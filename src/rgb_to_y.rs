/*
 * Copyright (c) Radzivon Bartoshyk, 10/2024. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 * 1.  Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2.  Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3.  Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use crate::avx2::avx2_rgb_to_y_row;
#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    feature = "nightly_avx512"
))]
use crate::avx512bw::avx512_row_rgb_to_y;
use crate::built_coefficients::get_built_forward_transform;
use crate::images::YuvGrayImageMut;
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use crate::neon::{neon_rgb_to_y_rdm, neon_rgb_to_y_row};
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use crate::sse::sse_rgb_to_y;
use crate::yuv_error::check_rgba_destination;
use crate::yuv_support::*;
use crate::YuvError;
#[cfg(feature = "rayon")]
use rayon::iter::{IndexedParallelIterator, ParallelIterator};
#[cfg(feature = "rayon")]
use rayon::prelude::{ParallelSlice, ParallelSliceMut};

// Chroma subsampling always assumed as YUV 400
fn rgbx_to_y<const ORIGIN_CHANNELS: u8>(
    gray_image: &mut YuvGrayImageMut<u8>,
    rgba: &[u8],
    rgba_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) -> Result<(), YuvError> {
    let source_channels: YuvSourceChannels = ORIGIN_CHANNELS.into();
    let channels = source_channels.get_channels_count();

    check_rgba_destination(
        rgba,
        rgba_stride,
        gray_image.width,
        gray_image.height,
        channels,
    )?;
    gray_image.check_constraints()?;

    let chroma_range = get_yuv_range(8, range);
    let kr_kb = matrix.get_kr_kb();
    let max_range_p8 = (1u32 << 8u32) - 1u32;
    const PRECISION: i32 = 13;
    let transform =
        if let Some(stored_t) = get_built_forward_transform(PRECISION as u32, 8, range, matrix) {
            stored_t
        } else {
            let transform_precise = get_forward_transform(
                max_range_p8,
                chroma_range.range_y,
                chroma_range.range_uv,
                kr_kb.kr,
                kr_kb.kb,
            );
            transform_precise.to_integers(PRECISION as u32)
        };
    let bias_y = (chroma_range.bias_y as f32 + 0.5f32) as i32 + ((1 << (PRECISION - 1)) - 1);

    let i_bias_y = chroma_range.bias_y as i32;
    let i_cap_y = chroma_range.range_y as i32 + i_bias_y;

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    let use_sse = std::arch::is_x86_feature_detected!("sse4.1");
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    let use_avx = std::arch::is_x86_feature_detected!("avx2");
    #[cfg(all(
        any(target_arch = "x86", target_arch = "x86_64"),
        feature = "nightly_avx512"
    ))]
    let use_avx512 = std::arch::is_x86_feature_detected!("avx512bw");
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    let is_rdm_available = std::arch::is_aarch64_feature_detected!("rdm");

    let iter;
    #[cfg(feature = "rayon")]
    {
        iter = gray_image
            .y_plane
            .borrow_mut()
            .par_chunks_exact_mut(gray_image.y_stride as usize)
            .zip(rgba.par_chunks_exact(rgba_stride as usize));
    }
    #[cfg(not(feature = "rayon"))]
    {
        iter = gray_image
            .y_plane
            .borrow_mut()
            .chunks_exact_mut(gray_image.y_stride as usize)
            .zip(rgba.chunks_exact(rgba_stride as usize));
    }

    iter.for_each(|(y_plane, rgba)| {
        let mut _cx = 0usize;

        let y_plane = &mut y_plane[0..gray_image.width as usize];

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            #[cfg(feature = "nightly_avx512")]
            if use_avx512 {
                let processed_offset = avx512_row_rgb_to_y::<ORIGIN_CHANNELS>(
                    &transform,
                    &chroma_range,
                    y_plane,
                    rgba,
                    _cx,
                    gray_image.width as usize,
                );
                _cx = processed_offset;
            }
            if use_avx {
                let processed_offset = avx2_rgb_to_y_row::<ORIGIN_CHANNELS>(
                    &transform,
                    &chroma_range,
                    y_plane,
                    rgba,
                    _cx,
                    gray_image.width as usize,
                );
                _cx = processed_offset;
            }
            if use_sse {
                let processed_offset = sse_rgb_to_y::<ORIGIN_CHANNELS>(
                    &transform,
                    &chroma_range,
                    y_plane,
                    rgba,
                    _cx,
                    gray_image.width as usize,
                );
                _cx = processed_offset;
            }
        }

        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        unsafe {
            if is_rdm_available {
                _cx = neon_rgb_to_y_rdm::<ORIGIN_CHANNELS>(
                    &transform,
                    &chroma_range,
                    y_plane.as_mut_ptr(),
                    rgba,
                    _cx,
                    gray_image.width as usize,
                );
            } else {
                _cx = neon_rgb_to_y_row::<ORIGIN_CHANNELS, PRECISION>(
                    &transform,
                    &chroma_range,
                    y_plane.as_mut_ptr(),
                    rgba,
                    _cx,
                    gray_image.width as usize,
                );
            }
        }

        for (y_dst, rgba) in y_plane
            .iter_mut()
            .zip(rgba.chunks_exact(channels))
            .skip(_cx)
        {
            let r = rgba[source_channels.get_r_channel_offset()] as i32;
            let g = rgba[source_channels.get_g_channel_offset()] as i32;
            let b = rgba[source_channels.get_b_channel_offset()] as i32;
            let y = (r * transform.yr + g * transform.yg + b * transform.yb + bias_y) >> PRECISION;
            *y_dst = y.max(i_bias_y).min(i_cap_y) as u8;
        }
    });

    Ok(())
}

/// Convert RGB image data to YUV 400 planar format.
///
/// This function performs RGB to YUV conversion and stores the result in YUV400 planar format,
/// with Y (luminance) plane
///
/// # Arguments
///
/// * `gray_image` - Target gray image.
/// * `rgb` - The input RGB image data slice.
/// * `rgb_stride` - The stride (elements per row) for the RGB image data.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input RGB data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn rgb_to_yuv400(
    gray_image: &mut YuvGrayImageMut<u8>,
    rgb: &[u8],
    rgb_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) -> Result<(), YuvError> {
    rgbx_to_y::<{ YuvSourceChannels::Rgb as u8 }>(gray_image, rgb, rgb_stride, range, matrix)
}

/// Convert RGBA image data to YUV 400 planar format.
///
/// This function performs RGBA to YUV conversion and stores the result in YUV400 planar format,
/// with Y (luminance) plane
///
/// # Arguments
///
/// * `gray_image` - Target gray image.
/// * `rgba` - The input RGBA image data slice.
/// * `rgba_stride` - The stride (components per row) for the RGBA image data.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input RGBA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn rgba_to_yuv400(
    gray_image: &mut YuvGrayImageMut<u8>,
    rgba: &[u8],
    rgba_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) -> Result<(), YuvError> {
    rgbx_to_y::<{ YuvSourceChannels::Rgba as u8 }>(gray_image, rgba, rgba_stride, range, matrix)
}

/// Convert BGRA image data to YUV 400 planar format.
///
/// This function performs BGRA to YUV conversion and stores the result in YUV420 planar format,
/// with Y (luminance) plane
///
/// # Arguments
///
/// * `gray_image` - Target gray image.
/// * `bgra` - The input BGRA image data slice.
/// * `bgra_stride` - The stride (components per row) for the BGRA image data.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGRA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn bgra_to_yuv400(
    gray_image: &mut YuvGrayImageMut<u8>,
    bgra: &[u8],
    bgra_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) -> Result<(), YuvError> {
    rgbx_to_y::<{ YuvSourceChannels::Bgra as u8 }>(gray_image, bgra, bgra_stride, range, matrix)
}

/// Convert BGR image data to YUV 400 planar format.
///
/// This function performs BGR to YUV conversion and stores the result in YUV400 planar format,
/// with Y (luminance) plane
///
/// # Arguments
///
/// * `gray_image` - Target gray image.
/// * `bgr` - The input BGR image data slice.
/// * `bgr_stride` - The stride (components per row) for the RGB image data.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input RGB data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn bgr_to_yuv400(
    gray_image: &mut YuvGrayImageMut<u8>,
    rgb: &[u8],
    rgb_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) -> Result<(), YuvError> {
    rgbx_to_y::<{ YuvSourceChannels::Bgr as u8 }>(gray_image, rgb, rgb_stride, range, matrix)
}
