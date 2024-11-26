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
#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    feature = "nightly_avx512"
))]
use crate::avx512bw::avx512_y_to_rgb_row;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unused_imports)]
use crate::internals::ProcessedOffset;
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use crate::neon::{neon_y_to_rgb_row, neon_y_to_rgb_row_rdm};
use crate::numerics::qrshr;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
use crate::wasm32::wasm_y_to_rgb_row;
use crate::yuv_error::check_rgba_destination;
use crate::yuv_support::*;
use crate::{YuvError, YuvGrayImage};
#[cfg(feature = "rayon")]
use rayon::iter::{IndexedParallelIterator, ParallelIterator};
#[cfg(feature = "rayon")]
use rayon::prelude::{ParallelSlice, ParallelSliceMut};

// Chroma subsampling always assumed as 400
fn y_to_rgbx<const DESTINATION_CHANNELS: u8>(
    image: &YuvGrayImage<u8>,
    rgba: &mut [u8],
    rgba_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) -> Result<(), YuvError> {
    let destination_channels: YuvSourceChannels = DESTINATION_CHANNELS.into();
    let channels = destination_channels.get_channels_count();

    check_rgba_destination(rgba, rgba_stride, image.width, image.height, channels)?;
    image.check_constraints()?;

    let chroma_range = get_yuv_range(8, range);
    let kr_kb = matrix.get_kr_kb();
    let transform = get_inverse_transform(
        255,
        chroma_range.range_y,
        chroma_range.range_uv,
        kr_kb.kr,
        kr_kb.kb,
    );

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    const PRECISION: i32 = 6;
    #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
    const PRECISION: i32 = 12;
    let inverse_transform = transform.to_integers(PRECISION as u32);
    let y_coef = inverse_transform.y_coef;

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    let is_rdm_available = std::arch::is_aarch64_feature_detected!("rdm");
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    let neon_wide_row_handler = if is_rdm_available {
        neon_y_to_rgb_row_rdm::<DESTINATION_CHANNELS>
    } else {
        neon_y_to_rgb_row::<PRECISION, DESTINATION_CHANNELS>
    };

    let bias_y = chroma_range.bias_y as i32;

    #[cfg(all(
        any(target_arch = "x86", target_arch = "x86_64"),
        feature = "nightly_avx512"
    ))]
    let use_avx512 = std::arch::is_x86_feature_detected!("avx512bw");

    let y_plane = image.y_plane;
    let y_stride = image.y_stride;

    let iter;
    let y_iter;
    #[cfg(feature = "rayon")]
    {
        iter = rgba.par_chunks_exact_mut(rgba_stride as usize);
        y_iter = y_plane.par_chunks_exact(y_stride as usize);
    }
    #[cfg(not(feature = "rayon"))]
    {
        iter = rgba.chunks_exact_mut(rgba_stride as usize);
        y_iter = y_plane.chunks_exact(y_stride as usize);
    }

    if range == YuvRange::Limited {
        iter.zip(y_iter).for_each(|(rgba, y_plane)| {
            let y_plane = &y_plane[0..image.width as usize];
            let mut _cx = 0usize;

            #[cfg(all(
                any(target_arch = "x86", target_arch = "x86_64"),
                feature = "nightly_avx512"
            ))]
            unsafe {
                if use_avx512 {
                    let processed = avx512_y_to_rgb_row::<DESTINATION_CHANNELS>(
                        &chroma_range,
                        &inverse_transform,
                        y_plane,
                        rgba,
                        _cx,
                        0,
                        0,
                        image.width as usize,
                    );
                    _cx = processed;
                }
            }

            #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
            unsafe {
                let offset = neon_wide_row_handler(
                    &chroma_range,
                    &inverse_transform,
                    y_plane,
                    rgba,
                    _cx,
                    image.width as usize,
                );
                _cx = offset;
            }

            #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
            unsafe {
                let offset = wasm_y_to_rgb_row::<DESTINATION_CHANNELS>(
                    &chroma_range,
                    &inverse_transform,
                    y_plane,
                    rgba,
                    _cx,
                    0,
                    0,
                    image.width as usize,
                );
                _cx = offset;
            }

            let rgba_sliced = &mut rgba[(_cx * channels)..];
            let y_sliced = &y_plane[_cx..];

            for (y_src, rgba) in y_sliced.iter().zip(rgba_sliced.chunks_exact_mut(channels)) {
                let y_value = (*y_src as i32 - bias_y) * y_coef;

                let r = qrshr::<PRECISION, 8>(y_value);
                rgba[destination_channels.get_r_channel_offset()] = r as u8;
                rgba[destination_channels.get_g_channel_offset()] = r as u8;
                rgba[destination_channels.get_b_channel_offset()] = r as u8;
                if destination_channels.has_alpha() {
                    rgba[destination_channels.get_a_channel_offset()] = 255;
                }
            }
        });
    } else {
        iter.zip(y_iter).for_each(|(rgba, y_plane)| {
            let mut _cx = 0usize;
            let y_plane = &y_plane[0..image.width as usize];
            let rgba_sliced = &mut rgba[(_cx * channels)..];
            let y_sliced = &y_plane[_cx..];

            for (y_src, rgba) in y_sliced.iter().zip(rgba_sliced.chunks_exact_mut(channels)) {
                let r = *y_src;
                rgba[destination_channels.get_r_channel_offset()] = r;
                rgba[destination_channels.get_g_channel_offset()] = r;
                rgba[destination_channels.get_b_channel_offset()] = r;
                if destination_channels.has_alpha() {
                    rgba[destination_channels.get_a_channel_offset()] = 255;
                }
            }
        });
    }

    Ok(())
}

/// Convert YUV 400 planar format to RGB format.
///
/// This function takes YUV 400 planar format data with 8-bit precision,
/// and converts it to RGB format with 8-bit per channel precision.
///
/// # Arguments
///
/// * `gray_image` - Source YUV gray image.
/// * `rgb_data` - A mutable slice to store the converted RGB data.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input RGB data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv400_to_rgb(
    gray_image: &YuvGrayImage<u8>,
    rgb: &mut [u8],
    rgb_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) -> Result<(), YuvError> {
    y_to_rgbx::<{ YuvSourceChannels::Rgb as u8 }>(gray_image, rgb, rgb_stride, range, matrix)
}

/// Convert YUV 400 planar format to BGR format.
///
/// This function takes YUV 400 planar format data with 8-bit precision,
/// and converts it to BGR format with 8-bit per channel precision.
///
/// # Arguments
///
/// * `gray_image` - Source YUV gray image.
/// * `rgb_data` - A mutable slice to store the converted BGR data.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGR data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv400_to_bgr(
    gray_image: &YuvGrayImage<u8>,
    bgr: &mut [u8],
    bgr_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) -> Result<(), YuvError> {
    y_to_rgbx::<{ YuvSourceChannels::Bgr as u8 }>(gray_image, bgr, bgr_stride, range, matrix)
}

/// Convert YUV 400 planar format to RGBA format.
///
/// This function takes YUV 400 planar format data with 8-bit precision,
/// and converts it to RGBA format with 8-bit per channel precision.
///
/// # Arguments
///
/// * `gray_image` - Source YUV gray image.
/// * `rgba_data` - A mutable slice to store the converted RGBA data.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGRA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv400_to_rgba(
    gray_image: &YuvGrayImage<u8>,
    rgba: &mut [u8],
    rgba_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) -> Result<(), YuvError> {
    y_to_rgbx::<{ YuvSourceChannels::Rgba as u8 }>(gray_image, rgba, rgba_stride, range, matrix)
}

/// Convert YUV 400 planar format to BGRA format.
///
/// This function takes YUV 400 planar format data with 8-bit precision,
/// and converts it to BGRA format with 8-bit per channel precision.
///
/// # Arguments
///
/// * `gray_image` - Source YUV gray image.
/// * `bgra_data` - A mutable slice to store the converted BGRA data.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGRA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv400_to_bgra(
    gray_image: &YuvGrayImage<u8>,
    bgra: &mut [u8],
    bgra_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) -> Result<(), YuvError> {
    y_to_rgbx::<{ YuvSourceChannels::Bgra as u8 }>(gray_image, bgra, bgra_stride, range, matrix)
}
