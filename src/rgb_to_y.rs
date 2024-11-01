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
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use crate::neon::neon_rgb_to_y_row;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use crate::sse::sse_rgb_to_y;
use crate::yuv_error::{check_rgba_destination, check_y8_channel};
use crate::yuv_support::*;
use crate::YuvError;

// Chroma subsampling always assumed as YUV 400
fn rgbx_to_y<const ORIGIN_CHANNELS: u8>(
    y_plane: &mut [u8],
    y_stride: u32,
    rgba: &[u8],
    rgba_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) -> Result<(), YuvError> {
    let source_channels: YuvSourceChannels = ORIGIN_CHANNELS.into();
    let channels = source_channels.get_channels_count();

    check_rgba_destination(rgba, rgba_stride, width, height, channels)?;
    check_y8_channel(y_plane, y_stride, width, height)?;

    let range = get_yuv_range(8, range);
    let kr_kb = matrix.get_kr_kb();
    let max_range_p8 = (1u32 << 8u32) - 1u32;
    let transform_precise = get_forward_transform(
        max_range_p8,
        range.range_y,
        range.range_uv,
        kr_kb.kr,
        kr_kb.kb,
    );
    let transform = transform_precise.to_integers(8);
    let precision_scale = (1 << 8) as f32;
    let bias_y = ((range.bias_y as f32 + 0.5f32) * precision_scale) as i32;

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    let mut _use_sse = std::arch::is_x86_feature_detected!("sse4.1");
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    let mut _use_avx = std::arch::is_x86_feature_detected!("avx2");
    #[cfg(all(
        any(target_arch = "x86", target_arch = "x86_64"),
        feature = "nightly_avx512"
    ))]
    let mut _use_avx512 = std::arch::is_x86_feature_detected!("avx512bw");

    let mut y_offset = 0usize;
    let mut rgba_offset = 0usize;

    for _ in 0..height as usize {
        let mut _cx = 0usize;

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        unsafe {
            #[cfg(feature = "nightly_avx512")]
            if _use_avx512 {
                let processed_offset = avx512_row_rgb_to_y::<ORIGIN_CHANNELS>(
                    &transform,
                    &range,
                    y_plane.as_mut_ptr(),
                    rgba,
                    y_offset,
                    rgba_offset,
                    _cx,
                    width as usize,
                );
                _cx = processed_offset;
            }
            if _use_avx {
                let processed_offset = avx2_rgb_to_y_row::<ORIGIN_CHANNELS>(
                    &transform,
                    &range,
                    y_plane.as_mut_ptr(),
                    rgba,
                    y_offset,
                    rgba_offset,
                    _cx,
                    width as usize,
                );
                _cx = processed_offset;
            }
            if _use_sse {
                let processed_offset = sse_rgb_to_y::<ORIGIN_CHANNELS>(
                    &transform,
                    &range,
                    y_plane.as_mut_ptr(),
                    rgba,
                    y_offset,
                    rgba_offset,
                    _cx,
                    width as usize,
                );
                _cx = processed_offset;
            }
        }

        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        unsafe {
            _cx = neon_rgb_to_y_row::<ORIGIN_CHANNELS>(
                &transform,
                &range,
                y_plane.as_mut_ptr(),
                rgba,
                y_offset,
                rgba_offset,
                _cx,
                width as usize,
            );
        }

        for x in _cx..width as usize {
            let px = x * channels;
            let dst_offset = rgba_offset + px;
            unsafe {
                let dst = rgba.get_unchecked(dst_offset..);
                let r = *dst.get_unchecked(source_channels.get_r_channel_offset()) as i32;
                let g = *dst.get_unchecked(source_channels.get_g_channel_offset()) as i32;
                let b = *dst.get_unchecked(source_channels.get_b_channel_offset()) as i32;
                let y = (r * transform.yr + g * transform.yg + b * transform.yb + bias_y) >> 8;
                *y_plane.get_unchecked_mut(y_offset + x) = y as u8;
            }
        }

        y_offset += y_stride as usize;
        rgba_offset += rgba_stride as usize;
    }

    Ok(())
}

/// Convert RGB image data to YUV 400 planar format.
///
/// This function performs RGB to YUV conversion and stores the result in YUV400 planar format,
/// with Y (luminance) plane
///
/// # Arguments
///
/// * `y_plane` - A mutable slice to store the Y (luminance) plane data.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `rgb` - The input RGB image data slice.
/// * `rgb_stride` - The stride (bytes per row) for the RGB image data.
/// * `width` - The width of the image in pixels.
/// * `height` - The height of the image in pixels.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input RGB data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn rgb_to_yuv400(
    y_plane: &mut [u8],
    y_stride: u32,
    rgb: &[u8],
    rgb_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) -> Result<(), YuvError> {
    rgbx_to_y::<{ YuvSourceChannels::Rgb as u8 }>(
        y_plane, y_stride, rgb, rgb_stride, width, height, range, matrix,
    )
}

/// Convert RGBA image data to YUV 400 planar format.
///
/// This function performs RGBA to YUV conversion and stores the result in YUV400 planar format,
/// with Y (luminance) plane
///
/// # Arguments
///
/// * `y_plane` - A mutable slice to store the Y (luminance) plane data.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `rgba` - The input RGBA image data slice.
/// * `rgba_stride` - The stride (bytes per row) for the RGBA image data.
/// * `width` - The width of the image in pixels.
/// * `height` - The height of the image in pixels.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input RGBA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn rgba_to_yuv400(
    y_plane: &mut [u8],
    y_stride: u32,
    rgba: &[u8],
    rgba_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) -> Result<(), YuvError> {
    rgbx_to_y::<{ YuvSourceChannels::Rgba as u8 }>(
        y_plane,
        y_stride,
        rgba,
        rgba_stride,
        width,
        height,
        range,
        matrix,
    )
}

/// Convert BGRA image data to YUV 400 planar format.
///
/// This function performs BGRA to YUV conversion and stores the result in YUV420 planar format,
/// with Y (luminance) plane
///
/// # Arguments
///
/// * `y_plane` - A mutable slice to store the Y (luminance) plane data.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `bgra` - The input BGRA image data slice.
/// * `bgra_stride` - The stride (bytes per row) for the BGRA image data.
/// * `width` - The width of the image in pixels.
/// * `height` - The height of the image in pixels.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGRA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn bgra_to_yuv400(
    y_plane: &mut [u8],
    y_stride: u32,
    bgra: &[u8],
    bgra_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) -> Result<(), YuvError> {
    rgbx_to_y::<{ YuvSourceChannels::Bgra as u8 }>(
        y_plane,
        y_stride,
        bgra,
        bgra_stride,
        width,
        height,
        range,
        matrix,
    )
}

/// Convert BGR image data to YUV 400 planar format.
///
/// This function performs BGR to YUV conversion and stores the result in YUV400 planar format,
/// with Y (luminance) plane
///
/// # Arguments
///
/// * `y_plane` - A mutable slice to store the Y (luminance) plane data.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `bgr` - The input BGR image data slice.
/// * `bgr_stride` - The stride (bytes per row) for the RGB image data.
/// * `width` - The width of the image in pixels.
/// * `height` - The height of the image in pixels.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input RGB data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn bgr_to_yuv400(
    y_plane: &mut [u8],
    y_stride: u32,
    rgb: &[u8],
    rgb_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) -> Result<(), YuvError> {
    rgbx_to_y::<{ YuvSourceChannels::Bgr as u8 }>(
        y_plane, y_stride, rgb, rgb_stride, width, height, range, matrix,
    )
}
