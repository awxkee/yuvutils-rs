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
use crate::avx2::avx2_ycgco_to_rgb_row;
#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    feature = "nightly_avx512"
))]
use crate::avx512bw::avx512_ycgco_to_rgb_row;
#[allow(unused_imports)]
use crate::internals::ProcessedOffset;
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use crate::neon::neon_ycgco_to_rgb_row;
use crate::numerics::qrshr;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use crate::sse::sse_ycgco_to_rgb_row;
use crate::yuv_error::{check_chroma_channel, check_rgba_destination, check_y8_channel};
#[allow(unused_imports)]
use crate::yuv_support::*;
use crate::YuvError;

fn ycgco_ro_rgbx<const DESTINATION_CHANNELS: u8, const SAMPLING: u8>(
    y_plane: &[u8],
    y_stride: u32,
    cg_plane: &[u8],
    cg_stride: u32,
    co_plane: &[u8],
    co_stride: u32,
    rgba: &mut [u8],
    rgba_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
) -> Result<(), YuvError> {
    let chroma_subsampling: YuvChromaSample = SAMPLING.into();
    let destination_channels: YuvSourceChannels = DESTINATION_CHANNELS.into();
    let channels = destination_channels.get_channels_count();

    check_rgba_destination(rgba, rgba_stride, width, height, channels)?;
    check_y8_channel(y_plane, y_stride, width, height)?;
    check_chroma_channel(cg_plane, cg_stride, width, height, chroma_subsampling)?;
    check_chroma_channel(co_plane, co_stride, width, height, chroma_subsampling)?;

    let range = get_yuv_range(8, range);
    let bias_y = range.bias_y as i32;
    let bias_uv = range.bias_uv as i32;

    const PRECISION: i32 = 6;

    let mut y_offset = 0usize;
    let mut u_offset = 0usize;
    let mut v_offset = 0usize;
    let mut rgba_offset = 0usize;

    let iterator_step = match chroma_subsampling {
        YuvChromaSample::Yuv420 => 2usize,
        YuvChromaSample::Yuv422 => 2usize,
        YuvChromaSample::Yuv444 => 1usize,
    };

    let max_colors = (1 << 8) - 1i32;
    let precision_scale = (1 << PRECISION) as f32;

    let range_reduction_y =
        (max_colors as f32 / range.range_y as f32 * precision_scale).round() as i32;
    let range_reduction_uv =
        (max_colors as f32 / range.range_uv as f32 * precision_scale).round() as i32;

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    let mut _use_sse = std::arch::is_x86_feature_detected!("sse4.1");
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    let mut _use_avx2 = std::arch::is_x86_feature_detected!("avx2");
    #[cfg(all(
        any(target_arch = "x86", target_arch = "x86_64"),
        feature = "nightly_avx512"
    ))]
    let mut _use_avx512 = std::arch::is_x86_feature_detected!("avx512bw");

    for y in 0..height as usize {
        #[allow(unused_variables)]
        #[allow(unused_mut)]
        let mut cx = 0usize;

        #[allow(unused_variables)]
        #[allow(unused_mut)]
        let mut uv_x = 0usize;

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        unsafe {
            #[cfg(feature = "nightly_avx512")]
            if _use_avx512 {
                let processed = avx512_ycgco_to_rgb_row::<DESTINATION_CHANNELS, SAMPLING>(
                    &range,
                    y_plane,
                    cg_plane,
                    co_plane,
                    rgba,
                    cx,
                    uv_x,
                    y_offset,
                    u_offset,
                    v_offset,
                    rgba_offset,
                    width as usize,
                );
                cx = processed.cx;
                uv_x = processed.ux;
            }
            if _use_avx2 {
                let processed = avx2_ycgco_to_rgb_row::<DESTINATION_CHANNELS, SAMPLING>(
                    &range,
                    y_plane,
                    cg_plane,
                    co_plane,
                    rgba,
                    cx,
                    uv_x,
                    y_offset,
                    u_offset,
                    v_offset,
                    rgba_offset,
                    width as usize,
                );
                cx = processed.cx;
                uv_x = processed.ux;
            }
            if _use_sse {
                let processed = sse_ycgco_to_rgb_row::<DESTINATION_CHANNELS, SAMPLING>(
                    &range,
                    y_plane,
                    cg_plane,
                    co_plane,
                    rgba,
                    cx,
                    uv_x,
                    y_offset,
                    u_offset,
                    v_offset,
                    rgba_offset,
                    width as usize,
                );
                cx = processed.cx;
                uv_x = processed.ux;
            }
        }

        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        unsafe {
            let processed = neon_ycgco_to_rgb_row::<DESTINATION_CHANNELS, SAMPLING>(
                &range,
                y_plane,
                cg_plane,
                co_plane,
                rgba,
                cx,
                uv_x,
                y_offset,
                u_offset,
                v_offset,
                rgba_offset,
                width as usize,
            );
            cx = processed.cx;
            uv_x = processed.ux;
        }

        for x in (cx..width as usize).step_by(iterator_step) {
            let y_value = (unsafe { *y_plane.get_unchecked(y_offset + x) } as i32 - bias_y)
                * range_reduction_y;

            let cg_pos = match chroma_subsampling {
                YuvChromaSample::Yuv420 | YuvChromaSample::Yuv422 => u_offset + uv_x,
                YuvChromaSample::Yuv444 => u_offset + uv_x,
            };

            let cg_value =
                (unsafe { *cg_plane.get_unchecked(cg_pos) } as i32 - bias_uv) * range_reduction_uv;

            let v_pos = match chroma_subsampling {
                YuvChromaSample::Yuv420 | YuvChromaSample::Yuv422 => v_offset + uv_x,
                YuvChromaSample::Yuv444 => v_offset + uv_x,
            };

            let co_value =
                (unsafe { *co_plane.get_unchecked(v_pos) } as i32 - bias_uv) * range_reduction_uv;

            let t = y_value - cg_value;

            let r = qrshr::<PRECISION, 8>(t + co_value);
            let b = qrshr::<PRECISION, 8>(t - co_value);
            let g = qrshr::<PRECISION, 8>(y_value + cg_value);

            let px = x * channels;

            let rgba_shift = rgba_offset + px;

            unsafe {
                *rgba.get_unchecked_mut(rgba_shift + destination_channels.get_r_channel_offset()) =
                    r as u8
            };
            unsafe {
                *rgba.get_unchecked_mut(rgba_shift + destination_channels.get_g_channel_offset()) =
                    g as u8
            };
            unsafe {
                *rgba.get_unchecked_mut(rgba_shift + destination_channels.get_b_channel_offset()) =
                    b as u8
            };
            if destination_channels.has_alpha() {
                unsafe {
                    *rgba.get_unchecked_mut(
                        rgba_shift + destination_channels.get_a_channel_offset(),
                    ) = 255
                };
            }

            if chroma_subsampling == YuvChromaSample::Yuv420
                || chroma_subsampling == YuvChromaSample::Yuv422
            {
                let next_x = x + 1;
                if next_x < width as usize {
                    let y_value = (unsafe { *y_plane.get_unchecked(y_offset + next_x) } as i32
                        - bias_y)
                        * range_reduction_y;

                    let r = qrshr::<PRECISION, 8>(t + co_value);
                    let b = qrshr::<PRECISION, 8>(t - co_value);
                    let g = qrshr::<PRECISION, 8>(y_value + cg_value);

                    let next_px = next_x * channels;

                    let rgba_shift = rgba_offset + next_px;

                    unsafe {
                        *rgba.get_unchecked_mut(
                            rgba_shift + destination_channels.get_r_channel_offset(),
                        ) = r as u8
                    };
                    unsafe {
                        *rgba.get_unchecked_mut(
                            rgba_shift + destination_channels.get_g_channel_offset(),
                        ) = g as u8
                    };
                    unsafe {
                        *rgba.get_unchecked_mut(
                            rgba_shift + destination_channels.get_b_channel_offset(),
                        ) = b as u8
                    };
                    if destination_channels.has_alpha() {
                        unsafe {
                            *rgba.get_unchecked_mut(
                                rgba_shift + destination_channels.get_a_channel_offset(),
                            ) = 255
                        };
                    }
                }
            }

            uv_x += 1;
        }

        y_offset += y_stride as usize;
        rgba_offset += rgba_stride as usize;
        match chroma_subsampling {
            YuvChromaSample::Yuv420 => {
                if y & 1 == 1 {
                    u_offset += cg_stride as usize;
                    v_offset += co_stride as usize;
                }
            }
            YuvChromaSample::Yuv444 | YuvChromaSample::Yuv422 => {
                u_offset += cg_stride as usize;
                v_offset += co_stride as usize;
            }
        }
    }

    Ok(())
}

/// Convert YCgCo 420 planar format to RGB format.
///
/// This function takes YCgCo 420 planar format data with 8-bit precision,
/// and converts it to RGB format with 8-bit per channel precision.
///
/// # Arguments
///
/// * `y_plane` - A slice to load the Y (luminance) plane data.
/// * `y_stride` - The stride (components per row) for the Y plane.
/// * `cg_plane` - A slice to load the U (chrominance) plane data.
/// * `cg_stride` - The stride (components per row) for the U plane.
/// * `co_plane` - A slice to load the V (chrominance) plane data.
/// * `co_stride` - The stride (components per row) for the V plane.
/// * `width` - The width of the YUV image.
/// * `height` - The height of the YUV image.
/// * `rgb_data` - A mutable slice to store the converted RGB data.
/// * `range` - The YUV range (limited or full).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input RGB data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn ycgco420_to_rgb(
    y_plane: &[u8],
    y_stride: u32,
    cg_plane: &[u8],
    cg_stride: u32,
    co_plane: &[u8],
    co_stride: u32,
    rgb: &mut [u8],
    rgb_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
) -> Result<(), YuvError> {
    ycgco_ro_rgbx::<{ YuvSourceChannels::Rgb as u8 }, { YuvChromaSample::Yuv420 as u8 }>(
        y_plane, y_stride, cg_plane, cg_stride, co_plane, co_stride, rgb, rgb_stride, width,
        height, range,
    )
}

/// Convert YCgCo 420 planar format to BGR format.
///
/// This function takes YCgCo 420 planar format data with 8-bit precision,
/// and converts it to BGR format with 8-bit per channel precision.
///
/// # Arguments
///
/// * `y_plane` - A slice to load the Y (luminance) plane data.
/// * `y_stride` - The stride (components per row) for the Y plane.
/// * `cg_plane` - A slice to load the U (chrominance) plane data.
/// * `cg_stride` - The stride (components per row) for the U plane.
/// * `co_plane` - A slice to load the V (chrominance) plane data.
/// * `co_stride` - The stride (components per row) for the V plane.
/// * `width` - The width of the YUV image.
/// * `height` - The height of the YUV image.
/// * `rgb_data` - A mutable slice to store the converted BGR data.
/// * `range` - The YUV range (limited or full).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGR data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn ycgco420_to_bgr(
    y_plane: &[u8],
    y_stride: u32,
    cg_plane: &[u8],
    cg_stride: u32,
    co_plane: &[u8],
    co_stride: u32,
    bgr: &mut [u8],
    bgr_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
) -> Result<(), YuvError> {
    ycgco_ro_rgbx::<{ YuvSourceChannels::Bgr as u8 }, { YuvChromaSample::Yuv420 as u8 }>(
        y_plane, y_stride, cg_plane, cg_stride, co_plane, co_stride, bgr, bgr_stride, width,
        height, range,
    )
}

/// Convert YCgCo 420 planar format to RGBA format.
///
/// This function takes YCgCo 420 planar format data with 8-bit precision,
/// and converts it to RGBA format with 8-bit per channel precision.
///
/// # Arguments
///
/// * `y_plane` - A slice to load the Y (luminance) plane data.
/// * `y_stride` - The stride (components per row) for the Y plane.
/// * `cg_plane` - A slice to load the U (chrominance) plane data.
/// * `cg_stride` - The stride (components per row) for the U plane.
/// * `co_plane` - A slice to load the V (chrominance) plane data.
/// * `co_stride` - The stride (components per row) for the V plane.
/// * `width` - The width of the YUV image.
/// * `height` - The height of the YUV image.
/// * `rgba_data` - A mutable slice to store the converted RGBA data.
/// * `range` - The YUV range (limited or full).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGRA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn ycgco420_to_rgba(
    y_plane: &[u8],
    y_stride: u32,
    cg_plane: &[u8],
    cg_stride: u32,
    co_plane: &[u8],
    co_stride: u32,
    rgba: &mut [u8],
    rgba_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
) -> Result<(), YuvError> {
    ycgco_ro_rgbx::<{ YuvSourceChannels::Rgba as u8 }, { YuvChromaSample::Yuv420 as u8 }>(
        y_plane,
        y_stride,
        cg_plane,
        cg_stride,
        co_plane,
        co_stride,
        rgba,
        rgba_stride,
        width,
        height,
        range,
    )
}

/// Convert YCgCo 420 planar format to BGRA format.
///
/// This function takes YCgCo 420 planar format data with 8-bit precision,
/// and converts it to BGRA format with 8-bit per channel precision.
///
/// # Arguments
///
/// * `y_plane` - A slice to load the Y (luminance) plane data.
/// * `y_stride` - The stride (components per row) for the Y plane.
/// * `cg_plane` - A slice to load the U (chrominance) plane data.
/// * `cg_stride` - The stride (components per row) for the U plane.
/// * `co_plane` - A slice to load the V (chrominance) plane data.
/// * `co_stride` - The stride (components per row) for the V plane.
/// * `width` - The width of the YUV image.
/// * `height` - The height of the YUV image.
/// * `bgra_data` - A mutable slice to store the converted BGRA data.
/// * `range` - The YUV range (limited or full).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGRA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn ycgco420_to_bgra(
    y_plane: &[u8],
    y_stride: u32,
    cg_plane: &[u8],
    cg_stride: u32,
    co_plane: &[u8],
    co_stride: u32,
    bgra: &mut [u8],
    bgra_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
) -> Result<(), YuvError> {
    ycgco_ro_rgbx::<{ YuvSourceChannels::Bgra as u8 }, { YuvChromaSample::Yuv420 as u8 }>(
        y_plane,
        y_stride,
        cg_plane,
        cg_stride,
        co_plane,
        co_stride,
        bgra,
        bgra_stride,
        width,
        height,
        range,
    )
}

/// Convert YCgCo 422 planar format to RGB format.
///
/// This function takes YCgCo 422 data with 8-bit precision,
/// and converts it to RGB format with 8-bit per channel precision.
///
/// # Arguments
///
/// * `y_plane` - A slice to load the Y (luminance) plane data.
/// * `y_stride` - The stride (components per row) for the Y plane.
/// * `cg_plane` - A slice to load the U (chrominance) plane data.
/// * `cg_stride` - The stride (components per row) for the U plane.
/// * `co_plane` - A slice to load the V (chrominance) plane data.
/// * `co_stride` - The stride (components per row) for the V plane.
/// * `width` - The width of the YUV image.
/// * `height` - The height of the YUV image.
/// * `rgb_data` - A mutable slice to store the converted RGB data.
/// * `range` - The YUV range (limited or full).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input RGB data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn ycgco422_to_rgb(
    y_plane: &[u8],
    y_stride: u32,
    cg_plane: &[u8],
    cg_stride: u32,
    co_plane: &[u8],
    co_stride: u32,
    rgb: &mut [u8],
    rgb_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
) -> Result<(), YuvError> {
    ycgco_ro_rgbx::<{ YuvSourceChannels::Rgb as u8 }, { YuvChromaSample::Yuv422 as u8 }>(
        y_plane, y_stride, cg_plane, cg_stride, co_plane, co_stride, rgb, rgb_stride, width,
        height, range,
    )
}

/// Convert YCgCo 422 planar format to BGR format.
///
/// This function takes YCgCo 422 data with 8-bit precision,
/// and converts it to BGR format with 8-bit per channel precision.
///
/// # Arguments
///
/// * `y_plane` - A slice to load the Y (luminance) plane data.
/// * `y_stride` - The stride (components per row) for the Y plane.
/// * `cg_plane` - A slice to load the U (chrominance) plane data.
/// * `cg_stride` - The stride (components per row) for the U plane.
/// * `co_plane` - A slice to load the V (chrominance) plane data.
/// * `co_stride` - The stride (components per row) for the V plane.
/// * `width` - The width of the YUV image.
/// * `height` - The height of the YUV image.
/// * `rgb_data` - A mutable slice to store the converted BGR data.
/// * `range` - The YUV range (limited or full).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGR data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn ycgco422_to_bgr(
    y_plane: &[u8],
    y_stride: u32,
    cg_plane: &[u8],
    cg_stride: u32,
    co_plane: &[u8],
    co_stride: u32,
    bgr: &mut [u8],
    bgr_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
) -> Result<(), YuvError> {
    ycgco_ro_rgbx::<{ YuvSourceChannels::Bgr as u8 }, { YuvChromaSample::Yuv422 as u8 }>(
        y_plane, y_stride, cg_plane, cg_stride, co_plane, co_stride, bgr, bgr_stride, width,
        height, range,
    )
}

/// Convert YCgCo 422 planar format to RGBA format.
///
/// This function takes YCgCo 422 data with 8-bit precision,
/// and converts it to RGBA format with 8-bit per channel precision.
///
/// # Arguments
///
/// * `y_plane` - A slice to load the Y (luminance) plane data.
/// * `y_stride` - The stride (components per row) for the Y plane.
/// * `cg_plane` - A slice to load the U (chrominance) plane data.
/// * `cg_stride` - The stride (components per row) for the U plane.
/// * `co_plane` - A slice to load the V (chrominance) plane data.
/// * `co_stride` - The stride (components per row) for the V plane.
/// * `width` - The width of the YUV image.
/// * `height` - The height of the YUV image.
/// * `bgra_data` - A mutable slice to store the converted RGBA data.
/// * `range` - The YUV range (limited or full).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGRA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn ycgco422_to_rgba(
    y_plane: &[u8],
    y_stride: u32,
    cg_plane: &[u8],
    cg_stride: u32,
    co_plane: &[u8],
    co_stride: u32,
    rgba: &mut [u8],
    rgba_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
) -> Result<(), YuvError> {
    ycgco_ro_rgbx::<{ YuvSourceChannels::Rgba as u8 }, { YuvChromaSample::Yuv422 as u8 }>(
        y_plane,
        y_stride,
        cg_plane,
        cg_stride,
        co_plane,
        co_stride,
        rgba,
        rgba_stride,
        width,
        height,
        range,
    )
}

/// Convert YCgCo 422 planar format to BGRA format.
///
/// This function takes YCgCo 422 data with 8-bit precision,
/// and converts it to BGRA format with 8-bit per channel precision.
///
/// # Arguments
///
/// * `y_plane` - A slice to load the Y (luminance) plane data.
/// * `y_stride` - The stride (components per row) for the Y plane.
/// * `cg_plane` - A slice to load the U (chrominance) plane data.
/// * `cg_stride` - The stride (components per row) for the U plane.
/// * `co_plane` - A slice to load the V (chrominance) plane data.
/// * `co_stride` - The stride (components per row) for the V plane.
/// * `width` - The width of the YUV image.
/// * `height` - The height of the YUV image.
/// * `bgra_data` - A mutable slice to store the converted BGRA data.
/// * `range` - The YUV range (limited or full).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGRA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn ycgco422_to_bgra(
    y_plane: &[u8],
    y_stride: u32,
    cg_plane: &[u8],
    cg_stride: u32,
    co_plane: &[u8],
    co_stride: u32,
    bgra: &mut [u8],
    bgra_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
) -> Result<(), YuvError> {
    ycgco_ro_rgbx::<{ YuvSourceChannels::Bgra as u8 }, { YuvChromaSample::Yuv422 as u8 }>(
        y_plane,
        y_stride,
        cg_plane,
        cg_stride,
        co_plane,
        co_stride,
        bgra,
        bgra_stride,
        width,
        height,
        range,
    )
}

/// Convert YCgCo 444 planar format to RGBA format.
///
/// This function takes YCgCo 444 data with 8-bit precision,
/// and converts it to RGBA format with 8-bit per channel precision.
///
/// # Arguments
///
/// * `y_plane` - A slice to load the Y (luminance) plane data.
/// * `y_stride` - The stride (components per row) for the Y plane.
/// * `cg_plane` - A slice to load the U (chrominance) plane data.
/// * `cg_stride` - The stride (components per row) for the U plane.
/// * `co_plane` - A slice to load the V (chrominance) plane data.
/// * `co_stride` - The stride (components per row) for the V plane.
/// * `width` - The width of the YUV image.
/// * `height` - The height of the YUV image.
/// * `rgba_data` - A mutable slice to store the converted RGBA data.
/// * `range` - The YUV range (limited or full).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGRA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn ycgco444_to_rgba(
    y_plane: &[u8],
    y_stride: u32,
    cg_plane: &[u8],
    cg_stride: u32,
    co_plane: &[u8],
    co_stride: u32,
    rgba: &mut [u8],
    rgba_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
) -> Result<(), YuvError> {
    ycgco_ro_rgbx::<{ YuvSourceChannels::Rgba as u8 }, { YuvChromaSample::Yuv444 as u8 }>(
        y_plane,
        y_stride,
        cg_plane,
        cg_stride,
        co_plane,
        co_stride,
        rgba,
        rgba_stride,
        width,
        height,
        range,
    )
}

/// Convert YCgCo 444 planar format to BGRA format.
///
/// This function takes YCgCo 444 data with 8-bit precision,
/// and converts it to BGRA format with 8-bit per channel precision.
///
/// # Arguments
///
/// * `y_plane` - A slice to load the Y (luminance) plane data.
/// * `y_stride` - The stride (components per row) for the Y plane.
/// * `cg_plane` - A slice to load the U (chrominance) plane data.
/// * `cg_stride` - The stride (components per row) for the U plane.
/// * `co_plane` - A slice to load the V (chrominance) plane data.
/// * `co_stride` - The stride (components per row) for the V plane.
/// * `width` - The width of the YUV image.
/// * `height` - The height of the YUV image.
/// * `bgra_data` - A mutable slice to store the converted BGRA data.
/// * `range` - The YUV range (limited or full).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGRA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn ycgco444_to_bgra(
    y_plane: &[u8],
    y_stride: u32,
    cg_plane: &[u8],
    cg_stride: u32,
    co_plane: &[u8],
    co_stride: u32,
    bgra: &mut [u8],
    bgra_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
) -> Result<(), YuvError> {
    ycgco_ro_rgbx::<{ YuvSourceChannels::Bgra as u8 }, { YuvChromaSample::Yuv444 as u8 }>(
        y_plane,
        y_stride,
        cg_plane,
        cg_stride,
        co_plane,
        co_stride,
        bgra,
        bgra_stride,
        width,
        height,
        range,
    )
}

/// Convert YCgCo 444 planar format to RGB format.
///
/// This function takes YCgCo 444 data with 8-bit precision,
/// and converts it to RGB format with 8-bit per channel precision.
///
/// # Arguments
///
/// * `y_plane` - A slice to load the Y (luminance) plane data.
/// * `y_stride` - The stride (components per row) for the Y plane.
/// * `cg_plane` - A slice to load the U (chrominance) plane data.
/// * `cg_stride` - The stride (components per row) for the U plane.
/// * `co_plane` - A slice to load the V (chrominance) plane data.
/// * `co_stride` - The stride (components per row) for the V plane.
/// * `width` - The width of the YUV image.
/// * `height` - The height of the YUV image.
/// * `rgb_data` - A mutable slice to store the converted RGB data.
/// * `range` - The YUV range (limited or full).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input RGB data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn ycgco444_to_rgb(
    y_plane: &[u8],
    y_stride: u32,
    cg_plane: &[u8],
    cg_stride: u32,
    co_plane: &[u8],
    co_stride: u32,
    rgb: &mut [u8],
    rgb_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
) -> Result<(), YuvError> {
    ycgco_ro_rgbx::<{ YuvSourceChannels::Rgb as u8 }, { YuvChromaSample::Yuv444 as u8 }>(
        y_plane, y_stride, cg_plane, cg_stride, co_plane, co_stride, rgb, rgb_stride, width,
        height, range,
    )
}

/// Convert YCgCo 444 planar format to BGR format.
///
/// This function takes YCgCo 444 data with 8-bit precision,
/// and converts it to BGR format with 8-bit per channel precision.
///
/// # Arguments
///
/// * `y_plane` - A slice to load the Y (luminance) plane data.
/// * `y_stride` - The stride (components per row) for the Y plane.
/// * `cg_plane` - A slice to load the U (chrominance) plane data.
/// * `cg_stride` - The stride (components per row) for the U plane.
/// * `co_plane` - A slice to load the V (chrominance) plane data.
/// * `co_stride` - The stride (components per row) for the V plane.
/// * `width` - The width of the YUV image.
/// * `height` - The height of the YUV image.
/// * `rgb_data` - A mutable slice to store the converted BGR data.
/// * `range` - The YUV range (limited or full).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGR data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn ycgco444_to_bgr(
    y_plane: &[u8],
    y_stride: u32,
    cg_plane: &[u8],
    cg_stride: u32,
    co_plane: &[u8],
    co_stride: u32,
    bgr: &mut [u8],
    bgr_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
) -> Result<(), YuvError> {
    ycgco_ro_rgbx::<{ YuvSourceChannels::Bgr as u8 }, { YuvChromaSample::Yuv444 as u8 }>(
        y_plane, y_stride, cg_plane, cg_stride, co_plane, co_stride, bgr, bgr_stride, width,
        height, range,
    )
}
