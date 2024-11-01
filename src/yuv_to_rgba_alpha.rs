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
use crate::avx2::avx2_yuv_to_rgba_alpha;
#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    feature = "nightly_avx512"
))]
use crate::avx512bw::avx512_yuv_to_rgba_alpha;
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use crate::neon::neon_yuv_to_rgba_alpha;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use crate::sse::sse_yuv_to_rgba_alpha_row;
use crate::yuv_error::{check_chroma_channel, check_rgba_destination, check_y8_channel};
#[allow(unused_imports)]
use crate::yuv_support::*;
use crate::{YuvError, YuvRange, YuvStandardMatrix};
#[cfg(feature = "rayon")]
use rayon::iter::{IndexedParallelIterator, ParallelIterator};
#[cfg(feature = "rayon")]
use rayon::prelude::ParallelSliceMut;

fn yuv_with_alpha_to_rgbx<const DESTINATION_CHANNELS: u8, const SAMPLING: u8>(
    y_plane: &[u8],
    y_stride: u32,
    u_plane: &[u8],
    u_stride: u32,
    v_plane: &[u8],
    v_stride: u32,
    a_plane: &[u8],
    a_stride: u32,
    rgba: &mut [u8],
    rgba_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    premultiply_alpha: bool,
) -> Result<(), YuvError> {
    let chroma_subsampling: YuvChromaSample = SAMPLING.into();
    let dst_chans: YuvSourceChannels = DESTINATION_CHANNELS.into();
    assert!(
        dst_chans.has_alpha(),
        "yuv_with_alpha_to_rgbx cannot be called on configuration without alpha"
    );
    let channels = dst_chans.get_channels_count();

    check_rgba_destination(rgba, rgba_stride, width, height, channels)?;
    check_y8_channel(y_plane, y_stride, width, height)?;
    check_y8_channel(a_plane, a_stride, width, height)?;
    check_chroma_channel(u_plane, u_stride, width, height, chroma_subsampling)?;
    check_chroma_channel(v_plane, v_stride, width, height, chroma_subsampling)?;

    let range = get_yuv_range(8, range);
    let kr_kb = matrix.get_kr_kb();
    let transform = get_inverse_transform(255, range.range_y, range.range_uv, kr_kb.kr, kr_kb.kb);
    const PRECISION: i32 = 6;
    const ROUNDING_CONST: i32 = 1 << (PRECISION - 1);
    let inverse_transform = transform.to_integers(PRECISION as u32);

    let cr_coef = inverse_transform.cr_coef;
    let cb_coef = inverse_transform.cb_coef;
    let y_coef = inverse_transform.y_coef;
    let g_coef_1 = inverse_transform.g_coeff_1;
    let g_coef_2 = inverse_transform.g_coeff_2;

    let bias_y = range.bias_y as i32;
    let bias_uv = range.bias_uv as i32;

    let iterator_step = match chroma_subsampling {
        YuvChromaSample::YUV420 => 2usize,
        YuvChromaSample::YUV422 => 2usize,
        YuvChromaSample::YUV444 => 1usize,
    };

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    let mut _use_avx2 = std::arch::is_x86_feature_detected!("avx2");
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    let mut _use_sse = std::arch::is_x86_feature_detected!("sse4.1");
    #[cfg(all(
        any(target_arch = "x86", target_arch = "x86_64"),
        feature = "nightly_avx512"
    ))]
    let mut _use_avx512 = std::arch::is_x86_feature_detected!("avx512bw");

    let iter;
    #[cfg(feature = "rayon")]
    {
        iter = rgba.par_chunks_exact_mut(rgba_stride as usize);
    }
    #[cfg(not(feature = "rayon"))]
    {
        iter = rgba.chunks_exact_mut(rgba_stride as usize);
    }

    iter.enumerate().for_each(|(y, rgba)| {
        let y_offset = y * (y_stride as usize);
        let u_offset = if chroma_subsampling == YuvChromaSample::YUV420 {
            (y >> 1) * (u_stride as usize)
        } else {
            y * (u_stride as usize)
        };
        let v_offset = if chroma_subsampling == YuvChromaSample::YUV420 {
            (y >> 1) * (v_stride as usize)
        } else {
            y * (v_stride as usize)
        };
        let a_offset = y * (a_stride as usize);
        let rgba_offset = 0;

        #[allow(unused_variables)]
        #[allow(unused_mut)]
        let mut cx = 0usize;

        #[allow(unused_variables)]
        #[allow(unused_mut)]
        let mut uv_x = 0usize;

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        unsafe {
            #[cfg(feature = "nightly_avx512")]
            {
                if _use_avx512 {
                    let processed = avx512_yuv_to_rgba_alpha::<DESTINATION_CHANNELS, SAMPLING>(
                        &range,
                        &inverse_transform,
                        y_plane,
                        u_plane,
                        v_plane,
                        a_plane,
                        rgba,
                        cx,
                        uv_x,
                        y_offset,
                        u_offset,
                        v_offset,
                        a_offset,
                        rgba_offset,
                        width as usize,
                        premultiply_alpha,
                    );
                    cx = processed.cx;
                    uv_x = processed.ux;
                }
            }
            if _use_avx2 {
                let processed = avx2_yuv_to_rgba_alpha::<DESTINATION_CHANNELS, SAMPLING>(
                    &range,
                    &inverse_transform,
                    y_plane,
                    u_plane,
                    v_plane,
                    a_plane,
                    rgba,
                    cx,
                    uv_x,
                    y_offset,
                    u_offset,
                    v_offset,
                    a_offset,
                    rgba_offset,
                    width as usize,
                    premultiply_alpha,
                );
                cx = processed.cx;
                uv_x = processed.ux;
            }
            if _use_sse {
                let processed = sse_yuv_to_rgba_alpha_row::<DESTINATION_CHANNELS, SAMPLING>(
                    &range,
                    &inverse_transform,
                    y_plane,
                    u_plane,
                    v_plane,
                    a_plane,
                    rgba,
                    cx,
                    uv_x,
                    y_offset,
                    u_offset,
                    v_offset,
                    a_offset,
                    rgba_offset,
                    width as usize,
                    premultiply_alpha,
                );
                cx = processed.cx;
                uv_x = processed.ux;
            }
        }

        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        unsafe {
            let processed = neon_yuv_to_rgba_alpha::<DESTINATION_CHANNELS, SAMPLING>(
                &range,
                &inverse_transform,
                y_plane,
                u_plane,
                v_plane,
                a_plane,
                rgba,
                cx,
                uv_x,
                y_offset,
                u_offset,
                v_offset,
                a_offset,
                rgba_offset,
                width as usize,
                premultiply_alpha,
            );
            cx = processed.cx;
            uv_x = processed.ux;
        }

        for x in (cx..width as usize).step_by(iterator_step) {
            let y_value =
                (unsafe { *y_plane.get_unchecked(y_offset + x) } as i32 - bias_y) * y_coef;

            let u_pos = match chroma_subsampling {
                YuvChromaSample::YUV420 | YuvChromaSample::YUV422 => u_offset + uv_x,
                YuvChromaSample::YUV444 => u_offset + uv_x,
            };

            let cb_value = unsafe { *u_plane.get_unchecked(u_pos) } as i32 - bias_uv;

            let v_pos = match chroma_subsampling {
                YuvChromaSample::YUV420 | YuvChromaSample::YUV422 => v_offset + uv_x,
                YuvChromaSample::YUV444 => v_offset + uv_x,
            };

            let cr_value = unsafe { *v_plane.get_unchecked(v_pos) } as i32 - bias_uv;

            let mut r = ((y_value + cr_coef * cr_value + ROUNDING_CONST) >> PRECISION)
                .min(255)
                .max(0);
            let mut b = ((y_value + cb_coef * cb_value + ROUNDING_CONST) >> PRECISION)
                .min(255)
                .max(0);
            let mut g = ((y_value - g_coef_1 * cr_value - g_coef_2 * cb_value + ROUNDING_CONST)
                >> PRECISION)
                .min(255)
                .max(0);

            let px = x * channels;

            let rgba_shift = rgba_offset + px;

            let a_value = unsafe { *a_plane.get_unchecked(a_offset + x) };
            if premultiply_alpha {
                r = (r * a_value as i32) / 255;
                g = (g * a_value as i32) / 255;
                b = (b * a_value as i32) / 255;
            }

            unsafe {
                let dst = rgba.get_unchecked_mut(rgba_shift..);
                *dst.get_unchecked_mut(dst_chans.get_r_channel_offset()) = r as u8;
                *dst.get_unchecked_mut(dst_chans.get_g_channel_offset()) = g as u8;
                *dst.get_unchecked_mut(dst_chans.get_b_channel_offset()) = b as u8;
                *dst.get_unchecked_mut(dst_chans.get_a_channel_offset()) = a_value;
            }

            if chroma_subsampling == YuvChromaSample::YUV420
                || chroma_subsampling == YuvChromaSample::YUV422
            {
                let next_x = x + 1;
                if x + 1 < width as usize {
                    let y_value = (unsafe { *y_plane.get_unchecked(y_offset + x + 1) } as i32
                        - bias_y)
                        * y_coef;

                    let mut r = ((y_value + cr_coef * cr_value + ROUNDING_CONST) >> PRECISION)
                        .min(255)
                        .max(0);
                    let mut b = ((y_value + cb_coef * cb_value + ROUNDING_CONST) >> PRECISION)
                        .min(255)
                        .max(0);
                    let mut g = ((y_value - g_coef_1 * cr_value - g_coef_2 * cb_value
                        + ROUNDING_CONST)
                        >> PRECISION)
                        .min(255)
                        .max(0);

                    let next_px = next_x * channels;

                    let rgba_shift = rgba_offset + next_px;

                    let a_value = unsafe { *a_plane.get_unchecked(a_offset + next_x) };
                    if premultiply_alpha {
                        r = (r * a_value as i32) / 255;
                        g = (g * a_value as i32) / 255;
                        b = (b * a_value as i32) / 255;
                    }

                    unsafe {
                        let dst = rgba.get_unchecked_mut(rgba_shift..);
                        *dst.get_unchecked_mut(dst_chans.get_r_channel_offset()) = r as u8;
                        *dst.get_unchecked_mut(dst_chans.get_g_channel_offset()) = g as u8;
                        *dst.get_unchecked_mut(dst_chans.get_b_channel_offset()) = b as u8;
                        *dst.get_unchecked_mut(dst_chans.get_a_channel_offset()) = a_value;
                    }
                }
            }

            uv_x += 1;
        }
    });

    Ok(())
}

/// Convert YUV 420 planar format to RGBA format and appends provided alpha channel.
///
/// This function takes YUV 420 planar format data with 8-bit precision,
/// and converts it to RGBA format with 8-bit per channel precision.
///
/// # Arguments
///
/// * `y_plane` - A slice to load the Y (luminance) plane data.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `u_plane` - A slice to load the U (chrominance) plane data.
/// * `u_stride` - The stride (bytes per row) for the U plane.
/// * `v_plane` - A slice to load the V (chrominance) plane data.
/// * `v_stride` - The stride (bytes per row) for the V plane.
/// * `a_plane` - A slice to load alpha plane to append to result.
/// * `a_stride` - The stride (bytes per row) for the alpha plane.
/// * `width` - The width of the YUV image.
/// * `height` - The height of the YUV image.
/// * `rgba_data` - A mutable slice to store the converted RGBA data.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
/// * `premultiply_alpha` - Flag to premultiply alpha or not
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGRA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv420_with_alpha_to_rgba(
    y_plane: &[u8],
    y_stride: u32,
    u_plane: &[u8],
    u_stride: u32,
    v_plane: &[u8],
    v_stride: u32,
    a_plane: &[u8],
    a_stride: u32,
    rgba: &mut [u8],
    rgba_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    premultiply_alpha: bool,
) -> Result<(), YuvError> {
    yuv_with_alpha_to_rgbx::<{ YuvSourceChannels::Rgba as u8 }, { YuvChromaSample::YUV420 as u8 }>(
        y_plane,
        y_stride,
        u_plane,
        u_stride,
        v_plane,
        v_stride,
        a_plane,
        a_stride,
        rgba,
        rgba_stride,
        width,
        height,
        range,
        matrix,
        premultiply_alpha,
    )
}

/// Convert YUV 420 planar format to BGRA format and appends provided alpha channel.
///
/// This function takes YUV 420 planar format data with 8-bit precision,
/// and converts it to BGRA format with 8-bit per channel precision.
///
/// # Arguments
///
/// * `y_plane` - A slice to load the Y (luminance) plane data.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `u_plane` - A slice to load the U (chrominance) plane data.
/// * `u_stride` - The stride (bytes per row) for the U plane.
/// * `v_plane` - A slice to load the V (chrominance) plane data.
/// * `v_stride` - The stride (bytes per row) for the V plane.
/// * `a_plane` - A slice to load alpha plane to append to result.
/// * `a_stride` - The stride (bytes per row) for the alpha plane.
/// * `width` - The width of the YUV image.
/// * `height` - The height of the YUV image.
/// * `bgra_data` - A mutable slice to store the converted BGRA data.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
/// * `premultiply_alpha` - Flag to premultiply alpha or not
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGRA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv420_with_alpha_to_bgra(
    y_plane: &[u8],
    y_stride: u32,
    u_plane: &[u8],
    u_stride: u32,
    v_plane: &[u8],
    v_stride: u32,
    a_plane: &[u8],
    a_stride: u32,
    bgra: &mut [u8],
    bgra_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    premultiply_alpha: bool,
) -> Result<(), YuvError> {
    yuv_with_alpha_to_rgbx::<{ YuvSourceChannels::Bgra as u8 }, { YuvChromaSample::YUV420 as u8 }>(
        y_plane,
        y_stride,
        u_plane,
        u_stride,
        v_plane,
        v_stride,
        a_plane,
        a_stride,
        bgra,
        bgra_stride,
        width,
        height,
        range,
        matrix,
        premultiply_alpha,
    )
}

/// Convert YUV 422 planar format to RGBA format and appends provided alpha channel.
///
/// This function takes YUV 422 planar format data with 8-bit precision,
/// and converts it to RGBA format with 8-bit per channel precision.
///
/// # Arguments
///
/// * `y_plane` - A slice to load the Y (luminance) plane data.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `u_plane` - A slice to load the U (chrominance) plane data.
/// * `u_stride` - The stride (bytes per row) for the U plane.
/// * `v_plane` - A slice to load the V (chrominance) plane data.
/// * `v_stride` - The stride (bytes per row) for the V plane.
/// * `a_plane` - A slice to load alpha plane to append to result.
/// * `a_stride` - The stride (bytes per row) for the alpha plane.
/// * `width` - The width of the YUV image.
/// * `height` - The height of the YUV image.
/// * `rgba_data` - A mutable slice to store the converted RGBA data.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
/// * `premultiply_alpha` - Flag to premultiply alpha or not
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGRA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv422_with_alpha_to_rgba(
    y_plane: &[u8],
    y_stride: u32,
    u_plane: &[u8],
    u_stride: u32,
    v_plane: &[u8],
    v_stride: u32,
    a_plane: &[u8],
    a_stride: u32,
    rgba: &mut [u8],
    rgba_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    premultiply_alpha: bool,
) -> Result<(), YuvError> {
    yuv_with_alpha_to_rgbx::<{ YuvSourceChannels::Rgba as u8 }, { YuvChromaSample::YUV422 as u8 }>(
        y_plane,
        y_stride,
        u_plane,
        u_stride,
        v_plane,
        v_stride,
        a_plane,
        a_stride,
        rgba,
        rgba_stride,
        width,
        height,
        range,
        matrix,
        premultiply_alpha,
    )
}

/// Convert YUV 422 planar format to BGRA format and appends provided alpha channel.
///
/// This function takes YUV 422 planar format data with 8-bit precision,
/// and converts it to BGRA format with 8-bit per channel precision.
///
/// # Arguments
///
/// * `y_plane` - A slice to load the Y (luminance) plane data.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `u_plane` - A slice to load the U (chrominance) plane data.
/// * `u_stride` - The stride (bytes per row) for the U plane.
/// * `v_plane` - A slice to load the V (chrominance) plane data.
/// * `v_stride` - The stride (bytes per row) for the V plane.
/// * `a_plane` - A slice to load alpha plane to append to result.
/// * `a_stride` - The stride (bytes per row) for the alpha plane.
/// * `width` - The width of the YUV image.
/// * `height` - The height of the YUV image.
/// * `bgra_data` - A mutable slice to store the converted BGRA data.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
/// * `premultiply_alpha` - Flag to premultiply alpha or not
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGRA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv422_with_alpha_to_bgra(
    y_plane: &[u8],
    y_stride: u32,
    u_plane: &[u8],
    u_stride: u32,
    v_plane: &[u8],
    v_stride: u32,
    a_plane: &[u8],
    a_stride: u32,
    bgra: &mut [u8],
    bgra_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    premultiply_alpha: bool,
) -> Result<(), YuvError> {
    yuv_with_alpha_to_rgbx::<{ YuvSourceChannels::Bgra as u8 }, { YuvChromaSample::YUV422 as u8 }>(
        y_plane,
        y_stride,
        u_plane,
        u_stride,
        v_plane,
        v_stride,
        a_plane,
        a_stride,
        bgra,
        bgra_stride,
        width,
        height,
        range,
        matrix,
        premultiply_alpha,
    )
}

/// Convert YUV 444 planar format to RGBA format and appends provided alpha channel.
///
/// This function takes YUV 444 planar format data with 8-bit precision,
/// and converts it to RGBA format with 8-bit per channel precision.
///
/// # Arguments
///
/// * `y_plane` - A slice to load the Y (luminance) plane data.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `u_plane` - A slice to load the U (chrominance) plane data.
/// * `u_stride` - The stride (bytes per row) for the U plane.
/// * `v_plane` - A slice to load the V (chrominance) plane data.
/// * `v_stride` - The stride (bytes per row) for the V plane.
/// * `a_plane` - A slice to load alpha plane to append to result.
/// * `a_stride` - The stride (bytes per row) for the alpha plane.
/// * `width` - The width of the YUV image.
/// * `height` - The height of the YUV image.
/// * `rgba_data` - A mutable slice to store the converted RGBA data.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
/// * `premultiply_alpha` - Flag to premultiply alpha or not
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGRA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv444_with_alpha_to_rgba(
    y_plane: &[u8],
    y_stride: u32,
    u_plane: &[u8],
    u_stride: u32,
    v_plane: &[u8],
    v_stride: u32,
    a_plane: &[u8],
    a_stride: u32,
    rgba: &mut [u8],
    rgba_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    premultiply_alpha: bool,
) -> Result<(), YuvError> {
    yuv_with_alpha_to_rgbx::<{ YuvSourceChannels::Rgba as u8 }, { YuvChromaSample::YUV444 as u8 }>(
        y_plane,
        y_stride,
        u_plane,
        u_stride,
        v_plane,
        v_stride,
        a_plane,
        a_stride,
        rgba,
        rgba_stride,
        width,
        height,
        range,
        matrix,
        premultiply_alpha,
    )
}

/// Convert YUV 444 planar format to BGRA format and appends provided alpha channel.
///
/// This function takes YUV 444 planar format data with 8-bit precision,
/// and converts it to BGRA format with 8-bit per channel precision.
///
/// # Arguments
///
/// * `y_plane` - A slice to load the Y (luminance) plane data.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `u_plane` - A slice to load the U (chrominance) plane data.
/// * `u_stride` - The stride (bytes per row) for the U plane.
/// * `v_plane` - A slice to load the V (chrominance) plane data.
/// * `v_stride` - The stride (bytes per row) for the V plane.
/// * `a_plane` - A slice to load alpha plane to append to result.
/// * `a_stride` - The stride (bytes per row) for the alpha plane.
/// * `width` - The width of the YUV image.
/// * `height` - The height of the YUV image.
/// * `bgra_data` - A mutable slice to store the converted BGRA data.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
/// * `premultiply_alpha` - Flag to premultiply alpha or not
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGRA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv444_with_alpha_to_bgra(
    y_plane: &[u8],
    y_stride: u32,
    u_plane: &[u8],
    u_stride: u32,
    v_plane: &[u8],
    v_stride: u32,
    a_plane: &[u8],
    a_stride: u32,
    bgra: &mut [u8],
    bgra_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    premultiply_alpha: bool,
) -> Result<(), YuvError> {
    yuv_with_alpha_to_rgbx::<{ YuvSourceChannels::Bgra as u8 }, { YuvChromaSample::YUV444 as u8 }>(
        y_plane,
        y_stride,
        u_plane,
        u_stride,
        v_plane,
        v_stride,
        a_plane,
        a_stride,
        bgra,
        bgra_stride,
        width,
        height,
        range,
        matrix,
        premultiply_alpha,
    )
}
