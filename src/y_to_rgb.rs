/*
 * // Copyright (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use std::arch::aarch64::*;

use crate::yuv_support::{
    get_inverse_transform, get_kr_kb, get_yuv_range, YuvRange, YuvSourceChannels, YuvStandardMatrix,
};

// Chroma subsampling always assumed as 400
fn y_to_rgbx<const DESTINATION_CHANNELS: u8>(
    y_plane: &[u8],
    y_stride: u32,
    rgba: &mut [u8],
    rgba_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) {
    let destination_channels: YuvSourceChannels = DESTINATION_CHANNELS.into();
    let channels = destination_channels.get_channels_count();
    let range = get_yuv_range(8, range);
    let kr_kb = get_kr_kb(matrix);
    let transform = get_inverse_transform(255, range.range_y, range.range_uv, kr_kb.kr, kr_kb.kb);
    let precision_scale: i32 = 1i32 << 6i32;
    let y_coef = (transform.y_coef * precision_scale as f32).round() as i32;

    let bias_y = range.bias_y as i32;

    let mut y_offset = 0usize;
    let mut rgba_offset = 0usize;

    for _ in 0..height as usize {
        #[allow(unused_variables)]
        #[allow(unused_mut)]
        let mut cx = 0usize;

        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        unsafe {
            let y_ptr = y_plane.as_ptr();
            let rgba_ptr = rgba.as_mut_ptr();

            let y_corr = vdupq_n_u8(bias_y as u8);
            let v_luma_coeff = vdupq_n_u8(y_coef as u8);
            let v_luma_coeff_8 = vdup_n_u8(y_coef as u8);
            let v_min_values = vdupq_n_s16(0i16);
            let v_alpha = vdupq_n_u8(255u8);

            while cx + 16 < width as usize {
                let y_values = vsubq_u8(vld1q_u8(y_ptr.add(y_offset + cx)), y_corr);

                let y_high = vreinterpretq_s16_u16(vmull_high_u8(y_values, v_luma_coeff));

                let r_high = vqshrun_n_s16::<6>(vmaxq_s16(y_high, v_min_values));

                let y_low = vreinterpretq_s16_u16(vmull_u8(vget_low_u8(y_values), v_luma_coeff_8));

                let r_low = vqshrun_n_s16::<6>(vmaxq_s16(y_low, v_min_values));

                let r_values = vcombine_u8(r_low, r_high);

                let dst_shift = rgba_offset + cx * channels;

                match destination_channels {
                    YuvSourceChannels::Rgb => {
                        let dst_pack: uint8x16x3_t = uint8x16x3_t(r_values, r_values, r_values);
                        vst3q_u8(rgba_ptr.add(dst_shift), dst_pack);
                    }
                    YuvSourceChannels::Rgba => {
                        let dst_pack: uint8x16x4_t =
                            uint8x16x4_t(r_values, r_values, r_values, v_alpha);
                        vst4q_u8(rgba_ptr.add(dst_shift), dst_pack);
                    }
                    YuvSourceChannels::Bgra => {
                        let dst_pack: uint8x16x4_t =
                            uint8x16x4_t(r_values, r_values, r_values, v_alpha);
                        vst4q_u8(rgba_ptr.add(dst_shift), dst_pack);
                    }
                }

                cx += 16;
            }
        }

        for x in cx..width as usize {
            let y_value = (y_plane[y_offset + x] as i32 - bias_y) * y_coef;

            let r = (y_value >> 6).min(255i32).max(0);

            let px = x * channels;

            let rgba_shift = rgba_offset + px;

            rgba[rgba_shift + destination_channels.get_r_channel_offset()] = r as u8;
            rgba[rgba_shift + destination_channels.get_g_channel_offset()] = r as u8;
            rgba[rgba_shift + destination_channels.get_b_channel_offset()] = r as u8;
            if destination_channels.has_alpha() {
                rgba[rgba_shift + destination_channels.get_a_channel_offset()] = 255;
            }
        }

        y_offset += y_stride as usize;
        rgba_offset += rgba_stride as usize;
    }
}

/// Convert YUV 400 planar format to RGB format.
///
/// This function takes YUV 400 planar format data with 8-bit precision,
/// and converts it to RGB format with 8-bit per channel precision.
///
/// # Arguments
///
/// * `y_plane` - A slice to load the Y (luminance) plane data.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `width` - The width of the YUV image.
/// * `height` - The height of the YUV image.
/// * `rgb_data` - A mutable slice to store the converted RGB data.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGRA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv400_to_rgb(
    y_plane: &[u8],
    y_stride: u32,
    rgb: &mut [u8],
    rgb_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) {
    y_to_rgbx::<{ YuvSourceChannels::Rgb as u8 }>(
        y_plane, y_stride, rgb, rgb_stride, width, height, range, matrix,
    )
}

/// Convert YUV 400 planar format to RGBA format.
///
/// This function takes YUV 400 planar format data with 8-bit precision,
/// and converts it to RGBA format with 8-bit per channel precision.
///
/// # Arguments
///
/// * `y_plane` - A slice to load the Y (luminance) plane data.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `width` - The width of the YUV image.
/// * `height` - The height of the YUV image.
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
    y_plane: &[u8],
    y_stride: u32,
    rgba: &mut [u8],
    rgba_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) {
    y_to_rgbx::<{ YuvSourceChannels::Rgba as u8 }>(
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

/// Convert YUV 400 planar format to BGRA format.
///
/// This function takes YUV 400 planar format data with 8-bit precision,
/// and converts it to BGRA format with 8-bit per channel precision.
///
/// # Arguments
///
/// * `y_plane` - A slice to load the Y (luminance) plane data.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `width` - The width of the YUV image.
/// * `height` - The height of the YUV image.
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
    y_plane: &[u8],
    y_stride: u32,
    bgra: &mut [u8],
    bgra_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) {
    y_to_rgbx::<{ YuvSourceChannels::Bgra as u8 }>(
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
