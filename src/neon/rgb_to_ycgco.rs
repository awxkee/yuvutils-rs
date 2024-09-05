/*
 * // Copyright (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

use crate::internals::ProcessedOffset;
use crate::neon::neon_ycgco::neon_rgb_to_ycgco;
use crate::yuv_support::{YuvChromaRange, YuvChromaSample, YuvSourceChannels};
use std::arch::aarch64::*;

#[inline(always)]
pub unsafe fn neon_rgb_to_ycgco_row<const ORIGIN_CHANNELS: u8, const SAMPLING: u8>(
    range: &YuvChromaRange,
    y_plane: *mut u8,
    cg_plane: *mut u8,
    co_plane: *mut u8,
    rgba: &[u8],
    y_offset: usize,
    cg_offset: usize,
    co_offset: usize,
    rgba_offset: usize,
    start_cx: usize,
    start_ux: usize,
    width: usize,
) -> ProcessedOffset {
    let chroma_subsampling: YuvChromaSample = SAMPLING.into();
    let source_channels: YuvSourceChannels = ORIGIN_CHANNELS.into();
    let channels = source_channels.get_channels_count();

    let y_ptr = y_plane.add(y_offset);
    let cg_ptr = cg_plane.add(cg_offset);
    let co_ptr = co_plane.add(co_offset);
    let rgba_ptr = rgba.as_ptr().add(rgba_offset);

    let mut cx = start_cx;
    let mut uv_x = start_ux;

    let bias_y = ((range.bias_y as f32 + 0.5f32) * (1i32 << 8i32) as f32) as i32;
    let bias_uv = ((range.bias_uv as f32 + 0.5f32) * (1i32 << 8i32) as f32) as i32;

    let precision_scale = (1 << 8) as f32;
    let max_colors = (1 << 8) - 1i32;

    let range_reduction_y =
        (range.range_y as f32 / max_colors as f32 * precision_scale).round() as i32;
    let range_reduction_uv =
        (range.range_uv as f32 / max_colors as f32 * precision_scale).round() as i32;

    let v_range_reduction_y = vdupq_n_s16(range_reduction_y as i16);
    let v_range_reduction_uv = vdupq_n_s16(range_reduction_uv as i16);
    let v_bias_y = vdupq_n_s32(bias_y);
    let v_bias_uv = vdupq_n_s32(bias_uv);

    while cx + 16 < width {
        let r_values_u8: uint8x16_t;
        let g_values_u8: uint8x16_t;
        let b_values_u8: uint8x16_t;

        match source_channels {
            YuvSourceChannels::Rgb | YuvSourceChannels::Bgr => {
                let rgb_values = vld3q_u8(rgba_ptr.add(cx * channels));
                if source_channels == YuvSourceChannels::Rgb {
                    r_values_u8 = rgb_values.0;
                    g_values_u8 = rgb_values.1;
                    b_values_u8 = rgb_values.2;
                } else {
                    r_values_u8 = rgb_values.2;
                    g_values_u8 = rgb_values.1;
                    b_values_u8 = rgb_values.0;
                }
            }
            YuvSourceChannels::Rgba => {
                let rgb_values = vld4q_u8(rgba_ptr.add(cx * channels));
                r_values_u8 = rgb_values.0;
                g_values_u8 = rgb_values.1;
                b_values_u8 = rgb_values.2;
            }
            YuvSourceChannels::Bgra => {
                let rgb_values = vld4q_u8(rgba_ptr.add(cx * channels));
                r_values_u8 = rgb_values.2;
                g_values_u8 = rgb_values.1;
                b_values_u8 = rgb_values.0;
            }
        }

        let r_high = vreinterpretq_s16_u16(vmovl_high_u8(r_values_u8));
        let g_high = vreinterpretq_s16_u16(vmovl_high_u8(g_values_u8));
        let b_high = vreinterpretq_s16_u16(vmovl_high_u8(b_values_u8));

        let (y_v_high, cg_high, co_high) = neon_rgb_to_ycgco(
            r_high,
            g_high,
            b_high,
            v_range_reduction_y,
            v_range_reduction_uv,
            v_bias_y,
            v_bias_uv,
        );

        let r_low = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(r_values_u8)));
        let g_low = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(g_values_u8)));
        let b_low = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(b_values_u8)));

        let (y_v_low, cg_low, co_low) = neon_rgb_to_ycgco(
            r_low,
            g_low,
            b_low,
            v_range_reduction_y,
            v_range_reduction_uv,
            v_bias_y,
            v_bias_uv,
        );

        let y = vcombine_u8(vqmovn_u16(y_v_low), vqmovn_u16(y_v_high));
        let cg = vcombine_u8(vqmovn_u16(cg_low), vqmovn_u16(cg_high));
        let co = vcombine_u8(vqmovn_u16(co_low), vqmovn_u16(co_high));
        vst1q_u8(y_ptr.add(cx), y);

        match chroma_subsampling {
            YuvChromaSample::YUV420 | YuvChromaSample::YUV422 => {
                let cg_s = vrshrn_n_u16::<1>(vpaddlq_u8(cg));
                let co_s = vrshrn_n_u16::<1>(vpaddlq_u8(co));
                vst1_u8(cg_ptr.add(uv_x), cg_s);
                vst1_u8(co_ptr.add(uv_x), co_s);

                uv_x += 8;
            }
            YuvChromaSample::YUV444 => {
                vst1q_u8(cg_ptr.add(uv_x), cg);
                vst1q_u8(co_ptr.add(uv_x), co);

                uv_x += 16;
            }
        }

        cx += 16;
    }

    ProcessedOffset { cx, ux: uv_x }
}
