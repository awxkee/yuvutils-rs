/*
 * // Copyright (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

use crate::internals::ProcessedOffset;
use crate::neon::neon_ycgco_r::neon_rgb_to_ycgco_r;
use crate::yuv_support::{YuvChromaRange, YuvChromaSample, YuvSourceChannels};
use std::arch::aarch64::*;

#[inline(always)]
pub unsafe fn neon_rgb_to_ycgcor_row<const ORIGIN_CHANNELS: u8, const SAMPLING: u8>(
    range: &YuvChromaRange,
    y_plane: *mut u16,
    cg_plane: *mut u16,
    co_plane: *mut u16,
    rgba: &[u8],
    rgba_offset: usize,
    start_cx: usize,
    start_ux: usize,
    width: usize,
) -> ProcessedOffset {
    let chroma_subsampling: YuvChromaSample = SAMPLING.into();
    let source_channels: YuvSourceChannels = ORIGIN_CHANNELS.into();
    let channels = source_channels.get_channels_count();

    let y_ptr = y_plane;
    let cg_ptr = cg_plane;
    let co_ptr = co_plane;
    let rgba_ptr = rgba.as_ptr().add(rgba_offset);

    let mut cx = start_cx;
    let mut uv_x = start_ux;

    let bias_y = ((range.bias_y as f32 + 0.5f32) * (1i32 << 8i32) as f32) as i32;

    let precision_scale = (1 << 8) as f32;
    let max_colors = 2i32.pow(8) - 1i32;

    let range_reduction_y =
        (range.range_y as f32 / max_colors as f32 * precision_scale).round() as i32;

    let v_range_reduction_y = vdup_n_s16(range_reduction_y as i16);
    let v_bias_y = vdupq_n_s32(bias_y);

    while cx + 16 < width {
        let r_values_u8: uint8x16_t;
        let g_values_u8: uint8x16_t;
        let b_values_u8: uint8x16_t;

        match source_channels {
            YuvSourceChannels::Rgb => {
                let rgb_values = vld3q_u8(rgba_ptr.add(cx * channels));
                r_values_u8 = rgb_values.0;
                g_values_u8 = rgb_values.1;
                b_values_u8 = rgb_values.2;
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

        let (y_v_high, cg_high, co_high) =
            neon_rgb_to_ycgco_r(r_high, g_high, b_high, v_range_reduction_y, v_bias_y);

        let r_low = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(r_values_u8)));
        let g_low = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(g_values_u8)));
        let b_low = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(b_values_u8)));

        let (y_v_low, cg_low, co_low) =
            neon_rgb_to_ycgco_r(r_low, g_low, b_low, v_range_reduction_y, v_bias_y);

        let y_store = uint16x8x2_t(y_v_low, y_v_high);
        vst1q_u16_x2(y_ptr.add(cx), y_store);

        match chroma_subsampling {
            YuvChromaSample::YUV420 | YuvChromaSample::YUV422 => {
                let cg_l_s = vshrn_n_u32::<1>(vpaddlq_u16(cg_low));
                let cg_h_s = vshrn_n_u32::<1>(vpaddlq_u16(cg_high));
                let co_l_s = vshrn_n_u32::<1>(vpaddlq_u16(co_low));
                let co_h_s = vshrn_n_u32::<1>(vpaddlq_u16(co_high));
                let cg = vcombine_u16(cg_l_s, cg_h_s);
                let co = vcombine_u16(co_l_s, co_h_s);
                vst1q_u16(cg_ptr.add(uv_x), cg);
                vst1q_u16(co_ptr.add(uv_x), co);

                uv_x += 8;
            }
            YuvChromaSample::YUV444 => {
                let cg = uint16x8x2_t(cg_low, cg_high);
                vst1q_u16_x2(cg_ptr.add(uv_x), cg);
                let co = uint16x8x2_t(co_low, co_high);
                vst1q_u16_x2(co_ptr.add(uv_x), co);

                uv_x += 16;
            }
        }

        cx += 16;
    }

    return ProcessedOffset { cx, ux: uv_x };
}
