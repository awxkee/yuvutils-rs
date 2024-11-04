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
    let bias_uv = ((range.bias_uv as f32 + 0.5f32) * (1i32 << 8i32) as f32) as i32;

    let precision_scale = (1 << 8) as f32;
    let max_colors = (1 << 8) - 1i32;

    let range_reduction_y =
        (range.range_y as f32 / max_colors as f32 * precision_scale).round() as i32;
    let range_reduction_uv =
        (range.range_uv as f32 / max_colors as f32 * precision_scale).round() as i32;

    let v_range_reduction_y = vdupq_n_s32(range_reduction_y);
    let v_range_reduction_uv = vdupq_n_s32(range_reduction_uv);
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

        let (y_v_high, cg_high, co_high) = neon_rgb_to_ycgco_r(
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

        let (y_v_low, cg_low, co_low) = neon_rgb_to_ycgco_r(
            r_low,
            g_low,
            b_low,
            v_range_reduction_y,
            v_range_reduction_uv,
            v_bias_y,
            v_bias_uv,
        );

        let y_store = uint16x8x2_t(y_v_low, y_v_high);
        vst1q_u16_x2(y_ptr.add(cx), y_store);

        match chroma_subsampling {
            YuvChromaSample::Yuv420 | YuvChromaSample::Yuv422 => {
                let cg_l_s = vrshrn_n_u32::<1>(vpaddlq_u16(cg_low));
                let cg_h_s = vrshrn_n_u32::<1>(vpaddlq_u16(cg_high));
                let co_l_s = vrshrn_n_u32::<1>(vpaddlq_u16(co_low));
                let co_h_s = vrshrn_n_u32::<1>(vpaddlq_u16(co_high));
                let cg = vcombine_u16(cg_l_s, cg_h_s);
                let co = vcombine_u16(co_l_s, co_h_s);
                vst1q_u16(cg_ptr.add(uv_x), cg);
                vst1q_u16(co_ptr.add(uv_x), co);

                uv_x += 8;
            }
            YuvChromaSample::Yuv444 => {
                let cg = uint16x8x2_t(cg_low, cg_high);
                vst1q_u16_x2(cg_ptr.add(uv_x), cg);
                let co = uint16x8x2_t(co_low, co_high);
                vst1q_u16_x2(co_ptr.add(uv_x), co);

                uv_x += 16;
            }
        }

        cx += 16;
    }

    ProcessedOffset { cx, ux: uv_x }
}
