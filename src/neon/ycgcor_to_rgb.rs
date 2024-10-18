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
use crate::neon::neon_ycgco_r::neon_ycgco_r_to_rgb;
use crate::yuv_support::{YuvChromaRange, YuvChromaSample, YuvSourceChannels};
use std::arch::aarch64::*;

#[inline(always)]
pub unsafe fn neon_ycgcor_to_rgb_row<const DESTINATION_CHANNELS: u8, const SAMPLING: u8>(
    range: &YuvChromaRange,
    y_plane: *const u16,
    cg_plane: *const u16,
    v_plane: *const u16,
    rgba: &mut [u8],
    start_cx: usize,
    start_ux: usize,
    rgba_offset: usize,
    width: usize,
) -> ProcessedOffset {
    let chroma_subsampling: YuvChromaSample = SAMPLING.into();
    let destination_channels: YuvSourceChannels = DESTINATION_CHANNELS.into();
    let channels = destination_channels.get_channels_count();
    let bias_y = range.bias_y as i32;
    let bias_uv = range.bias_uv as i32;

    let mut cx = start_cx;
    let mut uv_x = start_ux;

    let y_ptr = y_plane;
    let u_ptr = cg_plane;
    let v_ptr = v_plane;
    let rgba_ptr = rgba.as_mut_ptr().add(rgba_offset);

    let max_colors = (1 << 8) - 1i32;
    let precision_scale = (1 << 6) as f32;

    let range_reduction_y =
        (max_colors as f32 / range.range_y as f32 * precision_scale).round() as i32;
    let range_reduction_uv =
        (max_colors as f32 / range.range_uv as f32 * precision_scale).round() as i32;

    let y_corr = vdupq_n_s16(bias_y as i16);
    let uv_corr = vdupq_n_s16(bias_uv as i16);
    let y_reduction = vdupq_n_s16(range_reduction_y as i16);
    let uv_reduction = vdupq_n_s16(range_reduction_uv as i16);
    let v_alpha = vdupq_n_u8(255u8);

    while cx + 16 < width {
        let full_y_values = vld1q_u16_x2(y_ptr.add(cx));
        let y_values_lo = vreinterpretq_s16_u16(full_y_values.0);
        let y_values_hi = vreinterpretq_s16_u16(full_y_values.1);

        let u_high_s16: int16x8_t;
        let v_high_s16: int16x8_t;
        let u_low_s16: int16x8_t;
        let v_low_s16: int16x8_t;

        match chroma_subsampling {
            YuvChromaSample::YUV420 | YuvChromaSample::YUV422 => {
                let u_values = vreinterpretq_s16_u16(vld1q_u16(u_ptr.add(uv_x)));
                let v_values = vreinterpretq_s16_u16(vld1q_u16(v_ptr.add(uv_x)));

                u_high_s16 = vzip2q_s16(u_values, u_values);
                v_high_s16 = vzip2q_s16(v_values, v_values);
                u_low_s16 = vzip1q_s16(u_values, u_values);
                v_low_s16 = vzip1q_s16(v_values, v_values);
            }
            YuvChromaSample::YUV444 => {
                let full_u_values = vld1q_u16_x2(u_ptr.add(uv_x));
                let full_v_values = vld1q_u16_x2(v_ptr.add(uv_x));

                u_high_s16 = vreinterpretq_s16_u16(full_u_values.1);
                v_high_s16 = vreinterpretq_s16_u16(full_v_values.1);
                u_low_s16 = vreinterpretq_s16_u16(full_u_values.0);
                v_low_s16 = vreinterpretq_s16_u16(full_v_values.1);
            }
        }

        let (r_l, g_l, b_l) = neon_ycgco_r_to_rgb(
            y_values_lo,
            u_low_s16,
            v_low_s16,
            y_reduction,
            uv_reduction,
            y_corr,
            uv_corr,
        );
        let (r_h, g_h, b_h) = neon_ycgco_r_to_rgb(
            y_values_hi,
            u_high_s16,
            v_high_s16,
            y_reduction,
            uv_reduction,
            y_corr,
            uv_corr,
        );

        let r_values = vcombine_u8(r_l, r_h);
        let g_values = vcombine_u8(g_l, g_h);
        let b_values = vcombine_u8(b_l, b_h);

        let dst_shift = cx * channels;

        match destination_channels {
            YuvSourceChannels::Rgb => {
                let dst_pack: uint8x16x3_t = uint8x16x3_t(r_values, g_values, b_values);
                vst3q_u8(rgba_ptr.add(dst_shift), dst_pack);
            }
            YuvSourceChannels::Bgr => {
                let dst_pack: uint8x16x3_t = uint8x16x3_t(b_values, g_values, r_values);
                vst3q_u8(rgba_ptr.add(dst_shift), dst_pack);
            }
            YuvSourceChannels::Rgba => {
                let dst_pack: uint8x16x4_t = uint8x16x4_t(r_values, g_values, b_values, v_alpha);
                vst4q_u8(rgba_ptr.add(dst_shift), dst_pack);
            }
            YuvSourceChannels::Bgra => {
                let dst_pack: uint8x16x4_t = uint8x16x4_t(b_values, g_values, r_values, v_alpha);
                vst4q_u8(rgba_ptr.add(dst_shift), dst_pack);
            }
        }

        cx += 16;

        match chroma_subsampling {
            YuvChromaSample::YUV420 | YuvChromaSample::YUV422 => {
                uv_x += 8;
            }
            YuvChromaSample::YUV444 => {
                uv_x += 16;
            }
        }
    }

    ProcessedOffset { cx, ux: uv_x }
}
