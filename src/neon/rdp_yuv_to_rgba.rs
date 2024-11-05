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
use crate::yuv_support::{CbCrInverseTransform, YuvSourceChannels};
use std::arch::aarch64::*;

#[inline(always)]
pub unsafe fn rdp_neon_yuv_to_rgba_row<const DESTINATION_CHANNELS: u8>(
    transform: &CbCrInverseTransform<i32>,
    y_plane: &[u16],
    u_plane: &[u16],
    v_plane: &[u16],
    rgba: &mut [u8],
    start_cx: usize,
    start_ux: usize,
    width: usize,
) -> ProcessedOffset {
    let destination_channels: YuvSourceChannels = DESTINATION_CHANNELS.into();
    let channels = destination_channels.get_channels_count();

    let mut cx = start_cx;
    let mut uv_x = start_ux;

    let y_ptr = y_plane.as_ptr();
    let u_ptr = u_plane.as_ptr();
    let v_ptr = v_plane.as_ptr();
    let rgba_ptr = rgba.as_mut_ptr();

    let v_min_values = vdupq_n_s16(0i16);
    let v_alpha = vdupq_n_u8(255u8);
    let y_corr = vdupq_n_s16(4096);

    const V_SHR: i32 = 7;
    const UV_SCALE: i32 = 5;

    while cx + 16 < width {
        let y_u_values = vld1q_u16_x2(y_ptr.add(cx));
        let y_s_values = int16x8x2_t(
            vshlq_n_s16::<2>(vaddq_s16(vreinterpretq_s16_u16(y_u_values.0), y_corr)),
            vshlq_n_s16::<2>(vaddq_s16(vreinterpretq_s16_u16(y_u_values.1), y_corr)),
        );

        let u_high;
        let v_high;
        let u_low;
        let v_low;

        let u_values = vld1q_u16_x2(u_ptr.add(uv_x));
        let v_values = vld1q_u16_x2(v_ptr.add(uv_x));

        u_high = vreinterpretq_s16_u16(u_values.1);
        v_high = vreinterpretq_s16_u16(v_values.1);
        u_low = vreinterpretq_s16_u16(u_values.0);
        v_low = vreinterpretq_s16_u16(v_values.0);

        let r_high = vqrshrun_n_s16::<V_SHR>(vmaxq_s16(
            vaddq_s16(
                y_s_values.1,
                vshlq_n_s16::<UV_SCALE>(vqrdmulhq_n_s16(v_high, transform.cr_coef as i16)),
            ),
            v_min_values,
        ));
        let b_high = vqrshrun_n_s16::<V_SHR>(vmaxq_s16(
            vaddq_s16(
                y_s_values.1,
                vshlq_n_s16::<UV_SCALE>(vqrdmulhq_n_s16(u_high, transform.cb_coef as i16)),
            ),
            v_min_values,
        ));
        let g_high = vqrshrun_n_s16::<V_SHR>(vmaxq_s16(
            vsubq_s16(
                y_s_values.1,
                vaddq_s16(
                    vshlq_n_s16::<UV_SCALE>(vqrdmulhq_n_s16(v_high, transform.g_coeff_2 as i16)),
                    vshlq_n_s16::<UV_SCALE>(vqrdmulhq_n_s16(u_high, transform.g_coeff_1 as i16)),
                ),
            ),
            v_min_values,
        ));

        let r_low = vqrshrun_n_s16::<V_SHR>(vmaxq_s16(
            vaddq_s16(
                y_s_values.0,
                vshlq_n_s16::<UV_SCALE>(vqrdmulhq_n_s16(v_low, transform.cr_coef as i16)),
            ),
            v_min_values,
        ));
        let b_low = vqrshrun_n_s16::<V_SHR>(vmaxq_s16(
            vaddq_s16(
                y_s_values.0,
                vshlq_n_s16::<UV_SCALE>(vqrdmulhq_n_s16(u_low, transform.cb_coef as i16)),
            ),
            v_min_values,
        ));
        let g_low = vqrshrun_n_s16::<V_SHR>(vmaxq_s16(
            vsubq_s16(
                y_s_values.0,
                vaddq_s16(
                    vshlq_n_s16::<UV_SCALE>(vqrdmulhq_n_s16(v_low, transform.g_coeff_2 as i16)),
                    vshlq_n_s16::<UV_SCALE>(vqrdmulhq_n_s16(u_low, transform.g_coeff_1 as i16)),
                ),
            ),
            v_min_values,
        ));

        let r_values = vcombine_u8(r_low, r_high);
        let g_values = vcombine_u8(g_low, g_high);
        let b_values = vcombine_u8(b_low, b_high);

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
        uv_x += 16;
    }

    ProcessedOffset { cx, ux: uv_x }
}
