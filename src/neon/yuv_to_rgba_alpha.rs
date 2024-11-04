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
use crate::neon::neon_simd_support::neon_premultiply_alpha;
use crate::yuv_support::{
    CbCrInverseTransform, YuvChromaRange, YuvChromaSample, YuvSourceChannels,
};
use std::arch::aarch64::*;

#[inline(always)]
pub unsafe fn neon_yuv_to_rgba_alpha<const DESTINATION_CHANNELS: u8, const SAMPLING: u8>(
    range: &YuvChromaRange,
    transform: &CbCrInverseTransform<i32>,
    y_plane: &[u8],
    u_plane: &[u8],
    v_plane: &[u8],
    a_plane: &[u8],
    rgba: &mut [u8],
    start_cx: usize,
    start_ux: usize,
    y_offset: usize,
    u_offset: usize,
    v_offset: usize,
    a_offset: usize,
    rgba_offset: usize,
    width: usize,
    use_premultiply: bool,
) -> ProcessedOffset {
    let chroma_subsampling: YuvChromaSample = SAMPLING.into();
    let destination_channels: YuvSourceChannels = DESTINATION_CHANNELS.into();
    let channels = destination_channels.get_channels_count();

    let mut cx = start_cx;
    let mut uv_x = start_ux;
    let y_ptr = y_plane.as_ptr();
    let u_ptr = u_plane.as_ptr();
    let v_ptr = v_plane.as_ptr();
    let a_ptr = a_plane.as_ptr();
    let rgba_ptr = rgba.as_mut_ptr();

    let y_corr = vdupq_n_u8(range.bias_y as u8);
    let uv_corr = vdupq_n_s16(range.bias_uv as i16);
    let v_min_values = vdupq_n_s16(0i16);

    while cx + 16 < width {
        let y_values = vqsubq_u8(vld1q_u8(y_ptr.add(y_offset + cx)), y_corr);
        let a_values = vld1q_u8(a_ptr.add(a_offset + cx));

        let u_high_u8: uint8x8_t;
        let v_high_u8: uint8x8_t;
        let u_low_u8: uint8x8_t;
        let v_low_u8: uint8x8_t;

        match chroma_subsampling {
            YuvChromaSample::Yuv420 | YuvChromaSample::Yuv422 => {
                let u_values = vld1_u8(u_ptr.add(u_offset + uv_x));
                let v_values = vld1_u8(v_ptr.add(v_offset + uv_x));

                u_high_u8 = vzip2_u8(u_values, u_values);
                v_high_u8 = vzip2_u8(v_values, v_values);
                u_low_u8 = vzip1_u8(u_values, u_values);
                v_low_u8 = vzip1_u8(v_values, v_values);
            }
            YuvChromaSample::Yuv444 => {
                let u_values = vld1q_u8(u_ptr.add(u_offset + uv_x));
                let v_values = vld1q_u8(v_ptr.add(v_offset + uv_x));

                u_high_u8 = vget_high_u8(u_values);
                v_high_u8 = vget_high_u8(v_values);
                u_low_u8 = vget_low_u8(u_values);
                v_low_u8 = vget_low_u8(v_values);
            }
        }

        let u_high = vshlq_n_s16::<7>(vsubq_s16(
            vreinterpretq_s16_u16(vmovl_u8(u_high_u8)),
            uv_corr,
        ));
        let v_high = vshlq_n_s16::<7>(vsubq_s16(
            vreinterpretq_s16_u16(vmovl_u8(v_high_u8)),
            uv_corr,
        ));
        let y_high = vqrdmulhq_n_s16(
            vreinterpretq_s16_u16(vshll_high_n_u8::<7>(y_values)),
            transform.y_coef as i16,
        );

        let r_high = vqrshrun_n_s16::<4>(vmaxq_s16(
            vaddq_s16(y_high, vqrdmulhq_n_s16(v_high, transform.cr_coef as i16)),
            v_min_values,
        ));
        let b_high = vqrshrun_n_s16::<4>(vmaxq_s16(
            vaddq_s16(y_high, vqrdmulhq_n_s16(u_high, transform.cb_coef as i16)),
            v_min_values,
        ));
        let g_high = vqrshrun_n_s16::<4>(vmaxq_s16(
            vsubq_s16(
                y_high,
                vaddq_s16(
                    vqrdmulhq_n_s16(v_high, transform.g_coeff_1 as i16),
                    vqrdmulhq_n_s16(u_high, transform.g_coeff_2 as i16),
                ),
            ),
            v_min_values,
        ));

        let u_low = vshlq_n_s16::<7>(vsubq_s16(
            vreinterpretq_s16_u16(vmovl_u8(u_low_u8)),
            uv_corr,
        ));
        let v_low = vshlq_n_s16::<7>(vsubq_s16(
            vreinterpretq_s16_u16(vmovl_u8(v_low_u8)),
            uv_corr,
        ));
        let y_low = vqrdmulhq_n_s16(
            vreinterpretq_s16_u16(vshll_n_u8::<7>(vget_low_u8(y_values))),
            transform.y_coef as i16,
        );

        let r_low = vqrshrun_n_s16::<4>(vmaxq_s16(
            vqaddq_s16(y_low, vqrdmulhq_n_s16(v_low, transform.cr_coef as i16)),
            v_min_values,
        ));
        let b_low = vqrshrun_n_s16::<4>(vmaxq_s16(
            vqaddq_s16(y_low, vqrdmulhq_n_s16(u_low, transform.cb_coef as i16)),
            v_min_values,
        ));
        let g_low = vqrshrun_n_s16::<4>(vmaxq_s16(
            vsubq_s16(
                y_low,
                vaddq_s16(
                    vqrdmulhq_n_s16(v_low, transform.g_coeff_1 as i16),
                    vqrdmulhq_n_s16(u_low, transform.g_coeff_2 as i16),
                ),
            ),
            v_min_values,
        ));

        let mut r_values = vcombine_u8(r_low, r_high);
        let mut g_values = vcombine_u8(g_low, g_high);
        let mut b_values = vcombine_u8(b_low, b_high);

        let dst_shift = rgba_offset + cx * channels;

        if use_premultiply {
            r_values = neon_premultiply_alpha(r_values, a_values);
            g_values = neon_premultiply_alpha(g_values, a_values);
            b_values = neon_premultiply_alpha(b_values, a_values);
        }

        match destination_channels {
            YuvSourceChannels::Rgb | YuvSourceChannels::Bgr => {
                panic!("Should not be reached");
            }
            YuvSourceChannels::Rgba => {
                let dst_pack: uint8x16x4_t = uint8x16x4_t(r_values, g_values, b_values, a_values);
                vst4q_u8(rgba_ptr.add(dst_shift), dst_pack);
            }
            YuvSourceChannels::Bgra => {
                let dst_pack: uint8x16x4_t = uint8x16x4_t(b_values, g_values, r_values, a_values);
                vst4q_u8(rgba_ptr.add(dst_shift), dst_pack);
            }
        }

        cx += 16;

        match chroma_subsampling {
            YuvChromaSample::Yuv420 | YuvChromaSample::Yuv422 => {
                uv_x += 8;
            }
            YuvChromaSample::Yuv444 => {
                uv_x += 16;
            }
        }
    }

    ProcessedOffset { cx, ux: uv_x }
}
