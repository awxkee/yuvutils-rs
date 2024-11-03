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
use crate::yuv_support::{
    CbCrInverseTransform, YuvChromaRange, YuvChromaSample, YuvNVOrder, YuvSourceChannels,
};
use std::arch::aarch64::*;

pub unsafe fn neon_yuv_nv_to_rgba_row<
    const UV_ORDER: u8,
    const DESTINATION_CHANNELS: u8,
    const YUV_CHROMA_SAMPLING: u8,
>(
    range: &YuvChromaRange,
    transform: &CbCrInverseTransform<i32>,
    y_plane: &[u8],
    uv_plane: &[u8],
    rgba: &mut [u8],
    start_cx: usize,
    start_ux: usize,
    y_offset: usize,
    uv_offset: usize,
    rgba_offset: usize,
    width: usize,
) -> ProcessedOffset {
    let order: YuvNVOrder = UV_ORDER.into();
    let destination_channels: YuvSourceChannels = DESTINATION_CHANNELS.into();
    let chroma_subsampling: YuvChromaSample = YUV_CHROMA_SAMPLING.into();
    let channels = destination_channels.get_channels_count();

    let y_ptr = y_plane.as_ptr();
    let uv_ptr = uv_plane.as_ptr();
    let bgra_ptr = rgba.as_mut_ptr();

    let y_corr = vdupq_n_u8(range.bias_y as u8);
    let uv_corr = vdupq_n_s16(range.bias_uv as i16);
    let v_min_values = vdupq_n_s16(0i16);
    let v_alpha = vdupq_n_u8(255u8);

    let mut cx = start_cx;
    let mut ux = start_ux;

    while cx + 16 < width {
        let y_values = vqsubq_u8(vld1q_u8(y_ptr.add(y_offset + cx)), y_corr);

        let u_high_u8: uint8x8_t;
        let v_high_u8: uint8x8_t;
        let u_low_u8: uint8x8_t;
        let v_low_u8: uint8x8_t;

        match chroma_subsampling {
            YuvChromaSample::YUV420 | YuvChromaSample::YUV422 => {
                let mut uv_values = vld2_u8(uv_ptr.add(uv_offset + ux));
                if order == YuvNVOrder::VU {
                    uv_values = uint8x8x2_t(uv_values.1, uv_values.0);
                }

                u_high_u8 = vzip2_u8(uv_values.0, uv_values.0);
                v_high_u8 = vzip2_u8(uv_values.1, uv_values.1);
                u_low_u8 = vzip1_u8(uv_values.0, uv_values.0);
                v_low_u8 = vzip1_u8(uv_values.1, uv_values.1);
            }
            YuvChromaSample::YUV444 => {
                let mut uv_values = vld2q_u8(uv_ptr.add(uv_offset + ux));
                if order == YuvNVOrder::VU {
                    uv_values = uint8x16x2_t(uv_values.1, uv_values.0);
                }
                u_high_u8 = vget_high_u8(uv_values.0);
                v_high_u8 = vget_high_u8(uv_values.1);
                u_low_u8 = vget_low_u8(uv_values.0);
                v_low_u8 = vget_low_u8(uv_values.1);
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
        let y_v_shl = vshll_high_n_u8::<7>(y_values);
        let y_high = vqrdmulhq_n_s16(vreinterpretq_s16_u16(y_v_shl), transform.y_coef as i16);

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
        let y_v_shl = vshll_n_u8::<7>(vget_low_u8(y_values));
        let y_low = vqrdmulhq_n_s16(vreinterpretq_s16_u16(y_v_shl), transform.y_coef as i16);

        let r_low = vqrshrun_n_s16::<4>(vmaxq_s16(
            vaddq_s16(y_low, vqrdmulhq_n_s16(v_low, transform.cr_coef as i16)),
            v_min_values,
        ));
        let b_low = vqrshrun_n_s16::<4>(vmaxq_s16(
            vaddq_s16(y_low, vqrdmulhq_n_s16(u_low, transform.cb_coef as i16)),
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

        let r_values = vcombine_u8(r_low, r_high);
        let g_values = vcombine_u8(g_low, g_high);
        let b_values = vcombine_u8(b_low, b_high);

        let dst_shift = rgba_offset + cx * channels;

        match destination_channels {
            YuvSourceChannels::Rgb => {
                let dst_pack: uint8x16x3_t = uint8x16x3_t(r_values, g_values, b_values);
                vst3q_u8(bgra_ptr.add(dst_shift), dst_pack);
            }
            YuvSourceChannels::Bgr => {
                let dst_pack: uint8x16x3_t = uint8x16x3_t(b_values, g_values, r_values);
                vst3q_u8(bgra_ptr.add(dst_shift), dst_pack);
            }
            YuvSourceChannels::Rgba => {
                let dst_pack: uint8x16x4_t = uint8x16x4_t(r_values, g_values, b_values, v_alpha);
                vst4q_u8(bgra_ptr.add(dst_shift), dst_pack);
            }
            YuvSourceChannels::Bgra => {
                let dst_pack: uint8x16x4_t = uint8x16x4_t(b_values, g_values, r_values, v_alpha);
                vst4q_u8(bgra_ptr.add(dst_shift), dst_pack);
            }
        }

        cx += 16;

        match chroma_subsampling {
            YuvChromaSample::YUV420 | YuvChromaSample::YUV422 => {
                ux += 16;
            }
            YuvChromaSample::YUV444 => {
                ux += 32;
            }
        }
    }

    let shuffle_u = vld1_u8([0, 0, 2, 2, 4, 4, 6, 6].as_ptr());
    let shuffle_v = vld1_u8([1, 1, 3, 3, 5, 5, 7, 7].as_ptr());

    while cx + 8 < width {
        let y_values = vqsub_u8(vld1_u8(y_ptr.add(y_offset + cx)), vget_low_u8(y_corr));

        let mut u_low_u8: uint8x8_t;
        let mut v_low_u8: uint8x8_t;

        match chroma_subsampling {
            YuvChromaSample::YUV420 | YuvChromaSample::YUV422 => {
                let uv_values = vld1_u8(uv_ptr.add(uv_offset + ux));

                u_low_u8 = vtbl1_u8(uv_values, shuffle_u);
                v_low_u8 = vtbl1_u8(uv_values, shuffle_v);

                #[allow(clippy::manual_swap)]
                if order == YuvNVOrder::VU {
                    let new_v = u_low_u8;
                    u_low_u8 = v_low_u8;
                    v_low_u8 = new_v;
                }
            }
            YuvChromaSample::YUV444 => {
                let mut uv_values = vld2_u8(uv_ptr.add(uv_offset + ux));
                if order == YuvNVOrder::VU {
                    uv_values = uint8x8x2_t(uv_values.1, uv_values.0);
                }
                u_low_u8 = uv_values.0;
                v_low_u8 = uv_values.0;
            }
        }

        let u_low = vshlq_n_s16::<7>(vsubq_s16(
            vreinterpretq_s16_u16(vmovl_u8(u_low_u8)),
            uv_corr,
        ));
        let v_low = vshlq_n_s16::<7>(vsubq_s16(
            vreinterpretq_s16_u16(vmovl_u8(v_low_u8)),
            uv_corr,
        ));
        let y_low = vqrdmulhq_n_s16(
            vreinterpretq_s16_u16(vshll_n_u8::<7>(y_values)),
            transform.y_coef as i16,
        );

        let r_low = vqrshrun_n_s16::<4>(vmaxq_s16(
            vaddq_s16(y_low, vqrdmulhq_n_s16(v_low, transform.cr_coef as i16)),
            v_min_values,
        ));
        let b_low = vqrshrun_n_s16::<4>(vmaxq_s16(
            vaddq_s16(y_low, vqrdmulhq_n_s16(u_low, transform.cb_coef as i16)),
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

        let r_values = r_low;
        let g_values = g_low;
        let b_values = b_low;

        let dst_shift = rgba_offset + cx * channels;

        match destination_channels {
            YuvSourceChannels::Rgb => {
                let dst_pack: uint8x8x3_t = uint8x8x3_t(r_values, g_values, b_values);
                vst3_u8(bgra_ptr.add(dst_shift), dst_pack);
            }
            YuvSourceChannels::Bgr => {
                let dst_pack: uint8x8x3_t = uint8x8x3_t(b_values, g_values, r_values);
                vst3_u8(bgra_ptr.add(dst_shift), dst_pack);
            }
            YuvSourceChannels::Rgba => {
                let dst_pack: uint8x8x4_t =
                    uint8x8x4_t(r_values, g_values, b_values, vget_low_u8(v_alpha));
                vst4_u8(bgra_ptr.add(dst_shift), dst_pack);
            }
            YuvSourceChannels::Bgra => {
                let dst_pack: uint8x8x4_t =
                    uint8x8x4_t(b_values, g_values, r_values, vget_low_u8(v_alpha));
                vst4_u8(bgra_ptr.add(dst_shift), dst_pack);
            }
        }

        cx += 8;

        match chroma_subsampling {
            YuvChromaSample::YUV420 | YuvChromaSample::YUV422 => {
                ux += 8;
            }
            YuvChromaSample::YUV444 => {
                ux += 16;
            }
        }
    }

    ProcessedOffset { cx, ux }
}
