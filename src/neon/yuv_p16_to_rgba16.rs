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

use std::arch::aarch64::*;

use crate::internals::ProcessedOffset;
use crate::yuv_support::{
    CbCrInverseTransform, YuvBytesPacking, YuvChromaRange, YuvChromaSample, YuvEndianness,
    YuvSourceChannels,
};

#[inline(always)]
pub unsafe fn neon_yuv_p16_to_rgba16_row<
    const DESTINATION_CHANNELS: u8,
    const SAMPLING: u8,
    const ENDIANNESS: u8,
    const BYTES_POSITION: u8,
>(
    y_ld_ptr: *const u16,
    u_ld_ptr: *const u16,
    v_ld_ptr: *const u16,
    rgba: *mut u16,
    dst_offset: usize,
    width: u32,
    range: &YuvChromaRange,
    transform: &CbCrInverseTransform<i32>,
    start_cx: usize,
    start_ux: usize,
    bit_depth: usize,
) -> ProcessedOffset {
    let destination_channels: YuvSourceChannels = DESTINATION_CHANNELS.into();
    let channels = destination_channels.get_channels_count();
    let chroma_subsampling: YuvChromaSample = SAMPLING.into();
    let endianness: YuvEndianness = ENDIANNESS.into();
    let bytes_position: YuvBytesPacking = BYTES_POSITION.into();
    let dst_ptr = rgba;

    let y_corr = vdupq_n_s16(range.bias_y as i16);
    let uv_corr = vdup_n_s16(range.bias_uv as i16);
    let v_luma_coeff = vdupq_n_s16(transform.y_coef as i16);
    let v_cr_coeff = vdup_n_s16(transform.cr_coef as i16);
    let v_cb_coeff = vdup_n_s16(transform.cb_coef as i16);
    let v_min_values = vdupq_n_s16(0i16);
    let v_g_coeff_1 = vdup_n_s16(-(transform.g_coeff_1 as i16));
    let v_g_coeff_2 = vdup_n_s16(-(transform.g_coeff_2 as i16));
    let v_alpha = vdupq_n_u16((1 << bit_depth) - 1);
    let v_msb_shift = vdupq_n_s16(bit_depth as i16 - 16);

    let mut cx = start_cx;
    let mut ux = start_ux;

    while cx + 8 < width as usize {
        let y_values: int16x8_t;

        let u_values_c: int16x4_t;
        let v_values_c: int16x4_t;

        let u_values_l = vld1_u16(u_ld_ptr.add(ux));
        let v_values_l = vld1_u16(v_ld_ptr.add(ux));

        match endianness {
            YuvEndianness::BigEndian => {
                let mut y_u_values = vreinterpretq_u16_u8(vrev16q_u8(vreinterpretq_u8_u16(
                    vld1q_u16(y_ld_ptr.add(cx)),
                )));
                if bytes_position == YuvBytesPacking::MostSignificantBytes {
                    y_u_values = vshlq_u16(y_u_values, v_msb_shift);
                }
                y_values = vsubq_s16(vreinterpretq_s16_u16(y_u_values), y_corr);

                let mut u_v = vreinterpret_u16_u8(vrev16_u8(vreinterpret_u8_u16(u_values_l)));
                let mut v_v = vreinterpret_u16_u8(vrev16_u8(vreinterpret_u8_u16(v_values_l)));
                if bytes_position == YuvBytesPacking::MostSignificantBytes {
                    u_v = vshl_u16(u_v, vget_low_s16(v_msb_shift));
                    v_v = vshl_u16(v_v, vget_low_s16(v_msb_shift));
                }
                u_values_c = vsub_s16(vreinterpret_s16_u16(u_v), uv_corr);
                v_values_c = vsub_s16(vreinterpret_s16_u16(v_v), uv_corr);
            }
            YuvEndianness::LittleEndian => {
                let mut y_vl = vld1q_u16(y_ld_ptr.add(cx));
                if bytes_position == YuvBytesPacking::MostSignificantBytes {
                    y_vl = vshlq_u16(y_vl, v_msb_shift);
                }
                y_values = vsubq_s16(vreinterpretq_s16_u16(y_vl), y_corr);

                let mut u_vl = u_values_l;
                let mut v_vl = v_values_l;
                if bytes_position == YuvBytesPacking::MostSignificantBytes {
                    u_vl = vshl_u16(u_vl, vget_low_s16(v_msb_shift));
                    v_vl = vshl_u16(v_vl, vget_low_s16(v_msb_shift));
                }
                u_values_c = vsub_s16(vreinterpret_s16_u16(u_vl), uv_corr);
                v_values_c = vsub_s16(vreinterpret_s16_u16(v_vl), uv_corr);
            }
        }

        let u_high = vzip2_s16(u_values_c, u_values_c);
        let v_high = vzip2_s16(v_values_c, v_values_c);

        let y_high = vmull_high_s16(y_values, v_luma_coeff);

        let r_high = vrshrn_n_s32::<6>(vmlal_s16(y_high, v_high, v_cr_coeff));
        let b_high = vrshrn_n_s32::<6>(vmlal_s16(y_high, u_high, v_cb_coeff));
        let g_high = vrshrn_n_s32::<6>(vmlal_s16(
            vmlal_s16(y_high, v_high, v_g_coeff_1),
            u_high,
            v_g_coeff_2,
        ));

        let y_low = vmull_s16(vget_low_s16(y_values), vget_low_s16(v_luma_coeff));
        let u_low = vzip1_s16(u_values_c, u_values_c);
        let v_low = vzip1_s16(v_values_c, v_values_c);

        let r_low = vrshrn_n_s32::<6>(vmlal_s16(y_low, v_low, v_cr_coeff));
        let b_low = vrshrn_n_s32::<6>(vmlal_s16(y_low, u_low, v_cb_coeff));
        let g_low = vrshrn_n_s32::<6>(vmlal_s16(
            vmlal_s16(y_low, v_low, v_g_coeff_1),
            u_low,
            v_g_coeff_2,
        ));

        let r_values = vreinterpretq_u16_s16(vmaxq_s16(vcombine_s16(r_low, r_high), v_min_values));
        let g_values = vreinterpretq_u16_s16(vmaxq_s16(vcombine_s16(g_low, g_high), v_min_values));
        let b_values = vreinterpretq_u16_s16(vmaxq_s16(vcombine_s16(b_low, b_high), v_min_values));

        match destination_channels {
            YuvSourceChannels::Rgb => {
                let dst_pack = uint16x8x3_t(r_values, g_values, b_values);
                vst3q_u16(dst_ptr.add(dst_offset + cx * channels), dst_pack);
            }
            YuvSourceChannels::Bgr => {
                let dst_pack = uint16x8x3_t(b_values, g_values, r_values);
                vst3q_u16(dst_ptr.add(dst_offset + cx * channels), dst_pack);
            }
            YuvSourceChannels::Rgba => {
                let dst_pack = uint16x8x4_t(r_values, g_values, b_values, v_alpha);
                vst4q_u16(dst_ptr.add(dst_offset + cx * channels), dst_pack);
            }
            YuvSourceChannels::Bgra => {
                let dst_pack = uint16x8x4_t(b_values, g_values, r_values, v_alpha);
                vst4q_u16(dst_ptr.add(dst_offset + cx * channels), dst_pack);
            }
        }

        cx += 8;

        match chroma_subsampling {
            YuvChromaSample::YUV420 | YuvChromaSample::YUV422 => {
                ux += 4;
            }
            YuvChromaSample::YUV444 => {
                ux += 8;
            }
        }
    }

    ProcessedOffset { cx, ux }
}
