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
    YuvNVOrder, YuvSourceChannels,
};

#[inline(always)]
pub unsafe fn neon_yuv_nv12_p10_to_rgba_row<
    const DESTINATION_CHANNELS: u8,
    const NV_ORDER: u8,
    const SAMPLING: u8,
    const ENDIANNESS: u8,
    const BYTES_POSITION: u8,
>(
    y_ld_ptr: *const u16,
    uv_ld_ptr: *const u16,
    bgra: &mut [u8],
    dst_offset: usize,
    width: u32,
    range: &YuvChromaRange,
    transform: &CbCrInverseTransform<i32>,
    start_cx: usize,
    start_ux: usize,
) -> ProcessedOffset {
    let destination_channels: YuvSourceChannels = DESTINATION_CHANNELS.into();
    let channels = destination_channels.get_channels_count();
    let uv_order: YuvNVOrder = NV_ORDER.into();
    let chroma_subsampling: YuvChromaSample = SAMPLING.into();
    let endianness: YuvEndianness = ENDIANNESS.into();
    let bytes_position: YuvBytesPacking = BYTES_POSITION.into();
    let dst_ptr = bgra.as_mut_ptr();
    let cr_coef = transform.cr_coef;
    let cb_coef = transform.cb_coef;
    let y_coef = transform.y_coef;
    let g_coef_1 = transform.g_coeff_1;
    let g_coef_2 = transform.g_coeff_2;

    let bias_y = range.bias_y as i32;
    let bias_uv = range.bias_uv as i32;

    let y_corr = vdupq_n_s16(bias_y as i16);
    let uv_corr = vdup_n_s16(bias_uv as i16);
    let uv_corr_q = vdupq_n_s16(bias_uv as i16);
    let v_luma_coeff = vdupq_n_s16(y_coef as i16);
    let v_cr_coeff = vdup_n_s16(cr_coef as i16);
    let v_cb_coeff = vdup_n_s16(cb_coef as i16);
    let v_min_values = vdupq_n_s16(0i16);
    let v_g_coeff_1 = vdup_n_s16(-(g_coef_1 as i16));
    let v_g_coeff_2 = vdup_n_s16(-(g_coef_2 as i16));
    let v_alpha = vdup_n_u8(255u8);
    let rounding_const = vdupq_n_s32(1 << 5);

    let mut cx = start_cx;
    let mut ux = start_ux;

    while cx + 8 < width as usize {
        let u_high: int16x4_t;
        let v_high: int16x4_t;
        let u_low: int16x4_t;
        let v_low: int16x4_t;

        let mut y_vl = vld1q_u16(y_ld_ptr.add(cx));
        if endianness == YuvEndianness::BigEndian {
            y_vl = vreinterpretq_u16_u8(vrev16q_u8(vreinterpretq_u8_u16(y_vl)));
        }
        if bytes_position == YuvBytesPacking::MostSignificantBytes {
            y_vl = vshrq_n_u16::<6>(y_vl);
        }
        let y_values: int16x8_t = vsubq_s16(vreinterpretq_s16_u16(y_vl), y_corr);

        match chroma_subsampling {
            YuvChromaSample::Yuv420 | YuvChromaSample::Yuv422 => {
                let mut uv_values_u = vld2_u16(uv_ld_ptr.add(ux));

                if uv_order == YuvNVOrder::VU {
                    uv_values_u = uint16x4x2_t(uv_values_u.1, uv_values_u.0);
                }

                let mut u_vl = uv_values_u.0;
                if endianness == YuvEndianness::BigEndian {
                    u_vl = vreinterpret_u16_u8(vrev16_u8(vreinterpret_u8_u16(u_vl)));
                }
                let mut v_vl = uv_values_u.1;
                if endianness == YuvEndianness::BigEndian {
                    v_vl = vreinterpret_u16_u8(vrev16_u8(vreinterpret_u8_u16(v_vl)));
                }
                if bytes_position == YuvBytesPacking::MostSignificantBytes {
                    u_vl = vshr_n_u16::<6>(u_vl);
                    v_vl = vshr_n_u16::<6>(v_vl);
                }
                let u_values_c = vsub_s16(vreinterpret_s16_u16(u_vl), uv_corr);
                let v_values_c = vsub_s16(vreinterpret_s16_u16(v_vl), uv_corr);

                u_high = vzip2_s16(u_values_c, u_values_c);
                v_high = vzip2_s16(v_values_c, v_values_c);
                u_low = vzip1_s16(u_values_c, u_values_c);
                v_low = vzip1_s16(v_values_c, v_values_c);
            }
            YuvChromaSample::Yuv444 => {
                let mut uv_values_u = vld2q_u16(uv_ld_ptr.add(ux));

                if uv_order == YuvNVOrder::VU {
                    uv_values_u = uint16x8x2_t(uv_values_u.1, uv_values_u.0);
                }
                let mut u_vl = uv_values_u.0;
                if endianness == YuvEndianness::BigEndian {
                    u_vl = vreinterpretq_u16_u8(vrev16q_u8(vreinterpretq_u8_u16(u_vl)));
                }
                let mut v_vl = uv_values_u.1;
                if endianness == YuvEndianness::BigEndian {
                    v_vl = vreinterpretq_u16_u8(vrev16q_u8(vreinterpretq_u8_u16(v_vl)));
                }
                if bytes_position == YuvBytesPacking::MostSignificantBytes {
                    u_vl = vshrq_n_u16::<6>(u_vl);
                    v_vl = vshrq_n_u16::<6>(v_vl);
                }
                let u_values_c = vsubq_s16(vreinterpretq_s16_u16(u_vl), uv_corr_q);
                let v_values_c = vsubq_s16(vreinterpretq_s16_u16(v_vl), uv_corr_q);
                u_high = vget_high_s16(u_values_c);
                v_high = vget_high_s16(v_values_c);
                u_low = vget_low_s16(u_values_c);
                v_low = vget_low_s16(v_values_c);
            }
        }

        let y_high = vmull_high_s16(y_values, v_luma_coeff);

        let r_high = vshrn_n_s32::<6>(vaddq_s32(
            vmlal_s16(y_high, v_high, v_cr_coeff),
            rounding_const,
        ));
        let b_high = vshrn_n_s32::<6>(vaddq_s32(
            vmlal_s16(y_high, u_high, v_cb_coeff),
            rounding_const,
        ));
        let g_high = vshrn_n_s32::<6>(vaddq_s32(
            vmlal_s16(vmlal_s16(y_high, v_high, v_g_coeff_1), u_high, v_g_coeff_2),
            rounding_const,
        ));

        let y_low = vmull_s16(vget_low_s16(y_values), vget_low_s16(v_luma_coeff));

        let r_low = vshrn_n_s32::<6>(vaddq_s32(
            vmlal_s16(y_low, v_low, v_cr_coeff),
            rounding_const,
        ));
        let b_low = vshrn_n_s32::<6>(vaddq_s32(
            vmlal_s16(y_low, u_low, v_cb_coeff),
            rounding_const,
        ));
        let g_low = vshrn_n_s32::<6>(vaddq_s32(
            vmlal_s16(vmlal_s16(y_low, v_low, v_g_coeff_1), u_low, v_g_coeff_2),
            rounding_const,
        ));

        let r_values = vqshrun_n_s16::<2>(vmaxq_s16(vcombine_s16(r_low, r_high), v_min_values));
        let g_values = vqshrun_n_s16::<2>(vmaxq_s16(vcombine_s16(g_low, g_high), v_min_values));
        let b_values = vqshrun_n_s16::<2>(vmaxq_s16(vcombine_s16(b_low, b_high), v_min_values));

        match destination_channels {
            YuvSourceChannels::Rgb => {
                let dst_pack: uint8x8x3_t = uint8x8x3_t(r_values, g_values, b_values);
                vst3_u8(dst_ptr.add(dst_offset + cx * channels), dst_pack);
            }
            YuvSourceChannels::Bgr => {
                let dst_pack: uint8x8x3_t = uint8x8x3_t(b_values, g_values, r_values);
                vst3_u8(dst_ptr.add(dst_offset + cx * channels), dst_pack);
            }
            YuvSourceChannels::Rgba => {
                let dst_pack: uint8x8x4_t = uint8x8x4_t(r_values, g_values, b_values, v_alpha);
                vst4_u8(dst_ptr.add(dst_offset + cx * channels), dst_pack);
            }
            YuvSourceChannels::Bgra => {
                let dst_pack: uint8x8x4_t = uint8x8x4_t(b_values, g_values, r_values, v_alpha);
                vst4_u8(dst_ptr.add(dst_offset + cx * channels), dst_pack);
            }
        }

        cx += 8;

        match chroma_subsampling {
            YuvChromaSample::Yuv420 | YuvChromaSample::Yuv422 => {
                ux += 8;
            }
            YuvChromaSample::Yuv444 => {
                ux += 16;
            }
        }
    }

    ProcessedOffset { cx, ux }
}
