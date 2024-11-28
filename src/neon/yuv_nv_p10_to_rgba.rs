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
    CbCrInverseTransform, YuvBytesPacking, YuvChromaRange, YuvChromaSubsampling, YuvEndianness,
    YuvNVOrder, YuvSourceChannels,
};

pub(crate) unsafe fn neon_yuv_nv12_p10_to_rgba_row<
    const DESTINATION_CHANNELS: u8,
    const NV_ORDER: u8,
    const SAMPLING: u8,
    const ENDIANNESS: u8,
    const BYTES_POSITION: u8,
>(
    y_ld_ptr: &[u16],
    uv_ld_ptr: &[u16],
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
    let chroma_subsampling: YuvChromaSubsampling = SAMPLING.into();
    let endianness: YuvEndianness = ENDIANNESS.into();
    let bytes_position: YuvBytesPacking = BYTES_POSITION.into();
    let dst_ptr = bgra.as_mut_ptr();

    let bias_y = range.bias_y as i32;
    let bias_uv = range.bias_uv as i32;

    let y_corr = vdupq_n_u16(bias_y as u16);
    let uv_corr_q = vdupq_n_s16(bias_uv as i16);

    let weights_arr: [i16; 8] = [
        transform.y_coef as i16,
        transform.cr_coef as i16,
        transform.cb_coef as i16,
        -transform.g_coeff_1 as i16,
        -transform.g_coeff_2 as i16,
        0,
        0,
        0,
    ];

    let v_weights = vld1q_s16(weights_arr.as_ptr());

    let v_alpha = vdup_n_u8(255u8);

    let mut cx = start_cx;
    let mut ux = start_ux;

    while cx + 8 < width as usize {
        let u_high: int16x4_t;
        let v_high: int16x4_t;
        let u_low: int16x4_t;
        let v_low: int16x4_t;

        let mut y_vl = vld1q_u16(y_ld_ptr.get_unchecked(cx..).as_ptr());
        if endianness == YuvEndianness::BigEndian {
            y_vl = vreinterpretq_u16_u8(vrev16q_u8(vreinterpretq_u8_u16(y_vl)));
        }
        if bytes_position == YuvBytesPacking::MostSignificantBytes {
            y_vl = vshrq_n_u16::<6>(y_vl);
        }

        let y_values: int16x8_t = vreinterpretq_s16_u16(vqsubq_u16(y_vl, y_corr));

        match chroma_subsampling {
            YuvChromaSubsampling::Yuv420 | YuvChromaSubsampling::Yuv422 => {
                let mut uv_values_u = vld2_u16(uv_ld_ptr.get_unchecked(ux..).as_ptr());

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
                let u_values_c = vsub_s16(vreinterpret_s16_u16(u_vl), vget_low_s16(uv_corr_q));
                let v_values_c = vsub_s16(vreinterpret_s16_u16(v_vl), vget_low_s16(uv_corr_q));

                u_high = vzip2_s16(u_values_c, u_values_c);
                v_high = vzip2_s16(v_values_c, v_values_c);
                u_low = vzip1_s16(u_values_c, u_values_c);
                v_low = vzip1_s16(v_values_c, v_values_c);
            }
            YuvChromaSubsampling::Yuv444 => {
                let mut uv_values_u = vld2q_u16(uv_ld_ptr.get_unchecked(ux..).as_ptr());

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

        let y_high = vmull_high_laneq_s16::<0>(y_values, v_weights);

        let r_high = vrshrn_n_s32::<6>(vmlal_laneq_s16::<1>(y_high, v_high, v_weights));
        let b_high = vrshrn_n_s32::<6>(vmlal_laneq_s16::<2>(y_high, u_high, v_weights));
        let g_high = vshrn_n_s32::<6>(vmlal_laneq_s16::<4>(
            vmlal_laneq_s16::<3>(y_high, v_high, v_weights),
            u_high,
            v_weights,
        ));

        let y_low = vmull_laneq_s16::<0>(vget_low_s16(y_values), v_weights);

        let r_low = vshrn_n_s32::<6>(vmlal_laneq_s16::<1>(y_low, v_low, v_weights));
        let b_low = vshrn_n_s32::<6>(vmlal_laneq_s16::<2>(y_low, u_low, v_weights));
        let g_low = vshrn_n_s32::<6>(vmlal_laneq_s16::<4>(
            vmlal_laneq_s16::<3>(y_low, v_low, v_weights),
            u_low,
            v_weights,
        ));

        let r_values = vqrshrun_n_s16::<2>(vcombine_s16(r_low, r_high));
        let g_values = vqrshrun_n_s16::<2>(vcombine_s16(g_low, g_high));
        let b_values = vqrshrun_n_s16::<2>(vcombine_s16(b_low, b_high));

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
            YuvChromaSubsampling::Yuv420 | YuvChromaSubsampling::Yuv422 => {
                ux += 8;
            }
            YuvChromaSubsampling::Yuv444 => {
                ux += 16;
            }
        }
    }

    ProcessedOffset { cx, ux }
}
