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
use crate::neon::neon_simd_support::neon_store_half_rgb8;
use crate::yuv_support::{
    CbCrInverseTransform, YuvBytesPacking, YuvChromaRange, YuvChromaSubsampling, YuvEndianness,
    YuvSourceChannels,
};

#[inline(always)]
pub(crate) unsafe fn neon_yuv_p16_to_rgba_alpha_row<
    const DESTINATION_CHANNELS: u8,
    const SAMPLING: u8,
    const ENDIANNESS: u8,
    const BYTES_POSITION: u8,
    const BIT_DEPTH: usize,
    const PRECISION: i32,
>(
    y_ld_ptr: &[u16],
    u_ld_ptr: &[u16],
    v_ld_ptr: &[u16],
    a_ld_ptr: &[u16],
    rgba: &mut [u8],
    width: u32,
    range: &YuvChromaRange,
    transform: &CbCrInverseTransform<i32>,
    start_cx: usize,
    start_ux: usize,
) -> ProcessedOffset {
    let destination_channels: YuvSourceChannels = DESTINATION_CHANNELS.into();
    if destination_channels == YuvSourceChannels::Rgb
        || destination_channels == YuvSourceChannels::Bgr
    {
        unreachable!("Cannot call YUV p16 to Rgb8 with alpha without real alpha");
    }
    let channels = destination_channels.get_channels_count();
    let chroma_subsampling: YuvChromaSubsampling = SAMPLING.into();
    let endianness: YuvEndianness = ENDIANNESS.into();
    let bytes_position: YuvBytesPacking = BYTES_POSITION.into();
    let dst_ptr = rgba.as_mut_ptr();

    let y_corr = vdupq_n_s16(range.bias_y as i16);
    let uv_corr = vdup_n_s16(range.bias_uv as i16);
    let v_msb_shift = vdupq_n_s16(BIT_DEPTH as i16 - 16);
    let v_store_shift = vdupq_n_s16(8 - (BIT_DEPTH as i16));

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

    let mut cx = start_cx;
    let mut ux = start_ux;

    while cx + 8 < width as usize {
        let y_values: int16x8_t;

        let u_values_c: int16x4_t;
        let v_values_c: int16x4_t;

        let u_values_l = vld1_u16(u_ld_ptr.get_unchecked(ux..).as_ptr());
        let v_values_l = vld1_u16(v_ld_ptr.get_unchecked(ux..).as_ptr());
        let mut a_values_l = vld1q_u16(a_ld_ptr.get_unchecked(cx..).as_ptr());

        if endianness == YuvEndianness::BigEndian {
            a_values_l = vreinterpretq_u16_u8(vrev16q_u8(vreinterpretq_u8_u16(a_values_l)));
        }

        if bytes_position == YuvBytesPacking::MostSignificantBytes {
            a_values_l = vshlq_u16(a_values_l, v_msb_shift);
        }

        match endianness {
            YuvEndianness::BigEndian => {
                let mut y_u_values = vreinterpretq_u16_u8(vrev16q_u8(vreinterpretq_u8_u16(
                    vld1q_u16(y_ld_ptr.get_unchecked(cx..).as_ptr()),
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
                let mut y_vl = vld1q_u16(y_ld_ptr.get_unchecked(cx..).as_ptr());
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

        let y_high = vmull_high_laneq_s16::<0>(y_values, v_weights);

        let r_high = vqrshrun_n_s32::<PRECISION>(vmlal_laneq_s16::<1>(y_high, v_high, v_weights));
        let b_high = vqrshrun_n_s32::<PRECISION>(vmlal_laneq_s16::<2>(y_high, u_high, v_weights));
        let g_high = vqrshrun_n_s32::<PRECISION>(vmlal_laneq_s16::<4>(
            vmlal_laneq_s16::<3>(y_high, v_high, v_weights),
            u_high,
            v_weights,
        ));

        let y_low = vmull_laneq_s16::<0>(vget_low_s16(y_values), v_weights);
        let u_low = vzip1_s16(u_values_c, u_values_c);
        let v_low = vzip1_s16(v_values_c, v_values_c);

        let r_low = vqrshrun_n_s32::<PRECISION>(vmlal_laneq_s16::<1>(y_low, v_low, v_weights));
        let b_low = vqrshrun_n_s32::<PRECISION>(vmlal_laneq_s16::<2>(y_low, u_low, v_weights));
        let g_low = vqrshrun_n_s32::<PRECISION>(vmlal_laneq_s16::<4>(
            vmlal_laneq_s16::<3>(y_low, v_low, v_weights),
            u_low,
            v_weights,
        ));

        let r_values = vqmovn_u16(vqshlq_u16(vcombine_u16(r_low, r_high), v_store_shift));
        let g_values = vqmovn_u16(vqshlq_u16(vcombine_u16(g_low, g_high), v_store_shift));
        let b_values = vqmovn_u16(vqshlq_u16(vcombine_u16(b_low, b_high), v_store_shift));

        let v_alpha = vqmovn_u16(vshlq_u16(a_values_l, v_store_shift));

        neon_store_half_rgb8::<DESTINATION_CHANNELS>(
            dst_ptr.add(cx * channels),
            r_values,
            g_values,
            b_values,
            v_alpha,
        );

        cx += 8;

        match chroma_subsampling {
            YuvChromaSubsampling::Yuv420 | YuvChromaSubsampling::Yuv422 => {
                ux += 4;
            }
            YuvChromaSubsampling::Yuv444 => {
                ux += 8;
            }
        }
    }

    ProcessedOffset { cx, ux }
}
