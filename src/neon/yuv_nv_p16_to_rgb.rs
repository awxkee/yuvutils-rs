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
use crate::neon::utils::{neon_store_rgb16, vexpand_high_bp_by_2, vfrommsb_u16, vfrommsbq_u16};
use crate::yuv_support::{
    to_channels_layout, to_subsampling, CbCrInverseTransform, YuvBytesPacking, YuvChromaRange,
    YuvChromaSubsampling, YuvEndianness, YuvNVOrder, YuvSourceChannels,
};

pub(crate) unsafe fn neon_yuv_nv_p16_to_rgba_row<
    const DESTINATION_CHANNELS: u8,
    const NV_ORDER: u8,
    const SAMPLING: u8,
    const ENDIANNESS: u8,
    const BYTES_POSITION: u8,
    const BIT_DEPTH: usize,
    const PRECISION: i32,
>(
    y_ld_ptr: &[u16],
    uv_ld_ptr: &[u16],
    bgra: &mut [u16],
    width: u32,
    range: &YuvChromaRange,
    transform: &CbCrInverseTransform<i32>,
    start_cx: usize,
    start_ux: usize,
) -> ProcessedOffset {
    let destination_channels: YuvSourceChannels = to_channels_layout(DESTINATION_CHANNELS);
    let channels = destination_channels.get_channels_count();
    let uv_order: YuvNVOrder = NV_ORDER.into();
    let chroma_subsampling: YuvChromaSubsampling = to_subsampling(SAMPLING);
    let endianness: YuvEndianness = ENDIANNESS.into();
    let bytes_position: YuvBytesPacking = BYTES_POSITION.into();

    let bias_y = range.bias_y as i32;
    let bias_uv = range.bias_uv as i32;

    let v_max_colors = vdupq_n_u16((1u16 << BIT_DEPTH as u16) - 1);

    let y_corr = vdupq_n_s16(bias_y as i16);
    let uv_corr = vdupq_n_s16(bias_uv as i16);

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
        let dst_ptr = bgra.get_unchecked_mut(cx * channels..).as_mut_ptr();

        let u_high: int16x4_t;
        let v_high: int16x4_t;
        let u_low: int16x4_t;
        let v_low: int16x4_t;

        let mut y_vl = vld1q_u16(y_ld_ptr.get_unchecked(cx..).as_ptr());
        if endianness == YuvEndianness::BigEndian {
            y_vl = vreinterpretq_u16_u8(vrev16q_u8(vreinterpretq_u8_u16(y_vl)));
        }
        if bytes_position == YuvBytesPacking::MostSignificantBytes {
            y_vl = vfrommsbq_u16::<BIT_DEPTH>(y_vl);
        }

        let y_values: int16x8_t = vsubq_s16(vreinterpretq_s16_u16(y_vl), y_corr);

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
                    u_vl = vfrommsb_u16::<BIT_DEPTH>(u_vl);
                    v_vl = vfrommsb_u16::<BIT_DEPTH>(v_vl);
                }
                let u_values_c = vsub_s16(vreinterpret_s16_u16(u_vl), vget_low_s16(uv_corr));
                let v_values_c = vsub_s16(vreinterpret_s16_u16(v_vl), vget_low_s16(uv_corr));

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
                    u_vl = vfrommsbq_u16::<BIT_DEPTH>(u_vl);
                    v_vl = vfrommsbq_u16::<BIT_DEPTH>(v_vl);
                }
                let u_values_c = vsubq_s16(vreinterpretq_s16_u16(u_vl), uv_corr);
                let v_values_c = vsubq_s16(vreinterpretq_s16_u16(v_vl), uv_corr);
                u_high = vget_high_s16(u_values_c);
                v_high = vget_high_s16(v_values_c);
                u_low = vget_low_s16(u_values_c);
                v_low = vget_low_s16(v_values_c);
            }
        }

        let y_high = vmull_high_laneq_s16::<0>(y_values, v_weights);

        let r_high = vqrshrun_n_s32::<PRECISION>(vmlal_laneq_s16::<1>(y_high, v_high, v_weights));
        let b_high = vqrshrun_n_s32::<PRECISION>(vmlal_laneq_s16::<2>(y_high, u_high, v_weights));
        let g_high = vqrshrun_n_s32::<PRECISION>(vmlal_laneq_s16::<4>(
            vmlal_laneq_s16::<3>(y_high, v_high, v_weights),
            u_high,
            v_weights,
        ));

        let y_low = vmull_laneq_s16::<0>(vget_low_s16(y_values), v_weights);

        let r_low = vqrshrun_n_s32::<PRECISION>(vmlal_laneq_s16::<1>(y_low, v_low, v_weights));
        let b_low = vqrshrun_n_s32::<PRECISION>(vmlal_laneq_s16::<2>(y_low, u_low, v_weights));
        let g_low = vqrshrun_n_s32::<PRECISION>(vmlal_laneq_s16::<4>(
            vmlal_laneq_s16::<3>(y_low, v_low, v_weights),
            u_low,
            v_weights,
        ));

        let r_values = vminq_u16(vcombine_u16(r_low, r_high), v_max_colors);
        let g_values = vminq_u16(vcombine_u16(g_low, g_high), v_max_colors);
        let b_values = vminq_u16(vcombine_u16(b_low, b_high), v_max_colors);

        neon_store_rgb16::<DESTINATION_CHANNELS>(
            dst_ptr,
            r_values,
            g_values,
            b_values,
            v_max_colors,
        );
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

#[target_feature(enable = "rdm")]
pub(crate) unsafe fn neon_yuv_nv_p16_to_rgba_row_rdm<
    const DESTINATION_CHANNELS: u8,
    const NV_ORDER: u8,
    const SAMPLING: u8,
    const ENDIANNESS: u8,
    const BYTES_POSITION: u8,
    const BIT_DEPTH: usize,
    const PRECISION: i32,
>(
    y_ld_ptr: &[u16],
    uv_ld_ptr: &[u16],
    bgra: &mut [u16],
    width: u32,
    range: &YuvChromaRange,
    transform: &CbCrInverseTransform<i32>,
    start_cx: usize,
    start_ux: usize,
) -> ProcessedOffset {
    let destination_channels: YuvSourceChannels = to_channels_layout(DESTINATION_CHANNELS);
    let channels = destination_channels.get_channels_count();
    let uv_order: YuvNVOrder = NV_ORDER.into();
    let chroma_subsampling: YuvChromaSubsampling = to_subsampling(SAMPLING);
    let endianness: YuvEndianness = ENDIANNESS.into();
    let bytes_position: YuvBytesPacking = BYTES_POSITION.into();

    let bias_y = range.bias_y as i32;
    let bias_uv = range.bias_uv as i32;

    let v_max_colors = vdupq_n_u16((1u16 << BIT_DEPTH as u16) - 1);

    let y_corr = vdupq_n_u16(bias_y as u16);
    let uv_corr = vdupq_n_s16(bias_uv as i16);

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

    let zeros = vdupq_n_s16(0);

    let mut cx = start_cx;
    let mut ux = start_ux;

    const V_SCALE: i32 = 2;

    while cx + 8 < width as usize {
        let dst_ptr = bgra.get_unchecked_mut(cx * channels..).as_mut_ptr();

        let u_values: int16x8_t;
        let v_values: int16x8_t;

        let mut y_vl = vld1q_u16(y_ld_ptr.get_unchecked(cx..).as_ptr());
        if endianness == YuvEndianness::BigEndian {
            y_vl = vreinterpretq_u16_u8(vrev16q_u8(vreinterpretq_u8_u16(y_vl)));
        }
        if bytes_position == YuvBytesPacking::MostSignificantBytes {
            y_vl = vfrommsbq_u16::<BIT_DEPTH>(y_vl);
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
                    u_vl = vfrommsb_u16::<BIT_DEPTH>(u_vl);
                    v_vl = vfrommsb_u16::<BIT_DEPTH>(v_vl);
                }
                let u_values_c = vsub_s16(vreinterpret_s16_u16(u_vl), vget_low_s16(uv_corr));
                let v_values_c = vsub_s16(vreinterpret_s16_u16(v_vl), vget_low_s16(uv_corr));

                let u_high = vzip2_s16(u_values_c, u_values_c);
                let v_high = vzip2_s16(v_values_c, v_values_c);
                let u_low = vzip1_s16(u_values_c, u_values_c);
                let v_low = vzip1_s16(v_values_c, v_values_c);

                u_values = vshlq_n_s16::<V_SCALE>(vcombine_s16(u_low, u_high));
                v_values = vshlq_n_s16::<V_SCALE>(vcombine_s16(v_low, v_high));
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
                    u_vl = vfrommsbq_u16::<BIT_DEPTH>(u_vl);
                    v_vl = vfrommsbq_u16::<BIT_DEPTH>(v_vl);
                }
                let u_values_c = vsubq_s16(vreinterpretq_s16_u16(u_vl), uv_corr);
                let v_values_c = vsubq_s16(vreinterpretq_s16_u16(v_vl), uv_corr);
                let u_high = vget_high_s16(u_values_c);
                let v_high = vget_high_s16(v_values_c);
                let u_low = vget_low_s16(u_values_c);
                let v_low = vget_low_s16(v_values_c);

                u_values = vshlq_n_s16::<V_SCALE>(vcombine_s16(u_low, u_high));
                v_values = vshlq_n_s16::<V_SCALE>(vcombine_s16(v_low, v_high));
            }
        }

        let y_high =
            vqrdmulhq_laneq_s16::<0>(vexpand_high_bp_by_2::<BIT_DEPTH>(y_values), v_weights);

        let r_vals = vqrdmlahq_laneq_s16::<1>(y_high, v_values, v_weights);
        let b_vals = vqrdmlahq_laneq_s16::<2>(y_high, u_values, v_weights);
        let g_vals = vqrdmlahq_laneq_s16::<4>(
            vqrdmlahq_laneq_s16::<3>(y_high, v_values, v_weights),
            u_values,
            v_weights,
        );

        let r_values = vminq_u16(
            vreinterpretq_u16_s16(vmaxq_s16(r_vals, zeros)),
            v_max_colors,
        );
        let g_values = vminq_u16(
            vreinterpretq_u16_s16(vmaxq_s16(g_vals, zeros)),
            v_max_colors,
        );
        let b_values = vminq_u16(
            vreinterpretq_u16_s16(vmaxq_s16(b_vals, zeros)),
            v_max_colors,
        );

        neon_store_rgb16::<DESTINATION_CHANNELS>(
            dst_ptr,
            r_values,
            g_values,
            b_values,
            v_max_colors,
        );

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
