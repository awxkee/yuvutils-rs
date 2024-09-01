/*
 * // Copyright (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

use std::arch::aarch64::*;

use crate::internals::ProcessedOffset;
use crate::yuv_support::{
    CbCrInverseTransform, YuvBytesPacking, YuvChromaRange, YuvChromaSample, YuvEndianness,
    YuvNVOrder, YuvSourceChannels,
};

pub unsafe fn neon_yuv_nv_p16_to_rgba_row<
    const DESTINATION_CHANNELS: u8,
    const NV_ORDER: u8,
    const SAMPLING: u8,
    const ENDIANNESS: u8,
    const BYTES_POSITION: u8,
    const BIT_DEPTH: u8,
>(
    y_ld_ptr: *const u16,
    uv_ld_ptr: *const u16,
    bgra: *mut u16,
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
    let cr_coef = transform.cr_coef;
    let cb_coef = transform.cb_coef;
    let y_coef = transform.y_coef;
    let g_coef_1 = transform.g_coeff_1;
    let g_coef_2 = transform.g_coeff_2;

    let bias_y = range.bias_y as i32;
    let bias_uv = range.bias_uv as i32;

    let mut dst_ptr = bgra;

    let v_max_colors = vdupq_n_u16((1u16 << BIT_DEPTH as u16) - 1);

    let y_corr = vdupq_n_s16(bias_y as i16);
    let uv_corr = vdup_n_s16(bias_uv as i16);
    let uv_corr_q = vdupq_n_s16(bias_uv as i16);
    let v_luma_coeff = vdupq_n_s16(y_coef as i16);
    let v_cr_coeff = vdup_n_s16(cr_coef as i16);
    let v_cb_coeff = vdup_n_s16(cb_coef as i16);
    let v_min_values = vdupq_n_s16(0i16);
    let v_g_coeff_1 = vdup_n_s16(-1i16 * (g_coef_1 as i16));
    let v_g_coeff_2 = vdup_n_s16(-1i16 * (g_coef_2 as i16));

    let mut cx = start_cx;
    let mut ux = start_ux;

    let v_big_shift_count = vdupq_n_s16(-(16i16 - BIT_DEPTH as i16));

    while cx + 8 < width as usize {
        let y_values: int16x8_t;

        let u_high: int16x4_t;
        let v_high: int16x4_t;
        let u_low: int16x4_t;
        let v_low: int16x4_t;

        let mut y_vl = vld1q_u16(y_ld_ptr.add(cx));
        if endianness == YuvEndianness::BigEndian {
            y_vl = vreinterpretq_u16_u8(vrev16q_u8(vreinterpretq_u8_u16(y_vl)));
        }
        if bytes_position == YuvBytesPacking::MostSignificantBytes {
            y_vl = vshlq_u16(y_vl, v_big_shift_count);
        }
        y_values = vsubq_s16(vreinterpretq_s16_u16(y_vl), y_corr);

        match chroma_subsampling {
            YuvChromaSample::YUV420 | YuvChromaSample::YUV422 => {
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
                    u_vl = vshl_u16(u_vl, vget_low_s16(v_big_shift_count));
                    v_vl = vshl_u16(v_vl, vget_low_s16(v_big_shift_count));
                }
                let u_values_c = vsub_s16(vreinterpret_s16_u16(u_vl), uv_corr);
                let v_values_c = vsub_s16(vreinterpret_s16_u16(v_vl), uv_corr);

                u_high = vzip2_s16(u_values_c, u_values_c);
                v_high = vzip2_s16(v_values_c, v_values_c);
                u_low = vzip1_s16(u_values_c, u_values_c);
                v_low = vzip1_s16(v_values_c, v_values_c);
            }
            YuvChromaSample::YUV444 => {
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
                    u_vl = vshlq_u16(u_vl, v_big_shift_count);
                    v_vl = vshlq_u16(v_vl, v_big_shift_count);
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

        let r_high = vshrn_n_s32::<6>(vmlal_s16(y_high, v_high, v_cr_coeff));
        let b_high = vshrn_n_s32::<6>(vmlal_s16(y_high, u_high, v_cb_coeff));
        let g_high = vshrn_n_s32::<6>(vmlal_s16(
            vmlal_s16(y_high, v_high, v_g_coeff_1),
            u_high,
            v_g_coeff_2,
        ));

        let y_low = vmull_s16(vget_low_s16(y_values), vget_low_s16(v_luma_coeff));

        let r_low = vshrn_n_s32::<6>(vmlal_s16(y_low, v_low, v_cr_coeff));
        let b_low = vshrn_n_s32::<6>(vmlal_s16(y_low, u_low, v_cb_coeff));
        let g_low = vshrn_n_s32::<6>(vmlal_s16(
            vmlal_s16(y_low, v_low, v_g_coeff_1),
            u_low,
            v_g_coeff_2,
        ));

        let r_values = vminq_u16(
            vreinterpretq_u16_s16(vmaxq_s16(vcombine_s16(r_low, r_high), v_min_values)),
            v_max_colors,
        );
        let g_values = vminq_u16(
            vreinterpretq_u16_s16(vmaxq_s16(vcombine_s16(g_low, g_high), v_min_values)),
            v_max_colors,
        );
        let b_values = vminq_u16(
            vreinterpretq_u16_s16(vmaxq_s16(vcombine_s16(b_low, b_high), v_min_values)),
            v_max_colors,
        );

        match destination_channels {
            YuvSourceChannels::Rgb => {
                let dst_pack = uint16x8x3_t(r_values, g_values, b_values);
                vst3q_u16(dst_ptr, dst_pack);
            }
            YuvSourceChannels::Bgr => {
                let dst_pack = uint16x8x3_t(b_values, g_values, r_values);
                vst3q_u16(dst_ptr, dst_pack);
            }
            YuvSourceChannels::Rgba => {
                let dst_pack = uint16x8x4_t(r_values, g_values, b_values, v_max_colors);
                vst4q_u16(dst_ptr, dst_pack);
            }
            YuvSourceChannels::Bgra => {
                let dst_pack = uint16x8x4_t(b_values, g_values, r_values, v_max_colors);
                vst4q_u16(dst_ptr, dst_pack);
            }
        }

        cx += 8;
        dst_ptr = dst_ptr.add(8 * channels);

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
