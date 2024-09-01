/*
 * // Copyright (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::internals::ProcessedOffset;
use crate::sse::{_mm_deinterleave_x2_epi16, _mm_interleave_rgb_epi16, _mm_interleave_rgba_epi16};
use crate::yuv_support::{
    CbCrInverseTransform, YuvBytesPacking, YuvChromaRange, YuvChromaSample, YuvEndianness,
    YuvNVOrder, YuvSourceChannels,
};

pub unsafe fn sse_yuv_nv_p16_to_rgba_row<
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

    let v_max_colors = _mm_set1_epi16((1i16 << BIT_DEPTH as i16) - 1);

    let y_corr = _mm_set1_epi16(bias_y as i16);
    let uv_corr = _mm_set1_epi16(bias_uv as i16);
    let uv_corr_q = _mm_set1_epi16(bias_uv as i16);
    let v_luma_coeff = _mm_set1_epi16(y_coef as i16);
    let v_cr_coeff = _mm_set1_epi16(cr_coef as i16);
    let v_cb_coeff = _mm_set1_epi16(cb_coef as i16);
    let zeros = _mm_setzero_si128();
    let v_g_coeff_1 = _mm_set1_epi16(-1i16 * (g_coef_1 as i16));
    let v_g_coeff_2 = _mm_set1_epi16(-1i16 * (g_coef_2 as i16));

    let mut cx = start_cx;
    let mut ux = start_ux;

    let v_big_shift_count = _mm_set1_epi64x(16i64 - BIT_DEPTH as i64);

    let big_endian_shuffle_flag =
        _mm_setr_epi8(1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14);

    while cx + 8 < width as usize {
        let y_values;

        let u_high;
        let v_high;
        let u_low;
        let v_low;

        let mut y_vl = _mm_loadu_si128(y_ld_ptr.add(cx) as *const __m128i);
        if endianness == YuvEndianness::BigEndian {
            y_vl = _mm_shuffle_epi8(y_vl, big_endian_shuffle_flag);
        }
        if bytes_position == YuvBytesPacking::MostSignificantBytes {
            y_vl = _mm_srl_epi16(y_vl, v_big_shift_count);
        }
        y_values = _mm_sub_epi16(y_vl, y_corr);

        match chroma_subsampling {
            YuvChromaSample::YUV420 | YuvChromaSample::YUV422 => {
                let uv_ld = uv_ld_ptr.add(ux);

                let row0 = _mm_loadu_si128(uv_ld as *const __m128i);

                let mut uv_values_u = _mm_deinterleave_x2_epi16(row0, zeros);

                if uv_order == YuvNVOrder::VU {
                    uv_values_u = (uv_values_u.1, uv_values_u.0);
                }

                let mut u_vl = uv_values_u.0;
                if endianness == YuvEndianness::BigEndian {
                    u_vl = _mm_shuffle_epi8(u_vl, big_endian_shuffle_flag);
                }
                let mut v_vl = uv_values_u.1;
                if endianness == YuvEndianness::BigEndian {
                    v_vl = _mm_shuffle_epi8(v_vl, big_endian_shuffle_flag);
                }
                if bytes_position == YuvBytesPacking::MostSignificantBytes {
                    u_vl = _mm_srl_epi16(u_vl, v_big_shift_count);
                    v_vl = _mm_srl_epi16(v_vl, v_big_shift_count);
                }
                let u_values_c = _mm_sub_epi16(u_vl, uv_corr);
                let v_values_c = _mm_sub_epi16(v_vl, uv_corr);

                let u_values_32 = _mm_cvtepi16_epi32(u_values_c);
                let v_values_32 = _mm_cvtepi16_epi32(v_values_c);

                u_high = _mm_unpackhi_epi32(u_values_32, u_values_32);
                v_high = _mm_unpackhi_epi32(v_values_32, v_values_32);
                u_low = _mm_unpacklo_epi32(u_values_32, u_values_32);
                v_low = _mm_unpacklo_epi32(v_values_32, v_values_32);
            }
            YuvChromaSample::YUV444 => {
                let uv_ld = uv_ld_ptr.add(ux);
                let row0 = _mm_loadu_si128(uv_ld as *const __m128i);
                let row1 = _mm_loadu_si128(uv_ld.add(8) as *const __m128i);
                let mut uv_values_u = _mm_deinterleave_x2_epi16(row0, row1);

                if uv_order == YuvNVOrder::VU {
                    uv_values_u = (uv_values_u.1, uv_values_u.0);
                }
                let mut u_vl = uv_values_u.0;
                if endianness == YuvEndianness::BigEndian {
                    u_vl = _mm_shuffle_epi8(u_vl, big_endian_shuffle_flag);
                }
                let mut v_vl = uv_values_u.1;
                if endianness == YuvEndianness::BigEndian {
                    v_vl = _mm_shuffle_epi8(v_vl, big_endian_shuffle_flag);
                }
                if bytes_position == YuvBytesPacking::MostSignificantBytes {
                    u_vl = _mm_srl_epi16(u_vl, v_big_shift_count);
                    v_vl = _mm_srl_epi16(v_vl, v_big_shift_count);
                }
                let u_values_c = _mm_sub_epi16(u_vl, uv_corr_q);
                let v_values_c = _mm_sub_epi16(v_vl, uv_corr_q);
                u_high = _mm_cvtepi16_epi32(_mm_slli_si128::<8>(u_values_c));
                v_high = _mm_cvtepi16_epi32(_mm_slli_si128::<8>(v_values_c));
                u_low = _mm_cvtepi16_epi32(u_values_c);
                v_low = _mm_cvtepi16_epi32(v_values_c);
            }
        }

        let y_high = _mm_madd_epi16(_mm_unpackhi_epi16(y_values, zeros), v_luma_coeff);

        let r_high = _mm_srai_epi32::<6>(_mm_add_epi32(y_high, _mm_madd_epi16(v_high, v_cr_coeff)));
        let b_high = _mm_srai_epi32::<6>(_mm_add_epi32(y_high, _mm_madd_epi16(u_high, v_cb_coeff)));
        let g_high = _mm_srai_epi32::<6>(_mm_add_epi32(
            _mm_add_epi32(y_high, _mm_madd_epi16(v_high, v_g_coeff_1)),
            _mm_madd_epi16(u_high, v_g_coeff_2),
        ));

        let y_low = _mm_madd_epi16(_mm_unpacklo_epi16(y_values, zeros), v_luma_coeff);

        let r_low = _mm_srai_epi32::<6>(_mm_add_epi32(y_low, _mm_madd_epi16(v_low, v_cr_coeff)));
        let b_low = _mm_srai_epi32::<6>(_mm_add_epi32(y_low, _mm_madd_epi16(u_low, v_cb_coeff)));
        let g_low = _mm_srai_epi32::<6>(_mm_add_epi32(
            _mm_add_epi32(y_low, _mm_madd_epi16(v_low, v_g_coeff_1)),
            _mm_madd_epi16(u_low, v_g_coeff_2),
        ));

        let r_values = _mm_min_epi16(
            _mm_max_epi16(_mm_packs_epi32(r_low, r_high), zeros),
            v_max_colors,
        );
        let g_values = _mm_min_epi16(
            _mm_max_epi16(_mm_packs_epi32(g_low, g_high), zeros),
            v_max_colors,
        );
        let b_values = _mm_min_epi16(
            _mm_max_epi16(_mm_packs_epi32(b_low, b_high), zeros),
            v_max_colors,
        );

        match destination_channels {
            YuvSourceChannels::Rgb => {
                let dst_pack = _mm_interleave_rgb_epi16(r_values, g_values, b_values);
                _mm_storeu_si128(dst_ptr as *mut __m128i, dst_pack.0);
                _mm_storeu_si128(dst_ptr.add(8) as *mut __m128i, dst_pack.1);
                _mm_storeu_si128(dst_ptr.add(16) as *mut __m128i, dst_pack.2);
            }
            YuvSourceChannels::Bgr => {
                let dst_pack = _mm_interleave_rgb_epi16(b_values, g_values, r_values);
                _mm_storeu_si128(dst_ptr as *mut __m128i, dst_pack.0);
                _mm_storeu_si128(dst_ptr.add(8) as *mut __m128i, dst_pack.1);
                _mm_storeu_si128(dst_ptr.add(16) as *mut __m128i, dst_pack.2);
            }
            YuvSourceChannels::Rgba => {
                let dst_pack =
                    _mm_interleave_rgba_epi16(r_values, g_values, b_values, v_max_colors);
                _mm_storeu_si128(dst_ptr as *mut __m128i, dst_pack.0);
                _mm_storeu_si128(dst_ptr.add(8) as *mut __m128i, dst_pack.1);
                _mm_storeu_si128(dst_ptr.add(16) as *mut __m128i, dst_pack.2);
                _mm_storeu_si128(dst_ptr.add(24) as *mut __m128i, dst_pack.3);
            }
            YuvSourceChannels::Bgra => {
                let dst_pack =
                    _mm_interleave_rgba_epi16(b_values, g_values, r_values, v_max_colors);
                _mm_storeu_si128(dst_ptr as *mut __m128i, dst_pack.0);
                _mm_storeu_si128(dst_ptr.add(8) as *mut __m128i, dst_pack.1);
                _mm_storeu_si128(dst_ptr.add(16) as *mut __m128i, dst_pack.2);
                _mm_storeu_si128(dst_ptr.add(24) as *mut __m128i, dst_pack.3);
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