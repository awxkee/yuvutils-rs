/*
 * // Copyright (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

use crate::avx2::avx2_utils::*;
use crate::internals::ProcessedOffset;
use crate::yuv_support::{
    CbCrInverseTransform, YuvChromaRange, YuvChromaSample, YuvNVOrder, YuvSourceChannels,
};
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn avx2_yuv_nv_to_rgba_row<
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

    let mut cx = start_cx;
    let mut uv_x = start_ux;
    let y_ptr = y_plane.as_ptr();
    let uv_ptr = uv_plane.as_ptr();
    let rgba_ptr = rgba.as_mut_ptr();

    let y_corr = _mm256_set1_epi8(range.bias_y as i8);
    let uv_corr = _mm256_set1_epi16(range.bias_uv as i16);
    let v_luma_coeff = _mm256_set1_epi16(transform.y_coef as i16);
    let v_cr_coeff = _mm256_set1_epi16(transform.cr_coef as i16);
    let v_cb_coeff = _mm256_set1_epi16(transform.cb_coef as i16);
    let v_min_values = _mm256_setzero_si256();
    let v_g_coeff_1 = _mm256_set1_epi16(-1 * transform.g_coeff_1 as i16);
    let v_g_coeff_2 = _mm256_set1_epi16(-1 * transform.g_coeff_2 as i16);
    let v_alpha = _mm256_set1_epi8(255u8 as i8);

    while cx + 32 < width {
        let y_values = _mm256_subs_epi8(
            _mm256_loadu_si256(y_ptr.add(y_offset + cx) as *const __m256i),
            y_corr,
        );

        let (u_high_u8, v_high_u8, u_low_u8, v_low_u8);

        match chroma_subsampling {
            YuvChromaSample::YUV420 | YuvChromaSample::YUV422 => {
                let uv_values = _mm256_loadu_si256(uv_ptr.add(uv_offset + uv_x) as *const __m256i);

                let u_values = avx2_interleave_even(uv_values);
                let v_values = avx2_interleave_odd(uv_values);

                match order {
                    YuvNVOrder::UV => {
                        u_high_u8 = _mm256_extracti128_si256::<1>(u_values);
                        v_high_u8 = _mm256_extracti128_si256::<1>(v_values);
                        u_low_u8 = _mm256_castsi256_si128(u_values);
                        v_low_u8 = _mm256_castsi256_si128(v_values);
                    }
                    YuvNVOrder::VU => {
                        u_high_u8 = _mm256_extracti128_si256::<1>(v_values);
                        v_high_u8 = _mm256_extracti128_si256::<1>(u_values);
                        u_low_u8 = _mm256_castsi256_si128(v_values);
                        v_low_u8 = _mm256_castsi256_si128(u_values);
                    }
                }
            }
            YuvChromaSample::YUV444 => {
                let offset = uv_offset + uv_x;
                let uv_values_l = _mm256_loadu_si256(uv_ptr.add(offset) as *const __m256i);
                let uv_values_h = _mm256_loadu_si256(uv_ptr.add(offset + 32) as *const __m256i);

                let full_v = avx2_interleave_even_2_epi8(uv_values_l, uv_values_h);
                let full_u = avx2_interleave_odd_2_epi8(uv_values_l, uv_values_h);

                match order {
                    YuvNVOrder::UV => {
                        u_high_u8 = _mm256_extracti128_si256::<1>(full_u);
                        v_high_u8 = _mm256_extracti128_si256::<1>(full_v);
                        u_low_u8 = _mm256_castsi256_si128(full_u);
                        v_low_u8 = _mm256_castsi256_si128(full_v);
                    }
                    YuvNVOrder::VU => {
                        u_high_u8 = _mm256_extracti128_si256::<1>(full_v);
                        v_high_u8 = _mm256_extracti128_si256::<1>(full_u);
                        u_low_u8 = _mm256_castsi256_si128(full_v);
                        v_low_u8 = _mm256_castsi256_si128(full_u);
                    }
                }
            }
        }

        let u_high = _mm256_subs_epi16(_mm256_cvtepu8_epi16(u_high_u8), uv_corr);
        let v_high = _mm256_subs_epi16(_mm256_cvtepu8_epi16(v_high_u8), uv_corr);
        let y_high = _mm256_mullo_epi16(
            _mm256_cvtepu8_epi16(_mm256_extracti128_si256::<1>(y_values)),
            v_luma_coeff,
        );

        let r_high = _mm256_srli_epi16::<6>(_mm256_max_epi16(
            _mm256_adds_epi16(y_high, _mm256_mullo_epi16(v_high, v_cr_coeff)),
            v_min_values,
        ));
        let b_high = _mm256_srli_epi16::<6>(_mm256_max_epi16(
            _mm256_adds_epi16(y_high, _mm256_mullo_epi16(u_high, v_cb_coeff)),
            v_min_values,
        ));
        let g_high = _mm256_srli_epi16::<6>(_mm256_max_epi16(
            _mm256_adds_epi16(
                y_high,
                _mm256_adds_epi16(
                    _mm256_mullo_epi16(v_high, v_g_coeff_1),
                    _mm256_mullo_epi16(u_high, v_g_coeff_2),
                ),
            ),
            v_min_values,
        ));

        let u_low = _mm256_subs_epi16(_mm256_cvtepu8_epi16(u_low_u8), uv_corr);
        let v_low = _mm256_subs_epi16(_mm256_cvtepu8_epi16(v_low_u8), uv_corr);
        let y_low = _mm256_mullo_epi16(
            _mm256_cvtepu8_epi16(_mm256_castsi256_si128(y_values)),
            v_luma_coeff,
        );

        let r_low = _mm256_srli_epi16::<6>(_mm256_max_epi16(
            _mm256_adds_epi16(y_low, _mm256_mullo_epi16(v_low, v_cr_coeff)),
            v_min_values,
        ));
        let b_low = _mm256_srli_epi16::<6>(_mm256_max_epi16(
            _mm256_adds_epi16(y_low, _mm256_mullo_epi16(u_low, v_cb_coeff)),
            v_min_values,
        ));
        let g_low = _mm256_srli_epi16::<6>(_mm256_max_epi16(
            _mm256_adds_epi16(
                y_low,
                _mm256_adds_epi16(
                    _mm256_mullo_epi16(v_low, v_g_coeff_1),
                    _mm256_mullo_epi16(u_low, v_g_coeff_2),
                ),
            ),
            v_min_values,
        ));

        let r_values = avx2_pack_u16(r_low, r_high);
        let g_values = avx2_pack_u16(g_low, g_high);
        let b_values = avx2_pack_u16(b_low, b_high);

        let dst_shift = rgba_offset + cx * channels;

        match destination_channels {
            YuvSourceChannels::Rgb => {
                let ptr = rgba_ptr.add(dst_shift);
                avx2_store_u8_rgb(ptr, r_values, g_values, b_values);
            }
            YuvSourceChannels::Rgba => {
                avx2_store_u8_rgba(
                    rgba_ptr.add(dst_shift),
                    r_values,
                    g_values,
                    b_values,
                    v_alpha,
                );
            }
            YuvSourceChannels::Bgra => {
                avx2_store_u8_rgba(
                    rgba_ptr.add(dst_shift),
                    b_values,
                    g_values,
                    r_values,
                    v_alpha,
                );
            }
        }

        cx += 32;

        match chroma_subsampling {
            YuvChromaSample::YUV420 | YuvChromaSample::YUV422 => {
                uv_x += 32;
            }
            YuvChromaSample::YUV444 => {
                uv_x += 64;
            }
        }
    }

    return ProcessedOffset { cx, ux: uv_x };
}
