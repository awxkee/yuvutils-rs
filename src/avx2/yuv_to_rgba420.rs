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

use crate::avx2::avx2_utils::*;
use crate::internals::ProcessedOffset;
use crate::yuv_support::{CbCrInverseTransform, YuvChromaRange, YuvSourceChannels};
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

pub(crate) fn avx2_yuv_to_rgba_row420<const DESTINATION_CHANNELS: u8>(
    range: &YuvChromaRange,
    transform: &CbCrInverseTransform<i32>,
    y_plane0: &[u8],
    y_plane1: &[u8],
    u_plane: &[u8],
    v_plane: &[u8],
    rgba0: &mut [u8],
    rgba1: &mut [u8],
    start_cx: usize,
    start_ux: usize,
    width: usize,
) -> ProcessedOffset {
    unsafe {
        avx2_yuv_to_rgba_row_impl420::<DESTINATION_CHANNELS>(
            range, transform, y_plane0, y_plane1, u_plane, v_plane, rgba0, rgba1, start_cx,
            start_ux, width,
        )
    }
}

#[target_feature(enable = "avx2")]
unsafe fn avx2_yuv_to_rgba_row_impl420<const DESTINATION_CHANNELS: u8>(
    range: &YuvChromaRange,
    transform: &CbCrInverseTransform<i32>,
    y_plane0: &[u8],
    y_plane1: &[u8],
    u_plane: &[u8],
    v_plane: &[u8],
    rgba0: &mut [u8],
    rgba1: &mut [u8],
    start_cx: usize,
    start_ux: usize,
    width: usize,
) -> ProcessedOffset {
    let destination_channels: YuvSourceChannels = DESTINATION_CHANNELS.into();
    let channels = destination_channels.get_channels_count();

    let mut cx = start_cx;
    let mut uv_x = start_ux;
    let u_ptr = u_plane.as_ptr();
    let v_ptr = v_plane.as_ptr();

    let y_corr = _mm256_set1_epi8(range.bias_y as i8);
    let uv_corr = _mm256_set1_epi16(range.bias_uv as i16);
    let v_luma_coeff = _mm256_set1_epi16(transform.y_coef as i16);
    let v_cr_coeff = _mm256_set1_epi16(transform.cr_coef as i16);
    let v_cb_coeff = _mm256_set1_epi16(transform.cb_coef as i16);
    let v_g_coeff_1 = _mm256_set1_epi16(transform.g_coeff_1 as i16);
    let v_g_coeff_2 = _mm256_set1_epi16(transform.g_coeff_2 as i16);
    let v_alpha = _mm256_set1_epi8(255u8 as i8);

    const SCALE: i32 = 2;

    while cx + 32 < width {
        let y_values0 = _mm256_subs_epu8(
            _mm256_loadu_si256(y_plane0.get_unchecked(cx..).as_ptr() as *const __m256i),
            y_corr,
        );
        let y_values1 = _mm256_subs_epu8(
            _mm256_loadu_si256(y_plane1.get_unchecked(cx..).as_ptr() as *const __m256i),
            y_corr,
        );

        let u_values = _mm_loadu_si128(u_ptr.add(uv_x) as *const __m128i);
        let v_values = _mm_loadu_si128(v_ptr.add(uv_x) as *const __m128i);

        let u_high_u16 = _mm256_cvtepu8_epi16(_mm_unpackhi_epi8(u_values, u_values));
        let v_high_u16 = _mm256_cvtepu8_epi16(_mm_unpackhi_epi8(v_values, v_values));
        let u_low_u16 = _mm256_cvtepu8_epi16(_mm_unpacklo_epi8(u_values, u_values));
        let v_low_u16 = _mm256_cvtepu8_epi16(_mm_unpacklo_epi8(v_values, v_values));

        let u_high = _mm256_slli_epi16::<SCALE>(_mm256_sub_epi16(u_high_u16, uv_corr));
        let v_high = _mm256_slli_epi16::<SCALE>(_mm256_sub_epi16(v_high_u16, uv_corr));
        let y_high0 = _mm256_mulhrs_epi16(
            _mm256_slli_epi16::<SCALE>(_mm256_cvtepu8_epi16(_mm256_extracti128_si256::<1>(
                y_values0,
            ))),
            v_luma_coeff,
        );
        let y_high1 = _mm256_mulhrs_epi16(
            _mm256_slli_epi16::<SCALE>(_mm256_cvtepu8_epi16(_mm256_extracti128_si256::<1>(
                y_values1,
            ))),
            v_luma_coeff,
        );

        let g_coeff_hi = _mm256_add_epi16(
            _mm256_mulhrs_epi16(v_high, v_g_coeff_1),
            _mm256_mulhrs_epi16(u_high, v_g_coeff_2),
        );

        let r_high0 = _mm256_add_epi16(y_high0, _mm256_mulhrs_epi16(v_high, v_cr_coeff));
        let b_high0 = _mm256_add_epi16(y_high0, _mm256_mulhrs_epi16(u_high, v_cb_coeff));
        let g_high0 = _mm256_sub_epi16(y_high0, g_coeff_hi);

        let r_high1 = _mm256_add_epi16(y_high1, _mm256_mulhrs_epi16(v_high, v_cr_coeff));
        let b_high1 = _mm256_add_epi16(y_high1, _mm256_mulhrs_epi16(u_high, v_cb_coeff));
        let g_high1 = _mm256_sub_epi16(y_high1, g_coeff_hi);

        let u_low = _mm256_slli_epi16::<SCALE>(_mm256_sub_epi16(u_low_u16, uv_corr));
        let v_low = _mm256_slli_epi16::<SCALE>(_mm256_sub_epi16(v_low_u16, uv_corr));
        let y_low0 = _mm256_mulhrs_epi16(
            _mm256_slli_epi16::<SCALE>(_mm256_cvtepu8_epi16(_mm256_castsi256_si128(y_values0))),
            v_luma_coeff,
        );
        let y_low1 = _mm256_mulhrs_epi16(
            _mm256_slli_epi16::<SCALE>(_mm256_cvtepu8_epi16(_mm256_castsi256_si128(y_values1))),
            v_luma_coeff,
        );

        let g_coeff_lo = _mm256_add_epi16(
            _mm256_mulhrs_epi16(v_low, v_g_coeff_1),
            _mm256_mulhrs_epi16(u_low, v_g_coeff_2),
        );

        let r_low0 = _mm256_add_epi16(y_low0, _mm256_mulhrs_epi16(v_low, v_cr_coeff));
        let b_low0 = _mm256_add_epi16(y_low0, _mm256_mulhrs_epi16(u_low, v_cb_coeff));
        let g_low0 = _mm256_sub_epi16(y_low0, g_coeff_lo);

        let r_low1 = _mm256_add_epi16(y_low1, _mm256_mulhrs_epi16(v_low, v_cr_coeff));
        let b_low1 = _mm256_add_epi16(y_low1, _mm256_mulhrs_epi16(u_low, v_cb_coeff));
        let g_low1 = _mm256_sub_epi16(y_low1, g_coeff_lo);
        let r_values0 = avx2_pack_u16(r_low0, r_high0);
        let g_values0 = avx2_pack_u16(g_low0, g_high0);
        let b_values0 = avx2_pack_u16(b_low0, b_high0);

        let r_values1 = avx2_pack_u16(r_low1, r_high1);
        let g_values1 = avx2_pack_u16(g_low1, g_high1);
        let b_values1 = avx2_pack_u16(b_low1, b_high1);

        let dst_shift = cx * channels;

        _mm256_store_interleave_rgb_for_yuv::<DESTINATION_CHANNELS>(
            rgba0.get_unchecked_mut(dst_shift..).as_mut_ptr(),
            r_values0,
            g_values0,
            b_values0,
            v_alpha,
        );
        _mm256_store_interleave_rgb_for_yuv::<DESTINATION_CHANNELS>(
            rgba1.get_unchecked_mut(dst_shift..).as_mut_ptr(),
            r_values1,
            g_values1,
            b_values1,
            v_alpha,
        );

        cx += 32;
        uv_x += 16;
    }

    ProcessedOffset { cx, ux: uv_x }
}
