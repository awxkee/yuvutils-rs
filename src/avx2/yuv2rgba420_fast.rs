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
use crate::internals::{interleaved_epi8, ProcessedOffset};
use crate::yuv_support::{CbCrInverseTransform, YuvChromaRange, YuvSourceChannels};
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

pub(crate) fn avx2_yuv_to_rgba_fast_row420<const DESTINATION_CHANNELS: u8>(
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
        avx2_yuv_to_rgba_row_fast_impl420::<DESTINATION_CHANNELS>(
            range, transform, y_plane0, y_plane1, u_plane, v_plane, rgba0, rgba1, start_cx,
            start_ux, width,
        )
    }
}

#[target_feature(enable = "avx2")]
unsafe fn avx2_yuv_to_rgba_row_fast_impl420<const DESTINATION_CHANNELS: u8>(
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

    const PRECISION: i32 = 6;

    let y_corr = _mm256_set1_epi8(range.bias_y as i8);
    let v_luma_coeff = _mm256_set1_epi16((transform.y_coef as u16 * 256) as i16);
    let v_cr_coeff = _mm256_set1_epi16(interleaved_epi8(
        transform.cr_coef as i8,
        -transform.cr_coef as i8,
    ));
    let v_cb_coeff = _mm256_set1_epi16(interleaved_epi8(
        transform.cb_coef as i8,
        -transform.cb_coef as i8,
    ));
    let v_g_coeff_1 = _mm256_set1_epi16(interleaved_epi8(
        transform.g_coeff_1 as i8,
        -transform.g_coeff_1 as i8,
    ));
    let v_g_coeff_2 = _mm256_set1_epi16(interleaved_epi8(
        transform.g_coeff_2 as i8,
        -transform.g_coeff_2 as i8,
    ));
    let u_bias_uv = _mm256_set1_epi8(range.bias_uv as i8);

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

        let (u_low_u16, u_high_u16) = _mm256_interleave_epi8(
            avx2_create(
                _mm_unpacklo_epi8(u_values, u_values),
                _mm_unpackhi_epi8(u_values, u_values),
            ),
            u_bias_uv,
        );

        let (v_low_u16, v_high_u16) = _mm256_interleave_epi8(
            avx2_create(
                _mm_unpacklo_epi8(v_values, v_values),
                _mm_unpackhi_epi8(v_values, v_values),
            ),
            u_bias_uv,
        );

        let y0_10 = _mm256_interleave_epi8(y_values0, y_values0);
        let y1_10 = _mm256_interleave_epi8(y_values1, y_values1);

        let y_high0 = _mm256_mulhi_epu16(y0_10.1, v_luma_coeff);
        let y_high1 = _mm256_mulhi_epu16(y1_10.1, v_luma_coeff);

        let g_coeff_hi = _mm256_adds_epi16(
            _mm256_maddubs_epi16(v_high_u16, v_g_coeff_1),
            _mm256_maddubs_epi16(u_high_u16, v_g_coeff_2),
        );

        let r_high0 = _mm256_adds_epi16(y_high0, _mm256_maddubs_epi16(v_high_u16, v_cr_coeff));
        let b_high0 = _mm256_adds_epi16(y_high0, _mm256_maddubs_epi16(u_high_u16, v_cb_coeff));
        let g_high0 = _mm256_subs_epi16(y_high0, g_coeff_hi);

        let r_high1 = _mm256_adds_epi16(y_high1, _mm256_maddubs_epi16(v_high_u16, v_cr_coeff));
        let b_high1 = _mm256_adds_epi16(y_high1, _mm256_maddubs_epi16(u_high_u16, v_cb_coeff));
        let g_high1 = _mm256_subs_epi16(y_high1, g_coeff_hi);

        let y_low0 = _mm256_mulhi_epu16(y0_10.0, v_luma_coeff);
        let y_low1 = _mm256_mulhi_epu16(y1_10.0, v_luma_coeff);

        let g_coeff_lo = _mm256_adds_epi16(
            _mm256_maddubs_epi16(v_low_u16, v_g_coeff_1),
            _mm256_maddubs_epi16(u_low_u16, v_g_coeff_2),
        );

        let r_low0 = _mm256_adds_epi16(y_low0, _mm256_maddubs_epi16(v_low_u16, v_cr_coeff));
        let b_low0 = _mm256_adds_epi16(y_low0, _mm256_maddubs_epi16(u_low_u16, v_cb_coeff));
        let g_low0 = _mm256_subs_epi16(y_low0, g_coeff_lo);

        let r_low1 = _mm256_adds_epi16(y_low1, _mm256_maddubs_epi16(v_low_u16, v_cr_coeff));
        let b_low1 = _mm256_adds_epi16(y_low1, _mm256_maddubs_epi16(u_low_u16, v_cb_coeff));
        let g_low1 = _mm256_subs_epi16(y_low1, g_coeff_lo);

        let r_values0 = avx2_pack_u16(
            _mm256_srai_epi16::<PRECISION>(r_low0),
            _mm256_srai_epi16::<PRECISION>(r_high0),
        );
        let g_values0 = avx2_pack_u16(
            _mm256_srai_epi16::<PRECISION>(g_low0),
            _mm256_srai_epi16::<PRECISION>(g_high0),
        );
        let b_values0 = avx2_pack_u16(
            _mm256_srai_epi16::<PRECISION>(b_low0),
            _mm256_srai_epi16::<PRECISION>(b_high0),
        );

        let r_values1 = avx2_pack_u16(
            _mm256_srai_epi16::<PRECISION>(r_low1),
            _mm256_srai_epi16::<PRECISION>(r_high1),
        );
        let g_values1 = avx2_pack_u16(
            _mm256_srai_epi16::<PRECISION>(g_low1),
            _mm256_srai_epi16::<PRECISION>(g_high1),
        );
        let b_values1 = avx2_pack_u16(
            _mm256_srai_epi16::<PRECISION>(b_low1),
            _mm256_srai_epi16::<PRECISION>(b_high1),
        );

        let dst_shift = cx * channels;

        let v_alpha = _mm256_set1_epi8(255u8 as i8);

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

    while cx + 16 < width {
        let y_values0 = _mm256_subs_epu8(
            _mm256_castsi128_si256(_mm_loadu_si128(
                y_plane0.get_unchecked(cx..).as_ptr() as *const __m128i
            )),
            y_corr,
        );
        let y_values1 = _mm256_subs_epu8(
            _mm256_castsi128_si256(_mm_loadu_si128(
                y_plane1.get_unchecked(cx..).as_ptr() as *const __m128i
            )),
            y_corr,
        );

        let u_values = _mm_loadu_si64(u_ptr.add(uv_x));
        let v_values = _mm_loadu_si64(v_ptr.add(uv_x));

        let (u_low_u16, _) = _mm256_interleave_epi8(
            avx2_create(_mm_unpacklo_epi8(u_values, u_values), _mm_setzero_si128()),
            u_bias_uv,
        );

        let (v_low_u16, _) = _mm256_interleave_epi8(
            avx2_create(_mm_unpacklo_epi8(v_values, v_values), _mm_setzero_si128()),
            u_bias_uv,
        );

        let y0_10 = _mm256_interleave_epi8(y_values0, y_values0);
        let y1_10 = _mm256_interleave_epi8(y_values1, y_values1);

        let y_low0 = _mm256_mulhi_epu16(y0_10.0, v_luma_coeff);
        let y_low1 = _mm256_mulhi_epu16(y1_10.0, v_luma_coeff);

        let g_coeff_lo = _mm256_adds_epi16(
            _mm256_maddubs_epi16(v_low_u16, v_g_coeff_1),
            _mm256_maddubs_epi16(u_low_u16, v_g_coeff_2),
        );

        let r_low0 = _mm256_adds_epi16(y_low0, _mm256_maddubs_epi16(v_low_u16, v_cr_coeff));
        let b_low0 = _mm256_adds_epi16(y_low0, _mm256_maddubs_epi16(u_low_u16, v_cb_coeff));
        let g_low0 = _mm256_subs_epi16(y_low0, g_coeff_lo);

        let r_low1 = _mm256_adds_epi16(y_low1, _mm256_maddubs_epi16(v_low_u16, v_cr_coeff));
        let b_low1 = _mm256_adds_epi16(y_low1, _mm256_maddubs_epi16(u_low_u16, v_cb_coeff));
        let g_low1 = _mm256_subs_epi16(y_low1, g_coeff_lo);

        let zeros = _mm256_setzero_si256();

        let r_values0 = avx2_pack_u16(_mm256_srai_epi16::<PRECISION>(r_low0), zeros);
        let g_values0 = avx2_pack_u16(_mm256_srai_epi16::<PRECISION>(g_low0), zeros);
        let b_values0 = avx2_pack_u16(_mm256_srai_epi16::<PRECISION>(b_low0), zeros);

        let r_values1 = avx2_pack_u16(_mm256_srai_epi16::<PRECISION>(r_low1), zeros);
        let g_values1 = avx2_pack_u16(_mm256_srai_epi16::<PRECISION>(g_low1), zeros);
        let b_values1 = avx2_pack_u16(_mm256_srai_epi16::<PRECISION>(b_low1), zeros);

        let dst_shift = cx * channels;

        let v_alpha = _mm256_set1_epi8(255u8 as i8);
        _mm256_store_interleave_rgb_half_for_yuv::<DESTINATION_CHANNELS>(
            rgba0.get_unchecked_mut(dst_shift..).as_mut_ptr(),
            r_values0,
            g_values0,
            b_values0,
            v_alpha,
        );
        _mm256_store_interleave_rgb_half_for_yuv::<DESTINATION_CHANNELS>(
            rgba1.get_unchecked_mut(dst_shift..).as_mut_ptr(),
            r_values1,
            g_values1,
            b_values1,
            v_alpha,
        );

        cx += 16;
        uv_x += 8;
    }

    ProcessedOffset { cx, ux: uv_x }
}