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

use crate::internals::ProcessedOffset;
use crate::sse::{
    _mm_deinterleave_x2_epi8, _mm_store_interleave_rgb_for_yuv, sse_interleave_rgb,
    sse_interleave_rgba,
};
use crate::yuv_support::{CbCrInverseTransform, YuvChromaRange, YuvNVOrder, YuvSourceChannels};
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

pub(crate) fn sse_yuv_nv_to_rgba420<const UV_ORDER: u8, const DESTINATION_CHANNELS: u8>(
    range: &YuvChromaRange,
    transform: &CbCrInverseTransform<i32>,
    y_plane0: &[u8],
    y_plane1: &[u8],
    uv_plane: &[u8],
    rgba0: &mut [u8],
    rgba1: &mut [u8],
    start_cx: usize,
    start_ux: usize,
    width: usize,
) -> ProcessedOffset {
    unsafe {
        sse_yuv_nv_to_rgba_impl420::<UV_ORDER, DESTINATION_CHANNELS>(
            range, transform, y_plane0, y_plane1, uv_plane, rgba0, rgba1, start_cx, start_ux, width,
        )
    }
}

#[target_feature(enable = "sse4.1")]
unsafe fn sse_yuv_nv_to_rgba_impl420<const UV_ORDER: u8, const DESTINATION_CHANNELS: u8>(
    range: &YuvChromaRange,
    transform: &CbCrInverseTransform<i32>,
    y_plane0: &[u8],
    y_plane1: &[u8],
    uv_plane: &[u8],
    rgba0: &mut [u8],
    rgba1: &mut [u8],
    start_cx: usize,
    start_ux: usize,
    width: usize,
) -> ProcessedOffset {
    let order: YuvNVOrder = UV_ORDER.into();
    let destination_channels: YuvSourceChannels = DESTINATION_CHANNELS.into();
    let channels = destination_channels.get_channels_count();

    let mut cx = start_cx;
    let mut uv_x = start_ux;

    let uv_ptr = uv_plane.as_ptr();

    const SCALE: i32 = 2;

    let y_corr = _mm_set1_epi8(range.bias_y as i8);
    let uv_corr = _mm_set1_epi16(range.bias_uv as i16);
    let v_luma_coeff = _mm_set1_epi16(transform.y_coef as i16);
    let v_cr_coeff = _mm_set1_epi16(transform.cr_coef as i16);
    let v_cb_coeff = _mm_set1_epi16(transform.cb_coef as i16);
    let v_g_coeff_1 = _mm_set1_epi16(transform.g_coeff_1 as i16);
    let v_g_coeff_2 = _mm_set1_epi16(transform.g_coeff_2 as i16);
    let v_alpha = _mm_set1_epi8(255u8 as i8);
    let zeros = _mm_setzero_si128();

    let distribute_shuffle = _mm_setr_epi8(0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7);

    while cx + 16 < width {
        let y_values0 = _mm_subs_epu8(
            _mm_loadu_si128(y_plane0.get_unchecked(cx..).as_ptr() as *const __m128i),
            y_corr,
        );
        let y_values1 = _mm_subs_epu8(
            _mm_loadu_si128(y_plane1.get_unchecked(cx..).as_ptr() as *const __m128i),
            y_corr,
        );

        let (u_high_u16, v_high_u16, u_low_u16, v_low_u16);

        let uv_values_ = _mm_loadu_si128(uv_ptr.add(uv_x) as *const __m128i);
        let (mut u, mut v) = _mm_deinterleave_x2_epi8(uv_values_, zeros);

        u = _mm_shuffle_epi8(u, distribute_shuffle);
        v = _mm_shuffle_epi8(v, distribute_shuffle);

        match order {
            YuvNVOrder::UV => {
                u_high_u16 = _mm_unpackhi_epi8(u, zeros);
                v_high_u16 = _mm_unpackhi_epi8(v, zeros);
                u_low_u16 = _mm_unpacklo_epi8(u, zeros);
                v_low_u16 = _mm_unpacklo_epi8(v, zeros);
            }
            YuvNVOrder::VU => {
                u_high_u16 = _mm_unpackhi_epi8(v, zeros);
                v_high_u16 = _mm_unpackhi_epi8(u, zeros);
                u_low_u16 = _mm_unpacklo_epi8(v, zeros);
                v_low_u16 = _mm_unpacklo_epi8(u, zeros);
            }
        }

        let u_high = _mm_slli_epi16::<SCALE>(_mm_sub_epi16(u_high_u16, uv_corr));
        let v_high = _mm_slli_epi16::<SCALE>(_mm_sub_epi16(v_high_u16, uv_corr));
        let y_high0 = _mm_mulhrs_epi16(
            _mm_slli_epi16::<SCALE>(_mm_unpackhi_epi8(y_values0, zeros)),
            v_luma_coeff,
        );
        let y_high1 = _mm_mulhrs_epi16(
            _mm_slli_epi16::<SCALE>(_mm_unpackhi_epi8(y_values1, zeros)),
            v_luma_coeff,
        );

        let g_coeff_hi = _mm_add_epi16(
            _mm_mulhrs_epi16(v_high, v_g_coeff_1),
            _mm_mulhrs_epi16(u_high, v_g_coeff_2),
        );

        let r_high0 = _mm_add_epi16(y_high0, _mm_mulhrs_epi16(v_high, v_cr_coeff));
        let b_high0 = _mm_add_epi16(y_high0, _mm_mulhrs_epi16(u_high, v_cb_coeff));
        let g_high0 = _mm_sub_epi16(y_high0, g_coeff_hi);
        let r_high1 = _mm_add_epi16(y_high1, _mm_mulhrs_epi16(v_high, v_cr_coeff));
        let b_high1 = _mm_add_epi16(y_high1, _mm_mulhrs_epi16(u_high, v_cb_coeff));
        let g_high1 = _mm_sub_epi16(y_high1, g_coeff_hi);

        let u_low = _mm_slli_epi16::<SCALE>(_mm_sub_epi16(u_low_u16, uv_corr));
        let v_low = _mm_slli_epi16::<SCALE>(_mm_sub_epi16(v_low_u16, uv_corr));
        let y_low0 = _mm_mulhrs_epi16(
            _mm_slli_epi16::<SCALE>(_mm_cvtepu8_epi16(y_values0)),
            v_luma_coeff,
        );
        let y_low1 = _mm_mulhrs_epi16(
            _mm_slli_epi16::<SCALE>(_mm_cvtepu8_epi16(y_values1)),
            v_luma_coeff,
        );

        let g_coeff_lo = _mm_add_epi16(
            _mm_mulhrs_epi16(v_low, v_g_coeff_1),
            _mm_mulhrs_epi16(u_low, v_g_coeff_2),
        );

        let r_low0 = _mm_add_epi16(y_low0, _mm_mulhrs_epi16(v_low, v_cr_coeff));
        let b_low0 = _mm_add_epi16(y_low0, _mm_mulhrs_epi16(u_low, v_cb_coeff));
        let g_low0 = _mm_sub_epi16(y_low0, g_coeff_lo);
        let r_low1 = _mm_add_epi16(y_low1, _mm_mulhrs_epi16(v_low, v_cr_coeff));
        let b_low1 = _mm_add_epi16(y_low1, _mm_mulhrs_epi16(u_low, v_cb_coeff));
        let g_low1 = _mm_sub_epi16(y_low1, g_coeff_lo);

        let r_values0 = _mm_packus_epi16(r_low0, r_high0);
        let g_values0 = _mm_packus_epi16(g_low0, g_high0);
        let b_values0 = _mm_packus_epi16(b_low0, b_high0);

        let r_values1 = _mm_packus_epi16(r_low1, r_high1);
        let g_values1 = _mm_packus_epi16(g_low1, g_high1);
        let b_values1 = _mm_packus_epi16(b_low1, b_high1);

        let dst_shift = cx * channels;

        _mm_store_interleave_rgb_for_yuv::<DESTINATION_CHANNELS>(
            rgba0.get_unchecked_mut(dst_shift..).as_mut_ptr(),
            r_values0,
            g_values0,
            b_values0,
            v_alpha,
        );
        _mm_store_interleave_rgb_for_yuv::<DESTINATION_CHANNELS>(
            rgba1.get_unchecked_mut(dst_shift..).as_mut_ptr(),
            r_values1,
            g_values1,
            b_values1,
            v_alpha,
        );

        cx += 16;
        uv_x += 16;
    }

    while cx + 8 < width {
        let y_values0 = _mm_subs_epi8(
            _mm_loadu_si64(y_plane0.get_unchecked(cx..).as_ptr()),
            y_corr,
        );
        let y_values1 = _mm_subs_epi8(
            _mm_loadu_si64(y_plane1.get_unchecked(cx..).as_ptr()),
            y_corr,
        );

        let (u_low_u16, v_low_u16);

        let uv_values_ = _mm_loadu_si64(uv_ptr.add(uv_x));
        let (mut u, mut v) = _mm_deinterleave_x2_epi8(uv_values_, zeros);

        u = _mm_shuffle_epi8(u, distribute_shuffle);
        v = _mm_shuffle_epi8(v, distribute_shuffle);

        match order {
            YuvNVOrder::UV => {
                u_low_u16 = _mm_unpacklo_epi8(u, zeros);
                v_low_u16 = _mm_unpacklo_epi8(v, zeros);
            }
            YuvNVOrder::VU => {
                u_low_u16 = _mm_unpacklo_epi8(v, zeros);
                v_low_u16 = _mm_unpacklo_epi8(u, zeros);
            }
        }

        let u_low = _mm_slli_epi16::<SCALE>(_mm_sub_epi16(u_low_u16, uv_corr));
        let v_low = _mm_slli_epi16::<SCALE>(_mm_sub_epi16(v_low_u16, uv_corr));
        let y_low0 = _mm_mulhrs_epi16(
            _mm_slli_epi16::<SCALE>(_mm_cvtepu8_epi16(y_values0)),
            v_luma_coeff,
        );
        let y_low1 = _mm_mulhrs_epi16(
            _mm_slli_epi16::<SCALE>(_mm_cvtepu8_epi16(y_values1)),
            v_luma_coeff,
        );

        let g_coeff_lo = _mm_add_epi16(
            _mm_mulhrs_epi16(v_low, v_g_coeff_1),
            _mm_mulhrs_epi16(u_low, v_g_coeff_2),
        );

        let r_low0 = _mm_add_epi16(y_low0, _mm_mulhrs_epi16(v_low, v_cr_coeff));
        let b_low0 = _mm_add_epi16(y_low0, _mm_mulhrs_epi16(u_low, v_cb_coeff));
        let g_low0 = _mm_sub_epi16(y_low0, g_coeff_lo);

        let r_low1 = _mm_add_epi16(y_low1, _mm_mulhrs_epi16(v_low, v_cr_coeff));
        let b_low1 = _mm_add_epi16(y_low1, _mm_mulhrs_epi16(u_low, v_cb_coeff));
        let g_low1 = _mm_sub_epi16(y_low1, g_coeff_lo);

        let r_values0 = _mm_packus_epi16(r_low0, zeros);
        let g_values0 = _mm_packus_epi16(g_low0, zeros);
        let b_values0 = _mm_packus_epi16(b_low0, zeros);

        let r_values1 = _mm_packus_epi16(r_low1, zeros);
        let g_values1 = _mm_packus_epi16(g_low1, zeros);
        let b_values1 = _mm_packus_epi16(b_low1, zeros);

        let dst_shift = cx * channels;
        let dst_ptr0 = rgba0.get_unchecked_mut(dst_shift..).as_mut_ptr();
        let dst_ptr1 = rgba1.get_unchecked_mut(dst_shift..).as_mut_ptr();
        match destination_channels {
            YuvSourceChannels::Rgb => {
                let (v0, v1, _) = sse_interleave_rgb(r_values0, g_values0, b_values0);
                _mm_storeu_si128(dst_ptr0 as *mut __m128i, v0);
                std::ptr::copy_nonoverlapping(&v1 as *const _ as *const u8, dst_ptr0.add(16), 8);

                let (v0, v1, _) = sse_interleave_rgb(r_values1, g_values1, b_values1);
                _mm_storeu_si128(dst_ptr1 as *mut __m128i, v0);
                std::ptr::copy_nonoverlapping(&v1 as *const _ as *const u8, dst_ptr1.add(16), 8);
            }
            YuvSourceChannels::Bgr => {
                let (v0, v1, _) = sse_interleave_rgb(b_values0, g_values0, r_values0);
                _mm_storeu_si128(dst_ptr0 as *mut __m128i, v0);
                std::ptr::copy_nonoverlapping(&v1 as *const _ as *const u8, dst_ptr0.add(16), 8);

                let (v0, v1, _) = sse_interleave_rgb(b_values1, g_values1, r_values1);
                _mm_storeu_si128(dst_ptr1 as *mut __m128i, v0);
                std::ptr::copy_nonoverlapping(&v1 as *const _ as *const u8, dst_ptr1.add(16), 8);
            }
            YuvSourceChannels::Rgba => {
                let (row1, row2, _, _) =
                    sse_interleave_rgba(r_values0, g_values0, b_values0, v_alpha);
                _mm_storeu_si128(dst_ptr0 as *mut __m128i, row1);
                _mm_storeu_si128(dst_ptr0.add(16) as *mut __m128i, row2);

                let (row1, row2, _, _) =
                    sse_interleave_rgba(r_values1, g_values1, b_values1, v_alpha);
                _mm_storeu_si128(dst_ptr1 as *mut __m128i, row1);
                _mm_storeu_si128(dst_ptr1.add(16) as *mut __m128i, row2);
            }
            YuvSourceChannels::Bgra => {
                let (row1, row2, _, _) =
                    sse_interleave_rgba(b_values0, g_values0, r_values0, v_alpha);
                _mm_storeu_si128(dst_ptr0 as *mut __m128i, row1);
                _mm_storeu_si128(dst_ptr0.add(16) as *mut __m128i, row2);

                let (row1, row2, _, _) =
                    sse_interleave_rgba(b_values1, g_values1, r_values1, v_alpha);
                _mm_storeu_si128(dst_ptr1 as *mut __m128i, row1);
                _mm_storeu_si128(dst_ptr1.add(16) as *mut __m128i, row2);
            }
        }

        cx += 8;
        uv_x += 8;
    }

    ProcessedOffset { cx, ux: uv_x }
}
