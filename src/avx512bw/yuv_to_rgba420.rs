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

use crate::avx2::{_mm256_interleave_epi8, _xx256_load_si256};
use crate::avx512bw::avx512_utils::{
    _mm512_expand8_to_10, _xx512_load_si512, avx512_pack_u16, avx512_store_rgba_for_yuv_u8,
};
use crate::internals::{is_slice_aligned, ProcessedOffset};
use crate::yuv_support::{CbCrInverseTransform, YuvChromaRange, YuvSourceChannels};
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

pub(crate) fn avx512_yuv_to_rgba420<const DESTINATION_CHANNELS: u8, const HAS_VBMI: bool>(
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
        let y_aligned0 = is_slice_aligned(y_plane0, 64);
        let y_aligned1 = is_slice_aligned(y_plane1, 64);
        let u_aligned = is_slice_aligned(u_plane, 64);
        let v_aligned = is_slice_aligned(v_plane, 64);
        if y_aligned0 && y_aligned1 && u_aligned && v_aligned {
            if HAS_VBMI {
                avx512_yuv_to_rgba_bmi_impl420::<DESTINATION_CHANNELS, true>(
                    range, transform, y_plane0, y_plane1, u_plane, v_plane, rgba0, rgba1, start_cx,
                    start_ux, width,
                )
            } else {
                avx512_yuv_to_rgba_def_impl420::<DESTINATION_CHANNELS, true>(
                    range, transform, y_plane0, y_plane1, u_plane, v_plane, rgba0, rgba1, start_cx,
                    start_ux, width,
                )
            }
        } else {
            if HAS_VBMI {
                avx512_yuv_to_rgba_bmi_impl420::<DESTINATION_CHANNELS, false>(
                    range, transform, y_plane0, y_plane1, u_plane, v_plane, rgba0, rgba1, start_cx,
                    start_ux, width,
                )
            } else {
                avx512_yuv_to_rgba_def_impl420::<DESTINATION_CHANNELS, false>(
                    range, transform, y_plane0, y_plane1, u_plane, v_plane, rgba0, rgba1, start_cx,
                    start_ux, width,
                )
            }
        }
    }
}

#[target_feature(enable = "avx512bw", enable = "avx512f", enable = "avx512vbmi")]
unsafe fn avx512_yuv_to_rgba_bmi_impl420<const DESTINATION_CHANNELS: u8, const ALIGNED: bool>(
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
    avx512_yuv_to_rgba_impl420::<DESTINATION_CHANNELS, true, ALIGNED>(
        range, transform, y_plane0, y_plane1, u_plane, v_plane, rgba0, rgba1, start_cx, start_ux,
        width,
    )
}

#[target_feature(enable = "avx512bw", enable = "avx512f")]
unsafe fn avx512_yuv_to_rgba_def_impl420<const DESTINATION_CHANNELS: u8, const ALIGNED: bool>(
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
    avx512_yuv_to_rgba_impl420::<DESTINATION_CHANNELS, false, ALIGNED>(
        range, transform, y_plane0, y_plane1, u_plane, v_plane, rgba0, rgba1, start_cx, start_ux,
        width,
    )
}

#[inline(always)]
unsafe fn avx512_yuv_to_rgba_impl420<
    const DESTINATION_CHANNELS: u8,
    const HAS_VBMI: bool,
    const ALIGNED: bool,
>(
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
    if ALIGNED {
        debug_assert!(y_plane0.as_ptr() as usize % 64 == 0);
        debug_assert!(y_plane1.as_ptr() as usize % 64 == 0);
        debug_assert!(u_plane.as_ptr() as usize % 64 == 0);
        debug_assert!(v_plane.as_ptr() as usize % 64 == 0);
    }
    let destination_channels: YuvSourceChannels = DESTINATION_CHANNELS.into();
    let channels = destination_channels.get_channels_count();

    let mut cx = start_cx;
    let mut uv_x = start_ux;
    let u_ptr = u_plane.as_ptr();
    let v_ptr = v_plane.as_ptr();

    let v_luma_coeff = _mm512_set1_epi16(transform.y_coef as i16);
    let v_cr_coeff = _mm512_set1_epi16(transform.cr_coef as i16);
    let v_cb_coeff = _mm512_set1_epi16(transform.cb_coef as i16);
    let v_g_coeff_1 = _mm512_set1_epi16(transform.g_coeff_1 as i16);
    let v_g_coeff_2 = _mm512_set1_epi16(transform.g_coeff_2 as i16);

    const SCALE: u32 = 2;

    while cx + 64 < width {
        let y_corr = _mm512_set1_epi8(range.bias_y as i8);
        let uv_corr = _mm512_set1_epi16(range.bias_uv as i16);
        let y_values0 = _mm512_subs_epu8(
            _xx512_load_si512::<ALIGNED>(y_plane0.get_unchecked(cx..).as_ptr() as *const i32),
            y_corr,
        );
        let y_values1 = _mm512_subs_epu8(
            _xx512_load_si512::<ALIGNED>(y_plane1.get_unchecked(cx..).as_ptr() as *const i32),
            y_corr,
        );

        let y0_10 = _mm512_expand8_to_10::<HAS_VBMI>(y_values0);
        let y1_10 = _mm512_expand8_to_10::<HAS_VBMI>(y_values1);

        let u_values = _xx256_load_si256::<ALIGNED>(u_ptr.add(uv_x) as *const __m256i);
        let v_values = _xx256_load_si256::<ALIGNED>(v_ptr.add(uv_x) as *const __m256i);

        let i_u = _mm256_interleave_epi8(u_values, u_values);
        let i_v = _mm256_interleave_epi8(v_values, v_values);

        let u_high0 = i_u.1;
        let v_high0 = i_v.1;
        let u_low0 = i_u.0;
        let v_low0 = i_v.0;

        let u_high =
            _mm512_slli_epi16::<SCALE>(_mm512_sub_epi16(_mm512_cvtepu8_epi16(u_high0), uv_corr));
        let v_high =
            _mm512_slli_epi16::<SCALE>(_mm512_sub_epi16(_mm512_cvtepu8_epi16(v_high0), uv_corr));

        let y_high0 = _mm512_mulhrs_epi16(y0_10.1, v_luma_coeff);
        let y_high1 = _mm512_mulhrs_epi16(y1_10.1, v_luma_coeff);

        let g_coeff_hi = _mm512_add_epi16(
            _mm512_mulhrs_epi16(v_high, v_g_coeff_1),
            _mm512_mulhrs_epi16(u_high, v_g_coeff_2),
        );

        let r_high0 = _mm512_add_epi16(y_high0, _mm512_mulhrs_epi16(v_high, v_cr_coeff));
        let b_high0 = _mm512_add_epi16(y_high0, _mm512_mulhrs_epi16(u_high, v_cb_coeff));
        let g_high0 = _mm512_sub_epi16(y_high0, g_coeff_hi);

        let r_high1 = _mm512_add_epi16(y_high1, _mm512_mulhrs_epi16(v_high, v_cr_coeff));
        let b_high1 = _mm512_add_epi16(y_high1, _mm512_mulhrs_epi16(u_high, v_cb_coeff));
        let g_high1 = _mm512_sub_epi16(y_high1, g_coeff_hi);

        let u_low =
            _mm512_slli_epi16::<SCALE>(_mm512_sub_epi16(_mm512_cvtepu8_epi16(u_low0), uv_corr));
        let v_low =
            _mm512_slli_epi16::<SCALE>(_mm512_sub_epi16(_mm512_cvtepu8_epi16(v_low0), uv_corr));

        let y_low0 = _mm512_mulhrs_epi16(y0_10.0, v_luma_coeff);
        let y_low1 = _mm512_mulhrs_epi16(y1_10.0, v_luma_coeff);

        let g_coeff_lo = _mm512_add_epi16(
            _mm512_mulhrs_epi16(v_low, v_g_coeff_1),
            _mm512_mulhrs_epi16(u_low, v_g_coeff_2),
        );

        let r_low0 = _mm512_add_epi16(y_low0, _mm512_mulhrs_epi16(v_low, v_cr_coeff));
        let b_low0 = _mm512_add_epi16(y_low0, _mm512_mulhrs_epi16(u_low, v_cb_coeff));
        let g_low0 = _mm512_sub_epi16(y_low0, g_coeff_lo);

        let r_low1 = _mm512_add_epi16(y_low1, _mm512_mulhrs_epi16(v_low, v_cr_coeff));
        let b_low1 = _mm512_add_epi16(y_low1, _mm512_mulhrs_epi16(u_low, v_cb_coeff));
        let g_low1 = _mm512_sub_epi16(y_low1, g_coeff_lo);

        let r_values0 = avx512_pack_u16(r_low0, r_high0);
        let g_values0 = avx512_pack_u16(g_low0, g_high0);
        let b_values0 = avx512_pack_u16(b_low0, b_high0);

        let r_values1 = avx512_pack_u16(r_low1, r_high1);
        let g_values1 = avx512_pack_u16(g_low1, g_high1);
        let b_values1 = avx512_pack_u16(b_low1, b_high1);

        let dst_shift = cx * channels;

        let v_alpha = _mm512_set1_epi8(255u8 as i8);

        avx512_store_rgba_for_yuv_u8::<DESTINATION_CHANNELS, HAS_VBMI>(
            rgba0.get_unchecked_mut(dst_shift..).as_mut_ptr(),
            r_values0,
            g_values0,
            b_values0,
            v_alpha,
        );

        avx512_store_rgba_for_yuv_u8::<DESTINATION_CHANNELS, HAS_VBMI>(
            rgba1.get_unchecked_mut(dst_shift..).as_mut_ptr(),
            r_values1,
            g_values1,
            b_values1,
            v_alpha,
        );

        cx += 64;
        uv_x += 32;
    }

    ProcessedOffset { cx, ux: uv_x }
}
