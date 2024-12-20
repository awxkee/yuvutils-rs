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

use crate::avx512bw::avx512_utils::{
    avx2_unzip_epi8, avx2_zip_epi8, avx512_pack_u16, avx512_store_u8,
};
use crate::internals::ProcessedOffset;
use crate::yuv_support::{CbCrInverseTransform, YuvChromaRange, YuvNVOrder, YuvSourceChannels};
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

pub(crate) fn avx512_yuv_nv_to_rgba420<
    const UV_ORDER: u8,
    const DESTINATION_CHANNELS: u8,
    const HAS_VBMI: bool,
>(
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
        avx512_yuv_nv_to_rgba_impl420::<UV_ORDER, DESTINATION_CHANNELS, HAS_VBMI>(
            range, transform, y_plane0, y_plane1, uv_plane, rgba0, rgba1, start_cx, start_ux, width,
        )
    }
}

#[target_feature(enable = "avx512bw", enable = "avx512f")]
unsafe fn avx512_yuv_nv_to_rgba_impl420<
    const UV_ORDER: u8,
    const DESTINATION_CHANNELS: u8,
    const HAS_VBMI: bool,
>(
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

    const SCALE: u32 = 2;

    let v_luma_coeff = _mm512_set1_epi16(transform.y_coef as i16);
    let v_cr_coeff = _mm512_set1_epi16(transform.cr_coef as i16);
    let v_cb_coeff = _mm512_set1_epi16(transform.cb_coef as i16);
    let v_g_coeff_1 = _mm512_set1_epi16(transform.g_coeff_1 as i16);
    let v_g_coeff_2 = _mm512_set1_epi16(transform.g_coeff_2 as i16);

    while cx + 32 < width {
        let y_corr = _mm512_set1_epi8(range.bias_y as i8);
        let uv_corr = _mm512_set1_epi16(range.bias_uv as i16);
        let y_values0 = _mm512_subs_epu8(
            _mm512_loadu_si512(y_plane0.get_unchecked(cx..).as_ptr() as *const i32),
            y_corr,
        );
        let y_values1 = _mm512_subs_epu8(
            _mm512_loadu_si512(y_plane1.get_unchecked(cx..).as_ptr() as *const i32),
            y_corr,
        );

        let (u_high, v_high0, u_low0, v_low0);

        let uv_values = _mm512_loadu_si512(uv_ptr.add(uv_x) as *const i32);

        let (u_values0, v_values0) = avx2_unzip_epi8::<HAS_VBMI>(uv_values, _mm512_setzero_si512());

        let (u_values, _) = avx2_zip_epi8::<HAS_VBMI>(u_values0, u_values0);
        let (v_values, _) = avx2_zip_epi8::<HAS_VBMI>(v_values0, v_values0);

        match order {
            YuvNVOrder::UV => {
                u_high = _mm512_extracti64x4_epi64::<1>(u_values);
                v_high0 = _mm512_extracti64x4_epi64::<1>(v_values);
                u_low0 = _mm512_castsi512_si256(u_values);
                v_low0 = _mm512_castsi512_si256(v_values);
            }
            YuvNVOrder::VU => {
                u_high = _mm512_extracti64x4_epi64::<1>(v_values);
                v_high0 = _mm512_extracti64x4_epi64::<1>(u_values);
                u_low0 = _mm512_castsi512_si256(v_values);
                v_low0 = _mm512_castsi512_si256(u_values);
            }
        }

        let u_high =
            _mm512_slli_epi16::<SCALE>(_mm512_sub_epi16(_mm512_cvtepu8_epi16(u_high), uv_corr));
        let v_high =
            _mm512_slli_epi16::<SCALE>(_mm512_sub_epi16(_mm512_cvtepu8_epi16(v_high0), uv_corr));
        let y_high0 = _mm512_mulhrs_epi16(
            _mm512_slli_epi16::<SCALE>(_mm512_cvtepu8_epi16(_mm512_extracti64x4_epi64::<1>(
                y_values0,
            ))),
            v_luma_coeff,
        );
        let y_high1 = _mm512_mulhrs_epi16(
            _mm512_slli_epi16::<SCALE>(_mm512_cvtepu8_epi16(_mm512_extracti64x4_epi64::<1>(
                y_values1,
            ))),
            v_luma_coeff,
        );

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
        let y_low0 = _mm512_mulhrs_epi16(
            _mm512_slli_epi16::<SCALE>(_mm512_cvtepu8_epi16(_mm512_castsi512_si256(y_values0))),
            v_luma_coeff,
        );
        let y_low1 = _mm512_mulhrs_epi16(
            _mm512_slli_epi16::<SCALE>(_mm512_cvtepu8_epi16(_mm512_castsi512_si256(y_values0))),
            v_luma_coeff,
        );

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

        avx512_store_u8::<DESTINATION_CHANNELS>(
            rgba0.get_unchecked_mut(dst_shift..).as_mut_ptr(),
            r_values0,
            g_values0,
            b_values0,
            v_alpha,
        );

        avx512_store_u8::<DESTINATION_CHANNELS>(
            rgba1.get_unchecked_mut(dst_shift..).as_mut_ptr(),
            r_values1,
            g_values1,
            b_values1,
            v_alpha,
        );

        cx += 64;
        uv_x += 64;
    }

    ProcessedOffset { cx, ux: uv_x }
}
