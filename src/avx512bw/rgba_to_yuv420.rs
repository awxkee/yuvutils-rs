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
    _mm512_affine_dot, avx512_load_rgb_u8, avx512_pack_u16, avx512_pairwise_avg_epi16_epi8,
};
use crate::internals::ProcessedOffset;
use crate::yuv_support::{CbCrForwardTransform, YuvChromaRange, YuvSourceChannels};
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
use std::ops::Shl;

pub(crate) fn avx512_rgba_to_yuv420<const ORIGIN_CHANNELS: u8, const HAS_VBMI: bool>(
    transform: &CbCrForwardTransform<i32>,
    range: &YuvChromaRange,
    y_plane0: &mut [u8],
    y_plane1: &mut [u8],
    u_plane: &mut [u8],
    v_plane: &mut [u8],
    rgba0: &[u8],
    rgba1: &[u8],
    start_cx: usize,
    start_ux: usize,
    width: usize,
) -> ProcessedOffset {
    unsafe {
        if HAS_VBMI {
            avx512_rgba_to_yuv_bmi_impl420::<ORIGIN_CHANNELS>(
                transform, range, y_plane0, y_plane1, u_plane, v_plane, rgba0, rgba1, start_cx,
                start_ux, width,
            )
        } else {
            avx512_rgba_to_yuv_def_impl420::<ORIGIN_CHANNELS>(
                transform, range, y_plane0, y_plane1, u_plane, v_plane, rgba0, rgba1, start_cx,
                start_ux, width,
            )
        }
    }
}

#[target_feature(enable = "avx512bw", enable = "avx512f", enable = "avx512vbmi")]
unsafe fn avx512_rgba_to_yuv_bmi_impl420<const ORIGIN_CHANNELS: u8>(
    transform: &CbCrForwardTransform<i32>,
    range: &YuvChromaRange,
    y_plane0: &mut [u8],
    y_plane1: &mut [u8],
    u_plane: &mut [u8],
    v_plane: &mut [u8],
    rgba0: &[u8],
    rgba1: &[u8],
    start_cx: usize,
    start_ux: usize,
    width: usize,
) -> ProcessedOffset {
    avx512_rgba_to_yuv_impl420::<ORIGIN_CHANNELS, true>(
        transform, range, y_plane0, y_plane1, u_plane, v_plane, rgba0, rgba1, start_cx, start_ux,
        width,
    )
}

#[target_feature(enable = "avx512bw", enable = "avx512f")]
unsafe fn avx512_rgba_to_yuv_def_impl420<const ORIGIN_CHANNELS: u8>(
    transform: &CbCrForwardTransform<i32>,
    range: &YuvChromaRange,
    y_plane0: &mut [u8],
    y_plane1: &mut [u8],
    u_plane: &mut [u8],
    v_plane: &mut [u8],
    rgba0: &[u8],
    rgba1: &[u8],
    start_cx: usize,
    start_ux: usize,
    width: usize,
) -> ProcessedOffset {
    avx512_rgba_to_yuv_impl420::<ORIGIN_CHANNELS, false>(
        transform, range, y_plane0, y_plane1, u_plane, v_plane, rgba0, rgba1, start_cx, start_ux,
        width,
    )
}

#[inline(always)]
unsafe fn avx512_rgba_to_yuv_impl420<const ORIGIN_CHANNELS: u8, const HAS_VBMI: bool>(
    transform: &CbCrForwardTransform<i32>,
    range: &YuvChromaRange,
    y_plane0: &mut [u8],
    y_plane1: &mut [u8],
    u_plane: &mut [u8],
    v_plane: &mut [u8],
    rgba0: &[u8],
    rgba1: &[u8],
    start_cx: usize,
    start_ux: usize,
    width: usize,
) -> ProcessedOffset {
    let source_channels: YuvSourceChannels = ORIGIN_CHANNELS.into();
    let channels = source_channels.get_channels_count();

    let u_ptr = u_plane.as_mut_ptr();
    let v_ptr = v_plane.as_mut_ptr();

    let mut cx = start_cx;
    let mut uv_x = start_ux;

    const V_SCALE: u32 = 2;

    let bias_y = range.bias_y as i16;
    let bias_uv = range.bias_uv as i16;

    let cap_uv = range.bias_y as i16 + range.range_uv as i16;

    const PREC: u32 = 13;

    let y_bias = _mm512_set1_epi16(bias_y);
    let y_base = _mm512_set1_epi32(bias_y as i32 * (1 << PREC) + (1 << (PREC - 1)) - 1);
    let uv_bias = _mm512_set1_epi16(bias_uv);
    let v_yr_yg = _mm512_set1_epi32(transform.yg.shl(16) | transform.yr);
    let v_yb = _mm512_set1_epi16(transform.yb as i16);
    let v_cb_r = _mm512_set1_epi16(transform.cb_r as i16);
    let v_cb_g = _mm512_set1_epi16(transform.cb_g as i16);
    let v_cb_b = _mm512_set1_epi16(transform.cb_b as i16);
    let v_cr_r = _mm512_set1_epi16(transform.cr_r as i16);
    let v_cr_g = _mm512_set1_epi16(transform.cr_g as i16);
    let v_cr_b = _mm512_set1_epi16(transform.cr_b as i16);

    while cx + 64 < width {
        let px = cx * channels;

        let (r_values0, g_values0, b_values0) =
            avx512_load_rgb_u8::<ORIGIN_CHANNELS, HAS_VBMI>(rgba0.get_unchecked(px..).as_ptr());

        let (r_values1, g_values1, b_values1) =
            avx512_load_rgb_u8::<ORIGIN_CHANNELS, HAS_VBMI>(rgba1.get_unchecked(px..).as_ptr());

        let r0_lo16 = _mm512_cvtepu8_epi16(_mm512_castsi512_si256(r_values0));
        let r0_hi16 = _mm512_cvtepu8_epi16(_mm512_extracti64x4_epi64::<1>(r_values0));
        let g0_lo16 = _mm512_cvtepu8_epi16(_mm512_castsi512_si256(g_values0));
        let g0_hi16 = _mm512_cvtepu8_epi16(_mm512_extracti64x4_epi64::<1>(g_values0));
        let b0_lo16 = _mm512_cvtepu8_epi16(_mm512_castsi512_si256(b_values0));
        let b0_hi16 = _mm512_cvtepu8_epi16(_mm512_extracti64x4_epi64::<1>(b_values0));

        let y_l0 = _mm512_affine_dot::<PREC>(y_base, r0_lo16, g0_lo16, b0_lo16, v_yr_yg, v_yb);
        let y_h0 = _mm512_affine_dot::<PREC>(y_base, r0_hi16, g0_hi16, b0_hi16, v_yr_yg, v_yb);

        let r1_lo16 = _mm512_cvtepu8_epi16(_mm512_castsi512_si256(r_values1));
        let r1_hi16 = _mm512_cvtepu8_epi16(_mm512_extracti64x4_epi64::<1>(r_values1));
        let g1_lo16 = _mm512_cvtepu8_epi16(_mm512_castsi512_si256(g_values1));
        let g1_hi16 = _mm512_cvtepu8_epi16(_mm512_extracti64x4_epi64::<1>(g_values1));
        let b1_lo16 = _mm512_cvtepu8_epi16(_mm512_castsi512_si256(b_values1));
        let b1_hi16 = _mm512_cvtepu8_epi16(_mm512_extracti64x4_epi64::<1>(b_values1));

        let y_yuv0 = avx512_pack_u16(y_l0, y_h0);
        _mm512_storeu_si512(
            y_plane0.get_unchecked_mut(cx..).as_mut_ptr() as *mut i32,
            y_yuv0,
        );

        let y_l1 = _mm512_affine_dot::<PREC>(y_base, r1_lo16, g1_lo16, b1_lo16, v_yr_yg, v_yb);
        let y_h1 = _mm512_affine_dot::<PREC>(y_base, r1_hi16, g1_hi16, b1_hi16, v_yr_yg, v_yb);

        let y_yuv1 = avx512_pack_u16(y_l1, y_h1);
        _mm512_storeu_si512(
            y_plane1.get_unchecked_mut(cx..).as_mut_ptr() as *mut i32,
            y_yuv1,
        );

        let r1 = _mm512_slli_epi16::<V_SCALE>(avx512_pairwise_avg_epi16_epi8(r_values0, r_values1));
        let g1 = _mm512_slli_epi16::<V_SCALE>(avx512_pairwise_avg_epi16_epi8(g_values0, g_values1));
        let b1 = _mm512_slli_epi16::<V_SCALE>(avx512_pairwise_avg_epi16_epi8(b_values0, b_values1));

        let i_cap_uv = _mm512_set1_epi16(cap_uv);

        let cbk = _mm512_max_epi16(
            _mm512_min_epi16(
                _mm512_add_epi16(
                    uv_bias,
                    _mm512_add_epi16(
                        _mm512_add_epi16(
                            _mm512_mulhrs_epi16(r1, v_cb_r),
                            _mm512_mulhrs_epi16(g1, v_cb_g),
                        ),
                        _mm512_mulhrs_epi16(b1, v_cb_b),
                    ),
                ),
                i_cap_uv,
            ),
            y_bias,
        );

        let crk = _mm512_max_epi16(
            _mm512_min_epi16(
                _mm512_add_epi16(
                    uv_bias,
                    _mm512_add_epi16(
                        _mm512_add_epi16(
                            _mm512_mulhrs_epi16(r1, v_cr_r),
                            _mm512_mulhrs_epi16(g1, v_cr_g),
                        ),
                        _mm512_mulhrs_epi16(b1, v_cr_b),
                    ),
                ),
                i_cap_uv,
            ),
            y_bias,
        );

        let cb = avx512_pack_u16(cbk, cbk);
        let cr = avx512_pack_u16(crk, crk);

        _mm256_storeu_si256(
            u_ptr.add(uv_x) as *mut _ as *mut __m256i,
            _mm512_castsi512_si256(cb),
        );
        _mm256_storeu_si256(
            v_ptr.add(uv_x) as *mut _ as *mut __m256i,
            _mm512_castsi512_si256(cr),
        );
        uv_x += 32;

        cx += 64;
    }

    ProcessedOffset { cx, ux: uv_x }
}
