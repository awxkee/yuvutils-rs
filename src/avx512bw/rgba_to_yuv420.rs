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
    avx512_load_rgb_u8, avx512_pack_u16, avx512_pairwise_avg_epi16_epi8,
};
use crate::internals::ProcessedOffset;
use crate::yuv_support::{CbCrForwardTransform, YuvChromaRange, YuvSourceChannels};
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

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

    const V_S: u32 = 4;
    const A_E: u32 = 2;
    let y_bias = _mm512_set1_epi16(range.bias_y as i16 * (1 << A_E) + (1 << (A_E - 1)));
    let uv_bias = _mm512_set1_epi16(range.bias_uv as i16 * (1 << A_E) + (1 << (A_E - 1)));
    let v_yr = _mm512_set1_epi16(transform.yr as i16);
    let v_yg = _mm512_set1_epi16(transform.yg as i16);
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

        let r0_lo16 = _mm512_srli_epi16::<V_S>(_mm512_unpacklo_epi8(r_values0, r_values0));
        let r0_hi16 = _mm512_srli_epi16::<V_S>(_mm512_unpackhi_epi8(r_values0, r_values0));
        let g0_lo16 = _mm512_srli_epi16::<V_S>(_mm512_unpacklo_epi8(g_values0, g_values0));
        let g0_hi16 = _mm512_srli_epi16::<V_S>(_mm512_unpackhi_epi8(g_values0, g_values0));
        let b0_lo16 = _mm512_srli_epi16::<V_S>(_mm512_unpacklo_epi8(b_values0, b_values0));
        let b0_hi16 = _mm512_srli_epi16::<V_S>(_mm512_unpackhi_epi8(b_values0, b_values0));

        let y_l0 = _mm512_srli_epi16::<A_E>(_mm512_add_epi16(
            y_bias,
            _mm512_add_epi16(
                _mm512_add_epi16(
                    _mm512_mulhrs_epi16(r0_lo16, v_yr),
                    _mm512_mulhrs_epi16(g0_lo16, v_yg),
                ),
                _mm512_mulhrs_epi16(b0_lo16, v_yb),
            ),
        ));

        let y_h0 = _mm512_srli_epi16::<A_E>(_mm512_add_epi16(
            y_bias,
            _mm512_add_epi16(
                _mm512_add_epi16(
                    _mm512_mulhrs_epi16(r0_hi16, v_yr),
                    _mm512_mulhrs_epi16(g0_hi16, v_yg),
                ),
                _mm512_mulhrs_epi16(b0_hi16, v_yb),
            ),
        ));

        let y_yuv0 = _mm512_packus_epi16(y_l0, y_h0);
        _mm512_storeu_si512(
            y_plane0.get_unchecked_mut(cx..).as_mut_ptr() as *mut _,
            y_yuv0,
        );

        let r1_lo16 = _mm512_srli_epi16::<V_S>(_mm512_unpacklo_epi8(r_values1, r_values1));
        let r1_hi16 = _mm512_srli_epi16::<V_S>(_mm512_unpackhi_epi8(r_values1, r_values1));
        let g1_lo16 = _mm512_srli_epi16::<V_S>(_mm512_unpacklo_epi8(g_values1, g_values1));
        let g1_hi16 = _mm512_srli_epi16::<V_S>(_mm512_unpackhi_epi8(g_values1, g_values1));
        let b1_lo16 = _mm512_srli_epi16::<V_S>(_mm512_unpacklo_epi8(b_values1, b_values1));
        let b1_hi16 = _mm512_srli_epi16::<V_S>(_mm512_unpackhi_epi8(b_values1, b_values1));

        let y_l1 = _mm512_srli_epi16::<A_E>(_mm512_add_epi16(
            y_bias,
            _mm512_add_epi16(
                _mm512_add_epi16(
                    _mm512_mulhrs_epi16(r1_lo16, v_yr),
                    _mm512_mulhrs_epi16(g1_lo16, v_yg),
                ),
                _mm512_mulhrs_epi16(b1_lo16, v_yb),
            ),
        ));
        let y_h1 = _mm512_srli_epi16::<A_E>(_mm512_add_epi16(
            y_bias,
            _mm512_add_epi16(
                _mm512_add_epi16(
                    _mm512_mulhrs_epi16(r1_hi16, v_yr),
                    _mm512_mulhrs_epi16(g1_hi16, v_yg),
                ),
                _mm512_mulhrs_epi16(b1_hi16, v_yb),
            ),
        ));

        let y_yuv1 = _mm512_packus_epi16(y_l1, y_h1);
        _mm512_storeu_si512(
            y_plane1.get_unchecked_mut(cx..).as_mut_ptr() as *mut _,
            y_yuv1,
        );

        let r1 = avx512_pairwise_avg_epi16_epi8(r_values0, r_values1, 1 << (16 - V_S - 8));
        let g1 = avx512_pairwise_avg_epi16_epi8(g_values0, g_values1, 1 << (16 - V_S - 8));
        let b1 = avx512_pairwise_avg_epi16_epi8(b_values0, b_values1, 1 << (16 - V_S - 8));

        let cbk = _mm512_srli_epi16::<A_E>(_mm512_add_epi16(
            uv_bias,
            _mm512_add_epi16(
                _mm512_add_epi16(
                    _mm512_mulhrs_epi16(r1, v_cb_r),
                    _mm512_mulhrs_epi16(g1, v_cb_g),
                ),
                _mm512_mulhrs_epi16(b1, v_cb_b),
            ),
        ));

        let crk = _mm512_srli_epi16::<A_E>(_mm512_add_epi16(
            uv_bias,
            _mm512_add_epi16(
                _mm512_add_epi16(
                    _mm512_mulhrs_epi16(r1, v_cr_r),
                    _mm512_mulhrs_epi16(g1, v_cr_g),
                ),
                _mm512_mulhrs_epi16(b1, v_cr_b),
            ),
        ));

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
