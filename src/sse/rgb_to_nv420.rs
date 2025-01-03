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
use crate::sse::{_mm_load_deinterleave_rgb_for_yuv, sse_pairwise_avg_epi8_f};
use crate::yuv_support::{CbCrForwardTransform, YuvChromaRange, YuvNVOrder, YuvSourceChannels};
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

pub(crate) fn sse_rgba_to_nv_row420<
    const ORIGIN_CHANNELS: u8,
    const UV_ORDER: u8,
    const PRECISION: i32,
>(
    y_plane0: &mut [u8],
    y_plane1: &mut [u8],
    uv_plane: &mut [u8],
    rgba0: &[u8],
    rgba1: &[u8],
    width: u32,
    range: &YuvChromaRange,
    transform: &CbCrForwardTransform<i32>,
    start_cx: usize,
    start_ux: usize,
) -> ProcessedOffset {
    unsafe {
        sse_rgba_to_nv_row_impl420::<ORIGIN_CHANNELS, UV_ORDER, PRECISION>(
            y_plane0, y_plane1, uv_plane, rgba0, rgba1, width, range, transform, start_cx, start_ux,
        )
    }
}

#[target_feature(enable = "sse4.1")]
unsafe fn sse_rgba_to_nv_row_impl420<
    const ORIGIN_CHANNELS: u8,
    const UV_ORDER: u8,
    const PRECISION: i32,
>(
    y_plane0: &mut [u8],
    y_plane1: &mut [u8],
    uv_plane: &mut [u8],
    rgba0: &[u8],
    rgba1: &[u8],
    width: u32,
    range: &YuvChromaRange,
    transform: &CbCrForwardTransform<i32>,
    start_cx: usize,
    start_ux: usize,
) -> ProcessedOffset {
    let order: YuvNVOrder = UV_ORDER.into();
    let source_channels: YuvSourceChannels = ORIGIN_CHANNELS.into();
    let channels = source_channels.get_channels_count();

    let uv_ptr = uv_plane.as_mut_ptr();

    let mut cx = start_cx;
    let mut uv_x = start_ux;

    const V_S: i32 = 4;
    const A_E: i32 = 2;
    let y_bias = _mm_set1_epi16(range.bias_y as i16 * (1 << A_E));
    let uv_bias = _mm_set1_epi16(range.bias_uv as i16 * (1 << A_E) + (1 << (A_E - 1)) - 1);
    let v_yr = _mm_set1_epi16(transform.yr as i16);
    let v_yg = _mm_set1_epi16(transform.yg as i16);
    let v_yb = _mm_set1_epi16(transform.yb as i16);

    while cx + 16 < width as usize {
        let px = cx * channels;
        let row_start0 = rgba0.get_unchecked(px..).as_ptr();
        let (r_values0, g_values0, b_values0) =
            _mm_load_deinterleave_rgb_for_yuv::<ORIGIN_CHANNELS>(row_start0);

        let r0_lo16 = _mm_srli_epi16::<V_S>(_mm_unpacklo_epi8(r_values0, r_values0));
        let r0_hi16 = _mm_srli_epi16::<V_S>(_mm_unpackhi_epi8(r_values0, r_values0));
        let g0_lo16 = _mm_srli_epi16::<V_S>(_mm_unpacklo_epi8(g_values0, g_values0));
        let g0_hi16 = _mm_srli_epi16::<V_S>(_mm_unpackhi_epi8(g_values0, g_values0));
        let b0_lo16 = _mm_srli_epi16::<V_S>(_mm_unpacklo_epi8(b_values0, b_values0));
        let b0_hi16 = _mm_srli_epi16::<V_S>(_mm_unpackhi_epi8(b_values0, b_values0));

        let y0_l = _mm_srli_epi16::<A_E>(_mm_add_epi16(
            y_bias,
            _mm_add_epi16(
                _mm_add_epi16(
                    _mm_mulhrs_epi16(r0_lo16, v_yr),
                    _mm_mulhrs_epi16(g0_lo16, v_yg),
                ),
                _mm_mulhrs_epi16(b0_lo16, v_yb),
            ),
        ));

        let y0_h = _mm_srli_epi16::<A_E>(_mm_add_epi16(
            y_bias,
            _mm_add_epi16(
                _mm_add_epi16(
                    _mm_mulhrs_epi16(r0_hi16, v_yr),
                    _mm_mulhrs_epi16(g0_hi16, v_yg),
                ),
                _mm_mulhrs_epi16(b0_hi16, v_yb),
            ),
        ));

        let y0_yuv = _mm_packus_epi16(y0_l, y0_h);
        _mm_storeu_si128(
            y_plane0.get_unchecked_mut(cx..).as_mut_ptr() as *mut __m128i,
            y0_yuv,
        );

        let row_start1 = rgba1.get_unchecked(px..).as_ptr();
        let (r_values1, g_values1, b_values1) =
            _mm_load_deinterleave_rgb_for_yuv::<ORIGIN_CHANNELS>(row_start1);

        let r1_lo = _mm_srli_epi16::<V_S>(_mm_unpacklo_epi8(r_values1, r_values1));
        let r1_hi = _mm_srli_epi16::<V_S>(_mm_unpackhi_epi8(r_values1, r_values1));
        let g1_lo = _mm_srli_epi16::<V_S>(_mm_unpacklo_epi8(g_values1, g_values1));
        let g1_hi = _mm_srli_epi16::<V_S>(_mm_unpackhi_epi8(g_values1, g_values1));
        let b1_lo = _mm_srli_epi16::<V_S>(_mm_unpacklo_epi8(b_values1, b_values1));
        let b1_hi = _mm_srli_epi16::<V_S>(_mm_unpackhi_epi8(b_values1, b_values1));

        let y1_l = _mm_srli_epi16::<A_E>(_mm_add_epi16(
            y_bias,
            _mm_add_epi16(
                _mm_add_epi16(_mm_mulhrs_epi16(r1_lo, v_yr), _mm_mulhrs_epi16(g1_lo, v_yg)),
                _mm_mulhrs_epi16(b1_lo, v_yb),
            ),
        ));

        let y1_h = _mm_srli_epi16::<A_E>(_mm_add_epi16(
            y_bias,
            _mm_add_epi16(
                _mm_add_epi16(_mm_mulhrs_epi16(r1_hi, v_yr), _mm_mulhrs_epi16(g1_hi, v_yg)),
                _mm_mulhrs_epi16(b1_hi, v_yb),
            ),
        ));

        let y1_yuv = _mm_packus_epi16(y1_l, y1_h);

        _mm_storeu_si128(
            y_plane1.get_unchecked_mut(cx..).as_mut_ptr() as *mut __m128i,
            y1_yuv,
        );

        let r1 = sse_pairwise_avg_epi8_f(_mm_avg_epu8(r_values0, r_values1), 1 << (16 - V_S - 8));
        let g1 = sse_pairwise_avg_epi8_f(_mm_avg_epu8(g_values0, g_values1), 1 << (16 - V_S - 8));
        let b1 = sse_pairwise_avg_epi8_f(_mm_avg_epu8(b_values0, b_values1), 1 << (16 - V_S - 8));

        let v_cb_r = _mm_set1_epi16(transform.cb_r as i16);
        let v_cb_g = _mm_set1_epi16(transform.cb_g as i16);
        let v_cb_b = _mm_set1_epi16(transform.cb_b as i16);
        let v_cr_r = _mm_set1_epi16(transform.cr_r as i16);
        let v_cr_g = _mm_set1_epi16(transform.cr_g as i16);
        let v_cr_b = _mm_set1_epi16(transform.cr_b as i16);

        let cbk = _mm_srli_epi16::<A_E>(_mm_add_epi16(
            uv_bias,
            _mm_add_epi16(
                _mm_add_epi16(_mm_mulhrs_epi16(r1, v_cb_r), _mm_mulhrs_epi16(g1, v_cb_g)),
                _mm_mulhrs_epi16(b1, v_cb_b),
            ),
        ));

        let crk = _mm_srli_epi16::<A_E>(_mm_add_epi16(
            uv_bias,
            _mm_add_epi16(
                _mm_add_epi16(_mm_mulhrs_epi16(r1, v_cr_r), _mm_mulhrs_epi16(g1, v_cr_g)),
                _mm_mulhrs_epi16(b1, v_cr_b),
            ),
        ));

        let cb = _mm_packus_epi16(cbk, cbk);
        let cr = _mm_packus_epi16(crk, crk);

        let row0 = match order {
            YuvNVOrder::UV => _mm_unpacklo_epi8(cb, cr),
            YuvNVOrder::VU => _mm_unpacklo_epi8(cr, cb),
        };
        let dst_ptr = uv_ptr.add(uv_x);
        _mm_storeu_si128(dst_ptr as *mut __m128i, row0);
        uv_x += 16;

        cx += 16;
    }

    ProcessedOffset { cx, ux: uv_x }
}
