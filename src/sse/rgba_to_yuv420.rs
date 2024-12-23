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
    _mm_affine_dot, _mm_load_deinterleave_half_rgb_for_yuv, _mm_load_deinterleave_rgb_for_yuv,
    sse_pairwise_avg_epi16_epi8,
};
use crate::yuv_support::{CbCrForwardTransform, YuvChromaRange, YuvSourceChannels};
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

pub(crate) fn sse_rgba_to_yuv_row420<const ORIGIN_CHANNELS: u8, const PRECISION: i32>(
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
        sse_rgba_to_yuv_row_impl420::<ORIGIN_CHANNELS, PRECISION>(
            transform, range, y_plane0, y_plane1, u_plane, v_plane, rgba0, rgba1, start_cx,
            start_ux, width,
        )
    }
}

#[target_feature(enable = "sse4.1")]
unsafe fn sse_rgba_to_yuv_row_impl420<const ORIGIN_CHANNELS: u8, const PRECISION: i32>(
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

    const V_SCALE: i32 = 2;
    let bias_y = range.bias_y as i16;
    let bias_uv = range.bias_uv as i16;

    let i_cap_uv = _mm_set1_epi16(range.bias_y as i16 + range.range_uv as i16);

    let zeros = _mm_setzero_si128();

    let y_bias = _mm_set1_epi16(bias_y);
    let uv_bias = _mm_set1_epi16(bias_uv);
    let y_base = _mm_set1_epi32(bias_y as i32 * (1 << PRECISION) + (1 << (PRECISION - 1)) - 1);
    let v_yr_yg = _mm_set1_epi32(transform.interleaved_yr_yg());
    let v_yb = _mm_set1_epi16(transform.yb as i16);
    let v_cb_r = _mm_set1_epi16(transform.cb_r as i16);
    let v_cb_g = _mm_set1_epi16(transform.cb_g as i16);
    let v_cb_b = _mm_set1_epi16(transform.cb_b as i16);
    let v_cr_r = _mm_set1_epi16(transform.cr_r as i16);
    let v_cr_g = _mm_set1_epi16(transform.cr_g as i16);
    let v_cr_b = _mm_set1_epi16(transform.cr_b as i16);

    while cx + 16 < width {
        let px = cx * channels;

        let row_start0 = rgba0.get_unchecked(px..).as_ptr();
        let (r_values0, g_values0, b_values0) =
            _mm_load_deinterleave_rgb_for_yuv::<ORIGIN_CHANNELS>(row_start0);

        let row_start1 = rgba1.get_unchecked(px..).as_ptr();
        let (r_values1, g_values1, b_values1) =
            _mm_load_deinterleave_rgb_for_yuv::<ORIGIN_CHANNELS>(row_start1);

        let r0_lo16 = _mm_unpacklo_epi8(r_values0, zeros);
        let r0_hi16 = _mm_unpackhi_epi8(r_values0, zeros);
        let g0_lo16 = _mm_unpacklo_epi8(g_values0, zeros);
        let g0_hi16 = _mm_unpackhi_epi8(g_values0, zeros);
        let b0_lo16 = _mm_unpacklo_epi8(b_values0, zeros);
        let b0_hi16 = _mm_unpackhi_epi8(b_values0, zeros);

        let y0_l = _mm_affine_dot::<PRECISION>(y_base, r0_lo16, g0_lo16, b0_lo16, v_yr_yg, v_yb);

        let y0_h = _mm_affine_dot::<PRECISION>(y_base, r0_hi16, g0_hi16, b0_hi16, v_yr_yg, v_yb);

        let r1_low = _mm_unpacklo_epi8(r_values1, zeros);
        let r1_high = _mm_unpackhi_epi8(r_values1, zeros);
        let g1_low = _mm_unpacklo_epi8(g_values1, zeros);
        let g1_high = _mm_unpackhi_epi8(g_values1, zeros);
        let b1_low = _mm_unpacklo_epi8(b_values1, zeros);
        let b1_high = _mm_unpackhi_epi8(b_values1, zeros);

        let y1_l = _mm_affine_dot::<PRECISION>(y_base, r1_low, g1_low, b1_low, v_yr_yg, v_yb);

        let y1_h = _mm_affine_dot::<PRECISION>(y_base, r1_high, g1_high, b1_high, v_yr_yg, v_yb);

        let y0_yuv = _mm_packus_epi16(y0_l, y0_h);
        let y1_yuv = _mm_packus_epi16(y1_l, y1_h);

        _mm_storeu_si128(
            y_plane0.get_unchecked_mut(cx..).as_mut_ptr() as *mut __m128i,
            y0_yuv,
        );
        _mm_storeu_si128(
            y_plane1.get_unchecked_mut(cx..).as_mut_ptr() as *mut __m128i,
            y1_yuv,
        );

        let r1 = _mm_slli_epi16::<V_SCALE>(sse_pairwise_avg_epi16_epi8(r_values0, r_values1));
        let g1 = _mm_slli_epi16::<V_SCALE>(sse_pairwise_avg_epi16_epi8(g_values0, g_values1));
        let b1 = _mm_slli_epi16::<V_SCALE>(sse_pairwise_avg_epi16_epi8(b_values0, b_values1));

        let cbk = _mm_max_epi16(
            _mm_min_epi16(
                _mm_add_epi16(
                    uv_bias,
                    _mm_add_epi16(
                        _mm_add_epi16(_mm_mulhrs_epi16(r1, v_cb_r), _mm_mulhrs_epi16(g1, v_cb_g)),
                        _mm_mulhrs_epi16(b1, v_cb_b),
                    ),
                ),
                i_cap_uv,
            ),
            y_bias,
        );

        let crk = _mm_max_epi16(
            _mm_min_epi16(
                _mm_add_epi16(
                    uv_bias,
                    _mm_add_epi16(
                        _mm_add_epi16(_mm_mulhrs_epi16(r1, v_cr_r), _mm_mulhrs_epi16(g1, v_cr_g)),
                        _mm_mulhrs_epi16(b1, v_cr_b),
                    ),
                ),
                i_cap_uv,
            ),
            y_bias,
        );

        let cb = _mm_packus_epi16(cbk, cbk);
        let cr = _mm_packus_epi16(crk, crk);

        std::ptr::copy_nonoverlapping(&cb as *const _ as *const u8, u_ptr.add(uv_x), 8);
        std::ptr::copy_nonoverlapping(&cr as *const _ as *const u8, v_ptr.add(uv_x), 8);

        uv_x += 8;
        cx += 16;
    }

    while cx + 8 < width {
        let px = cx * channels;
        let row_start0 = rgba0.get_unchecked(px..).as_ptr();
        let (r_values0, g_values0, b_values0) =
            _mm_load_deinterleave_half_rgb_for_yuv::<ORIGIN_CHANNELS>(row_start0);
        let row_start1 = rgba1.get_unchecked(px..).as_ptr();
        let (r_values1, g_values1, b_values1) =
            _mm_load_deinterleave_half_rgb_for_yuv::<ORIGIN_CHANNELS>(row_start1);

        let r0_low = _mm_unpacklo_epi8(r_values0, zeros);
        let g0_low = _mm_unpacklo_epi8(g_values0, zeros);
        let b0_low = _mm_unpacklo_epi8(b_values0, zeros);

        let y0_l = _mm_affine_dot::<PRECISION>(y_base, r0_low, g0_low, b0_low, v_yr_yg, v_yb);

        let r1_low = _mm_unpacklo_epi8(r_values1, zeros);
        let g1_low = _mm_unpacklo_epi8(g_values1, zeros);
        let b1_low = _mm_unpacklo_epi8(b_values1, zeros);

        let y1_l = _mm_affine_dot::<PRECISION>(y_base, r1_low, g1_low, b1_low, v_yr_yg, v_yb);

        let y0_yuv = _mm_packus_epi16(y0_l, _mm_setzero_si128());
        let y1_yuv = _mm_packus_epi16(y1_l, _mm_setzero_si128());

        std::ptr::copy_nonoverlapping(
            &y0_yuv as *const _ as *const u8,
            y_plane0.get_unchecked_mut(cx..).as_mut_ptr(),
            8,
        );
        std::ptr::copy_nonoverlapping(
            &y1_yuv as *const _ as *const u8,
            y_plane1.get_unchecked_mut(cx..).as_mut_ptr(),
            8,
        );

        let r1 = _mm_slli_epi16::<V_SCALE>(sse_pairwise_avg_epi16_epi8(r_values0, r_values1));
        let g1 = _mm_slli_epi16::<V_SCALE>(sse_pairwise_avg_epi16_epi8(g_values0, g_values1));
        let b1 = _mm_slli_epi16::<V_SCALE>(sse_pairwise_avg_epi16_epi8(b_values0, b_values1));

        let cbk = _mm_max_epi16(
            _mm_min_epi16(
                _mm_add_epi16(
                    uv_bias,
                    _mm_add_epi16(
                        _mm_add_epi16(_mm_mulhrs_epi16(r1, v_cb_r), _mm_mulhrs_epi16(g1, v_cb_g)),
                        _mm_mulhrs_epi16(b1, v_cb_b),
                    ),
                ),
                i_cap_uv,
            ),
            y_bias,
        );

        let crk = _mm_max_epi16(
            _mm_min_epi16(
                _mm_add_epi16(
                    uv_bias,
                    _mm_add_epi16(
                        _mm_add_epi16(_mm_mulhrs_epi16(r1, v_cr_r), _mm_mulhrs_epi16(g1, v_cr_g)),
                        _mm_mulhrs_epi16(b1, v_cr_b),
                    ),
                ),
                i_cap_uv,
            ),
            y_bias,
        );

        let cb = _mm_packus_epi16(cbk, cbk);
        let cr = _mm_packus_epi16(crk, crk);

        (u_ptr.add(uv_x) as *mut i32).write_unaligned(_mm_extract_epi32::<0>(cb));
        (v_ptr.add(uv_x) as *mut i32).write_unaligned(_mm_extract_epi32::<0>(cr));

        uv_x += 4;
        cx += 8;
    }

    ProcessedOffset { cx, ux: uv_x }
}
