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
use crate::sse::sse_pairwise_wide_avg;
use crate::sse::sse_support::{sse_deinterleave_rgb, sse_deinterleave_rgba};
use crate::yuv_support::{CbCrForwardTransform, YuvChromaRange, YuvSourceChannels};
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

pub(crate) fn sse_rgba_to_yuv_row420<const ORIGIN_CHANNELS: u8>(
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
        sse_rgba_to_yuv_row_impl420::<ORIGIN_CHANNELS>(
            transform, range, y_plane0, y_plane1, u_plane, v_plane, rgba0, rgba1, start_cx,
            start_ux, width,
        )
    }
}

#[target_feature(enable = "sse4.1")]
unsafe fn sse_rgba_to_yuv_row_impl420<const ORIGIN_CHANNELS: u8>(
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

    let i_bias_y = _mm_set1_epi16(range.bias_y as i16);
    let i_cap_y = _mm_set1_epi16(range.range_y as i16 + range.bias_y as i16);
    let i_cap_uv = _mm_set1_epi16(range.bias_y as i16 + range.range_uv as i16);

    let zeros = _mm_setzero_si128();

    let y_bias = _mm_set1_epi16(bias_y);
    let uv_bias = _mm_set1_epi16(bias_uv);
    let v_yr = _mm_set1_epi16(transform.yr as i16);
    let v_yg = _mm_set1_epi16(transform.yg as i16);
    let v_yb = _mm_set1_epi16(transform.yb as i16);
    let v_cb_r = _mm_set1_epi16(transform.cb_r as i16);
    let v_cb_g = _mm_set1_epi16(transform.cb_g as i16);
    let v_cb_b = _mm_set1_epi16(transform.cb_b as i16);
    let v_cr_r = _mm_set1_epi16(transform.cr_r as i16);
    let v_cr_g = _mm_set1_epi16(transform.cr_g as i16);
    let v_cr_b = _mm_set1_epi16(transform.cr_b as i16);

    while cx + 16 < width {
        let (r_values0, g_values0, b_values0);
        let (r_values1, g_values1, b_values1);

        let px = cx * channels;

        match source_channels {
            YuvSourceChannels::Rgb | YuvSourceChannels::Bgr => {
                let row_start0 = rgba0.get_unchecked(px..).as_ptr();
                let row_1 = _mm_loadu_si128(row_start0 as *const __m128i);
                let row_2 = _mm_loadu_si128(row_start0.add(16) as *const __m128i);
                let row_3 = _mm_loadu_si128(row_start0.add(32) as *const __m128i);

                let (it1, it2, it3) = sse_deinterleave_rgb(row_1, row_2, row_3);
                if source_channels == YuvSourceChannels::Rgb {
                    r_values0 = it1;
                    g_values0 = it2;
                    b_values0 = it3;
                } else {
                    r_values0 = it3;
                    g_values0 = it2;
                    b_values0 = it1;
                }

                let row_start1 = rgba1.get_unchecked(px..).as_ptr();
                let row_11 = _mm_loadu_si128(row_start1 as *const __m128i);
                let row_21 = _mm_loadu_si128(row_start1.add(16) as *const __m128i);
                let row_31 = _mm_loadu_si128(row_start1.add(32) as *const __m128i);

                let (it11, it21, it31) = sse_deinterleave_rgb(row_11, row_21, row_31);
                if source_channels == YuvSourceChannels::Rgb {
                    r_values1 = it11;
                    g_values1 = it21;
                    b_values1 = it31;
                } else {
                    r_values1 = it31;
                    g_values1 = it21;
                    b_values1 = it11;
                }
            }
            YuvSourceChannels::Rgba | YuvSourceChannels::Bgra => {
                let row_start0 = rgba0.get_unchecked(px..).as_ptr();
                let row_1 = _mm_loadu_si128(row_start0 as *const __m128i);
                let row_2 = _mm_loadu_si128(row_start0.add(16) as *const __m128i);
                let row_3 = _mm_loadu_si128(row_start0.add(32) as *const __m128i);
                let row_4 = _mm_loadu_si128(row_start0.add(48) as *const __m128i);

                let (it1, it2, it3, _) = sse_deinterleave_rgba(row_1, row_2, row_3, row_4);
                if source_channels == YuvSourceChannels::Rgba {
                    r_values0 = it1;
                    g_values0 = it2;
                    b_values0 = it3;
                } else {
                    r_values0 = it3;
                    g_values0 = it2;
                    b_values0 = it1;
                }

                let row_start1 = rgba1.get_unchecked(px..).as_ptr();
                let row_11 = _mm_loadu_si128(row_start1 as *const __m128i);
                let row_21 = _mm_loadu_si128(row_start1.add(16) as *const __m128i);
                let row_31 = _mm_loadu_si128(row_start1.add(32) as *const __m128i);
                let row_41 = _mm_loadu_si128(row_start1.add(48) as *const __m128i);

                let (it11, it21, it31, _) = sse_deinterleave_rgba(row_11, row_21, row_31, row_41);
                if source_channels == YuvSourceChannels::Rgba {
                    r_values1 = it11;
                    g_values1 = it21;
                    b_values1 = it31;
                } else {
                    r_values1 = it31;
                    g_values1 = it21;
                    b_values1 = it11;
                }
            }
        }

        let r0_low = _mm_slli_epi16::<V_SCALE>(_mm_cvtepu8_epi16(r_values0));
        let r0_high = _mm_slli_epi16::<V_SCALE>(_mm_unpackhi_epi8(r_values0, zeros));
        let g0_low = _mm_slli_epi16::<V_SCALE>(_mm_cvtepu8_epi16(g_values0));
        let g0_high = _mm_slli_epi16::<V_SCALE>(_mm_unpackhi_epi8(g_values0, zeros));
        let b0_low = _mm_slli_epi16::<V_SCALE>(_mm_cvtepu8_epi16(b_values0));
        let b0_high = _mm_slli_epi16::<V_SCALE>(_mm_unpackhi_epi8(b_values0, zeros));

        let y0_l = _mm_max_epi16(
            _mm_min_epi16(
                _mm_add_epi16(
                    y_bias,
                    _mm_add_epi16(
                        _mm_add_epi16(
                            _mm_mulhrs_epi16(r0_low, v_yr),
                            _mm_mulhrs_epi16(g0_low, v_yg),
                        ),
                        _mm_mulhrs_epi16(b0_low, v_yb),
                    ),
                ),
                i_cap_y,
            ),
            i_bias_y,
        );

        let y0_h = _mm_max_epi16(
            _mm_min_epi16(
                _mm_add_epi16(
                    y_bias,
                    _mm_add_epi16(
                        _mm_add_epi16(
                            _mm_mulhrs_epi16(r0_high, v_yr),
                            _mm_mulhrs_epi16(g0_high, v_yg),
                        ),
                        _mm_mulhrs_epi16(b0_high, v_yb),
                    ),
                ),
                i_cap_y,
            ),
            i_bias_y,
        );

        let r1_low = _mm_slli_epi16::<V_SCALE>(_mm_cvtepu8_epi16(r_values1));
        let r1_high = _mm_slli_epi16::<V_SCALE>(_mm_unpackhi_epi8(r_values1, zeros));
        let g1_low = _mm_slli_epi16::<V_SCALE>(_mm_cvtepu8_epi16(g_values1));
        let g1_high = _mm_slli_epi16::<V_SCALE>(_mm_unpackhi_epi8(g_values1, zeros));
        let b1_low = _mm_slli_epi16::<V_SCALE>(_mm_cvtepu8_epi16(b_values1));
        let b1_high = _mm_slli_epi16::<V_SCALE>(_mm_unpackhi_epi8(b_values1, zeros));

        let y1_l = _mm_max_epi16(
            _mm_min_epi16(
                _mm_add_epi16(
                    y_bias,
                    _mm_add_epi16(
                        _mm_add_epi16(
                            _mm_mulhrs_epi16(r1_low, v_yr),
                            _mm_mulhrs_epi16(g1_low, v_yg),
                        ),
                        _mm_mulhrs_epi16(b1_low, v_yb),
                    ),
                ),
                i_cap_y,
            ),
            i_bias_y,
        );

        let y1_h = _mm_max_epi16(
            _mm_min_epi16(
                _mm_add_epi16(
                    y_bias,
                    _mm_add_epi16(
                        _mm_add_epi16(
                            _mm_mulhrs_epi16(r1_high, v_yr),
                            _mm_mulhrs_epi16(g1_high, v_yg),
                        ),
                        _mm_mulhrs_epi16(b1_high, v_yb),
                    ),
                ),
                i_cap_y,
            ),
            i_bias_y,
        );

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

        let r1 = _mm_slli_epi16::<V_SCALE>(_mm_avg_epu16(
            sse_pairwise_wide_avg(r_values0),
            sse_pairwise_wide_avg(r_values1),
        ));
        let g1 = _mm_slli_epi16::<V_SCALE>(_mm_avg_epu16(
            sse_pairwise_wide_avg(g_values0),
            sse_pairwise_wide_avg(g_values1),
        ));
        let b1 = _mm_slli_epi16::<V_SCALE>(_mm_avg_epu16(
            sse_pairwise_wide_avg(b_values0),
            sse_pairwise_wide_avg(b_values1),
        ));

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
            i_bias_y,
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
            i_bias_y,
        );

        let cb = _mm_packus_epi16(cbk, cbk);
        let cr = _mm_packus_epi16(crk, crk);

        std::ptr::copy_nonoverlapping(&cb as *const _ as *const u8, u_ptr.add(uv_x), 8);
        std::ptr::copy_nonoverlapping(&cr as *const _ as *const u8, v_ptr.add(uv_x), 8);
        uv_x += 8;
        cx += 16;
    }

    ProcessedOffset { cx, ux: uv_x }
}
