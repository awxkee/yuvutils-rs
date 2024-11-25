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
use crate::sse::sse_support::{sse_deinterleave_rgb, sse_deinterleave_rgba};
use crate::yuv_support::{
    CbCrForwardTransform, YuvChromaRange, YuvChromaSubsampling, YuvNVOrder, YuvSourceChannels,
};
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

pub(crate) fn sse_rgba_to_nv_row<
    const ORIGIN_CHANNELS: u8,
    const UV_ORDER: u8,
    const SAMPLING: u8,
>(
    y_plane: &mut [u8],
    uv_plane: &mut [u8],
    rgba: &[u8],
    width: u32,
    range: &YuvChromaRange,
    transform: &CbCrForwardTransform<i32>,
    start_cx: usize,
    start_ux: usize,
    compute_uv_row: bool,
) -> ProcessedOffset {
    unsafe {
        sse_rgba_to_nv_row_impl::<ORIGIN_CHANNELS, UV_ORDER, SAMPLING>(
            y_plane,
            uv_plane,
            rgba,
            width,
            range,
            transform,
            start_cx,
            start_ux,
            compute_uv_row,
        )
    }
}

#[target_feature(enable = "sse4.1")]
unsafe fn sse_rgba_to_nv_row_impl<
    const ORIGIN_CHANNELS: u8,
    const UV_ORDER: u8,
    const SAMPLING: u8,
>(
    y_plane: &mut [u8],
    uv_plane: &mut [u8],
    rgba: &[u8],
    width: u32,
    range: &YuvChromaRange,
    transform: &CbCrForwardTransform<i32>,
    start_cx: usize,
    start_ux: usize,
    compute_uv_row: bool,
) -> ProcessedOffset {
    let order: YuvNVOrder = UV_ORDER.into();
    let chroma_subsampling: YuvChromaSubsampling = SAMPLING.into();
    let source_channels: YuvSourceChannels = ORIGIN_CHANNELS.into();
    let channels = source_channels.get_channels_count();

    let y_ptr = y_plane.as_mut_ptr();
    let uv_ptr = uv_plane.as_mut_ptr();
    let rgba_ptr = rgba.as_ptr();

    let mut cx = start_cx;
    let mut uv_x = start_ux;

    const V_SCALE: i32 = 3;
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

    while cx + 16 < width as usize {
        let (r_values, g_values, b_values);

        let px = cx * channels;

        match source_channels {
            YuvSourceChannels::Rgb | YuvSourceChannels::Bgr => {
                let row_start = rgba_ptr.add(px);
                let row_1 = _mm_loadu_si128(row_start as *const __m128i);
                let row_2 = _mm_loadu_si128(row_start.add(16) as *const __m128i);
                let row_3 = _mm_loadu_si128(row_start.add(32) as *const __m128i);

                let (it1, it2, it3) = sse_deinterleave_rgb(row_1, row_2, row_3);
                if source_channels == YuvSourceChannels::Rgb {
                    r_values = it1;
                    g_values = it2;
                    b_values = it3;
                } else {
                    r_values = it3;
                    g_values = it2;
                    b_values = it1;
                }
            }
            YuvSourceChannels::Rgba | YuvSourceChannels::Bgra => {
                let row_start = rgba_ptr.add(px);
                let row_1 = _mm_loadu_si128(row_start as *const __m128i);
                let row_2 = _mm_loadu_si128(row_start.add(16) as *const __m128i);
                let row_3 = _mm_loadu_si128(row_start.add(32) as *const __m128i);
                let row_4 = _mm_loadu_si128(row_start.add(48) as *const __m128i);

                let (it1, it2, it3, _) = sse_deinterleave_rgba(row_1, row_2, row_3, row_4);
                if source_channels == YuvSourceChannels::Rgba {
                    r_values = it1;
                    g_values = it2;
                    b_values = it3;
                } else {
                    r_values = it3;
                    g_values = it2;
                    b_values = it1;
                }
            }
        }

        let r_low = _mm_slli_epi16::<V_SCALE>(_mm_cvtepu8_epi16(r_values));
        let r_high = _mm_slli_epi16::<V_SCALE>(_mm_unpackhi_epi8(r_values, zeros));
        let g_low = _mm_slli_epi16::<V_SCALE>(_mm_cvtepu8_epi16(g_values));
        let g_high = _mm_slli_epi16::<V_SCALE>(_mm_unpackhi_epi8(g_values, zeros));
        let b_low = _mm_slli_epi16::<V_SCALE>(_mm_cvtepu8_epi16(b_values));
        let b_high = _mm_slli_epi16::<V_SCALE>(_mm_unpackhi_epi8(b_values, zeros));

        let y_l = _mm_max_epi16(
            _mm_min_epi16(
                (_mm_add_epi16(
                    y_bias,
                    _mm_add_epi16(
                        _mm_add_epi16(_mm_mulhrs_epi16(r_low, v_yr), _mm_mulhrs_epi16(g_low, v_yg)),
                        _mm_mulhrs_epi16(b_low, v_yb),
                    ),
                )),
                i_cap_y,
            ),
            i_bias_y,
        );

        let y_h = _mm_max_epi16(
            _mm_min_epi16(
                (_mm_add_epi16(
                    y_bias,
                    _mm_add_epi16(
                        _mm_add_epi16(
                            _mm_mulhrs_epi16(r_high, v_yr),
                            _mm_mulhrs_epi16(g_high, v_yg),
                        ),
                        _mm_mulhrs_epi16(b_high, v_yb),
                    ),
                )),
                i_cap_y,
            ),
            i_bias_y,
        );

        let y_yuv = _mm_packus_epi16(y_l, y_h);
        _mm_storeu_si128(y_ptr.add(cx) as *mut __m128i, y_yuv);

        if chroma_subsampling == YuvChromaSubsampling::Yuv444 {
            let cb_l = _mm_max_epi16(
                _mm_min_epi16(
                    (_mm_add_epi16(
                        uv_bias,
                        _mm_add_epi16(
                            _mm_add_epi16(
                                _mm_mulhrs_epi16(r_low, v_cb_r),
                                _mm_mulhrs_epi16(g_low, v_cb_g),
                            ),
                            _mm_mulhrs_epi16(b_low, v_cb_b),
                        ),
                    )),
                    i_cap_uv,
                ),
                i_bias_y,
            );
            let cr_l = _mm_max_epi16(
                _mm_min_epi16(
                    _mm_add_epi16(
                        uv_bias,
                        _mm_add_epi16(
                            _mm_add_epi16(
                                _mm_mulhrs_epi16(r_low, v_cr_r),
                                _mm_mulhrs_epi16(g_low, v_cr_g),
                            ),
                            _mm_mulhrs_epi16(b_low, v_cr_b),
                        ),
                    ),
                    i_cap_uv,
                ),
                i_bias_y,
            );
            let cb_h = _mm_max_epi16(
                _mm_min_epi16(
                    (_mm_add_epi16(
                        uv_bias,
                        _mm_add_epi16(
                            _mm_add_epi16(
                                _mm_mulhrs_epi16(r_high, v_cb_r),
                                _mm_mulhrs_epi16(g_high, v_cb_g),
                            ),
                            _mm_mulhrs_epi16(b_high, v_cb_b),
                        ),
                    )),
                    i_cap_uv,
                ),
                i_bias_y,
            );
            let cr_h = _mm_max_epi16(
                _mm_min_epi16(
                    (_mm_add_epi16(
                        uv_bias,
                        _mm_add_epi16(
                            _mm_add_epi16(
                                _mm_mulhrs_epi16(r_high, v_cr_r),
                                _mm_mulhrs_epi16(g_high, v_cr_g),
                            ),
                            _mm_mulhrs_epi16(b_high, v_cr_b),
                        ),
                    )),
                    i_cap_uv,
                ),
                i_bias_y,
            );

            let cb = _mm_packus_epi16(cb_l, cb_h);
            let cr = _mm_packus_epi16(cr_l, cr_h);

            let row0 = match order {
                YuvNVOrder::UV => _mm_unpacklo_epi8(cb, cr),
                YuvNVOrder::VU => _mm_unpacklo_epi8(cr, cb),
            };
            let row1 = match order {
                YuvNVOrder::UV => _mm_unpackhi_epi8(cb, cr),
                YuvNVOrder::VU => _mm_unpackhi_epi8(cr, cb),
            };

            let dst_ptr = uv_ptr.add(uv_x);
            _mm_storeu_si128(dst_ptr as *mut __m128i, row0);
            _mm_storeu_si128(dst_ptr.add(16) as *mut __m128i, row1);
            uv_x += 32;
        } else if chroma_subsampling == YuvChromaSubsampling::Yuv422
            || (chroma_subsampling == YuvChromaSubsampling::Yuv420 && compute_uv_row)
        {
            let r1 = _mm_avg_epu16(r_low, r_high);
            let g1 = _mm_avg_epu16(g_low, g_high);
            let b1 = _mm_avg_epu16(b_low, b_high);

            let cbk = _mm_max_epi16(
                _mm_min_epi16(
                    (_mm_add_epi16(
                        uv_bias,
                        _mm_add_epi16(
                            _mm_add_epi16(
                                _mm_mulhrs_epi16(r1, v_cb_r),
                                _mm_mulhrs_epi16(g1, v_cb_g),
                            ),
                            _mm_mulhrs_epi16(b1, v_cb_b),
                        ),
                    )),
                    i_cap_uv,
                ),
                i_bias_y,
            );

            let crk = _mm_max_epi16(
                _mm_min_epi16(
                    (_mm_add_epi16(
                        uv_bias,
                        _mm_add_epi16(
                            _mm_add_epi16(
                                _mm_mulhrs_epi16(r1, v_cr_r),
                                _mm_mulhrs_epi16(g1, v_cr_g),
                            ),
                            _mm_mulhrs_epi16(b1, v_cr_b),
                        ),
                    )),
                    i_cap_uv,
                ),
                i_bias_y,
            );

            let cb = _mm_packus_epi16(cbk, cbk);
            let cr = _mm_packus_epi16(crk, crk);

            let row0 = match order {
                YuvNVOrder::UV => _mm_unpacklo_epi8(cb, cr),
                YuvNVOrder::VU => _mm_unpacklo_epi8(cr, cb),
            };
            let dst_ptr = uv_ptr.add(uv_x);
            _mm_storeu_si128(dst_ptr as *mut __m128i, row0);
            uv_x += 16;
        }

        cx += 16;
    }

    ProcessedOffset { cx, ux: uv_x }
}
