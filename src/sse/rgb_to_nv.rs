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
use crate::sse::{_mm_affine_dot, _mm_load_deinterleave_rgb_for_yuv, sse_pairwise_avg_epi8};
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
    const PRECISION: i32,
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
        sse_rgba_to_nv_row_impl::<ORIGIN_CHANNELS, UV_ORDER, SAMPLING, PRECISION>(
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
    const PRECISION: i32,
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

    const V_SCALE: i32 = 2;
    let bias_y = range.bias_y as i16;
    let bias_uv = range.bias_uv as i16;

    let i_cap_uv = _mm_set1_epi16(range.bias_y as i16 + range.range_uv as i16);

    let zeros = _mm_setzero_si128();

    let y_bias = _mm_set1_epi16(bias_y);
    let uv_bias = _mm_set1_epi16(bias_uv);
    let y_base = _mm_set1_epi32(bias_y as i32 * (1 << PRECISION) + (1 << (PRECISION - 1)) - 1);
    let v_yr_yg = _mm_set1_epi32(transform._interleaved_yr_yg());
    let v_yb = _mm_set1_epi16(transform.yb as i16);
    let v_cb_r = _mm_set1_epi16(transform.cb_r as i16);
    let v_cb_g = _mm_set1_epi16(transform.cb_g as i16);
    let v_cb_b = _mm_set1_epi16(transform.cb_b as i16);
    let v_cr_r = _mm_set1_epi16(transform.cr_r as i16);
    let v_cr_g = _mm_set1_epi16(transform.cr_g as i16);
    let v_cr_b = _mm_set1_epi16(transform.cr_b as i16);

    while cx + 16 < width as usize {
        let px = cx * channels;
        let (r_values, g_values, b_values) =
            _mm_load_deinterleave_rgb_for_yuv::<ORIGIN_CHANNELS>(rgba_ptr.add(px));

        let r_lo16 = _mm_unpacklo_epi8(r_values, zeros);
        let r_hi16 = _mm_unpackhi_epi8(r_values, zeros);
        let g_lo16 = _mm_unpacklo_epi8(g_values, zeros);
        let g_hi16 = _mm_unpackhi_epi8(g_values, zeros);
        let b_lo16 = _mm_unpacklo_epi8(b_values, zeros);
        let b_hi16 = _mm_unpackhi_epi8(b_values, zeros);

        let y_l = _mm_affine_dot::<PRECISION>(y_base, r_lo16, g_lo16, b_lo16, v_yr_yg, v_yb);
        let y_h = _mm_affine_dot::<PRECISION>(y_base, r_hi16, g_hi16, b_hi16, v_yr_yg, v_yb);

        let y_yuv = _mm_packus_epi16(y_l, y_h);
        _mm_storeu_si128(y_ptr.add(cx) as *mut __m128i, y_yuv);

        if chroma_subsampling == YuvChromaSubsampling::Yuv444 {
            let r_low = _mm_slli_epi16::<V_SCALE>(r_lo16);
            let r_high = _mm_slli_epi16::<V_SCALE>(r_hi16);
            let g_low = _mm_slli_epi16::<V_SCALE>(g_lo16);
            let g_high = _mm_slli_epi16::<V_SCALE>(g_hi16);
            let b_low = _mm_slli_epi16::<V_SCALE>(b_lo16);
            let b_high = _mm_slli_epi16::<V_SCALE>(b_hi16);

            let cb_l = _mm_max_epi16(
                _mm_min_epi16(
                    _mm_add_epi16(
                        uv_bias,
                        _mm_add_epi16(
                            _mm_add_epi16(
                                _mm_mulhrs_epi16(r_low, v_cb_r),
                                _mm_mulhrs_epi16(g_low, v_cb_g),
                            ),
                            _mm_mulhrs_epi16(b_low, v_cb_b),
                        ),
                    ),
                    i_cap_uv,
                ),
                y_bias,
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
                y_bias,
            );
            let cb_h = _mm_max_epi16(
                _mm_min_epi16(
                    _mm_add_epi16(
                        uv_bias,
                        _mm_add_epi16(
                            _mm_add_epi16(
                                _mm_mulhrs_epi16(r_high, v_cb_r),
                                _mm_mulhrs_epi16(g_high, v_cb_g),
                            ),
                            _mm_mulhrs_epi16(b_high, v_cb_b),
                        ),
                    ),
                    i_cap_uv,
                ),
                y_bias,
            );
            let cr_h = _mm_max_epi16(
                _mm_min_epi16(
                    _mm_add_epi16(
                        uv_bias,
                        _mm_add_epi16(
                            _mm_add_epi16(
                                _mm_mulhrs_epi16(r_high, v_cr_r),
                                _mm_mulhrs_epi16(g_high, v_cr_g),
                            ),
                            _mm_mulhrs_epi16(b_high, v_cr_b),
                        ),
                    ),
                    i_cap_uv,
                ),
                y_bias,
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
            let r1 = _mm_slli_epi16::<V_SCALE>(sse_pairwise_avg_epi8(r_values));
            let g1 = _mm_slli_epi16::<V_SCALE>(sse_pairwise_avg_epi8(g_values));
            let b1 = _mm_slli_epi16::<V_SCALE>(sse_pairwise_avg_epi8(b_values));

            let cbk = _mm_max_epi16(
                _mm_min_epi16(
                    _mm_add_epi16(
                        uv_bias,
                        _mm_add_epi16(
                            _mm_add_epi16(
                                _mm_mulhrs_epi16(r1, v_cb_r),
                                _mm_mulhrs_epi16(g1, v_cb_g),
                            ),
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
                            _mm_add_epi16(
                                _mm_mulhrs_epi16(r1, v_cr_r),
                                _mm_mulhrs_epi16(g1, v_cr_g),
                            ),
                            _mm_mulhrs_epi16(b1, v_cr_b),
                        ),
                    ),
                    i_cap_uv,
                ),
                y_bias,
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
