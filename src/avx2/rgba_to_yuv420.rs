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

use crate::avx2::avx2_utils::{
    _mm256_affine_dot, _mm256_load_deinterleave_rgb_for_yuv, avx2_pack_u16,
    avx_pairwise_avg_epi16_epi8,
};
use crate::internals::ProcessedOffset;
use crate::yuv_support::{CbCrForwardTransform, YuvChromaRange, YuvSourceChannels};
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

pub(crate) fn avx2_rgba_to_yuv420<const ORIGIN_CHANNELS: u8, const PRECISION: i32>(
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
        avx2_rgba_to_yuv_impl420::<ORIGIN_CHANNELS, PRECISION>(
            transform, range, y_plane0, y_plane1, u_plane, v_plane, rgba0, rgba1, start_cx,
            start_ux, width,
        )
    }
}

#[target_feature(enable = "avx2")]
unsafe fn avx2_rgba_to_yuv_impl420<const ORIGIN_CHANNELS: u8, const PRECISION: i32>(
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

    let i_cap_uv = _mm256_set1_epi16(range.bias_y as i16 + range.range_uv as i16);

    let y_bias = _mm256_set1_epi16(bias_y);
    let uv_bias = _mm256_set1_epi16(bias_uv);
    let y_base = _mm256_set1_epi32(bias_y as i32 * (1 << PRECISION) + (1 << (PRECISION - 1)) - 1);

    let v_yr_yg = _mm256_set1_epi32(transform._interleaved_yr_yg());
    let v_yb = _mm256_set1_epi16(transform.yb as i16);
    let v_cb_r = _mm256_set1_epi16(transform.cb_r as i16);
    let v_cb_g = _mm256_set1_epi16(transform.cb_g as i16);
    let v_cb_b = _mm256_set1_epi16(transform.cb_b as i16);
    let v_cr_r = _mm256_set1_epi16(transform.cr_r as i16);
    let v_cr_g = _mm256_set1_epi16(transform.cr_g as i16);
    let v_cr_b = _mm256_set1_epi16(transform.cr_b as i16);

    while cx + 32 < width {
        let px = cx * channels;
        let (r_values0, g_values0, b_values0) = _mm256_load_deinterleave_rgb_for_yuv::<
            ORIGIN_CHANNELS,
        >(rgba0.get_unchecked(px..).as_ptr());
        let (r_values1, g_values1, b_values1) = _mm256_load_deinterleave_rgb_for_yuv::<
            ORIGIN_CHANNELS,
        >(rgba1.get_unchecked(px..).as_ptr());

        let r0_low = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(r_values0));
        let r0_high = _mm256_cvtepu8_epi16(_mm256_extracti128_si256::<1>(r_values0));
        let g0_low = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(g_values0));
        let g0_high = _mm256_cvtepu8_epi16(_mm256_extracti128_si256::<1>(g_values0));
        let b0_low = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(b_values0));
        let b0_high = _mm256_cvtepu8_epi16(_mm256_extracti128_si256::<1>(b_values0));

        let y0_l = _mm256_affine_dot::<PRECISION>(y_base, r0_low, g0_low, b0_low, v_yr_yg, v_yb);

        let y0_h = _mm256_affine_dot::<PRECISION>(y_base, r0_high, g0_high, b0_high, v_yr_yg, v_yb);

        let r1_low = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(r_values1));
        let r1_high = _mm256_cvtepu8_epi16(_mm256_extracti128_si256::<1>(r_values1));
        let g1_low = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(g_values1));
        let g1_high = _mm256_cvtepu8_epi16(_mm256_extracti128_si256::<1>(g_values1));
        let b1_low = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(b_values1));
        let b1_high = _mm256_cvtepu8_epi16(_mm256_extracti128_si256::<1>(b_values1));

        let y1_l = _mm256_affine_dot::<PRECISION>(y_base, r1_low, g1_low, b1_low, v_yr_yg, v_yb);

        let y1_h = _mm256_affine_dot::<PRECISION>(y_base, r1_high, g1_high, b1_high, v_yr_yg, v_yb);

        let y0_yuv = avx2_pack_u16(y0_l, y0_h);
        let y1_yuv = avx2_pack_u16(y1_l, y1_h);

        _mm256_storeu_si256(
            y_plane0.get_unchecked_mut(cx..).as_mut_ptr() as *mut __m256i,
            y0_yuv,
        );
        _mm256_storeu_si256(
            y_plane1.get_unchecked_mut(cx..).as_mut_ptr() as *mut __m256i,
            y1_yuv,
        );

        let r_uv = _mm256_slli_epi16::<V_SCALE>(avx_pairwise_avg_epi16_epi8(_mm256_avg_epu8(
            r_values0, r_values1,
        )));
        let g_uv = _mm256_slli_epi16::<V_SCALE>(avx_pairwise_avg_epi16_epi8(_mm256_avg_epu8(
            g_values0, g_values1,
        )));
        let b_uv = _mm256_slli_epi16::<V_SCALE>(avx_pairwise_avg_epi16_epi8(_mm256_avg_epu8(
            b_values0, b_values1,
        )));

        let cb = _mm256_max_epi16(
            _mm256_min_epi16(
                _mm256_add_epi16(
                    uv_bias,
                    _mm256_add_epi16(
                        _mm256_add_epi16(
                            _mm256_mulhrs_epi16(r_uv, v_cb_r),
                            _mm256_mulhrs_epi16(g_uv, v_cb_g),
                        ),
                        _mm256_mulhrs_epi16(b_uv, v_cb_b),
                    ),
                ),
                i_cap_uv,
            ),
            y_bias,
        );

        let cr = _mm256_max_epi16(
            _mm256_min_epi16(
                _mm256_add_epi16(
                    uv_bias,
                    _mm256_add_epi16(
                        _mm256_add_epi16(
                            _mm256_mulhrs_epi16(r_uv, v_cr_r),
                            _mm256_mulhrs_epi16(g_uv, v_cr_g),
                        ),
                        _mm256_mulhrs_epi16(b_uv, v_cr_b),
                    ),
                ),
                i_cap_uv,
            ),
            y_bias,
        );

        let cb = avx2_pack_u16(cb, cb);
        let cr = avx2_pack_u16(cr, cr);

        _mm_storeu_si128(
            u_ptr.add(uv_x) as *mut _ as *mut __m128i,
            _mm256_castsi256_si128(cb),
        );
        _mm_storeu_si128(
            v_ptr.add(uv_x) as *mut _ as *mut __m128i,
            _mm256_castsi256_si128(cr),
        );

        uv_x += 16;
        cx += 32;
    }

    ProcessedOffset { cx, ux: uv_x }
}
