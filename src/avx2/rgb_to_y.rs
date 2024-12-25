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
    _mm256_load_deinterleave_half_rgb_for_yuv, _mm256_load_deinterleave_rgb_for_yuv, avx2_pack_u16,
};
use crate::yuv_support::{CbCrForwardTransform, YuvChromaRange, YuvSourceChannels};
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

pub(crate) fn avx2_rgb_to_y_row<const ORIGIN_CHANNELS: u8, const PRECISION: i32>(
    transform: &CbCrForwardTransform<i32>,
    range: &YuvChromaRange,
    y_plane: &mut [u8],
    rgba: &[u8],
    start_cx: usize,
    width: usize,
) -> usize {
    unsafe {
        avx2_rgb_to_y_row_impl::<ORIGIN_CHANNELS, PRECISION>(
            transform, range, y_plane, rgba, start_cx, width,
        )
    }
}

#[target_feature(enable = "avx2")]
pub(crate) unsafe fn avx2_rgb_to_y_row_impl<const ORIGIN_CHANNELS: u8, const PRECISION: i32>(
    transform: &CbCrForwardTransform<i32>,
    range: &YuvChromaRange,
    y_plane: &mut [u8],
    rgba: &[u8],
    start_cx: usize,
    width: usize,
) -> usize {
    let source_channels: YuvSourceChannels = ORIGIN_CHANNELS.into();
    let channels = source_channels.get_channels_count();

    let y_ptr = y_plane.as_mut_ptr();
    let rgba_ptr = rgba.as_ptr();

    let mut cx = start_cx;

    let bias_y = range.bias_y as i16;

    const V_SCALE: i32 = 2;

    let i_cap_y = _mm256_set1_epi16(range.bias_y as i16 + range.range_y as i16);

    let y_bias = _mm256_set1_epi16(bias_y);
    let v_yr = _mm256_set1_epi16(transform.yr as i16);
    let v_yg = _mm256_set1_epi16(transform.yg as i16);
    let v_yb = _mm256_set1_epi16(transform.yb as i16);

    while cx + 32 < width {
        let px = cx * channels;
        let (r_values, g_values, b_values) =
            _mm256_load_deinterleave_rgb_for_yuv::<ORIGIN_CHANNELS>(rgba_ptr.add(px));

        let r_low =
            _mm256_slli_epi16::<V_SCALE>(_mm256_cvtepu8_epi16(_mm256_castsi256_si128(r_values)));
        let r_high = _mm256_slli_epi16::<V_SCALE>(_mm256_cvtepu8_epi16(
            _mm256_extracti128_si256::<1>(r_values),
        ));
        let g_low =
            _mm256_slli_epi16::<V_SCALE>(_mm256_cvtepu8_epi16(_mm256_castsi256_si128(g_values)));
        let g_high = _mm256_slli_epi16::<V_SCALE>(_mm256_cvtepu8_epi16(
            _mm256_extracti128_si256::<1>(g_values),
        ));
        let b_low =
            _mm256_slli_epi16::<V_SCALE>(_mm256_cvtepu8_epi16(_mm256_castsi256_si128(b_values)));
        let b_high = _mm256_slli_epi16::<V_SCALE>(_mm256_cvtepu8_epi16(
            _mm256_extracti128_si256::<1>(b_values),
        ));

        let y_l = _mm256_min_epi16(
            _mm256_add_epi16(
                y_bias,
                _mm256_add_epi16(
                    _mm256_add_epi16(
                        _mm256_mulhrs_epi16(r_low, v_yr),
                        _mm256_mulhrs_epi16(g_low, v_yg),
                    ),
                    _mm256_mulhrs_epi16(b_low, v_yb),
                ),
            ),
            i_cap_y,
        );

        let y_h = _mm256_min_epi16(
            _mm256_add_epi16(
                y_bias,
                _mm256_add_epi16(
                    _mm256_add_epi16(
                        _mm256_mulhrs_epi16(r_high, v_yr),
                        _mm256_mulhrs_epi16(g_high, v_yg),
                    ),
                    _mm256_mulhrs_epi16(b_high, v_yb),
                ),
            ),
            i_cap_y,
        );

        let y_yuv = avx2_pack_u16(y_l, y_h);
        _mm256_storeu_si256(y_ptr.add(cx) as *mut __m256i, y_yuv);

        cx += 32;
    }

    while cx + 16 < width {
        let px = cx * channels;
        let (r_values, g_values, b_values) =
            _mm256_load_deinterleave_half_rgb_for_yuv::<ORIGIN_CHANNELS>(rgba_ptr.add(px));

        let r_low =
            _mm256_slli_epi16::<V_SCALE>(_mm256_cvtepu8_epi16(_mm256_castsi256_si128(r_values)));
        let g_low =
            _mm256_slli_epi16::<V_SCALE>(_mm256_cvtepu8_epi16(_mm256_castsi256_si128(g_values)));
        let b_low =
            _mm256_slli_epi16::<V_SCALE>(_mm256_cvtepu8_epi16(_mm256_castsi256_si128(b_values)));

        let y_l = _mm256_min_epi16(
            _mm256_add_epi16(
                y_bias,
                _mm256_add_epi16(
                    _mm256_add_epi16(
                        _mm256_mulhrs_epi16(r_low, v_yr),
                        _mm256_mulhrs_epi16(g_low, v_yg),
                    ),
                    _mm256_mulhrs_epi16(b_low, v_yb),
                ),
            ),
            i_cap_y,
        );

        let y_yuv = avx2_pack_u16(y_l, _mm256_setzero_si256());
        _mm_storeu_si128(y_ptr.add(cx) as *mut __m128i, _mm256_castsi256_si128(y_yuv));

        cx += 16;
    }

    cx
}
