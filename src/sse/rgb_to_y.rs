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

use crate::sse::sse_support::{sse_deinterleave_rgb, sse_deinterleave_rgba};
use crate::yuv_support::{CbCrForwardTransform, YuvChromaRange, YuvSourceChannels};
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

pub fn sse_rgb_to_y<const ORIGIN_CHANNELS: u8>(
    transform: &CbCrForwardTransform<i32>,
    range: &YuvChromaRange,
    y_plane: &mut [u8],
    rgba: &[u8],
    start_cx: usize,
    width: usize,
) -> usize {
    unsafe {
        sse_rgb_to_y_impl::<ORIGIN_CHANNELS>(transform, range, y_plane, rgba, start_cx, width)
    }
}

#[target_feature(enable = "sse4.1")]
unsafe fn sse_rgb_to_y_impl<const ORIGIN_CHANNELS: u8>(
    transform: &CbCrForwardTransform<i32>,
    range: &YuvChromaRange,
    y_plane: &mut [u8],
    rgba: &[u8],
    start_cx: usize,
    width: usize,
) -> usize {
    let source_channels: YuvSourceChannels = ORIGIN_CHANNELS.into();
    let channels = source_channels.get_channels_count();

    let y_ptr = y_plane;
    let rgba_ptr = rgba.as_ptr();

    let mut cx = start_cx;

    const V_SHR: i32 = 3;
    const V_SCALE: i32 = 7;
    let rounding_const_bias: i16 = 1 << (V_SHR - 1);
    let bias_y = range.bias_y as i16 * (1 << V_SHR) + rounding_const_bias;

    let zeros = _mm_setzero_si128();

    let i_bias_y = _mm_set1_epi16(range.bias_y as i16);
    let i_cap_y = _mm_set1_epi16(range.range_y as i16 + range.bias_y as i16);
    let y_bias = _mm_set1_epi16(bias_y);
    let v_yr = _mm_set1_epi16(transform.yr as i16);
    let v_yg = _mm_set1_epi16(transform.yg as i16);
    let v_yb = _mm_set1_epi16(transform.yb as i16);

    while cx + 16 < width {
        let (r_values, g_values, b_values);

        let px = cx * channels;

        match source_channels {
            YuvSourceChannels::Rgb | YuvSourceChannels::Bgr => {
                let source_ptr = rgba_ptr.add(px);
                let row_1 = _mm_loadu_si128(source_ptr as *const __m128i);
                let row_2 = _mm_loadu_si128(source_ptr.add(16) as *const __m128i);
                let row_3 = _mm_loadu_si128(source_ptr.add(32) as *const __m128i);

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
                let source_ptr = rgba_ptr.add(px);
                let row_1 = _mm_loadu_si128(source_ptr as *const __m128i);
                let row_2 = _mm_loadu_si128(source_ptr.add(16) as *const __m128i);
                let row_3 = _mm_loadu_si128(source_ptr.add(32) as *const __m128i);
                let row_4 = _mm_loadu_si128(source_ptr.add(48) as *const __m128i);

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
                _mm_srai_epi16::<V_SHR>(_mm_add_epi16(
                    y_bias,
                    _mm_add_epi16(
                        _mm_add_epi16(_mm_mulhi_epi16(r_low, v_yr), _mm_mulhi_epi16(g_low, v_yg)),
                        _mm_mulhi_epi16(b_low, v_yb),
                    ),
                )),
                i_cap_y,
            ),
            i_bias_y,
        );

        let y_h = _mm_max_epi16(
            _mm_min_epi16(
                _mm_srai_epi16::<V_SHR>(_mm_add_epi16(
                    y_bias,
                    _mm_add_epi16(
                        _mm_add_epi16(_mm_mulhi_epi16(r_high, v_yr), _mm_mulhi_epi16(g_high, v_yg)),
                        _mm_mulhi_epi16(b_high, v_yb),
                    ),
                )),
                i_cap_y,
            ),
            i_bias_y,
        );

        let y_yuv = _mm_packus_epi16(y_l, y_h);

        _mm_storeu_si128(
            y_ptr.get_unchecked_mut(cx..).as_mut_ptr() as *mut __m128i,
            y_yuv,
        );

        cx += 16;
    }

    cx
}
