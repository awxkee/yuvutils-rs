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

use crate::avx512bw::avx512_utils::{avx512_load_rgb_u8, avx512_pack_u16};
use crate::yuv_support::{CbCrForwardTransform, YuvChromaRange, YuvSourceChannels};
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

pub(crate) fn avx512_row_rgb_to_y<const ORIGIN_CHANNELS: u8, const HAS_VBMI: bool>(
    transform: &CbCrForwardTransform<i32>,
    range: &YuvChromaRange,
    y_plane: &mut [u8],
    rgba: &[u8],
    start_cx: usize,
    width: usize,
) -> usize {
    unsafe {
        if HAS_VBMI {
            avx512_row_rgb_to_y_bmi_impl::<ORIGIN_CHANNELS>(
                transform, range, y_plane, rgba, start_cx, width,
            )
        } else {
            avx512_row_rgb_to_y_def_impl::<ORIGIN_CHANNELS>(
                transform, range, y_plane, rgba, start_cx, width,
            )
        }
    }
}

#[target_feature(enable = "avx512bw", enable = "avx512f")]
unsafe fn avx512_row_rgb_to_y_def_impl<const ORIGIN_CHANNELS: u8>(
    transform: &CbCrForwardTransform<i32>,
    range: &YuvChromaRange,
    y_plane: &mut [u8],
    rgba: &[u8],
    start_cx: usize,
    width: usize,
) -> usize {
    avx512_row_rgb_to_y_impl::<ORIGIN_CHANNELS, false>(
        transform, range, y_plane, rgba, start_cx, width,
    )
}

#[target_feature(enable = "avx512bw", enable = "avx512f", enable = "avx512vbmi")]
unsafe fn avx512_row_rgb_to_y_bmi_impl<const ORIGIN_CHANNELS: u8>(
    transform: &CbCrForwardTransform<i32>,
    range: &YuvChromaRange,
    y_plane: &mut [u8],
    rgba: &[u8],
    start_cx: usize,
    width: usize,
) -> usize {
    avx512_row_rgb_to_y_impl::<ORIGIN_CHANNELS, true>(
        transform, range, y_plane, rgba, start_cx, width,
    )
}

#[inline(always)]
unsafe fn avx512_row_rgb_to_y_impl<const ORIGIN_CHANNELS: u8, const HAS_VBMI: bool>(
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

    const V_SCALE: u32 = 2;
    let bias_y = range.bias_y as i16;

    let y_bias = _mm512_set1_epi16(bias_y);
    let v_yr = _mm512_set1_epi16(transform.yr as i16);
    let v_yg = _mm512_set1_epi16(transform.yg as i16);
    let v_yb = _mm512_set1_epi16(transform.yb as i16);
    let i_cap_y = _mm512_set1_epi16(range.range_y as i16 + range.bias_y as i16);

    while cx + 64 < width {
        let px = cx * channels;

        let (r_values, g_values, b_values) =
            avx512_load_rgb_u8::<ORIGIN_CHANNELS, HAS_VBMI>(rgba_ptr.add(px));

        let r_low =
            _mm512_slli_epi16::<V_SCALE>(_mm512_cvtepu8_epi16(_mm512_castsi512_si256(r_values)));
        let r_high = _mm512_slli_epi16::<V_SCALE>(_mm512_cvtepu8_epi16(
            _mm512_extracti64x4_epi64::<1>(r_values),
        ));
        let g_low =
            _mm512_slli_epi16::<V_SCALE>(_mm512_cvtepu8_epi16(_mm512_castsi512_si256(g_values)));
        let g_high = _mm512_slli_epi16::<V_SCALE>(_mm512_cvtepu8_epi16(
            _mm512_extracti64x4_epi64::<1>(g_values),
        ));
        let b_low =
            _mm512_slli_epi16::<V_SCALE>(_mm512_cvtepu8_epi16(_mm512_castsi512_si256(b_values)));
        let b_high = _mm512_slli_epi16::<V_SCALE>(_mm512_cvtepu8_epi16(
            _mm512_extracti64x4_epi64::<1>(b_values),
        ));

        let y_l = _mm512_min_epi16(
            _mm512_add_epi16(
                y_bias,
                _mm512_add_epi16(
                    _mm512_add_epi16(
                        _mm512_mulhrs_epi16(r_low, v_yr),
                        _mm512_mulhrs_epi16(g_low, v_yg),
                    ),
                    _mm512_mulhrs_epi16(b_low, v_yb),
                ),
            ),
            i_cap_y,
        );

        let y_h = _mm512_min_epi16(
            _mm512_add_epi16(
                y_bias,
                _mm512_add_epi16(
                    _mm512_add_epi16(
                        _mm512_mulhrs_epi16(r_high, v_yr),
                        _mm512_mulhrs_epi16(g_high, v_yg),
                    ),
                    _mm512_mulhrs_epi16(b_high, v_yb),
                ),
            ),
            i_cap_y,
        );

        let y_yuv = avx512_pack_u16(y_l, y_h);

        _mm512_storeu_si512(
            y_ptr.get_unchecked_mut(cx..).as_mut_ptr() as *mut i32,
            y_yuv,
        );

        cx += 64;
    }

    cx
}
