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

use crate::avx512bw::avx512_utils::{avx512_pack_u16, avx512_rgb_u8, avx512_rgba_u8};
use crate::yuv_support::{CbCrInverseTransform, YuvChromaRange, YuvSourceChannels};
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[target_feature(enable = "avx512bw")]
pub unsafe fn avx512_y_to_rgb_row<const DESTINATION_CHANNELS: u8>(
    range: &YuvChromaRange,
    transform: &CbCrInverseTransform<i32>,
    y_plane: &[u8],
    rgba: &mut [u8],
    start_cx: usize,
    y_offset: usize,
    rgba_offset: usize,
    width: usize,
) -> usize {
    let destination_channels: YuvSourceChannels = DESTINATION_CHANNELS.into();
    let channels = destination_channels.get_channels_count();

    let mut cx = start_cx;
    let y_ptr = y_plane.as_ptr();
    let rgba_ptr = rgba.as_mut_ptr();

    let y_corr = _mm512_set1_epi8(range.bias_y as i8);
    let v_luma_coeff = _mm512_set1_epi16(transform.y_coef as i16);
    let v_min_values = _mm512_setzero_si512();
    let v_alpha = _mm512_set1_epi8(255u8 as i8);
    let rounding_const = _mm512_set1_epi16(1 << 5);

    while cx + 64 < width {
        let y_values = _mm512_subs_epi8(
            _mm512_slli_epi16::<7>(_mm512_loadu_si512(y_ptr.add(y_offset + cx) as *const i32)),
            y_corr,
        );

        let y_high = _mm512_mulhi_epi16(
            _mm512_cvtepu8_epi16(_mm512_extracti64x4_epi64::<1>(y_values)),
            v_luma_coeff,
        );

        let r_high = _mm512_srli_epi16::<3>(_mm512_max_epi16(y_high, v_min_values));

        let y_low = _mm512_mulhi_epi16(
            _mm512_slli_epi16::<7>(_mm512_cvtepu8_epi16(_mm512_castsi512_si256(y_values))),
            v_luma_coeff,
        );

        let r_low = _mm512_srli_epi16::<3>(_mm512_adds_epi16(
            _mm512_max_epi16(y_low, v_min_values),
            rounding_const,
        ));

        let r_values = avx512_pack_u16(r_low, r_high);

        let dst_shift = rgba_offset + cx * channels;

        match destination_channels {
            YuvSourceChannels::Rgb | YuvSourceChannels::Bgr => {
                let ptr = rgba_ptr.add(dst_shift);
                avx512_rgb_u8(ptr, r_values, r_values, r_values);
            }
            YuvSourceChannels::Rgba => {
                avx512_rgba_u8(
                    rgba_ptr.add(dst_shift),
                    r_values,
                    r_values,
                    r_values,
                    v_alpha,
                );
            }
            YuvSourceChannels::Bgra => {
                avx512_rgba_u8(
                    rgba_ptr.add(dst_shift),
                    r_values,
                    r_values,
                    r_values,
                    v_alpha,
                );
            }
        }

        cx += 64;
    }

    cx
}
