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

use crate::avx2::avx2_utils::*;
use crate::yuv_support::{CbCrInverseTransform, YuvChromaRange, YuvSourceChannels};
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

pub(crate) fn avx2_y_to_rgba_row<const DESTINATION_CHANNELS: u8>(
    range: &YuvChromaRange,
    transform: &CbCrInverseTransform<i32>,
    y_plane: &[u8],
    rgba: &mut [u8],
    start_cx: usize,
    width: usize,
) -> usize {
    unsafe {
        avx2_y_to_rgba_row_impl::<DESTINATION_CHANNELS>(
            range, transform, y_plane, rgba, start_cx, width,
        )
    }
}

#[target_feature(enable = "avx2")]
unsafe fn avx2_y_to_rgba_row_impl<const DESTINATION_CHANNELS: u8>(
    range: &YuvChromaRange,
    transform: &CbCrInverseTransform<i32>,
    y_plane: &[u8],
    rgba: &mut [u8],
    start_cx: usize,
    width: usize,
) -> usize {
    let destination_channels: YuvSourceChannels = DESTINATION_CHANNELS.into();
    let channels = destination_channels.get_channels_count();

    let mut cx = start_cx;
    let y_ptr = y_plane.as_ptr();
    let rgba_ptr = rgba.as_mut_ptr();

    let y_corr = _mm256_set1_epi8(range.bias_y as i8);
    let v_luma_coeff = _mm256_set1_epi16(transform.y_coef as i16);

    const SCALE: i32 = 2;

    while cx + 64 < width {
        let y_values0 =
            _mm256_subs_epu8(_mm256_loadu_si256(y_ptr.add(cx) as *const __m256i), y_corr);
        let y_values1 = _mm256_subs_epu8(
            _mm256_loadu_si256(y_ptr.add(cx + 32) as *const __m256i),
            y_corr,
        );

        let y_high0 = _mm256_mulhrs_epi16(
            _mm256_slli_epi16::<SCALE>(_mm256_cvtepu8_epi16(_mm256_extracti128_si256::<1>(
                y_values0,
            ))),
            v_luma_coeff,
        );

        let y_high1 = _mm256_mulhrs_epi16(
            _mm256_slli_epi16::<SCALE>(_mm256_cvtepu8_epi16(_mm256_extracti128_si256::<1>(
                y_values1,
            ))),
            v_luma_coeff,
        );

        let y_low0 = _mm256_mulhrs_epi16(
            _mm256_slli_epi16::<SCALE>(_mm256_cvtepu8_epi16(_mm256_castsi256_si128(y_values0))),
            v_luma_coeff,
        );

        let y_low1 = _mm256_mulhrs_epi16(
            _mm256_slli_epi16::<SCALE>(_mm256_cvtepu8_epi16(_mm256_castsi256_si128(y_values1))),
            v_luma_coeff,
        );

        let v_values0 = avx2_pack_u16(y_low0, y_high0);
        let v_values1 = avx2_pack_u16(y_low1, y_high1);

        let dst_shift = cx * channels;

        let v_alpha = _mm256_set1_epi8(255u8 as i8);
        _mm256_store_interleave_rgb_for_yuv::<DESTINATION_CHANNELS>(
            rgba_ptr.add(dst_shift),
            v_values0,
            v_values0,
            v_values0,
            v_alpha,
        );

        _mm256_store_interleave_rgb_for_yuv::<DESTINATION_CHANNELS>(
            rgba_ptr.add(dst_shift + channels * 32),
            v_values1,
            v_values1,
            v_values1,
            v_alpha,
        );

        cx += 64;
    }

    while cx + 32 < width {
        let y_values =
            _mm256_subs_epu8(_mm256_loadu_si256(y_ptr.add(cx) as *const __m256i), y_corr);

        let y_high = _mm256_mulhrs_epi16(
            _mm256_slli_epi16::<SCALE>(_mm256_cvtepu8_epi16(_mm256_extracti128_si256::<1>(
                y_values,
            ))),
            v_luma_coeff,
        );

        let y_low = _mm256_mulhrs_epi16(
            _mm256_slli_epi16::<SCALE>(_mm256_cvtepu8_epi16(_mm256_castsi256_si128(y_values))),
            v_luma_coeff,
        );

        let v_values = avx2_pack_u16(y_low, y_high);

        let dst_shift = cx * channels;

        let v_alpha = _mm256_set1_epi8(255u8 as i8);
        _mm256_store_interleave_rgb_for_yuv::<DESTINATION_CHANNELS>(
            rgba_ptr.add(dst_shift),
            v_values,
            v_values,
            v_values,
            v_alpha,
        );

        cx += 32;
    }

    while cx + 16 < width {
        let y_values = _mm256_subs_epu8(
            _mm256_castsi128_si256(_mm_loadu_si128(y_ptr.add(cx) as *const __m128i)),
            y_corr,
        );

        let y_low = _mm256_mulhrs_epi16(
            _mm256_slli_epi16::<SCALE>(_mm256_cvtepu8_epi16(_mm256_castsi256_si128(y_values))),
            v_luma_coeff,
        );

        let v_values = avx2_pack_u16(y_low, _mm256_setzero_si256());

        let dst_shift = cx * channels;

        let v_alpha = _mm256_set1_epi8(255u8 as i8);

        _mm256_store_interleave_rgb_half_for_yuv::<DESTINATION_CHANNELS>(
            rgba_ptr.add(dst_shift),
            v_values,
            v_values,
            v_values,
            v_alpha,
        );

        cx += 16;
    }

    cx
}