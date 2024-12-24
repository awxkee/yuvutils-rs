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
use crate::internals::{interleaved_epi8, ProcessedOffset};
use crate::yuv_support::{
    CbCrInverseTransform, YuvChromaRange, YuvChromaSubsampling, YuvSourceChannels,
};
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

pub(crate) fn avx2_yuv_to_rgba_fast_row<const DESTINATION_CHANNELS: u8, const SAMPLING: u8>(
    range: &YuvChromaRange,
    transform: &CbCrInverseTransform<i32>,
    y_plane: &[u8],
    u_plane: &[u8],
    v_plane: &[u8],
    rgba: &mut [u8],
    start_cx: usize,
    start_ux: usize,
    width: usize,
) -> ProcessedOffset {
    unsafe {
        avx2_yuv_to_rgba_row_fast_impl::<DESTINATION_CHANNELS, SAMPLING>(
            range, transform, y_plane, u_plane, v_plane, rgba, start_cx, start_ux, width,
        )
    }
}

#[target_feature(enable = "avx2")]
unsafe fn avx2_yuv_to_rgba_row_fast_impl<const DESTINATION_CHANNELS: u8, const SAMPLING: u8>(
    range: &YuvChromaRange,
    transform: &CbCrInverseTransform<i32>,
    y_plane: &[u8],
    u_plane: &[u8],
    v_plane: &[u8],
    rgba: &mut [u8],
    start_cx: usize,
    start_ux: usize,
    width: usize,
) -> ProcessedOffset {
    let chroma_subsampling: YuvChromaSubsampling = SAMPLING.into();
    let destination_channels: YuvSourceChannels = DESTINATION_CHANNELS.into();
    let channels = destination_channels.get_channels_count();

    const PRECISION: i32 = 6;

    let mut cx = start_cx;
    let mut uv_x = start_ux;
    let y_ptr = y_plane.as_ptr();
    let u_ptr = u_plane.as_ptr();
    let v_ptr = v_plane.as_ptr();
    let rgba_ptr = rgba.as_mut_ptr();

    let y_corr = _mm256_set1_epi8(range.bias_y as i8);
    let v_luma_coeff = _mm256_set1_epi16((transform.y_coef as u16 * 256) as i16);
    let v_cr_coeff = _mm256_set1_epi16(interleaved_epi8(
        transform.cr_coef as i8,
        -transform.cr_coef as i8,
    ));
    let v_cb_coeff = _mm256_set1_epi16(interleaved_epi8(
        transform.cb_coef as i8,
        -transform.cb_coef as i8,
    ));
    let v_g_coeff_1 = _mm256_set1_epi16(interleaved_epi8(
        transform.g_coeff_1 as i8,
        -transform.g_coeff_1 as i8,
    ));
    let v_g_coeff_2 = _mm256_set1_epi16(interleaved_epi8(
        transform.g_coeff_2 as i8,
        -transform.g_coeff_2 as i8,
    ));
    let u_bias_uv = _mm256_set1_epi8(range.bias_uv as i8);

    while cx + 32 < width {
        let y_values =
            _mm256_subs_epu8(_mm256_loadu_si256(y_ptr.add(cx) as *const __m256i), y_corr);

        let (u_high_u16, v_high_u16, u_low_u16, v_low_u16);

        match chroma_subsampling {
            YuvChromaSubsampling::Yuv420 | YuvChromaSubsampling::Yuv422 => {
                let u_values = _mm_loadu_si128(u_ptr.add(uv_x) as *const __m128i);
                let v_values = _mm_loadu_si128(v_ptr.add(uv_x) as *const __m128i);

                (u_low_u16, u_high_u16) = _mm256_interleave_epi8(
                    avx2_create(
                        _mm_unpacklo_epi8(u_values, u_values),
                        _mm_unpackhi_epi8(u_values, u_values),
                    ),
                    u_bias_uv,
                );

                (v_low_u16, v_high_u16) = _mm256_interleave_epi8(
                    avx2_create(
                        _mm_unpacklo_epi8(v_values, v_values),
                        _mm_unpackhi_epi8(v_values, v_values),
                    ),
                    u_bias_uv,
                );
            }
            YuvChromaSubsampling::Yuv444 => {
                let u_values = _mm256_loadu_si256(u_ptr.add(uv_x) as *const __m256i);
                let v_values = _mm256_loadu_si256(v_ptr.add(uv_x) as *const __m256i);

                (u_low_u16, u_high_u16) = _mm256_interleave_epi8(u_values, u_bias_uv);
                (v_low_u16, v_high_u16) = _mm256_interleave_epi8(v_values, u_bias_uv);
            }
        }

        let interleaved_y = _mm256_interleave_epi8(y_values, y_values);

        let y_high = _mm256_mulhi_epu16(interleaved_y.1, v_luma_coeff);

        let r_high = _mm256_adds_epi16(y_high, _mm256_maddubs_epi16(v_high_u16, v_cr_coeff));
        let b_high = _mm256_adds_epi16(y_high, _mm256_maddubs_epi16(u_high_u16, v_cb_coeff));
        let g_high = _mm256_subs_epi16(
            y_high,
            _mm256_adds_epi16(
                _mm256_maddubs_epi16(v_high_u16, v_g_coeff_1),
                _mm256_maddubs_epi16(u_high_u16, v_g_coeff_2),
            ),
        );

        let y_low = _mm256_mulhi_epu16(interleaved_y.0, v_luma_coeff);

        let r_low = _mm256_adds_epi16(y_low, _mm256_maddubs_epi16(v_low_u16, v_cr_coeff));
        let b_low = _mm256_adds_epi16(y_low, _mm256_maddubs_epi16(u_low_u16, v_cb_coeff));
        let g_low = _mm256_subs_epi16(
            y_low,
            _mm256_adds_epi16(
                _mm256_maddubs_epi16(v_low_u16, v_g_coeff_1),
                _mm256_maddubs_epi16(u_low_u16, v_g_coeff_2),
            ),
        );

        let r_values = avx2_pack_u16(
            _mm256_srai_epi16::<PRECISION>(r_low),
            _mm256_srai_epi16::<PRECISION>(r_high),
        );
        let g_values = avx2_pack_u16(
            _mm256_srai_epi16::<PRECISION>(g_low),
            _mm256_srai_epi16::<PRECISION>(g_high),
        );
        let b_values = avx2_pack_u16(
            _mm256_srai_epi16::<PRECISION>(b_low),
            _mm256_srai_epi16::<PRECISION>(b_high),
        );

        let dst_shift = cx * channels;

        let v_alpha = _mm256_set1_epi8(255u8 as i8);
        _mm256_store_interleave_rgb_for_yuv::<DESTINATION_CHANNELS>(
            rgba_ptr.add(dst_shift),
            r_values,
            g_values,
            b_values,
            v_alpha,
        );

        cx += 32;

        match chroma_subsampling {
            YuvChromaSubsampling::Yuv420 | YuvChromaSubsampling::Yuv422 => {
                uv_x += 16;
            }
            YuvChromaSubsampling::Yuv444 => {
                uv_x += 32;
            }
        }
    }

    ProcessedOffset { cx, ux: uv_x }
}
