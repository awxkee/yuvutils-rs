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

use crate::avx512bw::avx512_utils::{avx512_pack_u16, avx512_rgb_u8, avx512_rgba_u8, shuffle};
use crate::internals::ProcessedOffset;
use crate::yuv_support::{
    CbCrInverseTransform, YuvChromaRange, YuvChromaSubsampling, YuvSourceChannels,
};
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

pub(crate) fn avx512_yuv_to_rgba<const DESTINATION_CHANNELS: u8, const SAMPLING: u8>(
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
        avx512_yuv_to_rgba_impl::<DESTINATION_CHANNELS, SAMPLING>(
            range, transform, y_plane, u_plane, v_plane, rgba, start_cx, start_ux, width,
        )
    }
}

#[target_feature(enable = "avx512bw")]
unsafe fn avx512_yuv_to_rgba_impl<const DESTINATION_CHANNELS: u8, const SAMPLING: u8>(
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

    let mut cx = start_cx;
    let mut uv_x = start_ux;
    let y_ptr = y_plane.as_ptr();
    let u_ptr = u_plane.as_ptr();
    let v_ptr = v_plane.as_ptr();
    let rgba_ptr = rgba.as_mut_ptr();

    let y_corr = _mm512_set1_epi8(range.bias_y as i8);
    let uv_corr = _mm512_set1_epi16(range.bias_uv as i16);
    let v_luma_coeff = _mm512_set1_epi16(transform.y_coef as i16);
    let v_cr_coeff = _mm512_set1_epi16(transform.cr_coef as i16);
    let v_cb_coeff = _mm512_set1_epi16(transform.cb_coef as i16);
    let v_g_coeff_1 = _mm512_set1_epi16(transform.g_coeff_1 as i16);
    let v_g_coeff_2 = _mm512_set1_epi16(transform.g_coeff_2 as i16);
    let v_alpha = _mm512_set1_epi8(255u8 as i8);

    const SCALE: u32 = 2;

    while cx + 64 < width {
        let y_values = _mm512_subs_epu8(_mm512_loadu_si512(y_ptr.add(cx) as *const i32), y_corr);

        let (u_high_u8, v_high_u8, u_low_u8, v_low_u8);

        match chroma_subsampling {
            YuvChromaSubsampling::Yuv420 | YuvChromaSubsampling::Yuv422 => {
                let u_values = _mm256_loadu_si256(u_ptr.add(uv_x) as *const __m256i);
                let v_values = _mm256_loadu_si256(v_ptr.add(uv_x) as *const __m256i);

                const MASK: i32 = shuffle(3, 1, 2, 0);
                u_high_u8 =
                    _mm256_permute4x64_epi64::<MASK>(_mm256_unpackhi_epi8(u_values, u_values));
                v_high_u8 =
                    _mm256_permute4x64_epi64::<MASK>(_mm256_unpackhi_epi8(v_values, v_values));
                u_low_u8 =
                    _mm256_permute4x64_epi64::<MASK>(_mm256_unpacklo_epi8(u_values, u_values));
                v_low_u8 =
                    _mm256_permute4x64_epi64::<MASK>(_mm256_unpacklo_epi8(v_values, v_values));
            }
            YuvChromaSubsampling::Yuv444 => {
                let u_values = _mm512_loadu_si512(u_ptr.add(uv_x) as *const i32);
                let v_values = _mm512_loadu_si512(v_ptr.add(uv_x) as *const i32);

                u_high_u8 = _mm512_extracti64x4_epi64::<1>(u_values);
                v_high_u8 = _mm512_extracti64x4_epi64::<1>(v_values);
                u_low_u8 = _mm512_castsi512_si256(u_values);
                v_low_u8 = _mm512_castsi512_si256(v_values);
            }
        }

        let u_high =
            _mm512_slli_epi16::<SCALE>(_mm512_sub_epi16(_mm512_cvtepu8_epi16(u_high_u8), uv_corr));
        let v_high =
            _mm512_slli_epi16::<SCALE>(_mm512_sub_epi16(_mm512_cvtepu8_epi16(v_high_u8), uv_corr));
        let y_high = _mm512_mulhrs_epi16(
            _mm512_slli_epi16::<SCALE>(_mm512_cvtepu8_epi16(_mm512_extracti64x4_epi64::<1>(
                y_values,
            ))),
            v_luma_coeff,
        );

        let r_high = _mm512_add_epi16(y_high, _mm512_mulhrs_epi16(v_high, v_cr_coeff));
        let b_high = _mm512_add_epi16(y_high, _mm512_mulhrs_epi16(u_high, v_cb_coeff));
        let g_high = _mm512_sub_epi16(
            y_high,
            _mm512_add_epi16(
                _mm512_mulhrs_epi16(v_high, v_g_coeff_1),
                _mm512_mulhrs_epi16(u_high, v_g_coeff_2),
            ),
        );

        let u_low =
            _mm512_slli_epi16::<SCALE>(_mm512_sub_epi16(_mm512_cvtepu8_epi16(u_low_u8), uv_corr));
        let v_low =
            _mm512_slli_epi16::<SCALE>(_mm512_sub_epi16(_mm512_cvtepu8_epi16(v_low_u8), uv_corr));
        let y_low = _mm512_mulhrs_epi16(
            _mm512_slli_epi16::<SCALE>(_mm512_cvtepu8_epi16(_mm512_castsi512_si256(y_values))),
            v_luma_coeff,
        );

        let r_low = _mm512_add_epi16(y_low, _mm512_mulhrs_epi16(v_low, v_cr_coeff));
        let b_low = _mm512_adds_epi16(y_low, _mm512_mulhrs_epi16(u_low, v_cb_coeff));
        let g_low = _mm512_sub_epi16(
            y_low,
            _mm512_add_epi16(
                _mm512_mulhrs_epi16(v_low, v_g_coeff_1),
                _mm512_mulhrs_epi16(u_low, v_g_coeff_2),
            ),
        );

        let r_values = avx512_pack_u16(r_low, r_high);
        let g_values = avx512_pack_u16(g_low, g_high);
        let b_values = avx512_pack_u16(b_low, b_high);

        let dst_shift = cx * channels;

        match destination_channels {
            YuvSourceChannels::Rgb => {
                let ptr = rgba_ptr.add(dst_shift);
                avx512_rgb_u8(ptr, r_values, g_values, b_values);
            }
            YuvSourceChannels::Bgr => {
                let ptr = rgba_ptr.add(dst_shift);
                avx512_rgb_u8(ptr, b_values, g_values, r_values);
            }
            YuvSourceChannels::Rgba => {
                avx512_rgba_u8(
                    rgba_ptr.add(dst_shift),
                    r_values,
                    g_values,
                    b_values,
                    v_alpha,
                );
            }
            YuvSourceChannels::Bgra => {
                avx512_rgba_u8(
                    rgba_ptr.add(dst_shift),
                    b_values,
                    g_values,
                    r_values,
                    v_alpha,
                );
            }
        }

        cx += 64;

        match chroma_subsampling {
            YuvChromaSubsampling::Yuv420 | YuvChromaSubsampling::Yuv422 => {
                uv_x += 32;
            }
            YuvChromaSubsampling::Yuv444 => {
                uv_x += 64;
            }
        }
    }

    ProcessedOffset { cx, ux: uv_x }
}
