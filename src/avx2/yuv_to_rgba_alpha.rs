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
use crate::internals::ProcessedOffset;
use crate::yuv_support::{
    CbCrInverseTransform, YuvChromaRange, YuvChromaSubsampling, YuvSourceChannels,
};
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

pub(crate) fn avx2_yuv_to_rgba_alpha<const DESTINATION_CHANNELS: u8, const SAMPLING: u8>(
    range: &YuvChromaRange,
    transform: &CbCrInverseTransform<i32>,
    y_plane: &[u8],
    u_plane: &[u8],
    v_plane: &[u8],
    a_plane: &[u8],
    rgba: &mut [u8],
    start_cx: usize,
    start_ux: usize,
    width: usize,
    use_premultiply: bool,
) -> ProcessedOffset {
    unsafe {
        avx2_yuv_to_rgba_alpha_impl::<DESTINATION_CHANNELS, SAMPLING>(
            range,
            transform,
            y_plane,
            u_plane,
            v_plane,
            a_plane,
            rgba,
            start_cx,
            start_ux,
            width,
            use_premultiply,
        )
    }
}

#[target_feature(enable = "avx2")]
unsafe fn avx2_yuv_to_rgba_alpha_impl<const DESTINATION_CHANNELS: u8, const SAMPLING: u8>(
    range: &YuvChromaRange,
    transform: &CbCrInverseTransform<i32>,
    y_plane: &[u8],
    u_plane: &[u8],
    v_plane: &[u8],
    a_plane: &[u8],
    rgba: &mut [u8],
    start_cx: usize,
    start_ux: usize,
    width: usize,
    use_premultiply: bool,
) -> ProcessedOffset {
    let chroma_subsampling: YuvChromaSubsampling = SAMPLING.into();
    let destination_channels: YuvSourceChannels = DESTINATION_CHANNELS.into();
    let channels = destination_channels.get_channels_count();

    let mut cx = start_cx;
    let mut uv_x = start_ux;
    let y_ptr = y_plane.as_ptr();
    let u_ptr = u_plane.as_ptr();
    let v_ptr = v_plane.as_ptr();
    let a_ptr = a_plane.as_ptr();
    let rgba_ptr = rgba.as_mut_ptr();

    const SCALE: i32 = 2;

    let y_corr = _mm256_set1_epi8(range.bias_y as i8);
    let uv_corr = _mm256_set1_epi16(range.bias_uv as i16);
    let v_luma_coeff = _mm256_set1_epi16(transform.y_coef as i16);
    let v_cr_coeff = _mm256_set1_epi16(transform.cr_coef as i16);
    let v_cb_coeff = _mm256_set1_epi16(transform.cb_coef as i16);
    let v_g_coeff_1 = _mm256_set1_epi16(transform.g_coeff_1 as i16);
    let v_g_coeff_2 = _mm256_set1_epi16(transform.g_coeff_2 as i16);

    while cx + 64 < width {
        let y_values0 =
            _mm256_subs_epu8(_mm256_loadu_si256(y_ptr.add(cx) as *const __m256i), y_corr);
        let y_values1 = _mm256_subs_epu8(
            _mm256_loadu_si256(y_ptr.add(cx + 32) as *const __m256i),
            y_corr,
        );

        let (u_high1, v_high1, u_high0, v_high0, u_low0, v_low0, u_low1, v_low1);

        match chroma_subsampling {
            YuvChromaSubsampling::Yuv420 | YuvChromaSubsampling::Yuv422 => {
                let u_values_v = _mm256_loadu_si256(u_ptr.add(uv_x) as *const __m256i);
                let v_values_v = _mm256_loadu_si256(v_ptr.add(uv_x) as *const __m256i);

                let i_u = _mm256_interleave_epi8(u_values_v, u_values_v);
                let i_v = _mm256_interleave_epi8(v_values_v, v_values_v);

                u_high0 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256::<1>(i_u.0));
                v_high0 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256::<1>(i_v.0));
                u_low0 = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(i_u.0));
                v_low0 = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(i_v.0));

                u_high1 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256::<1>(i_u.1));
                v_high1 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256::<1>(i_v.1));
                u_low1 = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(i_u.1));
                v_low1 = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(i_v.1));
            }
            YuvChromaSubsampling::Yuv444 => {
                let u_values_v0 = _mm256_loadu_si256(u_ptr.add(uv_x) as *const __m256i);
                let u_values_v1 = _mm256_loadu_si256(u_ptr.add(uv_x + 32) as *const __m256i);
                let v_values_v0 = _mm256_loadu_si256(v_ptr.add(uv_x) as *const __m256i);
                let v_values_v1 = _mm256_loadu_si256(v_ptr.add(uv_x + 32) as *const __m256i);

                u_high0 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256::<1>(u_values_v0));
                v_high0 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256::<1>(v_values_v0));
                u_low0 = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(u_values_v0));
                v_low0 = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(v_values_v0));

                u_high1 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256::<1>(u_values_v1));
                v_high1 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256::<1>(v_values_v1));
                u_low1 = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(u_values_v1));
                v_low1 = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(v_values_v1));
            }
        }

        let y0_10 = _mm256_expand8_to_10(y_values0);
        let y1_10 = _mm256_expand8_to_10(y_values1);

        let u_high0 = _mm256_slli_epi16::<SCALE>(_mm256_sub_epi16(u_high0, uv_corr));
        let v_high0 = _mm256_slli_epi16::<SCALE>(_mm256_sub_epi16(v_high0, uv_corr));
        let y_high0 = _mm256_mulhrs_epi16(y0_10.1, v_luma_coeff);

        let u_high1 = _mm256_slli_epi16::<SCALE>(_mm256_sub_epi16(u_high1, uv_corr));
        let v_high1 = _mm256_slli_epi16::<SCALE>(_mm256_sub_epi16(v_high1, uv_corr));
        let y_high1 = _mm256_mulhrs_epi16(y1_10.1, v_luma_coeff);

        let r_high0 = _mm256_add_epi16(y_high0, _mm256_mulhrs_epi16(v_high0, v_cr_coeff));
        let b_high0 = _mm256_add_epi16(y_high0, _mm256_mulhrs_epi16(u_high0, v_cb_coeff));
        let g_high0 = _mm256_sub_epi16(
            y_high0,
            _mm256_add_epi16(
                _mm256_mulhrs_epi16(v_high0, v_g_coeff_1),
                _mm256_mulhrs_epi16(u_high0, v_g_coeff_2),
            ),
        );

        let r_high1 = _mm256_add_epi16(y_high1, _mm256_mulhrs_epi16(v_high1, v_cr_coeff));
        let b_high1 = _mm256_add_epi16(y_high1, _mm256_mulhrs_epi16(u_high1, v_cb_coeff));
        let g_high1 = _mm256_sub_epi16(
            y_high1,
            _mm256_add_epi16(
                _mm256_mulhrs_epi16(v_high1, v_g_coeff_1),
                _mm256_mulhrs_epi16(u_high1, v_g_coeff_2),
            ),
        );

        let u_low0 = _mm256_slli_epi16::<SCALE>(_mm256_sub_epi16(u_low0, uv_corr));
        let v_low0 = _mm256_slli_epi16::<SCALE>(_mm256_sub_epi16(v_low0, uv_corr));
        let y_low0 = _mm256_mulhrs_epi16(y0_10.0, v_luma_coeff);

        let u_low1 = _mm256_slli_epi16::<SCALE>(_mm256_sub_epi16(u_low1, uv_corr));
        let v_low1 = _mm256_slli_epi16::<SCALE>(_mm256_sub_epi16(v_low1, uv_corr));
        let y_low1 = _mm256_mulhrs_epi16(y1_10.0, v_luma_coeff);

        let r_low0 = _mm256_add_epi16(y_low0, _mm256_mulhrs_epi16(v_low0, v_cr_coeff));
        let b_low0 = _mm256_add_epi16(y_low0, _mm256_mulhrs_epi16(u_low0, v_cb_coeff));
        let g_low0 = _mm256_sub_epi16(
            y_low0,
            _mm256_add_epi16(
                _mm256_mulhrs_epi16(v_low0, v_g_coeff_1),
                _mm256_mulhrs_epi16(u_low0, v_g_coeff_2),
            ),
        );

        let r_low1 = _mm256_add_epi16(y_low1, _mm256_mulhrs_epi16(v_low1, v_cr_coeff));
        let b_low1 = _mm256_add_epi16(y_low1, _mm256_mulhrs_epi16(u_low1, v_cb_coeff));
        let g_low1 = _mm256_sub_epi16(
            y_low1,
            _mm256_add_epi16(
                _mm256_mulhrs_epi16(v_low1, v_g_coeff_1),
                _mm256_mulhrs_epi16(u_low1, v_g_coeff_2),
            ),
        );

        let a_values0 = _mm256_loadu_si256(a_ptr.add(cx) as *const __m256i);
        let a_values1 = _mm256_loadu_si256(a_ptr.add(cx + 32) as *const __m256i);

        let (r_values0, g_values0, b_values0);
        let (r_values1, g_values1, b_values1);

        if use_premultiply {
            let a_high0 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256::<1>(a_values0));
            let a_low0 = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(a_values0));

            let r_l = avx2_div_by255(_mm256_mullo_epi16(r_low0, a_low0));
            let r_h = avx2_div_by255(_mm256_mullo_epi16(r_high0, a_high0));
            let g_l = avx2_div_by255(_mm256_mullo_epi16(g_low0, a_low0));
            let g_h = avx2_div_by255(_mm256_mullo_epi16(g_high0, a_high0));
            let b_l = avx2_div_by255(_mm256_mullo_epi16(b_low0, a_low0));
            let b_h = avx2_div_by255(_mm256_mullo_epi16(b_high0, a_high0));

            r_values0 = avx2_pack_u16(r_l, r_h);
            g_values0 = avx2_pack_u16(g_l, g_h);
            b_values0 = avx2_pack_u16(b_l, b_h);

            let a_high1 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256::<1>(a_values1));
            let a_low1 = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(a_values1));

            let r_l = avx2_div_by255(_mm256_mullo_epi16(r_low1, a_low1));
            let r_h = avx2_div_by255(_mm256_mullo_epi16(r_high1, a_high1));
            let g_l = avx2_div_by255(_mm256_mullo_epi16(g_low1, a_low1));
            let g_h = avx2_div_by255(_mm256_mullo_epi16(g_high1, a_high1));
            let b_l = avx2_div_by255(_mm256_mullo_epi16(b_low1, a_low1));
            let b_h = avx2_div_by255(_mm256_mullo_epi16(b_high1, a_high1));

            r_values1 = avx2_pack_u16(r_l, r_h);
            g_values1 = avx2_pack_u16(g_l, g_h);
            b_values1 = avx2_pack_u16(b_l, b_h);
        } else {
            r_values0 = avx2_pack_u16(r_low0, r_high0);
            g_values0 = avx2_pack_u16(g_low0, g_high0);
            b_values0 = avx2_pack_u16(b_low0, b_high0);

            r_values1 = avx2_pack_u16(r_low1, r_high1);
            g_values1 = avx2_pack_u16(g_low1, g_high1);
            b_values1 = avx2_pack_u16(b_low1, b_high1);
        }

        let dst_shift = cx * channels;

        _mm256_store_interleave_rgb_for_yuv::<DESTINATION_CHANNELS>(
            rgba_ptr.add(dst_shift),
            r_values0,
            g_values0,
            b_values0,
            a_values0,
        );

        _mm256_store_interleave_rgb_for_yuv::<DESTINATION_CHANNELS>(
            rgba_ptr.add(dst_shift + channels * 32),
            r_values1,
            g_values1,
            b_values1,
            a_values1,
        );

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

    while cx + 32 < width {
        let y_values =
            _mm256_subs_epu8(_mm256_loadu_si256(y_ptr.add(cx) as *const __m256i), y_corr);

        let a_values = _mm256_loadu_si256(a_ptr.add(cx) as *const __m256i);

        let (u_high_u16, v_high_u16, u_low_u16, v_low_u16);

        match chroma_subsampling {
            YuvChromaSubsampling::Yuv420 | YuvChromaSubsampling::Yuv422 => {
                let u_values = _mm_loadu_si128(u_ptr.add(uv_x) as *const __m128i);
                let v_values = _mm_loadu_si128(v_ptr.add(uv_x) as *const __m128i);

                u_high_u16 = _mm256_cvtepu8_epi16(_mm_unpackhi_epi8(u_values, u_values));
                v_high_u16 = _mm256_cvtepu8_epi16(_mm_unpackhi_epi8(v_values, v_values));
                u_low_u16 = _mm256_cvtepu8_epi16(_mm_unpacklo_epi8(u_values, u_values));
                v_low_u16 = _mm256_cvtepu8_epi16(_mm_unpacklo_epi8(v_values, v_values));
            }
            YuvChromaSubsampling::Yuv444 => {
                let u_values = _mm256_loadu_si256(u_ptr.add(uv_x) as *const __m256i);
                let v_values = _mm256_loadu_si256(v_ptr.add(uv_x) as *const __m256i);

                u_high_u16 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256::<1>(u_values));
                v_high_u16 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256::<1>(v_values));
                u_low_u16 = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(u_values));
                v_low_u16 = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(v_values));
            }
        }

        let y0_10 = _mm256_expand8_to_10(y_values);

        let u_high = _mm256_slli_epi16::<SCALE>(_mm256_sub_epi16(u_high_u16, uv_corr));
        let v_high = _mm256_slli_epi16::<SCALE>(_mm256_sub_epi16(v_high_u16, uv_corr));
        let y_high = _mm256_mulhrs_epi16(y0_10.1, v_luma_coeff);

        let r_high = _mm256_add_epi16(y_high, _mm256_mulhrs_epi16(v_high, v_cr_coeff));
        let b_high = _mm256_add_epi16(y_high, _mm256_mulhrs_epi16(u_high, v_cb_coeff));
        let g_high = _mm256_sub_epi16(
            y_high,
            _mm256_add_epi16(
                _mm256_mulhrs_epi16(v_high, v_g_coeff_1),
                _mm256_mulhrs_epi16(u_high, v_g_coeff_2),
            ),
        );

        let u_low = _mm256_slli_epi16::<SCALE>(_mm256_sub_epi16(u_low_u16, uv_corr));
        let v_low = _mm256_slli_epi16::<SCALE>(_mm256_sub_epi16(v_low_u16, uv_corr));
        let y_low = _mm256_mulhrs_epi16(y0_10.0, v_luma_coeff);

        let r_low = _mm256_add_epi16(y_low, _mm256_mulhrs_epi16(v_low, v_cr_coeff));
        let b_low = _mm256_add_epi16(y_low, _mm256_mulhrs_epi16(u_low, v_cb_coeff));
        let g_low = _mm256_sub_epi16(
            y_low,
            _mm256_add_epi16(
                _mm256_mulhrs_epi16(v_low, v_g_coeff_1),
                _mm256_mulhrs_epi16(u_low, v_g_coeff_2),
            ),
        );

        let (r_values, g_values, b_values);

        if use_premultiply {
            let a_high = _mm256_cvtepu8_epi16(_mm256_extracti128_si256::<1>(a_values));
            let a_low = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(a_values));

            let r_l = avx2_div_by255(_mm256_mullo_epi16(r_low, a_low));
            let r_h = avx2_div_by255(_mm256_mullo_epi16(r_high, a_high));
            let g_l = avx2_div_by255(_mm256_mullo_epi16(g_low, a_low));
            let g_h = avx2_div_by255(_mm256_mullo_epi16(g_high, a_high));
            let b_l = avx2_div_by255(_mm256_mullo_epi16(b_low, a_low));
            let b_h = avx2_div_by255(_mm256_mullo_epi16(b_high, a_high));

            r_values = avx2_pack_u16(r_l, r_h);
            g_values = avx2_pack_u16(g_l, g_h);
            b_values = avx2_pack_u16(b_l, b_h);
        } else {
            r_values = avx2_pack_u16(r_low, r_high);
            g_values = avx2_pack_u16(g_low, g_high);
            b_values = avx2_pack_u16(b_low, b_high);
        }

        let dst_shift = cx * channels;

        _mm256_store_interleave_rgb_for_yuv::<DESTINATION_CHANNELS>(
            rgba_ptr.add(dst_shift),
            r_values,
            g_values,
            b_values,
            a_values,
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
