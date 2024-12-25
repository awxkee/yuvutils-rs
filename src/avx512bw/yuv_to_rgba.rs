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

use crate::avx2::{_mm256_interleave_epi8, _xx256_load_si256};
use crate::avx512bw::avx512_utils::{
    _mm512_expand8_to_10, _xx512_load_si512, avx512_pack_u16, avx512_store_rgba_for_yuv_u8,
    avx512_zip_epi8,
};
use crate::internals::{is_slice_aligned, ProcessedOffset};
use crate::yuv_support::{
    CbCrInverseTransform, YuvChromaRange, YuvChromaSubsampling, YuvSourceChannels,
};
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

pub(crate) fn avx512_yuv_to_rgba<
    const DESTINATION_CHANNELS: u8,
    const SAMPLING: u8,
    const HAS_VBMI: bool,
>(
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
        let y_aligned = is_slice_aligned(y_plane, 64);
        let u_aligned = is_slice_aligned(u_plane, 64);
        let v_aligned = is_slice_aligned(v_plane, 64);
        if y_aligned && u_aligned && v_aligned {
            if HAS_VBMI {
                avx512_yuv_to_rgba_bmi_impl::<DESTINATION_CHANNELS, SAMPLING, true>(
                    range, transform, y_plane, u_plane, v_plane, rgba, start_cx, start_ux, width,
                )
            } else {
                avx512_yuv_to_rgba_def_impl::<DESTINATION_CHANNELS, SAMPLING, true>(
                    range, transform, y_plane, u_plane, v_plane, rgba, start_cx, start_ux, width,
                )
            }
        } else {
            if HAS_VBMI {
                avx512_yuv_to_rgba_bmi_impl::<DESTINATION_CHANNELS, SAMPLING, false>(
                    range, transform, y_plane, u_plane, v_plane, rgba, start_cx, start_ux, width,
                )
            } else {
                avx512_yuv_to_rgba_def_impl::<DESTINATION_CHANNELS, SAMPLING, false>(
                    range, transform, y_plane, u_plane, v_plane, rgba, start_cx, start_ux, width,
                )
            }
        }
    }
}

#[target_feature(enable = "avx512bw", enable = "avx512f", enable = "avx512vbmi")]
unsafe fn avx512_yuv_to_rgba_bmi_impl<
    const DESTINATION_CHANNELS: u8,
    const SAMPLING: u8,
    const ALIGNED: bool,
>(
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
    avx512_yuv_to_rgba_impl::<DESTINATION_CHANNELS, SAMPLING, true, ALIGNED>(
        range, transform, y_plane, u_plane, v_plane, rgba, start_cx, start_ux, width,
    )
}

#[target_feature(enable = "avx512bw", enable = "avx512f")]
unsafe fn avx512_yuv_to_rgba_def_impl<
    const DESTINATION_CHANNELS: u8,
    const SAMPLING: u8,
    const ALIGNED: bool,
>(
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
    avx512_yuv_to_rgba_impl::<DESTINATION_CHANNELS, SAMPLING, false, ALIGNED>(
        range, transform, y_plane, u_plane, v_plane, rgba, start_cx, start_ux, width,
    )
}

#[inline(always)]
unsafe fn avx512_yuv_to_rgba_impl<
    const DESTINATION_CHANNELS: u8,
    const SAMPLING: u8,
    const HAS_VBMI: bool,
    const ALIGNED: bool,
>(
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
    if ALIGNED {
        debug_assert!(y_plane.as_ptr() as usize % 64 == 0);
        debug_assert!(u_plane.as_ptr() as usize % 64 == 0);
        debug_assert!(v_plane.as_ptr() as usize % 64 == 0);
    }
    let chroma_subsampling: YuvChromaSubsampling = SAMPLING.into();
    let destination_channels: YuvSourceChannels = DESTINATION_CHANNELS.into();
    let channels = destination_channels.get_channels_count();

    let mut cx = start_cx;
    let mut uv_x = start_ux;
    let y_ptr = y_plane.as_ptr();
    let u_ptr = u_plane.as_ptr();
    let v_ptr = v_plane.as_ptr();
    let rgba_ptr = rgba.as_mut_ptr();

    let v_luma_coeff = _mm512_set1_epi16(transform.y_coef as i16);
    let v_cr_coeff = _mm512_set1_epi16(transform.cr_coef as i16);
    let v_cb_coeff = _mm512_set1_epi16(transform.cb_coef as i16);
    let v_g_coeff_1 = _mm512_set1_epi16(transform.g_coeff_1 as i16);
    let v_g_coeff_2 = _mm512_set1_epi16(transform.g_coeff_2 as i16);

    const SCALE: u32 = 2;

    while cx + 128 < width {
        let y_corr = _mm512_set1_epi8(range.bias_y as i8);
        let uv_corr = _mm512_set1_epi16(range.bias_uv as i16);
        let y_values0 = _mm512_subs_epu8(
            _xx512_load_si512::<ALIGNED>(y_ptr.add(cx) as *const i32),
            y_corr,
        );
        let y_values1 = _mm512_subs_epu8(
            _xx512_load_si512::<ALIGNED>(y_ptr.add(cx + 64) as *const i32),
            y_corr,
        );

        let (u_high00, v_high00, u_low00, v_low00, u_high10, v_high10, u_low10, v_low10);

        match chroma_subsampling {
            YuvChromaSubsampling::Yuv420 | YuvChromaSubsampling::Yuv422 => {
                let u_values_full = _xx512_load_si512::<ALIGNED>(u_ptr.add(uv_x) as *const i32);
                let v_values_full = _xx512_load_si512::<ALIGNED>(v_ptr.add(uv_x) as *const i32);

                let (u_values0, u_values1) =
                    avx512_zip_epi8::<HAS_VBMI>(u_values_full, u_values_full);
                let (v_values0, v_values1) =
                    avx512_zip_epi8::<HAS_VBMI>(v_values_full, v_values_full);

                u_high00 = _mm512_extracti64x4_epi64::<1>(u_values0);
                v_high00 = _mm512_extracti64x4_epi64::<1>(v_values0);
                u_low00 = _mm512_castsi512_si256(u_values0);
                v_low00 = _mm512_castsi512_si256(v_values0);

                u_high10 = _mm512_extracti64x4_epi64::<1>(u_values1);
                v_high10 = _mm512_extracti64x4_epi64::<1>(v_values1);
                u_low10 = _mm512_castsi512_si256(u_values1);
                v_low10 = _mm512_castsi512_si256(v_values1);
            }
            YuvChromaSubsampling::Yuv444 => {
                let u_values0 = _xx512_load_si512::<ALIGNED>(u_ptr.add(uv_x) as *const i32);
                let u_values1 = _xx512_load_si512::<ALIGNED>(u_ptr.add(uv_x + 64) as *const i32);
                let v_values0 = _xx512_load_si512::<ALIGNED>(v_ptr.add(uv_x) as *const i32);
                let v_values1 = _xx512_load_si512::<ALIGNED>(v_ptr.add(uv_x + 64) as *const i32);

                u_high00 = _mm512_extracti64x4_epi64::<1>(u_values0);
                v_high00 = _mm512_extracti64x4_epi64::<1>(v_values0);
                u_low00 = _mm512_castsi512_si256(u_values0);
                v_low00 = _mm512_castsi512_si256(v_values0);

                u_high10 = _mm512_extracti64x4_epi64::<1>(u_values1);
                v_high10 = _mm512_extracti64x4_epi64::<1>(v_values1);
                u_low10 = _mm512_castsi512_si256(u_values1);
                v_low10 = _mm512_castsi512_si256(v_values1);
            }
        }

        let y0_10 = _mm512_expand8_to_10::<HAS_VBMI>(y_values0);
        let y1_10 = _mm512_expand8_to_10::<HAS_VBMI>(y_values1);

        let u_high0 =
            _mm512_slli_epi16::<SCALE>(_mm512_sub_epi16(_mm512_cvtepu8_epi16(u_high00), uv_corr));
        let v_high0 =
            _mm512_slli_epi16::<SCALE>(_mm512_sub_epi16(_mm512_cvtepu8_epi16(v_high00), uv_corr));
        let y_high0 = _mm512_mulhrs_epi16(y0_10.1, v_luma_coeff);

        let u_high1 =
            _mm512_slli_epi16::<SCALE>(_mm512_sub_epi16(_mm512_cvtepu8_epi16(u_high10), uv_corr));
        let v_high1 =
            _mm512_slli_epi16::<SCALE>(_mm512_sub_epi16(_mm512_cvtepu8_epi16(v_high10), uv_corr));
        let y_high1 = _mm512_mulhrs_epi16(y1_10.1, v_luma_coeff);

        let r_high0 = _mm512_add_epi16(y_high0, _mm512_mulhrs_epi16(v_high0, v_cr_coeff));
        let b_high0 = _mm512_add_epi16(y_high0, _mm512_mulhrs_epi16(u_high0, v_cb_coeff));
        let g_high0 = _mm512_sub_epi16(
            y_high0,
            _mm512_add_epi16(
                _mm512_mulhrs_epi16(v_high0, v_g_coeff_1),
                _mm512_mulhrs_epi16(u_high0, v_g_coeff_2),
            ),
        );

        let r_high1 = _mm512_add_epi16(y_high1, _mm512_mulhrs_epi16(v_high1, v_cr_coeff));
        let b_high1 = _mm512_add_epi16(y_high1, _mm512_mulhrs_epi16(u_high1, v_cb_coeff));
        let g_high1 = _mm512_sub_epi16(
            y_high1,
            _mm512_add_epi16(
                _mm512_mulhrs_epi16(v_high1, v_g_coeff_1),
                _mm512_mulhrs_epi16(u_high1, v_g_coeff_2),
            ),
        );

        let u_low0 =
            _mm512_slli_epi16::<SCALE>(_mm512_sub_epi16(_mm512_cvtepu8_epi16(u_low00), uv_corr));
        let v_low0 =
            _mm512_slli_epi16::<SCALE>(_mm512_sub_epi16(_mm512_cvtepu8_epi16(v_low00), uv_corr));
        let y_low0 = _mm512_mulhrs_epi16(y0_10.0, v_luma_coeff);

        let u_low1 =
            _mm512_slli_epi16::<SCALE>(_mm512_sub_epi16(_mm512_cvtepu8_epi16(u_low10), uv_corr));
        let v_low1 =
            _mm512_slli_epi16::<SCALE>(_mm512_sub_epi16(_mm512_cvtepu8_epi16(v_low10), uv_corr));
        let y_low1 = _mm512_mulhrs_epi16(y1_10.0, v_luma_coeff);

        let r_low0 = _mm512_add_epi16(y_low0, _mm512_mulhrs_epi16(v_low0, v_cr_coeff));
        let b_low0 = _mm512_add_epi16(y_low0, _mm512_mulhrs_epi16(u_low0, v_cb_coeff));
        let g_low0 = _mm512_sub_epi16(
            y_low0,
            _mm512_add_epi16(
                _mm512_mulhrs_epi16(v_low0, v_g_coeff_1),
                _mm512_mulhrs_epi16(u_low0, v_g_coeff_2),
            ),
        );

        let r_low1 = _mm512_add_epi16(y_low1, _mm512_mulhrs_epi16(v_low1, v_cr_coeff));
        let b_low1 = _mm512_add_epi16(y_low1, _mm512_mulhrs_epi16(u_low1, v_cb_coeff));
        let g_low1 = _mm512_sub_epi16(
            y_low1,
            _mm512_add_epi16(
                _mm512_mulhrs_epi16(v_low1, v_g_coeff_1),
                _mm512_mulhrs_epi16(u_low1, v_g_coeff_2),
            ),
        );

        let r_values0 = avx512_pack_u16(r_low0, r_high0);
        let g_values0 = avx512_pack_u16(g_low0, g_high0);
        let b_values0 = avx512_pack_u16(b_low0, b_high0);

        let r_values1 = avx512_pack_u16(r_low1, r_high1);
        let g_values1 = avx512_pack_u16(g_low1, g_high1);
        let b_values1 = avx512_pack_u16(b_low1, b_high1);

        let dst_shift = cx * channels;

        let v_alpha = _mm512_set1_epi8(255u8 as i8);
        avx512_store_rgba_for_yuv_u8::<DESTINATION_CHANNELS, HAS_VBMI>(
            rgba_ptr.add(dst_shift),
            r_values0,
            g_values0,
            b_values0,
            v_alpha,
        );

        avx512_store_rgba_for_yuv_u8::<DESTINATION_CHANNELS, HAS_VBMI>(
            rgba_ptr.add(dst_shift + 64 * channels),
            r_values1,
            g_values1,
            b_values1,
            v_alpha,
        );

        cx += 128;

        match chroma_subsampling {
            YuvChromaSubsampling::Yuv420 | YuvChromaSubsampling::Yuv422 => {
                uv_x += 64;
            }
            YuvChromaSubsampling::Yuv444 => {
                uv_x += 128;
            }
        }
    }

    while cx + 64 < width {
        let y_corr = _mm512_set1_epi8(range.bias_y as i8);
        let uv_corr = _mm512_set1_epi16(range.bias_uv as i16);

        let y_values = _mm512_subs_epu8(
            _xx512_load_si512::<ALIGNED>(y_ptr.add(cx) as *const i32),
            y_corr,
        );

        let (u_high0, v_high0, u_low0, v_low0);

        match chroma_subsampling {
            YuvChromaSubsampling::Yuv420 | YuvChromaSubsampling::Yuv422 => {
                let u_values = _xx256_load_si256::<ALIGNED>(u_ptr.add(uv_x) as *const __m256i);
                let v_values = _xx256_load_si256::<ALIGNED>(v_ptr.add(uv_x) as *const __m256i);

                (u_low0, u_high0) = _mm256_interleave_epi8(u_values, u_values);
                (v_low0, v_high0) = _mm256_interleave_epi8(v_values, v_values);
            }
            YuvChromaSubsampling::Yuv444 => {
                let u_values = _xx512_load_si512::<ALIGNED>(u_ptr.add(uv_x) as *const i32);
                let v_values = _xx512_load_si512::<ALIGNED>(v_ptr.add(uv_x) as *const i32);

                u_high0 = _mm512_extracti64x4_epi64::<1>(u_values);
                v_high0 = _mm512_extracti64x4_epi64::<1>(v_values);
                u_low0 = _mm512_castsi512_si256(u_values);
                v_low0 = _mm512_castsi512_si256(v_values);
            }
        }

        let y_10 = _mm512_expand8_to_10::<HAS_VBMI>(y_values);

        let u_high =
            _mm512_slli_epi16::<SCALE>(_mm512_sub_epi16(_mm512_cvtepu8_epi16(u_high0), uv_corr));
        let v_high =
            _mm512_slli_epi16::<SCALE>(_mm512_sub_epi16(_mm512_cvtepu8_epi16(v_high0), uv_corr));
        let y_high = _mm512_mulhrs_epi16(y_10.1, v_luma_coeff);

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
            _mm512_slli_epi16::<SCALE>(_mm512_sub_epi16(_mm512_cvtepu8_epi16(u_low0), uv_corr));
        let v_low =
            _mm512_slli_epi16::<SCALE>(_mm512_sub_epi16(_mm512_cvtepu8_epi16(v_low0), uv_corr));
        let y_low = _mm512_mulhrs_epi16(y_10.0, v_luma_coeff);

        let r_low = _mm512_add_epi16(y_low, _mm512_mulhrs_epi16(v_low, v_cr_coeff));
        let b_low = _mm512_add_epi16(y_low, _mm512_mulhrs_epi16(u_low, v_cb_coeff));
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

        let v_alpha = _mm512_set1_epi8(255u8 as i8);

        avx512_store_rgba_for_yuv_u8::<DESTINATION_CHANNELS, HAS_VBMI>(
            rgba_ptr.add(dst_shift),
            r_values,
            g_values,
            b_values,
            v_alpha,
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

    ProcessedOffset { cx, ux: uv_x }
}
