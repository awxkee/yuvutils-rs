/*
 * // Copyright (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

use crate::avx2::avx2_utils::{
    avx2_deinterleave_rgb, avx2_deinterleave_rgba, avx2_pack_u16, avx2_pairwise_widen_avg,
};
use crate::avx2::avx2_ycgco::avx2_rgb_to_ycgco;
use crate::internals::ProcessedOffset;
use crate::yuv_support::{YuvChromaRange, YuvChromaSample, YuvSourceChannels};
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn avx2_rgb_to_ycgco_row<const ORIGIN_CHANNELS: u8, const SAMPLING: u8>(
    range: &YuvChromaRange,
    y_plane: *mut u8,
    cg_plane: *mut u8,
    co_plane: *mut u8,
    rgba: &[u8],
    y_offset: usize,
    cg_offset: usize,
    co_offset: usize,
    rgba_offset: usize,
    start_cx: usize,
    start_ux: usize,
    width: usize,
) -> ProcessedOffset {
    let chroma_subsampling: YuvChromaSample = SAMPLING.into();
    let source_channels: YuvSourceChannels = ORIGIN_CHANNELS.into();
    let channels = source_channels.get_channels_count();

    let y_ptr = y_plane.add(y_offset);
    let cg_ptr = cg_plane.add(cg_offset);
    let co_ptr = co_plane.add(co_offset);
    let rgba_ptr = rgba.as_ptr().add(rgba_offset);

    let mut cx = start_cx;
    let mut uv_x = start_ux;

    let bias_y = ((range.bias_y as f32 + 0.5f32) * (1i32 << 8i32) as f32) as i32;
    let bias_uv = ((range.bias_uv as f32 + 0.5f32) * (1i32 << 8i32) as f32) as i32;

    let precision_scale = (1 << 8) as f32;
    let max_colors = 2i32.pow(8) - 1i32;

    let range_reduction_y =
        (range.range_y as f32 / max_colors as f32 * precision_scale).round() as i32;
    let range_reduction_uv =
        (range.range_uv as f32 / max_colors as f32 * precision_scale).round() as i32;

    while cx + 32 < width {
        let y_bias = _mm256_set1_epi32(bias_y);
        let uv_bias = _mm256_set1_epi32(bias_uv);
        let v_range_reduction_y = _mm256_set1_epi32(range_reduction_y);
        let v_range_reduction_uv = _mm256_set1_epi32(range_reduction_uv);

        let (r_values, g_values, b_values);

        let px = cx * channels;

        match source_channels {
            YuvSourceChannels::Rgb => {
                let row_1 = _mm256_loadu_si256(rgba_ptr.add(px) as *const __m256i);
                let row_2 = _mm256_loadu_si256(rgba_ptr.add(px + 32) as *const __m256i);
                let row_3 = _mm256_loadu_si256(rgba_ptr.add(px + 64) as *const __m256i);

                let (it1, it2, it3) = avx2_deinterleave_rgb(row_1, row_2, row_3);
                r_values = it1;
                g_values = it2;
                b_values = it3;
            }
            YuvSourceChannels::Rgba | YuvSourceChannels::Bgra => {
                let row_1 = _mm256_loadu_si256(rgba_ptr.add(px) as *const __m256i);
                let row_2 = _mm256_loadu_si256(rgba_ptr.add(px + 32) as *const __m256i);
                let row_3 = _mm256_loadu_si256(rgba_ptr.add(px + 64) as *const __m256i);
                let row_4 = _mm256_loadu_si256(rgba_ptr.add(px + 96) as *const __m256i);

                let (it1, it2, it3, _) = avx2_deinterleave_rgba(row_1, row_2, row_3, row_4);
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

        let r_low = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(r_values));
        let r_high = _mm256_cvtepu8_epi16(_mm256_extracti128_si256::<1>(r_values));
        let g_low = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(g_values));
        let g_high = _mm256_cvtepu8_epi16(_mm256_extracti128_si256::<1>(g_values));
        let b_low = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(b_values));
        let b_high = _mm256_cvtepu8_epi16(_mm256_extracti128_si256::<1>(b_values));

        let (y_l, cg_l, co_l) = avx2_rgb_to_ycgco(
            r_low,
            g_low,
            b_low,
            v_range_reduction_y,
            v_range_reduction_uv,
            y_bias,
            uv_bias,
        );
        let (y_h, cg_h, co_h) = avx2_rgb_to_ycgco(
            r_high,
            g_high,
            b_high,
            v_range_reduction_y,
            v_range_reduction_uv,
            y_bias,
            uv_bias,
        );

        let y_intensity = avx2_pack_u16(y_l, y_h);
        let cg = avx2_pack_u16(cg_l, cg_h);
        let co = avx2_pack_u16(co_l, co_h);

        _mm256_storeu_si256(y_ptr.add(cx) as *mut __m256i, y_intensity);

        match chroma_subsampling {
            YuvChromaSample::YUV420 | YuvChromaSample::YUV422 => {
                let cb_h = _mm256_castsi256_si128(avx2_pairwise_widen_avg(cg));
                let cr_h = _mm256_castsi256_si128(avx2_pairwise_widen_avg(co));
                _mm_storeu_si128(cg_ptr.add(uv_x) as *mut _ as *mut __m128i, cb_h);
                _mm_storeu_si128(co_ptr.add(uv_x) as *mut _ as *mut __m128i, cr_h);
                uv_x += 16;
            }
            YuvChromaSample::YUV444 => {
                _mm256_storeu_si256(cg_ptr.add(uv_x) as *mut __m256i, cg);
                _mm256_storeu_si256(co_ptr.add(uv_x) as *mut __m256i, co);
                uv_x += 32;
            }
        }

        cx += 32;
    }

    return ProcessedOffset { cx, ux: uv_x };
}
