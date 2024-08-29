/*
 * // Copyright (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

use crate::avx512bw::avx512_rgb_to_yuv::avx512_rgb_to_ycgco;
use crate::avx512bw::avx512_utils::{
    avx512_deinterleave_rgb, avx512_deinterleave_rgba, avx512_pack_u16, avx512_pairwise_widen_avg,
};
use crate::internals::ProcessedOffset;
use crate::yuv_support::{YuvChromaRange, YuvChromaSample, YuvSourceChannels};
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[target_feature(enable = "avx512bw")]
pub unsafe fn avx512_rgb_to_ycgco_row<const ORIGIN_CHANNELS: u8, const SAMPLING: u8>(
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
    let max_colors = (1i32 << 8i32) - 1i32;

    let range_reduction_y =
        (range.range_y as f32 / max_colors as f32 * precision_scale).round() as i32;
    let range_reduction_uv =
        (range.range_uv as f32 / max_colors as f32 * precision_scale).round() as i32;

    let y_bias = _mm512_set1_epi32(bias_y);
    let uv_bias = _mm512_set1_epi32(bias_uv);
    let v_range_reduction_y = _mm512_set1_epi32(range_reduction_y);
    let v_range_reduction_uv = _mm512_set1_epi32(range_reduction_uv);

    while cx + 64 < width {

        let (r_values, g_values, b_values);

        let px = cx * channels;

        match source_channels {
            YuvSourceChannels::Rgb | YuvSourceChannels::Bgr => {
                let source_ptr = rgba_ptr.add(px);
                let row_1 = _mm512_loadu_si512(source_ptr as *const i32);
                let row_2 = _mm512_loadu_si512(source_ptr.add(64) as *const i32);
                let row_3 = _mm512_loadu_si512(source_ptr.add(128) as *const i32);

                let (it1, it2, it3) = avx512_deinterleave_rgb(row_1, row_2, row_3);
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
                let row_1 = _mm512_loadu_si512(source_ptr as *const i32);
                let row_2 = _mm512_loadu_si512(source_ptr.add(64) as *const i32);
                let row_3 = _mm512_loadu_si512(source_ptr.add(128) as *const i32);
                let row_4 = _mm512_loadu_si512(source_ptr.add(192) as *const i32);

                let (it1, it2, it3, _) = avx512_deinterleave_rgba(row_1, row_2, row_3, row_4);
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

        let r_low = _mm512_cvtepu8_epi16(_mm512_castsi512_si256(r_values));
        let r_high = _mm512_cvtepu8_epi16(_mm512_extracti64x4_epi64::<1>(r_values));
        let g_low = _mm512_cvtepu8_epi16(_mm512_castsi512_si256(g_values));
        let g_high = _mm512_cvtepu8_epi16(_mm512_extracti64x4_epi64::<1>(g_values));
        let b_low = _mm512_cvtepu8_epi16(_mm512_castsi512_si256(b_values));
        let b_high = _mm512_cvtepu8_epi16(_mm512_extracti64x4_epi64::<1>(b_values));

        let (y_l, cg_l, co_l) = avx512_rgb_to_ycgco(
            r_low,
            g_low,
            b_low,
            v_range_reduction_y,
            v_range_reduction_uv,
            y_bias,
            uv_bias,
        );
        let (y_h, cg_h, co_h) = avx512_rgb_to_ycgco(
            r_high,
            g_high,
            b_high,
            v_range_reduction_y,
            v_range_reduction_uv,
            y_bias,
            uv_bias,
        );

        let y_intensity = avx512_pack_u16(y_l, y_h);
        let cg = avx512_pack_u16(cg_l, cg_h);
        let co = avx512_pack_u16(co_l, co_h);

        _mm512_storeu_si512(y_ptr.add(cx) as *mut i32, y_intensity);

        match chroma_subsampling {
            YuvChromaSample::YUV420 | YuvChromaSample::YUV422 => {
                let cb_h = _mm512_castsi512_si256(avx512_pairwise_widen_avg(cg));
                let cr_h = _mm512_castsi512_si256(avx512_pairwise_widen_avg(co));
                _mm256_storeu_si256(cg_ptr.add(uv_x) as *mut _ as *mut __m256i, cb_h);
                _mm256_storeu_si256(co_ptr.add(uv_x) as *mut _ as *mut __m256i, cr_h);
                uv_x += 32;
            }
            YuvChromaSample::YUV444 => {
                _mm512_storeu_si512(cg_ptr.add(uv_x) as *mut i32, cg);
                _mm512_storeu_si512(co_ptr.add(uv_x) as *mut i32, co);
                uv_x += 64;
            }
        }

        cx += 64;
    }

    ProcessedOffset { cx, ux: uv_x }
}
