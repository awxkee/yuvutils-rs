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

use crate::internals::ProcessedOffset;
use crate::sse::sse_ycbcr::sse_rgb_to_ycgco;
use crate::sse::utils::{sse_deinterleave_rgb, sse_deinterleave_rgba, sse_pairwise_widen_avg};
use crate::yuv_support::{
    to_channels_layout, to_subsampling, YuvChromaRange, YuvChromaSubsampling, YuvSourceChannels,
};
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[target_feature(enable = "sse4.1")]
pub(crate) unsafe fn sse_rgb_to_ycgco_row<const ORIGIN_CHANNELS: u8, const SAMPLING: u8>(
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
    compute_uv_row: bool,
) -> ProcessedOffset {
    let chroma_subsampling: YuvChromaSubsampling = to_subsampling(SAMPLING);
    let source_channels: YuvSourceChannels = to_channels_layout(ORIGIN_CHANNELS);
    let channels = source_channels.get_channels_count();

    let y_ptr = y_plane.add(y_offset);
    let cg_ptr = cg_plane.add(cg_offset);
    let co_ptr = co_plane.add(co_offset);
    let rgba_ptr = rgba.as_ptr().add(rgba_offset);

    let mut cx = start_cx;
    let mut uv_x = start_ux;

    const ROUNDING_CONST_BIAS: i32 = (1 << 7) - 1;
    let bias_y = range.bias_y as i32 * (1 << 8) + ROUNDING_CONST_BIAS;
    let bias_uv = range.bias_uv as i32 * (1 << 8) + ROUNDING_CONST_BIAS;

    let precision_scale = (1 << 8) as f32;
    let max_colors = (1 << 8) - 1i32;

    let range_reduction_y =
        (range.range_y as f32 / max_colors as f32 * precision_scale).round() as i32;
    let range_reduction_uv =
        (range.range_uv as f32 / max_colors as f32 * precision_scale).round() as i32;

    let zeros = _mm_setzero_si128();

    while cx + 16 < width {
        let y_bias = _mm_set1_epi32(bias_y);
        let uv_bias = _mm_set1_epi32(bias_uv);
        let v_range_reduction_y = _mm_set1_epi32(range_reduction_y);
        let v_range_reduction_uv = _mm_set1_epi32(range_reduction_uv);

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

        let r_low = _mm_cvtepu8_epi16(r_values);
        let r_high = _mm_unpackhi_epi8(r_values, zeros);
        let g_low = _mm_cvtepu8_epi16(g_values);
        let g_high = _mm_unpackhi_epi8(g_values, zeros);
        let b_low = _mm_cvtepu8_epi16(b_values);
        let b_high = _mm_unpackhi_epi8(b_values, zeros);

        let (y_l, cg_l, co_l) = sse_rgb_to_ycgco(
            r_low,
            g_low,
            b_low,
            v_range_reduction_y,
            v_range_reduction_uv,
            y_bias,
            uv_bias,
        );
        let (y_h, cg_h, co_h) = sse_rgb_to_ycgco(
            r_high,
            g_high,
            b_high,
            v_range_reduction_y,
            v_range_reduction_uv,
            y_bias,
            uv_bias,
        );

        let y_intensity = _mm_packus_epi16(y_l, y_h);

        _mm_storeu_si128(y_ptr.add(cx) as *mut __m128i, y_intensity);

        if compute_uv_row {
            let cg = _mm_packus_epi16(cg_l, cg_h);
            let co = _mm_packus_epi16(co_l, co_h);
            match chroma_subsampling {
                YuvChromaSubsampling::Yuv420 | YuvChromaSubsampling::Yuv422 => {
                    let cb_h = sse_pairwise_widen_avg(cg);
                    let cr_h = sse_pairwise_widen_avg(co);
                    _mm_storeu_si64(cg_ptr.add(uv_x), cb_h);
                    _mm_storeu_si64(co_ptr.add(uv_x), cr_h);
                    uv_x += 8;
                }
                YuvChromaSubsampling::Yuv444 => {
                    _mm_storeu_si128(cg_ptr.add(uv_x) as *mut __m128i, cg);
                    _mm_storeu_si128(co_ptr.add(uv_x) as *mut __m128i, co);
                    uv_x += 16;
                }
            }
        }
        cx += 16;
    }

    ProcessedOffset { cx, ux: uv_x }
}
