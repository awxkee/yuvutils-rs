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
use crate::sse::sse_support::{
    sse_deinterleave_rgb, sse_deinterleave_rgba, sse_pairwise_avg_epi16,
};
use crate::sse::sse_ycgco_r::sse_rgb_to_ycgco_r_epi16;
use crate::yuv_support::{YuvChromaRange, YuvChromaSample, YuvSourceChannels};
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[target_feature(enable = "sse4.1")]
pub unsafe fn sse_rgb_to_ycgcor_row<const ORIGIN_CHANNELS: u8, const SAMPLING: u8>(
    range: &YuvChromaRange,
    y_plane: *mut u16,
    cg_plane: *mut u16,
    co_plane: *mut u16,
    rgba: &[u8],
    rgba_offset: usize,
    start_cx: usize,
    start_ux: usize,
    width: usize,
) -> ProcessedOffset {
    let chroma_subsampling: YuvChromaSample = SAMPLING.into();
    let source_channels: YuvSourceChannels = ORIGIN_CHANNELS.into();
    let channels = source_channels.get_channels_count();

    let y_ptr = y_plane;
    let cg_ptr = cg_plane;
    let co_ptr = co_plane;
    let rgba_ptr = rgba.as_ptr().add(rgba_offset);

    let mut cx = start_cx;
    let mut uv_x = start_ux;

    let precision_scale = (1 << 8) as f32;
    let max_colors = (1 << 8) - 1i32;

    const ROUNDING_CONST_BIAS: i32 = 1 << 7;
    let bias_y = range.bias_y as i32 * (1 << 8) + ROUNDING_CONST_BIAS;
    let bias_uv = range.bias_uv as i32 * (1 << 8) + ROUNDING_CONST_BIAS;

    let range_reduction_y =
        (range.range_y as f32 / max_colors as f32 * precision_scale).round() as i32;
    let range_reduction_uv =
        (range.range_uv as f32 / max_colors as f32 * precision_scale).round() as i32;

    let zeros = _mm_setzero_si128();

    let y_bias = _mm_set1_epi32(bias_y);
    let uv_bias = _mm_set1_epi32(bias_uv);
    let v_range_reduction_y = _mm_set1_epi32(range_reduction_y);
    let v_range_reduction_uv = _mm_set1_epi32(range_reduction_uv);

    while cx + 16 < width {
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

        let (y_l, cg_l, co_l) = sse_rgb_to_ycgco_r_epi16(
            r_low,
            g_low,
            b_low,
            y_bias,
            uv_bias,
            v_range_reduction_y,
            v_range_reduction_uv,
        );
        let (y_h, cg_h, co_h) = sse_rgb_to_ycgco_r_epi16(
            r_high,
            g_high,
            b_high,
            y_bias,
            uv_bias,
            v_range_reduction_y,
            v_range_reduction_uv,
        );

        _mm_storeu_si128(y_ptr.add(cx) as *mut __m128i, y_l);
        _mm_storeu_si128(y_ptr.add(cx + 8) as *mut __m128i, y_h);

        match chroma_subsampling {
            YuvChromaSample::YUV420 | YuvChromaSample::YUV422 => {
                let cg_h = sse_pairwise_avg_epi16(cg_l, cg_h);
                let co_h = sse_pairwise_avg_epi16(co_l, co_h);
                _mm_storeu_si128(cg_ptr.add(uv_x) as *mut __m128i, cg_h);
                _mm_storeu_si128(co_ptr.add(uv_x) as *mut __m128i, co_h);
                uv_x += 8;
            }
            YuvChromaSample::YUV444 => {
                _mm_storeu_si128(cg_ptr.add(uv_x) as *mut __m128i, cg_l);
                _mm_storeu_si128(cg_ptr.add(uv_x).add(8) as *mut __m128i, cg_h);
                _mm_storeu_si128(co_ptr.add(uv_x) as *mut __m128i, co_l);
                _mm_storeu_si128(co_ptr.add(uv_x).add(8) as *mut __m128i, co_h);
                uv_x += 16;
            }
        }

        cx += 16;
    }

    ProcessedOffset { cx, ux: uv_x }
}
