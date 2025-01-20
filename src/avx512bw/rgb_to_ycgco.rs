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

use crate::avx512bw::avx512_rgb_to_yuv::avx512_rgb_to_ycgco;
use crate::avx512bw::avx512_utils::{
    avx512_load_rgb_u8, avx512_pack_u16, avx512_pairwise_widen_avg,
};
use crate::internals::ProcessedOffset;
use crate::yuv_support::{to_subsampling, YuvChromaRange, YuvChromaSubsampling, YuvSourceChannels};
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[target_feature(enable = "avx512bw", enable = "avx512f")]
pub(crate) unsafe fn avx512_rgb_to_ycgco_row<const ORIGIN_CHANNELS: u8, const SAMPLING: u8>(
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
    let source_channels: YuvSourceChannels = ORIGIN_CHANNELS.into();
    let channels = source_channels.get_channels_count();

    let y_ptr = y_plane.add(y_offset);
    let cg_ptr = cg_plane.add(cg_offset);
    let co_ptr = co_plane.add(co_offset);
    let rgba_ptr = rgba.as_ptr().add(rgba_offset);

    let mut cx = start_cx;
    let mut uv_x = start_ux;

    const ROUNDING_CONST_BIAS: i32 = 1 << 7;
    let bias_y = range.bias_y as i32 * (1 << 8) + ROUNDING_CONST_BIAS;
    let bias_uv = range.bias_uv as i32 * (1 << 8) + ROUNDING_CONST_BIAS;

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
        let px = cx * channels;

        let (r_values, g_values, b_values) =
            avx512_load_rgb_u8::<ORIGIN_CHANNELS, false>(rgba_ptr.add(px));

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

        _mm512_storeu_si512(y_ptr.add(cx) as *mut _, y_intensity);

        if compute_uv_row {
            let cg = avx512_pack_u16(cg_l, cg_h);
            let co = avx512_pack_u16(co_l, co_h);
            match chroma_subsampling {
                YuvChromaSubsampling::Yuv420 | YuvChromaSubsampling::Yuv422 => {
                    let cb_h = _mm512_castsi512_si256(avx512_pairwise_widen_avg(cg));
                    let cr_h = _mm512_castsi512_si256(avx512_pairwise_widen_avg(co));
                    _mm256_storeu_si256(cg_ptr.add(uv_x) as *mut _ as *mut __m256i, cb_h);
                    _mm256_storeu_si256(co_ptr.add(uv_x) as *mut _ as *mut __m256i, cr_h);
                    uv_x += 32;
                }
                YuvChromaSubsampling::Yuv444 => {
                    _mm512_storeu_si512(cg_ptr.add(uv_x) as *mut _, cg);
                    _mm512_storeu_si512(co_ptr.add(uv_x) as *mut _, co);
                    uv_x += 64;
                }
            }
        }

        cx += 64;
    }

    ProcessedOffset { cx, ux: uv_x }
}
