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
use crate::sse::sse_support::{sse_store_rgb_u8, sse_store_rgba};
use crate::yuv_support::{YuvChromaRange, YuvChromaSubsampling, YuvSourceChannels};
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[target_feature(enable = "sse4.1")]
pub unsafe fn sse_ycgco_to_rgb_row<const DESTINATION_CHANNELS: u8, const SAMPLING: u8>(
    range: &YuvChromaRange,
    y_plane: &[u8],
    cg_plane: &[u8],
    v_plane: &[u8],
    rgba: &mut [u8],
    start_cx: usize,
    start_ux: usize,
    y_offset: usize,
    u_offset: usize,
    v_offset: usize,
    rgba_offset: usize,
    width: usize,
) -> ProcessedOffset {
    let chroma_subsampling: YuvChromaSubsampling = SAMPLING.into();
    let destination_channels: YuvSourceChannels = DESTINATION_CHANNELS.into();
    let channels = destination_channels.get_channels_count();
    let bias_y = range.bias_y as i32;
    let bias_uv = range.bias_uv as i32;

    let mut cx = start_cx;
    let mut uv_x = start_ux;

    let y_ptr = y_plane.as_ptr().add(y_offset);
    let u_ptr = cg_plane.as_ptr().add(u_offset);
    let v_ptr = v_plane.as_ptr().add(v_offset);
    let rgba_ptr = rgba.as_mut_ptr().add(rgba_offset);

    let max_colors = (1 << 8) - 1i32;
    let precision_scale = (1 << 6) as f32;

    let range_reduction_y =
        (max_colors as f32 / range.range_y as f32 * precision_scale).round() as i32;
    let range_reduction_uv =
        (max_colors as f32 / range.range_uv as f32 * precision_scale).round() as i32;

    let y_corr = _mm_set1_epi16(bias_y as i16);
    let uv_corr = _mm_set1_epi16(bias_uv as i16);
    let y_reduction = _mm_set1_epi16(range_reduction_y as i16);
    let uv_reduction = _mm_set1_epi16(range_reduction_uv as i16);
    let v_alpha = _mm_set1_epi16(-128);
    let rounding_const = _mm_set1_epi16(1 << 5);

    let zeros = _mm_setzero_si128();

    while cx + 16 < width {
        let y_values = _mm_loadu_si128(y_ptr.add(cx) as *const __m128i);

        let (u_high_u16, v_high_u16, u_low_u16, v_low_u16);

        match chroma_subsampling {
            YuvChromaSubsampling::Yuv420 | YuvChromaSubsampling::Yuv422 => {
                let reshuffle = _mm_setr_epi8(0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7);

                let u_values = _mm_shuffle_epi8(_mm_loadu_si64(u_ptr.add(uv_x)), reshuffle);
                let v_values = _mm_shuffle_epi8(_mm_loadu_si64(v_ptr.add(uv_x)), reshuffle);

                u_high_u16 = _mm_unpackhi_epi8(u_values, zeros);
                v_high_u16 = _mm_unpackhi_epi8(v_values, zeros);
                u_low_u16 = _mm_unpacklo_epi8(u_values, zeros);
                v_low_u16 = _mm_unpacklo_epi8(v_values, zeros);
            }
            YuvChromaSubsampling::Yuv444 => {
                let u_values = _mm_loadu_si128(u_ptr.add(uv_x) as *const __m128i);
                let v_values = _mm_loadu_si128(v_ptr.add(uv_x) as *const __m128i);

                u_high_u16 = _mm_unpackhi_epi8(u_values, zeros);
                v_high_u16 = _mm_unpackhi_epi8(v_values, zeros);
                u_low_u16 = _mm_unpacklo_epi8(u_values, zeros);
                v_low_u16 = _mm_unpacklo_epi8(v_values, zeros);
            }
        }

        let cg_high = _mm_mullo_epi16(_mm_subs_epi16(u_high_u16, uv_corr), uv_reduction);
        let co_high = _mm_mullo_epi16(_mm_subs_epi16(v_high_u16, uv_corr), uv_reduction);
        let y_high = _mm_mullo_epi16(
            _mm_sub_epi16(_mm_unpackhi_epi8(y_values, zeros), y_corr),
            y_reduction,
        );

        let t_high = _mm_subs_epi16(y_high, cg_high);

        let r_high = _mm_srai_epi16::<6>(_mm_adds_epi16(
            _mm_max_epi16(_mm_adds_epi16(t_high, co_high), zeros),
            rounding_const,
        ));
        let b_high = _mm_srai_epi16::<6>(_mm_adds_epi16(
            _mm_max_epi16(_mm_subs_epi16(t_high, co_high), zeros),
            rounding_const,
        ));
        let g_high = _mm_srai_epi16::<6>(_mm_adds_epi16(
            _mm_max_epi16(_mm_adds_epi16(y_high, cg_high), zeros),
            rounding_const,
        ));

        let cg_low = _mm_mullo_epi16(_mm_subs_epi16(u_low_u16, uv_corr), uv_reduction);
        let co_low = _mm_mullo_epi16(_mm_subs_epi16(v_low_u16, uv_corr), uv_reduction);
        let y_low = _mm_mullo_epi16(
            _mm_sub_epi16(_mm_cvtepu8_epi16(y_values), y_corr),
            y_reduction,
        );

        let t_low = _mm_subs_epi16(y_low, cg_low);

        let r_low = _mm_srai_epi16::<6>(_mm_adds_epi16(
            _mm_max_epi16(_mm_adds_epi16(t_low, co_low), zeros),
            rounding_const,
        ));
        let b_low = _mm_srai_epi16::<6>(_mm_adds_epi16(
            _mm_max_epi16(_mm_subs_epi16(t_low, co_low), zeros),
            rounding_const,
        ));
        let g_low = _mm_srai_epi16::<6>(_mm_adds_epi16(
            _mm_max_epi16(_mm_adds_epi16(y_low, cg_low), zeros),
            rounding_const,
        ));

        let r_values = _mm_packus_epi16(r_low, r_high);
        let g_values = _mm_packus_epi16(g_low, g_high);
        let b_values = _mm_packus_epi16(b_low, b_high);

        let dst_shift = cx * channels;

        match destination_channels {
            YuvSourceChannels::Rgb => {
                sse_store_rgb_u8(rgba_ptr.add(dst_shift), r_values, g_values, b_values);
            }
            YuvSourceChannels::Bgr => {
                sse_store_rgb_u8(rgba_ptr.add(dst_shift), b_values, g_values, r_values);
            }
            YuvSourceChannels::Rgba => {
                sse_store_rgba(
                    rgba_ptr.add(dst_shift),
                    r_values,
                    g_values,
                    b_values,
                    v_alpha,
                );
            }
            YuvSourceChannels::Bgra => {
                sse_store_rgba(
                    rgba_ptr.add(dst_shift),
                    b_values,
                    g_values,
                    r_values,
                    v_alpha,
                );
            }
        }

        cx += 16;

        match chroma_subsampling {
            YuvChromaSubsampling::Yuv420 | YuvChromaSubsampling::Yuv422 => {
                uv_x += 8;
            }
            YuvChromaSubsampling::Yuv444 => {
                uv_x += 16;
            }
        }
    }

    ProcessedOffset { cx, ux: uv_x }
}
