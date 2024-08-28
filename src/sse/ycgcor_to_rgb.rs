/*
 * // Copyright (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

use crate::internals::ProcessedOffset;
use crate::sse::sse_support::{sse_store_rgb_u8, sse_store_rgba};
use crate::sse::sse_ycgco_r::sse_ycgco_r_to_rgb_epi16;
use crate::yuv_support::{YuvChromaRange, YuvChromaSample, YuvSourceChannels};
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[target_feature(enable = "sse4.1")]
pub unsafe fn sse_ycgcor_type_to_rgb_row<const DESTINATION_CHANNELS: u8, const SAMPLING: u8>(
    range: &YuvChromaRange,
    y_plane: *const u16,
    cg_plane: *const u16,
    v_plane: *const u16,
    rgba: &mut [u8],
    start_cx: usize,
    start_ux: usize,
    rgba_offset: usize,
    width: usize,
) -> ProcessedOffset {
    let chroma_subsampling: YuvChromaSample = SAMPLING.into();
    let destination_channels: YuvSourceChannels = DESTINATION_CHANNELS.into();
    let channels = destination_channels.get_channels_count();
    let bias_y = range.bias_y as i32;
    let bias_uv = range.bias_uv as i32;
    let max_colors = (1 << 8) - 1i32;
    let precision_scale = (1 << 6) as f32;
    let range_reduction_y =
        (max_colors as f32 / range.range_y as f32 * precision_scale).round() as i32;
    let range_reduction_uv =
        (max_colors as f32 / range.range_uv as f32 * precision_scale).round() as i32;

    let mut cx = start_cx;
    let mut uv_x = start_ux;

    let y_ptr = y_plane;
    let u_ptr = cg_plane;
    let v_ptr = v_plane;
    let rgba_ptr = rgba.as_mut_ptr().add(rgba_offset);

    let y_corr = _mm_set1_epi16(bias_y as i16);
    let uv_corr = _mm_set1_epi16(bias_uv as i16);
    let y_reduction = _mm_set1_epi16(range_reduction_y as i16);
    let uv_reduction = _mm_set1_epi16(range_reduction_uv as i16);
    let v_alpha = _mm_set1_epi16(-128);

    while cx + 16 < width {
        let y_values_lo = _mm_loadu_si128(y_ptr.add(cx) as *const __m128i);
        let y_values_hi = _mm_loadu_si128(y_ptr.add(cx + 8) as *const __m128i);

        let u_high_i16;
        let v_high_i16;
        let u_low_i16;
        let v_low_i16;

        match chroma_subsampling {
            YuvChromaSample::YUV420 | YuvChromaSample::YUV422 => {
                let u_values = _mm_loadu_si128(u_ptr.add(uv_x) as *const __m128i);
                let v_values = _mm_loadu_si128(v_ptr.add(uv_x) as *const __m128i);

                u_high_i16 = _mm_unpackhi_epi16(u_values, u_values);
                v_high_i16 = _mm_unpackhi_epi16(v_values, v_values);
                u_low_i16 = _mm_unpacklo_epi16(u_values, u_values);
                v_low_i16 = _mm_unpacklo_epi16(v_values, v_values);
            }
            YuvChromaSample::YUV444 => {
                let u_values_lo = _mm_loadu_si128(u_ptr.add(uv_x) as *const __m128i);
                let v_values_lo = _mm_loadu_si128(v_ptr.add(uv_x) as *const __m128i);
                let u_values_hi = _mm_loadu_si128(u_ptr.add(uv_x).add(8) as *const __m128i);
                let v_values_hi = _mm_loadu_si128(v_ptr.add(uv_x).add(8) as *const __m128i);

                u_high_i16 = u_values_hi;
                v_high_i16 = v_values_hi;
                u_low_i16 = u_values_lo;
                v_low_i16 = v_values_lo;
            }
        }

        let (r_l, g_l, b_l) = sse_ycgco_r_to_rgb_epi16(
            y_values_lo,
            u_low_i16,
            v_low_i16,
            y_corr,
            uv_corr,
            y_reduction,
            uv_reduction,
        );
        let (r_h, g_h, b_h) = sse_ycgco_r_to_rgb_epi16(
            y_values_hi,
            u_high_i16,
            v_high_i16,
            y_corr,
            uv_corr,
            y_reduction,
            uv_reduction,
        );

        let r_values = _mm_packus_epi16(r_l, r_h);
        let g_values = _mm_packus_epi16(g_l, g_h);
        let b_values = _mm_packus_epi16(b_l, b_h);

        let dst_shift = cx * channels;

        match destination_channels {
            YuvSourceChannels::Rgb => {
                sse_store_rgb_u8(rgba_ptr.add(dst_shift), r_values, g_values, b_values);
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
            YuvChromaSample::YUV420 | YuvChromaSample::YUV422 => {
                uv_x += 8;
            }
            YuvChromaSample::YUV444 => {
                uv_x += 16;
            }
        }
    }

    return ProcessedOffset { cx, ux: uv_x };
}
