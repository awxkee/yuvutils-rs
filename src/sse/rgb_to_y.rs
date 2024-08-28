/*
 * // Copyright (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

use crate::sse::sse_support::{sse_deinterleave_rgb, sse_deinterleave_rgba};
use crate::sse::sse_ycbcr::sse_rgb_to_ycbcr;
use crate::yuv_support::{CbCrForwardTransform, YuvChromaRange, YuvSourceChannels};
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[target_feature(enable = "sse4.1")]
pub unsafe fn sse_rgb_to_y<const ORIGIN_CHANNELS: u8>(
    transform: &CbCrForwardTransform<i32>,
    range: &YuvChromaRange,
    y_plane: *mut u8,
    rgba: &[u8],
    y_offset: usize,
    rgba_offset: usize,
    start_cx: usize,
    width: usize,
) -> usize {
    let source_channels: YuvSourceChannels = ORIGIN_CHANNELS.into();
    let channels = source_channels.get_channels_count();

    let y_ptr = y_plane.add(y_offset);
    let rgba_ptr = rgba.as_ptr().add(rgba_offset);

    let mut cx = start_cx;

    let bias_y = ((range.bias_y as f32 + 0.5f32) * (1i32 << 8i32) as f32) as i32;

    let zeros_si = _mm_setzero_si128();

    while cx + 16 < width {
        let y_bias = _mm_set1_epi32(bias_y);
        let v_yr = _mm_set1_epi16(transform.yr as i16);
        let v_yg = _mm_set1_epi16(transform.yg as i16);
        let v_yb = _mm_set1_epi16(transform.yb as i16);

        let (r_values, g_values, b_values);

        let px = cx * channels;

        match source_channels {
            YuvSourceChannels::Rgb => {
                let source_ptr = rgba_ptr.add(px);
                let row_1 = _mm_loadu_si128(source_ptr as *const __m128i);
                let row_2 = _mm_loadu_si128(source_ptr.add(16) as *const __m128i);
                let row_3 = _mm_loadu_si128(source_ptr.add(32) as *const __m128i);

                let (it1, it2, it3) = sse_deinterleave_rgb(row_1, row_2, row_3);
                r_values = it1;
                g_values = it2;
                b_values = it3;
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
        let r_high = _mm_unpackhi_epi8(r_values, zeros_si);
        let g_low = _mm_cvtepu8_epi16(g_values);
        let g_high = _mm_unpackhi_epi8(g_values, zeros_si);
        let b_low = _mm_cvtepu8_epi16(b_values);
        let b_high = _mm_unpackhi_epi8(b_values, zeros_si);

        let y_l = sse_rgb_to_ycbcr(r_low, g_low, b_low, y_bias, v_yr, v_yg, v_yb);

        let y_h = sse_rgb_to_ycbcr(r_high, g_high, b_high, y_bias, v_yr, v_yg, v_yb);

        let y_yuv = _mm_packus_epi16(y_l, y_h);

        _mm_storeu_si128(y_ptr.add(cx) as *mut __m128i, y_yuv);

        cx += 16;
    }

    return cx;
}
