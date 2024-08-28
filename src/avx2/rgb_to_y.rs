/*
 * // Copyright (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

use crate::avx2::avx2_utils::{avx2_deinterleave_rgb, avx2_deinterleave_rgba, avx2_pack_u16};
use crate::avx2::avx2_ycbcr::avx2_rgb_to_ycbcr;
use crate::yuv_support::{CbCrForwardTransform, YuvChromaRange, YuvSourceChannels};
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[target_feature(enable = "avx2")]
pub unsafe fn avx2_rgb_to_y_row<const ORIGIN_CHANNELS: u8>(
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

    while cx + 32 < width {
        let y_bias = _mm256_set1_epi32(bias_y);
        let v_yr = _mm256_set1_epi16(transform.yr as i16);
        let v_yg = _mm256_set1_epi16(transform.yg as i16);
        let v_yb = _mm256_set1_epi16(transform.yb as i16);

        let (r_values, g_values, b_values);

        let px = cx * channels;

        match source_channels {
            YuvSourceChannels::Rgb => {
                let source_ptr = rgba_ptr.add(px);
                let row_1 = _mm256_loadu_si256(source_ptr as *const __m256i);
                let row_2 = _mm256_loadu_si256(source_ptr.add(32) as *const __m256i);
                let row_3 = _mm256_loadu_si256(source_ptr.add(64) as *const __m256i);

                let (it1, it2, it3) = avx2_deinterleave_rgb(row_1, row_2, row_3);
                r_values = it1;
                g_values = it2;
                b_values = it3;
            }
            YuvSourceChannels::Rgba | YuvSourceChannels::Bgra => {
                let source_ptr = rgba_ptr.add(px);
                let row_1 = _mm256_loadu_si256(source_ptr as *const __m256i);
                let row_2 = _mm256_loadu_si256(source_ptr.add(32) as *const __m256i);
                let row_3 = _mm256_loadu_si256(source_ptr.add(64) as *const __m256i);
                let row_4 = _mm256_loadu_si256(source_ptr.add(96) as *const __m256i);

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

        let y_l = avx2_rgb_to_ycbcr(r_low, g_low, b_low, y_bias, v_yr, v_yg, v_yb);

        let y_h = avx2_rgb_to_ycbcr(r_high, g_high, b_high, y_bias, v_yr, v_yg, v_yb);

        let y_yuv = avx2_pack_u16(y_l, y_h);

        _mm256_storeu_si256(y_ptr.add(cx) as *mut __m256i, y_yuv);

        cx += 32;
    }

    cx
}
