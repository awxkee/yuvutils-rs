/*
 * // Copyright (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

use crate::sse::{_mm_deinterleave_rgb_epi16, _mm_interleave_rgb_epi16, _mm_interleave_rgba_epi16};
use crate::yuv_support::YuvSourceChannels;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[target_feature(enable = "sse4.1")]
pub unsafe fn gbr_to_image_sse_p16<const DESTINATION_CHANNELS: u8>(
    gbr: *const u16,
    rgb: *mut u16,
    bit_depth: u32,
    width: u32,
    start_cx: usize,
) -> usize {
    let mut _cx = start_cx;

    let destination_channels: YuvSourceChannels = DESTINATION_CHANNELS.into();
    let channels = destination_channels.get_channels_count();

    let mut gbr_start_ptr = gbr.add(3 * _cx);
    let mut rgb_start_ptr = rgb.add(channels * _cx);

    let rgb_part_size = channels * 8;

    let default_alpha = _mm_set1_epi16((bit_depth << 1) as i16);

    while _cx + 8 < width as usize {
        let row0 = _mm_loadu_si128(gbr_start_ptr as *const __m128i);
        let row1 = _mm_loadu_si128(gbr_start_ptr.add(8) as *const __m128i);
        let row2 = _mm_loadu_si128(gbr_start_ptr.add(16) as *const __m128i);
        let gbr_pixel = _mm_deinterleave_rgb_epi16(row0, row1, row2);

        let r_pixel = gbr_pixel.2;
        let g_pixel = gbr_pixel.0;
        let b_pixel = gbr_pixel.1;

        match destination_channels {
            YuvSourceChannels::Rgb => {
                let new_pixel = _mm_interleave_rgb_epi16(r_pixel, g_pixel, b_pixel);
                _mm_storeu_si128(rgb_start_ptr as *mut __m128i, new_pixel.0);
                _mm_storeu_si128(rgb_start_ptr.add(8) as *mut __m128i, new_pixel.1);
                _mm_storeu_si128(rgb_start_ptr.add(16) as *mut __m128i, new_pixel.2);
            }
            YuvSourceChannels::Rgba => {
                let new_pixel = _mm_interleave_rgba_epi16(r_pixel, g_pixel, b_pixel, default_alpha);
                _mm_storeu_si128(rgb_start_ptr as *mut __m128i, new_pixel.0);
                _mm_storeu_si128(rgb_start_ptr.add(8) as *mut __m128i, new_pixel.1);
                _mm_storeu_si128(rgb_start_ptr.add(16) as *mut __m128i, new_pixel.2);
                _mm_storeu_si128(rgb_start_ptr.add(24) as *mut __m128i, new_pixel.3);
            }
            YuvSourceChannels::Bgra => {
                let new_pixel = _mm_interleave_rgba_epi16(b_pixel, g_pixel, r_pixel, default_alpha);
                _mm_storeu_si128(rgb_start_ptr as *mut __m128i, new_pixel.0);
                _mm_storeu_si128(rgb_start_ptr.add(8) as *mut __m128i, new_pixel.1);
                _mm_storeu_si128(rgb_start_ptr.add(16) as *mut __m128i, new_pixel.2);
                _mm_storeu_si128(rgb_start_ptr.add(24) as *mut __m128i, new_pixel.3);
            }
            YuvSourceChannels::Bgr => {
                let new_pixel = _mm_interleave_rgb_epi16(b_pixel, g_pixel, r_pixel);
                _mm_storeu_si128(rgb_start_ptr as *mut __m128i, new_pixel.0);
                _mm_storeu_si128(rgb_start_ptr.add(8) as *mut __m128i, new_pixel.1);
                _mm_storeu_si128(rgb_start_ptr.add(16) as *mut __m128i, new_pixel.2);
            }
        }

        gbr_start_ptr = gbr_start_ptr.add(3 * 8);
        rgb_start_ptr = rgb_start_ptr.add(rgb_part_size);
        _cx += 8;
    }

    let rgb_part_size_small = channels * 4;

    while _cx + 4 < width as usize {
        let row0 = _mm_loadu_si128(gbr_start_ptr as *const __m128i);
        let row1 = _mm_loadu_si64(gbr_start_ptr.add(8) as *const u8);
        let gbr_pixel = _mm_deinterleave_rgb_epi16(row0, row1, _mm_setzero_si128());

        let r_pixel = gbr_pixel.2;
        let g_pixel = gbr_pixel.0;
        let b_pixel = gbr_pixel.1;

        match destination_channels {
            YuvSourceChannels::Rgb => {
                let new_pixel = _mm_interleave_rgb_epi16(r_pixel, g_pixel, b_pixel);
                _mm_storeu_si128(rgb_start_ptr as *mut __m128i, new_pixel.0);
                let row2 = new_pixel.1;
                std::ptr::copy_nonoverlapping(
                    &row2 as *const _ as *const u8,
                    rgb_start_ptr.add(8) as *mut u8,
                    8,
                );
            }
            YuvSourceChannels::Rgba => {
                let new_pixel = _mm_interleave_rgba_epi16(r_pixel, g_pixel, b_pixel, default_alpha);
                _mm_storeu_si128(rgb_start_ptr as *mut __m128i, new_pixel.0);
                _mm_storeu_si128(rgb_start_ptr.add(8) as *mut __m128i, new_pixel.1);
            }
            YuvSourceChannels::Bgra => {
                let new_pixel = _mm_interleave_rgba_epi16(b_pixel, g_pixel, r_pixel, default_alpha);
                _mm_storeu_si128(rgb_start_ptr as *mut __m128i, new_pixel.0);
                _mm_storeu_si128(rgb_start_ptr.add(8) as *mut __m128i, new_pixel.1);
            }
            YuvSourceChannels::Bgr => {
                let new_pixel = _mm_interleave_rgb_epi16(b_pixel, g_pixel, r_pixel);
                _mm_storeu_si128(rgb_start_ptr as *mut __m128i, new_pixel.0);
                let row2 = new_pixel.1;
                std::ptr::copy_nonoverlapping(
                    &row2 as *const _ as *const u8,
                    rgb_start_ptr.add(8) as *mut u8,
                    8,
                );
            }
        }

        gbr_start_ptr = gbr_start_ptr.add(3 * 4);
        rgb_start_ptr = rgb_start_ptr.add(rgb_part_size_small);
        _cx += 4;
    }

    _cx
}