/*
 * // Copyright (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

use crate::sse::{sse_deinterleave_rgb, sse_interleave_rgb, sse_interleave_rgba};
use crate::yuv_support::YuvSourceChannels;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[target_feature(enable = "sse4.1")]
pub unsafe fn gbr_to_image_sse<const DESTINATION_CHANNELS: u8>(
    gbr: &[u8],
    gbr_offset: usize,
    rgb: &mut [u8],
    rgb_offset: usize,
    width: u32,
    start_cx: usize,
) -> usize {
    let mut _cx = start_cx;

    let destination_channels: YuvSourceChannels = DESTINATION_CHANNELS.into();
    let channels = destination_channels.get_channels_count();

    let mut gbr_start_ptr = gbr.as_ptr().add(gbr_offset);
    let mut rgb_start_ptr = rgb.as_mut_ptr().add(rgb_offset);

    let rgb_part_size = channels * 16;

    let default_alpha = _mm_set1_epi8(-128);

    while _cx + 16 < width as usize {
        let row0 = _mm_loadu_si128(gbr_start_ptr as *const __m128i);
        let row1 = _mm_loadu_si128(gbr_start_ptr.add(16) as *const __m128i);
        let row2 = _mm_loadu_si128(gbr_start_ptr.add(32) as *const __m128i);
        let gbr_pixel = sse_deinterleave_rgb(row0, row1, row2);

        let r_pixel = gbr_pixel.2;
        let g_pixel = gbr_pixel.0;
        let b_pixel = gbr_pixel.1;

        match destination_channels {
            YuvSourceChannels::Rgb => {
                let new_pixel = sse_interleave_rgb(r_pixel, g_pixel, b_pixel);
                _mm_storeu_si128(rgb_start_ptr as *mut __m128i, new_pixel.0);
                _mm_storeu_si128(rgb_start_ptr.add(16) as *mut __m128i, new_pixel.1);
                _mm_storeu_si128(rgb_start_ptr.add(32) as *mut __m128i, new_pixel.2);
            }
            YuvSourceChannels::Rgba => {
                let new_pixel = sse_interleave_rgba(r_pixel, g_pixel, b_pixel, default_alpha);
                _mm_storeu_si128(rgb_start_ptr as *mut __m128i, new_pixel.0);
                _mm_storeu_si128(rgb_start_ptr.add(16) as *mut __m128i, new_pixel.1);
                _mm_storeu_si128(rgb_start_ptr.add(32) as *mut __m128i, new_pixel.2);
                _mm_storeu_si128(rgb_start_ptr.add(48) as *mut __m128i, new_pixel.3);
            }
            YuvSourceChannels::Bgra => {
                let new_pixel = sse_interleave_rgba(b_pixel, g_pixel, r_pixel, default_alpha);
                _mm_storeu_si128(rgb_start_ptr as *mut __m128i, new_pixel.0);
                _mm_storeu_si128(rgb_start_ptr.add(16) as *mut __m128i, new_pixel.1);
                _mm_storeu_si128(rgb_start_ptr.add(32) as *mut __m128i, new_pixel.2);
                _mm_storeu_si128(rgb_start_ptr.add(48) as *mut __m128i, new_pixel.3);
            }
            YuvSourceChannels::Bgr => {
                let new_pixel = sse_interleave_rgb(b_pixel, g_pixel, r_pixel);
                _mm_storeu_si128(rgb_start_ptr as *mut __m128i, new_pixel.0);
                _mm_storeu_si128(rgb_start_ptr.add(16) as *mut __m128i, new_pixel.1);
                _mm_storeu_si128(rgb_start_ptr.add(32) as *mut __m128i, new_pixel.2);
            }
        }

        gbr_start_ptr = gbr_start_ptr.add(3 * 16);
        rgb_start_ptr = rgb_start_ptr.add(rgb_part_size);
        _cx += 16;
    }

    let rgb_part_size_small = channels * 8;

    while _cx + 8 < width as usize {
        let row0 = _mm_loadu_si128(gbr_start_ptr as *const __m128i);
        let row1 = _mm_loadu_si64(gbr_start_ptr.add(16));
        let gbr_pixel = sse_deinterleave_rgb(row0, row1, _mm_setzero_si128());

        let r_pixel = gbr_pixel.2;
        let g_pixel = gbr_pixel.0;
        let b_pixel = gbr_pixel.1;

        match destination_channels {
            YuvSourceChannels::Rgb => {
                let new_pixel = sse_interleave_rgb(r_pixel, g_pixel, b_pixel);
                _mm_storeu_si128(rgb_start_ptr as *mut __m128i, new_pixel.0);
                let row2 = new_pixel.1;
                std::ptr::copy_nonoverlapping(
                    &row2 as *const _ as *const u8,
                    rgb_start_ptr.add(16),
                    8,
                );
            }
            YuvSourceChannels::Rgba => {
                let new_pixel = sse_interleave_rgba(r_pixel, g_pixel, b_pixel, default_alpha);
                _mm_storeu_si128(rgb_start_ptr as *mut __m128i, new_pixel.0);
                _mm_storeu_si128(rgb_start_ptr.add(16) as *mut __m128i, new_pixel.1);
            }
            YuvSourceChannels::Bgra => {
                let new_pixel = sse_interleave_rgba(b_pixel, g_pixel, r_pixel, default_alpha);
                _mm_storeu_si128(rgb_start_ptr as *mut __m128i, new_pixel.0);
                _mm_storeu_si128(rgb_start_ptr.add(16) as *mut __m128i, new_pixel.1);
            }
            YuvSourceChannels::Bgr => {
                let new_pixel = sse_interleave_rgb(b_pixel, g_pixel, r_pixel);
                _mm_storeu_si128(rgb_start_ptr as *mut __m128i, new_pixel.0);
                let row2 = new_pixel.1;
                std::ptr::copy_nonoverlapping(
                    &row2 as *const _ as *const u8,
                    rgb_start_ptr.add(16),
                    8,
                );
            }
        }

        gbr_start_ptr = gbr_start_ptr.add(3 * 8);
        rgb_start_ptr = rgb_start_ptr.add(rgb_part_size_small);
        _cx += 8;
    }

    _cx
}
