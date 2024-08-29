/*
 * // Copyright (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

use crate::avx2::avx2_utils::{
    _mm256_store_interleaved_epi8, avx2_deinterleave_rgb, avx2_interleave_rgb,
};
use crate::yuv_support::YuvSourceChannels;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[target_feature(enable = "avx2")]
pub unsafe fn gbr_to_image_avx<const DESTINATION_CHANNELS: u8>(
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

    let rgb_part_size = channels * 32;

    let default_alpha = _mm256_set1_epi8(-128);

    while _cx + 32 < width as usize {
        let row0 = _mm256_loadu_si256(gbr_start_ptr as *const __m256i);
        let row1 = _mm256_loadu_si256(gbr_start_ptr.add(32) as *const __m256i);
        let row2 = _mm256_loadu_si256(gbr_start_ptr.add(64) as *const __m256i);
        let gbr_pixel = avx2_deinterleave_rgb(row0, row1, row2);

        let r_pixel = gbr_pixel.2;
        let g_pixel = gbr_pixel.0;
        let b_pixel = gbr_pixel.1;

        match destination_channels {
            YuvSourceChannels::Rgb => {
                let new_pixel = avx2_interleave_rgb(r_pixel, g_pixel, b_pixel);
                _mm256_storeu_si256(rgb_start_ptr as *mut __m256i, new_pixel.0);
                _mm256_storeu_si256(rgb_start_ptr.add(32) as *mut __m256i, new_pixel.1);
                _mm256_storeu_si256(rgb_start_ptr.add(64) as *mut __m256i, new_pixel.2);
            }
            YuvSourceChannels::Rgba => {
                _mm256_store_interleaved_epi8(
                    rgb_start_ptr,
                    r_pixel,
                    g_pixel,
                    b_pixel,
                    default_alpha,
                );
            }
            YuvSourceChannels::Bgra => {
                _mm256_store_interleaved_epi8(
                    rgb_start_ptr,
                    b_pixel,
                    g_pixel,
                    r_pixel,
                    default_alpha,
                );
            }
            YuvSourceChannels::Bgr => {
                let new_pixel = avx2_interleave_rgb(b_pixel, g_pixel, r_pixel);
                _mm256_storeu_si256(rgb_start_ptr as *mut __m256i, new_pixel.0);
                _mm256_storeu_si256(rgb_start_ptr.add(32) as *mut __m256i, new_pixel.1);
                _mm256_storeu_si256(rgb_start_ptr.add(64) as *mut __m256i, new_pixel.2);
            }
        }

        gbr_start_ptr = gbr_start_ptr.add(3 * 32);
        rgb_start_ptr = rgb_start_ptr.add(rgb_part_size);
        _cx += 32;
    }

    _cx
}
