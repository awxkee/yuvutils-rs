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
use crate::avx2::avx2_utils::{
    _mm256_deinterleave_rgba_epi8, avx2_deinterleave_rgb, avx2_interleave_rgb,
};
use crate::yuv_support::YuvSourceChannels;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[target_feature(enable = "avx2")]
pub unsafe fn image_to_gbr_avx<const SOURCE_CHANNELS: u8>(
    rgb: &[u8],
    rgb_offset: usize,
    gbr: &mut [u8],
    gbr_offset: usize,
    width: u32,
    start_cx: usize,
) -> usize {
    let mut _cx = start_cx;

    let source_channels: YuvSourceChannels = SOURCE_CHANNELS.into();
    let channels = source_channels.get_channels_count();

    let mut gbr_start_ptr = gbr.as_mut_ptr().add(gbr_offset + 3 * _cx);
    let mut rgb_start_ptr = rgb.as_ptr().add(rgb_offset + channels * _cx);

    let rgb_part_size = channels * 32;

    while _cx + 32 < width as usize {
        let r_pixel;
        let g_pixel;
        let b_pixel;

        let row0 = _mm256_loadu_si256(rgb_start_ptr as *const __m256i);
        let row1 = _mm256_loadu_si256(rgb_start_ptr.add(32) as *const __m256i);
        let row2 = _mm256_loadu_si256(rgb_start_ptr.add(64) as *const __m256i);

        match source_channels {
            YuvSourceChannels::Rgb => {
                let rgb_pixel = avx2_deinterleave_rgb(row0, row1, row2);
                r_pixel = rgb_pixel.0;
                g_pixel = rgb_pixel.1;
                b_pixel = rgb_pixel.2;
            }
            YuvSourceChannels::Rgba => {
                let row3 = _mm256_loadu_si256(rgb_start_ptr.add(96) as *const __m256i);
                let rgb_pixel = _mm256_deinterleave_rgba_epi8(row0, row1, row2, row3);
                r_pixel = rgb_pixel.0;
                g_pixel = rgb_pixel.1;
                b_pixel = rgb_pixel.2;
            }
            YuvSourceChannels::Bgra => {
                let row3 = _mm256_loadu_si256(rgb_start_ptr.add(96) as *const __m256i);
                let rgb_pixel = _mm256_deinterleave_rgba_epi8(row0, row1, row2, row3);
                r_pixel = rgb_pixel.2;
                g_pixel = rgb_pixel.1;
                b_pixel = rgb_pixel.0;
            }
            YuvSourceChannels::Bgr => {
                let rgb_pixel = avx2_deinterleave_rgb(row0, row1, row2);
                r_pixel = rgb_pixel.2;
                g_pixel = rgb_pixel.1;
                b_pixel = rgb_pixel.0;
            }
        }

        let interleaved = avx2_interleave_rgb(g_pixel, b_pixel, r_pixel);
        _mm256_storeu_si256(gbr_start_ptr as *mut __m256i, interleaved.0);
        _mm256_storeu_si256(gbr_start_ptr.add(32) as *mut __m256i, interleaved.1);
        _mm256_storeu_si256(gbr_start_ptr.add(64) as *mut __m256i, interleaved.2);

        gbr_start_ptr = gbr_start_ptr.add(3 * 32);
        rgb_start_ptr = rgb_start_ptr.add(rgb_part_size);
        _cx += 32;
    }

    _cx
}
