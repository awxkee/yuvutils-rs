/*
 * // Copyright (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */
use crate::avx2::avx2_utils::{_mm256_deinterleave_x2_epi8, _mm256_store_interleaved_epi8};
use crate::yuv_support::{YuvChromaSample, Yuy2Description};
use crate::yuv_to_yuy2::YuvToYuy2Navigation;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[target_feature(enable = "avx2")]
pub unsafe fn yuv_to_yuy2_avx2_row<const SAMPLING: u8, const YUY2_TARGET: usize>(
    y_plane: &[u8],
    y_offset: usize,
    u_plane: &[u8],
    u_offset: usize,
    v_plane: &[u8],
    v_offset: usize,
    yuy2_store: &mut [u8],
    yuy2_offset: usize,
    width: u32,
    nav: YuvToYuy2Navigation,
) -> YuvToYuy2Navigation {
    let yuy2_target: Yuy2Description = YUY2_TARGET.into();
    let chroma_subsampling: YuvChromaSample = SAMPLING.into();

    let mut _cx = nav.cx;
    let mut _uv_x = nav.uv_x;
    let mut _yuy2_x = nav.x;
    unsafe {
        let max_x_32 = (width as usize / 2).saturating_sub(32);

        for x in (_yuy2_x..max_x_32).step_by(32) {
            let u_pos = u_offset + _uv_x;
            let v_pos = v_offset + _uv_x;
            let y_pos = y_offset + _cx;

            let u_pixels;
            let v_pixels;
            let y_pixels;

            let y_ptr = y_plane.as_ptr().add(y_pos);
            y_pixels = (
                _mm256_loadu_si256(y_ptr as *const __m256i),
                _mm256_loadu_si256(y_ptr.add(32) as *const __m256i),
            );

            if chroma_subsampling == YuvChromaSample::YUV444 {
                let u_ptr = u_plane.as_ptr().add(u_pos);
                let full_u = (
                    _mm256_loadu_si256(u_ptr as *const __m256i),
                    _mm256_loadu_si256(u_ptr.add(32) as *const __m256i),
                );
                let v_ptr = v_plane.as_ptr().add(v_pos);
                let full_v = (
                    _mm256_loadu_si256(v_ptr as *const __m256i),
                    _mm256_loadu_si256(v_ptr.add(32) as *const __m256i),
                );

                u_pixels = _mm256_avg_epu8(full_u.0, full_u.1);
                v_pixels = _mm256_avg_epu8(full_v.0, full_v.1);
            } else {
                u_pixels = _mm256_loadu_si256(u_plane.as_ptr().add(u_pos) as *const __m256i);
                v_pixels = _mm256_loadu_si256(v_plane.as_ptr().add(v_pos) as *const __m256i);
            }

            let (low_y, high_y) = _mm256_deinterleave_x2_epi8(y_pixels.0, y_pixels.1);

            let storage;
            match yuy2_target {
                Yuy2Description::YUYV => {
                    storage = (low_y, u_pixels, high_y, v_pixels);
                }
                Yuy2Description::UYVY => {
                    storage = (u_pixels, low_y, v_pixels, high_y);
                }
                Yuy2Description::YVYU => {
                    storage = (low_y, v_pixels, high_y, u_pixels);
                }
                Yuy2Description::VYUY => {
                    storage = (v_pixels, low_y, u_pixels, high_y);
                }
            }

            let dst_offset = yuy2_offset + x * 4;

            _mm256_store_interleaved_epi8(
                yuy2_store.as_mut_ptr().add(dst_offset),
                storage.0,
                storage.1,
                storage.2,
                storage.3,
            );

            _yuy2_x = x;

            if x + 32 < max_x_32 {
                _uv_x += match chroma_subsampling {
                    YuvChromaSample::YUV420 | YuvChromaSample::YUV422 => 32,
                    YuvChromaSample::YUV444 => 64,
                };
                _cx += 64;
            }
        }

        YuvToYuy2Navigation {
            cx: _cx,
            uv_x: _uv_x,
            x: _yuy2_x,
        }
    }
}
