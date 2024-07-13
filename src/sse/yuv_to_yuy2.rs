/*
 * // Copyright (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
use crate::sse::sse_support::{__mm128x4, _mm_combineh_epi8, _mm_combinel_epi8, _mm_gethigh_epi8, _mm_getlow_epi8, _mm_loadu_si128_x2, _mm_storeu_si128_x4, sse_interleave_rgba};
use crate::yuv_support::{YuvChromaSample, Yuy2Description};
use crate::yuv_to_yuy2::YuvToYuy2Navigation;

pub fn yuv_to_yuy2_sse_impl<const SAMPLING: u8, const YUY2_TARGET: usize>(
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
        #[rustfmt::skip]
        let v_shuffle = _mm_setr_epi8(0, 2, 4, 6, 8, 10, 12, 14,
                                                1, 3, 5, 7, 9, 11, 13, 15);

        let max_x_16 = (width.saturating_sub(1) as usize / 2).saturating_sub(16);
        let max_x_8 = (width.saturating_sub(1) as usize / 2).saturating_sub(8);

        for x in (_yuy2_x..max_x_16).step_by(16) {
            let u_pos = u_offset + _uv_x;
            let v_pos = v_offset + _uv_x;
            let y_pos = y_offset + _cx;

            let u_pixels;
            let v_pixels;
            let y_pixels;

            y_pixels = _mm_loadu_si128_x2(y_plane.as_ptr().add(y_pos));

            if chroma_subsampling == YuvChromaSample::YUV444 {
                let full_u = _mm_loadu_si128_x2(u_plane.as_ptr().add(u_pos));
                let full_v = _mm_loadu_si128_x2(v_plane.as_ptr().add(v_pos));

                u_pixels = _mm_avg_epu8(full_u.0, full_u.1);
                v_pixels = _mm_avg_epu8(full_v.0, full_v.1);
            } else {
                u_pixels = _mm_loadu_si128(u_plane.as_ptr().add(u_pos) as *const __m128i);
                v_pixels = _mm_loadu_si128(v_plane.as_ptr().add(v_pos) as *const __m128i);
            }

            let y_pixels_low = _mm_shuffle_epi8(y_pixels.0, v_shuffle);
            let y_pixels_high = _mm_shuffle_epi8(y_pixels.1, v_shuffle);

            let low_y = _mm_combinel_epi8(y_pixels_low, y_pixels_high);
            let high_y = _mm_combineh_epi8(y_pixels_low, y_pixels_high);

            let storage;
            match yuy2_target {
                Yuy2Description::YUYV => {
                    storage = __mm128x4(low_y, u_pixels, high_y, v_pixels);
                }
                Yuy2Description::UYVY => {
                    storage = __mm128x4(u_pixels, low_y, v_pixels, high_y);
                }
                Yuy2Description::YVYU => {
                    storage = __mm128x4(low_y, v_pixels, high_y, u_pixels);
                }
                Yuy2Description::VYUY => {
                    storage = __mm128x4(v_pixels, low_y, u_pixels, high_y);
                }
            }

            let dst_offset = yuy2_offset + x * 4;

            let inverleaved = sse_interleave_rgba(storage.0, storage.1, storage.2, storage.3);
            let converted = __mm128x4(inverleaved.0, inverleaved.1, inverleaved.2, inverleaved.3);

            _mm_storeu_si128_x4(yuy2_store.as_mut_ptr().add(dst_offset), converted);

            _yuy2_x = x;

            if x + 16 < max_x_16 {
                _uv_x += match chroma_subsampling {
                    YuvChromaSample::YUV420 | YuvChromaSample::YUV422 => 16,
                    YuvChromaSample::YUV444 => 32,
                };
                _cx += 32;
            }
        }

        for x in (_yuy2_x..max_x_8).step_by(8) {
            let u_pos = u_offset + _uv_x;
            let v_pos = v_offset + _uv_x;
            let y_pos = y_offset + _cx;

            let u_pixels;
            let v_pixels;
            let mut y_pixels;

            y_pixels = _mm_loadu_si128(y_plane.as_ptr().add(y_pos) as *const __m128i);

            if chroma_subsampling == YuvChromaSample::YUV444 {
                let full_u = _mm_loadu_si128(u_plane.as_ptr().add(u_pos) as *const __m128i);
                let full_v = _mm_loadu_si128(v_plane.as_ptr().add(v_pos) as *const __m128i);

                let low_u = _mm_getlow_epi8(full_u);
                let high_u = _mm_gethigh_epi8(full_u);
                u_pixels = _mm_avg_epu8(low_u, high_u);

                let low_v = _mm_getlow_epi8(full_v);
                let high_v = _mm_gethigh_epi8(full_v);

                v_pixels = _mm_avg_epu8(low_v, high_v);
            } else {
                u_pixels = _mm_loadu_si64(u_plane.as_ptr().add(u_pos));
                v_pixels = _mm_loadu_si64(v_plane.as_ptr().add(v_pos));
            }

            y_pixels = _mm_shuffle_epi8(y_pixels, v_shuffle);

            let low_y = _mm_getlow_epi8(y_pixels);
            let high_y = _mm_gethigh_epi8(y_pixels);

            let storage;
            match yuy2_target {
                Yuy2Description::YUYV => {
                    storage = __mm128x4(low_y, u_pixels, high_y, v_pixels);
                }
                Yuy2Description::UYVY => {
                    storage = __mm128x4(u_pixels, low_y, v_pixels, high_y);
                }
                Yuy2Description::YVYU => {
                    storage = __mm128x4(low_y, v_pixels, high_y, u_pixels);
                }
                Yuy2Description::VYUY => {
                    storage = __mm128x4(v_pixels, low_y, u_pixels, high_y);
                }
            }

            let inverleaved = sse_interleave_rgba(storage.0, storage.1, storage.2, storage.3);
            let converted = __mm128x4(inverleaved.0, inverleaved.1, inverleaved.2, inverleaved.3);

            let dst_offset = yuy2_offset + x * 4;

            let ptr = yuy2_store.as_mut_ptr().add(dst_offset);

            _mm_storeu_si128(ptr as *mut __m128i, converted.0);
            _mm_storeu_si128(ptr.add(16) as *mut __m128i, converted.1);

            _yuy2_x = x;

            if x + 8 < max_x_8 {
                _uv_x += match chroma_subsampling {
                    YuvChromaSample::YUV420 | YuvChromaSample::YUV422 => 8,
                    YuvChromaSample::YUV444 => 16,
                };
                _cx += 16;
            }
        }
    }

    return YuvToYuy2Navigation {
        cx: _cx,
        uv_x: _uv_x,
        x: _yuy2_x,
    };
}
