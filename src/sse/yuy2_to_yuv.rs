/*
 * // Copyright (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */
use crate::sse::sse_support::sse_deinterleave_rgba;
use crate::yuv_support::{YuvChromaSample, Yuy2Description};
use crate::yuv_to_yuy2::YuvToYuy2Navigation;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[target_feature(enable = "sse4.1")]
pub unsafe fn yuy2_to_yuv_sse_impl<const SAMPLING: u8, const YUY2_TARGET: usize>(
    y_plane: &mut [u8],
    y_offset: usize,
    u_plane: &mut [u8],
    u_offset: usize,
    v_plane: &mut [u8],
    v_offset: usize,
    yuy2_store: &[u8],
    yuy2_offset: usize,
    width: u32,
    nav: YuvToYuy2Navigation,
) -> YuvToYuy2Navigation {
    let yuy2_source: Yuy2Description = YUY2_TARGET.into();
    let chroma_subsampling: YuvChromaSample = SAMPLING.into();

    let mut _cx = nav.cx;
    let mut _uv_x = nav.uv_x;
    let mut _yuy2_x = nav.x;

    unsafe {
        let max_x_16 = (width.saturating_sub(1) as usize / 2).saturating_sub(16);
        let max_x_8 = (width.saturating_sub(1) as usize / 2).saturating_sub(8);

        for x in (_yuy2_x..max_x_16).step_by(16) {
            let yuy2_offset = yuy2_offset + x * 4;
            let u_pos = u_offset + _uv_x;
            let v_pos = v_offset + _uv_x;
            let y_pos = y_offset + _cx;

            let yuy2_ptr = yuy2_store.as_ptr().add(yuy2_offset);

            let j0 = _mm_loadu_si128(yuy2_ptr as *const __m128i);
            let j1 = _mm_loadu_si128(yuy2_ptr.add(16) as *const __m128i);
            let j2 = _mm_loadu_si128(yuy2_ptr.add(32) as *const __m128i);
            let j3 = _mm_loadu_si128(yuy2_ptr.add(48) as *const __m128i);

            let pixel_set = sse_deinterleave_rgba(j0, j1, j2, j3);
            let mut y_first = match yuy2_source {
                Yuy2Description::YUYV | Yuy2Description::YVYU => pixel_set.0,
                Yuy2Description::UYVY | Yuy2Description::VYUY => pixel_set.1,
            };
            let mut y_second = match yuy2_source {
                Yuy2Description::YUYV | Yuy2Description::YVYU => pixel_set.2,
                Yuy2Description::UYVY | Yuy2Description::VYUY => pixel_set.3,
            };

            let y_first_reconstructed = _mm_unpacklo_epi8(y_first, y_second);
            let y_second_reconstructed = _mm_unpackhi_epi8(y_first, y_second);
            y_first = y_first_reconstructed;
            y_second = y_second_reconstructed;

            let u_value = match yuy2_source {
                Yuy2Description::YUYV => pixel_set.1,
                Yuy2Description::UYVY => pixel_set.0,
                Yuy2Description::YVYU => pixel_set.3,
                Yuy2Description::VYUY => pixel_set.2,
            };
            let v_value = match yuy2_source {
                Yuy2Description::YUYV => pixel_set.3,
                Yuy2Description::UYVY => pixel_set.2,
                Yuy2Description::YVYU => pixel_set.1,
                Yuy2Description::VYUY => pixel_set.0,
            };

            if chroma_subsampling == YuvChromaSample::YUV444 {
                let low_u_value = _mm_unpacklo_epi8(u_value, u_value);
                let high_u_value = _mm_unpackhi_epi8(u_value, u_value);
                let low_v_value = _mm_unpacklo_epi8(v_value, v_value);
                let high_v_value = _mm_unpackhi_epi8(v_value, v_value);

                let u_plane_ptr = u_plane.as_mut_ptr().add(u_pos);
                let v_plane_ptr = v_plane.as_mut_ptr().add(v_pos);

                _mm_storeu_si128(u_plane_ptr as *mut __m128i, low_u_value);
                _mm_storeu_si128(u_plane_ptr.add(16) as *mut __m128i, high_u_value);
                _mm_storeu_si128(v_plane_ptr as *mut __m128i, low_v_value);
                _mm_storeu_si128(v_plane_ptr.add(16) as *mut __m128i, high_v_value);
            } else {
                _mm_storeu_si128(u_plane.as_mut_ptr().add(u_pos) as *mut __m128i, u_value);
                _mm_storeu_si128(v_plane.as_mut_ptr().add(v_pos) as *mut __m128i, v_value);
            }

            let y_plane_ptr = y_plane.as_mut_ptr().add(y_pos);

            _mm_storeu_si128(y_plane_ptr as *mut __m128i, y_first);
            _mm_storeu_si128(y_plane_ptr.add(16) as *mut __m128i, y_second);

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
            let yuy2_offset = yuy2_offset + x * 4;
            let u_pos = u_offset + _uv_x;
            let v_pos = v_offset + _uv_x;
            let y_pos = y_offset + _cx;

            let yuy2_ptr = yuy2_store.as_ptr().add(yuy2_offset);

            let j0 = _mm_loadu_si128(yuy2_ptr as *const __m128i);
            let j1 = _mm_loadu_si128(yuy2_ptr.add(16) as *const __m128i);

            let pixel_set = sse_deinterleave_rgba(j0, j1, _mm_setzero_si128(), _mm_setzero_si128());

            let y_first = match yuy2_source {
                Yuy2Description::YUYV | Yuy2Description::YVYU => pixel_set.0,
                Yuy2Description::UYVY | Yuy2Description::VYUY => pixel_set.1,
            };
            let y_second = match yuy2_source {
                Yuy2Description::YUYV | Yuy2Description::YVYU => pixel_set.2,
                Yuy2Description::UYVY | Yuy2Description::VYUY => pixel_set.3,
            };

            let y_reconstructed = _mm_unpacklo_epi8(y_first, y_second);

            let u_value = match yuy2_source {
                Yuy2Description::YUYV => pixel_set.1,
                Yuy2Description::UYVY => pixel_set.0,
                Yuy2Description::YVYU => pixel_set.3,
                Yuy2Description::VYUY => pixel_set.2,
            };
            let v_value = match yuy2_source {
                Yuy2Description::YUYV => pixel_set.3,
                Yuy2Description::UYVY => pixel_set.2,
                Yuy2Description::YVYU => pixel_set.1,
                Yuy2Description::VYUY => pixel_set.0,
            };

            _mm_storeu_si128(
                y_plane.as_mut_ptr().add(y_pos) as *mut __m128i,
                y_reconstructed,
            );

            if chroma_subsampling == YuvChromaSample::YUV444 {
                let u_value = _mm_unpacklo_epi8(u_value, u_value);
                let v_value = _mm_unpacklo_epi8(v_value, v_value);
                _mm_storeu_si128(u_plane.as_mut_ptr().add(u_pos) as *mut __m128i, u_value);
                _mm_storeu_si128(v_plane.as_mut_ptr().add(v_pos) as *mut __m128i, v_value);
            } else {
                std::ptr::copy_nonoverlapping(
                    &u_value as *const _ as *const u8,
                    u_plane.as_mut_ptr().add(u_pos),
                    8,
                );
                std::ptr::copy_nonoverlapping(
                    &v_value as *const _ as *const u8,
                    v_plane.as_mut_ptr().add(v_pos),
                    8,
                );
            }

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

    YuvToYuy2Navigation {
        cx: _cx,
        uv_x: _uv_x,
        x: _yuy2_x,
    }
}
