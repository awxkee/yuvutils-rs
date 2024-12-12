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
use crate::sse::utils::sse_deinterleave_rgba;
use crate::yuv_support::{YuvChromaSubsampling, Yuy2Description};
use crate::yuv_to_yuy2::YuvToYuy2Navigation;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

pub(crate) fn yuy2_to_yuv_sse<const SAMPLING: u8, const YUY2_TARGET: usize>(
    y_plane: &mut [u8],
    u_plane: &mut [u8],
    v_plane: &mut [u8],
    yuy2_store: &[u8],
    width: u32,
    nav: YuvToYuy2Navigation,
) -> YuvToYuy2Navigation {
    unsafe {
        yuy2_to_yuv_sse_impl::<SAMPLING, YUY2_TARGET>(
            y_plane, u_plane, v_plane, yuy2_store, width, nav,
        )
    }
}

#[target_feature(enable = "sse4.1")]
unsafe fn yuy2_to_yuv_sse_impl<const SAMPLING: u8, const YUY2_TARGET: usize>(
    y_plane: &mut [u8],
    u_plane: &mut [u8],
    v_plane: &mut [u8],
    yuy2_store: &[u8],
    width: u32,
    nav: YuvToYuy2Navigation,
) -> YuvToYuy2Navigation {
    let yuy2_source: Yuy2Description = YUY2_TARGET.into();
    let chroma_subsampling: YuvChromaSubsampling = SAMPLING.into();

    let mut _cx = nav.cx;
    let mut _uv_x = nav.uv_x;
    let mut _yuy2_x = nav.x;

    unsafe {
        while _cx + 32 < width as usize {
            let dst_offset = _cx * 2;
            let u_pos = _uv_x;
            let v_pos = _uv_x;
            let y_pos = _cx;

            let yuy2_ptr = yuy2_store.as_ptr().add(dst_offset);

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

            if chroma_subsampling == YuvChromaSubsampling::Yuv444 {
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

            _uv_x += match chroma_subsampling {
                YuvChromaSubsampling::Yuv420 | YuvChromaSubsampling::Yuv422 => 16,
                YuvChromaSubsampling::Yuv444 => 32,
            };
            _cx += 32;
        }

        while _cx + 16 < width as usize {
            let dst_offset = _cx * 2;
            let u_pos = _uv_x;
            let v_pos = _uv_x;
            let y_pos = _cx;

            let yuy2_ptr = yuy2_store.as_ptr().add(dst_offset);

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

            if chroma_subsampling == YuvChromaSubsampling::Yuv444 {
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

            _uv_x += match chroma_subsampling {
                YuvChromaSubsampling::Yuv420 | YuvChromaSubsampling::Yuv422 => 8,
                YuvChromaSubsampling::Yuv444 => 16,
            };
            _cx += 16;
        }

        _yuy2_x = _cx;
    }

    YuvToYuy2Navigation {
        cx: _cx,
        uv_x: _uv_x,
        x: _yuy2_x,
    }
}
