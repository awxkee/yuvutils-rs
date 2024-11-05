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
use crate::yuv_support::{YuvChromaSubsample, Yuy2Description};
use crate::yuv_to_yuy2::YuvToYuy2Navigation;
use std::arch::aarch64::*;

pub fn yuv_to_yuy2_neon_impl<const SAMPLING: u8, const YUY2_TARGET: usize>(
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
    let chroma_subsampling: YuvChromaSubsample = SAMPLING.into();

    let shuffle_table: [u8; 16] = [0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15];

    let mut _cx = nav.cx;
    let mut _uv_x = nav.uv_x;
    let mut _yuy2_x = nav.x;
    unsafe {
        let v_shuffle = vld1q_u8(shuffle_table.as_ptr());

        let max_x_16 = (width as usize / 2).saturating_sub(16);
        let max_x_8 = (width as usize / 2).saturating_sub(8);

        for x in (_yuy2_x..max_x_16).step_by(16) {
            let u_pos = u_offset + _uv_x;
            let v_pos = v_offset + _uv_x;
            let y_pos = y_offset + _cx;

            let u_pixels;
            let v_pixels;
            let y_pixels = vld1q_u8_x2(y_plane.as_ptr().add(y_pos));

            if chroma_subsampling == YuvChromaSubsample::Yuv444 {
                let full_u = vld1q_u8_x2(u_plane.as_ptr().add(u_pos));
                let full_v = vld1q_u8_x2(v_plane.as_ptr().add(v_pos));

                u_pixels = vhaddq_u8(full_u.0, full_u.1);
                v_pixels = vhaddq_u8(full_v.0, full_v.1);
            } else {
                u_pixels = vld1q_u8(u_plane.as_ptr().add(u_pos));
                v_pixels = vld1q_u8(v_plane.as_ptr().add(v_pos));
            }

            let y_pixels_low = vqtbl1q_u8(y_pixels.0, v_shuffle);
            let y_pixels_high = vqtbl1q_u8(y_pixels.1, v_shuffle);

            let low_y = vcombine_u8(vget_low_u8(y_pixels_low), vget_low_u8(y_pixels_high));
            let high_y = vcombine_u8(vget_high_u8(y_pixels_low), vget_high_u8(y_pixels_high));

            let storage = match yuy2_target {
                Yuy2Description::YUYV => uint8x16x4_t(low_y, u_pixels, high_y, v_pixels),
                Yuy2Description::UYVY => uint8x16x4_t(u_pixels, low_y, v_pixels, high_y),
                Yuy2Description::YVYU => uint8x16x4_t(low_y, v_pixels, high_y, u_pixels),
                Yuy2Description::VYUY => uint8x16x4_t(v_pixels, low_y, u_pixels, high_y),
            };

            let dst_offset = yuy2_offset + x * 4;

            vst4q_u8(yuy2_store.as_mut_ptr().add(dst_offset), storage);

            _yuy2_x = x;

            if x + 16 < max_x_16 {
                _uv_x += match chroma_subsampling {
                    YuvChromaSubsample::Yuv420 | YuvChromaSubsample::Yuv422 => 16,
                    YuvChromaSubsample::Yuv444 => 32,
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

            y_pixels = vld1q_u8(y_plane.as_ptr().add(y_pos));

            if chroma_subsampling == YuvChromaSubsample::Yuv444 {
                let full_u = vld1q_u8(u_plane.as_ptr().add(u_pos));
                let full_v = vld1q_u8(v_plane.as_ptr().add(v_pos));

                let low_u = vget_low_u8(full_u);
                let high_u = vget_high_u8(full_u);
                u_pixels = vhadd_u8(low_u, high_u);

                let low_v = vget_low_u8(full_v);
                let high_v = vget_high_u8(full_v);

                v_pixels = vhadd_u8(low_v, high_v);
            } else {
                u_pixels = vld1_u8(u_plane.as_ptr().add(u_pos));
                v_pixels = vld1_u8(v_plane.as_ptr().add(v_pos));
            }

            y_pixels = vqtbl1q_u8(y_pixels, v_shuffle);

            let low_y = vget_low_u8(y_pixels);
            let high_y = vget_high_u8(y_pixels);

            let storage = match yuy2_target {
                Yuy2Description::YUYV => uint8x8x4_t(low_y, u_pixels, high_y, v_pixels),
                Yuy2Description::UYVY => uint8x8x4_t(u_pixels, low_y, v_pixels, high_y),
                Yuy2Description::YVYU => uint8x8x4_t(low_y, v_pixels, high_y, u_pixels),
                Yuy2Description::VYUY => uint8x8x4_t(v_pixels, low_y, u_pixels, high_y),
            };

            let dst_offset = yuy2_offset + x * 4;

            vst4_u8(yuy2_store.as_mut_ptr().add(dst_offset), storage);

            _yuy2_x = x;

            if x + 8 < max_x_8 {
                _uv_x += match chroma_subsampling {
                    YuvChromaSubsample::Yuv420 | YuvChromaSubsample::Yuv422 => 8,
                    YuvChromaSubsample::Yuv444 => 16,
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
