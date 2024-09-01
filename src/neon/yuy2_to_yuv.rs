/*
 * // Copyright (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */
use crate::yuv_support::{YuvChromaSample, Yuy2Description};
use crate::yuv_to_yuy2::YuvToYuy2Navigation;
use std::arch::aarch64::*;

pub fn yuy2_to_yuv_neon_impl<const SAMPLING: u8, const YUY2_TARGET: usize>(
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
            let dst_offset = yuy2_offset + x * 4;
            let u_pos = u_offset + _uv_x;
            let v_pos = v_offset + _uv_x;
            let y_pos = y_offset + _cx;

            let pixel_set = vld4q_u8(yuy2_store.as_ptr().add(dst_offset));
            let mut y_first = match yuy2_source {
                Yuy2Description::YUYV | Yuy2Description::YVYU => pixel_set.0,
                Yuy2Description::UYVY | Yuy2Description::VYUY => pixel_set.1,
            };
            let mut y_second = match yuy2_source {
                Yuy2Description::YUYV | Yuy2Description::YVYU => pixel_set.2,
                Yuy2Description::UYVY | Yuy2Description::VYUY => pixel_set.3,
            };

            let y_first_reconstructed = vzip1q_u8(y_first, y_second);
            let y_second_reconstructed = vzip2q_u8(y_first, y_second);
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

            vst1q_u8_x2(
                y_plane.as_mut_ptr().add(y_pos),
                uint8x16x2_t(y_first, y_second),
            );

            if chroma_subsampling == YuvChromaSample::YUV444 {
                let low_u_value = vzip1q_u8(u_value, u_value);
                let high_u_value = vzip2q_u8(u_value, u_value);
                let low_v_value = vzip1q_u8(v_value, v_value);
                let high_v_value = vzip2q_u8(v_value, v_value);
                vst1q_u8_x2(
                    u_plane.as_mut_ptr().add(u_pos),
                    uint8x16x2_t(low_u_value, high_u_value),
                );
                vst1q_u8_x2(
                    v_plane.as_mut_ptr().add(v_pos),
                    uint8x16x2_t(low_v_value, high_v_value),
                );
            } else {
                vst1q_u8(u_plane.as_mut_ptr().add(u_pos), u_value);
                vst1q_u8(v_plane.as_mut_ptr().add(v_pos), v_value);
            }

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
            let dst_offset = yuy2_offset + x * 4;
            let u_pos = u_offset + _uv_x;
            let v_pos = v_offset + _uv_x;
            let y_pos = y_offset + _cx;

            let pixel_set = vld4_u8(yuy2_store.as_ptr().add(dst_offset));
            let mut y_first = match yuy2_source {
                Yuy2Description::YUYV | Yuy2Description::YVYU => pixel_set.0,
                Yuy2Description::UYVY | Yuy2Description::VYUY => pixel_set.1,
            };
            let mut y_second = match yuy2_source {
                Yuy2Description::YUYV | Yuy2Description::YVYU => pixel_set.2,
                Yuy2Description::UYVY | Yuy2Description::VYUY => pixel_set.3,
            };

            let y_first_reconstructed = vzip1_u8(y_first, y_second);
            let y_second_reconstructed = vzip2_u8(y_first, y_second);
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

            vst1q_u8(
                y_plane.as_mut_ptr().add(y_pos),
                vcombine_u8(y_first, y_second),
            );

            if chroma_subsampling == YuvChromaSample::YUV444 {
                let low_u_value = vzip1_u8(u_value, u_value);
                let high_u_value = vzip2_u8(u_value, u_value);
                let low_v_value = vzip1_u8(v_value, v_value);
                let high_v_value = vzip2_u8(v_value, v_value);
                vst1q_u8(
                    u_plane.as_mut_ptr().add(u_pos),
                    vcombine_u8(low_u_value, high_u_value),
                );
                vst1q_u8(
                    v_plane.as_mut_ptr().add(v_pos),
                    vcombine_u8(low_v_value, high_v_value),
                );
            } else {
                vst1_u8(u_plane.as_mut_ptr().add(u_pos), u_value);
                vst1_u8(v_plane.as_mut_ptr().add(v_pos), v_value);
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
