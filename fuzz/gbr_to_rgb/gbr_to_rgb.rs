/*
 * Copyright (c) Radzivon Bartoshyk, 12/2024. All rights reserved.
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

#![no_main]

use libfuzzer_sys::fuzz_target;
use yuv::{
    gb10_alpha_to_rgba10, gb10_to_rgb10, gb10_to_rgba10, gbr_to_rgb, gbr_to_rgba,
    gbr_with_alpha_to_rgba, YuvPlanarImage, YuvPlanarImageWithAlpha, YuvRange,
};

fuzz_target!(|data: (u8, u8, u8, u8, u8, u8)| {
    fuzz_yuv_444(data.0, data.1, data.2, data.3, data.4);
});

fn fuzz_yuv_444(i_width: u8, i_height: u8, y_value: u8, u_value: u8, v_value: u8) {
    if i_height == 0 || i_width == 0 {
        return;
    }
    let y_plane = vec![y_value; i_height as usize * i_width as usize];
    let a_plane = vec![y_value; i_height as usize * i_width as usize];
    let u_plane = vec![u_value; i_width as usize * i_height as usize];
    let v_plane = vec![v_value; i_width as usize * i_height as usize];

    let planar_image = YuvPlanarImage {
        y_plane: &y_plane,
        y_stride: i_width as u32,
        u_plane: &u_plane,
        u_stride: i_width as u32,
        v_plane: &v_plane,
        v_stride: i_width as u32,
        width: i_width as u32,
        height: i_height as u32,
    };

    let mut target_rgb = vec![0u8; i_width as usize * i_height as usize * 3];

    gbr_to_rgb(
        &planar_image,
        &mut target_rgb,
        i_width as u32 * 3,
        YuvRange::Limited,
    )
    .unwrap();

    gbr_to_rgb(
        &planar_image,
        &mut target_rgb,
        i_width as u32 * 3,
        YuvRange::Full,
    )
    .unwrap();

    let mut target_rgba = vec![0u8; i_width as usize * i_height as usize * 4];

    gbr_to_rgba(
        &planar_image,
        &mut target_rgba,
        i_width as u32 * 4,
        YuvRange::Limited,
    )
    .unwrap();

    gbr_to_rgba(
        &planar_image,
        &mut target_rgba,
        i_width as u32 * 4,
        YuvRange::Full,
    )
    .unwrap();

    let planar_image_with_alpha = YuvPlanarImageWithAlpha {
        y_plane: &y_plane,
        y_stride: i_width as u32,
        u_plane: &u_plane,
        u_stride: i_width as u32,
        v_plane: &v_plane,
        v_stride: i_width as u32,
        a_plane: &a_plane,
        a_stride: i_width as u32,
        width: i_width as u32,
        height: i_height as u32,
    };

    gbr_with_alpha_to_rgba(
        &planar_image_with_alpha,
        &mut target_rgba,
        i_width as u32 * 4,
        YuvRange::Limited,
    )
    .unwrap();
}

fn fuzz_yuv_444_p16(i_width: u8, i_height: u8, y_value: u8, u_value: u8, v_value: u8) {
    if i_height == 0 || i_width == 0 {
        return;
    }
    let y_plane = vec![y_value as u16; i_height as usize * i_width as usize];
    let a_plane = vec![y_value as u16; i_height as usize * i_width as usize];
    let u_plane = vec![u_value as u16; i_width as usize * i_height as usize];
    let v_plane = vec![v_value as u16; i_width as usize * i_height as usize];

    let planar_image = YuvPlanarImage {
        y_plane: &y_plane,
        y_stride: i_width as u32,
        u_plane: &u_plane,
        u_stride: i_width as u32,
        v_plane: &v_plane,
        v_stride: i_width as u32,
        width: i_width as u32,
        height: i_height as u32,
    };

    let mut target_rgb = vec![0u16; i_width as usize * i_height as usize * 3];

    gb10_to_rgb10(
        &planar_image,
        &mut target_rgb,
        i_width as u32 * 3,
        YuvRange::Limited,
    )
    .unwrap();

    gb10_to_rgb10(
        &planar_image,
        &mut target_rgb,
        i_width as u32 * 3,
        YuvRange::Full,
    )
    .unwrap();

    let mut target_rgba = vec![0u16; i_width as usize * i_height as usize * 4];

    gb10_to_rgba10(
        &planar_image,
        &mut target_rgba,
        i_width as u32 * 4,
        YuvRange::Limited,
    )
    .unwrap();

    gb10_to_rgba10(
        &planar_image,
        &mut target_rgba,
        i_width as u32 * 4,
        YuvRange::Full,
    )
    .unwrap();

    let planar_image_with_alpha = YuvPlanarImageWithAlpha {
        y_plane: &y_plane,
        y_stride: i_width as u32,
        u_plane: &u_plane,
        u_stride: i_width as u32,
        v_plane: &v_plane,
        v_stride: i_width as u32,
        a_plane: &a_plane,
        a_stride: i_width as u32,
        width: i_width as u32,
        height: i_height as u32,
    };

    gb10_alpha_to_rgba10(
        &planar_image_with_alpha,
        &mut target_rgba,
        i_width as u32 * 4,
        YuvRange::Limited,
    )
    .unwrap();
}
