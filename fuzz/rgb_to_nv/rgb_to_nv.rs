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
use yuvutils_rs::{
    rgb_to_yuv_nv12, rgb_to_yuv_nv16, rgb_to_yuv_nv24, rgba_to_yuv_nv12, rgba_to_yuv_nv16,
    rgba_to_yuv_nv24, BufferStoreMut, YuvBiPlanarImageMut, YuvRange, YuvStandardMatrix,
};

fuzz_target!(|data: (u8, u8, u8, u8, u8)| {
    fuzz_yuv_420(data.0, data.1, data.2, data.3);
    fuzz_yuv_422(data.0, data.1, data.2, data.3);
    fuzz_yuv_444(data.0, data.1, data.2, data.3);
});

fn fuzz_yuv_420(i_width: u8, i_height: u8, y_value: u8, u_value: u8) {
    if i_height == 0 || i_width == 0 {
        return;
    }
    let y_plane = vec![y_value; i_height as usize * i_width as usize];
    let uv_plane =
        vec![u_value; (i_width as usize).div_ceil(2) * (i_height as usize).div_ceil(2) * 2];

    let mut bi_planar_image = YuvBiPlanarImageMut {
        y_plane: BufferStoreMut::Owned(y_plane),
        y_stride: i_width as u32,
        uv_plane: BufferStoreMut::Owned(uv_plane),
        uv_stride: (i_width as u32).div_ceil(2) * 2,
        width: i_width as u32,
        height: i_height as u32,
    };

    let target_rgb = vec![0u8; i_width as usize * i_height as usize * 3];

    rgb_to_yuv_nv12(
        &mut bi_planar_image,
        &target_rgb,
        i_width as u32 * 3,
        YuvRange::Limited,
        YuvStandardMatrix::Bt601,
    )
    .unwrap();

    let target_rgba = vec![0u8; i_width as usize * i_height as usize * 4];

    rgba_to_yuv_nv12(
        &mut bi_planar_image,
        &target_rgba,
        i_width as u32 * 4,
        YuvRange::Limited,
        YuvStandardMatrix::Bt601,
    )
    .unwrap();
}

fn fuzz_yuv_422(i_width: u8, i_height: u8, y_value: u8, u_value: u8) {
    if i_height == 0 || i_width == 0 {
        return;
    }
    let y_plane = vec![y_value; i_height as usize * i_width as usize];
    let uv_plane = vec![u_value; (i_width as usize).div_ceil(2) * i_height as usize * 2];

    let mut planar_image = YuvBiPlanarImageMut {
        y_plane: BufferStoreMut::Owned(y_plane),
        y_stride: i_width as u32,
        uv_plane: BufferStoreMut::Owned(uv_plane),
        uv_stride: (i_width as u32).div_ceil(2) * 2,
        width: i_width as u32,
        height: i_height as u32,
    };

    let target_rgb = vec![0u8; i_width as usize * i_height as usize * 3];

    rgb_to_yuv_nv16(
        &mut planar_image,
        &target_rgb,
        i_width as u32 * 3,
        YuvRange::Limited,
        YuvStandardMatrix::Bt601,
    )
    .unwrap();

    let target_rgba = vec![0u8; i_width as usize * i_height as usize * 4];

    rgba_to_yuv_nv16(
        &mut planar_image,
        &target_rgba,
        i_width as u32 * 4,
        YuvRange::Limited,
        YuvStandardMatrix::Bt601,
    )
    .unwrap();
}

fn fuzz_yuv_444(i_width: u8, i_height: u8, y_value: u8, u_value: u8) {
    if i_height == 0 || i_width == 0 {
        return;
    }
    let y_plane = vec![y_value; i_height as usize * i_width as usize];
    let uv_plane = vec![u_value; i_width as usize * i_height as usize * 2];

    let mut planar_image = YuvBiPlanarImageMut {
        y_plane: BufferStoreMut::Owned(y_plane),
        y_stride: i_width as u32,
        uv_plane: BufferStoreMut::Owned(uv_plane),
        uv_stride: i_width as u32 * 2,
        width: i_width as u32,
        height: i_height as u32,
    };

    let target_rgb = vec![0u8; i_width as usize * i_height as usize * 3];

    rgb_to_yuv_nv24(
        &mut planar_image,
        &target_rgb,
        i_width as u32 * 3,
        YuvRange::Limited,
        YuvStandardMatrix::Bt601,
    )
    .unwrap();

    let target_rgba = vec![0u8; i_width as usize * i_height as usize * 4];

    rgba_to_yuv_nv24(
        &mut planar_image,
        &target_rgba,
        i_width as u32 * 4,
        YuvRange::Limited,
        YuvStandardMatrix::Bt601,
    )
    .unwrap();
}
