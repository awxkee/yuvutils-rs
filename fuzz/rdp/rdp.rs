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
    rdp_abgr_to_yuv444, rdp_argb_to_yuv444, rdp_bgr_to_yuv444, rdp_bgra_to_yuv444,
    rdp_rgb_to_yuv444, rdp_rgba_to_yuv444, rdp_yuv444_to_abgr, rdp_yuv444_to_argb,
    rdp_yuv444_to_bgra, rdp_yuv444_to_rgb, rdp_yuv444_to_rgba, BufferStoreMut, YuvPlanarImageMut,
};

fuzz_target!(|data: (u8, u8)| {
    fuzz_yuv_444(data.0, data.1);
});

fn fuzz_yuv_444(i_width: u8, i_height: u8) {
    if i_height == 0 || i_width == 0 {
        return;
    }
    let y_plane = vec![0i16; i_height as usize * i_width as usize];
    let u_plane = vec![0i16; i_width as usize * i_height as usize];
    let v_plane = vec![0i16; i_width as usize * i_height as usize];

    let mut planar_image = YuvPlanarImageMut {
        y_plane: BufferStoreMut::Owned(y_plane),
        y_stride: i_width as u32,
        u_plane: BufferStoreMut::Owned(u_plane),
        u_stride: i_width as u32,
        v_plane: BufferStoreMut::Owned(v_plane),
        v_stride: i_width as u32,
        width: i_width as u32,
        height: i_height as u32,
    };

    let src_rgb = vec![0u8; i_width as usize * i_height as usize * 3];

    rdp_rgb_to_yuv444(&mut planar_image, &src_rgb, i_width as u32 * 3).unwrap();
    rdp_bgr_to_yuv444(&mut planar_image, &src_rgb, i_width as u32 * 3).unwrap();

    let src_rgba = vec![0u8; i_width as usize * i_height as usize * 4];

    rdp_rgba_to_yuv444(&mut planar_image, &src_rgba, i_width as u32 * 4).unwrap();
    rdp_bgra_to_yuv444(&mut planar_image, &src_rgba, i_width as u32 * 4).unwrap();
    rdp_abgr_to_yuv444(&mut planar_image, &src_rgba, i_width as u32 * 4).unwrap();
    rdp_argb_to_yuv444(&mut planar_image, &src_rgba, i_width as u32 * 4).unwrap();

    let fixed_planar = planar_image.to_fixed();

    let mut target_rgba = vec![0u8; i_width as usize * i_height as usize * 4];

    rdp_yuv444_to_rgba(&fixed_planar, &mut target_rgba, i_width as u32 * 4).unwrap();
    rdp_yuv444_to_abgr(&fixed_planar, &mut target_rgba, i_width as u32 * 4).unwrap();
    rdp_yuv444_to_argb(&fixed_planar, &mut target_rgba, i_width as u32 * 4).unwrap();
    rdp_yuv444_to_bgra(&fixed_planar, &mut target_rgba, i_width as u32 * 4).unwrap();

    let mut target_rgb = vec![0u8; i_width as usize * i_height as usize * 4];

    rdp_yuv444_to_rgb(&fixed_planar, &mut target_rgb, i_width as u32 * 4).unwrap();
}
