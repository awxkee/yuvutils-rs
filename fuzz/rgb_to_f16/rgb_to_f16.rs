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
use core::f16;
use libfuzzer_sys::fuzz_target;
use yuvutils_rs::{
    convert_plane16_to_f16, convert_plane_f16_to_planar, convert_plane_to_f16, convert_rgb_to_f16,
    convert_rgba_to_f16,
};

fuzz_target!(|data: (u8, u8, u8)| {
    fuzz_f16_converter(data.0, data.1, data.2);
});

fn fuzz_f16_converter(i_width: u8, i_height: u8, y_value: u8) {
    if i_height == 0 || i_width == 0 {
        return;
    }
    let src_plane = vec![y_value; i_height as usize * i_width as usize];
    let mut dst_plane = vec![0.; i_width as usize * i_height as usize];
    convert_plane_to_f16(
        &src_plane,
        i_width as usize,
        &mut dst_plane,
        i_width as usize,
        i_width as usize,
        i_height as usize,
    )
    .unwrap();

    let src_plane1 = vec![y_value; i_height as usize * i_width as usize * 3];
    let mut dst_plane1 = vec![0.; i_width as usize * i_height as usize * 3];
    convert_rgb_to_f16(
        &src_plane1,
        i_width as usize * 3,
        &mut dst_plane1,
        i_width as usize * 3,
        i_width as usize,
        i_height as usize,
    )
    .unwrap();

    let src_plane2 = vec![y_value; i_height as usize * i_width as usize * 4];
    let mut dst_plane2 = vec![0.; i_width as usize * i_height as usize * 4];
    convert_rgba_to_f16(
        &src_plane2,
        i_width as usize * 4,
        &mut dst_plane2,
        i_width as usize * 4,
        i_width as usize,
        i_height as usize,
    )
    .unwrap();

    let src_plane3 = vec![y_value as u16; i_height as usize * i_width as usize];
    let mut dst_plane3 = vec![0.; i_width as usize * i_height as usize];
    convert_plane16_to_f16(
        &src_plane3,
        i_width as usize,
        &mut dst_plane3,
        i_width as usize,
        10,
        i_width as usize,
        i_height as usize,
    )
    .unwrap();

    let src_plane4 = vec![y_value as f16; i_height as usize * i_width as usize];
    let mut dst_plane4 = vec![0u8; i_width as usize * i_height as usize];
    convert_plane_f16_to_planar(
        &src_plane4,
        i_width as usize,
        &mut dst_plane4,
        i_width as usize,
        i_width as usize,
        i_height as usize,
    )
    .unwrap();
}
