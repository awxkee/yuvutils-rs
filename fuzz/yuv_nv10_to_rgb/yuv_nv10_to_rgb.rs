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
    yuv_nv16_p10_to_ar30, yuv_nv16_p10_to_rgb, yuv_nv16_p10_to_rgba, yuv_nv21_p10_to_ar30,
    yuv_nv21_p10_to_rgb, yuv_nv21_p10_to_rgba, Rgb30ByteOrder, YuvBiPlanarImage, YuvBytesPacking,
    YuvConversionMode, YuvEndianness, YuvRange, YuvStandardMatrix,
};

fuzz_target!(|data: (u8, u8, u8, u8, u8, u8)| {
    let mode = match data.5 % 3 {
        0 => YuvConversionMode::Fast,
        1 => YuvConversionMode::Balanced,
        _ => YuvConversionMode::Professional,
    };
    fuzz_yuv_420(data.0, data.1, data.2, data.3, mode);
    fuzz_yuv_422(data.0, data.1, data.2, data.3, mode);
});

fn fuzz_yuv_420(i_width: u8, i_height: u8, y_value: u8, uv_value: u8, mode: YuvConversionMode) {
    if i_height == 0 || i_width == 0 {
        return;
    }
    let y_plane = vec![y_value as u16; i_height as usize * i_width as usize];
    let uv_plane =
        vec![uv_value as u16; (i_width as usize).div_ceil(2) * 2 * (i_height as usize).div_ceil(2)];

    let planar_image = YuvBiPlanarImage {
        y_plane: &y_plane,
        y_stride: i_width as u32,
        uv_plane: &uv_plane,
        uv_stride: (i_width as u32).div_ceil(2) * 2,
        width: i_width as u32,
        height: i_height as u32,
    };

    let mut target_rgb = vec![0u8; i_width as usize * i_height as usize * 3];

    yuv_nv21_p10_to_rgb(
        &planar_image,
        &mut target_rgb,
        i_width as u32 * 3,
        YuvRange::Limited,
        YuvStandardMatrix::Bt601,
        YuvEndianness::LittleEndian,
        YuvBytesPacking::LeastSignificantBytes,
        mode,
    )
    .unwrap();

    let mut target_rgba = vec![0u8; i_width as usize * i_height as usize * 4];

    yuv_nv21_p10_to_rgba(
        &planar_image,
        &mut target_rgba,
        i_width as u32 * 4,
        YuvRange::Limited,
        YuvStandardMatrix::Bt601,
        YuvEndianness::LittleEndian,
        YuvBytesPacking::LeastSignificantBytes,
        mode,
    )
    .unwrap();

    yuv_nv21_p10_to_ar30(
        &planar_image,
        &mut target_rgba,
        i_width as u32 * 4,
        Rgb30ByteOrder::Host,
        YuvRange::Limited,
        YuvStandardMatrix::Bt601,
        YuvEndianness::LittleEndian,
        YuvBytesPacking::LeastSignificantBytes,
    )
    .unwrap();
}

fn fuzz_yuv_422(i_width: u8, i_height: u8, y_value: u8, uv_value: u8, mode: YuvConversionMode) {
    if i_height == 0 || i_width == 0 {
        return;
    }
    let y_plane = vec![y_value as u16; i_height as usize * i_width as usize];
    let uv_plane = vec![uv_value as u16; (i_width as usize).div_ceil(2) * 2 * i_height as usize];

    let planar_image = YuvBiPlanarImage {
        y_plane: &y_plane,
        y_stride: i_width as u32,
        uv_plane: &uv_plane,
        uv_stride: (i_width as u32).div_ceil(2) * 2,
        width: i_width as u32,
        height: i_height as u32,
    };

    let mut target_rgb = vec![0u8; i_width as usize * i_height as usize * 3];

    yuv_nv16_p10_to_rgb(
        &planar_image,
        &mut target_rgb,
        i_width as u32 * 3,
        YuvRange::Limited,
        YuvStandardMatrix::Bt601,
        YuvEndianness::LittleEndian,
        YuvBytesPacking::LeastSignificantBytes,
        mode,
    )
    .unwrap();

    let mut target_rgba = vec![0u8; i_width as usize * i_height as usize * 4];

    yuv_nv16_p10_to_rgba(
        &planar_image,
        &mut target_rgba,
        i_width as u32 * 4,
        YuvRange::Limited,
        YuvStandardMatrix::Bt601,
        YuvEndianness::LittleEndian,
        YuvBytesPacking::LeastSignificantBytes,
        mode,
    )
    .unwrap();

    yuv_nv16_p10_to_ar30(
        &planar_image,
        &mut target_rgba,
        i_width as u32 * 4,
        Rgb30ByteOrder::Host,
        YuvRange::Limited,
        YuvStandardMatrix::Bt601,
        YuvEndianness::LittleEndian,
        YuvBytesPacking::LeastSignificantBytes,
    )
    .unwrap();
}
