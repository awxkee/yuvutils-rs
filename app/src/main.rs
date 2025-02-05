#![feature(f16)]
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
mod support;

use image::{ColorType, DynamicImage, EncodableLayout, GenericImageView, ImageReader};
use std::fs::File;
use std::io::Read;
use std::time::Instant;
use yuvutils_rs::{
    ar30_to_rgb8, i010_to_rgb10, i010_to_rgb_f16, i012_to_rgb12, i014_to_rgb14, i016_to_rgb16,
    i210_to_rgb10, i210_to_rgb_f16, i210_to_rgba_f16, i214_to_rgb14, i214_to_rgb_f16,
    i214_to_rgba14, i216_to_rgb16, i410_to_rgb10, i410_to_rgb_f16, i410_to_rgba10, i414_to_rgb14,
    i414_to_rgb_f16, i416_to_rgb16, p210_to_ar30, p212_to_ar30, rgb10_to_i010, rgb10_to_i210,
    rgb10_to_i410, rgb10_to_p210, rgb12_to_i012, rgb12_to_p212, rgb14_to_i014, rgb14_to_i214,
    rgb14_to_i414, rgb16_to_i016, rgb16_to_i216, rgb16_to_i416, rgb_to_ycgco420, rgb_to_ycgco422,
    rgb_to_ycgco444, rgba14_to_i214, ycgco420_to_rgb, ycgco422_to_rgb, ycgco444_to_rgb,
    ycgco444_to_rgba, Rgb30ByteOrder, YuvBiPlanarImageMut, YuvChromaSubsampling, YuvPlanarImageMut,
    YuvRange, YuvStandardMatrix,
};

fn read_file_bytes(file_path: &str) -> Result<Vec<u8>, String> {
    // Open the file
    let mut file = File::open(file_path).unwrap();

    // Create a buffer to hold the file contents
    let mut buffer = Vec::new();

    // Read the file contents into the buffer
    file.read_to_end(&mut buffer).unwrap();

    // Return the buffer
    Ok(buffer)
}
use core::f16;

fn main() {
    let mut img = ImageReader::open("./assets/bench.png")
        .unwrap()
        .decode()
        .unwrap();
    let img = DynamicImage::ImageRgba8(img.to_rgba8());

    let dimensions = img.dimensions();

    let width = dimensions.0;
    let height = dimensions.1;

    let src_bytes = img.as_bytes();
    let mut components = match img.color() {
        ColorType::Rgb8 => 3,
        ColorType::Rgba8 => 4,
        _ => {
            panic!("Not accepted")
        }
    };

    let y_stride = width as usize + 100;
    let u_stride = (width + 1) / 2 + 100;
    let v_stride = (width + 1) / 2 + 100;
    let mut y_plane = vec![0u8; y_stride as usize * height as usize];
    let mut u_plane = vec![0u8; height as usize * u_stride as usize];
    let mut v_plane = vec![0u8; height as usize * v_stride as usize];

    let rgba_stride = width as usize * components;
    let mut rgba = vec![0u8; height as usize * rgba_stride];

    let start_time = Instant::now();

    let mut y_nv_plane = vec![0u8; width as usize * height as usize];
    let mut uv_nv_plane = vec![0u8; width as usize * (height as usize + 1) / 2];

    let mut bi_planar_image = YuvBiPlanarImageMut::<u16>::alloc(
        width as u32,
        height as u32,
        YuvChromaSubsampling::Yuv422,
    );

    let mut planar_image =
        YuvPlanarImageMut::<u8>::alloc(width as u32, height as u32, YuvChromaSubsampling::Yuv420
    //
    let mut bytes_16: Vec<u16> = src_bytes
        .iter()
        .map(|&x| ((x as u16) << 4) | ((x as u16) >> 4))
        .collect();

    let start_time = Instant::now();
    rgb_to_ycgco420(
        &mut planar_image,
        &src_bytes,
        rgba_stride as u32,
        YuvRange::Full,
    )
    .unwrap();

    println!("Forward time: {:?}", start_time.elapsed());
    let fixed = planar_image.to_fixed();
    rgba.fill(0);
    rdp_yuv444_to_rgba(&fixed, &mut rgba, rgba_stride as u32).unwrap();

    let fixed_biplanar = bi_planar_image.to_fixed();
    let fixed_planar = planar_image.to_fixed();
    // bytes_16.fill(0);

    let mut j_rgba = vec![0u8; dimensions.0 as usize * dimensions.1 as usize * 4];

    ycgco420_to_rgb(
        &fixed_planar,
        &mut rgba,
        dimensions.0 as u32 * 3,
        YuvRange::Full,
    )
    .unwrap();
    // let mut rgba_f16: Vec<f16> = vec![0.; rgba.len()];
    // //
    // i210_to_rgb_f16(
    //     &fixed_planar,
    //     &mut rgba_f16,
    //     rgba_stride as u32,
    //     YuvRange::Limited,
    //     YuvStandardMatrix::Bt709,
    // )
    // .unwrap();
    // //
    // println!("Backward time: {:?}", start_time.elapsed());
    //
    // rgba.fill(0);
    //
    // // convert_rgb_f16_to_rgb(&rgba_f16, rgba_stride, &mut rgba, rgba_stride, width as usize, height as usize).unwrap();
    //
    // rgba = bytes_16.iter().map(|&x| (x >> 2) as u8).collect();

    // rgba = rgba_f16.iter().map(|&x| (x as f32 * 255.) as u8).collect();

    image::save_buffer(
        "converted_sharp151.png",
        rgba.as_bytes(),
        dimensions.0,
        dimensions.1,
        if components == 3 {
            image::ExtendedColorType::Rgb8
        } else {
            image::ExtendedColorType::Rgba8
        },
    )
    .unwrap();
}
