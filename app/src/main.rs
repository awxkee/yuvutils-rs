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
    i010_to_rgb10, i010_to_rgb_f16, i012_to_rgb12, i014_to_rgb14, i016_to_rgb16, i214_to_rgb14,
    i214_to_rgb_f16, i214_to_rgba14, i216_to_rgb16, i410_to_rgb_f16, i410_to_rgba10, i414_to_rgb14,
    i414_to_rgb_f16, i416_to_rgb16, rgb10_to_i010, rgb10_to_i410, rgb12_to_i012, rgb14_to_i014,
    rgb14_to_i214, rgb14_to_i414, rgb16_to_i016, rgb16_to_i216, rgb16_to_i416, rgba14_to_i214,
    YuvBiPlanarImageMut, YuvChromaSubsampling, YuvPlanarImageMut, YuvRange, YuvStandardMatrix,
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
use image::imageops::FilterType;

fn main() {
    let mut img = ImageReader::open("./assets/bench.jpg")
        .unwrap()
        .decode()
        .unwrap();
    let img = DynamicImage::ImageRgb8(img.to_rgb8());

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
        YuvPlanarImageMut::<u16>::alloc(width as u32, height as u32, YuvChromaSubsampling::Yuv420);
    //
    let mut bytes_16: Vec<u16> = src_bytes
        .iter()
        .map(|&x| ((x as u16) << 2) | ((x as u16) >> 6))
        .collect();

    let start_time = Instant::now();
    rgb10_to_i010(
        &mut planar_image,
        &bytes_16,
        rgba_stride as u32,
        YuvRange::Limited,
        YuvStandardMatrix::Bt2020,
    )
    .unwrap();

    // bytes_16.fill(0);
    //
    // println!("rgb_to_yuv_nv12 time: {:?}", start_time.elapsed());
    //
    println!("Forward time: {:?}", start_time.elapsed());
    // // // //
    // let full_size = if width % 2 == 0 {
    //     2 * width as usize * height as usize
    // } else {
    //     2 * (width as usize + 1) * height as usize
    // };
    // // //
    // // // // println!("Full YUY2 {}", full_size);
    // // //
    // let yuy2_stride = if width % 2 == 0 {
    //     2 * width as usize
    // } else {
    //     2 * (width as usize + 1)
    // };
    //
    // let mut yuy2_plane = vec![0u8; full_size];
    // // // // //
    // // let start_time = Instant::now();
    // // // // //
    // let plane = planar_image.to_fixed();
    // // //
    // let mut packed_image_mut = YuvPackedImageMut {
    //     yuy: BufferStoreMut::Owned(yuy2_plane),
    //     yuy_stride: yuy2_stride as u32,
    //     width,
    //     height,
    // };
    // //
    // yuv420_to_yuyv422(&mut packed_image_mut, &plane).unwrap();
    //
    // save_yuy2_image("test_lanes_image.yuv2", packed_image_mut.width as usize, packed_image_mut.height as usize, packed_image_mut.yuy.borrow()).unwrap();
    //
    // // let end_time = Instant::now().sub(start_time);
    // // println!("yuv420_to_yuyv422 time: {:?}", end_time);
    // // // rgba.fill(0);
    // // // let start_time = Instant::now();
    // let yuy2_img = packed_image_mut.to_fixed();
    // yuyv422_to_rgb(
    //     &yuy2_img,
    //     &mut rgba,
    //     rgba_stride as u32,
    //     YuvRange::Limited,
    //     YuvStandardMatrix::Bt601,
    // )
    // .unwrap();
    //
    // let end_time = Instant::now().sub(start_time);
    // println!("yuyv422_to_rgb time: {:?}", end_time);

    // let start_time = Instant::now();
    // //
    //
    // let packed_image = packed_image_mut.to_fixed();
    //
    // yuyv422_to_yuv420(&mut planar_image, &packed_image).unwrap();
    // // //
    // let end_time = Instant::now().sub(start_time);
    // println!("yuyv422_to_yuv444 time: {:?}", end_time);
    // rgba.fill(0);
    // let mut bgra = vec![0u8; width as usize * height as usize * 4];
    // let start_time = Instant::now();
    // yuv420_to_rgb(
    //     &y_plane,
    //     y_stride as u32,
    //     &u_plane,
    //     u_stride as u32,
    //     &v_plane,
    //     v_stride as u32,
    //     &mut rgba,
    //     rgba_stride as u32,
    //     width as u32,
    //     height as u32,
    //     YuvRange::TV,
    //     YuvStandardMatrix::Bt601,
    // )
    // .unwrap();
    // let end_time = Instant::now().sub(start_time);

    let fixed_biplanar = bi_planar_image.to_fixed();
    let fixed_planar = planar_image.to_fixed();
    // bytes_16.fill(0);

    let mut j_rgba = vec![0u16; dimensions.0 as usize * dimensions.1 as usize * 4];

    i010_to_rgb10(
        &fixed_planar,
        &mut bytes_16,
        dimensions.0 as u32 * 3,
        YuvRange::Limited,
        YuvStandardMatrix::Bt2020,
    )
    .unwrap();

    // let a_plane = vec![1023u16; width as usize * height as usize];
    // let planar_with_alpha = YuvPlanarImageWithAlpha {
    //     y_plane: planar_image.y_plane.borrow(),
    //     y_stride: planar_image.y_stride,
    //     u_plane: planar_image.u_plane.borrow(),
    //     u_stride: planar_image.u_stride,
    //     v_plane: planar_image.v_plane.borrow(),
    //     v_stride: planar_image.v_stride,
    //     a_plane: &a_plane,
    //     a_stride: width,
    //     width,
    //     height,
    // };

    // components = 4;
    // bytes_16.resize(width as usize * height as usize * 4, 0u16);
    // rgba.resize(width as usize * height as usize * 4, 0u8);

    // let mut rgba_f16: Vec<f16> = vec![0.; rgba.len()];
    //
    // i010_to_rgb_f16(
    //     &fixed_planar,
    //     &mut rgba_f16,
    //     rgba_stride as u32,
    //     YuvRange::Full,
    //     YuvStandardMatrix::Bt2020,
    // )
    // .unwrap();
    //
    // println!("Backward time: {:?}", start_time.elapsed());
    //
    // rgba.fill(0);
    //
    // // convert_rgb_f16_to_rgb(&rgba_f16, rgba_stride, &mut rgba, rgba_stride, width as usize, height as usize).unwrap();
    //
    rgba = bytes_16.iter().map(|&x| (x >> 2) as u8).collect();
    //
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
