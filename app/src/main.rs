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

use image::io::Reader as ImageReader;
use image::{ColorType, EncodableLayout, GenericImageView};
use std::fs::File;
use std::io::Read;
use std::ops::Sub;
use std::time::Instant;
use image::imageops::FilterType;
use yuvutils_rs::{bgra_to_yuv444_p16, rgb_to_sharp_yuv420, rgb_to_yuv420, rgb_to_yuv420_p16, rgb_to_yuv422, rgb_to_yuv444, rgba_to_sharp_yuv420, rgba_to_yuv420_p16, rgba_to_yuv444_p16, yuv420_p16_to_rgb16, yuv420_to_rgb, yuv420_to_yuyv422, yuyv422_to_rgb, SharpYuvGammaTransfer, YuvBytesPacking, YuvEndianness, YuvRange, YuvStandardMatrix};

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

fn main() {
    let mut img = ImageReader::open("./assets/main_test1.jpg")
        .unwrap()
        .decode()
        .unwrap();
    img = img.resize_exact(img.dimensions().0 + 1, img.dimensions().1, FilterType::Nearest);
    let dimensions = img.dimensions();

    let width = dimensions.0;
    let height = dimensions.1;

    let src_bytes = img.as_bytes();
    let components = match img.color() {
        ColorType::Rgb8 => 3,
        ColorType::Rgba8 => 4,
        _ => {
            panic!("Not accepted")
        }
    };

    let y_stride = width as usize * std::mem::size_of::<u8>();
    let u_stride = width;
    let v_stride = width;
    let mut y_plane = vec![0u8; width as usize * height as usize];
    let mut u_plane = vec![0u8; height as usize * u_stride as usize];
    let mut v_plane = vec![0u8; height as usize * v_stride as usize];

    let rgba_stride = width as usize * components;
    let mut rgba = vec![0u8; height as usize * rgba_stride];

    let start_time = Instant::now();

    let mut y_nv_plane = vec![0u16; width as usize * height as usize];
    let mut uv_nv_plane = vec![0u16; width as usize * (height as usize + 1) / 2];

    let mut bytes_16: Vec<u16> = src_bytes.iter().map(|&x| (x as u16) << 2).collect();

    let start_time = Instant::now();
    // rgb_to_yuv_nv12_p16(
    //     &mut y_nv_plane,
    //     width * 2,
    //     &mut uv_nv_plane,
    //     width * 2,
    //     &bytes_16,
    //     rgba_stride as u32 * 2,
    //     10,
    //     width,
    //     height,
    //     YuvRange::Full,
    //     YuvStandardMatrix::Bt709,
    //     YuvEndianness::BigEndian,
    //     YuvBytesPacking::LeastSignificantBytes,
    // );
    // bytes_16.fill(0);
    //
    let end_time = Instant::now().sub(start_time);
    println!("rgb_to_yuv_nv12 time: {:?}", end_time);
    let start_time = Instant::now();
    // let vega = Vec::from(&bytes_16[bytes_16.len() / 2..(bytes_16.len() / 2 + 15)]);
    // println!("{:?}", vega);
    // yuv_nv12_to_rgb_p16(
    //     &y_nv_plane,
    //     width * 2,
    //     &uv_nv_plane,
    //     width * 2,
    //     &mut bytes_16,
    //     rgba_stride as u32 * 2,
    //     10,
    //     width,
    //     height,
    //     YuvRange::Full,
    //     YuvStandardMatrix::Bt709,
    //     YuvEndianness::BigEndian,
    //     YuvBytesPacking::LeastSignificantBytes,
    // );
    // yuv_nv12_p10_to_rgb(
    //     &y_nv_plane,
    //     width * 2,
    //     &uv_nv_plane,
    //     width * 2,
    //     &mut rgba,
    //     rgba_stride as u32,
    //     width,
    //     height,
    //     YuvRange::Full,
    //     YuvStandardMatrix::Bt709,
    //     YuvEndiannes::BigEndian,
    //     YuvBytesPacking::LeastSignificantBytes,
    // );
    rgba = bytes_16.iter().map(|&x| (x >> 2) as u8).collect();
    //
    let end_time = Instant::now().sub(start_time);
    println!("yuv_nv12_to_rgb time: {:?}", end_time);
    let start_time = Instant::now();
    rgb_to_yuv444(
        &mut y_plane,
        y_stride as u32,
        &mut u_plane,
        u_stride as u32,
        &mut v_plane,
        v_stride as u32,
        &rgba,
        width * components as u32,
        width,
        height,
        YuvRange::TV,
        YuvStandardMatrix::Bt601,
    );

    // let mut y_plane_16 = vec![0u16; width as usize * height as usize];
    // let mut u_plane_16 = vec![0u16; width as usize * height as usize];
    // let mut v_plane_16 = vec![0u16; width as usize * height as usize];
    //
    // let start_time = Instant::now();
    // rgb_to_yuv420_u16(
    //     &mut y_plane_16,
    //     y_stride as u32 * 2,
    //     &mut u_plane_16,
    //     y_stride as u32 * 2,
    //     &mut v_plane_16,
    //     y_stride as u32 * 2,
    //     &bytes_16,
    //     width * components * 2,
    //     10,
    //     width,
    //     height,
    //     YuvRange::TV,
    //     YuvStandardMatrix::Bt709,
    //     YuvEndiannes::BigEndian,
    //     YuvBytesPacking::LeastSignificantBytes,
    // );
    // let end_time = Instant::now().sub(start_time);
    // println!("rgb_to_yuv420_u16 time: {:?}", end_time);
    // let start_time = Instant::now();
    // yuv420_p10_to_rgb(
    //     &y_plane_16,
    //     y_stride as u32 * 2,
    //     &u_plane_16,
    //     y_stride as u32 * 2,
    //     &v_plane_16,
    //     y_stride as u32 * 2,
    //     &mut rgba,
    //     width * components,
    //     width,
    //     height,
    //     YuvRange::TV,
    //     YuvStandardMatrix::Bt709,
    //     YuvEndiannes::BigEndian,
    //     YuvBytesPacking::LeastSignificantBytes,
    // );

    let end_time = Instant::now().sub(start_time);
    println!("Forward time: {:?}", end_time);
    // //
    // let full_size = if width % 2 == 0 {
    //     2 * width as usize * height as usize
    // } else {
    //     2 * (width as usize + 1) * height as usize
    // };
    //
    // // println!("Full YUY2 {}", full_size);
    //
    // let yuy2_stride = if width % 2 == 0 {
    //     2 * width as usize
    // } else {
    //     2 * (width as usize + 1)
    // };
    //
    // let mut yuy2_plane = vec![0u8; full_size];
    // //
    // let start_time = Instant::now();
    // //
    // yuv420_to_yuyv422(
    //     &y_plane,
    //     y_stride as u32,
    //     &u_plane,
    //     y_stride as u32,
    //     &v_plane,
    //     y_stride as u32,
    //     &mut yuy2_plane,
    //     yuy2_stride as u32,
    //     width,
    //     height,
    // );
    // let end_time = Instant::now().sub(start_time);
    // println!("yuv420_to_yuyv422 time: {:?}", end_time);
    // rgba.fill(0);
    // let start_time = Instant::now();
    // yuyv422_to_rgb(
    //     &yuy2_plane,
    //     yuy2_stride as u32,
    //     &mut rgba,
    //     rgba_stride as u32,
    //     width,
    //     height,
    //     YuvRange::TV,
    //     YuvStandardMatrix::Bt709,
    // );
    //
    // let end_time = Instant::now().sub(start_time);
    // println!("yuyv422_to_rgb time: {:?}", end_time);

    // let start_time = Instant::now();
    //
    // yuyv422_to_yuv420(
    //     &mut y_plane,
    //     y_stride as u32,
    //     &mut u_plane,
    //     y_stride as u32,
    //     &mut v_plane,
    //     y_stride as u32,
    //     &yuy2_plane,
    //     yuy2_stride as u32,
    //     width,
    //     height,
    // );
    //
    // let end_time = Instant::now().sub(start_time);
    // println!("yuyv422_to_yuv444 time: {:?}", end_time);
    rgba.fill(0);
    let start_time = Instant::now();
    yuvs::yuv444_to_rgb(
        &y_plane,
        y_stride as usize,
        &u_plane,
        u_stride as usize,
        &v_plane,
        v_stride as usize,
        &mut rgba,
        width as usize,
        height as usize,
        yuvs::YuvRange::Tv,
        yuvs::YuvStandardMatrix::Bt601,
    ).unwrap();
    let end_time = Instant::now().sub(start_time);
    println!("Backward time: {:?}", end_time);

    // rgba = bytes_16.iter().map(|&x| (x >> 2) as u8).collect();

    //
    // let mut gbr = vec![0u8; rgba.len()];
    //
    // let start_time = Instant::now();
    // rgb_to_gbr(
    //     &rgba,
    //     rgba_stride as u32,
    //     &mut gbr,
    //     rgba_stride as u32,
    //     width,
    //     height,
    // );
    //
    // let end_time = Instant::now().sub(start_time);
    // println!("rgb_to_gbr time: {:?}", end_time);
    //
    // let start_time = Instant::now();
    // gbr_to_rgb(
    //     &gbr,
    //     rgba_stride as u32,
    //     &mut rgba,
    //     rgba_stride as u32,
    //     width,
    //     height,
    // );
    // let end_time = Instant::now().sub(start_time);
    // println!("gbr_to_rgb time: {:?}", end_time);
    //
    // rgba = Vec::from(gbr);

    image::save_buffer(
        "converted_sharp.png",
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
