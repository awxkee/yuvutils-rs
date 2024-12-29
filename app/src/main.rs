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
use std::cmp::max;
use std::fs::File;
use std::io::Read;
use std::thread::available_parallelism;
use std::time::Instant;
use yuv_sys::{rs_I420ToRGB24, rs_NV12ToRGB24, rs_NV21ToABGR, rs_NV21ToRGB24, rs_RGB24ToI420};
use yuvutils_rs::{
    bgra_to_rgba, gbr_to_rgb, rgb_to_bgra, rgb_to_gbr, rgb_to_sharp_yuv420, rgb_to_yuv400,
    rgb_to_yuv420, rgb_to_yuv420_p16, rgb_to_yuv422, rgb_to_yuv422_p16, rgb_to_yuv444,
    rgb_to_yuv444_p16, rgb_to_yuv_nv12, rgb_to_yuv_nv12_p16, rgb_to_yuv_nv16, rgb_to_yuv_nv24,
    rgba_to_bgra, rgba_to_yuv422, yuv400_to_rgb, yuv420_p16_to_rgb, yuv420_p16_to_rgb16,
    yuv420_to_rgb, yuv420_to_yuyv422, yuv422_p16_to_rgb, yuv422_p16_to_rgb16, yuv422_to_rgb,
    yuv422_to_rgba, yuv444_p16_to_rgb16, yuv444_to_rgb, yuv_nv12_to_rgb, yuv_nv12_to_rgb_p16,
    yuv_nv12_to_rgba, yuv_nv12_to_rgba_p16, yuv_nv16_to_rgb, yuv_nv24_to_rgb, yuyv422_to_rgb,
    yuyv422_to_yuv420, SharpYuvGammaTransfer, YuvBiPlanarImageMut, YuvBytesPacking,
    YuvChromaSubsampling, YuvEndianness, YuvGrayImageMut, YuvPackedImage, YuvPackedImageMut,
    YuvPlanarImageMut, YuvRange, YuvStandardMatrix,
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

    let mut bi_planar_image =
        YuvBiPlanarImageMut::<u8>::alloc(width as u32, height as u32, YuvChromaSubsampling::Yuv422);

    let mut planar_image =
        YuvPlanarImageMut::<u8>::alloc(width as u32, height as u32, YuvChromaSubsampling::Yuv422);
    //
    // let mut bytes_16: Vec<u16> = src_bytes.iter().map(|&x| (x as u16) << 2).collect();

    let start_time = Instant::now();
    rgb_to_yuv_nv16(
        &mut bi_planar_image,
        &src_bytes,
        rgba_stride as u32,
        YuvRange::Full,
        YuvStandardMatrix::Bt601,
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

    let mut ar30 = vec![0u32; width as usize * height as usize];

    // rgb8_to_ra30(
    //     &mut ar30,
    //     width,
    //     Rgb30ByteOrder::Host,
    //     &src_bytes,
    //     rgba_stride as u32,
    //     width,
    //     height,
    // )
    // .unwrap();

    // yuv422_p16_to_ab30(
    //     &fixed_planar,
    //     &mut ar30,
    //     width,
    //     Rgb30ByteOrder::Host,
    //     10,
    //     YuvRange::Limited,
    //     YuvStandardMatrix::Bt601,
    //     YuvEndianness::LittleEndian,
    //     YuvBytesPacking::LeastSignificantBytes,
    // )
    // .unwrap();
    // rgba.fill(0);
    // ra30_to_rgb8(
    //     &ar30,
    //     width,
    //     Rgb30ByteOrder::Host,
    //     &mut rgba,
    //     rgba_stride as u32,
    //     width,
    //     height,
    // )
    // .unwrap();
    let start_time = Instant::now();

    // let components = 4;
    // let rgba_stride = width as usize * 4;
    // let mut rgba = vec![0u8; height as usize * rgba_stride];

    // bytes_16.fill(0);

    yuv_nv16_to_rgb(
        &fixed_biplanar,
        &mut rgba,
        rgba_stride as u32,
        YuvRange::Full,
        YuvStandardMatrix::Bt601,
    )
    .unwrap();

    println!("Backward time: {:?}", start_time.elapsed());

    let start_time = Instant::now();

    // unsafe {
    //     let mut planar_image =
    //         YuvPlanarImageMut::<u8>::alloc(width as u32, height as u32, YuvChromaSubsampling::Yuv420);
    //
    //     let mut source_bgr = vec![0u8; src_bytes.len()];
    //
    //     rgba.chunks_exact(3).zip(source_bgr.chunks_exact_mut(3)).for_each(|(src, dst)| {
    //         let b = src[0];
    //         dst[0] = src[2];
    //         dst[1] = src[1];
    //         dst[2] = b;
    //     });
    //
    //     rs_RGB24ToI420(
    //         src_bytes.as_ptr(),
    //         rgba_stride as i32,
    //         planar_image.y_plane.borrow_mut().as_mut_ptr(),
    //         planar_image.y_stride as i32,
    //         planar_image.u_plane.borrow_mut().as_mut_ptr(),
    //         planar_image.u_stride as i32,
    //         planar_image.v_plane.borrow_mut().as_mut_ptr(),
    //         planar_image.v_stride as i32,
    //         dimensions.0 as i32,
    //         dimensions.1 as i32,
    //     );
    //     let fixed_planar = planar_image.to_fixed();
    //     rs_I420ToRGB24(
    //         fixed_planar.y_plane.as_ptr(),
    //         fixed_planar.y_stride as i32,
    //         fixed_planar.u_plane.as_ptr(),
    //         fixed_planar.u_stride as i32,
    //         fixed_planar.v_plane.as_ptr(),
    //         fixed_planar.v_stride as i32,
    //         rgba.as_mut_ptr(),
    //         rgba_stride as i32,
    //         fixed_planar.width as i32,
    //         fixed_planar.height as i32,
    //     );
    //
    //     // rgba.chunks_exact_mut(3).for_each(|chunk| {
    //     //     let b = chunk[0];
    //     //     chunk[0] = chunk[2];
    //     //     chunk[2] = b;
    //     // });
    // //     rs_NV12ToRGB24(
    // //         fixed_biplanar.y_plane.as_ptr(),
    // //         fixed_biplanar.y_stride as i32,
    // //         fixed_biplanar.uv_plane.as_ptr(),
    // //         fixed_biplanar.uv_stride as i32,
    // //         rgba.as_mut_ptr(),
    // //         rgba_stride as i32,
    // //         fixed_planar.width as i32,
    // //         fixed_planar.height as i32,
    // //     );
    // }

    // /    println!("Backward LIBYUV time: {:?}", start_time.elapsed());

    // rgba = bytes_16.iter().map(|&x| (x >> 2) as u8).collect();

    image::save_buffer(
        "converted_sharp15.png",
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
