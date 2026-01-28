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
mod bilinear;
mod support;

use image::{ColorType, DynamicImage, EncodableLayout, GenericImageView, ImageReader};
use std::fs::File;
use std::io::Read;
use std::time::Instant;
use yuv::{
    i010_alpha_to_rgba10, i010_to_rgb10_bilinear, i010_to_rgba10, i010_to_rgba10_bilinear,
    i210_to_rgba10, i210_to_rgba10_bilinear, icgc_re010_to_rgba, icgc_ro010_to_rgba,
    icgc_ro210_to_rgba, icgc_ro410_to_rgba, p010_to_rgba10, rgb10_to_i010, rgb10_to_i410,
    rgb10_to_p010, rgb_to_yuv420, rgb_to_yuv422, rgba10_to_i010, rgba10_to_i210, rgba10_to_p010,
    rgba12_to_i412, rgba_to_icgc_re010, rgba_to_icgc_ro010, rgba_to_icgc_ro210, rgba_to_icgc_ro410,
    rgba_to_ycgco420, rgba_to_ycgco444, rgba_to_yuv420, rgba_to_yuv422, rgba_to_yuv444,
    rgba_to_yuv_nv12, rgba_to_yuv_nv16, rgba_to_yuv_nv24, ycgco420_to_rgba, ycgco444_to_rgba,
    yuv420_alpha_to_rgba, yuv420_to_rgb, yuv420_to_rgb_bilinear, yuv420_to_rgba,
    yuv420_to_rgba_bilinear, yuv422_to_rgb_bilinear, yuv422_to_rgba, yuv422_to_rgba_bilinear,
    yuv444_to_rgba, yuv_nv12_to_rgba, yuv_nv16_to_rgba, yuv_nv24_to_rgba, YuvBiPlanarImageMut,
    YuvChromaSubsampling, YuvConversionMode, YuvPlanarImage, YuvPlanarImageMut,
    YuvPlanarImageWithAlpha, YuvRange, YuvStandardMatrix,
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
    let mut img = ImageReader::open("./assets/bench.png")
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

    let mut planar_image =
        YuvPlanarImageMut::<u8>::alloc(width as u32, height as u32, YuvChromaSubsampling::Yuv420);

    // let mut bi_planar_image = YuvBiPlanarImageMut::<u16>::alloc(
    //     width as u32,
    //     height as u32,
    //     YuvChromaSubsampling::Yuv420,
    // );

    // let mut bytes_16: Vec<u16> = src_bytes
    //     .iter()
    //     .map(|&x| ((x as u16) << 2) | ((x as u16) >> 6))
    //     .collect();

    let start_time = Instant::now();

    rgb_to_yuv420(
        &mut planar_image,
        &src_bytes,
        rgba_stride as u32,
        YuvRange::Limited,
        YuvStandardMatrix::Bt709,
        YuvConversionMode::Balanced,
    )
    .unwrap();

    println!("Forward time: {:?}", start_time.elapsed());
    let fixed = planar_image.to_fixed();
    rgba.fill(255);

    let start_time = Instant::now();
    yuv420_to_rgb(
        &fixed,
        &mut rgba,
        rgba_stride as u32,
        YuvRange::Limited,
        YuvStandardMatrix::Bt709,
    )
    .unwrap();
    println!("Backward time: {:?}", start_time.elapsed());

    // let fixed_biplanar = bi_planar_image.to_fixed();
    let fixed_planar = planar_image.to_fixed();
    // // bytes_16.fill(0);
    //
    // let mut j_rgba = vec![0u8; dimensions.0 as usize * dimensions.1 as usize * 4];

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

    // rgba = rgba_f16.iter().map(|&x| (x as f32 * 255.) as u8).collect();

    let mut img = ImageReader::open("./converted_sharp151_x86o.png")
        .unwrap()
        .decode()
        .unwrap();
    let img0 = DynamicImage::ImageRgb8(img.to_rgb8());
    for (y, (row_src, row_ref)) in rgba
        .chunks_exact(dimensions.0 as usize)
        .zip(img0.as_bytes().chunks_exact(dimensions.0 as usize))
        .enumerate()
    {
        for (x, (src, src_ref)) in row_src
            .chunks_exact(3)
            .zip(row_ref.chunks_exact(3))
            .enumerate()
        {
            if src[0] != src_ref[0] {
                // panic!(
                //     "disconvergence r on ({x}, {y}) vals {} vs {}",
                //     src[0], src_ref[0]
                // );
            }
        }
    }

    image::save_buffer(
        "converted_sharp151_x86.png",
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
