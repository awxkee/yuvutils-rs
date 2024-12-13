/*
 * Copyright (c) Radzivon Bartoshyk, 11/2024. All rights reserved.
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
use criterion::{criterion_group, criterion_main, Criterion};
use image::{GenericImageView, ImageReader};
use yuv_sys::{rs_I010ToABGR, rs_I210ToABGR};
use yuvutils_rs::{
    rgb_to_yuv420_p16, rgb_to_yuv422_p16, rgb_to_yuv444_p16, rgb_to_yuv_nv12_p16,
    rgba_to_yuv420_p16, rgba_to_yuv422_p16, rgba_to_yuv444_p16, yuv420_p16_to_rgb,
    yuv420_p16_to_rgb16, yuv420_p16_to_rgba16, yuv422_p16_to_rgba, yuv422_p16_to_rgba16,
    yuv444_p16_to_rgba16, yuv_nv12_to_rgba_p16, YuvBiPlanarImageMut, YuvBytesPacking,
    YuvChromaSubsampling, YuvEndianness, YuvPlanarImageMut, YuvRange, YuvStandardMatrix,
};

pub fn criterion_benchmark(c: &mut Criterion) {
    let img = ImageReader::open("../assets/test_image_2.jpg")
        .unwrap()
        .decode()
        .unwrap();
    let dimensions = img.dimensions();
    let components = 3;
    let stride = dimensions.0 as usize * components;
    let img_16 = img.to_rgb16();
    let src_bytes = img_16.iter().map(|&x| (x >> 6) as u16).collect::<Vec<_>>();

    let mut planar_image =
        YuvPlanarImageMut::<u16>::alloc(dimensions.0, dimensions.1, YuvChromaSubsampling::Yuv420);

    let mut bi_planar_image =
        YuvBiPlanarImageMut::<u16>::alloc(dimensions.0, dimensions.1, YuvChromaSubsampling::Yuv420);

    rgb_to_yuv420_p16(
        &mut planar_image,
        &src_bytes,
        stride as u32,
        10,
        YuvRange::Limited,
        YuvStandardMatrix::Bt601,
        YuvEndianness::LittleEndian,
        YuvBytesPacking::LeastSignificantBytes,
    )
    .unwrap();

    rgb_to_yuv_nv12_p16(
        &mut bi_planar_image,
        &src_bytes,
        stride as u32,
        10,
        YuvRange::Limited,
        YuvStandardMatrix::Bt601,
        YuvEndianness::LittleEndian,
        YuvBytesPacking::LeastSignificantBytes,
    )
    .unwrap();

    let fixed_bi_planar = bi_planar_image.to_fixed();

    let fixed_planar = planar_image.to_fixed();

    let rgba_image = img.to_rgba16().iter().map(|&x| x >> 6).collect::<Vec<_>>();

    c.bench_function("yuvutils RGB10 -> YUV10 4:2:0", |b| {
        let mut test_planar = YuvPlanarImageMut::<u16>::alloc(
            dimensions.0,
            dimensions.1,
            YuvChromaSubsampling::Yuv420,
        );
        b.iter(|| {
            rgb_to_yuv420_p16(
                &mut test_planar,
                &src_bytes,
                stride as u32,
                10,
                YuvRange::Limited,
                YuvStandardMatrix::Bt601,
                YuvEndianness::LittleEndian,
                YuvBytesPacking::LeastSignificantBytes,
            )
            .unwrap();
        })
    });

    c.bench_function("yuvutils RGBA10 -> YUV10 4:2:0", |b| {
        let mut test_planar = YuvPlanarImageMut::<u16>::alloc(
            dimensions.0,
            dimensions.1,
            YuvChromaSubsampling::Yuv420,
        );
        b.iter(|| {
            rgba_to_yuv420_p16(
                &mut test_planar,
                &rgba_image,
                dimensions.0 * 4,
                10,
                YuvRange::Limited,
                YuvStandardMatrix::Bt601,
                YuvEndianness::LittleEndian,
                YuvBytesPacking::LeastSignificantBytes,
            )
            .unwrap();
        })
    });

    c.bench_function("yuvutils RGBA10 -> YUV10 4:2:2", |b| {
        let mut test_planar = YuvPlanarImageMut::<u16>::alloc(
            dimensions.0,
            dimensions.1,
            YuvChromaSubsampling::Yuv422,
        );
        b.iter(|| {
            rgba_to_yuv422_p16(
                &mut test_planar,
                &rgba_image,
                dimensions.0 * 4,
                10,
                YuvRange::Limited,
                YuvStandardMatrix::Bt601,
                YuvEndianness::LittleEndian,
                YuvBytesPacking::LeastSignificantBytes,
            )
            .unwrap();
        })
    });

    c.bench_function("yuvutils RGBA10 -> YUV 4:4:4", |b| {
        let mut test_planar = YuvPlanarImageMut::<u16>::alloc(
            dimensions.0,
            dimensions.1,
            YuvChromaSubsampling::Yuv444,
        );
        b.iter(|| {
            rgba_to_yuv444_p16(
                &mut test_planar,
                &rgba_image,
                dimensions.0 * 4,
                10,
                YuvRange::Limited,
                YuvStandardMatrix::Bt601,
                YuvEndianness::LittleEndian,
                YuvBytesPacking::LeastSignificantBytes,
            )
            .unwrap();
        })
    });

    c.bench_function("yuvutils YUV10 NV12 -> RGB10", |b| {
        let mut rgb_bytes = vec![0u16; dimensions.0 as usize * 4 * dimensions.1 as usize];
        b.iter(|| {
            yuv_nv12_to_rgba_p16(
                &fixed_bi_planar,
                &mut rgb_bytes,
                dimensions.0 * 4u32,
                10,
                YuvRange::Limited,
                YuvStandardMatrix::Bt601,
                YuvEndianness::LittleEndian,
                YuvBytesPacking::LeastSignificantBytes,
            )
            .unwrap();
        })
    });

    c.bench_function("yuvutils YUV10 4:2:0 -> RGB10", |b| {
        let mut rgb_bytes = vec![0u16; dimensions.0 as usize * 3 * dimensions.1 as usize];
        b.iter(|| {
            yuv420_p16_to_rgb16(
                &fixed_planar,
                &mut rgb_bytes,
                dimensions.0 * 3u32,
                10,
                YuvRange::Limited,
                YuvStandardMatrix::Bt601,
                YuvEndianness::LittleEndian,
                YuvBytesPacking::LeastSignificantBytes,
            )
            .unwrap();
        })
    });

    c.bench_function("yuvutils YUV10 4:2:0 -> RGBA10", |b| {
        let mut rgb_bytes = vec![0u16; dimensions.0 as usize * 4 * dimensions.1 as usize];
        b.iter(|| {
            yuv420_p16_to_rgba16(
                &fixed_planar,
                &mut rgb_bytes,
                dimensions.0 * 4u32,
                10,
                YuvRange::Limited,
                YuvStandardMatrix::Bt601,
                YuvEndianness::LittleEndian,
                YuvBytesPacking::LeastSignificantBytes,
            )
            .unwrap();
        })
    });

    c.bench_function("yuvutils YUV10 4:2:0 -> RGBA8", |b| {
        let mut rgb_bytes = vec![0u8; dimensions.0 as usize * 4 * dimensions.1 as usize];
        b.iter(|| {
            yuv420_p16_to_rgb(
                &fixed_planar,
                &mut rgb_bytes,
                dimensions.0 * 4u32,
                10,
                YuvRange::Limited,
                YuvStandardMatrix::Bt601,
                YuvEndianness::LittleEndian,
                YuvBytesPacking::LeastSignificantBytes,
            )
            .unwrap();
        })
    });

    c.bench_function("libyuv YUV10 4:2:0 -> RGBA8", |b| {
        let mut rgb_bytes = vec![0u8; dimensions.0 as usize * 4 * dimensions.1 as usize];
        b.iter(|| unsafe {
            rs_I010ToABGR(
                fixed_planar.y_plane.as_ptr(),
                fixed_planar.y_stride as i32,
                fixed_planar.u_plane.as_ptr(),
                fixed_planar.u_stride as i32,
                fixed_planar.v_plane.as_ptr(),
                fixed_planar.v_stride as i32,
                rgb_bytes.as_mut_ptr(),
                dimensions.0 as i32 * 4i32,
                fixed_planar.width as i32,
                fixed_planar.height as i32,
            );
        })
    });

    let mut planar_image422 =
        YuvPlanarImageMut::<u16>::alloc(dimensions.0, dimensions.1, YuvChromaSubsampling::Yuv422);

    rgb_to_yuv422_p16(
        &mut planar_image422,
        &src_bytes,
        stride as u32,
        10,
        YuvRange::Limited,
        YuvStandardMatrix::Bt601,
        YuvEndianness::LittleEndian,
        YuvBytesPacking::LeastSignificantBytes,
    )
    .unwrap();

    let fixed_planar422 = planar_image422.to_fixed();

    c.bench_function("yuvutils YUV10 4:2:2 -> RGBA10", |b| {
        let mut rgb_bytes = vec![0u16; dimensions.0 as usize * 4 * dimensions.1 as usize];
        b.iter(|| {
            yuv422_p16_to_rgba16(
                &fixed_planar422,
                &mut rgb_bytes,
                dimensions.0 * 4u32,
                10,
                YuvRange::Limited,
                YuvStandardMatrix::Bt601,
                YuvEndianness::LittleEndian,
                YuvBytesPacking::LeastSignificantBytes,
            )
            .unwrap();
        })
    });

    c.bench_function("yuvutils YUV10 4:2:2 -> RGBA8", |b| {
        let mut rgb_bytes = vec![0u8; dimensions.0 as usize * 4 * dimensions.1 as usize];
        b.iter(|| {
            yuv422_p16_to_rgba(
                &fixed_planar422,
                &mut rgb_bytes,
                dimensions.0 * 4u32,
                10,
                YuvRange::Limited,
                YuvStandardMatrix::Bt601,
                YuvEndianness::LittleEndian,
                YuvBytesPacking::LeastSignificantBytes,
            )
            .unwrap();
        })
    });

    c.bench_function("libyuv YUV10 4:2:2 -> RGBA8", |b| {
        let mut rgb_bytes = vec![0u8; dimensions.0 as usize * 4 * dimensions.1 as usize];
        b.iter(|| unsafe {
            rs_I210ToABGR(
                fixed_planar422.y_plane.as_ptr(),
                fixed_planar422.y_stride as i32,
                fixed_planar422.u_plane.as_ptr(),
                fixed_planar422.u_stride as i32,
                fixed_planar422.v_plane.as_ptr(),
                fixed_planar422.v_stride as i32,
                rgb_bytes.as_mut_ptr(),
                dimensions.0 as i32 * 4i32,
                fixed_planar422.width as i32,
                fixed_planar422.height as i32,
            );
        })
    });

    let mut planar_image444 =
        YuvPlanarImageMut::<u16>::alloc(dimensions.0, dimensions.1, YuvChromaSubsampling::Yuv444);

    rgb_to_yuv444_p16(
        &mut planar_image444,
        &src_bytes,
        stride as u32,
        10,
        YuvRange::Limited,
        YuvStandardMatrix::Bt601,
        YuvEndianness::LittleEndian,
        YuvBytesPacking::LeastSignificantBytes,
    )
    .unwrap();

    let fixed_planar444 = planar_image444.to_fixed();

    c.bench_function("yuvutils YUV10 4:4:4 -> RGBA10", |b| {
        let mut rgb_bytes = vec![0u16; dimensions.0 as usize * 4 * dimensions.1 as usize];
        b.iter(|| {
            yuv444_p16_to_rgba16(
                &fixed_planar444,
                &mut rgb_bytes,
                dimensions.0 * 4u32,
                10,
                YuvRange::Limited,
                YuvStandardMatrix::Bt601,
                YuvEndianness::LittleEndian,
                YuvBytesPacking::LeastSignificantBytes,
            )
            .unwrap();
        })
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
