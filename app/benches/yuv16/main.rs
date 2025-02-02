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
#![feature(f16)]
use criterion::{criterion_group, criterion_main, Criterion};
use image::{GenericImageView, ImageReader};
use yuv_sys::{rs_I010ToABGR, rs_I210ToABGR};
use yuvutils_rs::{
    i010_to_rgb10, i010_to_rgba, i010_to_rgba10, i010_to_rgba_f16, i210_to_rgba, i210_to_rgba10,
    i410_to_rgba10, p010_to_rgba10, rgb10_to_i010, rgb10_to_i210, rgb10_to_i410, rgb10_to_p010,
    rgba10_to_i010, rgba10_to_i210, rgba10_to_i410, rgba16_to_i016, YuvBiPlanarImageMut,
    YuvChromaSubsampling, YuvPlanarImageMut, YuvRange, YuvStandardMatrix,
};

pub fn criterion_benchmark(c: &mut Criterion) {
    let img = ImageReader::open("../assets/bench.jpg")
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

    rgb10_to_i010(
        &mut planar_image,
        &src_bytes,
        stride as u32,
        YuvRange::Limited,
        YuvStandardMatrix::Bt601,
    )
    .unwrap();

    rgb10_to_p010(
        &mut bi_planar_image,
        &src_bytes,
        stride as u32,
        YuvRange::Limited,
        YuvStandardMatrix::Bt601,
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
            rgb10_to_i010(
                &mut test_planar,
                &src_bytes,
                stride as u32,
                YuvRange::Limited,
                YuvStandardMatrix::Bt601,
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
            rgba10_to_i010(
                &mut test_planar,
                &rgba_image,
                dimensions.0 * 4,
                YuvRange::Limited,
                YuvStandardMatrix::Bt601,
            )
            .unwrap();
        })
    });

    c.bench_function("yuvutils RGBA16 -> YUV16 4:2:0", |b| {
        let mut test_planar = YuvPlanarImageMut::<u16>::alloc(
            dimensions.0,
            dimensions.1,
            YuvChromaSubsampling::Yuv420,
        );
        b.iter(|| {
            rgba16_to_i016(
                &mut test_planar,
                &rgba_image,
                dimensions.0 * 4,
                YuvRange::Limited,
                YuvStandardMatrix::Bt601,
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
            rgba10_to_i210(
                &mut test_planar,
                &rgba_image,
                dimensions.0 * 4,
                YuvRange::Limited,
                YuvStandardMatrix::Bt601,
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
            rgba10_to_i410(
                &mut test_planar,
                &rgba_image,
                dimensions.0 * 4,
                YuvRange::Limited,
                YuvStandardMatrix::Bt601,
            )
            .unwrap();
        })
    });

    c.bench_function("yuvutils YUV10 NV12 -> RGB10", |b| {
        let mut rgb_bytes = vec![0u16; dimensions.0 as usize * 4 * dimensions.1 as usize];
        b.iter(|| {
            p010_to_rgba10(
                &fixed_bi_planar,
                &mut rgb_bytes,
                dimensions.0 * 4u32,
                YuvRange::Limited,
                YuvStandardMatrix::Bt601,
            )
            .unwrap();
        })
    });

    c.bench_function("yuvutils YUV10 4:2:0 -> RGB10", |b| {
        let mut rgb_bytes = vec![0u16; dimensions.0 as usize * 3 * dimensions.1 as usize];
        b.iter(|| {
            i010_to_rgb10(
                &fixed_planar,
                &mut rgb_bytes,
                dimensions.0 * 3u32,
                YuvRange::Limited,
                YuvStandardMatrix::Bt601,
            )
            .unwrap();
        })
    });

    c.bench_function("yuvutils YUV10 4:2:0 -> RGBA10", |b| {
        let mut rgb_bytes = vec![0u16; dimensions.0 as usize * 4 * dimensions.1 as usize];
        b.iter(|| {
            i010_to_rgba10(
                &fixed_planar,
                &mut rgb_bytes,
                dimensions.0 * 4u32,
                YuvRange::Limited,
                YuvStandardMatrix::Bt601,
            )
            .unwrap();
        })
    });

    c.bench_function("yuvutils YUV10 4:2:0 -> RGBAF16", |b| {
        use core::f16;
        let mut rgb_bytes: Vec<f16> = vec![0.; dimensions.0 as usize * 4 * dimensions.1 as usize];
        b.iter(|| {
            i010_to_rgba_f16(
                &fixed_planar,
                &mut rgb_bytes,
                dimensions.0 * 4u32,
                YuvRange::Limited,
                YuvStandardMatrix::Bt601,
            )
            .unwrap();
        })
    });

    c.bench_function("yuvutils YUV10 4:2:0 -> RGBA8", |b| {
        let mut rgb_bytes = vec![0u8; dimensions.0 as usize * 4 * dimensions.1 as usize];
        b.iter(|| {
            i010_to_rgba(
                &fixed_planar,
                &mut rgb_bytes,
                dimensions.0 * 4u32,
                YuvRange::Limited,
                YuvStandardMatrix::Bt601,
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

    rgb10_to_i210(
        &mut planar_image422,
        &src_bytes,
        stride as u32,
        YuvRange::Limited,
        YuvStandardMatrix::Bt601,
    )
    .unwrap();

    let fixed_planar422 = planar_image422.to_fixed();

    c.bench_function("yuvutils YUV10 4:2:2 -> RGBA10", |b| {
        let mut rgb_bytes = vec![0u16; dimensions.0 as usize * 4 * dimensions.1 as usize];
        b.iter(|| {
            i210_to_rgba10(
                &fixed_planar422,
                &mut rgb_bytes,
                dimensions.0 * 4u32,
                YuvRange::Limited,
                YuvStandardMatrix::Bt601,
            )
            .unwrap();
        })
    });

    c.bench_function("yuvutils YUV10 4:2:2 -> RGBA8", |b| {
        let mut rgb_bytes = vec![0u8; dimensions.0 as usize * 4 * dimensions.1 as usize];
        b.iter(|| {
            i210_to_rgba(
                &fixed_planar422,
                &mut rgb_bytes,
                dimensions.0 * 4u32,
                YuvRange::Limited,
                YuvStandardMatrix::Bt601,
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

    rgb10_to_i410(
        &mut planar_image444,
        &src_bytes,
        stride as u32,
        YuvRange::Limited,
        YuvStandardMatrix::Bt601,
    )
    .unwrap();

    let fixed_planar444 = planar_image444.to_fixed();

    c.bench_function("yuvutils YUV10 4:4:4 -> RGBA10", |b| {
        let mut rgb_bytes = vec![0u16; dimensions.0 as usize * 4 * dimensions.1 as usize];
        b.iter(|| {
            i410_to_rgba10(
                &fixed_planar444,
                &mut rgb_bytes,
                dimensions.0 * 4u32,
                YuvRange::Limited,
                YuvStandardMatrix::Bt601,
            )
            .unwrap();
        })
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
