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
use yuv_sys::{
    rs_ABGRToI420, rs_ABGRToJ422, rs_I420ToABGR, rs_I420ToRGB24, rs_I422ToABGR, rs_I444ToABGR,
    rs_NV21ToABGR, rs_RGB24ToI420,
};
use yuvutils_rs::{
    rgb_to_yuv420, rgb_to_yuv422, rgb_to_yuv444, rgb_to_yuv_nv12, rgba_to_yuv420, rgba_to_yuv422,
    rgba_to_yuv444, yuv420_to_rgb, yuv420_to_rgba, yuv422_to_rgba, yuv444_to_rgba,
    yuv_nv12_to_rgba, YuvBiPlanarImageMut, YuvChromaSubsampling, YuvPlanarImageMut, YuvRange,
    YuvStandardMatrix,
};

pub fn criterion_benchmark(c: &mut Criterion) {
    let img = ImageReader::open("../assets/test_image_2.jpg")
        .unwrap()
        .decode()
        .unwrap();
    let dimensions = img.dimensions();
    let components = 3;
    let stride = dimensions.0 as usize * components;
    let src_bytes = img.as_bytes();

    let mut planar_image =
        YuvPlanarImageMut::<u8>::alloc(dimensions.0, dimensions.1, YuvChromaSubsampling::Yuv420);

    let mut bi_planar_image =
        YuvBiPlanarImageMut::<u8>::alloc(dimensions.0, dimensions.1, YuvChromaSubsampling::Yuv420);

    rgb_to_yuv420(
        &mut planar_image,
        &src_bytes,
        stride as u32,
        YuvRange::Limited,
        YuvStandardMatrix::Bt601,
    )
    .unwrap();

    rgb_to_yuv_nv12(
        &mut bi_planar_image,
        &src_bytes,
        stride as u32,
        YuvRange::Limited,
        YuvStandardMatrix::Bt601,
    )
    .unwrap();

    let fixed_bi_planar = bi_planar_image.to_fixed();

    let fixed_planar = planar_image.to_fixed();

    let rgba_image = img.to_rgba8();

    c.bench_function("yuvutils RGB -> YUV 4:2:0", |b| {
        let mut test_planar = YuvPlanarImageMut::<u8>::alloc(
            dimensions.0,
            dimensions.1,
            YuvChromaSubsampling::Yuv420,
        );
        b.iter(|| {
            rgb_to_yuv420(
                &mut test_planar,
                &src_bytes,
                stride as u32,
                YuvRange::Limited,
                YuvStandardMatrix::Bt601,
            )
            .unwrap();
        })
    });

    c.bench_function("libyuv RGB -> YUV 4:2:0", |b| {
        let mut test_planar = YuvPlanarImageMut::<u8>::alloc(
            dimensions.0,
            dimensions.1,
            YuvChromaSubsampling::Yuv420,
        );
        b.iter(|| unsafe {
            rs_RGB24ToI420(
                src_bytes.as_ptr(),
                stride as i32,
                test_planar.y_plane.borrow_mut().as_mut_ptr(),
                test_planar.y_stride as i32,
                test_planar.u_plane.borrow_mut().as_mut_ptr(),
                test_planar.u_stride as i32,
                test_planar.v_plane.borrow_mut().as_mut_ptr(),
                test_planar.v_stride as i32,
                test_planar.width as i32,
                test_planar.height as i32,
            );
        })
    });

    c.bench_function("yuvutils RGBA -> YUV 4:2:0", |b| {
        let mut test_planar = YuvPlanarImageMut::<u8>::alloc(
            dimensions.0,
            dimensions.1,
            YuvChromaSubsampling::Yuv420,
        );
        b.iter(|| {
            rgba_to_yuv420(
                &mut test_planar,
                &rgba_image,
                dimensions.0 * 4,
                YuvRange::Limited,
                YuvStandardMatrix::Bt601,
            )
            .unwrap();
        })
    });

    c.bench_function("libyuv RGBA -> YUV 4:2:0", |b| {
        let mut test_planar = YuvPlanarImageMut::<u8>::alloc(
            dimensions.0,
            dimensions.1,
            YuvChromaSubsampling::Yuv420,
        );
        b.iter(|| unsafe {
            rs_ABGRToI420(
                rgba_image.as_ptr(),
                dimensions.0 as i32 * 4i32,
                test_planar.y_plane.borrow_mut().as_mut_ptr(),
                test_planar.y_stride as i32,
                test_planar.u_plane.borrow_mut().as_mut_ptr(),
                test_planar.u_stride as i32,
                test_planar.v_plane.borrow_mut().as_mut_ptr(),
                test_planar.v_stride as i32,
                test_planar.width as i32,
                test_planar.height as i32,
            );
        })
    });

    c.bench_function("yuvutils RGBA -> YUV 4:2:2", |b| {
        let mut test_planar = YuvPlanarImageMut::<u8>::alloc(
            dimensions.0,
            dimensions.1,
            YuvChromaSubsampling::Yuv422,
        );
        b.iter(|| {
            rgba_to_yuv422(
                &mut test_planar,
                &rgba_image,
                dimensions.0 * 4,
                YuvRange::Limited,
                YuvStandardMatrix::Bt601,
            )
            .unwrap();
        })
    });

    c.bench_function("libyuv RGBA -> YUV 4:2:2", |b| {
        let mut test_planar = YuvPlanarImageMut::<u8>::alloc(
            dimensions.0,
            dimensions.1,
            YuvChromaSubsampling::Yuv422,
        );
        b.iter(|| unsafe {
            rs_ABGRToJ422(
                rgba_image.as_ptr(),
                dimensions.0 as i32 * 4i32,
                test_planar.y_plane.borrow_mut().as_mut_ptr(),
                test_planar.y_stride as i32,
                test_planar.u_plane.borrow_mut().as_mut_ptr(),
                test_planar.u_stride as i32,
                test_planar.v_plane.borrow_mut().as_mut_ptr(),
                test_planar.v_stride as i32,
                test_planar.width as i32,
                test_planar.height as i32,
            );
        })
    });

    c.bench_function("yuvutils RGBA -> YUV 4:4:4", |b| {
        let mut test_planar = YuvPlanarImageMut::<u8>::alloc(
            dimensions.0,
            dimensions.1,
            YuvChromaSubsampling::Yuv444,
        );
        b.iter(|| {
            rgba_to_yuv444(
                &mut test_planar,
                &rgba_image,
                dimensions.0 * 4,
                YuvRange::Limited,
                YuvStandardMatrix::Bt601,
            )
            .unwrap();
        })
    });

    c.bench_function("yuvutils YUV NV12 -> RGB", |b| {
        let mut rgb_bytes = vec![0u8; dimensions.0 as usize * 4 * dimensions.1 as usize];
        b.iter(|| {
            yuv_nv12_to_rgba(
                &fixed_bi_planar,
                &mut rgb_bytes,
                dimensions.0 * 4u32,
                YuvRange::Limited,
                YuvStandardMatrix::Bt601,
            )
            .unwrap();
        })
    });
    //
    c.bench_function("libyuv YUV NV12 -> RGB", |b| {
        let mut rgb_bytes = vec![0u8; dimensions.0 as usize * 4 * dimensions.1 as usize];
        b.iter(|| unsafe {
            rs_NV21ToABGR(
                fixed_bi_planar.y_plane.as_ptr(),
                fixed_bi_planar.y_stride as i32,
                fixed_bi_planar.uv_plane.as_ptr(),
                fixed_bi_planar.uv_stride as i32,
                rgb_bytes.as_mut_ptr(),
                dimensions.0 as i32 * 4,
                fixed_bi_planar.width as i32,
                fixed_bi_planar.height as i32,
            );
        })
    });

    c.bench_function("yuvutils YUV 4:2:0 -> RGB", |b| {
        let mut rgb_bytes = vec![0u8; dimensions.0 as usize * 3 * dimensions.1 as usize];
        b.iter(|| {
            yuv420_to_rgb(
                &fixed_planar,
                &mut rgb_bytes,
                dimensions.0 * 3u32,
                YuvRange::Limited,
                YuvStandardMatrix::Bt601,
            )
            .unwrap();
        })
    });

    c.bench_function("libyuv YUV 4:2:0 -> BGR24", |b| {
        let mut rgb_bytes = vec![0u8; dimensions.0 as usize * 3 * dimensions.1 as usize];
        b.iter(|| unsafe {
            rs_I420ToRGB24(
                fixed_planar.y_plane.as_ptr(),
                fixed_planar.y_stride as i32,
                fixed_planar.u_plane.as_ptr(),
                fixed_planar.u_stride as i32,
                fixed_planar.v_plane.as_ptr(),
                fixed_planar.v_stride as i32,
                rgb_bytes.as_mut_ptr(),
                dimensions.0 as i32 * 3i32,
                fixed_planar.width as i32,
                fixed_planar.height as i32,
            );
        })
    });

    c.bench_function("yuvutils YUV 4:2:0 -> RGBA", |b| {
        let mut rgb_bytes = vec![0u8; dimensions.0 as usize * 4 * dimensions.1 as usize];
        b.iter(|| {
            yuv420_to_rgba(
                &fixed_planar,
                &mut rgb_bytes,
                dimensions.0 * 4u32,
                YuvRange::Limited,
                YuvStandardMatrix::Bt601,
            )
            .unwrap();
        })
    });

    c.bench_function("libyuv YUV 4:2:0 -> RGBA", |b| {
        let mut rgb_bytes = vec![0u8; dimensions.0 as usize * 4 * dimensions.1 as usize];
        b.iter(|| unsafe {
            rs_I420ToABGR(
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
        YuvPlanarImageMut::<u8>::alloc(dimensions.0, dimensions.1, YuvChromaSubsampling::Yuv422);

    rgb_to_yuv422(
        &mut planar_image422,
        &src_bytes,
        stride as u32,
        YuvRange::Limited,
        YuvStandardMatrix::Bt601,
    )
    .unwrap();

    let fixed_planar422 = planar_image422.to_fixed();

    c.bench_function("yuvutils YUV 4:2:2 -> RGBA", |b| {
        let mut rgb_bytes = vec![0u8; dimensions.0 as usize * 4 * dimensions.1 as usize];
        b.iter(|| {
            yuv422_to_rgba(
                &fixed_planar422,
                &mut rgb_bytes,
                dimensions.0 * 4u32,
                YuvRange::Limited,
                YuvStandardMatrix::Bt601,
            )
            .unwrap();
        })
    });

    c.bench_function("libyuv YUV 4:2:2 -> RGBA", |b| {
        let mut rgb_bytes = vec![0u8; dimensions.0 as usize * 4 * dimensions.1 as usize];
        b.iter(|| unsafe {
            rs_I422ToABGR(
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
        YuvPlanarImageMut::<u8>::alloc(dimensions.0, dimensions.1, YuvChromaSubsampling::Yuv444);

    rgb_to_yuv444(
        &mut planar_image444,
        &src_bytes,
        stride as u32,
        YuvRange::Limited,
        YuvStandardMatrix::Bt601,
    )
    .unwrap();

    let fixed_planar444 = planar_image444.to_fixed();

    c.bench_function("yuvutils YUV 4:4:4 -> RGBA", |b| {
        let mut rgb_bytes = vec![0u8; dimensions.0 as usize * 4 * dimensions.1 as usize];
        b.iter(|| {
            yuv444_to_rgba(
                &fixed_planar444,
                &mut rgb_bytes,
                dimensions.0 * 4u32,
                YuvRange::Limited,
                YuvStandardMatrix::Bt601,
            )
            .unwrap();
        })
    });

    c.bench_function("libyuv YUV 4:4:4 -> RGBA", |b| {
        let mut rgb_bytes = vec![0u8; dimensions.0 as usize * 4 * dimensions.1 as usize];
        b.iter(|| unsafe {
            rs_I444ToABGR(
                fixed_planar444.y_plane.as_ptr(),
                fixed_planar444.y_stride as i32,
                fixed_planar444.u_plane.as_ptr(),
                fixed_planar444.u_stride as i32,
                fixed_planar444.v_plane.as_ptr(),
                fixed_planar444.v_stride as i32,
                rgb_bytes.as_mut_ptr(),
                dimensions.0 as i32 * 4i32,
                fixed_planar444.width as i32,
                fixed_planar444.height as i32,
            );
        })
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
