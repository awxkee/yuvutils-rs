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
use image::{EncodableLayout, GenericImageView, ImageReader};
use yuv_sys::{RotationMode_kRotate180, RotationMode_kRotate270, RotationMode_kRotate90};
use yuvutils_rs::{rotate_plane, rotate_rgba, RotationMode};

pub fn criterion_benchmark(c: &mut Criterion) {
    let img = ImageReader::open("../assets/bench.jpg")
        .unwrap()
        .decode()
        .unwrap();
    let dimensions = img.dimensions();

    let rgba_img = img.to_rgba8();
    let plane_img = img.to_luma8();

    let rgba_src_bytes = rgba_img.as_bytes();
    let plane_src_bytes = plane_img.as_bytes();

    c.bench_function("yuvutils: Rotate 90 RGBA8", |b| {
        let mut rgb_bytes = vec![0u8; dimensions.0 as usize * 4 * dimensions.1 as usize];
        b.iter(|| {
            rotate_rgba(
                &rgba_src_bytes,
                dimensions.0 as usize * 4,
                &mut rgb_bytes,
                4 * dimensions.1 as usize,
                dimensions.0 as usize,
                dimensions.1 as usize,
                RotationMode::Rotate90,
            )
            .unwrap();
        })
    });

    c.bench_function("libyuv: Rotate 90 RGBA8", |b| {
        let mut rgb_bytes = vec![0u8; dimensions.0 as usize * 4 * dimensions.1 as usize];
        b.iter(|| unsafe {
            yuv_sys::rs_ARGBRotate(
                rgba_src_bytes.as_ptr(),
                dimensions.0 as i32 * 4,
                rgb_bytes.as_mut_ptr(),
                4 * dimensions.1 as i32,
                dimensions.0 as i32,
                dimensions.1 as i32,
                RotationMode_kRotate90,
            );
        })
    });

    c.bench_function("yuvutils: Rotate 180 RGBA8", |b| {
        let mut rgb_bytes = vec![0u8; dimensions.0 as usize * 4 * dimensions.1 as usize];
        b.iter(|| {
            rotate_rgba(
                &rgba_src_bytes,
                dimensions.0 as usize * 4,
                &mut rgb_bytes,
                4 * dimensions.0 as usize,
                dimensions.0 as usize,
                dimensions.1 as usize,
                RotationMode::Rotate180,
            )
            .unwrap();
        })
    });

    c.bench_function("libyuv: Rotate 180 RGBA8", |b| {
        let mut rgb_bytes = vec![0u8; dimensions.0 as usize * 4 * dimensions.1 as usize];
        b.iter(|| unsafe {
            yuv_sys::rs_ARGBRotate(
                rgba_src_bytes.as_ptr(),
                dimensions.0 as i32 * 4,
                rgb_bytes.as_mut_ptr(),
                4 * dimensions.0 as i32,
                dimensions.0 as i32,
                dimensions.1 as i32,
                RotationMode_kRotate180,
            );
        })
    });

    c.bench_function("yuvutils: Rotate 90 Plane8", |b| {
        let mut t_bytes = vec![0u8; dimensions.0 as usize * dimensions.1 as usize];
        b.iter(|| {
            rotate_plane(
                &plane_src_bytes,
                dimensions.0 as usize,
                &mut t_bytes,
                dimensions.1 as usize,
                dimensions.0 as usize,
                dimensions.1 as usize,
                RotationMode::Rotate90,
            )
            .unwrap();
        })
    });

    c.bench_function("libyuv: Rotate 90 Plane8", |b| {
        let mut rgb_bytes = vec![0u8; dimensions.0 as usize * dimensions.1 as usize];
        b.iter(|| unsafe {
            yuv_sys::rs_RotatePlane90(
                rgba_src_bytes.as_ptr(),
                dimensions.0 as i32,
                rgb_bytes.as_mut_ptr(),
                dimensions.1 as i32,
                dimensions.0 as i32,
                dimensions.1 as i32,
            );
        })
    });

    c.bench_function("yuvutils: Rotate 180 Plane8", |b| {
        let mut t_bytes = vec![0u8; dimensions.0 as usize * dimensions.1 as usize];
        b.iter(|| {
            rotate_plane(
                &plane_src_bytes,
                dimensions.0 as usize,
                &mut t_bytes,
                dimensions.0 as usize,
                dimensions.0 as usize,
                dimensions.1 as usize,
                RotationMode::Rotate180,
            )
            .unwrap();
        })
    });

    c.bench_function("libyuv: Rotate 180 Plane8", |b| {
        let mut rgb_bytes = vec![0u8; dimensions.0 as usize * dimensions.1 as usize];
        b.iter(|| unsafe {
            yuv_sys::rs_RotatePlane180(
                rgba_src_bytes.as_ptr(),
                dimensions.0 as i32,
                rgb_bytes.as_mut_ptr(),
                dimensions.0 as i32,
                dimensions.0 as i32,
                dimensions.1 as i32,
            );
        })
    });

    c.bench_function("yuvutils: Rotate 270 RGBA8", |b| {
        let mut rgb_bytes = vec![0u8; dimensions.0 as usize * 4 * dimensions.1 as usize];
        b.iter(|| {
            rotate_rgba(
                &rgba_src_bytes,
                dimensions.0 as usize * 4,
                &mut rgb_bytes,
                4 * dimensions.1 as usize,
                dimensions.0 as usize,
                dimensions.1 as usize,
                RotationMode::Rotate270,
            )
            .unwrap();
        })
    });

    c.bench_function("libyuv: Rotate 270 RGBA8", |b| {
        let mut rgb_bytes = vec![0u8; dimensions.0 as usize * 4 * dimensions.1 as usize];
        b.iter(|| unsafe {
            yuv_sys::rs_ARGBRotate(
                rgba_src_bytes.as_ptr(),
                dimensions.0 as i32 * 4,
                rgb_bytes.as_mut_ptr(),
                4 * dimensions.1 as i32,
                dimensions.0 as i32,
                dimensions.1 as i32,
                RotationMode_kRotate270,
            );
        })
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
