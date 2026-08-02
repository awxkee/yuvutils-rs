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
use yuv::{
    i010_alpha_to_rgba10, i010_to_rgb10, i010_to_rgb10_bilinear, i010_to_rgba10,
    i010_to_rgba10_bilinear, i012_to_rgba12, i012_to_rgba12_bilinear, i016_to_rgba16,
    i210_alpha_to_rgba10, i210_to_rgb10, i210_to_rgb10_bilinear, i210_to_rgba10,
    i210_to_rgba10_bilinear, i212_to_rgba12, i212_to_rgba12_bilinear, i216_to_rgba16,
    i410_alpha_to_rgba10, i410_to_rgb10, i410_to_rgba10, i412_to_rgba12, i416_to_rgba16,
    YuvPlanarImage, YuvPlanarImageWithAlpha, YuvRange, YuvStandardMatrix,
};

fuzz_target!(|data: (u8, u8, u8, u8, u8, u8)| {
    fuzz_yuv_420(
        data.0,
        data.1,
        data.2 as u16,
        data.3 as u16,
        data.4 as u16,
        data.5,
    );
    fuzz_yuv_422(
        data.0,
        data.1,
        data.2 as u16,
        data.3 as u16,
        data.4 as u16,
        data.5,
    );
    fuzz_yuv_444(data.0, data.1, data.2 as u16, data.3 as u16, data.4 as u16);
});

fn assert_rgba_alpha(rgba: &[u16], stride: usize, width: usize, height: usize, alpha: u16) {
    for row in 0..height {
        assert!(rgba[row * stride..row * stride + width * 4]
            .chunks_exact(4)
            .all(|pixel| pixel[3] == alpha));
    }
}

fn fuzz_yuv_420(
    i_width: u8,
    i_height: u8,
    y_value: u16,
    u_value: u16,
    v_value: u16,
    stride_seed: u8,
) {
    if i_height == 0 || i_width == 0 {
        return;
    }
    let width = i_width as usize;
    let height = i_height as usize;
    let chroma_width = width.div_ceil(2);
    let chroma_height = height.div_ceil(2);
    let y_stride = width + (stride_seed & 3) as usize;
    let u_stride = chroma_width + ((stride_seed >> 2) & 3) as usize;
    let v_stride = chroma_width + ((stride_seed >> 4) & 3) as usize;
    let dst_padding = ((stride_seed >> 6) & 3) as usize;
    let y_plane = vec![y_value; y_stride * (height - 1) + width];
    let u_plane = vec![u_value; u_stride * (chroma_height - 1) + chroma_width];
    let v_plane = vec![v_value; v_stride * (chroma_height - 1) + chroma_width];
    let tight_y_plane = vec![y_value; width * height];
    let tight_a_plane = vec![y_value; width * height];
    let tight_u_plane = vec![u_value; chroma_width * chroma_height];
    let tight_v_plane = vec![v_value; chroma_width * chroma_height];

    let bilinear_image = YuvPlanarImage {
        y_plane: &y_plane,
        y_stride: y_stride as u32,
        u_plane: &u_plane,
        u_stride: u_stride as u32,
        v_plane: &v_plane,
        v_stride: v_stride as u32,
        width: i_width as u32,
        height: i_height as u32,
    };
    let tight_image = YuvPlanarImage {
        y_plane: &tight_y_plane,
        y_stride: i_width as u32,
        u_plane: &tight_u_plane,
        u_stride: chroma_width as u32,
        v_plane: &tight_v_plane,
        v_stride: chroma_width as u32,
        width: i_width as u32,
        height: i_height as u32,
    };

    let rgb_width = width * 3;
    let rgb_stride = rgb_width + dst_padding;
    let mut target_rgb = vec![0u16; rgb_width * height];
    let mut bilinear_rgb = vec![0u16; rgb_stride * (height - 1) + rgb_width];

    i010_to_rgb10(
        &tight_image,
        &mut target_rgb,
        rgb_width as u32,
        YuvRange::Limited,
        YuvStandardMatrix::Bt601,
    )
    .unwrap();

    i010_to_rgb10_bilinear(
        &bilinear_image,
        &mut bilinear_rgb,
        rgb_stride as u32,
        YuvRange::Limited,
        YuvStandardMatrix::Bt601,
    )
    .unwrap();

    let rgba_width = width * 4;
    let rgba_stride = rgba_width + dst_padding;
    let mut target_rgba = vec![0u16; rgba_width * height];
    let mut bilinear_rgba = vec![0u16; rgba_stride * (height - 1) + rgba_width];

    i010_to_rgba10(
        &tight_image,
        &mut target_rgba,
        rgba_width as u32,
        YuvRange::Limited,
        YuvStandardMatrix::Bt601,
    )
    .unwrap();

    i010_to_rgba10_bilinear(
        &bilinear_image,
        &mut bilinear_rgba,
        rgba_stride as u32,
        YuvRange::Limited,
        YuvStandardMatrix::Bt601,
    )
    .unwrap();
    assert_rgba_alpha(&bilinear_rgba, rgba_stride, width, height, 1023);

    i012_to_rgba12(
        &tight_image,
        &mut target_rgba,
        rgba_width as u32,
        YuvRange::Limited,
        YuvStandardMatrix::Bt601,
    )
    .unwrap();

    bilinear_rgba.fill(0);
    i012_to_rgba12_bilinear(
        &bilinear_image,
        &mut bilinear_rgba,
        rgba_stride as u32,
        YuvRange::Limited,
        YuvStandardMatrix::Bt601,
    )
    .unwrap();
    assert_rgba_alpha(&bilinear_rgba, rgba_stride, width, height, 4095);

    i016_to_rgba16(
        &tight_image,
        &mut target_rgba,
        rgba_width as u32,
        YuvRange::Limited,
        YuvStandardMatrix::Bt601,
    )
    .unwrap();

    let planar_image_alpha = YuvPlanarImageWithAlpha {
        y_plane: &tight_y_plane,
        y_stride: i_width as u32,
        u_plane: &tight_u_plane,
        u_stride: chroma_width as u32,
        v_plane: &tight_v_plane,
        v_stride: chroma_width as u32,
        a_plane: &tight_a_plane,
        a_stride: i_width as u32,
        width: i_width as u32,
        height: i_height as u32,
    };

    i010_alpha_to_rgba10(
        &planar_image_alpha,
        &mut target_rgba,
        rgba_width as u32,
        YuvRange::Limited,
        YuvStandardMatrix::Bt601,
    )
    .unwrap();
}

fn fuzz_yuv_422(
    i_width: u8,
    i_height: u8,
    y_value: u16,
    u_value: u16,
    v_value: u16,
    stride_seed: u8,
) {
    if i_height == 0 || i_width == 0 {
        return;
    }
    let width = i_width as usize;
    let height = i_height as usize;
    let chroma_width = width.div_ceil(2);
    let y_stride = width + (stride_seed & 3) as usize;
    let u_stride = chroma_width + ((stride_seed >> 2) & 3) as usize;
    let v_stride = chroma_width + ((stride_seed >> 4) & 3) as usize;
    let dst_padding = ((stride_seed >> 6) & 3) as usize;
    let y_plane = vec![y_value; y_stride * (height - 1) + width];
    let u_plane = vec![u_value; u_stride * (height - 1) + chroma_width];
    let v_plane = vec![v_value; v_stride * (height - 1) + chroma_width];
    let tight_y_plane = vec![y_value; width * height];
    let tight_a_plane = vec![y_value; width * height];
    let tight_u_plane = vec![u_value; chroma_width * height];
    let tight_v_plane = vec![v_value; chroma_width * height];

    let bilinear_image = YuvPlanarImage {
        y_plane: &y_plane,
        y_stride: y_stride as u32,
        u_plane: &u_plane,
        u_stride: u_stride as u32,
        v_plane: &v_plane,
        v_stride: v_stride as u32,
        width: i_width as u32,
        height: i_height as u32,
    };
    let tight_image = YuvPlanarImage {
        y_plane: &tight_y_plane,
        y_stride: i_width as u32,
        u_plane: &tight_u_plane,
        u_stride: chroma_width as u32,
        v_plane: &tight_v_plane,
        v_stride: chroma_width as u32,
        width: i_width as u32,
        height: i_height as u32,
    };

    let rgb_width = width * 3;
    let rgb_stride = rgb_width + dst_padding;
    let mut target_rgb = vec![0u16; rgb_width * height];
    let mut bilinear_rgb = vec![0u16; rgb_stride * (height - 1) + rgb_width];

    i210_to_rgb10(
        &tight_image,
        &mut target_rgb,
        rgb_width as u32,
        YuvRange::Limited,
        YuvStandardMatrix::Bt601,
    )
    .unwrap();

    i210_to_rgb10_bilinear(
        &bilinear_image,
        &mut bilinear_rgb,
        rgb_stride as u32,
        YuvRange::Limited,
        YuvStandardMatrix::Bt601,
    )
    .unwrap();

    let rgba_width = width * 4;
    let rgba_stride = rgba_width + dst_padding;
    let mut target_rgba = vec![0u16; rgba_width * height];
    let mut bilinear_rgba = vec![0u16; rgba_stride * (height - 1) + rgba_width];

    i210_to_rgba10(
        &tight_image,
        &mut target_rgba,
        rgba_width as u32,
        YuvRange::Limited,
        YuvStandardMatrix::Bt601,
    )
    .unwrap();

    i210_to_rgba10_bilinear(
        &bilinear_image,
        &mut bilinear_rgba,
        rgba_stride as u32,
        YuvRange::Limited,
        YuvStandardMatrix::Bt601,
    )
    .unwrap();
    assert_rgba_alpha(&bilinear_rgba, rgba_stride, width, height, 1023);

    i212_to_rgba12(
        &tight_image,
        &mut target_rgba,
        rgba_width as u32,
        YuvRange::Limited,
        YuvStandardMatrix::Bt601,
    )
    .unwrap();

    bilinear_rgba.fill(0);
    i212_to_rgba12_bilinear(
        &bilinear_image,
        &mut bilinear_rgba,
        rgba_stride as u32,
        YuvRange::Limited,
        YuvStandardMatrix::Bt601,
    )
    .unwrap();
    assert_rgba_alpha(&bilinear_rgba, rgba_stride, width, height, 4095);

    i216_to_rgba16(
        &tight_image,
        &mut target_rgba,
        rgba_width as u32,
        YuvRange::Limited,
        YuvStandardMatrix::Bt601,
    )
    .unwrap();

    let planar_image_alpha = YuvPlanarImageWithAlpha {
        y_plane: &tight_y_plane,
        y_stride: i_width as u32,
        u_plane: &tight_u_plane,
        u_stride: chroma_width as u32,
        v_plane: &tight_v_plane,
        v_stride: chroma_width as u32,
        a_plane: &tight_a_plane,
        a_stride: i_width as u32,
        width: i_width as u32,
        height: i_height as u32,
    };

    i210_alpha_to_rgba10(
        &planar_image_alpha,
        &mut target_rgba,
        rgba_width as u32,
        YuvRange::Limited,
        YuvStandardMatrix::Bt601,
    )
    .unwrap();
}

fn fuzz_yuv_444(i_width: u8, i_height: u8, y_value: u16, u_value: u16, v_value: u16) {
    if i_height == 0 || i_width == 0 {
        return;
    }
    let y_plane = vec![y_value; i_height as usize * i_width as usize];
    let a_plane = vec![y_value; i_height as usize * i_width as usize];
    let u_plane = vec![u_value; i_width as usize * i_height as usize];
    let v_plane = vec![v_value; i_width as usize * i_height as usize];

    let planar_image = YuvPlanarImage {
        y_plane: &y_plane,
        y_stride: i_width as u32,
        u_plane: &u_plane,
        u_stride: i_width as u32,
        v_plane: &v_plane,
        v_stride: i_width as u32,
        width: i_width as u32,
        height: i_height as u32,
    };

    let mut target_rgb = vec![0u16; i_width as usize * i_height as usize * 3];

    i410_to_rgb10(
        &planar_image,
        &mut target_rgb,
        i_width as u32 * 3,
        YuvRange::Limited,
        YuvStandardMatrix::Bt601,
    )
    .unwrap();

    let mut target_rgba = vec![0u16; i_width as usize * i_height as usize * 4];

    i410_to_rgba10(
        &planar_image,
        &mut target_rgba,
        i_width as u32 * 4,
        YuvRange::Limited,
        YuvStandardMatrix::Bt601,
    )
    .unwrap();

    i412_to_rgba12(
        &planar_image,
        &mut target_rgba,
        i_width as u32 * 4,
        YuvRange::Limited,
        YuvStandardMatrix::Bt601,
    )
    .unwrap();

    i416_to_rgba16(
        &planar_image,
        &mut target_rgba,
        i_width as u32 * 4,
        YuvRange::Limited,
        YuvStandardMatrix::Bt601,
    )
    .unwrap();

    let planar_image_alpha = YuvPlanarImageWithAlpha {
        y_plane: &y_plane,
        y_stride: i_width as u32,
        u_plane: &u_plane,
        u_stride: i_width as u32,
        v_plane: &v_plane,
        v_stride: i_width as u32,
        a_plane: &a_plane,
        a_stride: i_width as u32,
        width: i_width as u32,
        height: i_height as u32,
    };

    i410_alpha_to_rgba10(
        &planar_image_alpha,
        &mut target_rgba,
        i_width as u32 * 4,
        YuvRange::Limited,
        YuvStandardMatrix::Bt601,
    )
    .unwrap();
}
