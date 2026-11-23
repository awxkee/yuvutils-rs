/*
 * Copyright (c) Radzivon Bartoshyk, 11/2026. All rights reserved.
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
    i010_alpha_to_rgba, i010_to_bgr, i010_to_bgra, i010_to_rgb, i010_to_rgba, i012_alpha_to_rgba,
    i012_to_bgr, i012_to_bgra, i012_to_rgb, i012_to_rgba, i210_alpha_to_rgba, i210_to_bgr,
    i210_to_bgra, i210_to_rgb, i210_to_rgba, i212_alpha_to_rgba, i212_to_bgr, i212_to_bgra,
    i212_to_rgb, i212_to_rgba, i410_alpha_to_rgba, i410_to_rgba, i412_alpha_to_rgba,
    i412_to_rgb, i412_to_rgba, YuvError, YuvPlanarImage, YuvPlanarImageWithAlpha, YuvRange,
    YuvStandardMatrix,
};

// Covers 10/12-bit planar YUV -> 8-bit RGB(A). Destination buffers are sized to the exact
// minimum (`stride * (height - 1) + width * channels`) so AddressSanitizer reports any write
// past the end, and row padding is filled with a sentinel so writes past the end of a row
// that stay inside the buffer are caught as well.

type Planar = fn(
    &YuvPlanarImage<u16>,
    &mut [u8],
    u32,
    YuvRange,
    YuvStandardMatrix,
) -> Result<(), YuvError>;

type PlanarAlpha = fn(
    &YuvPlanarImageWithAlpha<u16>,
    &mut [u8],
    u32,
    YuvRange,
    YuvStandardMatrix,
) -> Result<(), YuvError>;

#[derive(Copy, Clone, Eq, PartialEq)]
enum Subsampling {
    Yuv420,
    Yuv422,
    Yuv444,
}

const SENTINEL: u8 = 0xA5;

fuzz_target!(|data: (u8, u8, u8, u8, u8, u8, u8)| {
    let (w, h, y, u, v, a, stride_seed) = data;
    if w == 0 || h == 0 {
        return;
    }
    let range = if stride_seed & 0x80 != 0 {
        YuvRange::Full
    } else {
        YuvRange::Limited
    };

    for (depth, subsampling, planar, alpha) in [
        (
            10,
            Subsampling::Yuv420,
            &[
                (i010_to_rgb as Planar, 3usize, false),
                (i010_to_bgr, 3, false),
                (i010_to_rgba, 4, true),
                (i010_to_bgra, 4, true),
            ][..],
            i010_alpha_to_rgba as PlanarAlpha,
        ),
        (
            10,
            Subsampling::Yuv422,
            &[
                (i210_to_rgb as Planar, 3, false),
                (i210_to_bgr, 3, false),
                (i210_to_rgba, 4, true),
                (i210_to_bgra, 4, true),
            ][..],
            i210_alpha_to_rgba,
        ),
        (
            10,
            Subsampling::Yuv444,
            &[(i410_to_rgba as Planar, 4, true)][..],
            i410_alpha_to_rgba,
        ),
        (
            12,
            Subsampling::Yuv420,
            &[
                (i012_to_rgb as Planar, 3, false),
                (i012_to_bgr, 3, false),
                (i012_to_rgba, 4, true),
                (i012_to_bgra, 4, true),
            ][..],
            i012_alpha_to_rgba,
        ),
        (
            12,
            Subsampling::Yuv422,
            &[
                (i212_to_rgb as Planar, 3, false),
                (i212_to_bgr, 3, false),
                (i212_to_rgba, 4, true),
                (i212_to_bgra, 4, true),
            ][..],
            i212_alpha_to_rgba,
        ),
        (
            12,
            Subsampling::Yuv444,
            &[
                (i412_to_rgb as Planar, 3, false),
                (i412_to_rgba, 4, true),
            ][..],
            i412_alpha_to_rgba,
        ),
    ] {
        fuzz_planar(
            w as usize,
            h as usize,
            [y, u, v, a],
            stride_seed,
            range,
            depth,
            subsampling,
            planar,
            alpha,
        );
    }
});

#[allow(clippy::too_many_arguments)]
fn fuzz_planar(
    width: usize,
    height: usize,
    values: [u8; 4],
    stride_seed: u8,
    range: YuvRange,
    depth: u32,
    subsampling: Subsampling,
    planar: &[(Planar, usize, bool)],
    alpha: PlanarAlpha,
) {
    let shift = depth - 8;
    let [y_value, u_value, v_value, a_value] = values.map(|x| (x as u16) << shift);

    let (chroma_width, chroma_height) = match subsampling {
        Subsampling::Yuv420 => (width.div_ceil(2), height.div_ceil(2)),
        Subsampling::Yuv422 => (width.div_ceil(2), height),
        Subsampling::Yuv444 => (width, height),
    };

    let y_stride = width + (stride_seed & 3) as usize;
    let u_stride = chroma_width + ((stride_seed >> 2) & 3) as usize;
    let v_stride = chroma_width + ((stride_seed >> 4) & 3) as usize;
    let a_stride = width + ((stride_seed >> 1) & 3) as usize;
    let dst_padding = ((stride_seed >> 5) & 3) as usize;

    let y_plane = vec![y_value; y_stride * (height - 1) + width];
    let u_plane = vec![u_value; u_stride * (chroma_height - 1) + chroma_width];
    let v_plane = vec![v_value; v_stride * (chroma_height - 1) + chroma_width];
    let a_plane = vec![a_value; a_stride * (height - 1) + width];

    let image = YuvPlanarImage {
        y_plane: &y_plane,
        y_stride: y_stride as u32,
        u_plane: &u_plane,
        u_stride: u_stride as u32,
        v_plane: &v_plane,
        v_stride: v_stride as u32,
        width: width as u32,
        height: height as u32,
    };

    for &(convert, channels, has_alpha) in planar {
        let row = width * channels;
        let stride = row + dst_padding;
        let mut dst = vec![SENTINEL; stride * (height - 1) + row];
        convert(
            &image,
            &mut dst,
            stride as u32,
            range,
            YuvStandardMatrix::Bt601,
        )
        .unwrap();
        check_rows(&dst, stride, row, channels, has_alpha.then_some(255));
    }

    let image = YuvPlanarImageWithAlpha {
        y_plane: &y_plane,
        y_stride: y_stride as u32,
        u_plane: &u_plane,
        u_stride: u_stride as u32,
        v_plane: &v_plane,
        v_stride: v_stride as u32,
        a_plane: &a_plane,
        a_stride: a_stride as u32,
        width: width as u32,
        height: height as u32,
    };
    let row = width * 4;
    let stride = row + dst_padding;
    let mut dst = vec![SENTINEL; stride * (height - 1) + row];
    alpha(
        &image,
        &mut dst,
        stride as u32,
        range,
        YuvStandardMatrix::Bt601,
    )
    .unwrap();
    check_rows(&dst, stride, row, 4, Some(values[3]));
}

fn check_rows(dst: &[u8], stride: usize, row: usize, channels: usize, alpha: Option<u8>) {
    for (y, line) in dst.chunks(stride).enumerate() {
        let (pixels, padding) = line.split_at(row);
        assert!(
            padding.iter().all(|&x| x == SENTINEL),
            "row {y} padding was overwritten"
        );
        if let Some(alpha) = alpha {
            for (x, pixel) in pixels.chunks_exact(channels).enumerate() {
                assert_eq!(pixel[3], alpha, "row {y}, pixel {x}");
            }
        }
    }
}
