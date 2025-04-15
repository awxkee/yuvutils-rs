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
use arbitrary::{Arbitrary, Unstructured};
use libfuzzer_sys::fuzz_target;
use yuv::{
    rdp_abgr_to_yuv444, rdp_argb_to_yuv444, rdp_bgr_to_yuv444, rdp_bgra_to_yuv444,
    rdp_rgb_to_yuv444, rdp_rgba_to_yuv444, rdp_yuv444_to_abgr, rdp_yuv444_to_argb,
    rdp_yuv444_to_bgra, rdp_yuv444_to_rgb, rdp_yuv444_to_rgba, BufferStoreMut, YuvPlanarImageMut,
};

fuzz_target!(|data: (WidthStrides, u8)| {
    fuzz_yuv_444(data.0, data.1);
});

fn fuzz_yuv_444(i_width: WidthStrides, i_height: u8) {
    let [i_stride, y_stride, u_stride, v_stride] = i_width.1;
    if i_height == 0 || i_width.0 == 0 {
        return;
    }
    let y_plane = vec![0i16; y_stride as usize * (i_height - 1) as usize + i_width.0 as usize];
    let u_plane = vec![0i16; u_stride as usize * (i_height - 1) as usize + i_width.0 as usize];
    let v_plane = vec![0i16; v_stride as usize * (i_height - 1) as usize + i_width.0 as usize];

    let mut planar_image = YuvPlanarImageMut {
        y_plane: BufferStoreMut::Owned(y_plane),
        y_stride,
        u_plane: BufferStoreMut::Owned(u_plane),
        u_stride,
        v_plane: BufferStoreMut::Owned(v_plane),
        v_stride,
        width: i_width.0 as u32,
        height: i_height as u32,
    };

    let src_rgb = vec![0u8; (i_stride as usize * (i_height - 1) as usize + i_width.0 as usize) * 3];

    rdp_rgb_to_yuv444(&mut planar_image, &src_rgb, i_stride * 3).unwrap();
    rdp_bgr_to_yuv444(&mut planar_image, &src_rgb, i_stride * 3).unwrap();

    let src_rgba =
        vec![0u8; (i_stride as usize * (i_height - 1) as usize + i_width.0 as usize) * 4];

    rdp_rgba_to_yuv444(&mut planar_image, &src_rgba, i_stride * 4).unwrap();
    rdp_bgra_to_yuv444(&mut planar_image, &src_rgba, i_stride * 4).unwrap();
    rdp_abgr_to_yuv444(&mut planar_image, &src_rgba, i_stride * 4).unwrap();
    rdp_argb_to_yuv444(&mut planar_image, &src_rgba, i_stride * 4).unwrap();

    let fixed_planar = planar_image.to_fixed();

    let mut target_rgba =
        vec![0u8; (i_stride as usize * (i_height - 1) as usize + i_width.0 as usize) * 4];

    rdp_yuv444_to_rgba(&fixed_planar, &mut target_rgba, i_stride * 4).unwrap();
    rdp_yuv444_to_abgr(&fixed_planar, &mut target_rgba, i_stride * 4).unwrap();
    rdp_yuv444_to_argb(&fixed_planar, &mut target_rgba, i_stride * 4).unwrap();
    rdp_yuv444_to_bgra(&fixed_planar, &mut target_rgba, i_stride * 4).unwrap();

    let mut target_rgb =
        vec![0u8; (i_stride as usize * (i_height - 1) as usize + i_width.0 as usize) * 3];

    rdp_yuv444_to_rgb(&fixed_planar, &mut target_rgb, i_stride * 3).unwrap();
}

#[derive(Debug, Clone, Copy)]
struct WidthStrides(u8, [u32; 4]);

impl<'a> Arbitrary<'a> for WidthStrides {
    fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        let width = u8::arbitrary(u)?;
        let max_stride = u8::MAX - width;
        let strides: Vec<u32> = (0..4)
            .map(|_| (width + u.int_in_range(0..=max_stride).unwrap()) as u32)
            .collect();

        Ok(WidthStrides(width, strides.try_into().unwrap()))
    }
}
