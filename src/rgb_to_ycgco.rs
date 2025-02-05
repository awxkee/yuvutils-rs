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

use crate::yuv_error::check_rgba_destination;
#[allow(unused_imports)]
use crate::yuv_support::*;
use crate::{YuvError, YuvPlanarImageMut};
#[cfg(feature = "rayon")]
use rayon::iter::{IndexedParallelIterator, ParallelIterator};
#[cfg(feature = "rayon")]
use rayon::prelude::{ParallelSlice, ParallelSliceMut};

fn rgbx_to_ycgco<const ORIGIN_CHANNELS: u8, const SAMPLING: u8>(
    image: &mut YuvPlanarImageMut<u8>,
    rgba: &[u8],
    rgba_stride: u32,
    range: YuvRange,
) -> Result<(), YuvError> {
    let chroma_subsampling: YuvChromaSubsampling = SAMPLING.into();
    let src_chans: YuvSourceChannels = ORIGIN_CHANNELS.into();
    let channels = src_chans.get_channels_count();
    const PRECISION: i32 = 13;
    let range = get_yuv_range(8, range);
    let precision_scale = (1 << PRECISION) as f32;
    let rounding_const_bias: i32 = (1 << (PRECISION - 1)) - 1;
    let bias_y = range.bias_y as i32 * (1 << PRECISION) + rounding_const_bias;
    let bias_uv = range.bias_uv as i32 * (1 << PRECISION) + rounding_const_bias;
    let max_colors = (1 << 8) - 1i32;

    check_rgba_destination(rgba, rgba_stride, image.width, image.height, channels)?;
    image.check_constraints(chroma_subsampling)?;

    let range_reduction_y =
        (range.range_y as f32 / max_colors as f32 * precision_scale).round() as i32;
    let range_reduction_uv =
        (range.range_uv as f32 / max_colors as f32 * precision_scale).round() as i32;

    let process_halved_chroma_row =
        |y_plane: &mut [u8], u_plane: &mut [u8], v_plane: &mut [u8], rgba: &[u8]| {
            for (((y_dst, u_dst), v_dst), rgba) in y_plane
                .chunks_exact_mut(2)
                .zip(u_plane.iter_mut())
                .zip(v_plane.iter_mut())
                .zip(rgba.chunks_exact(channels * 2))
            {
                let src0 = &rgba[0..channels];

                let r0 = src0[src_chans.get_r_channel_offset()] as i32;
                let g0 = src0[src_chans.get_g_channel_offset()] as i32;
                let b0 = src0[src_chans.get_b_channel_offset()] as i32;

                let hg0 = (g0 * range_reduction_y) >> 1;
                let y_0 = (hg0 + ((r0 * range_reduction_y + b0 * range_reduction_y) >> 2) + bias_y)
                    >> PRECISION;

                y_dst[0] = y_0 as u8;

                let src1 = &rgba[channels..channels * 2];

                let r1 = src1[src_chans.get_r_channel_offset()] as i32;
                let g1 = src1[src_chans.get_g_channel_offset()] as i32;
                let b1 = src1[src_chans.get_b_channel_offset()] as i32;
                let hg1 = (g1 * range_reduction_y) >> 1;
                let y_1 = (hg1 + ((r1 * range_reduction_y + b1 * range_reduction_y) >> 2) + bias_y)
                    >> PRECISION;
                y_dst[1] = y_1 as u8;

                let r = ((r0 + r1 + 1) >> 1) * range_reduction_uv;
                let g = ((g0 + g1 + 1) >> 1) * range_reduction_uv;
                let b = ((b0 + b1 + 1) >> 1) * range_reduction_uv;

                let cg = (((g >> 1) - ((r + b) >> 2)) + bias_uv) >> PRECISION;
                let co = (((r - b) >> 1) + bias_uv) >> PRECISION;

                *u_dst = cg as u8;
                *v_dst = co as u8;
            }

            if image.width & 1 != 0 {
                let rgb_last = rgba.chunks_exact(channels * 2).remainder();
                let mut r0 = rgb_last[src_chans.get_r_channel_offset()] as i32;
                let mut g0 = rgb_last[src_chans.get_g_channel_offset()] as i32;
                let mut b0 = rgb_last[src_chans.get_b_channel_offset()] as i32;

                let y_last = y_plane.last_mut().unwrap();
                let u_last = u_plane.last_mut().unwrap();
                let v_last = v_plane.last_mut().unwrap();

                let hg0 = (g0 * range_reduction_y) >> 1;
                let y_0 = (hg0 + ((r0 * range_reduction_y + b0 * range_reduction_y) >> 2) + bias_y)
                    >> PRECISION;

                *y_last = y_0 as u8;

                r0 *= range_reduction_y;
                g0 *= range_reduction_uv;
                b0 *= range_reduction_uv;

                let cg = (((g0 >> 1) - ((r0 + b0) >> 2)) + bias_uv) >> PRECISION;
                let co = (((r0 - b0) >> 1) + bias_uv) >> PRECISION;
                *u_last = cg as u8;
                *v_last = co as u8;
            }
        };

    let process_doubled_row = |y_plane0: &mut [u8],
                               y_plane1: &mut [u8],
                               u_plane: &mut [u8],
                               v_plane: &mut [u8],
                               rgba0: &[u8],
                               rgba1: &[u8]| {
        for (((((y_dst0, y_dst1), u_dst), v_dst), rgba0), rgba1) in y_plane0
            .chunks_exact_mut(2)
            .zip(y_plane1.chunks_exact_mut(2))
            .zip(u_plane.iter_mut())
            .zip(v_plane.iter_mut())
            .zip(rgba0.chunks_exact(channels * 2))
            .zip(rgba1.chunks_exact(channels * 2))
        {
            let src00 = &rgba0[0..channels];

            let r00 = src00[src_chans.get_r_channel_offset()] as i32;
            let g00 = src00[src_chans.get_g_channel_offset()] as i32;
            let b00 = src00[src_chans.get_b_channel_offset()] as i32;
            let hg00 = (g00 * range_reduction_y) >> 1;
            let y_00 = (hg00 + ((r00 * range_reduction_y + b00 * range_reduction_y) >> 2) + bias_y)
                >> PRECISION;
            y_dst0[0] = y_00 as u8;

            let src1 = &rgba0[channels..channels * 2];

            let r01 = src1[src_chans.get_r_channel_offset()] as i32;
            let g01 = src1[src_chans.get_g_channel_offset()] as i32;
            let b01 = src1[src_chans.get_b_channel_offset()] as i32;
            let hg01 = (g01 * range_reduction_y) >> 1;
            let y_01 = (hg01 + ((r01 * range_reduction_y + b01 * range_reduction_y) >> 2) + bias_y)
                >> PRECISION;
            y_dst0[1] = y_01 as u8;

            let src10 = &rgba1[0..channels];

            let r10 = src10[src_chans.get_r_channel_offset()] as i32;
            let g10 = src10[src_chans.get_g_channel_offset()] as i32;
            let b10 = src10[src_chans.get_b_channel_offset()] as i32;
            let hg10 = (g10 * range_reduction_y) >> 1;
            let y_10 = (hg10 + ((r10 * range_reduction_y + b10 * range_reduction_y) >> 2) + bias_y)
                >> PRECISION;
            y_dst1[0] = y_10 as u8;

            let src11 = &rgba1[channels..channels * 2];

            let r11 = src11[src_chans.get_r_channel_offset()] as i32;
            let g11 = src11[src_chans.get_g_channel_offset()] as i32;
            let b11 = src11[src_chans.get_b_channel_offset()] as i32;
            let hg11 = (g11 * range_reduction_y) >> 1;
            let y_11 = (hg11 + ((r11 * range_reduction_y + b11 * range_reduction_y) >> 2) + bias_y)
                >> PRECISION;
            y_dst1[1] = y_11 as u8;

            let ruv = ((r00 + r01 + r10 + r11 + 2) >> 2) * range_reduction_uv;
            let guv = ((g00 + g01 + g10 + g11 + 2) >> 2) * range_reduction_uv;
            let buv = ((b00 + b01 + b10 + b11 + 2) >> 2) * range_reduction_uv;

            let cg = (((guv >> 1) - ((ruv + buv) >> 2)) + bias_uv) >> PRECISION;
            let co = (((ruv - buv) >> 1) + bias_uv) >> PRECISION;
            *u_dst = cg as u8;
            *v_dst = co as u8;
        }

        if image.width & 1 != 0 {
            let rgb_last0 = rgba0.chunks_exact(channels * 2).remainder();
            let rgb_last1 = rgba1.chunks_exact(channels * 2).remainder();
            let r0 = rgb_last0[src_chans.get_r_channel_offset()] as i32;
            let g0 = rgb_last0[src_chans.get_g_channel_offset()] as i32;
            let b0 = rgb_last0[src_chans.get_b_channel_offset()] as i32;

            let r1 = rgb_last1[src_chans.get_r_channel_offset()] as i32;
            let g1 = rgb_last1[src_chans.get_g_channel_offset()] as i32;
            let b1 = rgb_last1[src_chans.get_b_channel_offset()] as i32;

            let y0_last = y_plane0.last_mut().unwrap();
            let y1_last = y_plane1.last_mut().unwrap();
            let u_last = u_plane.last_mut().unwrap();
            let v_last = v_plane.last_mut().unwrap();

            let hg0 = (g0 * range_reduction_y) >> 1;
            let y_0 = (hg0 + ((r0 * range_reduction_y + b0 * range_reduction_y) >> 2) + bias_y)
                >> PRECISION;
            *y0_last = y_0 as u8;

            let hg1 = (g1 * range_reduction_y) >> 1;
            let y_1 = (hg1 + ((r1 * range_reduction_y + b1 * range_reduction_y) >> 2) + bias_y)
                >> PRECISION;
            *y1_last = y_1 as u8;

            let r0 = ((r0 + r1) >> 1) * range_reduction_uv;
            let g0 = ((g0 + g1) >> 1) * range_reduction_uv;
            let b0 = ((b0 + b1) >> 1) * range_reduction_uv;

            let cg = (((g0 >> 1) - ((r0 + b0) >> 2)) + bias_uv) >> PRECISION;
            let co = (((r0 - b0) >> 1) + bias_uv) >> PRECISION;
            *u_last = cg as u8;
            *v_last = co as u8;
        }
    };

    let y_plane = image.y_plane.borrow_mut();
    let u_plane = image.u_plane.borrow_mut();
    let v_plane = image.v_plane.borrow_mut();
    let y_stride = image.y_stride as usize;
    let u_stride = image.u_stride as usize;
    let v_stride = image.v_stride as usize;

    if chroma_subsampling == YuvChromaSubsampling::Yuv444 {
        let iter;
        #[cfg(feature = "rayon")]
        {
            iter = y_plane
                .par_chunks_exact_mut(y_stride)
                .zip(u_plane.par_chunks_exact_mut(u_stride))
                .zip(v_plane.par_chunks_exact_mut(v_stride))
                .zip(rgba.par_chunks_exact(rgba_stride as usize));
        }
        #[cfg(not(feature = "rayon"))]
        {
            iter = y_plane
                .chunks_exact_mut(y_stride)
                .zip(u_plane.chunks_exact_mut(u_stride))
                .zip(v_plane.chunks_exact_mut(v_stride))
                .zip(rgba.chunks_exact(rgba_stride as usize));
        }
        iter.for_each(|(((y_dst, u_plane), v_plane), rgba)| {
            let y_dst = &mut y_dst[0..image.width as usize];
            for (((y_dst, u_dst), v_dst), rgba) in y_dst
                .iter_mut()
                .zip(u_plane.iter_mut())
                .zip(v_plane.iter_mut())
                .zip(rgba.chunks_exact(channels))
            {
                let mut r0 = rgba[src_chans.get_r_channel_offset()] as i32;
                let mut g0 = rgba[src_chans.get_g_channel_offset()] as i32;
                let mut b0 = rgba[src_chans.get_b_channel_offset()] as i32;
                let hg0 = (g0 * range_reduction_y) >> 1;
                let y_0 = (hg0 + ((r0 * range_reduction_y + b0 * range_reduction_y) >> 2) + bias_y)
                    >> PRECISION;
                *y_dst = y_0 as u8;

                r0 *= range_reduction_y;
                g0 *= range_reduction_y;
                b0 *= range_reduction_y;

                let cg = (((g0 >> 1) - ((r0 + b0) >> 2)) + bias_uv) >> PRECISION;
                let co = (((r0 - b0) >> 1) + bias_uv) >> PRECISION;
                *u_dst = cg as u8;
                *v_dst = co as u8;
            }
        });
    } else if chroma_subsampling == YuvChromaSubsampling::Yuv422 {
        let iter;
        #[cfg(feature = "rayon")]
        {
            iter = y_plane
                .par_chunks_exact_mut(y_stride)
                .zip(u_plane.par_chunks_exact_mut(u_stride))
                .zip(v_plane.par_chunks_exact_mut(v_stride))
                .zip(rgba.par_chunks_exact(rgba_stride as usize));
        }
        #[cfg(not(feature = "rayon"))]
        {
            iter = y_plane
                .chunks_exact_mut(y_stride)
                .zip(u_plane.chunks_exact_mut(u_stride))
                .zip(v_plane.chunks_exact_mut(v_stride))
                .zip(rgba.chunks_exact(rgba_stride as usize));
        }

        iter.for_each(|(((y_plane, u_plane), v_plane), rgba)| {
            process_halved_chroma_row(
                &mut y_plane[0..image.width as usize],
                &mut u_plane[0..(image.width as usize).div_ceil(2)],
                &mut v_plane[0..(image.width as usize).div_ceil(2)],
                &rgba[0..image.width as usize * channels],
            );
        });
    } else if chroma_subsampling == YuvChromaSubsampling::Yuv420 {
        let iter;
        #[cfg(feature = "rayon")]
        {
            iter = y_plane
                .par_chunks_exact_mut(y_stride * 2)
                .zip(u_plane.par_chunks_exact_mut(u_stride))
                .zip(v_plane.par_chunks_exact_mut(v_stride))
                .zip(rgba.par_chunks_exact(rgba_stride as usize * 2));
        }
        #[cfg(not(feature = "rayon"))]
        {
            iter = y_plane
                .chunks_exact_mut(y_stride * 2)
                .zip(u_plane.chunks_exact_mut(u_stride))
                .zip(v_plane.chunks_exact_mut(v_stride))
                .zip(rgba.chunks_exact(rgba_stride as usize * 2));
        }
        iter.for_each(|(((y_plane, u_plane), v_plane), rgba)| {
            let (rgba0, rgba1) = rgba.split_at(rgba_stride as usize);
            let (y_plane0, y_plane1) = y_plane.split_at_mut(y_stride);
            process_doubled_row(
                &mut y_plane0[0..image.width as usize],
                &mut y_plane1[0..image.width as usize],
                &mut u_plane[0..(image.width as usize).div_ceil(2)],
                &mut v_plane[0..(image.width as usize).div_ceil(2)],
                &rgba0[0..image.width as usize * channels],
                &rgba1[0..image.width as usize * channels],
            );
        });

        if image.height & 1 != 0 {
            let remainder_y_plane = y_plane.chunks_exact_mut(y_stride * 2).into_remainder();
            let remainder_rgba = rgba.chunks_exact(rgba_stride as usize * 2).remainder();
            let u_plane = u_plane.chunks_exact_mut(u_stride).last().unwrap();
            let v_plane = v_plane.chunks_exact_mut(v_stride).last().unwrap();
            process_halved_chroma_row(
                &mut remainder_y_plane[0..image.width as usize],
                &mut u_plane[0..(image.width as usize).div_ceil(2)],
                &mut v_plane[0..(image.width as usize).div_ceil(2)],
                &remainder_rgba[0..image.width as usize * channels],
            );
        }
    } else {
        unreachable!();
    }

    Ok(())
}

/// Convert RGB image data to YCgCo 422 planar format.
///
/// This function performs RGB to YCgCo conversion and stores the result in YUV422 planar format,
/// with separate planes for Y (luminance), Cg (chrominance), and Co (chrominance) components.
///
/// # Arguments
///
/// * `image` - Target planar image.
/// * `rgb` - The input RGB image data slice.
/// * `rgb_stride` - The stride (components per row) for the RGB image data.
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input RGB data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn rgb_to_ycgco422(
    image: &mut YuvPlanarImageMut<u8>,
    rgb: &[u8],
    rgb_stride: u32,
    range: YuvRange,
) -> Result<(), YuvError> {
    rgbx_to_ycgco::<{ YuvSourceChannels::Rgb as u8 }, { YuvChromaSubsampling::Yuv422 as u8 }>(
        image, rgb, rgb_stride, range,
    )
}

/// Convert BGR image data to YCgCo 422 planar format.
///
/// This function performs BGR to YCgCo conversion and stores the result in YUV422 planar format,
/// with separate planes for Y (luminance), Cg (chrominance), and Co (chrominance) components.
///
/// # Arguments
///
/// * `image` - Target planar image.
/// * `bgr` - The input BGR image data slice.
/// * `bgr_stride` - The stride (components per row) for the BGR image data.
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input RGB data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn bgr_to_ycgco422(
    image: &mut YuvPlanarImageMut<u8>,
    bgr: &[u8],
    bgr_stride: u32,
    range: YuvRange,
) -> Result<(), YuvError> {
    rgbx_to_ycgco::<{ YuvSourceChannels::Bgr as u8 }, { YuvChromaSubsampling::Yuv422 as u8 }>(
        image, bgr, bgr_stride, range,
    )
}

/// Convert RGBA image data to YCgCo 422 planar format.
///
/// This function performs RGBA to YCgCo conversion and stores the result in YUV422 planar format,
/// with separate planes for Y (luminance), Cg (chrominance), and Co (chrominance) components.
///
/// # Arguments
///
/// * `image` - Target planar image.
/// * `rgba` - The input RGBA image data slice.
/// * `rgba_stride` - The stride (components per row) for the RGBA image data.
/// * `range` - The YUV range (limited or full).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input RGBA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn rgba_to_ycgco422(
    image: &mut YuvPlanarImageMut<u8>,
    rgba: &[u8],
    rgba_stride: u32,
    range: YuvRange,
) -> Result<(), YuvError> {
    rgbx_to_ycgco::<{ YuvSourceChannels::Rgba as u8 }, { YuvChromaSubsampling::Yuv422 as u8 }>(
        image,
        rgba,
        rgba_stride,
        range,
    )
}

/// Convert BGRA image data to YCgCo 422 planar format.
///
/// This function performs BGRA to YCgCo conversion and stores the result in YUV422 planar format,
/// with separate planes for Y (luminance), Cg (chrominance), and Co (chrominance) components.
///
/// # Arguments
///
/// * `image` - Target planar image.
/// * `bgra` - The input BGRA image data slice.
/// * `bgra_stride` - The stride (components per row) for the BGRA image data.
/// * `range` - The YUV range (limited or full).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGRA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn bgra_to_ycgco422(
    image: &mut YuvPlanarImageMut<u8>,
    bgra: &[u8],
    bgra_stride: u32,
    range: YuvRange,
) -> Result<(), YuvError> {
    rgbx_to_ycgco::<{ YuvSourceChannels::Bgra as u8 }, { YuvChromaSubsampling::Yuv422 as u8 }>(
        image,
        bgra,
        bgra_stride,
        range,
    )
}

/// Convert RGB image data to YCgCo 420 planar format.
///
/// This function performs RGB to YCgCo conversion and stores the result in YUV420 planar format,
/// with separate planes for Y (luminance), Cg (chrominance), and Co (chrominance) components.
///
/// # Arguments
///
/// * `image` - Target planar image.
/// * `rgb` - The input RGB image data slice.
/// * `rgb_stride` - The stride (components per row) for the RGB image data.
/// * `range` - The YUV range (limited or full).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input RGB data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn rgb_to_ycgco420(
    image: &mut YuvPlanarImageMut<u8>,
    rgb: &[u8],
    rgb_stride: u32,
    range: YuvRange,
) -> Result<(), YuvError> {
    rgbx_to_ycgco::<{ YuvSourceChannels::Rgb as u8 }, { YuvChromaSubsampling::Yuv420 as u8 }>(
        image, rgb, rgb_stride, range,
    )
}

/// Convert BGR image data to YCgCo 420 planar format.
///
/// This function performs BGR to YCgCo conversion and stores the result in YUV420 planar format,
/// with separate planes for Y (luminance), Cg (chrominance), and Co (chrominance) components.
///
/// # Arguments
///
/// * `image` - Target planar image.
/// * `bgr` - The input BGR image data slice.
/// * `bgr_stride` - The stride (components per row) for the BGR image data.
/// * `range` - The YUV range (limited or full).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGR data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn bgr_to_ycgco420(
    image: &mut YuvPlanarImageMut<u8>,
    bgr: &[u8],
    bgr_stride: u32,
    range: YuvRange,
) -> Result<(), YuvError> {
    rgbx_to_ycgco::<{ YuvSourceChannels::Bgr as u8 }, { YuvChromaSubsampling::Yuv420 as u8 }>(
        image, bgr, bgr_stride, range,
    )
}

/// Convert RGBA image data to YCgCo 420 planar format.
///
/// This function performs RGBA to YCgCo conversion and stores the result in YUV420 planar format,
/// with separate planes for Y (luminance), Cg (chrominance), and Co (chrominance) components.
///
/// # Arguments
///
/// * `image` - Target planar image.
/// * `rgba` - The input RGBA image data slice.
/// * `rgba_stride` - The stride (components per row) for the RGBA image data.
/// * `range` - The YUV range (limited or full).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input RGBA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn rgba_to_ycgco420(
    image: &mut YuvPlanarImageMut<u8>,
    rgba: &[u8],
    rgba_stride: u32,
    range: YuvRange,
) -> Result<(), YuvError> {
    rgbx_to_ycgco::<{ YuvSourceChannels::Rgba as u8 }, { YuvChromaSubsampling::Yuv420 as u8 }>(
        image,
        rgba,
        rgba_stride,
        range,
    )
}

/// Convert BGRA image data to YCgCo 420 planar format.
///
/// This function performs BGRA to YCgCo conversion and stores the result in YUV420 planar format,
/// with separate planes for Y (luminance), Cg (chrominance), and Co (chrominance) components.
///
/// # Arguments
///
/// * `image` - Target planar image.
/// * `bgra` - The input BGRA image data slice.
/// * `bgra_stride` - The stride (components per row) for the BGRA image data.
/// * `range` - The YUV range (limited or full).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGRA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn bgra_to_ycgco420(
    image: &mut YuvPlanarImageMut<u8>,
    bgra: &[u8],
    bgra_stride: u32,
    range: YuvRange,
) -> Result<(), YuvError> {
    rgbx_to_ycgco::<{ YuvSourceChannels::Bgra as u8 }, { YuvChromaSubsampling::Yuv420 as u8 }>(
        image,
        bgra,
        bgra_stride,
        range,
    )
}

/// Convert RGB image data to YCgCo 444 planar format.
///
/// This function performs RGB to YCgCo conversion and stores the result in YUV444 planar format,
/// with separate planes for Y (luminance), Cg (chrominance), and Co (chrominance) components.
///
/// # Arguments
///
/// * `image` - Target planar image.
/// * `rgb` - The input RGB image data slice.
/// * `rgb_stride` - The stride (components per row) for the RGB image data.
/// * `range` - The YUV range (limited or full).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input RGB data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn rgb_to_ycgco444(
    image: &mut YuvPlanarImageMut<u8>,
    rgb: &[u8],
    rgb_stride: u32,
    range: YuvRange,
) -> Result<(), YuvError> {
    rgbx_to_ycgco::<{ YuvSourceChannels::Rgb as u8 }, { YuvChromaSubsampling::Yuv444 as u8 }>(
        image, rgb, rgb_stride, range,
    )
}

/// Convert BGR image data to YCgCo 444 planar format.
///
/// This function performs BGR to YCgCo conversion and stores the result in YUV444 planar format,
/// with separate planes for Y (luminance), Cg (chrominance), and Co (chrominance) components.
///
/// # Arguments
///
/// * `image` - Target planar image.
/// * `bgr` - The input RGB image data slice.
/// * `bgr_stride` - The stride (components per row) for the BGR image data.
/// * `range` - The YUV range (limited or full).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGR data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn bgr_to_ycgco444(
    image: &mut YuvPlanarImageMut<u8>,
    bgr: &[u8],
    bgr_stride: u32,
    range: YuvRange,
) -> Result<(), YuvError> {
    rgbx_to_ycgco::<{ YuvSourceChannels::Bgr as u8 }, { YuvChromaSubsampling::Yuv444 as u8 }>(
        image, bgr, bgr_stride, range,
    )
}

/// Convert RGBA image data to YCgCo 444 planar format.
///
/// This function performs RGBA to YCgCo conversion and stores the result in YUV444 planar format,
/// with separate planes for Y (luminance), Cg (chrominance), and Co (chrominance) components.
///
/// # Arguments
///
/// * `image` - Target planar image.
/// * `rgba` - The input RGBA image data slice.
/// * `rgba_stride` - The stride (components per row) for the RGBA image data.
/// * `range` - The YUV range (limited or full).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input RGBA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn rgba_to_ycgco444(
    image: &mut YuvPlanarImageMut<u8>,
    rgba: &[u8],
    rgba_stride: u32,
    range: YuvRange,
) -> Result<(), YuvError> {
    rgbx_to_ycgco::<{ YuvSourceChannels::Rgba as u8 }, { YuvChromaSubsampling::Yuv444 as u8 }>(
        image,
        rgba,
        rgba_stride,
        range,
    )
}

/// Convert BGRA image data to YCgCo 444 planar format.
///
/// This function performs BGRA to YCgCo conversion and stores the result in YUV444 planar format,
/// with separate planes for Y (luminance), Cg (chrominance), and Co (chrominance) components.
///
/// # Arguments
///
/// * `image` - Target planar image.
/// * `bgra` - The input BGRA image data slice.
/// * `bgra_stride` - The stride (components per row) for the BGRA image data.
/// * `range` - The YUV range (limited or full).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGRA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn bgra_to_ycgco444(
    image: &mut YuvPlanarImageMut<u8>,
    bgra: &[u8],
    bgra_stride: u32,
    range: YuvRange,
) -> Result<(), YuvError> {
    rgbx_to_ycgco::<{ YuvSourceChannels::Bgra as u8 }, { YuvChromaSubsampling::Yuv444 as u8 }>(
        image,
        bgra,
        bgra_stride,
        range,
    )
}
