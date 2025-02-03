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

use crate::numerics::qrshr;
use crate::yuv_error::check_rgba_destination;
use crate::yuv_support::*;
use crate::{YuvError, YuvPlanarImage};
#[cfg(feature = "rayon")]
use rayon::iter::{IndexedParallelIterator, ParallelIterator};
#[cfg(feature = "rayon")]
use rayon::prelude::{ParallelSlice, ParallelSliceMut};

fn ycgco_ro_rgbx<const DESTINATION_CHANNELS: u8, const SAMPLING: u8>(
    image: &YuvPlanarImage<u8>,
    rgba: &mut [u8],
    rgba_stride: u32,
    range: YuvRange,
) -> Result<(), YuvError> {
    let chroma_subsampling: YuvChromaSubsampling = SAMPLING.into();
    let dst_chans: YuvSourceChannels = DESTINATION_CHANNELS.into();
    let channels = dst_chans.get_channels_count();

    check_rgba_destination(rgba, rgba_stride, image.width, image.height, channels)?;
    image.check_constraints(chroma_subsampling)?;

    let range = get_yuv_range(8, range);
    let bias_y = range.bias_y as i32;
    let bias_uv = range.bias_uv as i32;

    const PRECISION: i32 = 13;

    let max_colors = (1 << 8) - 1i32;
    let precision_scale = (1 << PRECISION) as f32;

    let range_reduction_y =
        (max_colors as f32 / range.range_y as f32 * precision_scale).round() as i32;
    let range_reduction_uv =
        (max_colors as f32 / range.range_uv as f32 * precision_scale).round() as i32;

    let process_halved_chroma_row =
        |y_plane: &[u8], u_plane: &[u8], v_plane: &[u8], rgba: &mut [u8]| {
            for (((rgba, y_src), &u_src), &v_src) in rgba
                .chunks_exact_mut(channels * 2)
                .zip(y_plane.chunks_exact(2))
                .zip(u_plane.iter())
                .zip(v_plane.iter())
            {
                let y_value0 = (y_src[0] as i32 - bias_y) * range_reduction_y;
                let cb_value = (u_src as i32 - bias_uv) * range_reduction_uv;
                let cr_value = (v_src as i32 - bias_uv) * range_reduction_uv;

                let t0 = y_value0 - cb_value;

                let r0 = qrshr::<PRECISION, 8>(t0 + cr_value);
                let b0 = qrshr::<PRECISION, 8>(t0 - cr_value);
                let g0 = qrshr::<PRECISION, 8>(y_value0 + cb_value);

                let rgba0 = &mut rgba[0..channels];

                rgba0[dst_chans.get_r_channel_offset()] = r0 as u8;
                rgba0[dst_chans.get_g_channel_offset()] = g0 as u8;
                rgba0[dst_chans.get_b_channel_offset()] = b0 as u8;
                if dst_chans.has_alpha() {
                    rgba0[dst_chans.get_a_channel_offset()] = 255u8;
                }

                let y_value1 = (y_src[1] as i32 - bias_y) * range_reduction_y;

                let t1 = y_value1 - cb_value;

                let r1 = qrshr::<PRECISION, 8>(t1 + cr_value);
                let b1 = qrshr::<PRECISION, 8>(t1 - cr_value);
                let g1 = qrshr::<PRECISION, 8>(y_value1 + cb_value);

                let rgba1 = &mut rgba[channels..channels * 2];

                rgba1[dst_chans.get_r_channel_offset()] = r1 as u8;
                rgba1[dst_chans.get_g_channel_offset()] = g1 as u8;
                rgba1[dst_chans.get_b_channel_offset()] = b1 as u8;
                if dst_chans.has_alpha() {
                    rgba1[dst_chans.get_a_channel_offset()] = 255u8;
                }
            }

            if image.width & 1 != 0 {
                let y_value0 = (*y_plane.last().unwrap() as i32 - bias_y) * range_reduction_y;
                let cb_value = (*u_plane.last().unwrap() as i32 - bias_uv) * range_reduction_uv;
                let cr_value = (*v_plane.last().unwrap() as i32 - bias_uv) * range_reduction_uv;
                let rgba = rgba.chunks_exact_mut(channels).last().unwrap();
                let rgba0 = &mut rgba[0..channels];

                let t0 = y_value0 - cb_value;

                let r0 = qrshr::<PRECISION, 8>(t0 + cr_value);
                let b0 = qrshr::<PRECISION, 8>(t0 - cr_value);
                let g0 = qrshr::<PRECISION, 8>(y_value0 + cb_value);
                rgba0[dst_chans.get_r_channel_offset()] = r0 as u8;
                rgba0[dst_chans.get_g_channel_offset()] = g0 as u8;
                rgba0[dst_chans.get_b_channel_offset()] = b0 as u8;
                if dst_chans.has_alpha() {
                    rgba0[dst_chans.get_a_channel_offset()] = 255;
                }
            }
        };

    let process_doubled_chroma_row = |y_plane0: &[u8],
                                      y_plane1: &[u8],
                                      u_plane: &[u8],
                                      v_plane: &[u8],
                                      rgba0: &mut [u8],
                                      rgba1: &mut [u8]| {
        for (((((rgba0, rgba1), y_src0), y_src1), &u_src), &v_src) in rgba0
            .chunks_exact_mut(channels * 2)
            .zip(rgba1.chunks_exact_mut(channels * 2))
            .zip(y_plane0.chunks_exact(2))
            .zip(y_plane1.chunks_exact(2))
            .zip(u_plane.iter())
            .zip(v_plane.iter())
        {
            let y_value0 = (y_src0[0] as i32 - bias_y) * range_reduction_y;
            let cb_value = (u_src as i32 - bias_uv) * range_reduction_uv;
            let cr_value = (v_src as i32 - bias_uv) * range_reduction_uv;

            let t0 = y_value0 - cb_value;

            let r0 = qrshr::<PRECISION, 8>(t0 + cr_value);
            let b0 = qrshr::<PRECISION, 8>(t0 - cr_value);
            let g0 = qrshr::<PRECISION, 8>(y_value0 + cb_value);

            let rgba00 = &mut rgba0[0..channels];

            rgba00[dst_chans.get_r_channel_offset()] = r0 as u8;
            rgba00[dst_chans.get_g_channel_offset()] = g0 as u8;
            rgba00[dst_chans.get_b_channel_offset()] = b0 as u8;
            if dst_chans.has_alpha() {
                rgba00[dst_chans.get_a_channel_offset()] = 255u8;
            }

            let y_value1 = (y_src0[1] as i32 - bias_y) * range_reduction_y;

            let t1 = y_value1 - cb_value;

            let r1 = qrshr::<PRECISION, 8>(t1 + cr_value);
            let b1 = qrshr::<PRECISION, 8>(t1 - cr_value);
            let g1 = qrshr::<PRECISION, 8>(y_value1 + cb_value);

            let rgba01 = &mut rgba0[channels..channels * 2];

            rgba01[dst_chans.get_r_channel_offset()] = r1 as u8;
            rgba01[dst_chans.get_g_channel_offset()] = g1 as u8;
            rgba01[dst_chans.get_b_channel_offset()] = b1 as u8;
            if dst_chans.has_alpha() {
                rgba01[dst_chans.get_a_channel_offset()] = 255u8;
            }

            let y_value10 = (y_src1[0] as i32 - bias_y) * range_reduction_y;

            let t10 = y_value10 - cb_value;

            let r10 = qrshr::<PRECISION, 8>(t10 + cr_value);
            let b10 = qrshr::<PRECISION, 8>(t10 - cr_value);
            let g10 = qrshr::<PRECISION, 8>(y_value10 + cb_value);

            let rgba10 = &mut rgba1[0..channels];

            rgba10[dst_chans.get_r_channel_offset()] = r10 as u8;
            rgba10[dst_chans.get_g_channel_offset()] = g10 as u8;
            rgba10[dst_chans.get_b_channel_offset()] = b10 as u8;
            if dst_chans.has_alpha() {
                rgba10[dst_chans.get_a_channel_offset()] = 255u8;
            }

            let y_value11 = (y_src1[1] as i32 - bias_y) * range_reduction_y;

            let t11 = y_value11 - cb_value;

            let r11 = qrshr::<PRECISION, 8>(t11 + cr_value);
            let b11 = qrshr::<PRECISION, 8>(t11 - cr_value);
            let g11 = qrshr::<PRECISION, 8>(y_value11 + cb_value);

            let rgba11 = &mut rgba1[channels..channels * 2];

            rgba11[dst_chans.get_r_channel_offset()] = r11 as u8;
            rgba11[dst_chans.get_g_channel_offset()] = g11 as u8;
            rgba11[dst_chans.get_b_channel_offset()] = b11 as u8;
            if dst_chans.has_alpha() {
                rgba11[dst_chans.get_a_channel_offset()] = 255u8;
            }
        }

        if image.width & 1 != 0 {
            let y_value0 = (*y_plane0.last().unwrap() as i32 - bias_y) * range_reduction_y;
            let y_value1 = (*y_plane1.last().unwrap() as i32 - bias_y) * range_reduction_y;
            let cb_value = (*u_plane.last().unwrap() as i32 - bias_uv) * range_reduction_uv;
            let cr_value = (*v_plane.last().unwrap() as i32 - bias_uv) * range_reduction_uv;
            let rgba = rgba0.chunks_exact_mut(channels).last().unwrap();
            let rgba0 = &mut rgba[0..channels];

            let t0 = y_value0 - cb_value;

            let r0 = qrshr::<PRECISION, 8>(t0 + cr_value);
            let b0 = qrshr::<PRECISION, 8>(t0 - cr_value);
            let g0 = qrshr::<PRECISION, 8>(y_value0 + cb_value);

            rgba0[dst_chans.get_r_channel_offset()] = r0 as u8;
            rgba0[dst_chans.get_g_channel_offset()] = g0 as u8;
            rgba0[dst_chans.get_b_channel_offset()] = b0 as u8;
            if dst_chans.has_alpha() {
                rgba0[dst_chans.get_a_channel_offset()] = 255;
            }

            let t1 = y_value1 - cb_value;

            let r1 = qrshr::<PRECISION, 8>(t1 + cr_value);
            let b1 = qrshr::<PRECISION, 8>(t1 - cr_value);
            let g1 = qrshr::<PRECISION, 8>(y_value1 + cb_value);

            let rgba = rgba1.chunks_exact_mut(channels).last().unwrap();
            let rgba1 = &mut rgba[0..channels];
            rgba1[dst_chans.get_r_channel_offset()] = r1 as u8;
            rgba1[dst_chans.get_g_channel_offset()] = g1 as u8;
            rgba1[dst_chans.get_b_channel_offset()] = b1 as u8;
            if dst_chans.has_alpha() {
                rgba1[dst_chans.get_a_channel_offset()] = 255;
            }
        }
    };

    if chroma_subsampling == YuvChromaSubsampling::Yuv444 {
        let iter;
        #[cfg(feature = "rayon")]
        {
            iter = rgba
                .par_chunks_exact_mut(rgba_stride as usize)
                .zip(image.y_plane.par_chunks_exact(image.y_stride as usize))
                .zip(image.u_plane.par_chunks_exact(image.u_stride as usize))
                .zip(image.v_plane.par_chunks_exact(image.v_stride as usize));
        }
        #[cfg(not(feature = "rayon"))]
        {
            iter = rgba
                .chunks_exact_mut(rgba_stride as usize)
                .zip(image.y_plane.chunks_exact(image.y_stride as usize))
                .zip(image.u_plane.chunks_exact(image.u_stride as usize))
                .zip(image.v_plane.chunks_exact(image.v_stride as usize));
        }
        iter.for_each(|(((rgba, y_plane), u_plane), v_plane)| {
            let y_plane = &y_plane[0..image.width as usize];
            for (((rgba, &y_src), &u_src), &v_src) in rgba
                .chunks_exact_mut(channels)
                .zip(y_plane.iter())
                .zip(u_plane.iter())
                .zip(v_plane.iter())
            {
                let y_value = (y_src as i32 - bias_y) * range_reduction_y;
                let cb_value = (u_src as i32 - bias_uv) * range_reduction_uv;
                let cr_value = (v_src as i32 - bias_uv) * range_reduction_uv;

                let t0 = y_value - cb_value;

                let r = qrshr::<PRECISION, 8>(t0 + cr_value);
                let b = qrshr::<PRECISION, 8>(t0 - cr_value);
                let g = qrshr::<PRECISION, 8>(y_value + cb_value);

                rgba[dst_chans.get_r_channel_offset()] = r as u8;
                rgba[dst_chans.get_g_channel_offset()] = g as u8;
                rgba[dst_chans.get_b_channel_offset()] = b as u8;
                if dst_chans.has_alpha() {
                    rgba[dst_chans.get_a_channel_offset()] = 255;
                }
            }
        });
    } else if chroma_subsampling == YuvChromaSubsampling::Yuv422 {
        let iter;
        #[cfg(feature = "rayon")]
        {
            iter = rgba
                .par_chunks_exact_mut(rgba_stride as usize)
                .zip(image.y_plane.par_chunks_exact(image.y_stride as usize))
                .zip(image.u_plane.par_chunks_exact(image.u_stride as usize))
                .zip(image.v_plane.par_chunks_exact(image.v_stride as usize));
        }
        #[cfg(not(feature = "rayon"))]
        {
            iter = rgba
                .chunks_exact_mut(rgba_stride as usize)
                .zip(image.y_plane.chunks_exact(image.y_stride as usize))
                .zip(image.u_plane.chunks_exact(image.u_stride as usize))
                .zip(image.v_plane.chunks_exact(image.v_stride as usize));
        }
        iter.for_each(|(((rgba, y_plane), u_plane), v_plane)| {
            process_halved_chroma_row(
                &y_plane[0..image.width as usize],
                &u_plane[0..(image.width as usize).div_ceil(2)],
                &v_plane[0..(image.width as usize).div_ceil(2)],
                &mut rgba[0..image.width as usize * channels],
            );
        });
    } else if chroma_subsampling == YuvChromaSubsampling::Yuv420 {
        let iter;
        #[cfg(feature = "rayon")]
        {
            iter = rgba
                .par_chunks_exact_mut(rgba_stride as usize * 2)
                .zip(image.y_plane.par_chunks_exact(image.y_stride as usize * 2))
                .zip(image.u_plane.par_chunks_exact(image.u_stride as usize))
                .zip(image.v_plane.par_chunks_exact(image.v_stride as usize));
        }
        #[cfg(not(feature = "rayon"))]
        {
            iter = rgba
                .chunks_exact_mut(rgba_stride as usize * 2)
                .zip(image.y_plane.chunks_exact(image.y_stride as usize * 2))
                .zip(image.u_plane.chunks_exact(image.u_stride as usize))
                .zip(image.v_plane.chunks_exact(image.v_stride as usize));
        }
        iter.for_each(|(((rgba, y_plane), u_plane), v_plane)| {
            let (rgba0, rgba1) = rgba.split_at_mut(rgba_stride as usize);
            let (y_plane0, y_plane1) = y_plane.split_at(image.y_stride as usize);
            process_doubled_chroma_row(
                &y_plane0[0..image.width as usize],
                &y_plane1[0..image.width as usize],
                &u_plane[0..(image.width as usize).div_ceil(2)],
                &v_plane[0..(image.width as usize).div_ceil(2)],
                &mut rgba0[0..image.width as usize * channels],
                &mut rgba1[0..image.width as usize * channels],
            );
        });

        if image.height & 1 != 0 {
            let rgba = rgba.chunks_exact_mut(rgba_stride as usize).last().unwrap();
            let u_plane = image
                .u_plane
                .chunks_exact(image.u_stride as usize)
                .last()
                .unwrap();
            let v_plane = image
                .v_plane
                .chunks_exact(image.v_stride as usize)
                .last()
                .unwrap();
            let y_plane = image
                .y_plane
                .chunks_exact(image.y_stride as usize)
                .last()
                .unwrap();
            process_halved_chroma_row(
                &y_plane[0..image.width as usize],
                &u_plane[0..(image.width as usize).div_ceil(2)],
                &v_plane[0..(image.width as usize).div_ceil(2)],
                &mut rgba[0..image.width as usize * channels],
            );
        }
    } else {
        unreachable!();
    }

    Ok(())
}

/// Convert YCgCo 420 planar format to RGB format.
///
/// This function takes YCgCo 420 planar format data with 8-bit precision,
/// and converts it to RGB format with 8-bit per channel precision.
///
/// # Arguments
///
/// * `planar_image` - Source planar image.
/// * `rgb` - A mutable slice to store the converted RGB data.
/// * `rgb_stride` - Elements per row.
/// * `range` - The YUV range (limited or full).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input RGB data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn ycgco420_to_rgb(
    planar_image: &YuvPlanarImage<u8>,
    rgb: &mut [u8],
    rgb_stride: u32,
    range: YuvRange,
) -> Result<(), YuvError> {
    ycgco_ro_rgbx::<{ YuvSourceChannels::Rgb as u8 }, { YuvChromaSubsampling::Yuv420 as u8 }>(
        planar_image,
        rgb,
        rgb_stride,
        range,
    )
}

/// Convert YCgCo 420 planar format to BGR format.
///
/// This function takes YCgCo 420 planar format data with 8-bit precision,
/// and converts it to BGR format with 8-bit per channel precision.
///
/// # Arguments
///
/// * `planar_image` - Source planar image.
/// * `bgr` - A mutable slice to store the converted BGR data.
/// * `bgr_stride` - Elements per row.
/// * `range` - The YUV range (limited or full).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGR data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn ycgco420_to_bgr(
    planar_image: &YuvPlanarImage<u8>,
    bgr: &mut [u8],
    bgr_stride: u32,
    range: YuvRange,
) -> Result<(), YuvError> {
    ycgco_ro_rgbx::<{ YuvSourceChannels::Bgr as u8 }, { YuvChromaSubsampling::Yuv420 as u8 }>(
        planar_image,
        bgr,
        bgr_stride,
        range,
    )
}

/// Convert YCgCo 420 planar format to RGBA format.
///
/// This function takes YCgCo 420 planar format data with 8-bit precision,
/// and converts it to RGBA format with 8-bit per channel precision.
///
/// # Arguments
///
/// * `planar_image` - Source planar image.
/// * `rgba` - A mutable slice to store the converted RGBA data.
/// * `rgba_stride` - Elements per row.
/// * `range` - The YUV range (limited or full).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGRA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn ycgco420_to_rgba(
    planar_image: &YuvPlanarImage<u8>,
    rgba: &mut [u8],
    rgba_stride: u32,
    range: YuvRange,
) -> Result<(), YuvError> {
    ycgco_ro_rgbx::<{ YuvSourceChannels::Rgba as u8 }, { YuvChromaSubsampling::Yuv420 as u8 }>(
        planar_image,
        rgba,
        rgba_stride,
        range,
    )
}

/// Convert YCgCo 420 planar format to BGRA format.
///
/// This function takes YCgCo 420 planar format data with 8-bit precision,
/// and converts it to BGRA format with 8-bit per channel precision.
///
/// # Arguments
///
/// * `planar_image` - Source planar image.
/// * `bgra` - A mutable slice to store the converted BGRA data.
/// * `bgra_stride` - Elements per row.
/// * `range` - The YUV range (limited or full).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGRA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn ycgco420_to_bgra(
    planar_image: &YuvPlanarImage<u8>,
    bgra: &mut [u8],
    bgra_stride: u32,
    range: YuvRange,
) -> Result<(), YuvError> {
    ycgco_ro_rgbx::<{ YuvSourceChannels::Bgra as u8 }, { YuvChromaSubsampling::Yuv420 as u8 }>(
        planar_image,
        bgra,
        bgra_stride,
        range,
    )
}

/// Convert YCgCo 422 planar format to RGB format.
///
/// This function takes YCgCo 422 data with 8-bit precision,
/// and converts it to RGB format with 8-bit per channel precision.
///
/// # Arguments
///
/// * `planar_image` - Source planar image.
/// * `width` - The width of the YUV image.
/// * `height` - The height of the YUV image.
/// * `rgb` - A mutable slice to store the converted RGB data.
/// * `rgb_stride` - Elements per row.
/// * `range` - The YUV range (limited or full).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input RGB data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn ycgco422_to_rgb(
    planar_image: &YuvPlanarImage<u8>,
    rgb: &mut [u8],
    rgb_stride: u32,
    range: YuvRange,
) -> Result<(), YuvError> {
    ycgco_ro_rgbx::<{ YuvSourceChannels::Rgb as u8 }, { YuvChromaSubsampling::Yuv422 as u8 }>(
        planar_image,
        rgb,
        rgb_stride,
        range,
    )
}

/// Convert YCgCo 422 planar format to BGR format.
///
/// This function takes YCgCo 422 data with 8-bit precision,
/// and converts it to BGR format with 8-bit per channel precision.
///
/// # Arguments
///
/// * `planar_image` - Source planar image.
/// * `bgr` - A mutable slice to store the converted BGR data.
/// * `bgr_stride` - Elements per row.
/// * `range` - The YUV range (limited or full).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGR data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn ycgco422_to_bgr(
    planar_image: &YuvPlanarImage<u8>,
    bgr: &mut [u8],
    bgr_stride: u32,
    range: YuvRange,
) -> Result<(), YuvError> {
    ycgco_ro_rgbx::<{ YuvSourceChannels::Bgr as u8 }, { YuvChromaSubsampling::Yuv422 as u8 }>(
        planar_image,
        bgr,
        bgr_stride,
        range,
    )
}

/// Convert YCgCo 422 planar format to RGBA format.
///
/// This function takes YCgCo 422 data with 8-bit precision,
/// and converts it to RGBA format with 8-bit per channel precision.
///
/// # Arguments
///
/// * `planar_image` - Source planar image.
/// * `rgba` - A mutable slice to store the converted RGBA data.
/// * `rgba_stride` - Elements per row.
/// * `range` - The YUV range (limited or full).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGRA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn ycgco422_to_rgba(
    planar_image: &YuvPlanarImage<u8>,
    rgba: &mut [u8],
    rgba_stride: u32,
    range: YuvRange,
) -> Result<(), YuvError> {
    ycgco_ro_rgbx::<{ YuvSourceChannels::Rgba as u8 }, { YuvChromaSubsampling::Yuv422 as u8 }>(
        planar_image,
        rgba,
        rgba_stride,
        range,
    )
}

/// Convert YCgCo 422 planar format to BGRA format.
///
/// This function takes YCgCo 422 data with 8-bit precision,
/// and converts it to BGRA format with 8-bit per channel precision.
///
/// # Arguments
///
/// * `planar_image` - Source planar image.
/// * `bgra` - A mutable slice to store the converted BGRA data.
/// * `bgra_stride` - Elements per row.
/// * `range` - The YUV range (limited or full).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGRA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn ycgco422_to_bgra(
    planar_image: &YuvPlanarImage<u8>,
    bgra: &mut [u8],
    bgra_stride: u32,
    range: YuvRange,
) -> Result<(), YuvError> {
    ycgco_ro_rgbx::<{ YuvSourceChannels::Bgra as u8 }, { YuvChromaSubsampling::Yuv422 as u8 }>(
        planar_image,
        bgra,
        bgra_stride,
        range,
    )
}

/// Convert YCgCo 444 planar format to RGBA format.
///
/// This function takes YCgCo 444 data with 8-bit precision,
/// and converts it to RGBA format with 8-bit per channel precision.
///
/// # Arguments
///
/// * `planar_image` - Source planar image.
/// * `rgba` - A mutable slice to store the converted RGBA data.
/// * `rgba_stride` - Elements per row.
/// * `range` - The YUV range (limited or full).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGRA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn ycgco444_to_rgba(
    planar_image: &YuvPlanarImage<u8>,
    rgba: &mut [u8],
    rgba_stride: u32,
    range: YuvRange,
) -> Result<(), YuvError> {
    ycgco_ro_rgbx::<{ YuvSourceChannels::Rgba as u8 }, { YuvChromaSubsampling::Yuv444 as u8 }>(
        planar_image,
        rgba,
        rgba_stride,
        range,
    )
}

/// Convert YCgCo 444 planar format to BGRA format.
///
/// This function takes YCgCo 444 data with 8-bit precision,
/// and converts it to BGRA format with 8-bit per channel precision.
///
/// # Arguments
///
/// * `planar_image` - Source planar image.
/// * `bgra` - A mutable slice to store the converted BGRA data.
/// * `bgra_stride` - Elements per row.
/// * `range` - The YUV range (limited or full).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGRA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn ycgco444_to_bgra(
    planar_image: &YuvPlanarImage<u8>,
    bgra: &mut [u8],
    bgra_stride: u32,
    range: YuvRange,
) -> Result<(), YuvError> {
    ycgco_ro_rgbx::<{ YuvSourceChannels::Bgra as u8 }, { YuvChromaSubsampling::Yuv444 as u8 }>(
        planar_image,
        bgra,
        bgra_stride,
        range,
    )
}

/// Convert YCgCo 444 planar format to RGB format.
///
/// This function takes YCgCo 444 data with 8-bit precision,
/// and converts it to RGB format with 8-bit per channel precision.
///
/// # Arguments
///
/// * `planar_image` - Source planar image.
/// * `rgb` - A mutable slice to store the converted RGB data.
/// * `rgb_stride` - Elements per row.
/// * `range` - The YUV range (limited or full).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input RGB data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn ycgco444_to_rgb(
    planar_image: &YuvPlanarImage<u8>,
    rgb: &mut [u8],
    rgb_stride: u32,
    range: YuvRange,
) -> Result<(), YuvError> {
    ycgco_ro_rgbx::<{ YuvSourceChannels::Rgb as u8 }, { YuvChromaSubsampling::Yuv444 as u8 }>(
        planar_image,
        rgb,
        rgb_stride,
        range,
    )
}

/// Convert YCgCo 444 planar format to BGR format.
///
/// This function takes YCgCo 444 data with 8-bit precision,
/// and converts it to BGR format with 8-bit per channel precision.
///
/// # Arguments
///
/// * `planar_image` - Source planar image.
/// * `bgr` - A mutable slice to store the converted BGR data.
/// * `bgr_stride` - Elements per row.
/// * `range` - The YUV range (limited or full).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGR data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn ycgco444_to_bgr(
    planar_image: &YuvPlanarImage<u8>,
    bgr: &mut [u8],
    bgr_stride: u32,
    range: YuvRange,
) -> Result<(), YuvError> {
    ycgco_ro_rgbx::<{ YuvSourceChannels::Bgr as u8 }, { YuvChromaSubsampling::Yuv444 as u8 }>(
        planar_image,
        bgr,
        bgr_stride,
        range,
    )
}
