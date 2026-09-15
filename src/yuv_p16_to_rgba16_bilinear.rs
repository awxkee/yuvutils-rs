/*
 * Copyright (c) Radzivon Bartoshyk, 6/2025. All rights reserved.
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
use crate::images::projected_rgba_plane_mut;
use crate::numerics::qrshr;
use crate::yuv_error::check_rgba_destination;
use crate::yuv_support::*;
use crate::{YuvError, YuvPlanarImage};
#[cfg(feature = "rayon")]
use rayon::iter::{IndexedParallelIterator, ParallelIterator};
#[cfg(feature = "rayon")]
use rayon::prelude::{ParallelSlice, ParallelSliceMut};

type OneRowInterpolator = fn(
    range: &YuvChromaRange,
    transform: &CbCrInverseTransform<i16>,
    y_plane: &[u16],
    u_plane: &[u16],
    v_plane: &[u16],
    rgba: &mut [u16],
    width: u32,
);

type DoubleRowInterpolator = fn(
    range: &YuvChromaRange,
    transform: &CbCrInverseTransform<i16>,
    y_plane: &[u16],
    u_plane0: &[u16],
    u_plane1: &[u16],
    v_plane0: &[u16],
    v_plane1: &[u16],
    rgba: &mut [u16],
    width: u32,
);

#[allow(dead_code)]
fn interpolate_1_row<const DESTINATION_CHANNELS: u8, const Q: i32, const BIT_DEPTH: usize>(
    range: &YuvChromaRange,
    transform: &CbCrInverseTransform<i16>,
    y_plane: &[u16],
    u_plane: &[u16],
    v_plane: &[u16],
    rgba: &mut [u16],
    _: u32,
) {
    let dst_chans: YuvSourceChannels = DESTINATION_CHANNELS.into();
    let channels = dst_chans.get_channels_count();

    let cr_coef = transform.cr_coef;
    let cb_coef = transform.cb_coef;
    let y_coef = transform.y_coef;
    let g_coef_1 = transform.g_coeff_1;
    let g_coef_2 = transform.g_coeff_2;

    let bias_y = range.bias_y as i32;
    let bias_uv = range.bias_uv as i32;

    let max_colors = ((1u32 << BIT_DEPTH) - 1) as u16;

    // Bilinear upscaling weights in Q0.2

    let blended_pairs = (y_plane.len() / 2).min(u_plane.len().saturating_sub(1));
    let (y_head, y_reminder) = y_plane.split_at(blended_pairs * 2);
    let (rgba_head, rgba_reminder) = rgba.split_at_mut(blended_pairs * channels * 2);

    for (((rgba, y_src), u_src), v_src) in rgba_head
        .chunks_exact_mut(channels * 2)
        .zip(y_head.chunks_exact(2))
        .zip(u_plane.windows(2))
        .zip(v_plane.windows(2))
    {
        let cb_0 = ((u_src[0] as u32 * 3 + u_src[1] as u32 + 2) >> 2) as u16;
        let cr_0 = ((v_src[0] as u32 * 3 + v_src[1] as u32 + 2) >> 2) as u16;

        let cb_1 = ((u_src[0] as u32 + (u_src[1] as u32) * 3 + 2) >> 2) as u16;
        let cr_1 = ((v_src[0] as u32 + (v_src[1] as u32) * 3 + 2) >> 2) as u16;

        let y_value0 = (y_src[0] as i32 - bias_y) * y_coef as i32;
        let cb_value0 = cb_0 as i32 - bias_uv;
        let cr_value0 = cr_0 as i32 - bias_uv;

        let r0 = qrshr::<Q, BIT_DEPTH>(y_value0 + cr_coef as i32 * cr_value0);
        let b0 = qrshr::<Q, BIT_DEPTH>(y_value0 + cb_coef as i32 * cb_value0);
        let g0 = qrshr::<Q, BIT_DEPTH>(
            y_value0 - g_coef_1 as i32 * cr_value0 - g_coef_2 as i32 * cb_value0,
        );

        let rgba0 = &mut rgba[..channels];

        rgba0[dst_chans.get_r_channel_offset()] = r0 as u16;
        rgba0[dst_chans.get_g_channel_offset()] = g0 as u16;
        rgba0[dst_chans.get_b_channel_offset()] = b0 as u16;
        if dst_chans.has_alpha() {
            rgba0[dst_chans.get_a_channel_offset()] = max_colors;
        }

        let y_value1 = (y_src[1] as i32 - bias_y) * y_coef as i32;
        let cb_value1 = cb_1 as i32 - bias_uv;
        let cr_value1 = cr_1 as i32 - bias_uv;

        let r0 = qrshr::<Q, BIT_DEPTH>(y_value1 + cr_coef as i32 * cr_value1);
        let b0 = qrshr::<Q, BIT_DEPTH>(y_value1 + cb_coef as i32 * cb_value1);
        let g0 = qrshr::<Q, BIT_DEPTH>(
            y_value1 - g_coef_1 as i32 * cr_value1 - g_coef_2 as i32 * cb_value1,
        );

        let rgba1 = &mut rgba[channels..channels * 2];

        rgba1[dst_chans.get_r_channel_offset()] = r0 as u16;
        rgba1[dst_chans.get_g_channel_offset()] = g0 as u16;
        rgba1[dst_chans.get_b_channel_offset()] = b0 as u16;
        if dst_chans.has_alpha() {
            rgba1[dst_chans.get_a_channel_offset()] = max_colors;
        }
    }

    if let (Some(&u_last), Some(&v_last)) = (u_plane.last(), v_plane.last()) {
        let cb_value = u_last as i32 - bias_uv;
        let cr_value = v_last as i32 - bias_uv;

        for (rgba0, y_src) in rgba_reminder
            .chunks_exact_mut(channels)
            .zip(y_reminder.iter())
        {
            let y_value0 = (*y_src as i32 - bias_y) * y_coef as i32;

            let r0 = qrshr::<Q, BIT_DEPTH>(y_value0 + cr_coef as i32 * cr_value);
            let b0 = qrshr::<Q, BIT_DEPTH>(y_value0 + cb_coef as i32 * cb_value);
            let g0 = qrshr::<Q, BIT_DEPTH>(
                y_value0 - g_coef_1 as i32 * cr_value - g_coef_2 as i32 * cb_value,
            );

            rgba0[dst_chans.get_r_channel_offset()] = r0 as u16;
            rgba0[dst_chans.get_g_channel_offset()] = g0 as u16;
            rgba0[dst_chans.get_b_channel_offset()] = b0 as u16;
            if dst_chans.has_alpha() {
                rgba0[dst_chans.get_a_channel_offset()] = max_colors;
            }
        }
    }
}

#[allow(dead_code)]
fn interpolate_2_rows<const DESTINATION_CHANNELS: u8, const Q: i32, const BIT_DEPTH: usize>(
    range: &YuvChromaRange,
    transform: &CbCrInverseTransform<i16>,
    y_plane: &[u16],
    u_plane0: &[u16],
    u_plane1: &[u16],
    v_plane0: &[u16],
    v_plane1: &[u16],
    rgba: &mut [u16],
    _: u32,
) {
    let dst_chans: YuvSourceChannels = DESTINATION_CHANNELS.into();
    let channels = dst_chans.get_channels_count();

    let cr_coef = transform.cr_coef;
    let cb_coef = transform.cb_coef;
    let y_coef = transform.y_coef;
    let g_coef_1 = transform.g_coeff_1;
    let g_coef_2 = transform.g_coeff_2;

    let bias_y = range.bias_y as i32;
    let bias_uv = range.bias_uv as i32;

    let max_colors = ((1u32 << BIT_DEPTH) - 1) as u16;

    // Bilinear upscaling weights in Q0.4

    let chroma_width = u_plane0
        .len()
        .min(u_plane1.len())
        .min(v_plane0.len())
        .min(v_plane1.len());
    let blended_pairs = (y_plane.len() / 2).min(chroma_width.saturating_sub(1));
    let (y_head, y_reminder) = y_plane.split_at(blended_pairs * 2);
    let (rgba_head, rgba_reminder) = rgba.split_at_mut(blended_pairs * channels * 2);

    for (((((rgba0, y_src0), u_src), u_src_next), v_src), v_src_next) in rgba_head
        .chunks_exact_mut(channels * 2)
        .zip(y_head.chunks_exact(2))
        .zip(u_plane0.windows(2))
        .zip(u_plane1.windows(2))
        .zip(v_plane0.windows(2))
        .zip(v_plane1.windows(2))
    {
        let cb_0 = (u_src[0] as u32 * 9
            + u_src[1] as u32 * 3
            + u_src_next[0] as u32 * 3
            + u_src_next[1] as u32
            + (1 << 3))
            >> 4;
        let cr_0 = (v_src[0] as u32 * 9
            + v_src[1] as u32 * 3
            + v_src_next[0] as u32 * 3
            + v_src_next[1] as u32
            + (1 << 3))
            >> 4;

        let cb_1 = (u_src[0] as u32 * 3
            + u_src[1] as u32 * 9
            + u_src_next[0] as u32
            + u_src_next[1] as u32 * 3
            + (1 << 3))
            >> 4;
        let cr_1 = (v_src[0] as u32 * 3
            + v_src[1] as u32 * 9
            + v_src_next[0] as u32
            + v_src_next[1] as u32 * 3
            + (1 << 3))
            >> 4;

        let y_value0 = (y_src0[0] as i32 - bias_y) * y_coef as i32;
        let cb_value0 = cb_0 as i32 - bias_uv;
        let cr_value0 = cr_0 as i32 - bias_uv;

        let g_built_coeff0 = -g_coef_1 as i32 * cr_value0 - g_coef_2 as i32 * cb_value0;

        let r0 = qrshr::<Q, BIT_DEPTH>(y_value0 + cr_coef as i32 * cr_value0);
        let b0 = qrshr::<Q, BIT_DEPTH>(y_value0 + cb_coef as i32 * cb_value0);
        let g0 = qrshr::<Q, BIT_DEPTH>(y_value0 + g_built_coeff0);

        let rgba00 = &mut rgba0[..channels];

        rgba00[dst_chans.get_r_channel_offset()] = r0 as u16;
        rgba00[dst_chans.get_g_channel_offset()] = g0 as u16;
        rgba00[dst_chans.get_b_channel_offset()] = b0 as u16;
        if dst_chans.has_alpha() {
            rgba00[dst_chans.get_a_channel_offset()] = max_colors;
        }

        let y_value1 = (y_src0[1] as i32 - bias_y) * y_coef as i32;
        let cb_value1 = cb_1 as i32 - bias_uv;
        let cr_value1 = cr_1 as i32 - bias_uv;

        let g_built_coeff1 = -g_coef_1 as i32 * cr_value1 - g_coef_2 as i32 * cb_value1;

        let r1 = qrshr::<Q, BIT_DEPTH>(y_value1 + cr_coef as i32 * cr_value1);
        let b1 = qrshr::<Q, BIT_DEPTH>(y_value1 + cb_coef as i32 * cb_value1);
        let g1 = qrshr::<Q, BIT_DEPTH>(y_value1 + g_built_coeff1);

        let rgba01 = &mut rgba0[channels..channels * 2];

        rgba01[dst_chans.get_r_channel_offset()] = r1 as u16;
        rgba01[dst_chans.get_g_channel_offset()] = g1 as u16;
        rgba01[dst_chans.get_b_channel_offset()] = b1 as u16;
        if dst_chans.has_alpha() {
            rgba01[dst_chans.get_a_channel_offset()] = max_colors;
        }
    }

    if let (Some(&u0_last), Some(&u1_last), Some(&v0_last), Some(&v1_last)) = (
        u_plane0.last(),
        u_plane1.last(),
        v_plane0.last(),
        v_plane1.last(),
    ) {
        let cb_value = ((u0_last as u32 * 3 + u1_last as u32 + 2) >> 2) as i32 - bias_uv;
        let cr_value = ((v0_last as u32 + v1_last as u32 * 3 + 2) >> 2) as i32 - bias_uv;
        let g_built_coeff = -g_coef_1 as i32 * cr_value - g_coef_2 as i32 * cb_value;

        for (rgba0, y_src0) in rgba_reminder
            .chunks_exact_mut(channels)
            .zip(y_reminder.iter())
        {
            let y_value0 = (*y_src0 as i32 - bias_y) * y_coef as i32;

            let r0 = qrshr::<Q, BIT_DEPTH>(y_value0 + cr_coef as i32 * cr_value);
            let b0 = qrshr::<Q, BIT_DEPTH>(y_value0 + cb_coef as i32 * cb_value);
            let g0 = qrshr::<Q, BIT_DEPTH>(y_value0 + g_built_coeff);

            rgba0[dst_chans.get_r_channel_offset()] = r0 as u16;
            rgba0[dst_chans.get_g_channel_offset()] = g0 as u16;
            rgba0[dst_chans.get_b_channel_offset()] = b0 as u16;
            if dst_chans.has_alpha() {
                rgba0[dst_chans.get_a_channel_offset()] = max_colors;
            }
        }
    }
}

fn make_1_row_interpolator<const DESTINATION_CHANNELS: u8, const Q: i32, const BIT_DEPTH: usize>(
) -> OneRowInterpolator {
    if BIT_DEPTH <= 14 {
        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        {
            use crate::neon::neon_planar16_bilinear_1_row_rgba16;
            return neon_planar16_bilinear_1_row_rgba16::<DESTINATION_CHANNELS, Q, BIT_DEPTH>;
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            #[cfg(feature = "avx")]
            if std::arch::is_x86_feature_detected!("avx2") {
                use crate::avx2::avx_planar16_bilinear_1_row_rgba;
                return avx_planar16_bilinear_1_row_rgba::<DESTINATION_CHANNELS, Q, BIT_DEPTH>;
            }
            #[cfg(feature = "sse")]
            if std::arch::is_x86_feature_detected!("sse4.1") {
                use crate::sse::sse_planar16_bilinear_1_row_rgba;
                return sse_planar16_bilinear_1_row_rgba::<DESTINATION_CHANNELS, Q, BIT_DEPTH>;
            }
        }
    }
    interpolate_1_row::<DESTINATION_CHANNELS, Q, BIT_DEPTH>
}

fn make_2_rows_interpolator<
    const DESTINATION_CHANNELS: u8,
    const Q: i32,
    const BIT_DEPTH: usize,
>() -> DoubleRowInterpolator {
    if BIT_DEPTH <= 14 {
        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        {
            use crate::neon::neon_planar16_bilinear_2_rows_rgba;
            return neon_planar16_bilinear_2_rows_rgba::<DESTINATION_CHANNELS, Q, BIT_DEPTH>;
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            #[cfg(feature = "avx")]
            if std::arch::is_x86_feature_detected!("avx2") {
                use crate::avx2::avx_planar16_bilinear_2_rows_rgba;
                return avx_planar16_bilinear_2_rows_rgba::<DESTINATION_CHANNELS, Q, BIT_DEPTH>;
            }
            #[cfg(feature = "sse")]
            if std::arch::is_x86_feature_detected!("sse4.1") {
                use crate::sse::sse_planar16_bilinear_2_rows_rgba;
                return sse_planar16_bilinear_2_rows_rgba::<DESTINATION_CHANNELS, Q, BIT_DEPTH>;
            }
        }
    }
    interpolate_2_rows::<DESTINATION_CHANNELS, Q, BIT_DEPTH>
}

fn yuv16_to_rgbx_impl_bilinear<
    const DESTINATION_CHANNELS: u8,
    const SAMPLING: u8,
    const Q: i32,
    const BIT_DEPTH: usize,
>(
    image: &YuvPlanarImage<u16>,
    rgba: &mut [u16],
    rgba_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) -> Result<(), YuvError> {
    let chroma_subsampling: YuvChromaSubsampling = SAMPLING.into();
    assert_ne!(chroma_subsampling, YuvChromaSubsampling::Yuv444);
    let dst_chans: YuvSourceChannels = DESTINATION_CHANNELS.into();
    let channels = dst_chans.get_channels_count();

    check_rgba_destination(rgba, rgba_stride, image.width, image.height, channels)?;
    image.check_constraints(chroma_subsampling)?;

    let chroma_range = get_yuv_range(BIT_DEPTH as u32, range);
    let kr_kb = matrix.get_kr_kb();

    let inverse_transform =
        search_inverse_transform(Q, BIT_DEPTH as u32, range, matrix, chroma_range, kr_kb).cast();

    let one_row_interpolator = make_1_row_interpolator::<DESTINATION_CHANNELS, Q, BIT_DEPTH>();
    let two_rows_interpolator = make_2_rows_interpolator::<DESTINATION_CHANNELS, Q, BIT_DEPTH>();
    let (y_plane, u_plane, v_plane) = image.projected_planes(chroma_subsampling);
    let rgba = projected_rgba_plane_mut(rgba, image.width, image.height, rgba_stride, dst_chans);

    if chroma_subsampling == YuvChromaSubsampling::Yuv422 {
        let iter;
        #[cfg(feature = "rayon")]
        {
            iter = rgba
                .par_chunks_mut(rgba_stride as usize)
                .zip(y_plane.par_chunks(image.y_stride as usize))
                .zip(u_plane.par_chunks(image.u_stride as usize))
                .zip(v_plane.par_chunks(image.v_stride as usize));
        }
        #[cfg(not(feature = "rayon"))]
        {
            iter = rgba
                .chunks_mut(rgba_stride as usize)
                .zip(y_plane.chunks(image.y_stride as usize))
                .zip(u_plane.chunks(image.u_stride as usize))
                .zip(v_plane.chunks(image.v_stride as usize));
        }
        iter.take(image.height as usize)
            .for_each(|(((rgba, y_plane), u_plane), v_plane)| {
                one_row_interpolator(
                    &chroma_range,
                    &inverse_transform,
                    &y_plane[..image.width as usize],
                    &u_plane[..(image.width as usize).div_ceil(2)],
                    &v_plane[..(image.width as usize).div_ceil(2)],
                    &mut rgba[..image.width as usize * channels],
                    image.width,
                );
            });
    } else if chroma_subsampling == YuvChromaSubsampling::Yuv420 {
        let width = image.width as usize;
        let chroma_width = width.div_ceil(2);
        let chroma_height = (image.height as usize).div_ceil(2);
        let row_pairs = image.height as usize / 2;
        let process_row_pair = |row: usize, rgba: &mut [u16], y_plane: &[u16]| {
            let (y_plane0, y_plane1) = y_plane.split_at(image.y_stride as usize);
            let (rgba0, rgba1) = rgba.split_at_mut(rgba_stride as usize);
            let next_row = (row + 1).min(chroma_height - 1);
            let u0_start = row * image.u_stride as usize;
            let u1_start = next_row * image.u_stride as usize;
            let v0_start = row * image.v_stride as usize;
            let v1_start = next_row * image.v_stride as usize;
            let u_plane0 = &u_plane[u0_start..u0_start + chroma_width];
            let u_plane1 = &u_plane[u1_start..u1_start + chroma_width];
            let v_plane0 = &v_plane[v0_start..v0_start + chroma_width];
            let v_plane1 = &v_plane[v1_start..v1_start + chroma_width];

            two_rows_interpolator(
                &chroma_range,
                &inverse_transform,
                &y_plane0[..width],
                u_plane0,
                u_plane1,
                v_plane0,
                v_plane1,
                &mut rgba0[..width * channels],
                image.width,
            );
            two_rows_interpolator(
                &chroma_range,
                &inverse_transform,
                &y_plane1[..width],
                u_plane1,
                u_plane0,
                v_plane1,
                v_plane0,
                &mut rgba1[..width * channels],
                image.width,
            );
        };

        #[cfg(feature = "rayon")]
        {
            rgba.par_chunks_mut(rgba_stride as usize * 2)
                .zip(y_plane.par_chunks(image.y_stride as usize * 2))
                .enumerate()
                .take(row_pairs)
                .for_each(|(row, (rgba, y_plane))| process_row_pair(row, rgba, y_plane));
        }
        #[cfg(not(feature = "rayon"))]
        {
            rgba.chunks_mut(rgba_stride as usize * 2)
                .zip(y_plane.chunks(image.y_stride as usize * 2))
                .enumerate()
                .take(row_pairs)
                .for_each(|(row, (rgba, y_plane))| process_row_pair(row, rgba, y_plane));
        }

        if image.height & 1 != 0 {
            let last_y_row = image.height as usize - 1;
            let last_chroma_row = chroma_height - 1;
            let rgba_start = last_y_row * rgba_stride as usize;
            let y_start = last_y_row * image.y_stride as usize;
            let u_start = last_chroma_row * image.u_stride as usize;
            let v_start = last_chroma_row * image.v_stride as usize;
            one_row_interpolator(
                &chroma_range,
                &inverse_transform,
                &y_plane[y_start..y_start + width],
                &u_plane[u_start..u_start + chroma_width],
                &v_plane[v_start..v_start + chroma_width],
                &mut rgba[rgba_start..rgba_start + width * channels],
                image.width,
            );
        }
    } else {
        unreachable!();
    }

    Ok(())
}

fn yuv_to_rgbx<const DESTINATION_CHANNELS: u8, const SAMPLING: u8, const BIT_DEPTH: usize>(
    image: &YuvPlanarImage<u16>,
    rgba: &mut [u16],
    rgba_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) -> Result<(), YuvError> {
    yuv16_to_rgbx_impl_bilinear::<DESTINATION_CHANNELS, SAMPLING, 13, BIT_DEPTH>(
        image,
        rgba,
        rgba_stride,
        range,
        matrix,
    )
}

macro_rules! d_cnv {
    ($method: ident, $px_fmt: expr, $sampling: expr, $sampling_written: expr, $px_written: expr, $px_written_small: expr, $bit_depth: expr) => {
        #[doc = concat!("
Convert ",$sampling_written, " planar format with ", stringify!($bit_depth), " bit pixel format to ", $px_written," ", stringify!($bit_depth), " bit-depth format using bi-linear upsampling.

This function takes ", $sampling_written, " planar data with ", stringify!($bit_depth), " bit precision.
and converts it to ", $px_written," format with ", stringify!($bit_depth), " bit-depth precision per channel
with bilinear upscampling.

# Arguments

* `planar_image` - Source ",$sampling_written," planar image.
* `", $px_written_small, "` - A mutable slice to store the converted ", $px_written," ", stringify!($bit_depth), " bit-depth data.
* `", $px_written_small, "_stride` - The stride (components per row) for ", $px_written," ", stringify!($bit_depth), " bit-depth data.
* `range` - The YUV range (limited or full).
* `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).

# Panics

This function panics if the lengths of the planes or the input ", $px_written," data are not valid based
on the specified width, height, and strides, or if invalid YUV range or matrix is provided.")]
        pub fn $method(
            planar_image: &YuvPlanarImage<u16>,
            dst: &mut [u16],
            dst_stride: u32,
            range: YuvRange,
            matrix: YuvStandardMatrix,
        ) -> Result<(), YuvError> {
            yuv_to_rgbx::<{ $px_fmt as u8 },
                            { $sampling as u8 }, $bit_depth>(
                planar_image, dst, dst_stride, range, matrix)
        }
    };
}

d_cnv!(
    i010_to_rgba10_bilinear,
    YuvSourceChannels::Rgba,
    YuvChromaSubsampling::Yuv420,
    "I010",
    "RGBA",
    "rgba",
    10
);
d_cnv!(
    i010_to_rgb10_bilinear,
    YuvSourceChannels::Rgb,
    YuvChromaSubsampling::Yuv420,
    "I010",
    "RGB",
    "rgb",
    10
);
d_cnv!(
    i210_to_rgba10_bilinear,
    YuvSourceChannels::Rgba,
    YuvChromaSubsampling::Yuv422,
    "I210",
    "RGBA",
    "rgba",
    10
);
d_cnv!(
    i210_to_rgb10_bilinear,
    YuvSourceChannels::Rgb,
    YuvChromaSubsampling::Yuv422,
    "I210",
    "RGB",
    "rgb",
    10
);

d_cnv!(
    i012_to_rgba12_bilinear,
    YuvSourceChannels::Rgba,
    YuvChromaSubsampling::Yuv420,
    "I012",
    "RGBA",
    "rgba",
    12
);
d_cnv!(
    i012_to_rgb12_bilinear,
    YuvSourceChannels::Rgb,
    YuvChromaSubsampling::Yuv420,
    "I012",
    "RGB",
    "rgb",
    12
);
d_cnv!(
    i212_to_rgba12_bilinear,
    YuvSourceChannels::Rgba,
    YuvChromaSubsampling::Yuv422,
    "I212",
    "RGBA",
    "rgba",
    12
);
d_cnv!(
    i212_to_rgb12_bilinear,
    YuvSourceChannels::Rgb,
    YuvChromaSubsampling::Yuv422,
    "I212",
    "RGB",
    "rgb",
    12
);

// 4:2:0, 14 bit

d_cnv!(
    i014_to_rgba14_bilinear,
    YuvSourceChannels::Rgba,
    YuvChromaSubsampling::Yuv420,
    "I014",
    "RGBA",
    "rgba",
    14
);
d_cnv!(
    i014_to_rgb14_bilinear,
    YuvSourceChannels::Rgb,
    YuvChromaSubsampling::Yuv420,
    "I014",
    "RGB",
    "rgb",
    14
);

// 14-bit, 4:2:2

d_cnv!(
    i214_to_rgba14_bilinear,
    YuvSourceChannels::Rgba,
    YuvChromaSubsampling::Yuv422,
    "I214",
    "RGBA",
    "rgba",
    14
);
d_cnv!(
    i214_to_rgb14_bilinear,
    YuvSourceChannels::Rgb,
    YuvChromaSubsampling::Yuv422,
    "I214",
    "RGB",
    "rgb",
    14
);

// 4:2:0, 16 bit

d_cnv!(
    i016_to_rgba16_bilinear,
    YuvSourceChannels::Rgba,
    YuvChromaSubsampling::Yuv420,
    "I016",
    "RGBA",
    "rgba",
    16
);
d_cnv!(
    i016_to_rgb16_bilinear,
    YuvSourceChannels::Rgb,
    YuvChromaSubsampling::Yuv420,
    "I016",
    "RGB",
    "rgb",
    16
);

// 16-bit, 4:2:2

d_cnv!(
    i216_to_rgba16_bilinear,
    YuvSourceChannels::Rgba,
    YuvChromaSubsampling::Yuv422,
    "I216",
    "RGBA",
    "rgba",
    16
);
d_cnv!(
    i216_to_rgb16_bilinear,
    YuvSourceChannels::Rgb,
    YuvChromaSubsampling::Yuv422,
    "I216",
    "RGB",
    "rgb",
    16
);

#[cfg(test)]
mod tests {
    use super::*;

    fn make_plane(
        width: usize,
        height: usize,
        stride: usize,
        mut value: impl FnMut(usize, usize) -> u16,
    ) -> Vec<u16> {
        let mut plane = vec![0xeeee; stride * (height - 1) + width];
        for row in 0..height {
            for x in 0..width {
                plane[row * stride + x] = value(row, x);
            }
        }
        plane
    }

    #[test]
    fn i010_bilinear_supports_minimal_padded_planes() {
        let width = 7usize;
        let chroma_width = width.div_ceil(2);

        for height in [4usize, 5] {
            let chroma_height = height.div_ceil(2);
            let y_stride = width + 3;
            let u_stride = chroma_width + 2;
            let v_stride = chroma_width + 4;
            let rgba_width = width * 4;
            let rgba_stride = rgba_width + 5;

            let y = make_plane(width, height, y_stride, |row, x| {
                128 + (row * 31 + x * 7) as u16
            });
            let u = make_plane(chroma_width, chroma_height, u_stride, |row, x| {
                384 + (row * 29 + x * 11) as u16
            });
            let v = make_plane(chroma_width, chroma_height, v_stride, |row, x| {
                640 - (row * 17 + x * 5) as u16
            });
            let tight_y = make_plane(width, height, width, |row, x| y[row * y_stride + x]);
            let tight_u = make_plane(chroma_width, chroma_height, chroma_width, |row, x| {
                u[row * u_stride + x]
            });
            let tight_v = make_plane(chroma_width, chroma_height, chroma_width, |row, x| {
                v[row * v_stride + x]
            });

            let image = YuvPlanarImage {
                y_plane: &y,
                y_stride: y_stride as u32,
                u_plane: &u,
                u_stride: u_stride as u32,
                v_plane: &v,
                v_stride: v_stride as u32,
                width: width as u32,
                height: height as u32,
            };
            let tight_image = YuvPlanarImage {
                y_plane: &tight_y,
                y_stride: width as u32,
                u_plane: &tight_u,
                u_stride: chroma_width as u32,
                v_plane: &tight_v,
                v_stride: chroma_width as u32,
                width: width as u32,
                height: height as u32,
            };
            let mut rgba = vec![0x5555; rgba_stride * (height - 1) + rgba_width];
            let mut tight_rgba = vec![0; rgba_width * height];

            i010_to_rgba10_bilinear(
                &image,
                &mut rgba,
                rgba_stride as u32,
                YuvRange::Full,
                YuvStandardMatrix::Bt601,
            )
            .unwrap();
            i010_to_rgba10_bilinear(
                &tight_image,
                &mut tight_rgba,
                rgba_width as u32,
                YuvRange::Full,
                YuvStandardMatrix::Bt601,
            )
            .unwrap();

            for row in 0..height {
                let rgba_row = &rgba[row * rgba_stride..row * rgba_stride + rgba_width];
                let tight_row = &tight_rgba[row * rgba_width..(row + 1) * rgba_width];
                assert_eq!(rgba_row, tight_row, "height {height}, row {row}");
                assert!(rgba_row.chunks_exact(4).all(|pixel| pixel[3] == 1023));
                if row + 1 < height {
                    assert!(
                        rgba[row * rgba_stride + rgba_width..(row + 1) * rgba_stride]
                            .iter()
                            .all(|&value| value == 0x5555)
                    );
                }
            }
        }
    }

    #[test]
    fn i016_bilinear_flat_color_16bit() {
        let width = 4usize;
        let height = 4usize;
        let (y_value, u_value, v_value) = (32768u16, 65535u16, 32768u16);

        let cb = f64::from(u_value) - 32768.;
        let expected = [
            f64::from(y_value),
            f64::from(y_value) - 0.344136 * cb,
            f64::from(y_value) + 1.772 * cb,
        ];
        let check = |rgb: &[u16], method: &str| {
            for row in rgb.chunks_exact(width * 3) {
                for pixel in row.chunks_exact(3) {
                    for (channel, (&value, expected)) in
                        pixel.iter().zip(expected.iter()).enumerate()
                    {
                        let expected = expected.clamp(0., 65535.);
                        assert!(
                            (f64::from(value) - expected).abs() <= 8.,
                            "{method}: channel {channel}: got {value}, expected about {expected}"
                        );
                    }
                }
            }
        };

        fn make_image<'a>(
            y: &'a [u16],
            u: &'a [u16],
            v: &'a [u16],
            width: usize,
            height: usize,
            chroma_width: usize,
        ) -> YuvPlanarImage<'a, u16> {
            YuvPlanarImage {
                y_plane: y,
                y_stride: width as u32,
                u_plane: u,
                u_stride: chroma_width as u32,
                v_plane: v,
                v_stride: chroma_width as u32,
                width: width as u32,
                height: height as u32,
            }
        }

        let y = vec![y_value; width * height];
        let chroma_width = width / 2;
        let mut rgb = vec![0u16; width * height * 3];

        // 4:2:0 goes through the two-row interpolator.
        let u = vec![u_value; chroma_width * (height / 2)];
        let v = vec![v_value; chroma_width * (height / 2)];
        i016_to_rgb16_bilinear(
            &make_image(&y, &u, &v, width, height, chroma_width),
            &mut rgb,
            (width * 3) as u32,
            YuvRange::Full,
            YuvStandardMatrix::Bt601,
        )
        .unwrap();
        check(&rgb, "i016");

        // 4:2:2 goes through the one-row interpolator.
        let u = vec![u_value; chroma_width * height];
        let v = vec![v_value; chroma_width * height];
        i216_to_rgb16_bilinear(
            &make_image(&y, &u, &v, width, height, chroma_width),
            &mut rgb,
            (width * 3) as u32,
            YuvRange::Full,
            YuvStandardMatrix::Bt601,
        )
        .unwrap();
        check(&rgb, "i216");
    }
    #[test]
    fn i216_bilinear_replicates_the_last_chroma_sample() {
        fn image<'a>(
            y: &'a [u16],
            u: &'a [u16],
            v: &'a [u16],
            width: usize,
            height: usize,
            chroma_width: usize,
        ) -> YuvPlanarImage<'a, u16> {
            YuvPlanarImage {
                y_plane: y,
                y_stride: width as u32,
                u_plane: u,
                u_stride: chroma_width as u32,
                v_plane: v,
                v_stride: chroma_width as u32,
                width: width as u32,
                height: height as u32,
            }
        }

        let width = 8usize;
        let height = 2usize;
        let chroma_width = width.div_ceil(2);
        let y = vec![30000u16; width * height];
        let u: Vec<u16> = (0..chroma_width * height)
            .map(|i| 32768 + (i % chroma_width) as u16 * 6000)
            .collect();
        let v: Vec<u16> = (0..chroma_width * height)
            .map(|i| 32768 + (i % chroma_width) as u16 * 3000)
            .collect();

        let mut rgb = vec![0u16; width * height * 3];
        i216_to_rgb16_bilinear(
            &image(&y, &u, &v, width, height, chroma_width),
            &mut rgb,
            (width * 3) as u32,
            YuvRange::Full,
            YuvStandardMatrix::Bt601,
        )
        .unwrap();

        // A two pixel wide image has a single chroma sample
        let narrow_y = vec![30000u16; 2];
        let narrow_u = [*u.last().unwrap()];
        let narrow_v = [*v.last().unwrap()];
        let mut narrow = vec![0u16; 2 * 3];
        i216_to_rgb16_bilinear(
            &image(&narrow_y, &narrow_u, &narrow_v, 2, 1, 1),
            &mut narrow,
            2 * 3,
            YuvRange::Full,
            YuvStandardMatrix::Bt601,
        )
        .unwrap();

        for (row, rgb_row) in rgb.chunks_exact(width * 3).enumerate() {
            let tail = &rgb_row[(width - 2) * 3..];
            assert_eq!(
                tail,
                narrow.as_slice(),
                "row {row}: the last pair is not written with the replicated chroma sample"
            );
        }
    }
}
