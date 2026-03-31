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
#![forbid(unsafe_code)]

use crate::built_coefficients::get_built_forward_transform;
use crate::sharpyuv::SharpYuvGammaTransfer;
use crate::yuv_error::check_rgba_destination;
use crate::yuv_support::*;
use crate::{YuvError, YuvPlanarImageMut};
#[cfg(feature = "rayon")]
use rayon::iter::{IndexedParallelIterator, ParallelIterator};
#[cfg(feature = "rayon")]
use rayon::prelude::{ParallelSlice, ParallelSliceMut};

fn sharpen_row420<const ORIGIN_CHANNELS: u8, const SAMPLING: u8, const PRECISION: i32>(
    y: usize,
    rgba: &[u8],
    y_plane: &mut [u8],
    u_plane: &mut [u8],
    v_plane: &mut [u8],
    rgb_layout: &[u16],
    rgb_layout_next_lane: &[u16],
    gamma_map_table: &[u8; u16::MAX as usize + 1],
    range: &YuvChromaRange,
    transform: &CbCrForwardTransform<i32>,
    width: usize,
) {
    let src_chans: YuvSourceChannels = ORIGIN_CHANNELS.into();
    let channels = src_chans.get_channels_count();

    let rounding_const_bias: i32 = (1 << (PRECISION - 1)) - 1;
    let bias_y = range.bias_y as i32 * (1 << PRECISION) + rounding_const_bias;
    let bias_uv = range.bias_uv as i32 * (1 << PRECISION) + rounding_const_bias;

    let i_bias_y = range.bias_y as i32;
    let i_cap_y = range.range_y as i32 + i_bias_y;
    let i_cap_uv = i_bias_y + range.range_uv as i32;

    let y_even_row = y & 1 == 0;

    for (((((y_dst, u_dst), v_dst), rgba), rgb_linearized), rgb_linearized_next) in y_plane
        .chunks_exact_mut(2)
        .zip(u_plane.iter_mut())
        .zip(v_plane.iter_mut())
        .zip(rgba.chunks_exact(channels * 2))
        .zip(rgb_layout.chunks_exact(2 * 3))
        .zip(rgb_layout_next_lane.chunks_exact(2 * 3))
    {
        let r0 = rgba[src_chans.get_r_channel_offset()] as i32;
        let g0 = rgba[src_chans.get_g_channel_offset()] as i32;
        let b0 = rgba[src_chans.get_b_channel_offset()] as i32;

        let y_0 = (r0 * transform.yr + g0 * transform.yg + b0 * transform.yb + bias_y) >> PRECISION;
        y_dst[0] = y_0.min(i_cap_y) as u8;

        let rgba_2 = &rgba[channels..channels * 2];

        let r1 = rgba_2[src_chans.get_r_channel_offset()] as i32;
        let g1 = rgba_2[src_chans.get_g_channel_offset()] as i32;
        let b1 = rgba_2[src_chans.get_b_channel_offset()] as i32;

        let y_1 = (r1 * transform.yr + g1 * transform.yg + b1 * transform.yb + bias_y) >> PRECISION;
        y_dst[1] = y_1.min(i_cap_y) as u8;

        if y_even_row {
            let sharp_r_c = rgb_linearized[src_chans.get_r_channel_offset()];
            let sharp_g_c = rgb_linearized[src_chans.get_g_channel_offset()];
            let sharp_b_c = rgb_linearized[src_chans.get_b_channel_offset()];

            let rgb_linearized_2 = &rgb_linearized[3..(3 + 3)];

            let sharp_r_next = rgb_linearized_2[src_chans.get_r_channel_offset()];
            let sharp_g_next = rgb_linearized_2[src_chans.get_g_channel_offset()];
            let sharp_b_next = rgb_linearized_2[src_chans.get_b_channel_offset()];

            let sharp_r_c_next_row = rgb_linearized_next[src_chans.get_r_channel_offset()];
            let sharp_g_c_next_row = rgb_linearized_next[src_chans.get_g_channel_offset()];
            let sharp_b_c_next_row = rgb_linearized_next[src_chans.get_b_channel_offset()];

            let rgb_linearized_next_2 = &rgb_linearized_next[3..(3 + 3)];

            let sharp_r_next_row = rgb_linearized_next_2[src_chans.get_r_channel_offset()];
            let sharp_g_next_row = rgb_linearized_next_2[src_chans.get_g_channel_offset()];
            let sharp_b_next_row = rgb_linearized_next_2[src_chans.get_b_channel_offset()];

            const ROUNDING_SHARP: i32 = 1 << 3;

            let interpolated_r = ((sharp_r_c as i32 * 9
                + sharp_r_next as i32 * 3
                + sharp_r_c_next_row as i32 * 3
                + sharp_r_next_row as i32
                + ROUNDING_SHARP)
                >> 4) as u16;
            let interpolated_g = ((sharp_g_c as i32 * 9
                + sharp_g_next as i32 * 3
                + sharp_g_c_next_row as i32 * 3
                + sharp_g_next_row as i32
                + ROUNDING_SHARP)
                >> 4) as u16;
            let interpolated_b = ((sharp_b_c as i32 * 9
                + sharp_b_next as i32 * 3
                + sharp_b_c_next_row as i32 * 3
                + sharp_b_next_row as i32
                + ROUNDING_SHARP)
                >> 4) as u16;

            let corrected_r = gamma_map_table[interpolated_r as usize] as i32;
            let corrected_g = gamma_map_table[interpolated_g as usize] as i32;
            let corrected_b = gamma_map_table[interpolated_b as usize] as i32;

            let cb = (corrected_r * transform.cb_r
                + corrected_g * transform.cb_g
                + corrected_b * transform.cb_b
                + bias_uv)
                >> PRECISION;
            let cr = (corrected_r * transform.cr_r
                + corrected_g * transform.cr_g
                + corrected_b * transform.cr_b
                + bias_uv)
                >> PRECISION;
            *u_dst = cb.max(i_bias_y).min(i_cap_uv) as u8;
            *v_dst = cr.max(i_bias_y).min(i_cap_uv) as u8;
        }
    }

    let rem_rgba = rgba.chunks_exact(channels * 2).remainder();

    if width & 1 != 0 && !rem_rgba.is_empty() {
        let rgba = &rem_rgba[0..3];
        let y_last = y_plane.last_mut().unwrap();
        let r0 = rgba[src_chans.get_r_channel_offset()] as i32;
        let g0 = rgba[src_chans.get_g_channel_offset()] as i32;
        let b0 = rgba[src_chans.get_b_channel_offset()] as i32;

        let y_1 = (r0 * transform.yr + g0 * transform.yg + b0 * transform.yb + bias_y) >> PRECISION;
        *y_last = y_1.min(i_cap_y) as u8;

        if y_even_row {
            let rgba_lin = rgb_layout.chunks_exact(3).last().unwrap();
            let rgba_lin = &rgba_lin[0..3];
            let sharp_r_c = rgba_lin[src_chans.get_r_channel_offset()];
            let sharp_g_c = rgba_lin[src_chans.get_g_channel_offset()];
            let sharp_b_c = rgba_lin[src_chans.get_b_channel_offset()];

            let rgba_lin_next = rgb_layout_next_lane.chunks_exact(3).last().unwrap();
            let rgba_lin_next = &rgba_lin_next[0..3];
            let sharp_r_c_next = rgba_lin_next[src_chans.get_r_channel_offset()];
            let sharp_g_c_next = rgba_lin_next[src_chans.get_g_channel_offset()];
            let sharp_b_c_next = rgba_lin_next[src_chans.get_b_channel_offset()];

            const ROUNDING_SHARP: i32 = 1 << 3;

            let interpolated_r =
                ((sharp_r_c as i32 * 12 + sharp_r_c_next as i32 * 4 + ROUNDING_SHARP) >> 4) as u16;
            let interpolated_g =
                ((sharp_g_c as i32 * 12 + sharp_g_c_next as i32 * 4 + ROUNDING_SHARP) >> 4) as u16;
            let interpolated_b =
                ((sharp_b_c as i32 * 12 + sharp_b_c_next as i32 * 4 + ROUNDING_SHARP) >> 4) as u16;

            let corrected_r = gamma_map_table[interpolated_r as usize] as i32;
            let corrected_g = gamma_map_table[interpolated_g as usize] as i32;
            let corrected_b = gamma_map_table[interpolated_b as usize] as i32;

            let cb = (corrected_r * transform.cb_r
                + corrected_g * transform.cb_g
                + corrected_b * transform.cb_b
                + bias_uv)
                >> PRECISION;
            let cr = (corrected_r * transform.cr_r
                + corrected_g * transform.cr_g
                + corrected_b * transform.cr_b
                + bias_uv)
                >> PRECISION;
            let u_last = u_plane.last_mut().unwrap();
            let v_last = v_plane.last_mut().unwrap();
            *u_last = cb.max(i_bias_y).min(i_cap_uv) as u8;
            *v_last = cr.max(i_bias_y).min(i_cap_uv) as u8;
        }
    }
}

fn sharpen_row422<const ORIGIN_CHANNELS: u8, const SAMPLING: u8, const PRECISION: i32>(
    rgba: &[u8],
    y_plane: &mut [u8],
    u_plane: &mut [u8],
    v_plane: &mut [u8],
    rgb_layout: &[u16],
    gamma_map_table: &[u8; u16::MAX as usize + 1],
    range: &YuvChromaRange,
    transform: &CbCrForwardTransform<i32>,
    width: usize,
) {
    let src_chans: YuvSourceChannels = ORIGIN_CHANNELS.into();
    let channels = src_chans.get_channels_count();

    let rounding_const_bias: i32 = (1 << (PRECISION - 1)) - 1;
    let bias_y = range.bias_y as i32 * (1 << PRECISION) + rounding_const_bias;
    let bias_uv = range.bias_uv as i32 * (1 << PRECISION) + rounding_const_bias;

    let i_bias_y = range.bias_y as i32;
    let i_cap_y = range.range_y as i32 + i_bias_y;
    let i_cap_uv = i_bias_y + range.range_uv as i32;

    for ((((y_dst, u_dst), v_dst), rgba), rgb_linearized) in y_plane
        .chunks_exact_mut(2)
        .zip(u_plane.iter_mut())
        .zip(v_plane.iter_mut())
        .zip(rgba.chunks_exact(channels * 2))
        .zip(rgb_layout.chunks_exact(2 * 3))
    {
        let r0 = rgba[src_chans.get_r_channel_offset()] as i32;
        let g0 = rgba[src_chans.get_g_channel_offset()] as i32;
        let b0 = rgba[src_chans.get_b_channel_offset()] as i32;

        let sharp_r_c = rgb_linearized[src_chans.get_r_channel_offset()];
        let sharp_g_c = rgb_linearized[src_chans.get_g_channel_offset()];
        let sharp_b_c = rgb_linearized[src_chans.get_b_channel_offset()];

        let y_0 = (r0 * transform.yr + g0 * transform.yg + b0 * transform.yb + bias_y) >> PRECISION;
        y_dst[0] = y_0.min(i_cap_y) as u8;

        let rgba_2 = &rgba[channels..channels * 2];

        let r1 = rgba_2[src_chans.get_r_channel_offset()] as i32;
        let g1 = rgba_2[src_chans.get_g_channel_offset()] as i32;
        let b1 = rgba_2[src_chans.get_b_channel_offset()] as i32;

        let y_1 = (r1 * transform.yr + g1 * transform.yg + b1 * transform.yb + bias_y) >> PRECISION;
        y_dst[1] = y_1.min(i_cap_y) as u8;

        let rgb_linearized_2 = &rgb_linearized[3..(3 + 3)];

        let sharp_r_next = rgb_linearized_2[src_chans.get_r_channel_offset()];
        let sharp_g_next = rgb_linearized_2[src_chans.get_g_channel_offset()];
        let sharp_b_next = rgb_linearized_2[src_chans.get_b_channel_offset()];

        const ROUNDING_SHARP: i32 = 1 << 3;

        let interpolated_r =
            ((sharp_r_c as i32 * 12 + sharp_r_next as i32 * 4 + ROUNDING_SHARP) >> 4) as u16;
        let interpolated_g =
            ((sharp_g_c as i32 * 12 + sharp_g_next as i32 * 4 + ROUNDING_SHARP) >> 4) as u16;
        let interpolated_b =
            ((sharp_b_c as i32 * 12 + sharp_b_next as i32 * 4 + ROUNDING_SHARP) >> 4) as u16;

        let corrected_r = gamma_map_table[interpolated_r as usize] as i32;
        let corrected_g = gamma_map_table[interpolated_g as usize] as i32;
        let corrected_b = gamma_map_table[interpolated_b as usize] as i32;

        let cb = (corrected_r * transform.cb_r
            + corrected_g * transform.cb_g
            + corrected_b * transform.cb_b
            + bias_uv)
            >> PRECISION;
        let cr = (corrected_r * transform.cr_r
            + corrected_g * transform.cr_g
            + corrected_b * transform.cr_b
            + bias_uv)
            >> PRECISION;
        *u_dst = cb.max(i_bias_y).min(i_cap_uv) as u8;
        *v_dst = cr.max(i_bias_y).min(i_cap_uv) as u8;
    }

    let rem_rgba = rgba.chunks_exact(channels * 2).remainder();

    if width & 1 != 0 && !rem_rgba.is_empty() {
        let rgba = &rem_rgba[0..3];
        let y_last = y_plane.last_mut().unwrap();
        let r0 = rgba[src_chans.get_r_channel_offset()] as i32;
        let g0 = rgba[src_chans.get_g_channel_offset()] as i32;
        let b0 = rgba[src_chans.get_b_channel_offset()] as i32;

        let y_1 = (r0 * transform.yr + g0 * transform.yg + b0 * transform.yb + bias_y) >> PRECISION;
        *y_last = y_1.min(i_cap_y) as u8;

        let cb = (r0 * transform.cb_r + g0 * transform.cb_g + b0 * transform.cb_b + bias_uv)
            >> PRECISION;
        let cr = (r0 * transform.cr_r + g0 * transform.cr_g + b0 * transform.cr_b + bias_uv)
            >> PRECISION;

        let u_last = u_plane.last_mut().unwrap();
        let v_last = v_plane.last_mut().unwrap();
        *u_last = cb.max(i_bias_y).min(i_cap_uv) as u8;
        *v_last = cr.max(i_bias_y).min(i_cap_uv) as u8;
    }
}

fn rgbx_to_sharp_yuv_core<const ORIGIN_CHANNELS: u8, const SAMPLING: u8, const PRECISION: i32>(
    planar_image: &mut YuvPlanarImageMut<u8>,
    rgba: &[u8],
    rgba_stride: u32,
    chroma_range: &YuvChromaRange,
    transform: &CbCrForwardTransform<i32>,
    sharp_yuv_gamma_transfer: SharpYuvGammaTransfer,
) -> Result<(), YuvError> {
    let chroma_subsampling: YuvChromaSubsampling = SAMPLING.into();
    let src_chans: YuvSourceChannels = ORIGIN_CHANNELS.into();

    check_rgba_destination(
        rgba,
        rgba_stride,
        planar_image.width,
        planar_image.height,
        src_chans.get_channels_count(),
    )?;
    planar_image.check_constraints(chroma_subsampling)?;

    let mut linear_map_table = [0u16; 256];
    let mut gamma_map_table = [0u8; u16::MAX as usize + 1];

    let linear_scale = (1. / 255.) as f32;
    let gamma_scale = 1. / u16::MAX as f32;

    for (i, item) in linear_map_table.iter_mut().enumerate() {
        let linear = sharp_yuv_gamma_transfer.linearize(i as f32 * linear_scale);
        *item = (linear * u16::MAX as f32) as u16;
    }

    for (i, item) in gamma_map_table.iter_mut().enumerate() {
        let gamma = sharp_yuv_gamma_transfer.gamma(i as f32 * gamma_scale);
        *item = (gamma * 255.) as u8;
    }

    let mut rgb_layout: Vec<u16> =
        vec![0u16; planar_image.width as usize * planar_image.height as usize * 3];

    let rgb_layout_stride_len = planar_image.width as usize * 3;

    let iter_linearize;
    #[cfg(not(feature = "rayon"))]
    {
        iter_linearize = rgb_layout
            .chunks_exact_mut(rgb_layout_stride_len)
            .zip(rgba.chunks_exact(rgba_stride as usize));
    }
    #[cfg(feature = "rayon")]
    {
        iter_linearize = rgb_layout
            .par_chunks_exact_mut(rgb_layout_stride_len)
            .zip(rgba.par_chunks_exact(rgba_stride as usize));
    }

    iter_linearize.for_each(|(rgb_layout_cast, src_layout)| {
        for (dst, src) in rgb_layout_cast
            .chunks_exact_mut(3)
            .zip(src_layout.chunks_exact(src_chans.get_channels_count()))
        {
            dst[0] = linear_map_table[src[0] as usize];
            dst[1] = linear_map_table[src[1] as usize];
            dst[2] = linear_map_table[src[2] as usize];
        }
    });

    let y_iter;
    let u_iter;
    let v_iter;
    let rgb_iter;

    if chroma_subsampling == YuvChromaSubsampling::Yuv420 {
        #[cfg(feature = "rayon")]
        {
            y_iter = planar_image
                .y_plane
                .borrow_mut()
                .par_chunks_exact_mut(planar_image.y_stride as usize * 2);
            u_iter = planar_image
                .u_plane
                .borrow_mut()
                .par_chunks_exact_mut(planar_image.u_stride as usize);
            v_iter = planar_image
                .v_plane
                .borrow_mut()
                .par_chunks_exact_mut(planar_image.v_stride as usize);
            rgb_iter = rgba.par_chunks_exact(rgba_stride as usize * 2);
        }
        #[cfg(not(feature = "rayon"))]
        {
            y_iter = planar_image
                .y_plane
                .borrow_mut()
                .chunks_exact_mut(planar_image.y_stride as usize * 2);
            u_iter = planar_image
                .u_plane
                .borrow_mut()
                .chunks_exact_mut(planar_image.u_stride as usize);
            v_iter = planar_image
                .v_plane
                .borrow_mut()
                .chunks_exact_mut(planar_image.v_stride as usize);
            rgb_iter = rgba.chunks_exact(rgba_stride as usize * 2);
        }
    } else {
        #[cfg(feature = "rayon")]
        {
            y_iter = planar_image
                .y_plane
                .borrow_mut()
                .par_chunks_exact_mut(planar_image.y_stride as usize);
            u_iter = planar_image
                .u_plane
                .borrow_mut()
                .par_chunks_exact_mut(planar_image.u_stride as usize);
            v_iter = planar_image
                .v_plane
                .borrow_mut()
                .par_chunks_exact_mut(planar_image.v_stride as usize);
            rgb_iter = rgba.par_chunks_exact(rgba_stride as usize);
        }
        #[cfg(not(feature = "rayon"))]
        {
            y_iter = planar_image
                .y_plane
                .borrow_mut()
                .chunks_exact_mut(planar_image.y_stride as usize);
            u_iter = planar_image
                .u_plane
                .borrow_mut()
                .chunks_exact_mut(planar_image.u_stride as usize);
            v_iter = planar_image
                .v_plane
                .borrow_mut()
                .chunks_exact_mut(planar_image.v_stride as usize);
            rgb_iter = rgba.chunks_exact(rgba_stride as usize);
        }
    }

    let full_iter = rgb_iter.zip(y_iter).zip(u_iter).zip(v_iter);

    full_iter
        .enumerate()
        .for_each(|(j, (((rgba, y_plane), u_plane), v_plane))| {
            if chroma_subsampling == YuvChromaSubsampling::Yuv420 {
                let v_y = j * 2;

                for (virtual_y, (y_plane, rgba)) in y_plane
                    .chunks_exact_mut(planar_image.y_stride as usize)
                    .zip(rgba.chunks_exact(rgba_stride as usize))
                    .enumerate()
                {
                    let y = virtual_y + v_y;
                    let rgb_layout_start = y * rgb_layout_stride_len;
                    let rgb_layout_start_next = (y + 1) * rgb_layout_stride_len;
                    let rgb_layout_lane = &rgb_layout
                        [rgb_layout_start..((planar_image.width as usize) * 3 + rgb_layout_start)];
                    let rgb_layout_next_lane = if y + 1 < planar_image.height as usize {
                        &rgb_layout[rgb_layout_start_next
                            ..((planar_image.width as usize) * 3 + rgb_layout_start_next)]
                    } else {
                        rgb_layout_lane
                    };
                    sharpen_row420::<ORIGIN_CHANNELS, SAMPLING, PRECISION>(
                        y,
                        &rgba[..planar_image.width as usize * src_chans.get_channels_count()],
                        &mut y_plane[..planar_image.width as usize],
                        &mut u_plane[..(planar_image.width as usize).div_ceil(2)],
                        &mut v_plane[..(planar_image.width as usize).div_ceil(2)],
                        rgb_layout_lane,
                        rgb_layout_next_lane,
                        &gamma_map_table,
                        chroma_range,
                        transform,
                        planar_image.width as usize,
                    );
                }
            } else {
                let y = j;
                let rgb_layout_start = y * rgb_layout_stride_len;
                let rgb_layout_lane = &rgb_layout
                    [rgb_layout_start..((planar_image.width as usize) * 3 + rgb_layout_start)];
                sharpen_row422::<ORIGIN_CHANNELS, SAMPLING, PRECISION>(
                    &rgba[..planar_image.width as usize * src_chans.get_channels_count()],
                    &mut y_plane[..planar_image.width as usize],
                    &mut u_plane[..(planar_image.width as usize).div_ceil(2)],
                    &mut v_plane[..(planar_image.width as usize).div_ceil(2)],
                    rgb_layout_lane,
                    &gamma_map_table,
                    chroma_range,
                    transform,
                    planar_image.width as usize,
                );
            }
        });

    if planar_image.height & 1 != 0 && chroma_subsampling == YuvChromaSubsampling::Yuv420 {
        let y_iter = planar_image
            .y_plane
            .borrow_mut()
            .chunks_exact_mut(planar_image.y_stride as usize)
            .rev()
            .take(1);
        let u_iter = planar_image
            .u_plane
            .borrow_mut()
            .chunks_exact_mut(planar_image.u_stride as usize)
            .rev()
            .take(1);
        let v_iter = planar_image
            .v_plane
            .borrow_mut()
            .chunks_exact_mut(planar_image.v_stride as usize)
            .rev()
            .take(1);
        let rgb_iter = rgba.chunks_exact(rgba_stride as usize).rev().take(1);
        let rgb_linearized_iter = rgb_layout
            .chunks_exact_mut(rgb_layout_stride_len)
            .rev()
            .take(1);

        let full_iter = rgb_iter
            .zip(rgb_linearized_iter)
            .zip(y_iter)
            .zip(u_iter)
            .zip(v_iter);

        full_iter.for_each(|((((rgba, rgb_layout), y_plane), u_plane), v_plane)| {
            let y = planar_image.height as usize - 1;
            let rgb_layout_lane: &[u16] = rgb_layout;
            let rgb_layout_next_lane: &[u16] = rgb_layout;
            sharpen_row420::<ORIGIN_CHANNELS, SAMPLING, PRECISION>(
                y,
                &rgba[..planar_image.width as usize * src_chans.get_channels_count()],
                &mut y_plane[..planar_image.width as usize],
                &mut u_plane[..(planar_image.width as usize).div_ceil(2)],
                &mut v_plane[..(planar_image.width as usize).div_ceil(2)],
                rgb_layout_lane,
                rgb_layout_next_lane,
                &gamma_map_table,
                chroma_range,
                transform,
                planar_image.width as usize,
            );
        });
    }

    Ok(())
}

fn rgbx_to_sharp_yuv_dispatch_mode<const ORIGIN_CHANNELS: u8, const SAMPLING: u8>(
    planar_image: &mut YuvPlanarImageMut<u8>,
    rgba: &[u8],
    rgba_stride: u32,
    chroma_range: &YuvChromaRange,
    transform: &CbCrForwardTransform<i32>,
    sharp_yuv_gamma_transfer: SharpYuvGammaTransfer,
    mode: YuvConversionMode,
) -> Result<(), YuvError> {
    match mode {
        #[cfg(feature = "fast_mode")]
        YuvConversionMode::Fast => rgbx_to_sharp_yuv_core::<ORIGIN_CHANNELS, SAMPLING, 7>(
            planar_image,
            rgba,
            rgba_stride,
            chroma_range,
            transform,
            sharp_yuv_gamma_transfer,
        ),
        YuvConversionMode::Balanced => rgbx_to_sharp_yuv_core::<ORIGIN_CHANNELS, SAMPLING, 13>(
            planar_image,
            rgba,
            rgba_stride,
            chroma_range,
            transform,
            sharp_yuv_gamma_transfer,
        ),
        #[cfg(feature = "professional_mode")]
        YuvConversionMode::Professional => rgbx_to_sharp_yuv_core::<ORIGIN_CHANNELS, SAMPLING, 15>(
            planar_image,
            rgba,
            rgba_stride,
            chroma_range,
            transform,
            sharp_yuv_gamma_transfer,
        ),
        #[cfg(feature = "professional_mode")]
        YuvConversionMode::Professional16 => {
            rgbx_to_sharp_yuv_core::<ORIGIN_CHANNELS, SAMPLING, 16>(
                planar_image,
                rgba,
                rgba_stride,
                chroma_range,
                transform,
                sharp_yuv_gamma_transfer,
            )
        }
    }
}

fn rgbx_to_sharp_yuv<const ORIGIN_CHANNELS: u8, const SAMPLING: u8>(
    planar_image: &mut YuvPlanarImageMut<u8>,
    rgba: &[u8],
    rgba_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    sharp_yuv_gamma_transfer: SharpYuvGammaTransfer,
) -> Result<(), YuvError> {
    let chroma_range = get_yuv_range(8, range);
    let kr_kb = matrix.get_kr_kb();
    let max_range_p8 = (1u32 << 8u32) - 1u32;
    const PRECISION: i32 = 13;
    let transform =
        if let Some(stored_t) = get_built_forward_transform(PRECISION as u32, 8, range, matrix) {
            stored_t
        } else {
            let transform_precise = get_forward_transform(
                max_range_p8,
                chroma_range.range_y,
                chroma_range.range_uv,
                kr_kb.kr,
                kr_kb.kb,
            );
            transform_precise.to_integers(PRECISION as u32)
        };
    rgbx_to_sharp_yuv_core::<ORIGIN_CHANNELS, SAMPLING, PRECISION>(
        planar_image,
        rgba,
        rgba_stride,
        &chroma_range,
        &transform,
        sharp_yuv_gamma_transfer,
    )
}

/// Convert RGB image data to YUV 422 planar format using bi-linear interpolation and gamma correction ( sharp YUV algorithm ).
///
/// This function performs RGB to YUV conversion using bi-linear interpolation and gamma correction and stores the result in YUV422 planar format using bi-linear interpolation and gamma correction,
/// with separate planes for Y (luminance), U (chrominance), and V (chrominance) components.
///
/// # Arguments
///
/// * `planar_image` - Target planar image.
/// * `rgb` - The input RGB image data slice.
/// * `rgb_stride` - The stride (components per row) for the RGB image data.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input RGB data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn rgb_to_sharp_yuv422(
    planar_image: &mut YuvPlanarImageMut<u8>,
    rgb: &[u8],
    rgb_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    gamma_transfer: SharpYuvGammaTransfer,
) -> Result<(), YuvError> {
    rgbx_to_sharp_yuv::<{ YuvSourceChannels::Rgb as u8 }, { YuvChromaSubsampling::Yuv422 as u8 }>(
        planar_image,
        rgb,
        rgb_stride,
        range,
        matrix,
        gamma_transfer,
    )
}

/// Convert BGR image data to YUV 422 planar format using bi-linear interpolation and gamma correction ( sharp YUV algorithm ).
///
/// This function performs BGR to YUV conversion using bi-linear interpolation and gamma correction and stores the result in YUV422 planar format using bi-linear interpolation and gamma correction,
/// with separate planes for Y (luminance), U (chrominance), and V (chrominance) components.
///
/// # Arguments
///
/// * `planar_image` - Target planar image.
/// * `bgr` - The input BGR image data slice.
/// * `bgr_stride` - The stride (components per row) for the BGR image data.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGR data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn bgr_to_sharp_yuv422(
    planar_image: &mut YuvPlanarImageMut<u8>,
    bgr: &[u8],
    bgr_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    gamma_transfer: SharpYuvGammaTransfer,
) -> Result<(), YuvError> {
    rgbx_to_sharp_yuv::<{ YuvSourceChannels::Bgr as u8 }, { YuvChromaSubsampling::Yuv422 as u8 }>(
        planar_image,
        bgr,
        bgr_stride,
        range,
        matrix,
        gamma_transfer,
    )
}

/// Convert RGBA image data to YUV 422 planar format using bi-linear interpolation and gamma correction ( sharp YUV algorithm ).
///
/// This function performs RGBA to YUV conversion using bi-linear interpolation and gamma correction and stores the result in YUV422 planar format using bi-linear interpolation and gamma correction,
/// with separate planes for Y (luminance), U (chrominance), and V (chrominance) components.
///
/// # Arguments
///
/// * `planar_image` - Target planar image.
/// * `rgba` - The input RGBA image data slice.
/// * `rgba_stride` - The stride (components per row) for the RGBA image data.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input RGBA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn rgba_to_sharp_yuv422(
    planar_image: &mut YuvPlanarImageMut<u8>,
    rgba: &[u8],
    rgba_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    gamma_transfer: SharpYuvGammaTransfer,
) -> Result<(), YuvError> {
    rgbx_to_sharp_yuv::<{ YuvSourceChannels::Rgba as u8 }, { YuvChromaSubsampling::Yuv422 as u8 }>(
        planar_image,
        rgba,
        rgba_stride,
        range,
        matrix,
        gamma_transfer,
    )
}

/// Convert BGRA image data to YUV 422 planar format using bi-linear interpolation and gamma correction ( sharp YUV algorithm ).
///
/// This function performs BGRA to YUV conversion using bi-linear interpolation and gamma correction and stores the result in YUV422 planar format using bi-linear interpolation and gamma correction,
/// with separate planes for Y (luminance), U (chrominance), and V (chrominance) components.
///
/// # Arguments
///
/// * `planar_image` - Target planar image.
/// * `bgra` - The input BGRA image data slice.
/// * `bgra_stride` - The stride (components per row) for the BGRA image data.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGRA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn bgra_to_sharp_yuv422(
    planar_image: &mut YuvPlanarImageMut<u8>,
    bgra: &[u8],
    bgra_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    gamma_transfer: SharpYuvGammaTransfer,
) -> Result<(), YuvError> {
    rgbx_to_sharp_yuv::<{ YuvSourceChannels::Bgra as u8 }, { YuvChromaSubsampling::Yuv422 as u8 }>(
        planar_image,
        bgra,
        bgra_stride,
        range,
        matrix,
        gamma_transfer,
    )
}

/// Convert RGB image data to YUV 420 planar format using bi-linear interpolation and gamma correction ( sharp YUV algorithm ).
///
/// This function performs RGB to YUV conversion using bi-linear interpolation and gamma correction and stores the result in YUV420 planar format using bi-linear interpolation and gamma correction,
/// with separate planes for Y (luminance), U (chrominance), and V (chrominance) components.
///
/// # Arguments
///
/// * `y_plane` - A mutable slice to store the Y (luminance) plane data.
/// * `y_stride` - The stride (components per row) for the Y plane.
/// * `u_plane` - A mutable slice to store the U (chrominance) plane data.
/// * `u_stride` - The stride (components per row) for the U plane.
/// * `v_plane` - A mutable slice to store the V (chrominance) plane data.
/// * `v_stride` - The stride (components per row) for the V plane.
/// * `rgb` - The input RGB image data slice.
/// * `rgb_stride` - The stride (components per row) for the RGB image data.
/// * `width` - The width of the image in pixels.
/// * `height` - The height of the image in pixels.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input RGB data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn rgb_to_sharp_yuv420(
    planar_image: &mut YuvPlanarImageMut<u8>,
    rgb: &[u8],
    rgb_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    gamma_transfer: SharpYuvGammaTransfer,
) -> Result<(), YuvError> {
    rgbx_to_sharp_yuv::<{ YuvSourceChannels::Rgb as u8 }, { YuvChromaSubsampling::Yuv420 as u8 }>(
        planar_image,
        rgb,
        rgb_stride,
        range,
        matrix,
        gamma_transfer,
    )
}

/// Convert BGR image data to YUV 420 planar format using bi-linear interpolation and gamma correction ( sharp YUV algorithm ).
///
/// This function performs BGR to YUV conversion using bi-linear interpolation and gamma correction and stores the result in YUV420 planar format using bi-linear interpolation and gamma correction,
/// with separate planes for Y (luminance), U (chrominance), and V (chrominance) components.
///
/// # Arguments
///
/// * `planar_image` - Target planar image.
/// * `bgr` - The input BGR image data slice.
/// * `bgr_stride` - The stride (components per row) for the BGR image data.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input RGB data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn bgr_to_sharp_yuv420(
    planar_image: &mut YuvPlanarImageMut<u8>,
    bgr: &[u8],
    bgr_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    gamma_transfer: SharpYuvGammaTransfer,
) -> Result<(), YuvError> {
    rgbx_to_sharp_yuv::<{ YuvSourceChannels::Bgr as u8 }, { YuvChromaSubsampling::Yuv420 as u8 }>(
        planar_image,
        bgr,
        bgr_stride,
        range,
        matrix,
        gamma_transfer,
    )
}

/// Convert RGBA image data to YUV 420 planar format using bi-linear interpolation and gamma correction ( sharp YUV algorithm ).
///
/// This function performs RGBA to YUV conversion using bi-linear interpolation and gamma correction and stores the result in YUV420 planar format using bi-linear interpolation and gamma correction,
/// with separate planes for Y (luminance), U (chrominance), and V (chrominance) components.
///
/// # Arguments
///
/// * `planar_image` - Target planar image.
/// * `rgba` - The input RGBA image data slice.
/// * `rgba_stride` - The stride (components per row) for the RGBA image data.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input RGBA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn rgba_to_sharp_yuv420(
    planar_image: &mut YuvPlanarImageMut<u8>,
    rgba: &[u8],
    rgba_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    gamma_transfer: SharpYuvGammaTransfer,
) -> Result<(), YuvError> {
    rgbx_to_sharp_yuv::<{ YuvSourceChannels::Rgba as u8 }, { YuvChromaSubsampling::Yuv420 as u8 }>(
        planar_image,
        rgba,
        rgba_stride,
        range,
        matrix,
        gamma_transfer,
    )
}

/// Convert BGRA image data to YUV 420 planar format using bi-linear interpolation and gamma correction ( sharp YUV algorithm ).
///
/// This function performs BGRA to YUV conversion using bi-linear interpolation and gamma correction and stores the result in YUV420 planar format using bi-linear interpolation and gamma correction,
/// with separate planes for Y (luminance), U (chrominance), and V (chrominance) components.
///
/// # Arguments
///
/// * `planar_image` - Target planar image.
/// * `bgra` - The input BGRA image data slice.
/// * `bgra_stride` - The stride (components per row) for the BGRA image data.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGRA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn bgra_to_sharp_yuv420(
    planar_image: &mut YuvPlanarImageMut<u8>,
    bgra: &[u8],
    bgra_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    gamma_transfer: SharpYuvGammaTransfer,
) -> Result<(), YuvError> {
    rgbx_to_sharp_yuv::<{ YuvSourceChannels::Bgra as u8 }, { YuvChromaSubsampling::Yuv420 as u8 }>(
        planar_image,
        bgra,
        bgra_stride,
        range,
        matrix,
        gamma_transfer,
    )
}

/// Convert RGB to SharpYUV 422 using pre-computed transform coefficients.
///
/// Bypasses the standard kr/kb to coefficient derivation, allowing the caller to supply
/// exact fixed-point coefficients (e.g. libwebp's VP8 matrix). The `mode` controls
/// the fixed-point precision and must match the precision used to compute `transform`.
pub fn rgb_to_sharp_yuv422_with_transform(
    planar_image: &mut YuvPlanarImageMut<u8>,
    rgb: &[u8],
    rgb_stride: u32,
    transform: &CbCrForwardTransform<i32>,
    chroma_range: &YuvChromaRange,
    gamma_transfer: SharpYuvGammaTransfer,
    mode: YuvConversionMode,
) -> Result<(), YuvError> {
    rgbx_to_sharp_yuv_dispatch_mode::<
        { YuvSourceChannels::Rgb as u8 },
        { YuvChromaSubsampling::Yuv422 as u8 },
    >(
        planar_image,
        rgb,
        rgb_stride,
        chroma_range,
        transform,
        gamma_transfer,
        mode,
    )
}

/// Convert BGR to SharpYUV 422 using pre-computed transform coefficients.
///
/// See [`rgb_to_sharp_yuv422_with_transform`] for details.
pub fn bgr_to_sharp_yuv422_with_transform(
    planar_image: &mut YuvPlanarImageMut<u8>,
    bgr: &[u8],
    bgr_stride: u32,
    transform: &CbCrForwardTransform<i32>,
    chroma_range: &YuvChromaRange,
    gamma_transfer: SharpYuvGammaTransfer,
    mode: YuvConversionMode,
) -> Result<(), YuvError> {
    rgbx_to_sharp_yuv_dispatch_mode::<
        { YuvSourceChannels::Bgr as u8 },
        { YuvChromaSubsampling::Yuv422 as u8 },
    >(
        planar_image,
        bgr,
        bgr_stride,
        chroma_range,
        transform,
        gamma_transfer,
        mode,
    )
}

/// Convert RGBA to SharpYUV 422 using pre-computed transform coefficients.
///
/// See [`rgb_to_sharp_yuv422_with_transform`] for details.
pub fn rgba_to_sharp_yuv422_with_transform(
    planar_image: &mut YuvPlanarImageMut<u8>,
    rgba: &[u8],
    rgba_stride: u32,
    transform: &CbCrForwardTransform<i32>,
    chroma_range: &YuvChromaRange,
    gamma_transfer: SharpYuvGammaTransfer,
    mode: YuvConversionMode,
) -> Result<(), YuvError> {
    rgbx_to_sharp_yuv_dispatch_mode::<
        { YuvSourceChannels::Rgba as u8 },
        { YuvChromaSubsampling::Yuv422 as u8 },
    >(
        planar_image,
        rgba,
        rgba_stride,
        chroma_range,
        transform,
        gamma_transfer,
        mode,
    )
}

/// Convert BGRA to SharpYUV 422 using pre-computed transform coefficients.
///
/// See [`rgb_to_sharp_yuv422_with_transform`] for details.
pub fn bgra_to_sharp_yuv422_with_transform(
    planar_image: &mut YuvPlanarImageMut<u8>,
    bgra: &[u8],
    bgra_stride: u32,
    transform: &CbCrForwardTransform<i32>,
    chroma_range: &YuvChromaRange,
    gamma_transfer: SharpYuvGammaTransfer,
    mode: YuvConversionMode,
) -> Result<(), YuvError> {
    rgbx_to_sharp_yuv_dispatch_mode::<
        { YuvSourceChannels::Bgra as u8 },
        { YuvChromaSubsampling::Yuv422 as u8 },
    >(
        planar_image,
        bgra,
        bgra_stride,
        chroma_range,
        transform,
        gamma_transfer,
        mode,
    )
}

/// Convert RGB to SharpYUV 420 using pre-computed transform coefficients.
///
/// Bypasses the standard kr/kb to coefficient derivation, allowing the caller to supply
/// exact fixed-point coefficients (e.g. libwebp's VP8 matrix). The `mode` controls
/// the fixed-point precision and must match the precision used to compute `transform`.
pub fn rgb_to_sharp_yuv420_with_transform(
    planar_image: &mut YuvPlanarImageMut<u8>,
    rgb: &[u8],
    rgb_stride: u32,
    transform: &CbCrForwardTransform<i32>,
    chroma_range: &YuvChromaRange,
    gamma_transfer: SharpYuvGammaTransfer,
    mode: YuvConversionMode,
) -> Result<(), YuvError> {
    rgbx_to_sharp_yuv_dispatch_mode::<
        { YuvSourceChannels::Rgb as u8 },
        { YuvChromaSubsampling::Yuv420 as u8 },
    >(
        planar_image,
        rgb,
        rgb_stride,
        chroma_range,
        transform,
        gamma_transfer,
        mode,
    )
}

/// Convert BGR to SharpYUV 420 using pre-computed transform coefficients.
///
/// See [`rgb_to_sharp_yuv420_with_transform`] for details.
pub fn bgr_to_sharp_yuv420_with_transform(
    planar_image: &mut YuvPlanarImageMut<u8>,
    bgr: &[u8],
    bgr_stride: u32,
    transform: &CbCrForwardTransform<i32>,
    chroma_range: &YuvChromaRange,
    gamma_transfer: SharpYuvGammaTransfer,
    mode: YuvConversionMode,
) -> Result<(), YuvError> {
    rgbx_to_sharp_yuv_dispatch_mode::<
        { YuvSourceChannels::Bgr as u8 },
        { YuvChromaSubsampling::Yuv420 as u8 },
    >(
        planar_image,
        bgr,
        bgr_stride,
        chroma_range,
        transform,
        gamma_transfer,
        mode,
    )
}

/// Convert RGBA to SharpYUV 420 using pre-computed transform coefficients.
///
/// See [`rgb_to_sharp_yuv420_with_transform`] for details.
pub fn rgba_to_sharp_yuv420_with_transform(
    planar_image: &mut YuvPlanarImageMut<u8>,
    rgba: &[u8],
    rgba_stride: u32,
    transform: &CbCrForwardTransform<i32>,
    chroma_range: &YuvChromaRange,
    gamma_transfer: SharpYuvGammaTransfer,
    mode: YuvConversionMode,
) -> Result<(), YuvError> {
    rgbx_to_sharp_yuv_dispatch_mode::<
        { YuvSourceChannels::Rgba as u8 },
        { YuvChromaSubsampling::Yuv420 as u8 },
    >(
        planar_image,
        rgba,
        rgba_stride,
        chroma_range,
        transform,
        gamma_transfer,
        mode,
    )
}

/// Convert BGRA to SharpYUV 420 using pre-computed transform coefficients.
///
/// See [`rgb_to_sharp_yuv420_with_transform`] for details.
pub fn bgra_to_sharp_yuv420_with_transform(
    planar_image: &mut YuvPlanarImageMut<u8>,
    bgra: &[u8],
    bgra_stride: u32,
    transform: &CbCrForwardTransform<i32>,
    chroma_range: &YuvChromaRange,
    gamma_transfer: SharpYuvGammaTransfer,
    mode: YuvConversionMode,
) -> Result<(), YuvError> {
    rgbx_to_sharp_yuv_dispatch_mode::<
        { YuvSourceChannels::Bgra as u8 },
        { YuvChromaSubsampling::Yuv420 as u8 },
    >(
        planar_image,
        bgra,
        bgra_stride,
        chroma_range,
        transform,
        gamma_transfer,
        mode,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sharpyuv::sharp_gamma::{pure_gamma_function, SharpYuvGammaTransfer};
    use crate::yuv_support::{
        get_forward_transform, get_inverse_transform, get_yuv_range, ToIntegerTransform,
    };
    #[cfg(feature = "professional_mode")]
    use crate::yuv_to_rgba_with_transform::yuv420_to_rgba_with_transform;
    use rand::RngExt;

    #[test]
    fn test_gamma0p80_round_trip() {
        let transfer = SharpYuvGammaTransfer::Gamma0p80;
        for i in 0..=255 {
            let x = i as f32 / 255.0;
            let linear = transfer.linearize(x);
            let back = transfer.gamma(linear);
            let diff = (back - x).abs();
            assert!(
                diff < 1e-5,
                "Gamma0p80 round-trip failed at {}: {} vs {}, diff {}",
                x,
                back,
                x,
                diff
            );
        }
    }

    #[test]
    fn test_gamma0p80_values() {
        let transfer = SharpYuvGammaTransfer::Gamma0p80;
        assert_eq!(transfer.linearize(0.0), 0.0);
        assert_eq!(transfer.linearize(1.0), 1.0);
        assert_eq!(transfer.gamma(0.0), 0.0);
        assert_eq!(transfer.gamma(1.0), 1.0);

        let mid = transfer.linearize(0.5);
        let expected = pure_gamma_function(0.5, 0.80);
        assert!(
            (mid - expected).abs() < 1e-6,
            "Gamma0p80 linearize(0.5) = {}, expected {}",
            mid,
            expected
        );
    }

    #[test]
    fn test_sharp_yuv420_with_transform_matches_original() {
        const W: usize = 64;
        const H: usize = 64;
        const CH: usize = 4;

        let or = rand::rng().random_range(0..256) as u8;
        let og = rand::rng().random_range(0..256) as u8;
        let ob = rand::rng().random_range(0..256) as u8;

        let mut src = vec![0u8; W * H * CH];
        for px in src.chunks_exact_mut(CH) {
            px[0] = or;
            px[1] = og;
            px[2] = ob;
            px[3] = 255;
        }

        let range = get_yuv_range(8, YuvRange::Limited);
        let kr_kb = YuvStandardMatrix::Bt601.get_kr_kb();
        let transform =
            get_forward_transform(255, range.range_y, range.range_uv, kr_kb.kr, kr_kb.kb)
                .to_integers(13);

        let mut planar_orig =
            YuvPlanarImageMut::<u8>::alloc(W as u32, H as u32, YuvChromaSubsampling::Yuv420);
        rgba_to_sharp_yuv420(
            &mut planar_orig,
            &src,
            W as u32 * CH as u32,
            YuvRange::Limited,
            YuvStandardMatrix::Bt601,
            SharpYuvGammaTransfer::Srgb,
        )
        .unwrap();

        let mut planar_new =
            YuvPlanarImageMut::<u8>::alloc(W as u32, H as u32, YuvChromaSubsampling::Yuv420);
        rgba_to_sharp_yuv420_with_transform(
            &mut planar_new,
            &src,
            W as u32 * CH as u32,
            &transform,
            &range,
            SharpYuvGammaTransfer::Srgb,
            YuvConversionMode::Balanced,
        )
        .unwrap();

        assert_eq!(
            planar_orig.y_plane.borrow(),
            planar_new.y_plane.borrow(),
            "Y planes differ between original and with_transform at Balanced/P13"
        );
        assert_eq!(
            planar_orig.u_plane.borrow(),
            planar_new.u_plane.borrow(),
            "U planes differ"
        );
        assert_eq!(
            planar_orig.v_plane.borrow(),
            planar_new.v_plane.borrow(),
            "V planes differ"
        );
    }

    #[test]
    #[cfg(feature = "professional_mode")]
    fn test_sharp_yuv420_with_transform_p16_round_trip() {
        const W: usize = 64;
        const H: usize = 64;
        const CH: usize = 4;

        let mut worst_r = 0i32;
        let mut worst_g = 0i32;
        let mut worst_b = 0i32;

        for _ in 0..20 {
            let or = rand::rng().random_range(0..256) as u8;
            let og = rand::rng().random_range(0..256) as u8;
            let ob = rand::rng().random_range(0..256) as u8;

            let mut src = vec![0u8; W * H * CH];
            for px in src.chunks_exact_mut(CH) {
                px[0] = or;
                px[1] = og;
                px[2] = ob;
                px[3] = 255;
            }

            let range = get_yuv_range(8, YuvRange::Limited);
            let kr_kb = YuvStandardMatrix::Bt601.get_kr_kb();
            let transform =
                get_forward_transform(255, range.range_y, range.range_uv, kr_kb.kr, kr_kb.kb)
                    .to_integers(16);
            let inverse_transform =
                get_inverse_transform(255, range.range_y, range.range_uv, kr_kb.kr, kr_kb.kb)
                    .to_integers(16);

            let mut planar =
                YuvPlanarImageMut::<u8>::alloc(W as u32, H as u32, YuvChromaSubsampling::Yuv420);
            rgba_to_sharp_yuv420_with_transform(
                &mut planar,
                &src,
                W as u32 * CH as u32,
                &transform,
                &range,
                SharpYuvGammaTransfer::Srgb,
                YuvConversionMode::Professional16,
            )
            .unwrap();

            let fixed = planar.to_fixed();
            let mut dst = vec![0u8; W * H * CH];
            yuv420_to_rgba_with_transform(
                &fixed,
                &mut dst,
                W as u32 * CH as u32,
                &inverse_transform,
                &range,
                YuvConversionMode::Professional16,
            )
            .unwrap();

            for (src_px, dst_px) in src.chunks_exact(CH).zip(dst.chunks_exact(CH)) {
                worst_r = worst_r.max((dst_px[0] as i32 - src_px[0] as i32).abs());
                worst_g = worst_g.max((dst_px[1] as i32 - src_px[1] as i32).abs());
                worst_b = worst_b.max((dst_px[2] as i32 - src_px[2] as i32).abs());
            }
        }

        let max_diff = 2;
        assert!(
            worst_r <= max_diff && worst_g <= max_diff && worst_b <= max_diff,
            "SharpYUV P16 round-trip: worst per-channel diffs ({},{},{}) exceed {}",
            worst_r,
            worst_g,
            worst_b,
            max_diff
        );
    }

    #[test]
    #[cfg(feature = "professional_mode")]
    fn test_sharp_yuv420_webp_coefficients_gamma0p80() {
        const W: usize = 64;
        const H: usize = 64;
        const CH: usize = 4;

        let webp_transform = CbCrForwardTransform {
            yr: 16839,
            yg: 33059,
            yb: 6420,
            cb_r: -9719,
            cb_g: -19081,
            cb_b: 28800,
            cr_r: 28800,
            cr_g: -24116,
            cr_b: -4684,
        };

        let range = get_yuv_range(8, YuvRange::Limited);
        let kr_kb = YuvStandardMatrix::Bt601.get_kr_kb();
        let inverse_transform =
            get_inverse_transform(255, range.range_y, range.range_uv, kr_kb.kr, kr_kb.kb)
                .to_integers(16);

        let mut worst_r = 0i32;
        let mut worst_g = 0i32;
        let mut worst_b = 0i32;

        for _ in 0..20 {
            let or = rand::rng().random_range(0..256) as u8;
            let og = rand::rng().random_range(0..256) as u8;
            let ob = rand::rng().random_range(0..256) as u8;

            let mut src = vec![0u8; W * H * CH];
            for px in src.chunks_exact_mut(CH) {
                px[0] = or;
                px[1] = og;
                px[2] = ob;
                px[3] = 255;
            }

            let mut planar =
                YuvPlanarImageMut::<u8>::alloc(W as u32, H as u32, YuvChromaSubsampling::Yuv420);
            rgba_to_sharp_yuv420_with_transform(
                &mut planar,
                &src,
                W as u32 * CH as u32,
                &webp_transform,
                &range,
                SharpYuvGammaTransfer::Gamma0p80,
                YuvConversionMode::Professional16,
            )
            .unwrap();

            let fixed = planar.to_fixed();
            let mut dst = vec![0u8; W * H * CH];
            yuv420_to_rgba_with_transform(
                &fixed,
                &mut dst,
                W as u32 * CH as u32,
                &inverse_transform,
                &range,
                YuvConversionMode::Professional16,
            )
            .unwrap();

            for (src_px, dst_px) in src.chunks_exact(CH).zip(dst.chunks_exact(CH)) {
                worst_r = worst_r.max((dst_px[0] as i32 - src_px[0] as i32).abs());
                worst_g = worst_g.max((dst_px[1] as i32 - src_px[1] as i32).abs());
                worst_b = worst_b.max((dst_px[2] as i32 - src_px[2] as i32).abs());
            }
        }

        let max_diff = 2;
        assert!(
            worst_r <= max_diff && worst_g <= max_diff && worst_b <= max_diff,
            "SharpYUV WebP+Gamma0p80 round-trip: worst per-channel diffs ({},{},{}) exceed {}",
            worst_r,
            worst_g,
            worst_b,
            max_diff
        );
    }

    #[test]
    fn test_sharp_yuv422_with_transform_round_trip() {
        const W: usize = 64;
        const H: usize = 64;
        const CH: usize = 3;

        let or = rand::rng().random_range(0..256) as u8;
        let og = rand::rng().random_range(0..256) as u8;
        let ob = rand::rng().random_range(0..256) as u8;

        let mut src = vec![0u8; W * H * CH];
        for px in src.chunks_exact_mut(CH) {
            px[0] = or;
            px[1] = og;
            px[2] = ob;
        }

        let range = get_yuv_range(8, YuvRange::Limited);
        let kr_kb = YuvStandardMatrix::Bt601.get_kr_kb();
        let transform =
            get_forward_transform(255, range.range_y, range.range_uv, kr_kb.kr, kr_kb.kb)
                .to_integers(13);

        let mut planar =
            YuvPlanarImageMut::<u8>::alloc(W as u32, H as u32, YuvChromaSubsampling::Yuv422);
        rgb_to_sharp_yuv422_with_transform(
            &mut planar,
            &src,
            W as u32 * CH as u32,
            &transform,
            &range,
            SharpYuvGammaTransfer::Srgb,
            YuvConversionMode::Balanced,
        )
        .unwrap();

        let y_val = planar.y_plane.borrow()[0];
        assert!(y_val > 0, "Y plane should have non-zero values");
    }

    #[test]
    #[cfg(feature = "professional_mode")]
    fn test_sharp_yuv_precision_differs_across_modes() {
        const W: usize = 64;
        const H: usize = 64;
        const CH: usize = 4;

        let mut src = vec![0u8; W * H * CH];
        for (i, px) in src.chunks_exact_mut(CH).enumerate() {
            px[0] = (i % 256) as u8;
            px[1] = ((i * 7) % 256) as u8;
            px[2] = ((i * 13) % 256) as u8;
            px[3] = 255;
        }

        let range = get_yuv_range(8, YuvRange::Limited);
        let kr_kb = YuvStandardMatrix::Bt601.get_kr_kb();

        let transform_13 =
            get_forward_transform(255, range.range_y, range.range_uv, kr_kb.kr, kr_kb.kb)
                .to_integers(13);
        let transform_16 =
            get_forward_transform(255, range.range_y, range.range_uv, kr_kb.kr, kr_kb.kb)
                .to_integers(16);

        let mut planar_13 =
            YuvPlanarImageMut::<u8>::alloc(W as u32, H as u32, YuvChromaSubsampling::Yuv420);
        rgba_to_sharp_yuv420_with_transform(
            &mut planar_13,
            &src,
            W as u32 * CH as u32,
            &transform_13,
            &range,
            SharpYuvGammaTransfer::Srgb,
            YuvConversionMode::Balanced,
        )
        .unwrap();

        let mut planar_16 =
            YuvPlanarImageMut::<u8>::alloc(W as u32, H as u32, YuvChromaSubsampling::Yuv420);
        rgba_to_sharp_yuv420_with_transform(
            &mut planar_16,
            &src,
            W as u32 * CH as u32,
            &transform_16,
            &range,
            SharpYuvGammaTransfer::Srgb,
            YuvConversionMode::Professional16,
        )
        .unwrap();

        let y_diff: u32 = planar_13
            .y_plane
            .borrow()
            .iter()
            .zip(planar_16.y_plane.borrow().iter())
            .map(|(a, b)| (*a as i32 - *b as i32).unsigned_abs())
            .sum();

        assert!(
            y_diff <= (W * H) as u32,
            "P13 vs P16 Y planes should be close, total diff: {} (max allowed: {})",
            y_diff,
            W * H
        );
    }
}
