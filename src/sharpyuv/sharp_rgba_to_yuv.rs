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
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use crate::sharpyuv::neon::neon_rgba_to_sharp_yuv;
use crate::sharpyuv::SharpYuvGammaTransfer;
use crate::yuv_support::*;
#[cfg(feature = "rayon")]
use rayon::iter::{IndexedParallelIterator, ParallelIterator};
#[cfg(feature = "rayon")]
use rayon::prelude::{ParallelSlice, ParallelSliceMut};

fn rgbx_to_sharp_yuv<const ORIGIN_CHANNELS: u8, const SAMPLING: u8>(
    y_plane: &mut [u8],
    y_stride: u32,
    u_plane: &mut [u8],
    u_stride: u32,
    v_plane: &mut [u8],
    v_stride: u32,
    rgba: &[u8],
    rgba_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    sharp_yuv_gamma_transfer: SharpYuvGammaTransfer,
) {
    let chroma_subsampling: YuvChromaSample = SAMPLING.into();
    let src_chans: YuvSourceChannels = ORIGIN_CHANNELS.into();

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

    // Always using 3 Channels ( RGB etc. ) layout since we do not need a alpha channel
    let mut rgb_layout: Vec<u16> = vec![0u16; width as usize * height as usize * 3];

    let rgb_layout_stride_len = width as usize * 3;

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

    iter_linearize.for_each(|(rgb_layout_cast, src_layout)| unsafe {
        for (dst, src) in rgb_layout_cast
            .chunks_exact_mut(3)
            .zip(src_layout.chunks_exact(src_chans.get_channels_count()))
        {
            dst[0] = *linear_map_table.get_unchecked(src[0] as usize);
            dst[1] = *linear_map_table.get_unchecked(src[1] as usize);
            dst[2] = *linear_map_table.get_unchecked(src[2] as usize);
        }
    });

    let channels = src_chans.get_channels_count();
    let range = get_yuv_range(8, range);
    let kr_kb = get_kr_kb(matrix);
    let max_range_p8 = (1u32 << 8u32) - 1u32;
    let transform_precise = get_forward_transform(
        max_range_p8,
        range.range_y,
        range.range_uv,
        kr_kb.kr,
        kr_kb.kb,
    );
    const PRECISION: i32 = 14;
    let transform = transform_precise.to_integers(PRECISION as u32);
    const ROUNDING_CONST_BIAS: i32 = 1 << (PRECISION - 1);
    let bias_y = range.bias_y as i32 * (1 << PRECISION) + ROUNDING_CONST_BIAS;
    let bias_uv = range.bias_uv as i32 * (1 << PRECISION) + ROUNDING_CONST_BIAS;

    let i_bias_y = range.bias_y as i32;
    let i_cap_y = range.range_y as i32 + i_bias_y;
    let i_cap_uv = i_bias_y + range.range_uv as i32;

    let iterator_step = match chroma_subsampling {
        YuvChromaSample::YUV420 => 2usize,
        YuvChromaSample::YUV422 => 2usize,
        YuvChromaSample::YUV444 => 1usize,
    };

    let y_iter;
    let u_iter;
    let v_iter;
    let rgb_iter;
    let rgb_linearized_iter;

    #[cfg(feature = "rayon")]
    {
        y_iter = y_plane.par_chunks_exact_mut(y_stride as usize * 2);
        u_iter = u_plane.par_chunks_exact_mut(u_stride as usize);
        v_iter = v_plane.par_chunks_exact_mut(v_stride as usize);
        rgb_iter = rgba.par_chunks_exact(rgba_stride as usize * 2);
        rgb_linearized_iter = rgb_layout.par_chunks_exact(rgb_layout_stride_len * 2);
    }
    #[cfg(not(feature = "rayon"))]
    {
        y_iter = y_plane.chunks_exact_mut(y_stride as usize * 2);
        u_iter = u_plane.chunks_exact_mut(u_stride as usize);
        v_iter = v_plane.chunks_exact_mut(v_stride as usize);
        rgb_iter = rgba.chunks_exact(rgba_stride as usize * 2);
        rgb_linearized_iter = rgb_layout.chunks_exact_mut(rgb_layout_stride_len * 2);
    }

    let full_iter = rgb_iter
        .zip(rgb_linearized_iter)
        .zip(y_iter)
        .zip(u_iter)
        .zip(v_iter);

    full_iter
        .enumerate()
        .for_each(|(j, ((((rgba, rgb_layout), y_plane), u_plane), v_plane))| {
            let mut y_offset = 0usize;
            let u_offset = 0usize;
            let v_offset = 0usize;
            let mut rgba_offset = 0usize;
            let mut rgb_layout_offset = 0usize;

            let v_y = j * 2;

            for y in v_y..=v_y + 1 {
                let y_even_row = y & 1 == 0;

                let mut _cx = 0usize;
                let mut _ux = 0usize;

                #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
                unsafe {
                    let rgb_layout_src = rgb_layout.get_unchecked(rgb_layout_offset..);
                    let rgb_layout_next_src = if y + 1 < height as usize {
                        rgb_layout_src.get_unchecked(rgb_layout_stride_len..)
                    } else {
                        rgb_layout_src
                    };
                    let offset = neon_rgba_to_sharp_yuv::<ORIGIN_CHANNELS, SAMPLING, PRECISION>(
                        &transform,
                        &range,
                        y_plane.as_mut_ptr().add(y_offset),
                        u_plane.as_mut_ptr().add(u_offset),
                        v_plane.as_mut_ptr().add(v_offset),
                        rgba,
                        rgba_offset,
                        rgb_layout_src,
                        rgb_layout_next_src,
                        &gamma_map_table,
                        _cx,
                        _ux,
                        width as usize,
                        y_even_row,
                    );
                    _cx = offset.cx;
                    _ux = offset.ux;
                }

                #[allow(clippy::explicit_counter_loop)]
                for x in (_cx..width as usize).step_by(iterator_step) {
                    unsafe {
                        let px = x * channels;
                        let rgba_shift = rgba_offset + px;
                        let src = rgba.get_unchecked(rgba_shift..);
                        let r = *src.get_unchecked(src_chans.get_r_channel_offset()) as i32;
                        let g = *src.get_unchecked(src_chans.get_g_channel_offset()) as i32;
                        let b = *src.get_unchecked(src_chans.get_b_channel_offset()) as i32;

                        let rgb_layout_src =
                            rgb_layout.get_unchecked((rgb_layout_offset + x * 3)..);

                        let sharp_r_c = if y_even_row {
                            *rgb_layout_src.get_unchecked(src_chans.get_r_channel_offset())
                        } else {
                            0
                        };
                        let sharp_g_c = if y_even_row {
                            *rgb_layout_src.get_unchecked(src_chans.get_g_channel_offset())
                        } else {
                            0
                        };
                        let sharp_b_c = if y_even_row {
                            *rgb_layout_src.get_unchecked(src_chans.get_b_channel_offset())
                        } else {
                            0
                        };

                        let mut sharp_r_next = sharp_r_c;
                        let mut sharp_g_next = sharp_g_c;
                        let mut sharp_b_next = sharp_b_c;

                        let y_0 = (r * transform.yr + g * transform.yg + b * transform.yb + bias_y)
                            >> PRECISION;

                        *y_plane.get_unchecked_mut(y_offset + x) =
                            y_0.clamp(i_bias_y, i_cap_y) as u8;

                        match chroma_subsampling {
                            YuvChromaSample::YUV420 | YuvChromaSample::YUV422 => {
                                if x + 1 < width as usize {
                                    let next_px = (x + 1) * channels;
                                    let rgba_shift = rgba_offset + next_px;
                                    let src = rgba.get_unchecked(rgba_shift..);
                                    let r =
                                        *src.get_unchecked(src_chans.get_r_channel_offset()) as i32;
                                    let g =
                                        *src.get_unchecked(src_chans.get_g_channel_offset()) as i32;
                                    let b =
                                        *src.get_unchecked(src_chans.get_b_channel_offset()) as i32;

                                    let y_1 = (r * transform.yr
                                        + g * transform.yg
                                        + b * transform.yb
                                        + bias_y)
                                        >> PRECISION;
                                    *y_plane.get_unchecked_mut(y_offset + x + 1) =
                                        y_1.clamp(i_bias_y, i_cap_y) as u8;

                                    if y_even_row {
                                        let rgb_layout_src_next = rgb_layout_src.get_unchecked(3..);

                                        sharp_r_next = *rgb_layout_src_next
                                            .get_unchecked(src_chans.get_r_channel_offset());
                                        sharp_g_next = *rgb_layout_src_next
                                            .get_unchecked(src_chans.get_g_channel_offset());
                                        sharp_b_next = *rgb_layout_src_next
                                            .get_unchecked(src_chans.get_b_channel_offset());
                                    }
                                }
                            }
                            _ => {}
                        }

                        if y_even_row {
                            let mut sharp_r_c_next_row = sharp_r_c;
                            let mut sharp_g_c_next_row = sharp_g_c;
                            let mut sharp_b_c_next_row = sharp_b_c;

                            let mut sharp_r_next_row = sharp_r_c;
                            let mut sharp_g_next_row = sharp_g_c;
                            let mut sharp_b_next_row = sharp_b_c;

                            if y + 1 < height as usize {
                                let mut rgb_layout_src_next_row =
                                    rgb_layout_src.get_unchecked(rgb_layout_stride_len..);
                                sharp_r_c_next_row = *rgb_layout_src_next_row
                                    .get_unchecked(src_chans.get_r_channel_offset());
                                sharp_g_c_next_row = *rgb_layout_src_next_row
                                    .get_unchecked(src_chans.get_g_channel_offset());
                                sharp_b_c_next_row = *rgb_layout_src_next_row
                                    .get_unchecked(src_chans.get_b_channel_offset());

                                if x + 1 < width as usize {
                                    rgb_layout_src_next_row =
                                        rgb_layout_src_next_row.get_unchecked(3..);
                                    sharp_r_next_row = *rgb_layout_src_next_row
                                        .get_unchecked(src_chans.get_r_channel_offset());
                                    sharp_g_next_row = *rgb_layout_src_next_row
                                        .get_unchecked(src_chans.get_g_channel_offset());
                                    sharp_b_next_row = *rgb_layout_src_next_row
                                        .get_unchecked(src_chans.get_b_channel_offset());
                                }
                            }

                            const ROUNDING_CONST: i32 = 1 << 3;

                            let interpolated_r = ((sharp_r_c as i32 * 9
                                + sharp_r_next as i32 * 3
                                + sharp_r_c_next_row as i32 * 3
                                + sharp_r_next_row as i32
                                + ROUNDING_CONST)
                                >> 4) as u16;
                            let interpolated_g = ((sharp_g_c as i32 * 9
                                + sharp_g_next as i32 * 3
                                + sharp_g_c_next_row as i32 * 3
                                + sharp_g_next_row as i32
                                + ROUNDING_CONST)
                                >> 4) as u16;
                            let interpolated_b = ((sharp_b_c as i32 * 9
                                + sharp_b_next as i32 * 3
                                + sharp_b_c_next_row as i32 * 3
                                + sharp_b_next_row as i32
                                + ROUNDING_CONST)
                                >> 4) as u16;

                            let corrected_r =
                                *gamma_map_table.get_unchecked(interpolated_r as usize) as i32;
                            let corrected_g =
                                *gamma_map_table.get_unchecked(interpolated_g as usize) as i32;
                            let corrected_b =
                                *gamma_map_table.get_unchecked(interpolated_b as usize) as i32;

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

                            let u_pos = match chroma_subsampling {
                                YuvChromaSample::YUV420 | YuvChromaSample::YUV422 => u_offset + _ux,
                                YuvChromaSample::YUV444 => u_offset + _ux,
                            };
                            *u_plane.get_unchecked_mut(u_pos) = cb.clamp(i_bias_y, i_cap_uv) as u8;
                            let v_pos = match chroma_subsampling {
                                YuvChromaSample::YUV420 | YuvChromaSample::YUV422 => v_offset + _ux,
                                YuvChromaSample::YUV444 => v_offset + _ux,
                            };
                            *v_plane.get_unchecked_mut(v_pos) = cr.clamp(i_bias_y, i_cap_uv) as u8;
                        }
                    }
                    _ux += 1;
                }

                rgba_offset += rgba_stride as usize;
                y_offset += y_stride as usize;
                rgb_layout_offset += rgb_layout_stride_len;
            }
        });

    // Handle last row if image is odd
    if height & 1 != 0 {
        let y_iter = y_plane.chunks_exact_mut(y_stride as usize).rev().take(1);
        let u_iter = u_plane.chunks_exact_mut(u_stride as usize).rev().take(1);
        let v_iter = v_plane.chunks_exact_mut(v_stride as usize).rev().take(1);
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
            let y_offset = 0usize;
            let rgba_offset = 0usize;

            let mut _cx = 0usize;
            let mut _ux = 0usize;

            #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
            unsafe {
                let rgb_layout_src = rgb_layout.get_unchecked(0..);
                let rgb_layout_next_src = rgb_layout_src;
                let offset = neon_rgba_to_sharp_yuv::<ORIGIN_CHANNELS, SAMPLING, PRECISION>(
                    &transform,
                    &range,
                    y_plane.as_mut_ptr(),
                    u_plane.as_mut_ptr(),
                    v_plane.as_mut_ptr(),
                    rgba,
                    rgba_offset,
                    rgb_layout_src,
                    rgb_layout_next_src,
                    &gamma_map_table,
                    _cx,
                    _ux,
                    width as usize,
                    true,
                );
                _cx = offset.cx;
                _ux = offset.ux;
            }

            #[allow(clippy::explicit_counter_loop)]
            for x in (_cx..width as usize).step_by(iterator_step) {
                unsafe {
                    let px = x * channels;
                    let rgba_shift = rgba_offset + px;
                    let src = rgba.get_unchecked(rgba_shift..);
                    let r = *src.get_unchecked(src_chans.get_r_channel_offset()) as i32;
                    let g = *src.get_unchecked(src_chans.get_g_channel_offset()) as i32;
                    let b = *src.get_unchecked(src_chans.get_b_channel_offset()) as i32;

                    let rgb_layout_src = rgb_layout.get_unchecked((x * 3)..);

                    let sharp_r_c = *rgb_layout_src.get_unchecked(src_chans.get_r_channel_offset());
                    let sharp_g_c = *rgb_layout_src.get_unchecked(src_chans.get_g_channel_offset());
                    let sharp_b_c = *rgb_layout_src.get_unchecked(src_chans.get_b_channel_offset());

                    let mut sharp_r_next = sharp_r_c;
                    let mut sharp_g_next = sharp_g_c;
                    let mut sharp_b_next = sharp_b_c;

                    let y_0 = (r * transform.yr + g * transform.yg + b * transform.yb + bias_y)
                        >> PRECISION;

                    *y_plane.get_unchecked_mut(y_offset + x) = y_0.clamp(i_bias_y, i_cap_y) as u8;

                    match chroma_subsampling {
                        YuvChromaSample::YUV420 | YuvChromaSample::YUV422 => {
                            if x + 1 < width as usize {
                                let next_px = (x + 1) * channels;
                                let rgba_shift = rgba_offset + next_px;
                                let src = rgba.get_unchecked(rgba_shift..);
                                let r = *src.get_unchecked(src_chans.get_r_channel_offset()) as i32;
                                let g = *src.get_unchecked(src_chans.get_g_channel_offset()) as i32;
                                let b = *src.get_unchecked(src_chans.get_b_channel_offset()) as i32;

                                let y_1 = (r * transform.yr
                                    + g * transform.yg
                                    + b * transform.yb
                                    + bias_y)
                                    >> PRECISION;
                                *y_plane.get_unchecked_mut(y_offset + x + 1) =
                                    y_1.clamp(i_bias_y, i_cap_y) as u8;

                                let rgb_layout_src_next = rgb_layout_src.get_unchecked(3..);

                                sharp_r_next = *rgb_layout_src_next
                                    .get_unchecked(src_chans.get_r_channel_offset());
                                sharp_g_next = *rgb_layout_src_next
                                    .get_unchecked(src_chans.get_g_channel_offset());
                                sharp_b_next = *rgb_layout_src_next
                                    .get_unchecked(src_chans.get_b_channel_offset());
                            }
                        }
                        _ => {}
                    }

                    let sharp_r_c_next_row = sharp_r_c;
                    let sharp_g_c_next_row = sharp_g_c;
                    let sharp_b_c_next_row = sharp_b_c;

                    let sharp_r_next_row = sharp_r_c;
                    let sharp_g_next_row = sharp_g_c;
                    let sharp_b_next_row = sharp_b_c;

                    const ROUNDING_CONST: i32 = 1 << 3;

                    let interpolated_r = ((sharp_r_c as i32 * 9
                        + sharp_r_next as i32 * 3
                        + sharp_r_c_next_row as i32 * 3
                        + sharp_r_next_row as i32
                        + ROUNDING_CONST)
                        >> 4) as u16;
                    let interpolated_g = ((sharp_g_c as i32 * 9
                        + sharp_g_next as i32 * 3
                        + sharp_g_c_next_row as i32 * 3
                        + sharp_g_next_row as i32
                        + ROUNDING_CONST)
                        >> 4) as u16;
                    let interpolated_b = ((sharp_b_c as i32 * 9
                        + sharp_b_next as i32 * 3
                        + sharp_b_c_next_row as i32 * 3
                        + sharp_b_next_row as i32
                        + ROUNDING_CONST)
                        >> 4) as u16;

                    let corrected_r =
                        *gamma_map_table.get_unchecked(interpolated_r as usize) as i32;
                    let corrected_g =
                        *gamma_map_table.get_unchecked(interpolated_g as usize) as i32;
                    let corrected_b =
                        *gamma_map_table.get_unchecked(interpolated_b as usize) as i32;

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

                    let u_pos = match chroma_subsampling {
                        YuvChromaSample::YUV420 | YuvChromaSample::YUV422 => _ux,
                        YuvChromaSample::YUV444 => _ux,
                    };
                    *u_plane.get_unchecked_mut(u_pos) = cb.clamp(i_bias_y, i_cap_uv) as u8;
                    let v_pos = match chroma_subsampling {
                        YuvChromaSample::YUV420 | YuvChromaSample::YUV422 => _ux,
                        YuvChromaSample::YUV444 => _ux,
                    };
                    *v_plane.get_unchecked_mut(v_pos) = cr.clamp(i_bias_y, i_cap_uv) as u8;
                }
                _ux += 1;
            }
        });
    }
}

/// Convert RGB image data to YUV 422 planar format using bi-linear interpolation and gamma correction ( sharp YUV algorithm ).
///
/// This function performs RGB to YUV conversion using bi-linear interpolation and gamma correction and stores the result in YUV422 planar format using bi-linear interpolation and gamma correction,
/// with separate planes for Y (luminance), U (chrominance), and V (chrominance) components.
///
/// # Arguments
///
/// * `y_plane` - A mutable slice to store the Y (luminance) plane data.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `u_plane` - A mutable slice to store the U (chrominance) plane data.
/// * `u_stride` - The stride (bytes per row) for the U plane.
/// * `v_plane` - A mutable slice to store the V (chrominance) plane data.
/// * `v_stride` - The stride (bytes per row) for the V plane.
/// * `rgb` - The input RGB image data slice.
/// * `rgb_stride` - The stride (bytes per row) for the RGB image data.
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
pub fn rgb_to_sharp_yuv422(
    y_plane: &mut [u8],
    y_stride: u32,
    u_plane: &mut [u8],
    u_stride: u32,
    v_plane: &mut [u8],
    v_stride: u32,
    rgb: &[u8],
    rgb_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    gamma_transfer: SharpYuvGammaTransfer,
) {
    rgbx_to_sharp_yuv::<{ YuvSourceChannels::Rgb as u8 }, { YuvChromaSample::YUV422 as u8 }>(
        y_plane,
        y_stride,
        u_plane,
        u_stride,
        v_plane,
        v_stride,
        rgb,
        rgb_stride,
        width,
        height,
        range,
        matrix,
        gamma_transfer,
    );
}

/// Convert BGR image data to YUV 422 planar format using bi-linear interpolation and gamma correction ( sharp YUV algorithm ).
///
/// This function performs BGR to YUV conversion using bi-linear interpolation and gamma correction and stores the result in YUV422 planar format using bi-linear interpolation and gamma correction,
/// with separate planes for Y (luminance), U (chrominance), and V (chrominance) components.
///
/// # Arguments
///
/// * `y_plane` - A mutable slice to store the Y (luminance) plane data.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `u_plane` - A mutable slice to store the U (chrominance) plane data.
/// * `u_stride` - The stride (bytes per row) for the U plane.
/// * `v_plane` - A mutable slice to store the V (chrominance) plane data.
/// * `v_stride` - The stride (bytes per row) for the V plane.
/// * `bgr` - The input BGR image data slice.
/// * `bgr_stride` - The stride (bytes per row) for the BGR image data.
/// * `width` - The width of the image in pixels.
/// * `height` - The height of the image in pixels.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGR data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn bgr_to_sharp_yuv422(
    y_plane: &mut [u8],
    y_stride: u32,
    u_plane: &mut [u8],
    u_stride: u32,
    v_plane: &mut [u8],
    v_stride: u32,
    bgr: &[u8],
    bgr_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    gamma_transfer: SharpYuvGammaTransfer,
) {
    rgbx_to_sharp_yuv::<{ YuvSourceChannels::Bgr as u8 }, { YuvChromaSample::YUV422 as u8 }>(
        y_plane,
        y_stride,
        u_plane,
        u_stride,
        v_plane,
        v_stride,
        bgr,
        bgr_stride,
        width,
        height,
        range,
        matrix,
        gamma_transfer,
    );
}

/// Convert RGBA image data to YUV 422 planar format using bi-linear interpolation and gamma correction ( sharp YUV algorithm ).
///
/// This function performs RGBA to YUV conversion using bi-linear interpolation and gamma correction and stores the result in YUV422 planar format using bi-linear interpolation and gamma correction,
/// with separate planes for Y (luminance), U (chrominance), and V (chrominance) components.
///
/// # Arguments
///
/// * `y_plane` - A mutable slice to store the Y (luminance) plane data.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `u_plane` - A mutable slice to store the U (chrominance) plane data.
/// * `u_stride` - The stride (bytes per row) for the U plane.
/// * `v_plane` - A mutable slice to store the V (chrominance) plane data.
/// * `v_stride` - The stride (bytes per row) for the V plane.
/// * `rgba` - The input RGBA image data slice.
/// * `rgba_stride` - The stride (bytes per row) for the RGBA image data.
/// * `width` - The width of the image in pixels.
/// * `height` - The height of the image in pixels.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input RGBA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn rgba_to_sharp_yuv422(
    y_plane: &mut [u8],
    y_stride: u32,
    u_plane: &mut [u8],
    u_stride: u32,
    v_plane: &mut [u8],
    v_stride: u32,
    rgba: &[u8],
    rgba_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    gamma_transfer: SharpYuvGammaTransfer,
) {
    rgbx_to_sharp_yuv::<{ YuvSourceChannels::Rgba as u8 }, { YuvChromaSample::YUV422 as u8 }>(
        y_plane,
        y_stride,
        u_plane,
        u_stride,
        v_plane,
        v_stride,
        rgba,
        rgba_stride,
        width,
        height,
        range,
        matrix,
        gamma_transfer,
    );
}

/// Convert BGRA image data to YUV 422 planar format using bi-linear interpolation and gamma correction ( sharp YUV algorithm ).
///
/// This function performs BGRA to YUV conversion using bi-linear interpolation and gamma correction and stores the result in YUV422 planar format using bi-linear interpolation and gamma correction,
/// with separate planes for Y (luminance), U (chrominance), and V (chrominance) components.
///
/// # Arguments
///
/// * `y_plane` - A mutable slice to store the Y (luminance) plane data.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `u_plane` - A mutable slice to store the U (chrominance) plane data.
/// * `u_stride` - The stride (bytes per row) for the U plane.
/// * `v_plane` - A mutable slice to store the V (chrominance) plane data.
/// * `v_stride` - The stride (bytes per row) for the V plane.
/// * `bgra` - The input BGRA image data slice.
/// * `bgra_stride` - The stride (bytes per row) for the BGRA image data.
/// * `width` - The width of the image in pixels.
/// * `height` - The height of the image in pixels.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGRA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn bgra_to_sharp_yuv422(
    y_plane: &mut [u8],
    y_stride: u32,
    u_plane: &mut [u8],
    u_stride: u32,
    v_plane: &mut [u8],
    v_stride: u32,
    bgra: &[u8],
    bgra_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    gamma_transfer: SharpYuvGammaTransfer,
) {
    rgbx_to_sharp_yuv::<{ YuvSourceChannels::Bgra as u8 }, { YuvChromaSample::YUV422 as u8 }>(
        y_plane,
        y_stride,
        u_plane,
        u_stride,
        v_plane,
        v_stride,
        bgra,
        bgra_stride,
        width,
        height,
        range,
        matrix,
        gamma_transfer,
    );
}

/// Convert RGB image data to YUV 420 planar format using bi-linear interpolation and gamma correction ( sharp YUV algorithm ).
///
/// This function performs RGB to YUV conversion using bi-linear interpolation and gamma correction and stores the result in YUV420 planar format using bi-linear interpolation and gamma correction,
/// with separate planes for Y (luminance), U (chrominance), and V (chrominance) components.
///
/// # Arguments
///
/// * `y_plane` - A mutable slice to store the Y (luminance) plane data.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `u_plane` - A mutable slice to store the U (chrominance) plane data.
/// * `u_stride` - The stride (bytes per row) for the U plane.
/// * `v_plane` - A mutable slice to store the V (chrominance) plane data.
/// * `v_stride` - The stride (bytes per row) for the V plane.
/// * `rgb` - The input RGB image data slice.
/// * `rgb_stride` - The stride (bytes per row) for the RGB image data.
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
    y_plane: &mut [u8],
    y_stride: u32,
    u_plane: &mut [u8],
    u_stride: u32,
    v_plane: &mut [u8],
    v_stride: u32,
    rgb: &[u8],
    rgb_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    gamma_transfer: SharpYuvGammaTransfer,
) {
    rgbx_to_sharp_yuv::<{ YuvSourceChannels::Rgb as u8 }, { YuvChromaSample::YUV420 as u8 }>(
        y_plane,
        y_stride,
        u_plane,
        u_stride,
        v_plane,
        v_stride,
        rgb,
        rgb_stride,
        width,
        height,
        range,
        matrix,
        gamma_transfer,
    );
}

/// Convert BGR image data to YUV 420 planar format using bi-linear interpolation and gamma correction ( sharp YUV algorithm ).
///
/// This function performs BGR to YUV conversion using bi-linear interpolation and gamma correction and stores the result in YUV420 planar format using bi-linear interpolation and gamma correction,
/// with separate planes for Y (luminance), U (chrominance), and V (chrominance) components.
///
/// # Arguments
///
/// * `y_plane` - A mutable slice to store the Y (luminance) plane data.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `u_plane` - A mutable slice to store the U (chrominance) plane data.
/// * `u_stride` - The stride (bytes per row) for the U plane.
/// * `v_plane` - A mutable slice to store the V (chrominance) plane data.
/// * `v_stride` - The stride (bytes per row) for the V plane.
/// * `bgr` - The input BGR image data slice.
/// * `bgr_stride` - The stride (bytes per row) for the BGR image data.
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
pub fn bgr_to_sharp_yuv420(
    y_plane: &mut [u8],
    y_stride: u32,
    u_plane: &mut [u8],
    u_stride: u32,
    v_plane: &mut [u8],
    v_stride: u32,
    bgr: &[u8],
    bgr_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    gamma_transfer: SharpYuvGammaTransfer,
) {
    rgbx_to_sharp_yuv::<{ YuvSourceChannels::Bgr as u8 }, { YuvChromaSample::YUV420 as u8 }>(
        y_plane,
        y_stride,
        u_plane,
        u_stride,
        v_plane,
        v_stride,
        bgr,
        bgr_stride,
        width,
        height,
        range,
        matrix,
        gamma_transfer,
    );
}

/// Convert RGBA image data to YUV 420 planar format using bi-linear interpolation and gamma correction ( sharp YUV algorithm ).
///
/// This function performs RGBA to YUV conversion using bi-linear interpolation and gamma correction and stores the result in YUV420 planar format using bi-linear interpolation and gamma correction,
/// with separate planes for Y (luminance), U (chrominance), and V (chrominance) components.
///
/// # Arguments
///
/// * `y_plane` - A mutable slice to store the Y (luminance) plane data.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `u_plane` - A mutable slice to store the U (chrominance) plane data.
/// * `u_stride` - The stride (bytes per row) for the U plane.
/// * `v_plane` - A mutable slice to store the V (chrominance) plane data.
/// * `v_stride` - The stride (bytes per row) for the V plane.
/// * `rgba` - The input RGBA image data slice.
/// * `rgba_stride` - The stride (bytes per row) for the RGBA image data.
/// * `width` - The width of the image in pixels.
/// * `height` - The height of the image in pixels.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input RGBA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn rgba_to_sharp_yuv420(
    y_plane: &mut [u8],
    y_stride: u32,
    u_plane: &mut [u8],
    u_stride: u32,
    v_plane: &mut [u8],
    v_stride: u32,
    rgba: &[u8],
    rgba_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    gamma_transfer: SharpYuvGammaTransfer,
) {
    rgbx_to_sharp_yuv::<{ YuvSourceChannels::Rgba as u8 }, { YuvChromaSample::YUV420 as u8 }>(
        y_plane,
        y_stride,
        u_plane,
        u_stride,
        v_plane,
        v_stride,
        rgba,
        rgba_stride,
        width,
        height,
        range,
        matrix,
        gamma_transfer,
    );
}

/// Convert BGRA image data to YUV 420 planar format using bi-linear interpolation and gamma correction ( sharp YUV algorithm ).
///
/// This function performs BGRA to YUV conversion using bi-linear interpolation and gamma correction and stores the result in YUV420 planar format using bi-linear interpolation and gamma correction,
/// with separate planes for Y (luminance), U (chrominance), and V (chrominance) components.
///
/// # Arguments
///
/// * `y_plane` - A mutable slice to store the Y (luminance) plane data.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `u_plane` - A mutable slice to store the U (chrominance) plane data.
/// * `u_stride` - The stride (bytes per row) for the U plane.
/// * `v_plane` - A mutable slice to store the V (chrominance) plane data.
/// * `v_stride` - The stride (bytes per row) for the V plane.
/// * `bgra` - The input BGRA image data slice.
/// * `bgra_stride` - The stride (bytes per row) for the BGRA image data.
/// * `width` - The width of the image in pixels.
/// * `height` - The height of the image in pixels.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGRA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn bgra_to_sharp_yuv420(
    y_plane: &mut [u8],
    y_stride: u32,
    u_plane: &mut [u8],
    u_stride: u32,
    v_plane: &mut [u8],
    v_stride: u32,
    bgra: &[u8],
    bgra_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    gamma_transfer: SharpYuvGammaTransfer,
) {
    rgbx_to_sharp_yuv::<{ YuvSourceChannels::Bgra as u8 }, { YuvChromaSample::YUV420 as u8 }>(
        y_plane,
        y_stride,
        u_plane,
        u_stride,
        v_plane,
        v_stride,
        bgra,
        bgra_stride,
        width,
        height,
        range,
        matrix,
        gamma_transfer,
    );
}
