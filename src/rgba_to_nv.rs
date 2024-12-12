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
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use crate::avx2::{avx2_rgba_to_nv, avx2_rgba_to_nv420};
use crate::built_coefficients::get_built_forward_transform;
use crate::images::YuvBiPlanarImageMut;
use crate::internals::ProcessedOffset;
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use crate::neon::{
    neon_rgbx_to_nv_row, neon_rgbx_to_nv_row420, neon_rgbx_to_nv_row_rdm,
    neon_rgbx_to_nv_row_rdm420,
};
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use crate::sse::{sse_rgba_to_nv_row, sse_rgba_to_nv_row420};
use crate::yuv_error::check_rgba_destination;
use crate::yuv_support::*;
use crate::YuvError;
#[cfg(feature = "rayon")]
use rayon::iter::{IndexedParallelIterator, ParallelIterator};
#[cfg(feature = "rayon")]
use rayon::prelude::{ParallelSlice, ParallelSliceMut};

fn rgbx_to_nv<const ORIGIN_CHANNELS: u8, const UV_ORDER: u8, const SAMPLING: u8>(
    image: &mut YuvBiPlanarImageMut<u8>,
    rgba: &[u8],
    rgba_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) -> Result<(), YuvError> {
    let order: YuvNVOrder = UV_ORDER.into();
    let chroma_subsampling: YuvChromaSubsampling = SAMPLING.into();
    let src_chans: YuvSourceChannels = ORIGIN_CHANNELS.into();
    let channels = src_chans.get_channels_count();

    check_rgba_destination(rgba, rgba_stride, image.width, image.height, channels)?;
    image.check_constraints(chroma_subsampling)?;

    let chroma_range = get_yuv_range(8, range);
    let kr_kb = matrix.get_kr_kb();
    let max_range_p8 = (1u32 << 8u32) - 1;

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

    const ROUNDING_CONST_BIAS: i32 = (1 << (PRECISION - 1)) - 1;
    let bias_y = chroma_range.bias_y as i32 * (1 << PRECISION) + ROUNDING_CONST_BIAS;
    let bias_uv = chroma_range.bias_uv as i32 * (1 << PRECISION) + ROUNDING_CONST_BIAS;

    let i_bias_y = chroma_range.bias_y as i32;
    let i_cap_y = chroma_range.range_y as i32 + i_bias_y;
    let i_cap_uv = i_bias_y + chroma_range.range_uv as i32;

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    let use_sse = std::arch::is_x86_feature_detected!("sse4.1");
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    let use_avx2 = std::arch::is_x86_feature_detected!("avx2");
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    let is_rdm_available = std::arch::is_aarch64_feature_detected!("rdm");
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    let neon_wide_row_handler = if is_rdm_available {
        neon_rgbx_to_nv_row_rdm::<ORIGIN_CHANNELS, UV_ORDER, SAMPLING>
    } else {
        neon_rgbx_to_nv_row::<ORIGIN_CHANNELS, UV_ORDER, SAMPLING, PRECISION>
    };
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    let neon_wide_row420_handler = if is_rdm_available {
        neon_rgbx_to_nv_row_rdm420::<ORIGIN_CHANNELS, UV_ORDER>
    } else {
        neon_rgbx_to_nv_row420::<ORIGIN_CHANNELS, UV_ORDER, PRECISION>
    };

    let width = image.width;

    let process_wide_row =
        |_y_plane: &mut [u8], _uv_plane: &mut [u8], _rgba: &[u8], _compute_uv_row| {
            let mut _offset: ProcessedOffset = ProcessedOffset { cx: 0, ux: 0 };
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            if use_avx2 {
                let offset = avx2_rgba_to_nv::<ORIGIN_CHANNELS, UV_ORDER, SAMPLING>(
                    _y_plane,
                    _uv_plane,
                    _rgba,
                    width,
                    &chroma_range,
                    &transform,
                    _offset.cx,
                    _offset.ux,
                    _compute_uv_row,
                );
                _offset = offset;
            }
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            if use_sse {
                let offset = sse_rgba_to_nv_row::<ORIGIN_CHANNELS, UV_ORDER, SAMPLING>(
                    _y_plane,
                    _uv_plane,
                    _rgba,
                    width,
                    &chroma_range,
                    &transform,
                    _offset.cx,
                    _offset.ux,
                    _compute_uv_row,
                );
                _offset = offset;
            }

            #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
            unsafe {
                let offset = neon_wide_row_handler(
                    _y_plane,
                    _uv_plane,
                    _rgba,
                    width,
                    &chroma_range,
                    &transform,
                    _offset.cx,
                    _offset.ux,
                    _compute_uv_row,
                );
                _offset = offset
            }
            _offset
        };

    let process_halved_row = |y_dst: &mut [u8], uv_dst: &mut [u8], rgba: &[u8]| {
        let offset = process_wide_row(y_dst, uv_dst, rgba, true);

        for ((y_dst, uv_dst), rgba) in y_dst
            .chunks_exact_mut(2)
            .zip(uv_dst.chunks_exact_mut(2))
            .zip(rgba.chunks_exact(channels * 2))
            .skip(offset.cx / 2)
        {
            let rgba0 = &rgba[0..channels];
            let r0 = rgba0[src_chans.get_r_channel_offset()] as i32;
            let g0 = rgba0[src_chans.get_g_channel_offset()] as i32;
            let b0 = rgba0[src_chans.get_b_channel_offset()] as i32;
            let y_0 =
                (r0 * transform.yr + g0 * transform.yg + b0 * transform.yb + bias_y) >> PRECISION;
            y_dst[0] = y_0.min(i_cap_y) as u8;

            let rgba1 = &rgba[channels..channels * 2];

            let r1 = rgba1[src_chans.get_r_channel_offset()] as i32;
            let g1 = rgba1[src_chans.get_g_channel_offset()] as i32;
            let b1 = rgba1[src_chans.get_b_channel_offset()] as i32;

            let y_1 =
                (r1 * transform.yr + g1 * transform.yg + b1 * transform.yb + bias_y) >> PRECISION;
            y_dst[1] = y_1.min(i_cap_y) as u8;

            let r = (r0 + r1 + 1) >> 1;
            let g = (g0 + g1 + 1) >> 1;
            let b = (b0 + b1 + 1) >> 1;

            let cb = (r * transform.cb_r + g * transform.cb_g + b * transform.cb_b + bias_uv)
                >> PRECISION;
            let cr = (r * transform.cr_r + g * transform.cr_g + b * transform.cr_b + bias_uv)
                >> PRECISION;
            uv_dst[order.get_u_position()] = cb.max(i_bias_y).min(i_cap_uv) as u8;
            uv_dst[order.get_v_position()] = cr.max(i_bias_y).min(i_cap_uv) as u8;
        }

        if width & 1 != 0 {
            let rgba = rgba.chunks_exact(channels * 2).remainder();
            let rgba = &rgba[0..channels];
            let uv_dst = uv_dst.chunks_exact_mut(2).last().unwrap();
            let y_dst = y_dst.chunks_exact_mut(2).into_remainder();

            let r0 = rgba[src_chans.get_r_channel_offset()] as i32;
            let g0 = rgba[src_chans.get_g_channel_offset()] as i32;
            let b0 = rgba[src_chans.get_b_channel_offset()] as i32;
            let y_0 =
                (r0 * transform.yr + g0 * transform.yg + b0 * transform.yb + bias_y) >> PRECISION;
            y_dst[0] = y_0.min(i_cap_y) as u8;

            let cb = (r0 * transform.cb_r + g0 * transform.cb_g + b0 * transform.cb_b + bias_uv)
                >> PRECISION;
            let cr = (r0 * transform.cr_r + g0 * transform.cr_g + b0 * transform.cr_b + bias_uv)
                >> PRECISION;
            uv_dst[order.get_u_position()] = cb.max(i_bias_y).min(i_cap_uv) as u8;
            uv_dst[order.get_v_position()] = cr.max(i_bias_y).min(i_cap_uv) as u8;
        }
    };

    let process_wide_row420 = |_y_plane0: &mut [u8],
                               _y_plane1: &mut [u8],
                               _uv_plane: &mut [u8],
                               _rgba0: &[u8],
                               _rgba1: &[u8]| {
        let mut _offset: ProcessedOffset = ProcessedOffset { cx: 0, ux: 0 };
        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        unsafe {
            let offset = neon_wide_row420_handler(
                _y_plane0,
                _y_plane1,
                _uv_plane,
                _rgba0,
                _rgba1,
                width,
                &chroma_range,
                &transform,
                _offset.cx,
                _offset.ux,
            );
            _offset = offset;
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if use_avx2 {
                let offset = avx2_rgba_to_nv420::<ORIGIN_CHANNELS, UV_ORDER>(
                    _y_plane0,
                    _y_plane1,
                    _uv_plane,
                    _rgba0,
                    _rgba1,
                    width,
                    &chroma_range,
                    &transform,
                    _offset.cx,
                    _offset.ux,
                );
                _offset = offset;
            }
            if use_sse {
                let offset = sse_rgba_to_nv_row420::<ORIGIN_CHANNELS, UV_ORDER>(
                    _y_plane0,
                    _y_plane1,
                    _uv_plane,
                    _rgba0,
                    _rgba1,
                    width,
                    &chroma_range,
                    &transform,
                    _offset.cx,
                    _offset.ux,
                );
                _offset = offset;
            }
        }
        _offset
    };

    let process_double_row =
        |y_dst0: &mut [u8], y_dst1: &mut [u8], uv_dst: &mut [u8], rgba0: &[u8], rgba1: &[u8]| {
            let offset = process_wide_row420(y_dst0, y_dst1, uv_dst, rgba0, rgba1);

            for ((((y_dst0, y_dst1), uv_dst), rgba0), rgba1) in y_dst0
                .chunks_exact_mut(2)
                .zip(y_dst1.chunks_exact_mut(2))
                .zip(uv_dst.chunks_exact_mut(2))
                .zip(rgba0.chunks_exact(channels * 2))
                .zip(rgba1.chunks_exact(channels * 2))
                .skip(offset.cx / 2)
            {
                let rgba00 = &rgba0[0..channels];
                let r00 = rgba00[src_chans.get_r_channel_offset()] as i32;
                let g00 = rgba00[src_chans.get_g_channel_offset()] as i32;
                let b00 = rgba00[src_chans.get_b_channel_offset()] as i32;
                let y_00 = (r00 * transform.yr + g00 * transform.yg + b00 * transform.yb + bias_y)
                    >> PRECISION;
                y_dst0[0] = y_00.min(i_cap_y) as u8;

                let rgba01 = &rgba0[channels..channels * 2];

                let r01 = rgba01[src_chans.get_r_channel_offset()] as i32;
                let g01 = rgba01[src_chans.get_g_channel_offset()] as i32;
                let b01 = rgba01[src_chans.get_b_channel_offset()] as i32;

                let y_1 = (r01 * transform.yr + g01 * transform.yg + b01 * transform.yb + bias_y)
                    >> PRECISION;
                y_dst0[1] = y_1.min(i_cap_y) as u8;

                let rgba10 = &rgba1[0..channels];
                let r10 = rgba10[src_chans.get_r_channel_offset()] as i32;
                let g10 = rgba10[src_chans.get_g_channel_offset()] as i32;
                let b10 = rgba10[src_chans.get_b_channel_offset()] as i32;
                let y_10 = (r10 * transform.yr + g10 * transform.yg + b10 * transform.yb + bias_y)
                    >> PRECISION;
                y_dst1[0] = y_10.min(i_cap_y) as u8;

                let rgba11 = &rgba1[channels..channels * 2];
                let r11 = rgba11[src_chans.get_r_channel_offset()] as i32;
                let g11 = rgba11[src_chans.get_g_channel_offset()] as i32;
                let b11 = rgba11[src_chans.get_b_channel_offset()] as i32;
                let y_11 = (r11 * transform.yr + g11 * transform.yg + b11 * transform.yb + bias_y)
                    >> PRECISION;
                y_dst1[1] = y_11.min(i_cap_y) as u8;

                let r = (r00 + r01 + r10 + r11 + 2) >> 2;
                let g = (g00 + g01 + g10 + g11 + 2) >> 2;
                let b = (b00 + b01 + b10 + b11 + 2) >> 2;

                let cb = (r * transform.cb_r + g * transform.cb_g + b * transform.cb_b + bias_uv)
                    >> PRECISION;
                let cr = (r * transform.cr_r + g * transform.cr_g + b * transform.cr_b + bias_uv)
                    >> PRECISION;
                uv_dst[order.get_u_position()] = cb.max(i_bias_y).min(i_cap_uv) as u8;
                uv_dst[order.get_v_position()] = cr.max(i_bias_y).min(i_cap_uv) as u8;
            }

            if width & 1 != 0 {
                let rgba0 = rgba0.chunks_exact(channels * 2).remainder();
                let rgba0 = &rgba0[0..channels];
                let rgba1 = rgba1.chunks_exact(channels * 2).remainder();
                let rgba1 = &rgba1[0..channels];
                let uv_dst = uv_dst.chunks_exact_mut(2).last().unwrap();
                let y_dst0 = y_dst0.chunks_exact_mut(2).into_remainder();

                let r0 = rgba0[src_chans.get_r_channel_offset()] as i32;
                let g0 = rgba0[src_chans.get_g_channel_offset()] as i32;
                let b0 = rgba0[src_chans.get_b_channel_offset()] as i32;
                let y_0 = (r0 * transform.yr + g0 * transform.yg + b0 * transform.yb + bias_y)
                    >> PRECISION;
                y_dst0[0] = y_0.min(i_cap_y) as u8;

                let r1 = rgba1[src_chans.get_r_channel_offset()] as i32;
                let g1 = rgba1[src_chans.get_g_channel_offset()] as i32;
                let b1 = rgba1[src_chans.get_b_channel_offset()] as i32;
                let y_1 = (r1 * transform.yr + g1 * transform.yg + b1 * transform.yb + bias_y)
                    >> PRECISION;
                y_dst1[0] = y_1.min(i_cap_y) as u8;

                let r = (r0 + r1 + 1) >> 1;
                let g = (g0 + g1 + 1) >> 1;
                let b = (b0 + b1 + 1) >> 1;

                let cb = (r * transform.cb_r + g * transform.cb_g + b * transform.cb_b + bias_uv)
                    >> PRECISION;
                let cr = (r * transform.cr_r + g * transform.cr_g + b * transform.cr_b + bias_uv)
                    >> PRECISION;
                uv_dst[order.get_u_position()] = cb.max(i_bias_y).min(i_cap_uv) as u8;
                uv_dst[order.get_v_position()] = cr.max(i_bias_y).min(i_cap_uv) as u8;
            }
        };

    let y_plane = image.y_plane.borrow_mut();
    let y_stride = image.y_stride;
    let uv_plane = image.uv_plane.borrow_mut();
    let uv_stride = image.uv_stride;

    if chroma_subsampling == YuvChromaSubsampling::Yuv444 {
        let iter;
        #[cfg(feature = "rayon")]
        {
            iter = y_plane
                .par_chunks_exact_mut(y_stride as usize)
                .zip(uv_plane.par_chunks_exact_mut(uv_stride as usize))
                .zip(rgba.par_chunks_exact(rgba_stride as usize));
        }
        #[cfg(not(feature = "rayon"))]
        {
            iter = y_plane
                .chunks_exact_mut(y_stride as usize)
                .zip(uv_plane.chunks_exact_mut(uv_stride as usize))
                .zip(rgba.chunks_exact(rgba_stride as usize));
        }
        iter.for_each(|((y_dst, uv_dst), rgba)| {
            let y_dst = &mut y_dst[0..image.width as usize];
            let offset = process_wide_row(y_dst, uv_dst, rgba, true);

            for ((y_dst, uv_dst), rgba) in y_dst
                .iter_mut()
                .zip(uv_dst.chunks_exact_mut(2))
                .zip(rgba.chunks_exact(channels))
                .skip(offset.cx)
            {
                let r0 = rgba[src_chans.get_r_channel_offset()] as i32;
                let g0 = rgba[src_chans.get_g_channel_offset()] as i32;
                let b0 = rgba[src_chans.get_b_channel_offset()] as i32;
                let y_0 = (r0 * transform.yr + g0 * transform.yg + b0 * transform.yb + bias_y)
                    >> PRECISION;
                *y_dst = y_0.min(i_cap_y) as u8;
                let cb =
                    (r0 * transform.cb_r + g0 * transform.cb_g + b0 * transform.cb_b + bias_uv)
                        >> PRECISION;
                let cr =
                    (r0 * transform.cr_r + g0 * transform.cr_g + b0 * transform.cr_b + bias_uv)
                        >> PRECISION;
                uv_dst[order.get_u_position()] = cb.max(i_bias_y).min(i_cap_uv) as u8;
                uv_dst[order.get_v_position()] = cr.max(i_bias_y).min(i_cap_uv) as u8;
            }
        });
    } else if chroma_subsampling == YuvChromaSubsampling::Yuv422 {
        let iter;
        #[cfg(feature = "rayon")]
        {
            iter = y_plane
                .par_chunks_exact_mut(y_stride as usize)
                .zip(uv_plane.par_chunks_exact_mut(uv_stride as usize))
                .zip(rgba.par_chunks_exact(rgba_stride as usize));
        }
        #[cfg(not(feature = "rayon"))]
        {
            iter = y_plane
                .chunks_exact_mut(y_stride as usize)
                .zip(uv_plane.chunks_exact_mut(uv_stride as usize))
                .zip(rgba.chunks_exact(rgba_stride as usize));
        }
        iter.for_each(|((y_dst, uv_dst), rgba)| {
            process_halved_row(
                &mut y_dst[0..image.width as usize],
                &mut uv_dst[0..(image.width as usize).div_ceil(2) * 2],
                &rgba[0..image.width as usize * channels],
            );
        });
    } else if chroma_subsampling == YuvChromaSubsampling::Yuv420 {
        let iter;
        #[cfg(feature = "rayon")]
        {
            iter = y_plane
                .par_chunks_exact_mut(y_stride as usize * 2)
                .zip(uv_plane.par_chunks_exact_mut(uv_stride as usize))
                .zip(rgba.par_chunks_exact(rgba_stride as usize * 2));
        }
        #[cfg(not(feature = "rayon"))]
        {
            iter = y_plane
                .chunks_exact_mut(y_stride as usize * 2)
                .zip(uv_plane.chunks_exact_mut(uv_stride as usize))
                .zip(rgba.chunks_exact(rgba_stride as usize * 2));
        }

        iter.for_each(|((y_dst, uv_dst), rgba)| {
            let (y_dst0, y_dst1) = y_dst.split_at_mut(image.y_stride as usize);
            let (rgba0, rgba1) = rgba.split_at(rgba_stride as usize);
            process_double_row(
                &mut y_dst0[0..image.width as usize],
                &mut y_dst1[0..image.width as usize],
                &mut uv_dst[0..(image.width as usize).div_ceil(2) * 2],
                &rgba0[0..image.width as usize * channels],
                &rgba1[0..image.width as usize * channels],
            );
        });

        if image.height & 1 != 0 {
            let y_dst = y_plane
                .chunks_exact_mut(y_stride as usize * 2)
                .into_remainder();
            let uv_dst = uv_plane
                .chunks_exact_mut(uv_stride as usize)
                .last()
                .unwrap();
            let rgba = rgba.chunks_exact(rgba_stride as usize * 2).remainder();
            process_halved_row(
                &mut y_dst[0..image.width as usize],
                &mut uv_dst[0..(image.width as usize).div_ceil(2) * 2],
                &rgba[0..image.width as usize * channels],
            );
        }
    }

    Ok(())
}

/// Convert RGB image data to YUV NV16 bi-planar format.
///
/// This function performs RGB to YUV conversion and stores the result in YUV NV16 bi-planar format,
/// with plane for Y (luminance), and bi-plane UV (chrominance) components.
///
/// # Arguments
///
/// * `bi_planar_image` - Target Bi-Planar image
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
pub fn rgb_to_yuv_nv16(
    bi_planar_image: &mut YuvBiPlanarImageMut<u8>,
    rgb: &[u8],
    rgb_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) -> Result<(), YuvError> {
    rgbx_to_nv::<
        { YuvSourceChannels::Rgb as u8 },
        { YuvNVOrder::UV as u8 },
        { YuvChromaSubsampling::Yuv422 as u8 },
    >(bi_planar_image, rgb, rgb_stride, range, matrix)
}

/// Convert RGB image data to YUV NV61 bi-planar format.
///
/// This function performs RGB to YUV conversion and stores the result in YUV NV61 bi-planar format,
/// with plane for Y (luminance), and bi-plane VU (chrominance) components.
///
/// # Arguments
///
/// * `bi_planar_image` - Target Bi-Planar image
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
pub fn rgb_to_yuv_nv61(
    bi_planar_image: &mut YuvBiPlanarImageMut<u8>,
    rgb: &[u8],
    rgb_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) -> Result<(), YuvError> {
    rgbx_to_nv::<
        { YuvSourceChannels::Rgb as u8 },
        { YuvNVOrder::VU as u8 },
        { YuvChromaSubsampling::Yuv422 as u8 },
    >(bi_planar_image, rgb, rgb_stride, range, matrix)
}

/// Convert BGR image data to YUV NV16 bi-planar format.
///
/// This function performs BGR to YUV conversion and stores the result in YUV NV16 bi-planar format,
/// with plane for Y (luminance), and bi-plane UV (chrominance) components.
///
/// # Arguments
///
/// * `bi_planar_image` - Target Bi-Planar image
/// * `rgb` - The input BGR image data slice.
/// * `rgb_stride` - The stride (components per row) for the BGR image data.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGR data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn bgr_to_yuv_nv16(
    bi_planar_image: &mut YuvBiPlanarImageMut<u8>,
    bgr: &[u8],
    bgr_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) -> Result<(), YuvError> {
    rgbx_to_nv::<
        { YuvSourceChannels::Bgr as u8 },
        { YuvNVOrder::UV as u8 },
        { YuvChromaSubsampling::Yuv422 as u8 },
    >(bi_planar_image, bgr, bgr_stride, range, matrix)
}

/// Convert BGR image data to YUV NV61 bi-planar format.
///
/// This function performs BGR to YUV conversion and stores the result in YUV NV61 bi-planar format,
/// with plane for Y (luminance), and bi-plane VU (chrominance) components.
///
/// # Arguments
///
/// * `bi_planar_image` - Target Bi-Planar image
/// * `rgb` - The input BGR image data slice.
/// * `rgb_stride` - The stride (components per row) for the BGR image data.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGR data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn bgr_to_yuv_nv61(
    bi_planar_image: &mut YuvBiPlanarImageMut<u8>,
    bgr: &[u8],
    bgr_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) -> Result<(), YuvError> {
    rgbx_to_nv::<
        { YuvSourceChannels::Bgr as u8 },
        { YuvNVOrder::VU as u8 },
        { YuvChromaSubsampling::Yuv422 as u8 },
    >(bi_planar_image, bgr, bgr_stride, range, matrix)
}

/// Convert RGBA image data to YUV NV16 bi-planar format.
///
/// This function performs RGBA to YUV conversion and stores the result in YUV NV16 bi-planar format,
/// with plane for Y (luminance), and bi-plane UV (chrominance) components.
///
/// # Arguments
///
/// * `bi_planar_image` - Target Bi-Planar image
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
pub fn rgba_to_yuv_nv16(
    bi_planar_image: &mut YuvBiPlanarImageMut<u8>,
    rgba: &[u8],
    rgba_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) -> Result<(), YuvError> {
    rgbx_to_nv::<
        { YuvSourceChannels::Rgba as u8 },
        { YuvNVOrder::UV as u8 },
        { YuvChromaSubsampling::Yuv422 as u8 },
    >(bi_planar_image, rgba, rgba_stride, range, matrix)
}

/// Convert RGBA image data to YUV NV61 bi-planar format.
///
/// This function performs RGBA to YUV conversion and stores the result in YUV NV61 bi-planar format,
/// with plane for Y (luminance), and bi-plane VU (chrominance) components.
///
/// # Arguments
///
/// * `bi_planar_image` - Target Bi-Planar image
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
pub fn rgba_to_yuv_nv61(
    bi_planar_image: &mut YuvBiPlanarImageMut<u8>,
    rgba: &[u8],
    rgba_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) -> Result<(), YuvError> {
    rgbx_to_nv::<
        { YuvSourceChannels::Rgba as u8 },
        { YuvNVOrder::VU as u8 },
        { YuvChromaSubsampling::Yuv422 as u8 },
    >(bi_planar_image, rgba, rgba_stride, range, matrix)
}

/// Convert BGRA image data to YUV NV16 bi-planar format.
///
/// This function performs BGRA to YUV conversion and stores the result in YUV NV16 bi-planar format,
/// with plane for Y (luminance), and bi-plane UV (chrominance) components.
///
/// # Arguments
///
/// * `bi_planar_image` - Target Bi-Planar image
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
pub fn bgra_to_yuv_nv16(
    bi_planar_image: &mut YuvBiPlanarImageMut<u8>,
    bgra: &[u8],
    bgra_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) -> Result<(), YuvError> {
    rgbx_to_nv::<
        { YuvSourceChannels::Bgra as u8 },
        { YuvNVOrder::UV as u8 },
        { YuvChromaSubsampling::Yuv422 as u8 },
    >(bi_planar_image, bgra, bgra_stride, range, matrix)
}

/// Convert BGRA image data to YUV NV61 bi-planar format.
///
/// This function performs BGRA to YUV conversion and stores the result in YUV NV61 bi-planar format,
/// with plane for Y (luminance), and bi-plane VU (chrominance) components.
///
/// # Arguments
///
/// * `bi_planar_image` - Target Bi-Planar image
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
pub fn bgra_to_yuv_nv61(
    bi_planar_image: &mut YuvBiPlanarImageMut<u8>,
    bgra: &[u8],
    bgra_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) -> Result<(), YuvError> {
    rgbx_to_nv::<
        { YuvSourceChannels::Bgra as u8 },
        { YuvNVOrder::VU as u8 },
        { YuvChromaSubsampling::Yuv422 as u8 },
    >(bi_planar_image, bgra, bgra_stride, range, matrix)
}

/// Convert RGB image data to YUV NV12 bi-planar format.
///
/// This function performs RGB to YUV conversion and stores the result in YUV NV12 bi-planar format,
/// with plane for Y (luminance), and bi-plane UV (chrominance) components.
///
/// # Arguments
///
/// * `bi_planar_image` - Target Bi-Planar image
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
pub fn rgb_to_yuv_nv12(
    bi_planar_image: &mut YuvBiPlanarImageMut<u8>,
    rgb: &[u8],
    rgb_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) -> Result<(), YuvError> {
    rgbx_to_nv::<
        { YuvSourceChannels::Rgb as u8 },
        { YuvNVOrder::UV as u8 },
        { YuvChromaSubsampling::Yuv420 as u8 },
    >(bi_planar_image, rgb, rgb_stride, range, matrix)
}

/// Convert RGB image data to YUV NV21 bi-planar format.
///
/// This function performs RGB to YUV conversion and stores the result in YUV NV21 bi-planar format,
/// with plane for Y (luminance), and bi-plane VU (chrominance) components.
///
/// # Arguments
///
/// * `bi_planar_image` - Target Bi-Planar image
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
pub fn rgb_to_yuv_nv21(
    bi_planar_image: &mut YuvBiPlanarImageMut<u8>,
    rgb: &[u8],
    rgb_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) -> Result<(), YuvError> {
    rgbx_to_nv::<
        { YuvSourceChannels::Rgb as u8 },
        { YuvNVOrder::VU as u8 },
        { YuvChromaSubsampling::Yuv420 as u8 },
    >(bi_planar_image, rgb, rgb_stride, range, matrix)
}

/// Convert BGR image data to YUV NV12 bi-planar format.
///
/// This function performs BGR to YUV conversion and stores the result in YUV NV12 bi-planar format,
/// with plane for Y (luminance), and bi-plane UV (chrominance) components.
///
/// # Arguments
///
/// * `bi_planar_image` - Target Bi-Planar image
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
pub fn bgr_to_yuv_nv12(
    bi_planar_image: &mut YuvBiPlanarImageMut<u8>,
    bgr: &[u8],
    bgr_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) -> Result<(), YuvError> {
    rgbx_to_nv::<
        { YuvSourceChannels::Bgr as u8 },
        { YuvNVOrder::UV as u8 },
        { YuvChromaSubsampling::Yuv420 as u8 },
    >(bi_planar_image, bgr, bgr_stride, range, matrix)
}

/// Convert BGR image data to YUV NV21 bi-planar format.
///
/// This function performs BGR to YUV conversion and stores the result in YUV NV21 bi-planar format,
/// with plane for Y (luminance), and bi-plane VU (chrominance) components.
///
/// # Arguments
///
/// * `bi_planar_image` - Target Bi-Planar image
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
pub fn bgr_to_yuv_nv21(
    bi_planar_image: &mut YuvBiPlanarImageMut<u8>,
    bgr: &[u8],
    bgr_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) -> Result<(), YuvError> {
    rgbx_to_nv::<
        { YuvSourceChannels::Bgr as u8 },
        { YuvNVOrder::VU as u8 },
        { YuvChromaSubsampling::Yuv420 as u8 },
    >(bi_planar_image, bgr, bgr_stride, range, matrix)
}

/// Convert RGBA image data to YUV NV12 bi-planar format.
///
/// This function performs RGBA to YUV conversion and stores the result in YUV NV12 bi-planar format,
/// with plane for Y (luminance), and bi-plane UV (chrominance) components.
///
/// # Arguments
///
/// * `bi_planar_image` - Target Bi-Planar image
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
pub fn rgba_to_yuv_nv12(
    bi_planar_image: &mut YuvBiPlanarImageMut<u8>,
    rgba: &[u8],
    rgba_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) -> Result<(), YuvError> {
    rgbx_to_nv::<
        { YuvSourceChannels::Rgba as u8 },
        { YuvNVOrder::UV as u8 },
        { YuvChromaSubsampling::Yuv420 as u8 },
    >(bi_planar_image, rgba, rgba_stride, range, matrix)
}

/// Convert RGBA image data to YUV NV21 bi-planar format.
///
/// This function performs RGBA to YUV conversion and stores the result in YUV NV21 bi-planar format,
/// with plane for Y (luminance), and bi-plane VU (chrominance) components.
///
/// # Arguments
///
/// * `bi_planar_image` - Target Bi-Planar image
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
pub fn rgba_to_yuv_nv21(
    bi_planar_image: &mut YuvBiPlanarImageMut<u8>,
    rgba: &[u8],
    rgba_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) -> Result<(), YuvError> {
    rgbx_to_nv::<
        { YuvSourceChannels::Rgba as u8 },
        { YuvNVOrder::VU as u8 },
        { YuvChromaSubsampling::Yuv420 as u8 },
    >(bi_planar_image, rgba, rgba_stride, range, matrix)
}

/// Convert BGRA image data to YUV NV12 bi-planar format.
///
/// This function performs BGRA to YUV conversion and stores the result in YUV NV12 bi-planar format,
/// with plane for Y (luminance), and bi-plane UV (chrominance) components.
///
/// # Arguments
///
/// * `bi_planar_image` - Target Bi-Planar image
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
pub fn bgra_to_yuv_nv12(
    bi_planar_image: &mut YuvBiPlanarImageMut<u8>,
    bgra: &[u8],
    bgra_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) -> Result<(), YuvError> {
    rgbx_to_nv::<
        { YuvSourceChannels::Bgra as u8 },
        { YuvNVOrder::UV as u8 },
        { YuvChromaSubsampling::Yuv420 as u8 },
    >(bi_planar_image, bgra, bgra_stride, range, matrix)
}

/// Convert BGRA image data to YUV NV21 bi-planar format.
///
/// This function performs BGRA to YUV conversion and stores the result in YUV NV21 bi-planar format,
/// with plane for Y (luminance), and bi-plane VU (chrominance) components.
///
/// # Arguments
///
/// * `bi_planar_image` - Target Bi-Planar image
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
pub fn bgra_to_yuv_nv21(
    bi_planar_image: &mut YuvBiPlanarImageMut<u8>,
    bgra: &[u8],
    bgra_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) -> Result<(), YuvError> {
    rgbx_to_nv::<
        { YuvSourceChannels::Bgra as u8 },
        { YuvNVOrder::VU as u8 },
        { YuvChromaSubsampling::Yuv420 as u8 },
    >(bi_planar_image, bgra, bgra_stride, range, matrix)
}

/// Convert RGB image data to YUV NV24 bi-planar format.
///
/// This function performs RGB to YUV conversion and stores the result in YUV NV24 bi-planar format,
/// with plane for Y (luminance), and bi-plane UV (chrominance) components.
///
/// # Arguments
///
/// * `bi_planar_image` - Target Bi-Planar image
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
pub fn rgb_to_yuv_nv24(
    bi_planar_image: &mut YuvBiPlanarImageMut<u8>,
    rgb: &[u8],
    rgb_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) -> Result<(), YuvError> {
    rgbx_to_nv::<
        { YuvSourceChannels::Rgb as u8 },
        { YuvNVOrder::UV as u8 },
        { YuvChromaSubsampling::Yuv444 as u8 },
    >(bi_planar_image, rgb, rgb_stride, range, matrix)
}

/// Convert RGB image data to YUV NV42 bi-planar format.
///
/// This function performs RGB to YUV conversion and stores the result in YUV NV42 bi-planar format,
/// with plane for Y (luminance), and bi-plane VU (chrominance) components.
///
/// # Arguments
///
/// * `bi_planar_image` - Target Bi-Planar image
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
pub fn rgb_to_yuv_nv42(
    bi_planar_image: &mut YuvBiPlanarImageMut<u8>,
    rgb: &[u8],
    rgb_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) -> Result<(), YuvError> {
    rgbx_to_nv::<
        { YuvSourceChannels::Rgb as u8 },
        { YuvNVOrder::VU as u8 },
        { YuvChromaSubsampling::Yuv444 as u8 },
    >(bi_planar_image, rgb, rgb_stride, range, matrix)
}

/// Convert BGR image data to YUV NV24 bi-planar format.
///
/// This function performs BGR to YUV conversion and stores the result in YUV NV24 bi-planar format,
/// with plane for Y (luminance), and bi-plane UV (chrominance) components.
///
/// # Arguments
///
/// * `bi_planar_image` - Target Bi-Planar image
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
pub fn bgr_to_yuv_nv24(
    bi_planar_image: &mut YuvBiPlanarImageMut<u8>,
    bgr: &[u8],
    bgr_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) -> Result<(), YuvError> {
    rgbx_to_nv::<
        { YuvSourceChannels::Bgr as u8 },
        { YuvNVOrder::UV as u8 },
        { YuvChromaSubsampling::Yuv444 as u8 },
    >(bi_planar_image, bgr, bgr_stride, range, matrix)
}

/// Convert BGR image data to YUV NV42 bi-planar format.
///
/// This function performs BGR to YUV conversion and stores the result in YUV NV42 bi-planar format,
/// with plane for Y (luminance), and bi-plane VU (chrominance) components.
///
/// # Arguments
///
/// * `bi_planar_image` - Target Bi-Planar image
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
pub fn bgr_to_yuv_nv42(
    bi_planar_image: &mut YuvBiPlanarImageMut<u8>,
    bgr: &[u8],
    bgr_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) -> Result<(), YuvError> {
    rgbx_to_nv::<
        { YuvSourceChannels::Bgr as u8 },
        { YuvNVOrder::VU as u8 },
        { YuvChromaSubsampling::Yuv444 as u8 },
    >(bi_planar_image, bgr, bgr_stride, range, matrix)
}

/// Convert RGBA image data to YUV NV24 bi-planar format.
///
/// This function performs RGBA to YUV conversion and stores the result in YUV NV24 bi-planar format,
/// with plane for Y (luminance), and bi-plane UV (chrominance) components.
///
/// # Arguments
///
/// * `bi_planar_image` - Target Bi-Planar image
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
pub fn rgba_to_yuv_nv24(
    bi_planar_image: &mut YuvBiPlanarImageMut<u8>,
    rgba: &[u8],
    rgba_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) -> Result<(), YuvError> {
    rgbx_to_nv::<
        { YuvSourceChannels::Rgba as u8 },
        { YuvNVOrder::UV as u8 },
        { YuvChromaSubsampling::Yuv444 as u8 },
    >(bi_planar_image, rgba, rgba_stride, range, matrix)
}

/// Convert RGBA image data to YUV NV42 bi-planar format.
///
/// This function performs RGBA to YUV conversion and stores the result in YUV NV42 bi-planar format,
/// with plane for Y (luminance), and bi-plane VU (chrominance) components.
///
/// # Arguments
///
/// * `bi_planar_image` - Target Bi-Planar image
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
pub fn rgba_to_yuv_nv42(
    bi_planar_image: &mut YuvBiPlanarImageMut<u8>,
    rgba: &[u8],
    rgba_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) -> Result<(), YuvError> {
    rgbx_to_nv::<
        { YuvSourceChannels::Rgba as u8 },
        { YuvNVOrder::VU as u8 },
        { YuvChromaSubsampling::Yuv444 as u8 },
    >(bi_planar_image, rgba, rgba_stride, range, matrix)
}

/// Convert BGRA image data to YUV NV24 bi-planar format.
///
/// This function performs BGRA to YUV conversion and stores the result in YUV NV24 bi-planar format,
/// with plane for Y (luminance), and bi-plane UV (chrominance) components.
///
/// # Arguments
///
/// * `bi_planar_image` - Target Bi-Planar image
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
pub fn bgra_to_yuv_nv24(
    bi_planar_image: &mut YuvBiPlanarImageMut<u8>,
    bgra: &[u8],
    bgra_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) -> Result<(), YuvError> {
    rgbx_to_nv::<
        { YuvSourceChannels::Bgra as u8 },
        { YuvNVOrder::UV as u8 },
        { YuvChromaSubsampling::Yuv444 as u8 },
    >(bi_planar_image, bgra, bgra_stride, range, matrix)
}

/// Convert BGRA image data to YUV NV42 bi-planar format.
///
/// This function performs BGRA to YUV conversion and stores the result in YUV NV42 bi-planar format,
/// with plane for Y (luminance), and bi-plane VU (chrominance) components.
///
/// # Arguments
///
/// * `bi_planar_image` - Target Bi-Planar image
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
pub fn bgra_to_yuv_nv42(
    bi_planar_image: &mut YuvBiPlanarImageMut<u8>,
    bgra: &[u8],
    bgra_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) -> Result<(), YuvError> {
    rgbx_to_nv::<
        { YuvSourceChannels::Bgra as u8 },
        { YuvNVOrder::VU as u8 },
        { YuvChromaSubsampling::Yuv444 as u8 },
    >(bi_planar_image, bgra, bgra_stride, range, matrix)
}
