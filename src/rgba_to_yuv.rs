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
use crate::avx2::{avx2_rgba_to_yuv, avx2_rgba_to_yuv420};
#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    feature = "nightly_avx512"
))]
use crate::avx512bw::{avx512_rgba_to_yuv, avx512_rgba_to_yuv420};
#[allow(unused_imports)]
use crate::internals::*;
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use crate::neon::{
    neon_rgba_to_yuv, neon_rgba_to_yuv420, neon_rgba_to_yuv_rdm, neon_rgba_to_yuv_rdm420,
};
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use crate::sse::{sse_rgba_to_yuv_row, sse_rgba_to_yuv_row420};
use crate::yuv_error::check_rgba_destination;
#[allow(unused_imports)]
use crate::yuv_support::*;
use crate::{YuvError, YuvPlanarImageMut};
#[cfg(feature = "rayon")]
use rayon::iter::{IndexedParallelIterator, ParallelIterator};
#[cfg(feature = "rayon")]
use rayon::prelude::{ParallelSlice, ParallelSliceMut};

fn rgbx_to_yuv8<const ORIGIN_CHANNELS: u8, const SAMPLING: u8>(
    image: &mut YuvPlanarImageMut<u8>,
    rgba: &[u8],
    rgba_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) -> Result<(), YuvError> {
    let chroma_subsampling: YuvChromaSubsampling = SAMPLING.into();
    let src_chans: YuvSourceChannels = ORIGIN_CHANNELS.into();
    let channels = src_chans.get_channels_count();

    check_rgba_destination(rgba, rgba_stride, image.width, image.height, channels)?;
    image.check_constraints(chroma_subsampling)?;

    let chroma_range = get_yuv_range(8, range);
    let kr_kb = matrix.get_kr_kb();
    const PRECISION: i32 = 13;

    let transform = search_forward_transform(PRECISION, 8, range, matrix, chroma_range, kr_kb);

    const ROUNDING_CONST_BIAS: i32 = (1 << (PRECISION - 1)) - 1;
    let bias_y = chroma_range.bias_y as i32 * (1 << PRECISION) + ROUNDING_CONST_BIAS;
    let bias_uv = chroma_range.bias_uv as i32 * (1 << PRECISION) + ROUNDING_CONST_BIAS;

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    let use_sse = std::arch::is_x86_feature_detected!("sse4.1");
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    let use_avx = std::arch::is_x86_feature_detected!("avx2");
    #[cfg(all(
        any(target_arch = "x86", target_arch = "x86_64"),
        feature = "nightly_avx512"
    ))]
    let use_avx512 = std::arch::is_x86_feature_detected!("avx512bw");
    #[cfg(all(
        any(target_arch = "x86", target_arch = "x86_64"),
        feature = "nightly_avx512"
    ))]
    let use_vbmi = std::arch::is_x86_feature_detected!("avx512vbmi");
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    let is_rdm_available = std::arch::is_aarch64_feature_detected!("rdm");
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    let neon_wide_row_handler = if is_rdm_available {
        neon_rgba_to_yuv_rdm::<ORIGIN_CHANNELS, SAMPLING, PRECISION>
    } else {
        neon_rgba_to_yuv::<ORIGIN_CHANNELS, SAMPLING, PRECISION>
    };
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    let neon_double_wide_row_handler = if is_rdm_available {
        neon_rgba_to_yuv_rdm420::<ORIGIN_CHANNELS, PRECISION>
    } else {
        neon_rgba_to_yuv420::<ORIGIN_CHANNELS, PRECISION>
    };
    #[cfg(all(
        any(target_arch = "x86", target_arch = "x86_64"),
        feature = "nightly_avx512"
    ))]
    let avx512_row_dispatch = if use_vbmi {
        avx512_rgba_to_yuv::<ORIGIN_CHANNELS, SAMPLING, true>
    } else {
        avx512_rgba_to_yuv::<ORIGIN_CHANNELS, SAMPLING, false>
    };
    #[cfg(all(
        any(target_arch = "x86", target_arch = "x86_64"),
        feature = "nightly_avx512"
    ))]
    let avx512_double_wide_row_handler = if use_vbmi {
        avx512_rgba_to_yuv420::<ORIGIN_CHANNELS, true>
    } else {
        avx512_rgba_to_yuv420::<ORIGIN_CHANNELS, false>
    };

    #[allow(unused_variables)]
    let process_wide_row = |_y_plane: &mut [u8],
                            _u_plane: &mut [u8],
                            _v_plane: &mut [u8],
                            _rgba: &[u8],
                            _cx,
                            _ux,
                            _compute_uv_row| {
        let mut _offset = ProcessedOffset { ux: _cx, cx: _ux };
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            #[cfg(feature = "nightly_avx512")]
            {
                if use_avx512 {
                    let processed_offset = avx512_row_dispatch(
                        &transform,
                        &chroma_range,
                        _y_plane,
                        _u_plane,
                        _v_plane,
                        _rgba,
                        _offset.cx,
                        _offset.ux,
                        image.width as usize,
                    );
                    _offset = processed_offset;
                }
            }

            if use_avx {
                let processed_offset = avx2_rgba_to_yuv::<ORIGIN_CHANNELS, SAMPLING, PRECISION>(
                    &transform,
                    &chroma_range,
                    _y_plane,
                    _u_plane,
                    _v_plane,
                    _rgba,
                    _offset.cx,
                    _offset.ux,
                    image.width as usize,
                );
                _offset = processed_offset;
            }

            if use_sse {
                let processed_offset = sse_rgba_to_yuv_row::<ORIGIN_CHANNELS, SAMPLING, PRECISION>(
                    &transform,
                    &chroma_range,
                    _y_plane,
                    _u_plane,
                    _v_plane,
                    _rgba,
                    _offset.cx,
                    _offset.ux,
                    image.width as usize,
                );
                _offset = processed_offset;
            }
        }

        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        unsafe {
            let offset = neon_wide_row_handler(
                &transform,
                &chroma_range,
                _y_plane,
                _u_plane,
                _v_plane,
                _rgba,
                _offset.cx,
                _offset.ux,
                image.width as usize,
            );
            _offset = offset;
        }

        _offset
    };
    let process_doubled_wide_row = |_y_plane0: &mut [u8],
                                    _y_plane1: &mut [u8],
                                    _u_plane: &mut [u8],
                                    _v_plane: &mut [u8],
                                    _rgba0: &[u8],
                                    _rgba1: &[u8]| {
        let mut _offset = ProcessedOffset { ux: 0, cx: 0 };
        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        unsafe {
            let offset = neon_double_wide_row_handler(
                &transform,
                &chroma_range,
                _y_plane0,
                _y_plane1,
                _u_plane,
                _v_plane,
                _rgba0,
                _rgba1,
                _offset.cx,
                _offset.ux,
                image.width as usize,
            );
            _offset = offset;
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            #[cfg(feature = "nightly_avx512")]
            {
                if use_avx512 {
                    let processed_offset = avx512_double_wide_row_handler(
                        &transform,
                        &chroma_range,
                        _y_plane0,
                        _y_plane1,
                        _u_plane,
                        _v_plane,
                        _rgba0,
                        _rgba1,
                        _offset.cx,
                        _offset.ux,
                        image.width as usize,
                    );
                    _offset = processed_offset;
                }
            }
            if use_avx {
                let processed_offset = avx2_rgba_to_yuv420::<ORIGIN_CHANNELS, PRECISION>(
                    &transform,
                    &chroma_range,
                    _y_plane0,
                    _y_plane1,
                    _u_plane,
                    _v_plane,
                    _rgba0,
                    _rgba1,
                    _offset.cx,
                    _offset.ux,
                    image.width as usize,
                );
                _offset = processed_offset;
            }

            if use_sse {
                let processed_offset = sse_rgba_to_yuv_row420::<ORIGIN_CHANNELS, PRECISION>(
                    &transform,
                    &chroma_range,
                    _y_plane0,
                    _y_plane1,
                    _u_plane,
                    _v_plane,
                    _rgba0,
                    _rgba1,
                    _offset.cx,
                    _offset.ux,
                    image.width as usize,
                );
                _offset = processed_offset;
            }
        }
        _offset
    };

    let process_halved_chroma_row = |y_plane: &mut [u8],
                                     u_plane: &mut [u8],
                                     v_plane: &mut [u8],
                                     rgba: &[u8]| {
        let processed_offset = process_wide_row(y_plane, u_plane, v_plane, rgba, 0, 0, true);
        let cx = processed_offset.cx;

        for (((y_dst, u_dst), v_dst), rgba) in y_plane
            .chunks_exact_mut(2)
            .zip(u_plane.iter_mut())
            .zip(v_plane.iter_mut())
            .zip(rgba.chunks_exact(channels * 2))
            .skip(cx / 2)
        {
            let src0 = &rgba[0..channels];

            let r0 = src0[src_chans.get_r_channel_offset()] as i32;
            let g0 = src0[src_chans.get_g_channel_offset()] as i32;
            let b0 = src0[src_chans.get_b_channel_offset()] as i32;
            let y_0 =
                (r0 * transform.yr + g0 * transform.yg + b0 * transform.yb + bias_y) >> PRECISION;
            y_dst[0] = y_0 as u8;

            let src1 = &rgba[channels..channels * 2];

            let r1 = src1[src_chans.get_r_channel_offset()] as i32;
            let g1 = src1[src_chans.get_g_channel_offset()] as i32;
            let b1 = src1[src_chans.get_b_channel_offset()] as i32;
            let y_1 =
                (r1 * transform.yr + g1 * transform.yg + b1 * transform.yb + bias_y) >> PRECISION;
            y_dst[1] = y_1 as u8;

            let r = (r0 + r1 + 1) >> 1;
            let g = (g0 + g1 + 1) >> 1;
            let b = (b0 + b1 + 1) >> 1;

            let cb = (r * transform.cb_r + g * transform.cb_g + b * transform.cb_b + bias_uv)
                >> PRECISION;
            let cr = (r * transform.cr_r + g * transform.cr_g + b * transform.cr_b + bias_uv)
                >> PRECISION;
            *u_dst = cb as u8;
            *v_dst = cr as u8;
        }

        if image.width & 1 != 0 {
            let rgb_last = rgba.chunks_exact(channels * 2).remainder();
            let r0 = rgb_last[src_chans.get_r_channel_offset()] as i32;
            let g0 = rgb_last[src_chans.get_g_channel_offset()] as i32;
            let b0 = rgb_last[src_chans.get_b_channel_offset()] as i32;

            let y_last = y_plane.last_mut().unwrap();
            let u_last = u_plane.last_mut().unwrap();
            let v_last = v_plane.last_mut().unwrap();

            let y_0 =
                (r0 * transform.yr + g0 * transform.yg + b0 * transform.yb + bias_y) >> PRECISION;
            *y_last = y_0 as u8;

            let cb = (r0 * transform.cb_r + g0 * transform.cb_g + b0 * transform.cb_b + bias_uv)
                >> PRECISION;
            let cr = (r0 * transform.cr_r + g0 * transform.cr_g + b0 * transform.cr_b + bias_uv)
                >> PRECISION;
            *u_last = cb as u8;
            *v_last = cr as u8;
        }
    };

    let process_doubled_row = |y_plane0: &mut [u8],
                               y_plane1: &mut [u8],
                               u_plane: &mut [u8],
                               v_plane: &mut [u8],
                               rgba0: &[u8],
                               rgba1: &[u8]| {
        let processed_offset =
            process_doubled_wide_row(y_plane0, y_plane1, u_plane, v_plane, rgba0, rgba1);
        let cx = processed_offset.cx;

        for (((((y_dst0, y_dst1), u_dst), v_dst), rgba0), rgba1) in y_plane0
            .chunks_exact_mut(2)
            .zip(y_plane1.chunks_exact_mut(2))
            .zip(u_plane.iter_mut())
            .zip(v_plane.iter_mut())
            .zip(rgba0.chunks_exact(channels * 2))
            .zip(rgba1.chunks_exact(channels * 2))
            .skip(cx / 2)
        {
            let src00 = &rgba0[0..channels];

            let r00 = src00[src_chans.get_r_channel_offset()] as i32;
            let g00 = src00[src_chans.get_g_channel_offset()] as i32;
            let b00 = src00[src_chans.get_b_channel_offset()] as i32;
            let y_00 = (r00 * transform.yr + g00 * transform.yg + b00 * transform.yb + bias_y)
                >> PRECISION;
            y_dst0[0] = y_00 as u8;

            let src1 = &rgba0[channels..channels * 2];

            let r01 = src1[src_chans.get_r_channel_offset()] as i32;
            let g01 = src1[src_chans.get_g_channel_offset()] as i32;
            let b01 = src1[src_chans.get_b_channel_offset()] as i32;
            let y_01 = (r01 * transform.yr + g01 * transform.yg + b01 * transform.yb + bias_y)
                >> PRECISION;
            y_dst0[1] = y_01 as u8;

            let src10 = &rgba1[0..channels];

            let r10 = src10[src_chans.get_r_channel_offset()] as i32;
            let g10 = src10[src_chans.get_g_channel_offset()] as i32;
            let b10 = src10[src_chans.get_b_channel_offset()] as i32;
            let y_10 = (r10 * transform.yr + g10 * transform.yg + b10 * transform.yb + bias_y)
                >> PRECISION;
            y_dst1[0] = y_10 as u8;

            let src11 = &rgba1[channels..channels * 2];

            let r11 = src11[src_chans.get_r_channel_offset()] as i32;
            let g11 = src11[src_chans.get_g_channel_offset()] as i32;
            let b11 = src11[src_chans.get_b_channel_offset()] as i32;
            let y_11 = (r11 * transform.yr + g11 * transform.yg + b11 * transform.yb + bias_y)
                >> PRECISION;
            y_dst1[1] = y_11 as u8;

            let ruv = (r00 + r01 + r10 + r11 + 2) >> 2;
            let guv = (g00 + g01 + g10 + g11 + 2) >> 2;
            let buv = (b00 + b01 + b10 + b11 + 2) >> 2;

            let cb = (ruv * transform.cb_r + guv * transform.cb_g + buv * transform.cb_b + bias_uv)
                >> PRECISION;
            let cr = (ruv * transform.cr_r + guv * transform.cr_g + buv * transform.cr_b + bias_uv)
                >> PRECISION;
            *u_dst = cb as u8;
            *v_dst = cr as u8;
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

            let y_0 =
                (r0 * transform.yr + g0 * transform.yg + b0 * transform.yb + bias_y) >> PRECISION;
            *y0_last = y_0 as u8;

            let y_1 =
                (r1 * transform.yr + g1 * transform.yg + b1 * transform.yb + bias_y) >> PRECISION;
            *y1_last = y_1 as u8;

            let r0 = (r0 + r1) >> 1;
            let g0 = (g0 + g1) >> 1;
            let b0 = (b0 + b1) >> 1;

            let cb = (r0 * transform.cb_r + g0 * transform.cb_g + b0 * transform.cb_b + bias_uv)
                >> PRECISION;
            let cr = (r0 * transform.cr_r + g0 * transform.cr_g + b0 * transform.cr_b + bias_uv)
                >> PRECISION;
            *u_last = cb as u8;
            *v_last = cr as u8;
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
            let processed_offset = process_wide_row(y_dst, u_plane, v_plane, rgba, 0, 0, true);
            let cx = processed_offset.cx;

            for (((y_dst, u_dst), v_dst), rgba) in y_dst
                .iter_mut()
                .zip(u_plane.iter_mut())
                .zip(v_plane.iter_mut())
                .zip(rgba.chunks_exact(channels))
                .skip(cx)
            {
                let r0 = rgba[src_chans.get_r_channel_offset()] as i32;
                let g0 = rgba[src_chans.get_g_channel_offset()] as i32;
                let b0 = rgba[src_chans.get_b_channel_offset()] as i32;
                let y_0 = (r0 * transform.yr + g0 * transform.yg + b0 * transform.yb + bias_y)
                    >> PRECISION;
                *y_dst = y_0 as u8;

                let cb =
                    (r0 * transform.cb_r + g0 * transform.cb_g + b0 * transform.cb_b + bias_uv)
                        >> PRECISION;
                let cr =
                    (r0 * transform.cr_r + g0 * transform.cr_g + b0 * transform.cr_b + bias_uv)
                        >> PRECISION;
                *u_dst = cb as u8;
                *v_dst = cr as u8;
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

/// Convert RGB image data to YUV 422 planar format.
///
/// This function performs RGB to YUV conversion and stores the result in YUV422 planar format,
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
pub fn rgb_to_yuv422(
    planar_image: &mut YuvPlanarImageMut<u8>,
    rgb: &[u8],
    rgb_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) -> Result<(), YuvError> {
    rgbx_to_yuv8::<{ YuvSourceChannels::Rgb as u8 }, { YuvChromaSubsampling::Yuv422 as u8 }>(
        planar_image,
        rgb,
        rgb_stride,
        range,
        matrix,
    )
}

/// Convert BGR image data to YUV 422 planar format.
///
/// This function performs BGR to YUV conversion and stores the result in YUV422 planar format,
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
pub fn bgr_to_yuv422(
    planar_image: &mut YuvPlanarImageMut<u8>,
    bgr: &[u8],
    bgr_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) -> Result<(), YuvError> {
    rgbx_to_yuv8::<{ YuvSourceChannels::Bgr as u8 }, { YuvChromaSubsampling::Yuv422 as u8 }>(
        planar_image,
        bgr,
        bgr_stride,
        range,
        matrix,
    )
}

/// Convert RGBA image data to YUV 422 planar format.
///
/// This function performs RGBA to YUV conversion and stores the result in YUV422 planar format,
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
pub fn rgba_to_yuv422(
    planar_image: &mut YuvPlanarImageMut<u8>,
    rgba: &[u8],
    rgba_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) -> Result<(), YuvError> {
    rgbx_to_yuv8::<{ YuvSourceChannels::Rgba as u8 }, { YuvChromaSubsampling::Yuv422 as u8 }>(
        planar_image,
        rgba,
        rgba_stride,
        range,
        matrix,
    )
}

/// Convert BGRA image data to YUV 422 planar format.
///
/// This function performs BGRA to YUV conversion and stores the result in YUV422 planar format,
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
pub fn bgra_to_yuv422(
    planar_image: &mut YuvPlanarImageMut<u8>,
    bgra: &[u8],
    bgra_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) -> Result<(), YuvError> {
    rgbx_to_yuv8::<{ YuvSourceChannels::Bgra as u8 }, { YuvChromaSubsampling::Yuv422 as u8 }>(
        planar_image,
        bgra,
        bgra_stride,
        range,
        matrix,
    )
}

/// Convert RGB image data to YUV 420 planar format.
///
/// This function performs RGB to YUV conversion and stores the result in YUV420 planar format,
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
pub fn rgb_to_yuv420(
    planar_image: &mut YuvPlanarImageMut<u8>,
    rgb: &[u8],
    rgb_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) -> Result<(), YuvError> {
    rgbx_to_yuv8::<{ YuvSourceChannels::Rgb as u8 }, { YuvChromaSubsampling::Yuv420 as u8 }>(
        planar_image,
        rgb,
        rgb_stride,
        range,
        matrix,
    )
}

/// Convert BGR image data to YUV 420 planar format.
///
/// This function performs BGR to YUV conversion and stores the result in YUV420 planar format,
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
pub fn bgr_to_yuv420(
    planar_image: &mut YuvPlanarImageMut<u8>,
    bgr: &[u8],
    bgr_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) -> Result<(), YuvError> {
    rgbx_to_yuv8::<{ YuvSourceChannels::Bgr as u8 }, { YuvChromaSubsampling::Yuv420 as u8 }>(
        planar_image,
        bgr,
        bgr_stride,
        range,
        matrix,
    )
}

/// Convert RGBA image data to YUV 420 planar format.
///
/// This function performs RGBA to YUV conversion and stores the result in YUV420 planar format,
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
pub fn rgba_to_yuv420(
    planar_image: &mut YuvPlanarImageMut<u8>,
    rgba: &[u8],
    rgba_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) -> Result<(), YuvError> {
    rgbx_to_yuv8::<{ YuvSourceChannels::Rgba as u8 }, { YuvChromaSubsampling::Yuv420 as u8 }>(
        planar_image,
        rgba,
        rgba_stride,
        range,
        matrix,
    )
}

/// Convert BGRA image data to YUV 420 planar format.
///
/// This function performs BGRA to YUV conversion and stores the result in YUV420 planar format,
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
pub fn bgra_to_yuv420(
    planar_image: &mut YuvPlanarImageMut<u8>,
    bgra: &[u8],
    bgra_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) -> Result<(), YuvError> {
    rgbx_to_yuv8::<{ YuvSourceChannels::Bgra as u8 }, { YuvChromaSubsampling::Yuv420 as u8 }>(
        planar_image,
        bgra,
        bgra_stride,
        range,
        matrix,
    )
}

/// Convert RGB image data to YUV 444 planar format.
///
/// This function performs RGB to YUV conversion and stores the result in YUV444 planar format,
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
pub fn rgb_to_yuv444(
    planar_image: &mut YuvPlanarImageMut<u8>,
    rgb: &[u8],
    rgb_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) -> Result<(), YuvError> {
    rgbx_to_yuv8::<{ YuvSourceChannels::Rgb as u8 }, { YuvChromaSubsampling::Yuv444 as u8 }>(
        planar_image,
        rgb,
        rgb_stride,
        range,
        matrix,
    )
}

/// Convert BGR image data to YUV 444 planar format.
///
/// This function performs BGR to YUV conversion and stores the result in YUV444 planar format,
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
pub fn bgr_to_yuv444(
    planar_image: &mut YuvPlanarImageMut<u8>,
    bgr: &[u8],
    bgr_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) -> Result<(), YuvError> {
    rgbx_to_yuv8::<{ YuvSourceChannels::Bgr as u8 }, { YuvChromaSubsampling::Yuv444 as u8 }>(
        planar_image,
        bgr,
        bgr_stride,
        range,
        matrix,
    )
}

/// Convert RGBA image data to YUV 444 planar format.
///
/// This function performs RGBA to YUV conversion and stores the result in YUV444 planar format,
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
pub fn rgba_to_yuv444(
    planar_image: &mut YuvPlanarImageMut<u8>,
    rgba: &[u8],
    rgba_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) -> Result<(), YuvError> {
    rgbx_to_yuv8::<{ YuvSourceChannels::Rgba as u8 }, { YuvChromaSubsampling::Yuv444 as u8 }>(
        planar_image,
        rgba,
        rgba_stride,
        range,
        matrix,
    )
}

/// Convert BGRA image data to YUV 444 planar format.
///
/// This function performs BGRA to YUV conversion and stores the result in YUV444 planar format,
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
pub fn bgra_to_yuv444(
    planar_image: &mut YuvPlanarImageMut<u8>,
    bgra: &[u8],
    bgra_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) -> Result<(), YuvError> {
    rgbx_to_yuv8::<{ YuvSourceChannels::Bgra as u8 }, { YuvChromaSubsampling::Yuv444 as u8 }>(
        planar_image,
        bgra,
        bgra_stride,
        range,
        matrix,
    )
}
