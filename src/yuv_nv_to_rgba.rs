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
use crate::avx2::{avx2_yuv_nv_to_rgba_row, avx2_yuv_nv_to_rgba_row420};
#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    feature = "nightly_avx512"
))]
use crate::avx512bw::{avx512_yuv_nv_to_rgba, avx512_yuv_nv_to_rgba420};
use crate::built_coefficients::get_built_inverse_transform;
#[allow(unused_imports)]
use crate::internals::*;
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use crate::neon::{
    neon_yuv_nv_to_rgba_row, neon_yuv_nv_to_rgba_row420, neon_yuv_nv_to_rgba_row_rdm,
    neon_yuv_nv_to_rgba_row_rdm420,
};
use crate::numerics::qrshr;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use crate::sse::{sse_yuv_nv_to_rgba, sse_yuv_nv_to_rgba420};
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
use crate::wasm32::{wasm_yuv_nv_to_rgba_row, wasm_yuv_nv_to_rgba_row420};
use crate::yuv_error::check_rgba_destination;
use crate::yuv_support::*;
use crate::{YuvBiPlanarImage, YuvError};
#[cfg(feature = "rayon")]
use rayon::iter::{IndexedParallelIterator, ParallelIterator};
#[cfg(feature = "rayon")]
use rayon::prelude::{ParallelSlice, ParallelSliceMut};

fn yuv_nv12_to_rgbx<
    const UV_ORDER: u8,
    const DESTINATION_CHANNELS: u8,
    const YUV_CHROMA_SAMPLING: u8,
>(
    image: &YuvBiPlanarImage<u8>,
    bgra: &mut [u8],
    bgra_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) -> Result<(), YuvError> {
    let order: YuvNVOrder = UV_ORDER.into();
    let dst_chans: YuvSourceChannels = DESTINATION_CHANNELS.into();
    let chroma_subsampling: YuvChromaSubsampling = YUV_CHROMA_SAMPLING.into();

    image.check_constraints(chroma_subsampling)?;
    check_rgba_destination(
        bgra,
        bgra_stride,
        image.width,
        image.height,
        dst_chans.get_channels_count(),
    )?;

    let chroma_range = get_yuv_range(8, range);
    let channels = dst_chans.get_channels_count();
    let kr_kb = matrix.get_kr_kb();
    const PRECISION: i32 = 13;

    let inverse_transform =
        if let Some(stored) = get_built_inverse_transform(PRECISION as u32, 8, range, matrix) {
            stored
        } else {
            let transform = get_inverse_transform(
                255,
                chroma_range.range_y,
                chroma_range.range_uv,
                kr_kb.kr,
                kr_kb.kb,
            );
            transform.to_integers(PRECISION as u32)
        };
    let cr_coef = inverse_transform.cr_coef;
    let cb_coef = inverse_transform.cb_coef;
    let y_coef = inverse_transform.y_coef;
    let g_coef_1 = inverse_transform.g_coeff_1;
    let g_coef_2 = inverse_transform.g_coeff_2;

    let bias_y = chroma_range.bias_y as i32;
    let bias_uv = chroma_range.bias_uv as i32;

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    let _use_avx2 = std::arch::is_x86_feature_detected!("avx2");
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    let use_sse = std::arch::is_x86_feature_detected!("sse4.1");
    #[cfg(all(
        any(target_arch = "x86", target_arch = "x86_64"),
        feature = "nightly_avx512"
    ))]
    let _use_avx512 = std::arch::is_x86_feature_detected!("avx512bw");
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    let is_rdm_available = std::arch::is_aarch64_feature_detected!("rdm");
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    let neon_wide_row = if is_rdm_available {
        neon_yuv_nv_to_rgba_row_rdm::<UV_ORDER, DESTINATION_CHANNELS, YUV_CHROMA_SAMPLING>
    } else {
        neon_yuv_nv_to_rgba_row::<PRECISION, UV_ORDER, DESTINATION_CHANNELS, YUV_CHROMA_SAMPLING>
    };
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    let neon_double_row = if is_rdm_available {
        neon_yuv_nv_to_rgba_row_rdm420::<UV_ORDER, DESTINATION_CHANNELS>
    } else {
        neon_yuv_nv_to_rgba_row420::<PRECISION, UV_ORDER, DESTINATION_CHANNELS>
    };
    #[cfg(all(
        any(target_arch = "x86", target_arch = "x86_64"),
        feature = "nightly_avx512"
    ))]
    let dispatch_wide_row_avx512 =
        avx512_yuv_nv_to_rgba::<UV_ORDER, DESTINATION_CHANNELS, YUV_CHROMA_SAMPLING, false>;

    let width = image.width;

    let process_wide_row = |_bgra: &mut [u8], _y_plane: &[u8], _uv_plane: &[u8]| {
        let mut _offset = ProcessedOffset { cx: 0, ux: 0 };
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            #[cfg(feature = "nightly_avx512")]
            if _use_avx512 {
                let processed = dispatch_wide_row_avx512(
                    &chroma_range,
                    &inverse_transform,
                    _y_plane,
                    _uv_plane,
                    _bgra,
                    _offset.cx,
                    _offset.ux,
                    width as usize,
                );
                _offset = processed;
            }

            if _use_avx2 {
                let processed =
                    avx2_yuv_nv_to_rgba_row::<UV_ORDER, DESTINATION_CHANNELS, YUV_CHROMA_SAMPLING>(
                        &chroma_range,
                        &inverse_transform,
                        _y_plane,
                        _uv_plane,
                        _bgra,
                        _offset.cx,
                        _offset.ux,
                        width as usize,
                    );
                _offset = processed;
            }

            if use_sse {
                let processed =
                    sse_yuv_nv_to_rgba::<UV_ORDER, DESTINATION_CHANNELS, YUV_CHROMA_SAMPLING>(
                        &chroma_range,
                        &inverse_transform,
                        _y_plane,
                        _uv_plane,
                        _bgra,
                        _offset.cx,
                        _offset.ux,
                        width as usize,
                    );
                _offset = processed;
            }
        }

        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        {
            unsafe {
                let processed = neon_wide_row(
                    &chroma_range,
                    &inverse_transform,
                    _y_plane,
                    _uv_plane,
                    _bgra,
                    _offset.cx,
                    _offset.ux,
                    width as usize,
                );
                _offset = processed;
            }
        }

        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        {
            unsafe {
                let processed =
                    wasm_yuv_nv_to_rgba_row::<UV_ORDER, DESTINATION_CHANNELS, YUV_CHROMA_SAMPLING>(
                        &chroma_range,
                        &inverse_transform,
                        _y_plane,
                        _uv_plane,
                        _bgra,
                        _offset.cx,
                        _offset.ux,
                        width as usize,
                    );
                _offset = processed;
            }
        }

        _offset
    };

    let process_double_wide_row = |_bgra0: &mut [u8],
                                   _bgra1: &mut [u8],
                                   _y_plane0: &[u8],
                                   _y_plane1: &[u8],
                                   _uv_plane: &[u8]| {
        let mut _offset = ProcessedOffset { cx: 0, ux: 0 };
        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        {
            unsafe {
                let processed = neon_double_row(
                    &chroma_range,
                    &inverse_transform,
                    _y_plane0,
                    _y_plane1,
                    _uv_plane,
                    _bgra0,
                    _bgra1,
                    _offset.cx,
                    _offset.ux,
                    width as usize,
                );
                _offset = processed;
            }
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            #[cfg(feature = "nightly_avx512")]
            if _use_avx512 {
                let processed = avx512_yuv_nv_to_rgba420::<UV_ORDER, DESTINATION_CHANNELS, false>(
                    &chroma_range,
                    &inverse_transform,
                    _y_plane0,
                    _y_plane1,
                    _uv_plane,
                    _bgra0,
                    _bgra1,
                    _offset.cx,
                    _offset.ux,
                    width as usize,
                );
                _offset = processed;
            }

            if _use_avx2 {
                let processed = avx2_yuv_nv_to_rgba_row420::<UV_ORDER, DESTINATION_CHANNELS>(
                    &chroma_range,
                    &inverse_transform,
                    _y_plane0,
                    _y_plane1,
                    _uv_plane,
                    _bgra0,
                    _bgra1,
                    _offset.cx,
                    _offset.ux,
                    width as usize,
                );
                _offset = processed;
            }
            if use_sse {
                let processed = sse_yuv_nv_to_rgba420::<UV_ORDER, DESTINATION_CHANNELS>(
                    &chroma_range,
                    &inverse_transform,
                    _y_plane0,
                    _y_plane1,
                    _uv_plane,
                    _bgra0,
                    _bgra1,
                    _offset.cx,
                    _offset.ux,
                    width as usize,
                );
                _offset = processed;
            }
        }
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        {
            unsafe {
                let processed = wasm_yuv_nv_to_rgba_row420::<
                    UV_ORDER,
                    DESTINATION_CHANNELS,
                    YUV_CHROMA_SAMPLING,
                >(
                    &chroma_range,
                    &inverse_transform,
                    _y_plane0,
                    _y_plane1,
                    _uv_plane,
                    _bgra0,
                    _bgra1,
                    _offset.cx,
                    _offset.ux,
                    width as usize,
                );
                _offset = processed;
            }
        }
        _offset
    };

    let process_double_chroma_row =
        |y_src0: &[u8], y_src1: &[u8], uv_src: &[u8], rgba0: &mut [u8], rgba1: &mut [u8]| {
            let processed = process_double_wide_row(rgba0, rgba1, y_src0, y_src1, uv_src);

            for ((((rgba0, rgba1), y_src0), y_src1), uv_src) in rgba0
                .chunks_exact_mut(channels * 2)
                .zip(rgba1.chunks_exact_mut(channels * 2))
                .zip(y_src0.chunks_exact(2))
                .zip(y_src1.chunks_exact(2))
                .zip(uv_src.chunks_exact(2))
                .skip(processed.cx / 2)
            {
                let y_vl00 = y_src0[0] as i32;
                let cb_value = (uv_src[order.get_u_position()] as i32) - bias_uv;
                let cr_value = (uv_src[order.get_v_position()] as i32) - bias_uv;

                let y_value00: i32 = (y_vl00 - bias_y) * y_coef;

                let g_built_coeff = -g_coef_1 * cr_value - g_coef_2 * cb_value;

                let r00 = qrshr::<PRECISION, 8>(y_value00 + cr_coef * cr_value);
                let b00 = qrshr::<PRECISION, 8>(y_value00 + cb_coef * cb_value);
                let g00 = qrshr::<PRECISION, 8>(y_value00 + g_built_coeff);

                let rgba00 = &mut rgba0[0..channels];

                rgba00[dst_chans.get_b_channel_offset()] = b00 as u8;
                rgba00[dst_chans.get_g_channel_offset()] = g00 as u8;
                rgba00[dst_chans.get_r_channel_offset()] = r00 as u8;

                if dst_chans.has_alpha() {
                    rgba00[dst_chans.get_a_channel_offset()] = 255;
                }

                let y_vl01 = y_src0[1] as i32;

                let y_value01: i32 = (y_vl01 - bias_y) * y_coef;

                let r01 = qrshr::<PRECISION, 8>(y_value01 + cr_coef * cr_value);
                let b01 = qrshr::<PRECISION, 8>(y_value01 + cb_coef * cb_value);
                let g01 = qrshr::<PRECISION, 8>(y_value01 + g_built_coeff);

                let rgba01 = &mut rgba0[channels..channels * 2];

                rgba01[dst_chans.get_b_channel_offset()] = b01 as u8;
                rgba01[dst_chans.get_g_channel_offset()] = g01 as u8;
                rgba01[dst_chans.get_r_channel_offset()] = r01 as u8;

                if dst_chans.has_alpha() {
                    rgba01[dst_chans.get_a_channel_offset()] = 255;
                }

                let y_vl10 = y_src1[0] as i32;

                let y_value00: i32 = (y_vl10 - bias_y) * y_coef;

                let r10 = qrshr::<PRECISION, 8>(y_value00 + cr_coef * cr_value);
                let b10 = qrshr::<PRECISION, 8>(y_value00 + cb_coef * cb_value);
                let g10 = qrshr::<PRECISION, 8>(y_value00 + g_built_coeff);

                let rgba10 = &mut rgba1[0..channels];

                rgba10[dst_chans.get_b_channel_offset()] = b10 as u8;
                rgba10[dst_chans.get_g_channel_offset()] = g10 as u8;
                rgba10[dst_chans.get_r_channel_offset()] = r10 as u8;

                if dst_chans.has_alpha() {
                    rgba10[dst_chans.get_a_channel_offset()] = 255;
                }

                let y_vl11 = y_src1[1] as i32;

                let y_value11: i32 = (y_vl11 - bias_y) * y_coef;

                let r11 = qrshr::<PRECISION, 8>(y_value11 + cr_coef * cr_value);
                let b11 = qrshr::<PRECISION, 8>(y_value11 + cb_coef * cb_value);
                let g11 = qrshr::<PRECISION, 8>(y_value11 + g_built_coeff);

                let rgba11 = &mut rgba1[channels..channels * 2];

                rgba11[dst_chans.get_b_channel_offset()] = b11 as u8;
                rgba11[dst_chans.get_g_channel_offset()] = g11 as u8;
                rgba11[dst_chans.get_r_channel_offset()] = r11 as u8;

                if dst_chans.has_alpha() {
                    rgba11[dst_chans.get_a_channel_offset()] = 255;
                }
            }

            if width & 1 != 0 {
                let rgba0 = rgba0.chunks_exact_mut(channels * 2).into_remainder();
                let rgba1 = rgba1.chunks_exact_mut(channels * 2).into_remainder();
                let rgba0 = &mut rgba0[0..channels];
                let rgba1 = &mut rgba1[0..channels];
                let uv_src = uv_src.chunks_exact(2).last().unwrap();
                let y_src0 = y_src0.chunks_exact(2).remainder();

                let y_vl0 = y_src0[0] as i32;
                let y_value0: i32 = (y_vl0 - bias_y) * y_coef;
                let cb_value = (uv_src[order.get_u_position()] as i32) - bias_uv;
                let cr_value = (uv_src[order.get_v_position()] as i32) - bias_uv;

                let g_built_coeff = -g_coef_1 * cr_value - g_coef_2 * cb_value;

                let r0 = qrshr::<PRECISION, 8>(y_value0 + cr_coef * cr_value);
                let b0 = qrshr::<PRECISION, 8>(y_value0 + cb_coef * cb_value);
                let g0 = qrshr::<PRECISION, 8>(y_value0 + g_built_coeff);

                rgba0[dst_chans.get_b_channel_offset()] = b0 as u8;
                rgba0[dst_chans.get_g_channel_offset()] = g0 as u8;
                rgba0[dst_chans.get_r_channel_offset()] = r0 as u8;

                if dst_chans.has_alpha() {
                    rgba0[dst_chans.get_a_channel_offset()] = 255;
                }

                let y_vl1 = y_src1[0] as i32;
                let y_value1: i32 = (y_vl1 - bias_y) * y_coef;

                let r1 = qrshr::<PRECISION, 8>(y_value1 + cr_coef * cr_value);
                let b1 = qrshr::<PRECISION, 8>(y_value1 + cb_coef * cb_value);
                let g1 = qrshr::<PRECISION, 8>(y_value1 + g_built_coeff);

                rgba1[dst_chans.get_b_channel_offset()] = b1 as u8;
                rgba1[dst_chans.get_g_channel_offset()] = g1 as u8;
                rgba1[dst_chans.get_r_channel_offset()] = r1 as u8;

                if dst_chans.has_alpha() {
                    rgba1[dst_chans.get_a_channel_offset()] = 255;
                }
            }
        };

    let process_halved_chroma_row = |y_src: &[u8], uv_src: &[u8], rgba: &mut [u8]| {
        let processed = process_wide_row(rgba, y_src, uv_src);

        for ((rgba, y_src), uv_src) in rgba
            .chunks_exact_mut(channels * 2)
            .zip(y_src.chunks_exact(2))
            .zip(uv_src.chunks_exact(2))
            .skip(processed.cx / 2)
        {
            let y_vl0 = y_src[0] as i32;
            let cb_value = (uv_src[order.get_u_position()] as i32) - bias_uv;
            let cr_value = (uv_src[order.get_v_position()] as i32) - bias_uv;

            let y_value0: i32 = (y_vl0 - bias_y) * y_coef;

            let r0 = qrshr::<PRECISION, 8>(y_value0 + cr_coef * cr_value);
            let b0 = qrshr::<PRECISION, 8>(y_value0 + cb_coef * cb_value);
            let g0 = qrshr::<PRECISION, 8>(y_value0 - g_coef_1 * cr_value - g_coef_2 * cb_value);

            rgba[dst_chans.get_b_channel_offset()] = b0 as u8;
            rgba[dst_chans.get_g_channel_offset()] = g0 as u8;
            rgba[dst_chans.get_r_channel_offset()] = r0 as u8;

            if dst_chans.has_alpha() {
                rgba[dst_chans.get_a_channel_offset()] = 255;
            }

            let y_vl1 = y_src[1] as i32;

            let y_value1: i32 = (y_vl1 - bias_y) * y_coef;

            let r1 = qrshr::<PRECISION, 8>(y_value1 + cr_coef * cr_value);
            let b1 = qrshr::<PRECISION, 8>(y_value1 + cb_coef * cb_value);
            let g1 = qrshr::<PRECISION, 8>(y_value1 - g_coef_1 * cr_value - g_coef_2 * cb_value);

            let rgba0 = &mut rgba[channels..channels * 2];

            rgba0[dst_chans.get_b_channel_offset()] = b1 as u8;
            rgba0[dst_chans.get_g_channel_offset()] = g1 as u8;
            rgba0[dst_chans.get_r_channel_offset()] = r1 as u8;

            if dst_chans.has_alpha() {
                rgba0[dst_chans.get_a_channel_offset()] = 255;
            }
        }

        if width & 1 != 0 {
            let rgba = rgba.chunks_exact_mut(channels * 2).into_remainder();
            let rgba = &mut rgba[0..channels];
            let uv_src = uv_src.chunks_exact(2).last().unwrap();
            let y_src = y_src.chunks_exact(2).remainder();

            let y_vl0 = y_src[0] as i32;
            let y_value0: i32 = (y_vl0 - bias_y) * y_coef;
            let cb_value = (uv_src[order.get_u_position()] as i32) - bias_uv;
            let cr_value = (uv_src[order.get_v_position()] as i32) - bias_uv;

            let r0 = qrshr::<PRECISION, 8>(y_value0 + cr_coef * cr_value);
            let b0 = qrshr::<PRECISION, 8>(y_value0 + cb_coef * cb_value);
            let g0 = qrshr::<PRECISION, 8>(y_value0 - g_coef_1 * cr_value - g_coef_2 * cb_value);

            rgba[dst_chans.get_b_channel_offset()] = b0 as u8;
            rgba[dst_chans.get_g_channel_offset()] = g0 as u8;
            rgba[dst_chans.get_r_channel_offset()] = r0 as u8;

            if dst_chans.has_alpha() {
                rgba[dst_chans.get_a_channel_offset()] = 255;
            }
        }
    };

    let y_stride = image.y_stride;
    let uv_stride = image.uv_stride;
    let y_plane = image.y_plane;
    let uv_plane = image.uv_plane;

    if chroma_subsampling == YuvChromaSubsampling::Yuv444 {
        let iter;
        #[cfg(feature = "rayon")]
        {
            iter = y_plane
                .par_chunks_exact(y_stride as usize)
                .zip(uv_plane.par_chunks_exact(uv_stride as usize))
                .zip(bgra.par_chunks_exact_mut(bgra_stride as usize));
        }
        #[cfg(not(feature = "rayon"))]
        {
            iter = y_plane
                .chunks_exact(y_stride as usize)
                .zip(uv_plane.chunks_exact(uv_stride as usize))
                .zip(bgra.chunks_exact_mut(bgra_stride as usize));
        }
        iter.for_each(|((y_src, uv_src), rgba)| {
            let y_src = &y_src[0..image.width as usize];
            let processed = process_wide_row(rgba, y_src, uv_src);

            for ((rgba, &y_src), uv_src) in rgba
                .chunks_exact_mut(channels)
                .zip(y_src.iter())
                .zip(uv_src.chunks_exact(2))
                .skip(processed.cx)
            {
                let y_vl = y_src as i32;
                let mut cb_value = uv_src[order.get_u_position()] as i32;
                let mut cr_value = uv_src[order.get_v_position()] as i32;

                let y_value: i32 = (y_vl - bias_y) * y_coef;

                cb_value -= bias_uv;
                cr_value -= bias_uv;

                let r = qrshr::<PRECISION, 8>(y_value + cr_coef * cr_value);
                let b = qrshr::<PRECISION, 8>(y_value + cb_coef * cb_value);
                let g = qrshr::<PRECISION, 8>(y_value - g_coef_1 * cr_value - g_coef_2 * cb_value);

                rgba[dst_chans.get_b_channel_offset()] = b as u8;
                rgba[dst_chans.get_g_channel_offset()] = g as u8;
                rgba[dst_chans.get_r_channel_offset()] = r as u8;

                if dst_chans.has_alpha() {
                    rgba[dst_chans.get_a_channel_offset()] = 255;
                }
            }
        });
    } else if chroma_subsampling == YuvChromaSubsampling::Yuv422 {
        let iter;
        #[cfg(feature = "rayon")]
        {
            iter = y_plane
                .par_chunks_exact(y_stride as usize)
                .zip(uv_plane.par_chunks_exact(uv_stride as usize))
                .zip(bgra.par_chunks_exact_mut(bgra_stride as usize));
        }
        #[cfg(not(feature = "rayon"))]
        {
            iter = y_plane
                .chunks_exact(y_stride as usize)
                .zip(uv_plane.chunks_exact(uv_stride as usize))
                .zip(bgra.chunks_exact_mut(bgra_stride as usize));
        }
        iter.for_each(|((y_src, uv_src), rgba)| {
            process_halved_chroma_row(
                &y_src[0..image.width as usize],
                &uv_src[0..(image.width as usize).div_ceil(2) * 2],
                &mut rgba[0..image.width as usize * channels],
            );
        });
    } else if chroma_subsampling == YuvChromaSubsampling::Yuv420 {
        let iter;
        #[cfg(feature = "rayon")]
        {
            iter = y_plane
                .par_chunks_exact(y_stride as usize * 2)
                .zip(uv_plane.par_chunks_exact(uv_stride as usize))
                .zip(bgra.par_chunks_exact_mut(bgra_stride as usize * 2));
        }
        #[cfg(not(feature = "rayon"))]
        {
            iter = y_plane
                .chunks_exact(y_stride as usize * 2)
                .zip(uv_plane.chunks_exact(uv_stride as usize))
                .zip(bgra.chunks_exact_mut(bgra_stride as usize * 2));
        }
        iter.for_each(|((y_src, uv_src), rgba)| {
            let (y_src0, y_src1) = y_src.split_at(y_stride as usize);
            let (rgba0, rgba1) = rgba.split_at_mut(bgra_stride as usize);
            process_double_chroma_row(
                &y_src0[0..image.width as usize],
                &y_src1[0..image.width as usize],
                &uv_src[0..(image.width as usize).div_ceil(2) * 2],
                &mut rgba0[0..image.width as usize * channels],
                &mut rgba1[0..image.width as usize * channels],
            );
        });
        if image.height & 1 != 0 {
            let y_src = y_plane.chunks_exact(y_stride as usize * 2).remainder();
            let uv_src = uv_plane.chunks_exact(uv_stride as usize).last().unwrap();
            let rgba = bgra
                .chunks_exact_mut(bgra_stride as usize * 2)
                .into_remainder();
            process_halved_chroma_row(
                &y_src[0..image.width as usize],
                &uv_src[0..(image.width as usize).div_ceil(2) * 2],
                &mut rgba[0..image.width as usize * channels],
            );
        }
    } else {
        unreachable!();
    }

    Ok(())
}

/// Convert YUV NV12 format to BGRA format.
///
/// This function takes YUV NV12 data with 8-bit precision,
/// and converts it to BGRA format with 8-bit per channel precision.
///
/// # Arguments
///
/// * `bi_planar_image` - Source Bi-Planar image.
/// * `bgra` - A mutable slice to store the converted BGRA data.
/// * `bgra_stride` - The stride (components per row) for the BGRA image data.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGRA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv_nv12_to_bgra(
    bi_planar_image: &YuvBiPlanarImage<u8>,
    bgra: &mut [u8],
    bgra_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) -> Result<(), YuvError> {
    yuv_nv12_to_rgbx::<
        { YuvNVOrder::UV as u8 },
        { YuvSourceChannels::Bgra as u8 },
        { YuvChromaSubsampling::Yuv420 as u8 },
    >(bi_planar_image, bgra, bgra_stride, range, matrix)
}

/// Convert YUV NV16 format to BGRA format.
///
/// This function takes YUV NV16 data with 8-bit precision,
/// and converts it to BGRA format with 8-bit per channel precision.
///
/// # Arguments
///
/// * `bi_planar_image` - Source Bi-Planar image.
/// * `bgra` - A mutable slice to store the converted BGRA data.
/// * `bgra_stride` - The stride (components per row) for the BGRA image data.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGRA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv_nv16_to_bgra(
    bi_planar_image: &YuvBiPlanarImage<u8>,
    bgra: &mut [u8],
    bgra_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) -> Result<(), YuvError> {
    yuv_nv12_to_rgbx::<
        { YuvNVOrder::UV as u8 },
        { YuvSourceChannels::Bgra as u8 },
        { YuvChromaSubsampling::Yuv422 as u8 },
    >(bi_planar_image, bgra, bgra_stride, range, matrix)
}

/// Convert YUV NV61 format to BGRA format.
///
/// This function takes YUV NV61 data with 8-bit precision,
/// and converts it to BGRA format with 8-bit per channel precision.
///
/// # Arguments
///
/// * `bi_planar_image` - Source Bi-Planar image.
/// * `bgra` - A mutable slice to store the converted BGRA data.
/// * `bgra_stride` - The stride (components per row) for the BGRA image data.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGRA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv_nv61_to_bgra(
    bi_planar_image: &YuvBiPlanarImage<u8>,
    bgra: &mut [u8],
    bgra_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) -> Result<(), YuvError> {
    yuv_nv12_to_rgbx::<
        { YuvNVOrder::VU as u8 },
        { YuvSourceChannels::Bgra as u8 },
        { YuvChromaSubsampling::Yuv422 as u8 },
    >(bi_planar_image, bgra, bgra_stride, range, matrix)
}

/// Convert YUV NV21 format to BGRA format.
///
/// This function takes YUV NV12 data with 8-bit precision,
/// and converts it to BGRA format with 8-bit per channel precision.
///
/// # Arguments
///
/// * `bi_planar_image` - Source Bi-Planar image.
/// * `bgra` - A mutable slice to store the converted BGRA data.
/// * `bgra_stride` - The stride (components per row) for the BGRA image data.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGRA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv_nv21_to_bgra(
    bi_planar_image: &YuvBiPlanarImage<u8>,
    bgra: &mut [u8],
    bgra_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) -> Result<(), YuvError> {
    yuv_nv12_to_rgbx::<
        { YuvNVOrder::VU as u8 },
        { YuvSourceChannels::Bgra as u8 },
        { YuvChromaSubsampling::Yuv420 as u8 },
    >(bi_planar_image, bgra, bgra_stride, range, matrix)
}

/// Convert YUV NV16 format to RGBA format.
///
/// This function takes YUV NV16 data with 8-bit precision,
/// and converts it to RGBA format with 8-bit per channel precision.
///
/// # Arguments
///
/// * `bi_planar_image` - Source Bi-Planar image.
/// * `rgba` - A mutable slice to store the converted RGBA data.
/// * `rgba_stride` - The stride (components per row) for the RGBA image data.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGRA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv_nv16_to_rgba(
    bi_planar_image: &YuvBiPlanarImage<u8>,
    rgba: &mut [u8],
    rgba_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) -> Result<(), YuvError> {
    yuv_nv12_to_rgbx::<
        { YuvNVOrder::UV as u8 },
        { YuvSourceChannels::Rgba as u8 },
        { YuvChromaSubsampling::Yuv422 as u8 },
    >(bi_planar_image, rgba, rgba_stride, range, matrix)
}

/// Convert YUV NV61 format to RGBA format.
///
/// This function takes YUV NV61 data with 8-bit precision,
/// and converts it to RGBA format with 8-bit per channel precision.
///
/// # Arguments
///
/// * `bi_planar_image` - Source Bi-Planar image.
/// * `rgba` - A mutable slice to store the converted RGBA data.
/// * `rgba_stride` - The stride (components per row) for the RGBA image data.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input RGBA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv_nv61_to_rgba(
    bi_planar_image: &YuvBiPlanarImage<u8>,
    rgba: &mut [u8],
    rgba_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) -> Result<(), YuvError> {
    yuv_nv12_to_rgbx::<
        { YuvNVOrder::VU as u8 },
        { YuvSourceChannels::Rgba as u8 },
        { YuvChromaSubsampling::Yuv422 as u8 },
    >(bi_planar_image, rgba, rgba_stride, range, matrix)
}

/// Convert YUV NV12 format to RGBA format.
///
/// This function takes YUV NV12 data with 8-bit precision,
/// and converts it to RGBA format with 8-bit per channel precision.
///
/// # Arguments
///
/// * `bi_planar_image` - Source Bi-Planar image.
/// * `rgba` - A mutable slice to store the converted RGBA data.
/// * `rgba_stride` - The stride (components per row) for the RGBA image data.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input RGBA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv_nv12_to_rgba(
    bi_planar_image: &YuvBiPlanarImage<u8>,
    rgba: &mut [u8],
    rgba_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) -> Result<(), YuvError> {
    yuv_nv12_to_rgbx::<
        { YuvNVOrder::UV as u8 },
        { YuvSourceChannels::Rgba as u8 },
        { YuvChromaSubsampling::Yuv420 as u8 },
    >(bi_planar_image, rgba, rgba_stride, range, matrix)
}

/// Convert YUV NV21 format to RGBA format.
///
/// This function takes YUV NV21 data with 8-bit precision,
/// and converts it to RGBA format with 8-bit per channel precision.
///
/// # Arguments
///
/// * `bi_planar_image` - Source Bi-Planar image.
/// * `rgba` - A mutable slice to store the converted RGBA data.
/// * `rgba_stride` - The stride (components per row) for the RGBA image data.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input RGBA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv_nv21_to_rgba(
    bi_planar_image: &YuvBiPlanarImage<u8>,
    rgba: &mut [u8],
    rgba_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) -> Result<(), YuvError> {
    yuv_nv12_to_rgbx::<
        { YuvNVOrder::VU as u8 },
        { YuvSourceChannels::Rgba as u8 },
        { YuvChromaSubsampling::Yuv420 as u8 },
    >(bi_planar_image, rgba, rgba_stride, range, matrix)
}

/// Convert YUV NV12 format to RGB format.
///
/// This function takes YUV NV12 data with 8-bit precision,
/// and converts it to RGB format with 8-bit per channel precision.
///
/// # Arguments
///
/// * `bi_planar_image` - Source Bi-Planar image.
/// * `rgb` - A mutable slice to store the converted RGB data.
/// * `rgb_stride` - The stride (components per row) for the RGB image data.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input RGB data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv_nv12_to_rgb(
    bi_planar_image: &YuvBiPlanarImage<u8>,
    rgb: &mut [u8],
    rgb_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) -> Result<(), YuvError> {
    yuv_nv12_to_rgbx::<
        { YuvNVOrder::UV as u8 },
        { YuvSourceChannels::Rgb as u8 },
        { YuvChromaSubsampling::Yuv420 as u8 },
    >(bi_planar_image, rgb, rgb_stride, range, matrix)
}

/// Convert YUV NV12 format to BGR format.
///
/// This function takes YUV NV12 data with 8-bit precision,
/// and converts it to BGR format with 8-bit per channel precision.
///
/// # Arguments
///
/// * `bi_planar_image` - Source Bi-Planar image.
/// * `bgr` - A mutable slice to store the converted BGR data.
/// * `bgr_stride` - The stride (components per row) for the BGR image data.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGR data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv_nv12_to_bgr(
    bi_planar_image: &YuvBiPlanarImage<u8>,
    bgr: &mut [u8],
    bgr_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) -> Result<(), YuvError> {
    yuv_nv12_to_rgbx::<
        { YuvNVOrder::UV as u8 },
        { YuvSourceChannels::Bgr as u8 },
        { YuvChromaSubsampling::Yuv420 as u8 },
    >(bi_planar_image, bgr, bgr_stride, range, matrix)
}

/// Convert YUV NV16 format to RGB format.
///
/// This function takes YUV NV16 data with 8-bit precision,
/// and converts it to RGB format with 8-bit per channel precision.
///
/// # Arguments
///
/// * `bi_planar_image` - Source Bi-Planar image.
/// * `rgb` - A mutable slice to store the converted RGB data.
/// * `rgb_stride` - The stride (components per row) for the RGB image data.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input RGB data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv_nv16_to_rgb(
    bi_planar_image: &YuvBiPlanarImage<u8>,
    rgb: &mut [u8],
    rgb_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) -> Result<(), YuvError> {
    yuv_nv12_to_rgbx::<
        { YuvNVOrder::UV as u8 },
        { YuvSourceChannels::Rgb as u8 },
        { YuvChromaSubsampling::Yuv422 as u8 },
    >(bi_planar_image, rgb, rgb_stride, range, matrix)
}

/// Convert YUV NV16 format to BGR format.
///
/// This function takes YUV NV16 data with 8-bit precision,
/// and converts it to BGR format with 8-bit per channel precision.
///
/// # Arguments
///
/// * `bi_planar_image` - Source Bi-Planar image.
/// * `bgr` - A mutable slice to store the converted BGR data.
/// * `bgr_stride` - The stride (components per row) for the BGR image data.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGR data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv_nv16_to_bgr(
    bi_planar_image: &YuvBiPlanarImage<u8>,
    bgr: &mut [u8],
    bgr_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) -> Result<(), YuvError> {
    yuv_nv12_to_rgbx::<
        { YuvNVOrder::UV as u8 },
        { YuvSourceChannels::Bgr as u8 },
        { YuvChromaSubsampling::Yuv422 as u8 },
    >(bi_planar_image, bgr, bgr_stride, range, matrix)
}

/// Convert YUV NV61 format to RGB format.
///
/// This function takes YUV NV61 data with 8-bit precision,
/// and converts it to RGB format with 8-bit per channel precision.
///
/// # Arguments
///
/// * `bi_planar_image` - Source Bi-Planar image.
/// * `rgb` - A mutable slice to store the converted RGB data.
/// * `rgb_stride` - The stride (components per row) for the RGB image data.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input RGB data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv_nv61_to_rgb(
    bi_planar_image: &YuvBiPlanarImage<u8>,
    rgb: &mut [u8],
    rgb_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) -> Result<(), YuvError> {
    yuv_nv12_to_rgbx::<
        { YuvNVOrder::VU as u8 },
        { YuvSourceChannels::Rgb as u8 },
        { YuvChromaSubsampling::Yuv422 as u8 },
    >(bi_planar_image, rgb, rgb_stride, range, matrix)
}

/// Convert YUV NV61 format to BGR format.
///
/// This function takes YUV NV61 data with 8-bit precision,
/// and converts it to BGR format with 8-bit per channel precision.
///
/// # Arguments
///
/// * `bi_planar_image` - Source Bi-Planar image.
/// * `bgr` - A mutable slice to store the converted BGR data.
/// * `bgr_stride` - The stride (components per row) for the BGR image data.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGR data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv_nv61_to_bgr(
    bi_planar_image: &YuvBiPlanarImage<u8>,
    bgr: &mut [u8],
    bgr_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) -> Result<(), YuvError> {
    yuv_nv12_to_rgbx::<
        { YuvNVOrder::VU as u8 },
        { YuvSourceChannels::Bgr as u8 },
        { YuvChromaSubsampling::Yuv422 as u8 },
    >(bi_planar_image, bgr, bgr_stride, range, matrix)
}

/// Convert YUV NV21 format to RGB format.
///
/// This function takes YUV NV21 data with 8-bit precision,
/// and converts it to RGB format with 8-bit per channel precision.
///
/// # Arguments
///
/// * `bi_planar_image` - Source Bi-Planar image.
/// * `rgb` - A mutable slice to store the converted RGB data.
/// * `rgb_stride` - The stride (components per row) for the RGB image data.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input RGB data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv_nv21_to_rgb(
    bi_planar_image: &YuvBiPlanarImage<u8>,
    rgb: &mut [u8],
    rgb_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) -> Result<(), YuvError> {
    yuv_nv12_to_rgbx::<
        { YuvNVOrder::VU as u8 },
        { YuvSourceChannels::Rgb as u8 },
        { YuvChromaSubsampling::Yuv420 as u8 },
    >(bi_planar_image, rgb, rgb_stride, range, matrix)
}

/// Convert YUV NV21 format to BGR format.
///
/// This function takes YUV NV21 data with 8-bit precision,
/// and converts it to BGR format with 8-bit per channel precision.
///
/// # Arguments
///
/// * `bi_planar_image` - Source Bi-Planar image.
/// * `bgr` - A mutable slice to store the converted BGR data.
/// * `bgr_stride` - The stride (components per row) for the BGR image data.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGR data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv_nv21_to_bgr(
    bi_planar_image: &YuvBiPlanarImage<u8>,
    bgr: &mut [u8],
    bgr_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) -> Result<(), YuvError> {
    yuv_nv12_to_rgbx::<
        { YuvNVOrder::VU as u8 },
        { YuvSourceChannels::Bgr as u8 },
        { YuvChromaSubsampling::Yuv420 as u8 },
    >(bi_planar_image, bgr, bgr_stride, range, matrix)
}

/// Convert YUV NV24 format to RGBA format.
///
/// This function takes YUV NV24 data with 8-bit precision,
/// and converts it to RGBA format with 8-bit per channel precision.
///
/// # Arguments
///
/// * `bi_planar_image` - Source Bi-Planar image.
/// * `rgba` - A mutable slice to store the converted RGBA data.
/// * `rgba_stride` - The stride (components per row) for the RGBA image data.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGRA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv_nv42_to_rgba(
    bi_planar_image: &YuvBiPlanarImage<u8>,
    rgba: &mut [u8],
    rgba_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) -> Result<(), YuvError> {
    yuv_nv12_to_rgbx::<
        { YuvNVOrder::VU as u8 },
        { YuvSourceChannels::Rgba as u8 },
        { YuvChromaSubsampling::Yuv444 as u8 },
    >(bi_planar_image, rgba, rgba_stride, range, matrix)
}

/// Convert YUV NV24 format to RGB format.
///
/// This function takes YUV NV24 data with 8-bit precision,
/// and converts it to RGB format with 8-bit per channel precision.
///
/// # Arguments
///
/// * `bi_planar_image` - Source Bi-Planar image.
/// * `rgb` - A mutable slice to store the converted RGB data.
/// * `rgb_stride` - The stride (components per row) for the RGB image data.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input RGB data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv_nv24_to_rgb(
    bi_planar_image: &YuvBiPlanarImage<u8>,
    rgb: &mut [u8],
    rgb_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) -> Result<(), YuvError> {
    yuv_nv12_to_rgbx::<
        { YuvNVOrder::UV as u8 },
        { YuvSourceChannels::Rgb as u8 },
        { YuvChromaSubsampling::Yuv444 as u8 },
    >(bi_planar_image, rgb, rgb_stride, range, matrix)
}

/// Convert YUV NV24 format to BGR format.
///
/// This function takes YUV NV24 data with 8-bit precision,
/// and converts it to BGR format with 8-bit per channel precision.
///
/// # Arguments
///
/// * `bi_planar_image` - Source Bi-Planar image.
/// * `bgr` - A mutable slice to store the converted BGR data.
/// * `bgr_stride` - The stride (components per row) for the BGR image data.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGR data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv_nv24_to_bgr(
    bi_planar_image: &YuvBiPlanarImage<u8>,
    bgr: &mut [u8],
    bgr_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) -> Result<(), YuvError> {
    yuv_nv12_to_rgbx::<
        { YuvNVOrder::UV as u8 },
        { YuvSourceChannels::Bgr as u8 },
        { YuvChromaSubsampling::Yuv444 as u8 },
    >(bi_planar_image, bgr, bgr_stride, range, matrix)
}

/// Convert YUV NV24 format to RGBA format.
///
/// This function takes YUV NV24 data with 8-bit precision,
/// and converts it to RGBA format with 8-bit per channel precision.
///
/// # Arguments
///
/// * `bi_planar_image` - Source Bi-Planar image.
/// * `rgba` - A mutable slice to store the converted RGBA data.
/// * `rgba_stride` - The stride (components per row) for the RGBA image data.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input RGBA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv_nv24_to_rgba(
    bi_planar_image: &YuvBiPlanarImage<u8>,
    rgb: &mut [u8],
    rgb_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) -> Result<(), YuvError> {
    yuv_nv12_to_rgbx::<
        { YuvNVOrder::UV as u8 },
        { YuvSourceChannels::Rgba as u8 },
        { YuvChromaSubsampling::Yuv444 as u8 },
    >(bi_planar_image, rgb, rgb_stride, range, matrix)
}

/// Convert YUV NV24 format to BGRA format.
///
/// This function takes YUV NV24 data with 8-bit precision,
/// and converts it to RGBA format with 8-bit per channel precision.
///
/// # Arguments
///
/// * `bi_planar_image` - Source Bi-Planar image.
/// * `bgra` - A mutable slice to store the converted BGRA data.
/// * `bgra_stride` - The stride (components per row) for the BGRA image data.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGRA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv_nv24_to_bgra(
    bi_planar_image: &YuvBiPlanarImage<u8>,
    rgb: &mut [u8],
    rgb_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) -> Result<(), YuvError> {
    yuv_nv12_to_rgbx::<
        { YuvNVOrder::UV as u8 },
        { YuvSourceChannels::Bgra as u8 },
        { YuvChromaSubsampling::Yuv444 as u8 },
    >(bi_planar_image, rgb, rgb_stride, range, matrix)
}

/// Convert YUV NV42 format to RGB format.
///
/// This function takes YUV NV42 data with 8-bit precision,
/// and converts it to RGB format with 8-bit per channel precision.
///
/// # Arguments
///
/// * `bi_planar_image` - Source Bi-Planar image.
/// * `rgb` - A mutable slice to store the converted RGB data.
/// * `rgb_stride` - The stride (components per row) for the RGB image data.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input RGB data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv_nv42_to_rgb(
    bi_planar_image: &YuvBiPlanarImage<u8>,
    rgb: &mut [u8],
    rgb_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) -> Result<(), YuvError> {
    yuv_nv12_to_rgbx::<
        { YuvNVOrder::VU as u8 },
        { YuvSourceChannels::Rgb as u8 },
        { YuvChromaSubsampling::Yuv444 as u8 },
    >(bi_planar_image, rgb, rgb_stride, range, matrix)
}

/// Convert YUV NV42 format to BGR format.
///
/// This function takes YUV NV42 data with 8-bit precision,
/// and converts it to BGR format with 8-bit per channel precision.
///
/// # Arguments
///
/// * `bi_planar_image` - Source Bi-Planar image.
/// * `bgr` - A mutable slice to store the converted BGR data.
/// * `bgr_stride` - The stride (components per row) for the BGR image data.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGR data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv_nv42_to_bgr(
    bi_planar_image: &YuvBiPlanarImage<u8>,
    bgr: &mut [u8],
    bgr_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) -> Result<(), YuvError> {
    yuv_nv12_to_rgbx::<
        { YuvNVOrder::VU as u8 },
        { YuvSourceChannels::Bgr as u8 },
        { YuvChromaSubsampling::Yuv444 as u8 },
    >(bi_planar_image, bgr, bgr_stride, range, matrix)
}

/// Convert YUV NV42 format to BGRA format.
///
/// This function takes YUV NV42 data with 8-bit precision,
/// and converts it to RGB format with 8-bit per channel precision.
///
/// # Arguments
///
/// * `bi_planar_image` - Source Bi-Planar image.
/// * `bgra` - A mutable slice to store the converted BGRA data.
/// * `bgra_stride` - The stride (components per row) for the BGRA image data.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGRA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv_nv42_to_bgra(
    bi_planar_image: &YuvBiPlanarImage<u8>,
    rgb: &mut [u8],
    rgb_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) -> Result<(), YuvError> {
    yuv_nv12_to_rgbx::<
        { YuvNVOrder::VU as u8 },
        { YuvSourceChannels::Bgra as u8 },
        { YuvChromaSubsampling::Yuv444 as u8 },
    >(bi_planar_image, rgb, rgb_stride, range, matrix)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{rgb_to_yuv_nv12, rgb_to_yuv_nv16, rgb_to_yuv_nv24, YuvBiPlanarImageMut};
    use rand::Rng;

    #[test]
    fn test_yuv444_nv_round_trip_full_range() {
        let image_width = 256usize;
        let image_height = 256usize;

        let random_point_x = rand::thread_rng().gen_range(0..image_width);
        let random_point_y = rand::thread_rng().gen_range(0..image_height);

        const CHANNELS: usize = 3;

        let pixel_points = [
            [0, 0],
            [image_width - 1, image_height - 1],
            [image_width - 1, 0],
            [0, image_height - 1],
            [(image_width - 1) / 2, (image_height - 1) / 2],
            [image_width / 5, image_height / 5],
            [0, image_height / 5],
            [image_width / 5, 0],
            [image_width / 5 * 3, image_height / 5],
            [image_width / 5 * 3, image_height / 5 * 3],
            [image_width / 5, image_height / 5 * 3],
            [random_point_x, random_point_y],
        ];

        let mut image_rgb = vec![0u8; image_width * image_height * 3];

        let or = rand::thread_rng().gen_range(0..256) as u8;
        let og = rand::thread_rng().gen_range(0..256) as u8;
        let ob = rand::thread_rng().gen_range(0..256) as u8;

        for point in &pixel_points {
            image_rgb[point[0] * CHANNELS + point[1] * image_width * CHANNELS] = or;
            image_rgb[point[0] * CHANNELS + point[1] * image_width * CHANNELS + 1] = og;
            image_rgb[point[0] * CHANNELS + point[1] * image_width * CHANNELS + 2] = ob;
        }

        let mut planar_image = YuvBiPlanarImageMut::<u8>::alloc(
            image_width as u32,
            image_height as u32,
            YuvChromaSubsampling::Yuv444,
        );

        rgb_to_yuv_nv24(
            &mut planar_image,
            &image_rgb,
            image_width as u32 * CHANNELS as u32,
            YuvRange::Full,
            YuvStandardMatrix::Bt709,
        )
        .unwrap();

        image_rgb.fill(0);

        let fixed_planar = planar_image.to_fixed();

        yuv_nv24_to_rgb(
            &fixed_planar,
            &mut image_rgb,
            image_width as u32 * CHANNELS as u32,
            YuvRange::Full,
            YuvStandardMatrix::Bt709,
        )
        .unwrap();

        for point in &pixel_points {
            let x = point[0];
            let y = point[1];
            let r = image_rgb[x * CHANNELS + y * image_width * CHANNELS];
            let g = image_rgb[x * CHANNELS + y * image_width * CHANNELS + 1];
            let b = image_rgb[x * CHANNELS + y * image_width * CHANNELS + 2];

            let diff_r = (r as i32 - or as i32).abs();
            let diff_g = (g as i32 - og as i32).abs();
            let diff_b = (b as i32 - ob as i32).abs();

            assert!(
                diff_r <= 3,
                "Original RGB {:?}, Round-tripped RGB {:?}",
                [or, og, ob],
                [r, g, b]
            );
            assert!(
                diff_g <= 3,
                "Original RGB {:?}, Round-tripped RGB {:?}",
                [or, og, ob],
                [r, g, b]
            );
            assert!(
                diff_b <= 3,
                "Original RGB {:?}, Round-tripped RGB {:?}",
                [or, og, ob],
                [r, g, b]
            );
        }
    }

    #[test]
    fn test_yuv444_nv_round_trip_limited_range() {
        let image_width = 256usize;
        let image_height = 256usize;

        let random_point_x = rand::thread_rng().gen_range(0..image_width);
        let random_point_y = rand::thread_rng().gen_range(0..image_height);

        let pixel_points = [
            [0, 0],
            [image_width - 1, image_height - 1],
            [image_width - 1, 0],
            [0, image_height - 1],
            [(image_width - 1) / 2, (image_height - 1) / 2],
            [image_width / 5, image_height / 5],
            [0, image_height / 5],
            [image_width / 5, 0],
            [image_width / 5 * 3, image_height / 5],
            [image_width / 5 * 3, image_height / 5 * 3],
            [image_width / 5, image_height / 5 * 3],
            [random_point_x, random_point_y],
        ];

        const CHANNELS: usize = 3;

        let mut image_rgb = vec![0u8; image_width * image_height * CHANNELS];

        let or = rand::thread_rng().gen_range(0..256) as u8;
        let og = rand::thread_rng().gen_range(0..256) as u8;
        let ob = rand::thread_rng().gen_range(0..256) as u8;

        for point in &pixel_points {
            image_rgb[point[0] * CHANNELS + point[1] * image_width * CHANNELS] = or;
            image_rgb[point[0] * CHANNELS + point[1] * image_width * CHANNELS + 1] = og;
            image_rgb[point[0] * CHANNELS + point[1] * image_width * CHANNELS + 2] = ob;
        }

        let mut planar_image = YuvBiPlanarImageMut::<u8>::alloc(
            image_width as u32,
            image_height as u32,
            YuvChromaSubsampling::Yuv444,
        );

        rgb_to_yuv_nv24(
            &mut planar_image,
            &image_rgb,
            image_width as u32 * CHANNELS as u32,
            YuvRange::Limited,
            YuvStandardMatrix::Bt709,
        )
        .unwrap();

        image_rgb.fill(0);

        let fixed_planar = planar_image.to_fixed();

        yuv_nv24_to_rgb(
            &fixed_planar,
            &mut image_rgb,
            image_width as u32 * CHANNELS as u32,
            YuvRange::Limited,
            YuvStandardMatrix::Bt709,
        )
        .unwrap();

        for point in &pixel_points {
            let x = point[0];
            let y = point[1];
            let r = image_rgb[x * CHANNELS + y * image_width * CHANNELS];
            let g = image_rgb[x * CHANNELS + y * image_width * CHANNELS + 1];
            let b = image_rgb[x * CHANNELS + y * image_width * CHANNELS + 2];

            let diff_r = (r as i32 - or as i32).abs();
            let diff_g = (g as i32 - og as i32).abs();
            let diff_b = (b as i32 - ob as i32).abs();

            assert!(
                diff_r <= 10,
                "Original RGB {:?}, Round-tripped RGB {:?}",
                [or, og, ob],
                [r, g, b]
            );
            assert!(
                diff_g <= 10,
                "Original RGB {:?}, Round-tripped RGB {:?}",
                [or, og, ob],
                [r, g, b]
            );
            assert!(
                diff_b <= 10,
                "Original RGB {:?}, Round-tripped RGB {:?}",
                [or, og, ob],
                [r, g, b]
            );
        }
    }

    #[test]
    fn test_yuv422_nv_round_trip_full_range() {
        let image_width = 256usize;
        let image_height = 256usize;

        let random_point_x = rand::thread_rng().gen_range(0..image_width);
        let random_point_y = rand::thread_rng().gen_range(0..image_height);

        const CHANNELS: usize = 3;

        let pixel_points = [
            [0, 0],
            [image_width - 1, image_height - 1],
            [image_width - 1, 0],
            [0, image_height - 1],
            [(image_width - 1) / 2, (image_height - 1) / 2],
            [image_width / 5, image_height / 5],
            [0, image_height / 5],
            [image_width / 5, 0],
            [image_width / 5 * 3, image_height / 5],
            [image_width / 5 * 3, image_height / 5 * 3],
            [image_width / 5, image_height / 5 * 3],
            [random_point_x, random_point_y],
        ];

        let mut source_rgb = vec![0u8; image_width * image_height * CHANNELS];

        let or = rand::thread_rng().gen_range(0..256) as u8;
        let og = rand::thread_rng().gen_range(0..256) as u8;
        let ob = rand::thread_rng().gen_range(0..256) as u8;

        for point in &pixel_points {
            source_rgb[point[0] * CHANNELS + point[1] * image_width * CHANNELS] = or;
            source_rgb[point[0] * CHANNELS + point[1] * image_width * CHANNELS + 1] = og;
            source_rgb[point[0] * CHANNELS + point[1] * image_width * CHANNELS + 2] = ob;

            let nx = (point[0] + 1).min(image_width - 1);
            let ny = point[1].min(image_height - 1);

            source_rgb[nx * CHANNELS + ny * image_width * CHANNELS] = or;
            source_rgb[nx * CHANNELS + ny * image_width * CHANNELS + 1] = og;
            source_rgb[nx * CHANNELS + ny * image_width * CHANNELS + 2] = ob;

            let nx = point[0].saturating_sub(1).min(image_width - 1);
            let ny = point[1].min(image_height - 1);

            source_rgb[nx * CHANNELS + ny * image_width * CHANNELS] = or;
            source_rgb[nx * CHANNELS + ny * image_width * CHANNELS + 1] = og;
            source_rgb[nx * CHANNELS + ny * image_width * CHANNELS + 2] = ob;
        }

        let mut planar_image = YuvBiPlanarImageMut::<u8>::alloc(
            image_width as u32,
            image_height as u32,
            YuvChromaSubsampling::Yuv422,
        );

        rgb_to_yuv_nv16(
            &mut planar_image,
            &source_rgb,
            image_width as u32 * CHANNELS as u32,
            YuvRange::Full,
            YuvStandardMatrix::Bt709,
        )
        .unwrap();

        let mut dest_rgb = vec![0u8; image_width * image_height * CHANNELS];

        let fixed_planar = planar_image.to_fixed();

        yuv_nv16_to_rgb(
            &fixed_planar,
            &mut dest_rgb,
            image_width as u32 * CHANNELS as u32,
            YuvRange::Full,
            YuvStandardMatrix::Bt709,
        )
        .unwrap();

        for point in &pixel_points {
            let x = point[0];
            let y = point[1];
            let px = x * CHANNELS + y * image_width * CHANNELS;

            let r = dest_rgb[px];
            let g = dest_rgb[px + 1];
            let b = dest_rgb[px + 2];

            let diff_r = r as i32 - or as i32;
            let diff_g = g as i32 - og as i32;
            let diff_b = b as i32 - ob as i32;

            assert!(
                diff_r <= 2,
                "Actual diff {}, Original RGB {:?}, Round-tripped RGB {:?}",
                diff_r,
                [or, og, ob],
                [r, g, b]
            );
            assert!(
                diff_g <= 2,
                "Actual diff {}, Original RGB {:?}, Round-tripped RGB {:?}",
                diff_g,
                [or, og, ob],
                [r, g, b]
            );
            assert!(
                diff_b <= 2,
                "Actual diff {}, Original RGB {:?}, Round-tripped RGB {:?}",
                diff_b,
                [or, og, ob],
                [r, g, b]
            );
        }
    }

    #[test]
    fn test_yuv422_nv_round_trip_limited_range() {
        let image_width = 256usize;
        let image_height = 256usize;

        let random_point_x = rand::thread_rng().gen_range(0..image_width);
        let random_point_y = rand::thread_rng().gen_range(0..image_height);

        const CHANNELS: usize = 3;

        let pixel_points = [
            [0, 0],
            [image_width - 1, image_height - 1],
            [image_width - 1, 0],
            [0, image_height - 1],
            [(image_width - 1) / 2, (image_height - 1) / 2],
            [image_width / 5, image_height / 5],
            [0, image_height / 5],
            [image_width / 5, 0],
            [image_width / 5 * 3, image_height / 5],
            [image_width / 5 * 3, image_height / 5 * 3],
            [image_width / 5, image_height / 5 * 3],
            [random_point_x, random_point_y],
        ];

        let mut source_rgb = vec![0u8; image_width * image_height * CHANNELS];

        let or = rand::thread_rng().gen_range(0..256) as u8;
        let og = rand::thread_rng().gen_range(0..256) as u8;
        let ob = rand::thread_rng().gen_range(0..256) as u8;

        for point in &pixel_points {
            source_rgb[point[0] * CHANNELS + point[1] * image_width * CHANNELS] = or;
            source_rgb[point[0] * CHANNELS + point[1] * image_width * CHANNELS + 1] = og;
            source_rgb[point[0] * CHANNELS + point[1] * image_width * CHANNELS + 2] = ob;

            let nx = (point[0] + 1).min(image_width - 1);
            let ny = point[1].min(image_height - 1);

            source_rgb[nx * CHANNELS + ny * image_width * CHANNELS] = or;
            source_rgb[nx * CHANNELS + ny * image_width * CHANNELS + 1] = og;
            source_rgb[nx * CHANNELS + ny * image_width * CHANNELS + 2] = ob;

            let nx = point[0].saturating_sub(1).min(image_width - 1);
            let ny = point[1].min(image_height - 1);

            source_rgb[nx * CHANNELS + ny * image_width * CHANNELS] = or;
            source_rgb[nx * CHANNELS + ny * image_width * CHANNELS + 1] = og;
            source_rgb[nx * CHANNELS + ny * image_width * CHANNELS + 2] = ob;
        }

        let mut planar_image = YuvBiPlanarImageMut::<u8>::alloc(
            image_width as u32,
            image_height as u32,
            YuvChromaSubsampling::Yuv422,
        );

        rgb_to_yuv_nv16(
            &mut planar_image,
            &source_rgb,
            image_width as u32 * CHANNELS as u32,
            YuvRange::Limited,
            YuvStandardMatrix::Bt709,
        )
        .unwrap();

        let mut dest_rgb = vec![0u8; image_width * image_height * CHANNELS];

        let fixed_planar = planar_image.to_fixed();

        yuv_nv16_to_rgb(
            &fixed_planar,
            &mut dest_rgb,
            image_width as u32 * CHANNELS as u32,
            YuvRange::Limited,
            YuvStandardMatrix::Bt709,
        )
        .unwrap();

        for point in &pixel_points {
            let x = point[0];
            let y = point[1];
            let px = x * CHANNELS + y * image_width * CHANNELS;

            let r = dest_rgb[px];
            let g = dest_rgb[px + 1];
            let b = dest_rgb[px + 2];

            let diff_r = r as i32 - or as i32;
            let diff_g = g as i32 - og as i32;
            let diff_b = b as i32 - ob as i32;

            assert!(
                diff_r <= 10,
                "Actual diff {}, Original RGB {:?}, Round-tripped RGB {:?}",
                diff_r,
                [or, og, ob],
                [r, g, b]
            );
            assert!(
                diff_g <= 10,
                "Actual diff {}, Original RGB {:?}, Round-tripped RGB {:?}",
                diff_g,
                [or, og, ob],
                [r, g, b]
            );
            assert!(
                diff_b <= 10,
                "Actual diff {}, Original RGB {:?}, Round-tripped RGB {:?}",
                diff_b,
                [or, og, ob],
                [r, g, b]
            );
        }
    }

    #[test]
    fn test_yuv420_nv_round_trip_full_range() {
        let image_width = 256usize;
        let image_height = 256usize;

        let random_point_x = rand::thread_rng().gen_range(0..image_width);
        let random_point_y = rand::thread_rng().gen_range(0..image_height);

        const CHANNELS: usize = 3;

        let pixel_points = [
            [0, 0],
            [image_width - 1, image_height - 1],
            [image_width - 1, 0],
            [0, image_height - 1],
            [(image_width - 1) / 2, (image_height - 1) / 2],
            [image_width / 5, image_height / 5],
            [0, image_height / 5],
            [image_width / 5, 0],
            [image_width / 5 * 3, image_height / 5],
            [image_width / 5 * 3, image_height / 5 * 3],
            [image_width / 5, image_height / 5 * 3],
            [random_point_x, random_point_y],
        ];

        let mut source_rgb = vec![0u8; image_width * image_height * CHANNELS];

        let or = rand::thread_rng().gen_range(0..256) as u8;
        let og = rand::thread_rng().gen_range(0..256) as u8;
        let ob = rand::thread_rng().gen_range(0..256) as u8;

        for point in &pixel_points {
            source_rgb[point[0] * CHANNELS + point[1] * image_width * CHANNELS] = or;
            source_rgb[point[0] * CHANNELS + point[1] * image_width * CHANNELS + 1] = og;
            source_rgb[point[0] * CHANNELS + point[1] * image_width * CHANNELS + 2] = ob;

            let nx = (point[0] + 1).min(image_width - 1);
            let ny = point[1].min(image_height - 1);

            source_rgb[nx * CHANNELS + ny * image_width * CHANNELS] = or;
            source_rgb[nx * CHANNELS + ny * image_width * CHANNELS + 1] = og;
            source_rgb[nx * CHANNELS + ny * image_width * CHANNELS + 2] = ob;

            let nx = (point[0] + 1).min(image_width - 1);
            let ny = (point[1] + 1).min(image_height - 1);

            source_rgb[nx * CHANNELS + ny * image_width * CHANNELS] = or;
            source_rgb[nx * CHANNELS + ny * image_width * CHANNELS + 1] = og;
            source_rgb[nx * CHANNELS + ny * image_width * CHANNELS + 2] = ob;

            let nx = point[0].min(image_width - 1);
            let ny = (point[1] + 1).min(image_height - 1);

            source_rgb[nx * CHANNELS + ny * image_width * CHANNELS] = or;
            source_rgb[nx * CHANNELS + ny * image_width * CHANNELS + 1] = og;
            source_rgb[nx * CHANNELS + ny * image_width * CHANNELS + 2] = ob;

            let nx = point[0].saturating_sub(1).min(image_width - 1);
            let ny = point[1].saturating_sub(1).min(image_height - 1);

            source_rgb[nx * CHANNELS + ny * image_width * CHANNELS] = or;
            source_rgb[nx * CHANNELS + ny * image_width * CHANNELS + 1] = og;
            source_rgb[nx * CHANNELS + ny * image_width * CHANNELS + 2] = ob;

            let nx = point[0].min(image_width - 1);
            let ny = point[1].saturating_sub(1).min(image_height - 1);

            source_rgb[nx * CHANNELS + ny * image_width * CHANNELS] = or;
            source_rgb[nx * CHANNELS + ny * image_width * CHANNELS + 1] = og;
            source_rgb[nx * CHANNELS + ny * image_width * CHANNELS + 2] = ob;

            let nx = point[0].saturating_sub(1).min(image_width - 1);
            let ny = point[1].min(image_height - 1);

            source_rgb[nx * CHANNELS + ny * image_width * CHANNELS] = or;
            source_rgb[nx * CHANNELS + ny * image_width * CHANNELS + 1] = og;
            source_rgb[nx * CHANNELS + ny * image_width * CHANNELS + 2] = ob;
        }

        let mut planar_image = YuvBiPlanarImageMut::<u8>::alloc(
            image_width as u32,
            image_height as u32,
            YuvChromaSubsampling::Yuv420,
        );

        rgb_to_yuv_nv12(
            &mut planar_image,
            &source_rgb,
            image_width as u32 * CHANNELS as u32,
            YuvRange::Full,
            YuvStandardMatrix::Bt709,
        )
        .unwrap();

        let mut dest_rgb = vec![0u8; image_width * image_height * CHANNELS];

        let fixed_planar = planar_image.to_fixed();

        yuv_nv12_to_rgb(
            &fixed_planar,
            &mut dest_rgb,
            image_width as u32 * CHANNELS as u32,
            YuvRange::Full,
            YuvStandardMatrix::Bt709,
        )
        .unwrap();

        for point in &pixel_points {
            let x = point[0];
            let y = point[1];
            let px = x * CHANNELS + y * image_width * CHANNELS;

            let r = dest_rgb[px];
            let g = dest_rgb[px + 1];
            let b = dest_rgb[px + 2];

            let diff_r = r as i32 - or as i32;
            let diff_g = g as i32 - og as i32;
            let diff_b = b as i32 - ob as i32;

            assert!(
                diff_r <= 47,
                "Actual diff {}, Original RGB {:?}, Round-tripped RGB {:?}",
                diff_r,
                [or, og, ob],
                [r, g, b]
            );
            assert!(
                diff_g <= 47,
                "Actual diff {}, Original RGB {:?}, Round-tripped RGB {:?}",
                diff_g,
                [or, og, ob],
                [r, g, b]
            );
            assert!(
                diff_b <= 47,
                "Actual diff {}, Original RGB {:?}, Round-tripped RGB {:?}",
                diff_b,
                [or, og, ob],
                [r, g, b]
            );
        }
    }

    #[test]
    fn test_yuv420_nv_round_trip_limited_range() {
        let image_width = 256usize;
        let image_height = 256usize;

        let random_point_x = rand::thread_rng().gen_range(0..image_width);
        let random_point_y = rand::thread_rng().gen_range(0..image_height);

        const CHANNELS: usize = 3;

        let pixel_points = [
            [0, 0],
            [image_width - 1, image_height - 1],
            [image_width - 1, 0],
            [0, image_height - 1],
            [(image_width - 1) / 2, (image_height - 1) / 2],
            [image_width / 5, image_height / 5],
            [0, image_height / 5],
            [image_width / 5, 0],
            [image_width / 5 * 3, image_height / 5],
            [image_width / 5 * 3, image_height / 5 * 3],
            [image_width / 5, image_height / 5 * 3],
            [random_point_x, random_point_y],
        ];

        let mut source_rgb = vec![0u8; image_width * image_height * CHANNELS];

        let or = rand::thread_rng().gen_range(0..256) as u8;
        let og = rand::thread_rng().gen_range(0..256) as u8;
        let ob = rand::thread_rng().gen_range(0..256) as u8;

        for point in &pixel_points {
            source_rgb[point[0] * CHANNELS + point[1] * image_width * CHANNELS] = or;
            source_rgb[point[0] * CHANNELS + point[1] * image_width * CHANNELS + 1] = og;
            source_rgb[point[0] * CHANNELS + point[1] * image_width * CHANNELS + 2] = ob;

            let nx = (point[0] + 1).min(image_width - 1);
            let ny = point[1].min(image_height - 1);

            source_rgb[nx * CHANNELS + ny * image_width * CHANNELS] = or;
            source_rgb[nx * CHANNELS + ny * image_width * CHANNELS + 1] = og;
            source_rgb[nx * CHANNELS + ny * image_width * CHANNELS + 2] = ob;

            let nx = (point[0] + 1).min(image_width - 1);
            let ny = (point[1] + 1).min(image_height - 1);

            source_rgb[nx * CHANNELS + ny * image_width * CHANNELS] = or;
            source_rgb[nx * CHANNELS + ny * image_width * CHANNELS + 1] = og;
            source_rgb[nx * CHANNELS + ny * image_width * CHANNELS + 2] = ob;

            let nx = point[0].min(image_width - 1);
            let ny = (point[1] + 1).min(image_height - 1);

            source_rgb[nx * CHANNELS + ny * image_width * CHANNELS] = or;
            source_rgb[nx * CHANNELS + ny * image_width * CHANNELS + 1] = og;
            source_rgb[nx * CHANNELS + ny * image_width * CHANNELS + 2] = ob;

            let nx = point[0].saturating_sub(1).min(image_width - 1);
            let ny = point[1].saturating_sub(1).min(image_height - 1);

            source_rgb[nx * CHANNELS + ny * image_width * CHANNELS] = or;
            source_rgb[nx * CHANNELS + ny * image_width * CHANNELS + 1] = og;
            source_rgb[nx * CHANNELS + ny * image_width * CHANNELS + 2] = ob;

            let nx = point[0].min(image_width - 1);
            let ny = point[1].saturating_sub(1).min(image_height - 1);

            source_rgb[nx * CHANNELS + ny * image_width * CHANNELS] = or;
            source_rgb[nx * CHANNELS + ny * image_width * CHANNELS + 1] = og;
            source_rgb[nx * CHANNELS + ny * image_width * CHANNELS + 2] = ob;

            let nx = point[0].saturating_sub(1).min(image_width - 1);
            let ny = point[1].min(image_height - 1);

            source_rgb[nx * CHANNELS + ny * image_width * CHANNELS] = or;
            source_rgb[nx * CHANNELS + ny * image_width * CHANNELS + 1] = og;
            source_rgb[nx * CHANNELS + ny * image_width * CHANNELS + 2] = ob;
        }

        let mut planar_image = YuvBiPlanarImageMut::<u8>::alloc(
            image_width as u32,
            image_height as u32,
            YuvChromaSubsampling::Yuv420,
        );

        rgb_to_yuv_nv12(
            &mut planar_image,
            &source_rgb,
            image_width as u32 * CHANNELS as u32,
            YuvRange::Limited,
            YuvStandardMatrix::Bt709,
        )
        .unwrap();

        let mut dest_rgb = vec![0u8; image_width * image_height * CHANNELS];

        let fixed_planar = planar_image.to_fixed();

        yuv_nv12_to_rgb(
            &fixed_planar,
            &mut dest_rgb,
            image_width as u32 * CHANNELS as u32,
            YuvRange::Limited,
            YuvStandardMatrix::Bt709,
        )
        .unwrap();

        for point in &pixel_points {
            let x = point[0];
            let y = point[1];
            let px = x * CHANNELS + y * image_width * CHANNELS;

            let r = dest_rgb[px];
            let g = dest_rgb[px + 1];
            let b = dest_rgb[px + 2];

            let diff_r = r as i32 - or as i32;
            let diff_g = g as i32 - og as i32;
            let diff_b = b as i32 - ob as i32;

            assert!(
                diff_r <= 55,
                "Actual diff {}, Original RGB {:?}, Round-tripped RGB {:?}",
                diff_r,
                [or, og, ob],
                [r, g, b]
            );
            assert!(
                diff_g <= 55,
                "Actual diff {}, Original RGB {:?}, Round-tripped RGB {:?}",
                diff_g,
                [or, og, ob],
                [r, g, b]
            );
            assert!(
                diff_b <= 55,
                "Actual diff {}, Original RGB {:?}, Round-tripped RGB {:?}",
                diff_b,
                [or, og, ob],
                [r, g, b]
            );
        }
    }
}
