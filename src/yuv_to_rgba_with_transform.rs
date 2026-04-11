/*
 * Copyright (c) Meta Platforms, Inc., 4/2026. All rights reserved.
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
//! YUV420 to RGBA/RGB conversion with pre-computed inverse transform coefficients.
//!
//! These functions bypass the runtime `search_inverse_transform()` lookup,
//! accepting caller-supplied `CbCrInverseTransform<i32>` and `YuvChromaRange`
//! directly. The SIMD handlers are identical to the standard path.

use crate::internals::ProcessedOffset;
use crate::numerics::qrshr;
use crate::yuv_error::check_rgba_destination;
#[allow(unused_imports)]
use crate::yuv_support::*;
use crate::{YuvError, YuvPlanarImage};

#[allow(unused_variables)]
#[inline(always)]
unsafe fn dispatch_inv_420<const DESTINATION_CHANNELS: u8, const PRECISION: i32>(
    range: &YuvChromaRange,
    transform: &CbCrInverseTransform<i32>,
    y_plane0: &[u8],
    y_plane1: &[u8],
    u_plane: &[u8],
    v_plane: &[u8],
    rgba0: &mut [u8],
    rgba1: &mut [u8],
    start_cx: usize,
    start_ux: usize,
    width: usize,
) -> ProcessedOffset {
    if PRECISION != 13 {
        return ProcessedOffset {
            cx: start_cx,
            ux: start_ux,
        };
    }
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        #[cfg(feature = "rdm")]
        {
            if std::arch::is_aarch64_feature_detected!("rdm") {
                use crate::neon::neon_yuv_to_rgba_row_rdm420;
                return neon_yuv_to_rgba_row_rdm420::<DESTINATION_CHANNELS>(
                    range, transform, y_plane0, y_plane1, u_plane, v_plane, rgba0, rgba1, start_cx,
                    start_ux, width,
                );
            }
        }
        use crate::neon::neon_yuv_to_rgba_row420;
        neon_yuv_to_rgba_row420::<DESTINATION_CHANNELS>(
            range, transform, y_plane0, y_plane1, u_plane, v_plane, rgba0, rgba1, start_cx,
            start_ux, width,
        )
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        #[cfg(feature = "nightly_avx512")]
        {
            if std::arch::is_x86_feature_detected!("avx512bw") {
                use crate::avx512bw::avx512_yuv_to_rgba420;
                let use_vbmi = std::arch::is_x86_feature_detected!("avx512vbmi");
                return if use_vbmi {
                    avx512_yuv_to_rgba420::<DESTINATION_CHANNELS, true>(
                        range, transform, y_plane0, y_plane1, u_plane, v_plane, rgba0, rgba1,
                        start_cx, start_ux, width,
                    )
                } else {
                    avx512_yuv_to_rgba420::<DESTINATION_CHANNELS, false>(
                        range, transform, y_plane0, y_plane1, u_plane, v_plane, rgba0, rgba1,
                        start_cx, start_ux, width,
                    )
                };
            }
        }
        #[cfg(feature = "avx")]
        if std::arch::is_x86_feature_detected!("avx2") {
            use crate::avx2::avx2_yuv_to_rgba_row420;
            return avx2_yuv_to_rgba_row420::<DESTINATION_CHANNELS>(
                range, transform, y_plane0, y_plane1, u_plane, v_plane, rgba0, rgba1, start_cx,
                start_ux, width,
            );
        }
        #[cfg(feature = "sse")]
        if std::arch::is_x86_feature_detected!("sse4.1") {
            use crate::sse::sse_yuv_to_rgba_row420;
            return sse_yuv_to_rgba_row420::<DESTINATION_CHANNELS>(
                range, transform, y_plane0, y_plane1, u_plane, v_plane, rgba0, rgba1, start_cx,
                start_ux, width,
            );
        }
    }
    #[cfg(not(any(all(target_arch = "aarch64", target_feature = "neon"),)))]
    ProcessedOffset {
        cx: start_cx,
        ux: start_ux,
    }
}

#[allow(unused_variables)]
#[inline(always)]
unsafe fn dispatch_inv_row<const DESTINATION_CHANNELS: u8, const PRECISION: i32>(
    range: &YuvChromaRange,
    transform: &CbCrInverseTransform<i32>,
    y_plane: &[u8],
    u_plane: &[u8],
    v_plane: &[u8],
    rgba: &mut [u8],
    start_cx: usize,
    start_ux: usize,
    width: usize,
) -> ProcessedOffset {
    if PRECISION != 13 {
        return ProcessedOffset {
            cx: start_cx,
            ux: start_ux,
        };
    }
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        #[cfg(feature = "rdm")]
        {
            if std::arch::is_aarch64_feature_detected!("rdm") {
                use crate::neon::neon_yuv_to_rgba_row_rdm;
                return neon_yuv_to_rgba_row_rdm::<
                    DESTINATION_CHANNELS,
                    { YuvChromaSubsampling::Yuv420 as u8 },
                >(
                    range, transform, y_plane, u_plane, v_plane, rgba, start_cx, start_ux, width,
                );
            }
        }
        use crate::neon::neon_yuv_to_rgba_row;
        neon_yuv_to_rgba_row::<DESTINATION_CHANNELS, { YuvChromaSubsampling::Yuv420 as u8 }>(
            range, transform, y_plane, u_plane, v_plane, rgba, start_cx, start_ux, width,
        )
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        #[cfg(feature = "nightly_avx512")]
        {
            if std::arch::is_x86_feature_detected!("avx512bw") {
                use crate::avx512bw::avx512_yuv_to_rgba422;
                let use_vbmi = std::arch::is_x86_feature_detected!("avx512vbmi");
                return if use_vbmi {
                    avx512_yuv_to_rgba422::<DESTINATION_CHANNELS, true>(
                        range, transform, y_plane, u_plane, v_plane, rgba, start_cx, start_ux,
                        width,
                    )
                } else {
                    avx512_yuv_to_rgba422::<DESTINATION_CHANNELS, false>(
                        range, transform, y_plane, u_plane, v_plane, rgba, start_cx, start_ux,
                        width,
                    )
                };
            }
        }
        #[cfg(feature = "avx")]
        if std::arch::is_x86_feature_detected!("avx2") {
            use crate::avx2::avx2_yuv_to_rgba_row422;
            return avx2_yuv_to_rgba_row422::<DESTINATION_CHANNELS>(
                range, transform, y_plane, u_plane, v_plane, rgba, start_cx, start_ux, width,
            );
        }
        #[cfg(feature = "sse")]
        if std::arch::is_x86_feature_detected!("sse4.1") {
            use crate::sse::sse_yuv_to_rgba_row422;
            return sse_yuv_to_rgba_row422::<DESTINATION_CHANNELS>(
                range, transform, y_plane, u_plane, v_plane, rgba, start_cx, start_ux, width,
            );
        }
    }
    #[cfg(not(any(all(target_arch = "aarch64", target_feature = "neon"),)))]
    ProcessedOffset {
        cx: start_cx,
        ux: start_ux,
    }
}

fn yuv420_to_rgbx_with_transform_impl<const DESTINATION_CHANNELS: u8, const PRECISION: i32>(
    image: &YuvPlanarImage<u8>,
    rgba: &mut [u8],
    rgba_stride: u32,
    inverse_transform: &CbCrInverseTransform<i32>,
    chroma_range: &YuvChromaRange,
) -> Result<(), YuvError> {
    let dst_chans: YuvSourceChannels = DESTINATION_CHANNELS.into();
    let channels = dst_chans.get_channels_count();

    check_rgba_destination(rgba, rgba_stride, image.width, image.height, channels)?;
    image.check_constraints(YuvChromaSubsampling::Yuv420)?;

    let cr_coef = inverse_transform.cr_coef;
    let cb_coef = inverse_transform.cb_coef;
    let y_coef = inverse_transform.y_coef;
    let g_coef_1 = inverse_transform.g_coeff_1;
    let g_coef_2 = inverse_transform.g_coeff_2;

    let bias_y = chroma_range.bias_y as i32;
    let bias_uv = chroma_range.bias_uv as i32;

    const BIT_DEPTH: usize = 8;

    let process_halved_chroma_row =
        |y_plane: &[u8], u_plane: &[u8], v_plane: &[u8], rgba: &mut [u8]| {
            let cx = 0usize;

            if cx != image.width as usize {
                for (((rgba, y_src), &u_src), &v_src) in rgba
                    .chunks_exact_mut(channels * 2)
                    .zip(y_plane.chunks_exact(2))
                    .zip(u_plane.iter())
                    .zip(v_plane.iter())
                    .skip(cx / 2)
                {
                    let y_value0 = (y_src[0] as i32 - bias_y) * y_coef;
                    let cb_value = u_src as i32 - bias_uv;
                    let cr_value = v_src as i32 - bias_uv;

                    let r0 = qrshr::<PRECISION, BIT_DEPTH>(y_value0 + cr_coef * cr_value);
                    let b0 = qrshr::<PRECISION, BIT_DEPTH>(y_value0 + cb_coef * cb_value);
                    let g0 = qrshr::<PRECISION, BIT_DEPTH>(
                        y_value0 - g_coef_1 * cr_value - g_coef_2 * cb_value,
                    );

                    let rgba0 = &mut rgba[0..channels];
                    rgba0[dst_chans.get_r_channel_offset()] = r0 as u8;
                    rgba0[dst_chans.get_g_channel_offset()] = g0 as u8;
                    rgba0[dst_chans.get_b_channel_offset()] = b0 as u8;
                    if dst_chans.has_alpha() {
                        rgba0[dst_chans.get_a_channel_offset()] = 255u8;
                    }

                    let y_value1 = (y_src[1] as i32 - bias_y) * y_coef;
                    let r1 = qrshr::<PRECISION, BIT_DEPTH>(y_value1 + cr_coef * cr_value);
                    let b1 = qrshr::<PRECISION, BIT_DEPTH>(y_value1 + cb_coef * cb_value);
                    let g1 = qrshr::<PRECISION, BIT_DEPTH>(
                        y_value1 - g_coef_1 * cr_value - g_coef_2 * cb_value,
                    );

                    let rgba1 = &mut rgba[channels..channels * 2];
                    rgba1[dst_chans.get_r_channel_offset()] = r1 as u8;
                    rgba1[dst_chans.get_g_channel_offset()] = g1 as u8;
                    rgba1[dst_chans.get_b_channel_offset()] = b1 as u8;
                    if dst_chans.has_alpha() {
                        rgba1[dst_chans.get_a_channel_offset()] = 255u8;
                    }
                }

                if image.width & 1 != 0 {
                    let y_value0 = (*y_plane.last().unwrap() as i32 - bias_y) * y_coef;
                    let cb_value = *u_plane.last().unwrap() as i32 - bias_uv;
                    let cr_value = *v_plane.last().unwrap() as i32 - bias_uv;
                    let rgba = rgba.chunks_exact_mut(channels).last().unwrap();

                    rgba[dst_chans.get_r_channel_offset()] =
                        qrshr::<PRECISION, BIT_DEPTH>(y_value0 + cr_coef * cr_value) as u8;
                    rgba[dst_chans.get_g_channel_offset()] = qrshr::<PRECISION, BIT_DEPTH>(
                        y_value0 - g_coef_1 * cr_value - g_coef_2 * cb_value,
                    ) as u8;
                    rgba[dst_chans.get_b_channel_offset()] =
                        qrshr::<PRECISION, BIT_DEPTH>(y_value0 + cb_coef * cb_value) as u8;
                    if dst_chans.has_alpha() {
                        rgba[dst_chans.get_a_channel_offset()] = 255;
                    }
                }
            }
        };

    let process_doubled_chroma_row = |y_plane0: &[u8],
                                      y_plane1: &[u8],
                                      u_plane: &[u8],
                                      v_plane: &[u8],
                                      rgba0: &mut [u8],
                                      rgba1: &mut [u8]| {
        let cx = unsafe {
            dispatch_inv_420::<DESTINATION_CHANNELS, PRECISION>(
                chroma_range,
                inverse_transform,
                y_plane0,
                y_plane1,
                u_plane,
                v_plane,
                rgba0,
                rgba1,
                0,
                0,
                image.width as usize,
            )
        }
        .cx;

        if cx != image.width as usize {
            for (((((rgba0, rgba1), y_src0), y_src1), &u_src), &v_src) in rgba0
                .chunks_exact_mut(channels * 2)
                .zip(rgba1.chunks_exact_mut(channels * 2))
                .zip(y_plane0.chunks_exact(2))
                .zip(y_plane1.chunks_exact(2))
                .zip(u_plane.iter())
                .zip(v_plane.iter())
                .skip(cx / 2)
            {
                let cb_value = u_src as i32 - bias_uv;
                let cr_value = v_src as i32 - bias_uv;
                let g_built = -g_coef_1 * cr_value - g_coef_2 * cb_value;

                let y_value0 = (y_src0[0] as i32 - bias_y) * y_coef;
                rgba0[dst_chans.get_r_channel_offset()] =
                    qrshr::<PRECISION, BIT_DEPTH>(y_value0 + cr_coef * cr_value) as u8;
                rgba0[dst_chans.get_g_channel_offset()] =
                    qrshr::<PRECISION, BIT_DEPTH>(y_value0 + g_built) as u8;
                rgba0[dst_chans.get_b_channel_offset()] =
                    qrshr::<PRECISION, BIT_DEPTH>(y_value0 + cb_coef * cb_value) as u8;
                if dst_chans.has_alpha() {
                    rgba0[dst_chans.get_a_channel_offset()] = 255u8;
                }

                let y_value1 = (y_src0[1] as i32 - bias_y) * y_coef;
                rgba0[channels + dst_chans.get_r_channel_offset()] =
                    qrshr::<PRECISION, BIT_DEPTH>(y_value1 + cr_coef * cr_value) as u8;
                rgba0[channels + dst_chans.get_g_channel_offset()] =
                    qrshr::<PRECISION, BIT_DEPTH>(y_value1 + g_built) as u8;
                rgba0[channels + dst_chans.get_b_channel_offset()] =
                    qrshr::<PRECISION, BIT_DEPTH>(y_value1 + cb_coef * cb_value) as u8;
                if dst_chans.has_alpha() {
                    rgba0[channels + dst_chans.get_a_channel_offset()] = 255u8;
                }

                let y_value10 = (y_src1[0] as i32 - bias_y) * y_coef;
                rgba1[dst_chans.get_r_channel_offset()] =
                    qrshr::<PRECISION, BIT_DEPTH>(y_value10 + cr_coef * cr_value) as u8;
                rgba1[dst_chans.get_g_channel_offset()] =
                    qrshr::<PRECISION, BIT_DEPTH>(y_value10 + g_built) as u8;
                rgba1[dst_chans.get_b_channel_offset()] =
                    qrshr::<PRECISION, BIT_DEPTH>(y_value10 + cb_coef * cb_value) as u8;
                if dst_chans.has_alpha() {
                    rgba1[dst_chans.get_a_channel_offset()] = 255u8;
                }

                let y_value11 = (y_src1[1] as i32 - bias_y) * y_coef;
                rgba1[channels + dst_chans.get_r_channel_offset()] =
                    qrshr::<PRECISION, BIT_DEPTH>(y_value11 + cr_coef * cr_value) as u8;
                rgba1[channels + dst_chans.get_g_channel_offset()] =
                    qrshr::<PRECISION, BIT_DEPTH>(y_value11 + g_built) as u8;
                rgba1[channels + dst_chans.get_b_channel_offset()] =
                    qrshr::<PRECISION, BIT_DEPTH>(y_value11 + cb_coef * cb_value) as u8;
                if dst_chans.has_alpha() {
                    rgba1[channels + dst_chans.get_a_channel_offset()] = 255u8;
                }
            }

            if image.width & 1 != 0 {
                let y_value0 = (*y_plane0.last().unwrap() as i32 - bias_y) * y_coef;
                let y_value1 = (*y_plane1.last().unwrap() as i32 - bias_y) * y_coef;
                let cb_value = *u_plane.last().unwrap() as i32 - bias_uv;
                let cr_value = *v_plane.last().unwrap() as i32 - bias_uv;
                let g_built = -g_coef_1 * cr_value - g_coef_2 * cb_value;

                let r0 = rgba0.chunks_exact_mut(channels).last().unwrap();
                r0[dst_chans.get_r_channel_offset()] =
                    qrshr::<PRECISION, BIT_DEPTH>(y_value0 + cr_coef * cr_value) as u8;
                r0[dst_chans.get_g_channel_offset()] =
                    qrshr::<PRECISION, BIT_DEPTH>(y_value0 + g_built) as u8;
                r0[dst_chans.get_b_channel_offset()] =
                    qrshr::<PRECISION, BIT_DEPTH>(y_value0 + cb_coef * cb_value) as u8;
                if dst_chans.has_alpha() {
                    r0[dst_chans.get_a_channel_offset()] = 255;
                }

                let r1 = rgba1.chunks_exact_mut(channels).last().unwrap();
                r1[dst_chans.get_r_channel_offset()] =
                    qrshr::<PRECISION, BIT_DEPTH>(y_value1 + cr_coef * cr_value) as u8;
                r1[dst_chans.get_g_channel_offset()] =
                    qrshr::<PRECISION, BIT_DEPTH>(y_value1 + g_built) as u8;
                r1[dst_chans.get_b_channel_offset()] =
                    qrshr::<PRECISION, BIT_DEPTH>(y_value1 + cb_coef * cb_value) as u8;
                if dst_chans.has_alpha() {
                    r1[dst_chans.get_a_channel_offset()] = 255;
                }
            }
        }
    };

    let iter = rgba
        .chunks_exact_mut(rgba_stride as usize * 2)
        .zip(image.y_plane.chunks_exact(image.y_stride as usize * 2))
        .zip(image.u_plane.chunks_exact(image.u_stride as usize))
        .zip(image.v_plane.chunks_exact(image.v_stride as usize));

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

    Ok(())
}

// Row-level conversion with pre-computed transform
//
// These avoid the frame-level `YuvPlanarImage` construction and validation
// overhead, accepting raw slices directly. Useful for streaming decoders
// that process one macroblock row at a time.

fn yuv420_row_pair_to_rgbx_with_transform_impl<const DESTINATION_CHANNELS: u8>(
    y0: &[u8],
    y1: &[u8],
    u: &[u8],
    v: &[u8],
    dst0: &mut [u8],
    dst1: &mut [u8],
    width: usize,
    inverse_transform: &CbCrInverseTransform<i32>,
    chroma_range: &YuvChromaRange,
) {
    let dst_chans: YuvSourceChannels = DESTINATION_CHANNELS.into();
    let channels = dst_chans.get_channels_count();

    let cr_coef = inverse_transform.cr_coef;
    let cb_coef = inverse_transform.cb_coef;
    let y_coef = inverse_transform.y_coef;
    let g_coef_1 = inverse_transform.g_coeff_1;
    let g_coef_2 = inverse_transform.g_coeff_2;
    let bias_y = chroma_range.bias_y as i32;
    let bias_uv = chroma_range.bias_uv as i32;
    const PRECISION: i32 = 13;
    const BIT_DEPTH: usize = 8;

    let cx = unsafe {
        dispatch_inv_420::<DESTINATION_CHANNELS, PRECISION>(
            chroma_range,
            inverse_transform,
            y0,
            y1,
            u,
            v,
            dst0,
            dst1,
            0,
            0,
            width,
        )
    }
    .cx;

    if cx != width {
        for (((((rgba0, rgba1), y_src0), y_src1), &u_src), &v_src) in dst0
            .chunks_exact_mut(channels * 2)
            .zip(dst1.chunks_exact_mut(channels * 2))
            .zip(y0.chunks_exact(2))
            .zip(y1.chunks_exact(2))
            .zip(u.iter())
            .zip(v.iter())
            .skip(cx / 2)
        {
            let cb_value = u_src as i32 - bias_uv;
            let cr_value = v_src as i32 - bias_uv;
            let g_built = -g_coef_1 * cr_value - g_coef_2 * cb_value;

            let y_value0 = (y_src0[0] as i32 - bias_y) * y_coef;
            rgba0[dst_chans.get_r_channel_offset()] =
                qrshr::<PRECISION, BIT_DEPTH>(y_value0 + cr_coef * cr_value) as u8;
            rgba0[dst_chans.get_g_channel_offset()] =
                qrshr::<PRECISION, BIT_DEPTH>(y_value0 + g_built) as u8;
            rgba0[dst_chans.get_b_channel_offset()] =
                qrshr::<PRECISION, BIT_DEPTH>(y_value0 + cb_coef * cb_value) as u8;
            if dst_chans.has_alpha() {
                rgba0[dst_chans.get_a_channel_offset()] = 255u8;
            }

            let y_value1 = (y_src0[1] as i32 - bias_y) * y_coef;
            rgba0[channels + dst_chans.get_r_channel_offset()] =
                qrshr::<PRECISION, BIT_DEPTH>(y_value1 + cr_coef * cr_value) as u8;
            rgba0[channels + dst_chans.get_g_channel_offset()] =
                qrshr::<PRECISION, BIT_DEPTH>(y_value1 + g_built) as u8;
            rgba0[channels + dst_chans.get_b_channel_offset()] =
                qrshr::<PRECISION, BIT_DEPTH>(y_value1 + cb_coef * cb_value) as u8;
            if dst_chans.has_alpha() {
                rgba0[channels + dst_chans.get_a_channel_offset()] = 255u8;
            }

            let y_value10 = (y_src1[0] as i32 - bias_y) * y_coef;
            rgba1[dst_chans.get_r_channel_offset()] =
                qrshr::<PRECISION, BIT_DEPTH>(y_value10 + cr_coef * cr_value) as u8;
            rgba1[dst_chans.get_g_channel_offset()] =
                qrshr::<PRECISION, BIT_DEPTH>(y_value10 + g_built) as u8;
            rgba1[dst_chans.get_b_channel_offset()] =
                qrshr::<PRECISION, BIT_DEPTH>(y_value10 + cb_coef * cb_value) as u8;
            if dst_chans.has_alpha() {
                rgba1[dst_chans.get_a_channel_offset()] = 255u8;
            }

            let y_value11 = (y_src1[1] as i32 - bias_y) * y_coef;
            rgba1[channels + dst_chans.get_r_channel_offset()] =
                qrshr::<PRECISION, BIT_DEPTH>(y_value11 + cr_coef * cr_value) as u8;
            rgba1[channels + dst_chans.get_g_channel_offset()] =
                qrshr::<PRECISION, BIT_DEPTH>(y_value11 + g_built) as u8;
            rgba1[channels + dst_chans.get_b_channel_offset()] =
                qrshr::<PRECISION, BIT_DEPTH>(y_value11 + cb_coef * cb_value) as u8;
            if dst_chans.has_alpha() {
                rgba1[channels + dst_chans.get_a_channel_offset()] = 255u8;
            }
        }

        if width & 1 != 0 {
            let y_value0 = (*y0.last().unwrap() as i32 - bias_y) * y_coef;
            let y_value1 = (*y1.last().unwrap() as i32 - bias_y) * y_coef;
            let cb_value = *u.last().unwrap() as i32 - bias_uv;
            let cr_value = *v.last().unwrap() as i32 - bias_uv;
            let g_built = -g_coef_1 * cr_value - g_coef_2 * cb_value;

            let r0 = dst0.chunks_exact_mut(channels).last().unwrap();
            r0[dst_chans.get_r_channel_offset()] =
                qrshr::<PRECISION, BIT_DEPTH>(y_value0 + cr_coef * cr_value) as u8;
            r0[dst_chans.get_g_channel_offset()] =
                qrshr::<PRECISION, BIT_DEPTH>(y_value0 + g_built) as u8;
            r0[dst_chans.get_b_channel_offset()] =
                qrshr::<PRECISION, BIT_DEPTH>(y_value0 + cb_coef * cb_value) as u8;
            if dst_chans.has_alpha() {
                r0[dst_chans.get_a_channel_offset()] = 255;
            }

            let r1 = dst1.chunks_exact_mut(channels).last().unwrap();
            r1[dst_chans.get_r_channel_offset()] =
                qrshr::<PRECISION, BIT_DEPTH>(y_value1 + cr_coef * cr_value) as u8;
            r1[dst_chans.get_g_channel_offset()] =
                qrshr::<PRECISION, BIT_DEPTH>(y_value1 + g_built) as u8;
            r1[dst_chans.get_b_channel_offset()] =
                qrshr::<PRECISION, BIT_DEPTH>(y_value1 + cb_coef * cb_value) as u8;
            if dst_chans.has_alpha() {
                r1[dst_chans.get_a_channel_offset()] = 255;
            }
        }
    }
}

fn yuv420_single_row_to_rgbx_with_transform_impl<const DESTINATION_CHANNELS: u8>(
    y: &[u8],
    u: &[u8],
    v: &[u8],
    dst: &mut [u8],
    width: usize,
    inverse_transform: &CbCrInverseTransform<i32>,
    chroma_range: &YuvChromaRange,
) {
    let dst_chans: YuvSourceChannels = DESTINATION_CHANNELS.into();
    let channels = dst_chans.get_channels_count();

    let cr_coef = inverse_transform.cr_coef;
    let cb_coef = inverse_transform.cb_coef;
    let y_coef = inverse_transform.y_coef;
    let g_coef_1 = inverse_transform.g_coeff_1;
    let g_coef_2 = inverse_transform.g_coeff_2;
    let bias_y = chroma_range.bias_y as i32;
    let bias_uv = chroma_range.bias_uv as i32;
    const PRECISION: i32 = 13;
    const BIT_DEPTH: usize = 8;

    let cx = unsafe {
        dispatch_inv_row::<DESTINATION_CHANNELS, PRECISION>(
            chroma_range,
            inverse_transform,
            y,
            u,
            v,
            dst,
            0,
            0,
            width,
        )
    }
    .cx;

    if cx != width {
        for (((rgba, y_src), &u_src), &v_src) in dst
            .chunks_exact_mut(channels * 2)
            .zip(y.chunks_exact(2))
            .zip(u.iter())
            .zip(v.iter())
            .skip(cx / 2)
        {
            let y_value0 = (y_src[0] as i32 - bias_y) * y_coef;
            let cb_value = u_src as i32 - bias_uv;
            let cr_value = v_src as i32 - bias_uv;

            let r0 = qrshr::<PRECISION, BIT_DEPTH>(y_value0 + cr_coef * cr_value);
            let b0 = qrshr::<PRECISION, BIT_DEPTH>(y_value0 + cb_coef * cb_value);
            let g0 =
                qrshr::<PRECISION, BIT_DEPTH>(y_value0 - g_coef_1 * cr_value - g_coef_2 * cb_value);

            let rgba0 = &mut rgba[0..channels];
            rgba0[dst_chans.get_r_channel_offset()] = r0 as u8;
            rgba0[dst_chans.get_g_channel_offset()] = g0 as u8;
            rgba0[dst_chans.get_b_channel_offset()] = b0 as u8;
            if dst_chans.has_alpha() {
                rgba0[dst_chans.get_a_channel_offset()] = 255u8;
            }

            let y_value1 = (y_src[1] as i32 - bias_y) * y_coef;
            let r1 = qrshr::<PRECISION, BIT_DEPTH>(y_value1 + cr_coef * cr_value);
            let b1 = qrshr::<PRECISION, BIT_DEPTH>(y_value1 + cb_coef * cb_value);
            let g1 =
                qrshr::<PRECISION, BIT_DEPTH>(y_value1 - g_coef_1 * cr_value - g_coef_2 * cb_value);

            let rgba1 = &mut rgba[channels..channels * 2];
            rgba1[dst_chans.get_r_channel_offset()] = r1 as u8;
            rgba1[dst_chans.get_g_channel_offset()] = g1 as u8;
            rgba1[dst_chans.get_b_channel_offset()] = b1 as u8;
            if dst_chans.has_alpha() {
                rgba1[dst_chans.get_a_channel_offset()] = 255u8;
            }
        }

        if width & 1 != 0 {
            let y_value0 = (*y.last().unwrap() as i32 - bias_y) * y_coef;
            let cb_value = *u.last().unwrap() as i32 - bias_uv;
            let cr_value = *v.last().unwrap() as i32 - bias_uv;
            let rgba = dst.chunks_exact_mut(channels).last().unwrap();

            rgba[dst_chans.get_r_channel_offset()] =
                qrshr::<PRECISION, BIT_DEPTH>(y_value0 + cr_coef * cr_value) as u8;
            rgba[dst_chans.get_g_channel_offset()] =
                qrshr::<PRECISION, BIT_DEPTH>(y_value0 - g_coef_1 * cr_value - g_coef_2 * cb_value)
                    as u8;
            rgba[dst_chans.get_b_channel_offset()] =
                qrshr::<PRECISION, BIT_DEPTH>(y_value0 + cb_coef * cb_value) as u8;
            if dst_chans.has_alpha() {
                rgba[dst_chans.get_a_channel_offset()] = 255;
            }
        }
    }
}

macro_rules! build_yuv420_row_pair_with_transform {
    ($method:ident, $px_fmt:expr, $px_name:expr, $channels:expr) => {
        #[doc = concat!("Convert a pair of I420 rows to ", $px_name, " using pre-computed inverse transform.")]
        pub fn $method(
            y0: &[u8],
            y1: &[u8],
            u: &[u8],
            v: &[u8],
            dst0: &mut [u8],
            dst1: &mut [u8],
            width: usize,
            inverse_transform: &CbCrInverseTransform<i32>,
            chroma_range: &YuvChromaRange,
        ) {
            yuv420_row_pair_to_rgbx_with_transform_impl::<{ $px_fmt as u8 }>(
                y0, y1, u, v, dst0, dst1, width, inverse_transform, chroma_range,
            );
        }
    };
}

build_yuv420_row_pair_with_transform!(
    yuv420_row_pair_to_rgba_with_transform,
    YuvSourceChannels::Rgba,
    "RGBA",
    4
);
build_yuv420_row_pair_with_transform!(
    yuv420_row_pair_to_rgb_with_transform,
    YuvSourceChannels::Rgb,
    "RGB",
    3
);
build_yuv420_row_pair_with_transform!(
    yuv420_row_pair_to_bgra_with_transform,
    YuvSourceChannels::Bgra,
    "BGRA",
    4
);
build_yuv420_row_pair_with_transform!(
    yuv420_row_pair_to_bgr_with_transform,
    YuvSourceChannels::Bgr,
    "BGR",
    3
);

macro_rules! build_yuv420_single_row_with_transform {
    ($method:ident, $px_fmt:expr, $px_name:expr) => {
        #[doc = concat!("Convert a single I420 row to ", $px_name, " using pre-computed inverse transform.")]
        pub fn $method(
            y: &[u8],
            u: &[u8],
            v: &[u8],
            dst: &mut [u8],
            width: usize,
            inverse_transform: &CbCrInverseTransform<i32>,
            chroma_range: &YuvChromaRange,
        ) {
            yuv420_single_row_to_rgbx_with_transform_impl::<{ $px_fmt as u8 }>(
                y, u, v, dst, width, inverse_transform, chroma_range,
            );
        }
    };
}

build_yuv420_single_row_with_transform!(
    yuv420_single_row_to_rgba_with_transform,
    YuvSourceChannels::Rgba,
    "RGBA"
);
build_yuv420_single_row_with_transform!(
    yuv420_single_row_to_rgb_with_transform,
    YuvSourceChannels::Rgb,
    "RGB"
);
build_yuv420_single_row_with_transform!(
    yuv420_single_row_to_bgra_with_transform,
    YuvSourceChannels::Bgra,
    "BGRA"
);
build_yuv420_single_row_with_transform!(
    yuv420_single_row_to_bgr_with_transform,
    YuvSourceChannels::Bgr,
    "BGR"
);

macro_rules! build_yuv420_frame_with_transform {
    ($method:ident, $px_fmt:expr, $px_name:expr, $px_small:expr) => {
        #[doc = concat!("Convert YUV 420 to ", $px_name, " using pre-computed inverse transform coefficients.")]
        pub fn $method(
            planar_image: &YuvPlanarImage<u8>,
            dst: &mut [u8],
            dst_stride: u32,
            config: &YuvInverseTransform,
        ) -> Result<(), YuvError> {
            match config.mode {
                #[cfg(feature = "fast_mode")]
                YuvConversionMode::Fast => {
                    yuv420_to_rgbx_with_transform_impl::<{ $px_fmt as u8 }, 7>(
                        planar_image, dst, dst_stride, &config.inverse_transform, &config.chroma_range,
                    )
                }
                YuvConversionMode::Balanced => {
                    yuv420_to_rgbx_with_transform_impl::<{ $px_fmt as u8 }, 13>(
                        planar_image, dst, dst_stride, &config.inverse_transform, &config.chroma_range,
                    )
                }
                #[cfg(feature = "professional_mode")]
                YuvConversionMode::Professional => {
                    yuv420_to_rgbx_with_transform_impl::<{ $px_fmt as u8 }, 16>(
                        planar_image, dst, dst_stride, &config.inverse_transform, &config.chroma_range,
                    )
                }
            }
        }
    };
}

build_yuv420_frame_with_transform!(
    yuv420_to_rgba_with_transform,
    YuvSourceChannels::Rgba,
    "RGBA",
    "rgba"
);
build_yuv420_frame_with_transform!(
    yuv420_to_rgb_with_transform,
    YuvSourceChannels::Rgb,
    "RGB",
    "rgb"
);
