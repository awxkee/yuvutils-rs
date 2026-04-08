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
//! RGBA/RGB to YUV420 conversion with pre-computed transform coefficients.
//!
//! These functions bypass the runtime `search_forward_transform()` lookup,
//! accepting caller-supplied `CbCrForwardTransform<i32>` and `YuvChromaRange`
//! directly. The SIMD handlers are identical to the standard path.

#[allow(unused_imports)]
use crate::internals::*;
use crate::yuv_error::check_rgba_destination;
#[allow(unused_imports)]
use crate::yuv_support::*;
use crate::{YuvError, YuvPlanarImageMut};

#[allow(unused_variables)]
#[inline(always)]
unsafe fn dispatch_fwd_420<const ORIGIN_CHANNELS: u8, const PRECISION: i32>(
    transform: &CbCrForwardTransform<i32>,
    range: &YuvChromaRange,
    y_plane0: &mut [u8],
    y_plane1: &mut [u8],
    u_plane: &mut [u8],
    v_plane: &mut [u8],
    rgba0: &[u8],
    rgba1: &[u8],
    start_cx: usize,
    start_ux: usize,
    width: usize,
) -> ProcessedOffset {
    #[cfg(feature = "fast_mode")]
    if PRECISION == 7 {
        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        {
            #[cfg(feature = "nightly_i8mm")]
            {
                if std::arch::is_aarch64_feature_detected!("i8mm") {
                    let chans: YuvSourceChannels = ORIGIN_CHANNELS.into();
                    if chans == YuvSourceChannels::Rgba || chans == YuvSourceChannels::Bgra {
                        use crate::neon::neon_rgba_to_yuv_dot_rgba420;
                        return neon_rgba_to_yuv_dot_rgba420::<ORIGIN_CHANNELS>(
                            transform, range, y_plane0, y_plane1, u_plane, v_plane, rgba0, rgba1,
                            start_cx, start_ux, width,
                        );
                    }
                }
            }
            use crate::neon::neon_rgbx_to_yuv_fast420;
            return neon_rgbx_to_yuv_fast420::<ORIGIN_CHANNELS>(
                transform, range, y_plane0, y_plane1, u_plane, v_plane, rgba0, rgba1, start_cx,
                start_ux, width,
            );
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            #[cfg(feature = "nightly_avx512")]
            if std::arch::is_x86_feature_detected!("avx512bw") {
                use crate::avx512bw::{
                    avx512_rgba_to_yuv_dot_rgba420, avx512_rgba_to_yuv_dot_rgba420_vbmi,
                };
                let chans: YuvSourceChannels = ORIGIN_CHANNELS.into();
                if chans == YuvSourceChannels::Rgba || chans == YuvSourceChannels::Bgra {
                    return avx512_rgba_to_yuv_dot_rgba420::<ORIGIN_CHANNELS>(
                        transform, range, y_plane0, y_plane1, u_plane, v_plane, rgba0, rgba1,
                        start_cx, start_ux, width,
                    );
                }
                let has_vbmi = std::arch::is_x86_feature_detected!("avx512vbmi");
                if (chans == YuvSourceChannels::Rgb || chans == YuvSourceChannels::Bgr) && has_vbmi
                {
                    return avx512_rgba_to_yuv_dot_rgba420_vbmi::<ORIGIN_CHANNELS>(
                        transform, range, y_plane0, y_plane1, u_plane, v_plane, rgba0, rgba1,
                        start_cx, start_ux, width,
                    );
                }
            }

            #[cfg(feature = "avx")]
            if std::arch::is_x86_feature_detected!("avx2") {
                use crate::avx2::avx2_rgba_to_yuv_dot_rgba420;
                return avx2_rgba_to_yuv_dot_rgba420::<ORIGIN_CHANNELS>(
                    transform, range, y_plane0, y_plane1, u_plane, v_plane, rgba0, rgba1, start_cx,
                    start_ux, width,
                );
            }

            #[cfg(feature = "sse")]
            if std::arch::is_x86_feature_detected!("sse4.1") {
                use crate::sse::sse_rgba_to_yuv_dot_rgba420;
                return sse_rgba_to_yuv_dot_rgba420::<ORIGIN_CHANNELS>(
                    transform, range, y_plane0, y_plane1, u_plane, v_plane, rgba0, rgba1, start_cx,
                    start_ux, width,
                );
            }
        }
    }

    if PRECISION == 16 {
        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        {
            #[cfg(feature = "professional_mode")]
            {
                use crate::neon::neon_rgba_to_yuv_prof420;
                return neon_rgba_to_yuv_prof420::<ORIGIN_CHANNELS>(
                    transform, range, y_plane0, y_plane1, u_plane, v_plane, rgba0, rgba1, start_cx,
                    start_ux, width,
                );
            }
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            #[cfg(feature = "avx")]
            if std::arch::is_x86_feature_detected!("avx2") {
                use crate::avx2::avx2_rgba_to_yuv420_p16;
                return avx2_rgba_to_yuv420_p16::<ORIGIN_CHANNELS>(
                    transform, range, y_plane0, y_plane1, u_plane, v_plane, rgba0, rgba1, start_cx,
                    start_ux, width,
                );
            }
        }
    }

    if PRECISION == 13 {
        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        {
            #[cfg(feature = "rdm")]
            {
                use crate::neon::neon_rgba_to_yuv_rdm420;
                if std::arch::is_aarch64_feature_detected!("rdm") {
                    return neon_rgba_to_yuv_rdm420::<ORIGIN_CHANNELS>(
                        transform, range, y_plane0, y_plane1, u_plane, v_plane, rgba0, rgba1,
                        start_cx, start_ux, width,
                    );
                }
            }
            use crate::neon::neon_rgba_to_yuv420;
            return neon_rgba_to_yuv420::<ORIGIN_CHANNELS>(
                transform, range, y_plane0, y_plane1, u_plane, v_plane, rgba0, rgba1, start_cx,
                start_ux, width,
            );
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            #[cfg(feature = "nightly_avx512")]
            {
                if std::arch::is_x86_feature_detected!("avx512bw") {
                    use crate::avx512bw::avx512_rgba_to_yuv420;
                    let use_vbmi = std::arch::is_x86_feature_detected!("avx512vbmi");
                    return if use_vbmi {
                        avx512_rgba_to_yuv420::<ORIGIN_CHANNELS, true>(
                            transform, range, y_plane0, y_plane1, u_plane, v_plane, rgba0, rgba1,
                            start_cx, start_ux, width,
                        )
                    } else {
                        avx512_rgba_to_yuv420::<ORIGIN_CHANNELS, false>(
                            transform, range, y_plane0, y_plane1, u_plane, v_plane, rgba0, rgba1,
                            start_cx, start_ux, width,
                        )
                    };
                }
            }
            #[cfg(feature = "avx")]
            if std::arch::is_x86_feature_detected!("avx2") {
                use crate::avx2::avx2_rgba_to_yuv420;
                return avx2_rgba_to_yuv420::<ORIGIN_CHANNELS, 13>(
                    transform, range, y_plane0, y_plane1, u_plane, v_plane, rgba0, rgba1, start_cx,
                    start_ux, width,
                );
            }
            #[cfg(feature = "sse")]
            if std::arch::is_x86_feature_detected!("sse4.1") {
                use crate::sse::sse_rgba_to_yuv_row420;
                return sse_rgba_to_yuv_row420::<ORIGIN_CHANNELS, 13>(
                    transform, range, y_plane0, y_plane1, u_plane, v_plane, rgba0, rgba1, start_cx,
                    start_ux, width,
                );
            }
        }
    }

    ProcessedOffset {
        cx: start_cx,
        ux: start_ux,
    }
}

fn rgbx_to_yuv420_with_transform_impl<const ORIGIN_CHANNELS: u8, const PRECISION: i32>(
    image: &mut YuvPlanarImageMut<u8>,
    rgba: &[u8],
    rgba_stride: u32,
    transform: &CbCrForwardTransform<i32>,
    chroma_range: &YuvChromaRange,
) -> Result<(), YuvError> {
    let src_chans: YuvSourceChannels = ORIGIN_CHANNELS.into();
    let channels = src_chans.get_channels_count();

    check_rgba_destination(rgba, rgba_stride, image.width, image.height, channels)?;
    image.check_constraints(YuvChromaSubsampling::Yuv420)?;

    let rounding_const_bias: i32 = (1 << (PRECISION - 1)) - 1;
    let bias_y = chroma_range.bias_y as i32 * (1 << PRECISION) + rounding_const_bias;
    let bias_uv = chroma_range.bias_uv as i32 * (1 << PRECISION) + rounding_const_bias;

    let process_halved_chroma_row = |y_plane: &mut [u8],
                                     u_plane: &mut [u8],
                                     v_plane: &mut [u8],
                                     rgba: &[u8]| {
        let cx = 0usize;

        if cx != image.width as usize {
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
                let y_0 = (r0 * transform.yr + g0 * transform.yg + b0 * transform.yb + bias_y)
                    >> PRECISION;
                y_dst[0] = y_0 as u8;

                let src1 = &rgba[channels..channels * 2];
                let r1 = src1[src_chans.get_r_channel_offset()] as i32;
                let g1 = src1[src_chans.get_g_channel_offset()] as i32;
                let b1 = src1[src_chans.get_b_channel_offset()] as i32;
                let y_1 = (r1 * transform.yr + g1 * transform.yg + b1 * transform.yb + bias_y)
                    >> PRECISION;
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

                *y_plane.last_mut().unwrap() =
                    ((r0 * transform.yr + g0 * transform.yg + b0 * transform.yb + bias_y)
                        >> PRECISION) as u8;
                *u_plane.last_mut().unwrap() =
                    ((r0 * transform.cb_r + g0 * transform.cb_g + b0 * transform.cb_b + bias_uv)
                        >> PRECISION) as u8;
                *v_plane.last_mut().unwrap() =
                    ((r0 * transform.cr_r + g0 * transform.cr_g + b0 * transform.cr_b + bias_uv)
                        >> PRECISION) as u8;
            }
        }
    };

    let process_doubled_row = |y_plane0: &mut [u8],
                               y_plane1: &mut [u8],
                               u_plane: &mut [u8],
                               v_plane: &mut [u8],
                               rgba0: &[u8],
                               rgba1: &[u8]| {
        let cx = unsafe {
            dispatch_fwd_420::<ORIGIN_CHANNELS, PRECISION>(
                transform,
                chroma_range,
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
                y_dst0[0] =
                    ((r00 * transform.yr + g00 * transform.yg + b00 * transform.yb + bias_y)
                        >> PRECISION) as u8;

                let src01 = &rgba0[channels..channels * 2];
                let r01 = src01[src_chans.get_r_channel_offset()] as i32;
                let g01 = src01[src_chans.get_g_channel_offset()] as i32;
                let b01 = src01[src_chans.get_b_channel_offset()] as i32;
                y_dst0[1] =
                    ((r01 * transform.yr + g01 * transform.yg + b01 * transform.yb + bias_y)
                        >> PRECISION) as u8;

                let src10 = &rgba1[0..channels];
                let r10 = src10[src_chans.get_r_channel_offset()] as i32;
                let g10 = src10[src_chans.get_g_channel_offset()] as i32;
                let b10 = src10[src_chans.get_b_channel_offset()] as i32;
                y_dst1[0] =
                    ((r10 * transform.yr + g10 * transform.yg + b10 * transform.yb + bias_y)
                        >> PRECISION) as u8;

                let src11 = &rgba1[channels..channels * 2];
                let r11 = src11[src_chans.get_r_channel_offset()] as i32;
                let g11 = src11[src_chans.get_g_channel_offset()] as i32;
                let b11 = src11[src_chans.get_b_channel_offset()] as i32;
                y_dst1[1] =
                    ((r11 * transform.yr + g11 * transform.yg + b11 * transform.yb + bias_y)
                        >> PRECISION) as u8;

                let ruv = (r00 + r01 + r10 + r11 + 2) >> 2;
                let guv = (g00 + g01 + g10 + g11 + 2) >> 2;
                let buv = (b00 + b01 + b10 + b11 + 2) >> 2;
                *u_dst =
                    ((ruv * transform.cb_r + guv * transform.cb_g + buv * transform.cb_b + bias_uv)
                        >> PRECISION) as u8;
                *v_dst =
                    ((ruv * transform.cr_r + guv * transform.cr_g + buv * transform.cr_b + bias_uv)
                        >> PRECISION) as u8;
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

                *y_plane0.last_mut().unwrap() =
                    ((r0 * transform.yr + g0 * transform.yg + b0 * transform.yb + bias_y)
                        >> PRECISION) as u8;
                *y_plane1.last_mut().unwrap() =
                    ((r1 * transform.yr + g1 * transform.yg + b1 * transform.yb + bias_y)
                        >> PRECISION) as u8;

                let r = (r0 + r1) >> 1;
                let g = (g0 + g1) >> 1;
                let b = (b0 + b1) >> 1;
                *u_plane.last_mut().unwrap() =
                    ((r * transform.cb_r + g * transform.cb_g + b * transform.cb_b + bias_uv)
                        >> PRECISION) as u8;
                *v_plane.last_mut().unwrap() =
                    ((r * transform.cr_r + g * transform.cr_g + b * transform.cr_b + bias_uv)
                        >> PRECISION) as u8;
            }
        }
    };

    let y_plane = image.y_plane.borrow_mut();
    let u_plane = image.u_plane.borrow_mut();
    let v_plane = image.v_plane.borrow_mut();
    let y_stride = image.y_stride as usize;
    let u_stride = image.u_stride as usize;
    let v_stride = image.v_stride as usize;

    let iter = y_plane
        .chunks_exact_mut(y_stride * 2)
        .zip(u_plane.chunks_exact_mut(u_stride))
        .zip(v_plane.chunks_exact_mut(v_stride))
        .zip(rgba.chunks_exact(rgba_stride as usize * 2));

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

    Ok(())
}

macro_rules! build_fwd_yuv420_with_transform {
    ($method:ident, $px_fmt:expr, $px_name:expr, $px_small:expr) => {
        #[doc = concat!("Convert ", $px_name, " to YUV 420 using pre-computed transform coefficients.")]
        pub fn $method(
            planar_image: &mut YuvPlanarImageMut<u8>,
            src: &[u8],
            src_stride: u32,
            config: &YuvForwardTransform,
        ) -> Result<(), YuvError> {
            let transform = &config.transform;
            let chroma_range = &config.chroma_range;
            #[cfg(any(
                any(target_arch = "x86", target_arch = "x86_64"),
                all(target_arch = "aarch64", target_feature = "neon")
            ))]
            {
                match config.mode {
                    #[cfg(feature = "fast_mode")]
                    YuvConversionMode::Fast => {
                        rgbx_to_yuv420_with_transform_impl::<{ $px_fmt as u8 }, 7>(
                            planar_image,
                            src,
                            src_stride,
                            transform,
                            chroma_range,
                        )
                    }
                    YuvConversionMode::Balanced => {
                        rgbx_to_yuv420_with_transform_impl::<{ $px_fmt as u8 }, 13>(
                            planar_image,
                            src,
                            src_stride,
                            transform,
                            chroma_range,
                        )
                    }
                    #[cfg(feature = "professional_mode")]
                    YuvConversionMode::Professional => {
                        rgbx_to_yuv420_with_transform_impl::<{ $px_fmt as u8 }, 16>(
                            planar_image,
                            src,
                            src_stride,
                            transform,
                            chroma_range,
                        )
                    }
                }
            }
            #[cfg(not(any(
                all(any(target_arch = "x86", target_arch = "x86_64"),),
                all(target_arch = "aarch64", target_feature = "neon",),
            )))]
            {
                match config.mode {
                    YuvConversionMode::Balanced => {}
                    #[cfg(feature = "fast_mode")]
                    YuvConversionMode::Fast => {}
                    #[cfg(feature = "professional_mode")]
                    YuvConversionMode::Professional => {
                        return rgbx_to_yuv420_with_transform_impl::<{ $px_fmt as u8 }, 16>(
                            planar_image,
                            src,
                            src_stride,
                            transform,
                            chroma_range,
                        );
                    }
                }
                rgbx_to_yuv420_with_transform_impl::<{ $px_fmt as u8 }, 13>(
                    planar_image,
                    src,
                    src_stride,
                    transform,
                    chroma_range,
                )
            }
        }
    };
}

build_fwd_yuv420_with_transform!(
    rgba_to_yuv420_with_transform,
    YuvSourceChannels::Rgba,
    "RGBA",
    "rgba"
);
build_fwd_yuv420_with_transform!(
    rgb_to_yuv420_with_transform,
    YuvSourceChannels::Rgb,
    "RGB",
    "rgb"
);

#[cfg(test)]
mod tests {
    #[cfg(feature = "professional_mode")]
    use super::*;
    #[cfg(feature = "professional_mode")]
    use crate::yuv_support::{
        get_forward_transform, get_inverse_transform, get_yuv_range, ToIntegerTransform,
        YuvForwardTransform, YuvInverseTransform,
    };
    #[cfg(feature = "professional_mode")]
    use crate::yuv_to_rgba_with_transform::yuv420_to_rgba_with_transform;
    #[cfg(feature = "professional_mode")]
    use rand::RngExt;

    #[test]
    #[cfg(feature = "professional_mode")]
    fn test_rgba_to_yuv420_p16_round_trip() {
        const IMAGE_WIDTH: usize = 256;
        const IMAGE_HEIGHT: usize = 256;
        const CHANNELS: usize = 4;

        let or = rand::rng().random_range(0..256) as u8;
        let og = rand::rng().random_range(0..256) as u8;
        let ob = rand::rng().random_range(0..256) as u8;

        let mut source_rgba = vec![0u8; IMAGE_WIDTH * IMAGE_HEIGHT * CHANNELS];
        for pixel in source_rgba.chunks_exact_mut(CHANNELS) {
            pixel[0] = or;
            pixel[1] = og;
            pixel[2] = ob;
            pixel[3] = 255;
        }

        let range = get_yuv_range(8, YuvRange::Limited);
        let kr_kb = YuvStandardMatrix::Bt709.get_kr_kb();
        let transform =
            get_forward_transform(255, range.range_y, range.range_uv, kr_kb.kr, kr_kb.kb)
                .to_integers(16);
        let inverse_transform =
            get_inverse_transform(255, range.range_y, range.range_uv, kr_kb.kr, kr_kb.kb)
                .to_integers(16);

        let mut planar_image = YuvPlanarImageMut::<u8>::alloc(
            IMAGE_WIDTH as u32,
            IMAGE_HEIGHT as u32,
            YuvChromaSubsampling::Yuv420,
        );

        let fwd_config = YuvForwardTransform {
            transform,
            chroma_range: range,
            mode: YuvConversionMode::Professional,
        };
        rgba_to_yuv420_with_transform(
            &mut planar_image,
            &source_rgba,
            IMAGE_WIDTH as u32 * CHANNELS as u32,
            &fwd_config,
        )
        .unwrap();

        let fixed_planar = planar_image.to_fixed();

        let mut dest_rgba = vec![0u8; IMAGE_WIDTH * IMAGE_HEIGHT * CHANNELS];

        let inv_config = YuvInverseTransform {
            inverse_transform,
            chroma_range: range,
            mode: YuvConversionMode::Professional,
        };
        yuv420_to_rgba_with_transform(
            &fixed_planar,
            &mut dest_rgba,
            IMAGE_WIDTH as u32 * CHANNELS as u32,
            &inv_config,
        )
        .unwrap();

        let random_point_x = rand::rng().random_range(2..IMAGE_WIDTH - 2);
        let random_point_y = rand::rng().random_range(2..IMAGE_HEIGHT - 2);

        let pixel_points = [
            [IMAGE_WIDTH / 2, IMAGE_HEIGHT / 2],
            [IMAGE_WIDTH / 4, IMAGE_HEIGHT / 4],
            [IMAGE_WIDTH / 4 * 3, IMAGE_HEIGHT / 4],
            [IMAGE_WIDTH / 4, IMAGE_HEIGHT / 4 * 3],
            [IMAGE_WIDTH / 4 * 3, IMAGE_HEIGHT / 4 * 3],
            [IMAGE_WIDTH / 5, IMAGE_HEIGHT / 5],
            [IMAGE_WIDTH / 5 * 3, IMAGE_HEIGHT / 5],
            [IMAGE_WIDTH / 5, IMAGE_HEIGHT / 5 * 3],
            [IMAGE_WIDTH / 5 * 3, IMAGE_HEIGHT / 5 * 3],
            [IMAGE_WIDTH / 3, IMAGE_HEIGHT / 3],
            [IMAGE_WIDTH / 3 * 2, IMAGE_HEIGHT / 3 * 2],
            [random_point_x, random_point_y],
        ];

        for point in &pixel_points {
            let x = point[0];
            let y = point[1];
            let px = x * CHANNELS + y * IMAGE_WIDTH * CHANNELS;

            let r = dest_rgba[px];
            let g = dest_rgba[px + 1];
            let b = dest_rgba[px + 2];

            let diff_r = (r as i32 - or as i32).abs();
            let diff_g = (g as i32 - og as i32).abs();
            let diff_b = (b as i32 - ob as i32).abs();

            let max_diff = 1;
            assert!(
                diff_r <= max_diff,
                "P16 Full/Bt709, R diff {}, Original RGBA {:?}, Round-tripped RGBA {:?}, point {:?}",
                diff_r,
                [or, og, ob],
                [r, g, b],
                point
            );
            assert!(
                diff_g <= max_diff,
                "P16 Full/Bt709, G diff {}, Original RGBA {:?}, Round-tripped RGBA {:?}, point {:?}",
                diff_g,
                [or, og, ob],
                [r, g, b],
                point
            );
            assert!(
                diff_b <= max_diff,
                "P16 Full/Bt709, B diff {}, Original RGBA {:?}, Round-tripped RGBA {:?}, point {:?}",
                diff_b,
                [or, og, ob],
                [r, g, b],
                point
            );
        }
    }

    #[test]
    #[cfg(feature = "professional_mode")]
    fn test_rgba_to_yuv420_p16_webp_constants() {
        const IMAGE_WIDTH: usize = 64;
        const IMAGE_HEIGHT: usize = 64;
        const CHANNELS: usize = 4;

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

            let mut source_rgba = vec![0u8; IMAGE_WIDTH * IMAGE_HEIGHT * CHANNELS];
            for px in source_rgba.chunks_exact_mut(CHANNELS) {
                px[0] = or;
                px[1] = og;
                px[2] = ob;
                px[3] = 255;
            }

            let mut planar_image = YuvPlanarImageMut::<u8>::alloc(
                IMAGE_WIDTH as u32,
                IMAGE_HEIGHT as u32,
                YuvChromaSubsampling::Yuv420,
            );

            let fwd_config = YuvForwardTransform {
                transform: webp_transform,
                chroma_range: range,
                mode: YuvConversionMode::Professional,
            };
            rgba_to_yuv420_with_transform(
                &mut planar_image,
                &source_rgba,
                IMAGE_WIDTH as u32 * CHANNELS as u32,
                &fwd_config,
            )
            .unwrap();

            let fixed_planar = planar_image.to_fixed();
            let mut dest_rgba = vec![0u8; IMAGE_WIDTH * IMAGE_HEIGHT * CHANNELS];

            let inv_config = YuvInverseTransform {
                inverse_transform,
                chroma_range: range,
                mode: YuvConversionMode::Professional,
            };
            yuv420_to_rgba_with_transform(
                &fixed_planar,
                &mut dest_rgba,
                IMAGE_WIDTH as u32 * CHANNELS as u32,
                &inv_config,
            )
            .unwrap();

            for (src_px, dst_px) in source_rgba
                .chunks_exact(CHANNELS)
                .zip(dest_rgba.chunks_exact(CHANNELS))
            {
                worst_r = worst_r.max((dst_px[0] as i32 - src_px[0] as i32).abs());
                worst_g = worst_g.max((dst_px[1] as i32 - src_px[1] as i32).abs());
                worst_b = worst_b.max((dst_px[2] as i32 - src_px[2] as i32).abs());
            }
        }

        let max_diff = 1;
        assert!(
            worst_r <= max_diff && worst_g <= max_diff && worst_b <= max_diff,
            "WebP P16 round-trip: worst per-channel diffs ({},{},{}) exceed {}",
            worst_r,
            worst_g,
            worst_b,
            max_diff
        );
    }
}
