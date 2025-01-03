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
use crate::avx2::{avx_rgba_to_yuv_p16, avx_rgba_to_yuv_p16_420};
#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    feature = "nightly_avx512"
))]
use crate::avx512bw::{avx512_rgba_to_yuv_p16, avx512_rgba_to_yuv_p16_420};
use crate::internals::ProcessedOffset;
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use crate::neon::{
    neon_rgba_to_yuv_p16, neon_rgba_to_yuv_p16_420, neon_rgba_to_yuv_p16_rdm,
    neon_rgba_to_yuv_p16_rdm_420,
};
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use crate::sse::{sse_rgba_to_yuv_p16, sse_rgba_to_yuv_p16_420};
use crate::yuv_error::check_rgba_destination;
use crate::yuv_support::{
    get_forward_transform, get_yuv_range, ToIntegerTransform, YuvChromaSubsampling,
    YuvSourceChannels,
};
use crate::{
    YuvBytesPacking, YuvEndianness, YuvError, YuvPlanarImageMut, YuvRange, YuvStandardMatrix,
};
#[cfg(feature = "rayon")]
use rayon::iter::{IndexedParallelIterator, ParallelIterator};
#[cfg(feature = "rayon")]
use rayon::prelude::{ParallelSlice, ParallelSliceMut};

#[inline(always)]
fn transform_integer<const ENDIANNESS: u8, const BYTES_POSITION: u8, const BIT_DEPTH: usize>(
    v: i32,
) -> u16 {
    let endianness: YuvEndianness = ENDIANNESS.into();
    let bytes_position: YuvBytesPacking = BYTES_POSITION.into();
    let packing: i32 = 16 - BIT_DEPTH as i32;
    let packed_bytes = match bytes_position {
        YuvBytesPacking::MostSignificantBytes => v << packing,
        YuvBytesPacking::LeastSignificantBytes => v,
    } as u16;
    match endianness {
        YuvEndianness::BigEndian => packed_bytes.to_be(),
        YuvEndianness::LittleEndian => packed_bytes.to_le(),
    }
}

fn rgbx_to_yuv_ant<
    const ORIGIN_CHANNELS: u8,
    const SAMPLING: u8,
    const ENDIANNESS: u8,
    const BYTES_POSITION: u8,
    const BIT_DEPTH: usize,
>(
    image: &mut YuvPlanarImageMut<u16>,
    rgba: &[u16],
    rgba_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) -> Result<(), YuvError> {
    let chroma_subsampling: YuvChromaSubsampling = SAMPLING.into();
    let src_chans: YuvSourceChannels = ORIGIN_CHANNELS.into();
    let channels = src_chans.get_channels_count();

    image.check_constraints(chroma_subsampling)?;
    check_rgba_destination(rgba, rgba_stride, image.width, image.height, channels)?;

    let range = get_yuv_range(BIT_DEPTH as u32, range);
    let kr_kb = matrix.get_kr_kb();
    let max_range_p8 = (1u32 << BIT_DEPTH) - 1u32;
    let transform_precise = get_forward_transform(
        max_range_p8,
        range.range_y,
        range.range_uv,
        kr_kb.kr,
        kr_kb.kb,
    );

    const PRECISION: i32 = 13;

    let transform = transform_precise.to_integers(PRECISION as u32);
    const ROUNDING_CONST_BIAS: i32 = (1 << (PRECISION - 1)) - 1;
    let bias_y = range.bias_y as i32 * (1 << PRECISION) + ROUNDING_CONST_BIAS;
    let bias_uv = range.bias_uv as i32 * (1 << PRECISION) + ROUNDING_CONST_BIAS;

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    let use_sse = std::arch::is_x86_feature_detected!("sse4.1");
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    let use_avx = std::arch::is_x86_feature_detected!("avx2");
    #[cfg(all(
        any(target_arch = "x86", target_arch = "x86_64"),
        feature = "nightly_avx512"
    ))]
    let use_avx512 = std::arch::is_x86_feature_detected!("avx512bw");
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    let is_rdm_available = std::arch::is_aarch64_feature_detected!("rdm");
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    let neon_wide_row_handler = if is_rdm_available && BIT_DEPTH <= 12 {
        neon_rgba_to_yuv_p16_rdm::<
            ORIGIN_CHANNELS,
            SAMPLING,
            ENDIANNESS,
            BYTES_POSITION,
            PRECISION,
            BIT_DEPTH,
        >
    } else {
        neon_rgba_to_yuv_p16::<
            ORIGIN_CHANNELS,
            SAMPLING,
            ENDIANNESS,
            BYTES_POSITION,
            PRECISION,
            BIT_DEPTH,
        >
    };
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    let neon_double_row_handler = if is_rdm_available && BIT_DEPTH <= 12 {
        neon_rgba_to_yuv_p16_rdm_420::<
            ORIGIN_CHANNELS,
            ENDIANNESS,
            BYTES_POSITION,
            PRECISION,
            BIT_DEPTH,
        >
    } else {
        neon_rgba_to_yuv_p16_420::<ORIGIN_CHANNELS, ENDIANNESS, BYTES_POSITION, PRECISION, BIT_DEPTH>
    };
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    let sse_dispatch = sse_rgba_to_yuv_p16::<
        ORIGIN_CHANNELS,
        SAMPLING,
        ENDIANNESS,
        BYTES_POSITION,
        PRECISION,
        BIT_DEPTH,
    >;

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    let sse_dispatch_420 = sse_rgba_to_yuv_p16_420::<
        ORIGIN_CHANNELS,
        ENDIANNESS,
        BYTES_POSITION,
        PRECISION,
        BIT_DEPTH,
    >;

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    let avx_dispatch_420 = avx_rgba_to_yuv_p16_420::<
        ORIGIN_CHANNELS,
        ENDIANNESS,
        BYTES_POSITION,
        PRECISION,
        BIT_DEPTH,
    >;

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    let avx_dispatch = avx_rgba_to_yuv_p16::<
        ORIGIN_CHANNELS,
        SAMPLING,
        ENDIANNESS,
        BYTES_POSITION,
        PRECISION,
        BIT_DEPTH,
    >;

    #[allow(unused_variables)]
    let process_wide_row = |_y_plane: &mut [u16],
                            _u_plane: &mut [u16],
                            _v_plane: &mut [u16],
                            rgba: &[u16],
                            _cx,
                            _ux,
                            _compute_uv_row| {
        let mut _offset = ProcessedOffset { ux: _cx, cx: _ux };
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            #[cfg(feature = "nightly_avx512")]
            if use_avx512 && BIT_DEPTH <= 12 {
                _offset = avx512_rgba_to_yuv_p16::<
                    ORIGIN_CHANNELS,
                    SAMPLING,
                    ENDIANNESS,
                    BYTES_POSITION,
                    PRECISION,
                    BIT_DEPTH,
                >(
                    &transform,
                    &range,
                    _y_plane,
                    _u_plane,
                    _v_plane,
                    rgba,
                    _offset.cx,
                    _offset.ux,
                    image.width as usize,
                );
            }
            if use_avx {
                _offset = avx_dispatch(
                    &transform,
                    &range,
                    _y_plane,
                    _u_plane,
                    _v_plane,
                    rgba,
                    _offset.cx,
                    _offset.ux,
                    image.width as usize,
                );
            }
            if use_sse {
                _offset = sse_dispatch(
                    &transform,
                    &range,
                    _y_plane,
                    _u_plane,
                    _v_plane,
                    rgba,
                    _offset.cx,
                    _offset.ux,
                    image.width as usize,
                );
            }
        }

        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        unsafe {
            _offset = neon_wide_row_handler(
                &transform,
                &range,
                _y_plane,
                _u_plane,
                _v_plane,
                rgba,
                _offset.cx,
                _offset.ux,
                image.width as usize,
            );
        }

        _offset
    };

    let process_halved_chroma_row = |y_plane: &mut [u16],
                                     u_plane: &mut [u16],
                                     v_plane: &mut [u16],
                                     rgba| {
        let processed_offset = process_wide_row(y_plane, u_plane, v_plane, rgba, 0, 0, true);
        let cx = processed_offset.cx;

        for (((y_dst, u_dst), v_dst), rgba) in y_plane
            .chunks_exact_mut(2)
            .zip(u_plane.iter_mut())
            .zip(v_plane.iter_mut())
            .zip(rgba.chunks_exact(channels * 2))
            .skip(cx / 2)
        {
            let r0 = rgba[src_chans.get_r_channel_offset()] as i32;
            let g0 = rgba[src_chans.get_g_channel_offset()] as i32;
            let b0 = rgba[src_chans.get_b_channel_offset()] as i32;
            let y_0 =
                (r0 * transform.yr + g0 * transform.yg + b0 * transform.yb + bias_y) >> PRECISION;
            y_dst[0] = transform_integer::<ENDIANNESS, BYTES_POSITION, BIT_DEPTH>(y_0);

            let r1 = rgba[channels + src_chans.get_r_channel_offset()] as i32;
            let g1 = rgba[channels + src_chans.get_g_channel_offset()] as i32;
            let b1 = rgba[channels + src_chans.get_b_channel_offset()] as i32;
            let y_1 =
                (r1 * transform.yr + g1 * transform.yg + b1 * transform.yb + bias_y) >> PRECISION;
            y_dst[1] = transform_integer::<ENDIANNESS, BYTES_POSITION, BIT_DEPTH>(y_1);

            let r = (r0 + r1 + 1) >> 1;
            let g = (g0 + g1 + 1) >> 1;
            let b = (b0 + b1 + 1) >> 1;

            let cb = (r * transform.cb_r + g * transform.cb_g + b * transform.cb_b + bias_uv)
                >> PRECISION;
            let cr = (r * transform.cr_r + g * transform.cr_g + b * transform.cr_b + bias_uv)
                >> PRECISION;
            *u_dst = transform_integer::<ENDIANNESS, BYTES_POSITION, BIT_DEPTH>(cb);
            *v_dst = transform_integer::<ENDIANNESS, BYTES_POSITION, BIT_DEPTH>(cr);
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
            *y_last = transform_integer::<ENDIANNESS, BYTES_POSITION, BIT_DEPTH>(y_0);

            let cb = (r0 * transform.cb_r + g0 * transform.cb_g + b0 * transform.cb_b + bias_uv)
                >> PRECISION;
            let cr = (r0 * transform.cr_r + g0 * transform.cr_g + b0 * transform.cr_b + bias_uv)
                >> PRECISION;
            *u_last = transform_integer::<ENDIANNESS, BYTES_POSITION, BIT_DEPTH>(cb);
            *v_last = transform_integer::<ENDIANNESS, BYTES_POSITION, BIT_DEPTH>(cr);
        }
    };

    let process_wide_double_chroma_row = |_y_plane0: &mut [u16],
                                          _y_plane1: &mut [u16],
                                          _u_plane: &mut [u16],
                                          _v_plane: &mut [u16],
                                          _rgba0: &[u16],
                                          _rgba1: &[u16]| {
        let mut _offset = ProcessedOffset { cx: 0, ux: 0 };
        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        unsafe {
            _offset = neon_double_row_handler(
                &transform,
                &range,
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
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            #[cfg(feature = "nightly_avx512")]
            if use_avx512 && BIT_DEPTH <= 12 {
                _offset = avx512_rgba_to_yuv_p16_420::<
                    ORIGIN_CHANNELS,
                    ENDIANNESS,
                    BYTES_POSITION,
                    PRECISION,
                    BIT_DEPTH,
                >(
                    &transform,
                    &range,
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
            }
            if use_avx {
                _offset = avx_dispatch_420(
                    &transform,
                    &range,
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
            }
            if use_sse {
                _offset = sse_dispatch_420(
                    &transform,
                    &range,
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
            }
        }
        _offset
    };

    let process_double_chroma_row = |y_plane0: &mut [u16],
                                     y_plane1: &mut [u16],
                                     u_plane: &mut [u16],
                                     v_plane: &mut [u16],
                                     rgba0: &[u16],
                                     rgba1: &[u16]| {
        let processed_offset =
            process_wide_double_chroma_row(y_plane0, y_plane1, u_plane, v_plane, rgba0, rgba1);
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
            let r00 = rgba0[src_chans.get_r_channel_offset()] as i32;
            let g00 = rgba0[src_chans.get_g_channel_offset()] as i32;
            let b00 = rgba0[src_chans.get_b_channel_offset()] as i32;
            let y_00 = (r00 * transform.yr + g00 * transform.yg + b00 * transform.yb + bias_y)
                >> PRECISION;
            y_dst0[0] = transform_integer::<ENDIANNESS, BYTES_POSITION, BIT_DEPTH>(y_00);

            let rgba01 = &rgba0[channels..channels * 2];
            let r01 = rgba01[src_chans.get_r_channel_offset()] as i32;
            let g01 = rgba01[src_chans.get_g_channel_offset()] as i32;
            let b01 = rgba01[src_chans.get_b_channel_offset()] as i32;
            let y_01 = (r01 * transform.yr + g01 * transform.yg + b01 * transform.yb + bias_y)
                >> PRECISION;
            y_dst0[1] = transform_integer::<ENDIANNESS, BYTES_POSITION, BIT_DEPTH>(y_01);

            let r10 = rgba1[src_chans.get_r_channel_offset()] as i32;
            let g10 = rgba1[src_chans.get_g_channel_offset()] as i32;
            let b10 = rgba1[src_chans.get_b_channel_offset()] as i32;
            let y_10 = (r10 * transform.yr + g10 * transform.yg + b10 * transform.yb + bias_y)
                >> PRECISION;
            y_dst1[0] = transform_integer::<ENDIANNESS, BYTES_POSITION, BIT_DEPTH>(y_10);

            let rgba11 = &rgba1[channels..channels * 2];
            let r11 = rgba11[src_chans.get_r_channel_offset()] as i32;
            let g11 = rgba11[src_chans.get_g_channel_offset()] as i32;
            let b11 = rgba11[src_chans.get_b_channel_offset()] as i32;
            let y_11 = (r01 * transform.yr + g01 * transform.yg + b01 * transform.yb + bias_y)
                >> PRECISION;
            y_dst1[1] = transform_integer::<ENDIANNESS, BYTES_POSITION, BIT_DEPTH>(y_11);

            let r = (r00 + r01 + r10 + r11 + 2) >> 2;
            let g = (g00 + g01 + g10 + g11 + 2) >> 2;
            let b = (b00 + b01 + b10 + b11 + 2) >> 2;

            let cb = (r * transform.cb_r + g * transform.cb_g + b * transform.cb_b + bias_uv)
                >> PRECISION;
            let cr = (r * transform.cr_r + g * transform.cr_g + b * transform.cr_b + bias_uv)
                >> PRECISION;
            *u_dst = transform_integer::<ENDIANNESS, BYTES_POSITION, BIT_DEPTH>(cb);
            *v_dst = transform_integer::<ENDIANNESS, BYTES_POSITION, BIT_DEPTH>(cr);
        }

        if image.width & 1 != 0 {
            let rgb_last0 = rgba0.chunks_exact(channels * 2).remainder();
            let r0 = rgb_last0[src_chans.get_r_channel_offset()] as i32;
            let g0 = rgb_last0[src_chans.get_g_channel_offset()] as i32;
            let b0 = rgb_last0[src_chans.get_b_channel_offset()] as i32;

            let rgb_last1 = rgba1.chunks_exact(channels * 2).remainder();
            let r1 = rgb_last1[src_chans.get_r_channel_offset()] as i32;
            let g1 = rgb_last1[src_chans.get_g_channel_offset()] as i32;
            let b1 = rgb_last1[src_chans.get_b_channel_offset()] as i32;

            let y0_last = y_plane0.last_mut().unwrap();
            let y1_last = y_plane1.last_mut().unwrap();
            let u_last = u_plane.last_mut().unwrap();
            let v_last = v_plane.last_mut().unwrap();

            let y_0 =
                (r0 * transform.yr + g0 * transform.yg + b0 * transform.yb + bias_y) >> PRECISION;
            *y0_last = transform_integer::<ENDIANNESS, BYTES_POSITION, BIT_DEPTH>(y_0);

            let y_1 =
                (r1 * transform.yr + g1 * transform.yg + b1 * transform.yb + bias_y) >> PRECISION;
            *y1_last = transform_integer::<ENDIANNESS, BYTES_POSITION, BIT_DEPTH>(y_1);

            let r = (r0 + r1 + 1) >> 1;
            let g = (g0 + g1 + 1) >> 1;
            let b = (b0 + b1 + 1) >> 1;

            let cb = (r * transform.cb_r + g * transform.cb_g + b * transform.cb_b + bias_uv)
                >> PRECISION;
            let cr = (r * transform.cr_r + g * transform.cr_g + b * transform.cr_b + bias_uv)
                >> PRECISION;
            *u_last = transform_integer::<ENDIANNESS, BYTES_POSITION, BIT_DEPTH>(cb);
            *v_last = transform_integer::<ENDIANNESS, BYTES_POSITION, BIT_DEPTH>(cr);
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
                *y_dst = transform_integer::<ENDIANNESS, BYTES_POSITION, BIT_DEPTH>(y_0);

                let cb =
                    (r0 * transform.cb_r + g0 * transform.cb_g + b0 * transform.cb_b + bias_uv)
                        >> PRECISION;
                let cr =
                    (r0 * transform.cr_r + g0 * transform.cr_g + b0 * transform.cr_b + bias_uv)
                        >> PRECISION;
                *u_dst = transform_integer::<ENDIANNESS, BYTES_POSITION, BIT_DEPTH>(cb);
                *v_dst = transform_integer::<ENDIANNESS, BYTES_POSITION, BIT_DEPTH>(cr);
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
            process_halved_chroma_row(y_plane, u_plane, v_plane, rgba);
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
            let (y_plane0, y_plane1) = y_plane.split_at_mut(y_stride);
            let (rgba0, rgba1) = rgba.split_at(rgba_stride as usize);
            process_double_chroma_row(
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

fn rgbx_to_yuv<
    const ORIGIN_CHANNELS: u8,
    const SAMPLING: u8,
    const ENDIANNESS: u8,
    const BYTES_POSITION: u8,
>(
    planar_image: &mut YuvPlanarImageMut<u16>,
    rgba: &[u16],
    rgba_stride: u32,
    bit_depth: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) -> Result<(), YuvError> {
    if bit_depth == 10 {
        rgbx_to_yuv_ant::<ORIGIN_CHANNELS, SAMPLING, ENDIANNESS, BYTES_POSITION, 10>(
            planar_image,
            rgba,
            rgba_stride,
            range,
            matrix,
        )
    } else if bit_depth == 12 {
        rgbx_to_yuv_ant::<ORIGIN_CHANNELS, SAMPLING, ENDIANNESS, BYTES_POSITION, 12>(
            planar_image,
            rgba,
            rgba_stride,
            range,
            matrix,
        )
    } else {
        unimplemented!("RGB16 to YUV16 implemented only for 10 and 12 bit-depth")
    }
}

/// Convert RGB image data to YUV 422 planar format with 10 or 12 bit depth.
///
/// This function performs RGB to YUV conversion and stores the result in YUV422 planar format,
/// with separate planes for Y (luminance), U (chrominance), and V (chrominance) components.
///
/// # Arguments
///
/// * `planar_image` - Target planar image.
/// * `rgb` - The input RGB image data slice.
/// * `rgb_stride` - The stride (components per row) for the RGB image data.
/// * `bit_depth` - Only 10 or 12 bit-depth is supported
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
/// * `endianness` - The endianness of final YUV
/// * `bytes_packing` - position of significant bytes for YUV ( most significant or least significant ) if it in most significant it should be stated as per Apple *kCVPixelFormatType_422YpCbCr10BiPlanarFullRange/kCVPixelFormatType_422YpCbCr10BiPlanarVideoRange*
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input RGB data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn rgb_to_yuv422_p16(
    planar_image: &mut YuvPlanarImageMut<u16>,
    rgb: &[u16],
    rgb_stride: u32,
    bit_depth: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    endianness: YuvEndianness,
    bytes_packing: YuvBytesPacking,
) -> Result<(), YuvError> {
    assert!(
        bit_depth == 10 || bit_depth == 12,
        "Only 10 and 12 bit depth is supported"
    );
    let dispatcher = match endianness {
        YuvEndianness::BigEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                rgbx_to_yuv::<
                    { YuvSourceChannels::Rgb as u8 },
                    { YuvChromaSubsampling::Yuv422 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                rgbx_to_yuv::<
                    { YuvSourceChannels::Rgb as u8 },
                    { YuvChromaSubsampling::Yuv422 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
        YuvEndianness::LittleEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                rgbx_to_yuv::<
                    { YuvSourceChannels::Rgb as u8 },
                    { YuvChromaSubsampling::Yuv422 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                rgbx_to_yuv::<
                    { YuvSourceChannels::Rgb as u8 },
                    { YuvChromaSubsampling::Yuv422 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
    };
    dispatcher(planar_image, rgb, rgb_stride, bit_depth, range, matrix)
}

/// Convert BGR image data to YUV 422 planar format with 10 or 12 bit depth.
///
/// This function performs BGR to YUV conversion and stores the result in YUV422 planar format,
/// with separate planes for Y (luminance), U (chrominance), and V (chrominance) components.
///
/// # Arguments
///
/// * `planar_image` - Target planar image.
/// * `bgr` - The input BGR image data slice.
/// * `bgr_stride` - The stride (components per row) for the BGR image data.
/// * `bit_depth` - Only 10 or 12 bit-depth is supported.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
/// * `endianness` - The endianness of final YUV
/// * `bytes_packing` - position of significant bytes for YUV ( most significant or least significant ) if it in most significant it should be stated as per Apple *kCVPixelFormatType_422YpCbCr10BiPlanarFullRange/kCVPixelFormatType_422YpCbCr10BiPlanarVideoRange*
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGR data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn bgr_to_yuv422_p16(
    planar_image: &mut YuvPlanarImageMut<u16>,
    bgr: &[u16],
    bgr_stride: u32,
    bit_depth: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    endianness: YuvEndianness,
    bytes_packing: YuvBytesPacking,
) -> Result<(), YuvError> {
    assert!(
        bit_depth == 10 || bit_depth == 12,
        "Only 10 and 12 bit depth is supported"
    );
    let dispatcher = match endianness {
        YuvEndianness::BigEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                rgbx_to_yuv::<
                    { YuvSourceChannels::Bgr as u8 },
                    { YuvChromaSubsampling::Yuv422 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                rgbx_to_yuv::<
                    { YuvSourceChannels::Bgr as u8 },
                    { YuvChromaSubsampling::Yuv422 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
        YuvEndianness::LittleEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                rgbx_to_yuv::<
                    { YuvSourceChannels::Bgr as u8 },
                    { YuvChromaSubsampling::Yuv422 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                rgbx_to_yuv::<
                    { YuvSourceChannels::Bgr as u8 },
                    { YuvChromaSubsampling::Yuv422 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
    };
    dispatcher(planar_image, bgr, bgr_stride, bit_depth, range, matrix)
}

/// Convert RGBA image data to YUV 422 planar format with 10 or 12 bit depth.
///
/// This function performs RGBA to YUV conversion and stores the result in YUV422 planar format,
/// with separate planes for Y (luminance), U (chrominance), and V (chrominance) components.
///
/// # Arguments
///
/// * `planar_image` - Target planar image.
/// * `rgba` - The input RGBA image data slice.
/// * `rgba_stride` - The stride (components per row) for the RGBA image data.
/// * `bit_depth` - Only 10 or 12 bit-depth is supported.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
/// * `endianness` - The endianness of final YUV
/// * `bytes_packing` - position of significant bytes for YUV ( most significant or least significant ) if it in most significant it should be stated as per Apple *kCVPixelFormatType_422YpCbCr10BiPlanarFullRange/kCVPixelFormatType_422YpCbCr10BiPlanarVideoRange*
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input RGBA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn rgba_to_yuv422_p16(
    planar_image: &mut YuvPlanarImageMut<u16>,
    rgba: &[u16],
    rgba_stride: u32,
    bit_depth: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    endianness: YuvEndianness,
    bytes_packing: YuvBytesPacking,
) -> Result<(), YuvError> {
    assert!(
        bit_depth == 10 || bit_depth == 12,
        "Only 10 and 12 bit depth is supported"
    );
    let dispatcher = match endianness {
        YuvEndianness::BigEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                rgbx_to_yuv::<
                    { YuvSourceChannels::Rgba as u8 },
                    { YuvChromaSubsampling::Yuv422 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                rgbx_to_yuv::<
                    { YuvSourceChannels::Rgba as u8 },
                    { YuvChromaSubsampling::Yuv422 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
        YuvEndianness::LittleEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                rgbx_to_yuv::<
                    { YuvSourceChannels::Rgba as u8 },
                    { YuvChromaSubsampling::Yuv422 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                rgbx_to_yuv::<
                    { YuvSourceChannels::Rgba as u8 },
                    { YuvChromaSubsampling::Yuv422 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
    };
    dispatcher(planar_image, rgba, rgba_stride, bit_depth, range, matrix)
}

/// Convert BGRA image data to YUV 422 planar format with 10 or 12 bit depth.
///
/// This function performs BGRA to YUV conversion and stores the result in YUV422 planar format,
/// with separate planes for Y (luminance), U (chrominance), and V (chrominance) components.
///
/// # Arguments
///
/// * `planar_image` - Target planar image.
/// * `bgra` - The input BGRA image data slice.
/// * `bgra_stride` - The stride (components per row) for the BGRA image data.
/// * `bit_depth` - Only 10 or 12 bit-depth is supported.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
/// * `endianness` - The endianness of final YUV
/// * `bytes_packing` - position of significant bytes for YUV ( most significant or least significant ) if it in most significant it should be stated as per Apple *kCVPixelFormatType_422YpCbCr10BiPlanarFullRange/kCVPixelFormatType_422YpCbCr10BiPlanarVideoRange*
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGRA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn bgra_to_yuv422_p16(
    planar_image: &mut YuvPlanarImageMut<u16>,
    bgra: &[u16],
    bgra_stride: u32,
    bit_depth: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    endianness: YuvEndianness,
    bytes_packing: YuvBytesPacking,
) -> Result<(), YuvError> {
    assert!(
        bit_depth == 10 || bit_depth == 12,
        "Only 10 and 12 bit depth is supported"
    );
    let dispatcher = match endianness {
        YuvEndianness::BigEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                rgbx_to_yuv::<
                    { YuvSourceChannels::Bgra as u8 },
                    { YuvChromaSubsampling::Yuv422 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                rgbx_to_yuv::<
                    { YuvSourceChannels::Bgra as u8 },
                    { YuvChromaSubsampling::Yuv422 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
        YuvEndianness::LittleEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                rgbx_to_yuv::<
                    { YuvSourceChannels::Bgra as u8 },
                    { YuvChromaSubsampling::Yuv422 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                rgbx_to_yuv::<
                    { YuvSourceChannels::Bgra as u8 },
                    { YuvChromaSubsampling::Yuv422 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
    };
    dispatcher(planar_image, bgra, bgra_stride, bit_depth, range, matrix)
}

/// Convert RGB image data to YUV 420 planar format with 10 or 12 bit depth.
///
/// This function performs RGB to YUV conversion and stores the result in YUV420 planar format,
/// with separate planes for Y (luminance), U (chrominance), and V (chrominance) components.
///
/// # Arguments
///
/// * `planar_image` - Target planar image.
/// * `rgb` - The input RGB image data slice.
/// * `rgb_stride` - The stride (components per row) for the RGB image data.
/// * `bit_depth` - Only 10 or 12 bit-depth is supported.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
/// * `endianness` - The endianness of final YUV
/// * `bytes_packing` - position of significant bytes for YUV ( most significant or least significant ) if it in most significant it should be stated as per Apple *kCVPixelFormatType_422YpCbCr10BiPlanarFullRange/kCVPixelFormatType_422YpCbCr10BiPlanarVideoRange*
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input RGB data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn rgb_to_yuv420_p16(
    planar_image: &mut YuvPlanarImageMut<u16>,
    rgb: &[u16],
    rgb_stride: u32,
    bit_depth: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    endianness: YuvEndianness,
    bytes_packing: YuvBytesPacking,
) -> Result<(), YuvError> {
    assert!(
        bit_depth == 10 || bit_depth == 12,
        "Only 10 and 12 bit depth is supported"
    );
    let dispatcher = match endianness {
        YuvEndianness::BigEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                rgbx_to_yuv::<
                    { YuvSourceChannels::Rgb as u8 },
                    { YuvChromaSubsampling::Yuv420 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                rgbx_to_yuv::<
                    { YuvSourceChannels::Rgb as u8 },
                    { YuvChromaSubsampling::Yuv420 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
        YuvEndianness::LittleEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                rgbx_to_yuv::<
                    { YuvSourceChannels::Rgb as u8 },
                    { YuvChromaSubsampling::Yuv420 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                rgbx_to_yuv::<
                    { YuvSourceChannels::Rgb as u8 },
                    { YuvChromaSubsampling::Yuv420 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
    };
    dispatcher(planar_image, rgb, rgb_stride, bit_depth, range, matrix)
}

/// Convert BGR image data to YUV 420 planar format with 10 or 12 bit depth.
///
/// This function performs BGR to YUV conversion and stores the result in YUV420 planar format,
/// with separate planes for Y (luminance), U (chrominance), and V (chrominance) components.
///
/// # Arguments
///
/// * `planar_image` - Target planar image.
/// * `bgr` - The input BGR image data slice.
/// * `bgr_stride` - The stride (components per row) for the BGR image data.
/// * `bit_depth` - Only 10 or 12 bit-depth is supported.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
/// * `endianness` - The endianness of final YUV
/// * `bytes_packing` - position of significant bytes for YUV ( most significant or least significant ) if it in most significant it should be stated as per Apple *kCVPixelFormatType_422YpCbCr10BiPlanarFullRange/kCVPixelFormatType_422YpCbCr10BiPlanarVideoRange*
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input RGB data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn bgr_to_yuv420_p16(
    planar_image: &mut YuvPlanarImageMut<u16>,
    bgr: &[u16],
    bgr_stride: u32,
    bit_depth: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    endianness: YuvEndianness,
    bytes_packing: YuvBytesPacking,
) -> Result<(), YuvError> {
    assert!(
        bit_depth == 10 || bit_depth == 12,
        "Only 10 and 12 bit depth is supported"
    );
    let dispatcher = match endianness {
        YuvEndianness::BigEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                rgbx_to_yuv::<
                    { YuvSourceChannels::Bgr as u8 },
                    { YuvChromaSubsampling::Yuv420 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                rgbx_to_yuv::<
                    { YuvSourceChannels::Bgr as u8 },
                    { YuvChromaSubsampling::Yuv420 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
        YuvEndianness::LittleEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                rgbx_to_yuv::<
                    { YuvSourceChannels::Bgr as u8 },
                    { YuvChromaSubsampling::Yuv420 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                rgbx_to_yuv::<
                    { YuvSourceChannels::Bgr as u8 },
                    { YuvChromaSubsampling::Yuv420 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
    };
    dispatcher(planar_image, bgr, bgr_stride, bit_depth, range, matrix)
}

/// Convert RGBA image data to YUV 420 planar format with 10 or 12 bit depth.
///
/// This function performs RGBA to YUV conversion and stores the result in YUV420 planar format,
/// with separate planes for Y (luminance), U (chrominance), and V (chrominance) components.
///
/// # Arguments
///
/// * `planar_image` - Target planar image.
/// * `rgba` - The input RGBA image data slice.
/// * `rgba_stride` - The stride (components per row) for the RGBA image data.
/// * `bit_depth` - Only 10 or 12 bit-depth is supported.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
/// * `endianness` - The endianness of final YUV
/// * `bytes_packing` - position of significant bytes for YUV ( most significant or least significant ) if it in most significant it should be stated as per Apple *kCVPixelFormatType_422YpCbCr10BiPlanarFullRange/kCVPixelFormatType_422YpCbCr10BiPlanarVideoRange*
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input RGBA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn rgba_to_yuv420_p16(
    planar_image: &mut YuvPlanarImageMut<u16>,
    rgba: &[u16],
    rgba_stride: u32,
    bit_depth: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    endianness: YuvEndianness,
    bytes_packing: YuvBytesPacking,
) -> Result<(), YuvError> {
    assert!(
        bit_depth == 10 || bit_depth == 12,
        "Only 10 and 12 bit depth is supported"
    );
    let dispatcher = match endianness {
        YuvEndianness::BigEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                rgbx_to_yuv::<
                    { YuvSourceChannels::Rgba as u8 },
                    { YuvChromaSubsampling::Yuv420 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                rgbx_to_yuv::<
                    { YuvSourceChannels::Rgba as u8 },
                    { YuvChromaSubsampling::Yuv420 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
        YuvEndianness::LittleEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                rgbx_to_yuv::<
                    { YuvSourceChannels::Rgba as u8 },
                    { YuvChromaSubsampling::Yuv420 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                rgbx_to_yuv::<
                    { YuvSourceChannels::Rgba as u8 },
                    { YuvChromaSubsampling::Yuv420 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
    };
    dispatcher(planar_image, rgba, rgba_stride, bit_depth, range, matrix)
}

/// Convert BGRA image data to YUV 420 planar format with 10 or 12 bit depth.
///
/// This function performs BGRA to YUV conversion and stores the result in YUV420 planar format,
/// with separate planes for Y (luminance), U (chrominance), and V (chrominance) components.
///
/// # Arguments
///
/// * `planar_image` - Target planar image.
/// * `bgra` - The input BGRA image data slice.
/// * `bgra_stride` - The stride (components per row) for the BGRA image data.
/// * `bit_depth` - Only 10 or 12 bit-depth is supported.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
/// * `endianness` - The endianness of final YUV
/// * `bytes_packing` - position of significant bytes for YUV ( most significant or least significant ) if it in most significant it should be stated as per Apple *kCVPixelFormatType_422YpCbCr10BiPlanarFullRange/kCVPixelFormatType_422YpCbCr10BiPlanarVideoRange*
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGRA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn bgra_to_yuv420_p16(
    planar_image: &mut YuvPlanarImageMut<u16>,
    bgra: &[u16],
    bgra_stride: u32,
    bit_depth: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    endianness: YuvEndianness,
    bytes_packing: YuvBytesPacking,
) -> Result<(), YuvError> {
    assert!(
        bit_depth == 10 || bit_depth == 12,
        "Only 10 and 12 bit depth is supported"
    );
    let dispatcher = match endianness {
        YuvEndianness::BigEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                rgbx_to_yuv::<
                    { YuvSourceChannels::Bgra as u8 },
                    { YuvChromaSubsampling::Yuv420 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                rgbx_to_yuv::<
                    { YuvSourceChannels::Bgra as u8 },
                    { YuvChromaSubsampling::Yuv420 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
        YuvEndianness::LittleEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                rgbx_to_yuv::<
                    { YuvSourceChannels::Bgra as u8 },
                    { YuvChromaSubsampling::Yuv420 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                rgbx_to_yuv::<
                    { YuvSourceChannels::Bgra as u8 },
                    { YuvChromaSubsampling::Yuv420 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
    };
    dispatcher(planar_image, bgra, bgra_stride, bit_depth, range, matrix)
}

/// Convert RGB image data to YUV 444 planar format with 10 or 12 bit depth.
///
/// This function performs RGB to YUV conversion and stores the result in YUV444 planar format,
/// with separate planes for Y (luminance), U (chrominance), and V (chrominance) components.
///
/// # Arguments
///
/// * `planar_image` - Target planar image.
/// * `rgb` - The input RGB image data slice.
/// * `rgb_stride` - The stride (components per row) for the RGB image data.
/// * `bit_depth` - Only 10 or 12 bit-depth is supported.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
/// * `endianness` - The endianness of final YUV
/// * `bytes_packing` - position of significant bytes for YUV ( most significant or least significant ) if it in most significant it should be stated as per Apple *kCVPixelFormatType_422YpCbCr10BiPlanarFullRange/kCVPixelFormatType_422YpCbCr10BiPlanarVideoRange*
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input RGB data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn rgb_to_yuv444_p16(
    planar_image: &mut YuvPlanarImageMut<u16>,
    rgb: &[u16],
    rgb_stride: u32,
    bit_depth: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    endianness: YuvEndianness,
    bytes_packing: YuvBytesPacking,
) -> Result<(), YuvError> {
    assert!(
        bit_depth == 10 || bit_depth == 12,
        "Only 10 and 12 bit depth is supported"
    );
    let dispatcher = match endianness {
        YuvEndianness::BigEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                rgbx_to_yuv::<
                    { YuvSourceChannels::Rgb as u8 },
                    { YuvChromaSubsampling::Yuv444 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                rgbx_to_yuv::<
                    { YuvSourceChannels::Rgb as u8 },
                    { YuvChromaSubsampling::Yuv444 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
        YuvEndianness::LittleEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                rgbx_to_yuv::<
                    { YuvSourceChannels::Rgb as u8 },
                    { YuvChromaSubsampling::Yuv444 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                rgbx_to_yuv::<
                    { YuvSourceChannels::Rgb as u8 },
                    { YuvChromaSubsampling::Yuv444 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
    };
    dispatcher(planar_image, rgb, rgb_stride, bit_depth, range, matrix)
}

/// Convert BGR image data to YUV 444 planar format with 10 or 12 bit depth.
///
/// This function performs BGR to YUV conversion and stores the result in YUV444 planar format,
/// with separate planes for Y (luminance), U (chrominance), and V (chrominance) components.
///
/// # Arguments
///
/// * `planar_image` - Target planar image.
/// * `bgr` - The input BGR image data slice.
/// * `bgr_stride` - The stride (components per row) for the BGR image data.
/// * `bit_depth` - Only 10 or 12 bit-depth is supported.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
/// * `endianness` - The endianness of final YUV
/// * `bytes_packing` - position of significant bytes for YUV ( most significant or least significant ) if it in most significant it should be stated as per Apple *kCVPixelFormatType_422YpCbCr10BiPlanarFullRange/kCVPixelFormatType_422YpCbCr10BiPlanarVideoRange*
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGR data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn bgr_to_yuv444_p16(
    planar_image: &mut YuvPlanarImageMut<u16>,
    bgr: &[u16],
    bgr_stride: u32,
    bit_depth: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    endianness: YuvEndianness,
    bytes_packing: YuvBytesPacking,
) -> Result<(), YuvError> {
    assert!(
        bit_depth == 10 || bit_depth == 12,
        "Only 10 and 12 bit depth is supported"
    );
    let dispatcher = match endianness {
        YuvEndianness::BigEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                rgbx_to_yuv::<
                    { YuvSourceChannels::Bgr as u8 },
                    { YuvChromaSubsampling::Yuv444 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                rgbx_to_yuv::<
                    { YuvSourceChannels::Bgr as u8 },
                    { YuvChromaSubsampling::Yuv444 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
        YuvEndianness::LittleEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                rgbx_to_yuv::<
                    { YuvSourceChannels::Bgr as u8 },
                    { YuvChromaSubsampling::Yuv444 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                rgbx_to_yuv::<
                    { YuvSourceChannels::Bgr as u8 },
                    { YuvChromaSubsampling::Yuv444 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
    };
    dispatcher(planar_image, bgr, bgr_stride, bit_depth, range, matrix)
}

/// Convert RGBA image data to YUV 444 planar format with 10 or 12 bit depth.
///
/// This function performs RGBA to YUV conversion and stores the result in YUV444 planar format,
/// with separate planes for Y (luminance), U (chrominance), and V (chrominance) components.
///
/// # Arguments
///
/// * `planar_image` - Target planar image.
/// * `rgba` - The input RGBA image data slice.
/// * `rgba_stride` - The stride (components per row) for the RGBA image data.
/// * `bit_depth` - Only 10 or 12 bit-depth is supported.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
/// * `endianness` - The endianness of final YUV
/// * `bytes_packing` - position of significant bytes for YUV ( most significant or least significant ) if it in most significant it should be stated as per Apple *kCVPixelFormatType_422YpCbCr10BiPlanarFullRange/kCVPixelFormatType_422YpCbCr10BiPlanarVideoRange*
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input RGBA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn rgba_to_yuv444_p16(
    planar_image: &mut YuvPlanarImageMut<u16>,
    rgba: &[u16],
    rgba_stride: u32,
    bit_depth: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    endianness: YuvEndianness,
    bytes_packing: YuvBytesPacking,
) -> Result<(), YuvError> {
    assert!(
        bit_depth == 10 || bit_depth == 12,
        "Only 10 and 12 bit depth is supported"
    );
    let dispatcher = match endianness {
        YuvEndianness::BigEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                rgbx_to_yuv::<
                    { YuvSourceChannels::Rgba as u8 },
                    { YuvChromaSubsampling::Yuv444 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                rgbx_to_yuv::<
                    { YuvSourceChannels::Rgba as u8 },
                    { YuvChromaSubsampling::Yuv444 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
        YuvEndianness::LittleEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                rgbx_to_yuv::<
                    { YuvSourceChannels::Rgba as u8 },
                    { YuvChromaSubsampling::Yuv444 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                rgbx_to_yuv::<
                    { YuvSourceChannels::Rgba as u8 },
                    { YuvChromaSubsampling::Yuv444 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
    };
    dispatcher(planar_image, rgba, rgba_stride, bit_depth, range, matrix)
}

/// Convert BGRA image data to YUV 444 planar format with 10 or 12 bit depth.
///
/// This function performs BGRA to YUV conversion and stores the result in YUV444 planar format,
/// with separate planes for Y (luminance), U (chrominance), and V (chrominance) components.
///
/// # Arguments
///
/// * `planar_image` - Target planar image.
/// * `bgra` - The input BGRA image data slice.
/// * `bgra_stride` - The stride (components per row) for the BGRA image data.
/// * `bit_depth` - Only 10 or 12 bit-depth is supported.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
/// * `endianness` - The endianness of final YUV
/// * `bytes_packing` - position of significant bytes for YUV ( most significant or least significant ) if it in most significant it should be stated as per Apple *kCVPixelFormatType_422YpCbCr10BiPlanarFullRange/kCVPixelFormatType_422YpCbCr10BiPlanarVideoRange*
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGRA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn bgra_to_yuv444_p16(
    planar_image: &mut YuvPlanarImageMut<u16>,
    bgra: &[u16],
    bgra_stride: u32,
    bit_depth: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    endianness: YuvEndianness,
    bytes_packing: YuvBytesPacking,
) -> Result<(), YuvError> {
    assert!(
        bit_depth == 10 || bit_depth == 12,
        "Only 10 and 12 bit depth is supported"
    );
    let dispatcher = match endianness {
        YuvEndianness::BigEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                rgbx_to_yuv::<
                    { YuvSourceChannels::Bgra as u8 },
                    { YuvChromaSubsampling::Yuv444 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                rgbx_to_yuv::<
                    { YuvSourceChannels::Bgra as u8 },
                    { YuvChromaSubsampling::Yuv444 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
        YuvEndianness::LittleEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                rgbx_to_yuv::<
                    { YuvSourceChannels::Bgra as u8 },
                    { YuvChromaSubsampling::Yuv444 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                rgbx_to_yuv::<
                    { YuvSourceChannels::Bgra as u8 },
                    { YuvChromaSubsampling::Yuv444 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
    };
    dispatcher(planar_image, bgra, bgra_stride, bit_depth, range, matrix)
}
