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
#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    feature = "nightly_avx512"
))]
use crate::avx512bw::avx512_yuv_p16_to_rgba8_row;
use crate::built_coefficients::get_built_inverse_transform;
#[allow(dead_code, unused_imports)]
use crate::internals::ProcessedOffset;
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use crate::neon::neon_yuv_p16_to_rgba_row;
use crate::numerics::to_ne;
use crate::yuv_error::check_rgba_destination;
use crate::yuv_support::{
    get_inverse_transform, get_yuv_range, YuvBytesPacking, YuvChromaSubsampling, YuvEndianness,
    YuvRange, YuvSourceChannels, YuvStandardMatrix,
};
use crate::{YuvError, YuvPlanarImage};
#[cfg(feature = "rayon")]
use rayon::iter::{IndexedParallelIterator, ParallelIterator};
#[cfg(feature = "rayon")]
use rayon::prelude::{ParallelSlice, ParallelSliceMut};

fn yuv_p16_to_image_ant<
    const DESTINATION_CHANNELS: u8,
    const SAMPLING: u8,
    const ENDIANNESS: u8,
    const BYTES_POSITION: u8,
    const BIT_DEPTH: usize,
>(
    image: &YuvPlanarImage<u16>,
    rgba: &mut [u8],
    rgba_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) -> Result<(), YuvError> {
    let dst_chans: YuvSourceChannels = DESTINATION_CHANNELS.into();
    let channels = dst_chans.get_channels_count();

    let chroma_subsampling: YuvChromaSubsampling = SAMPLING.into();

    assert!(
        BIT_DEPTH == 10 || BIT_DEPTH == 12,
        "YUV16 -> RGB8 implemented only 10 and 12 bit depth"
    );

    image.check_constraints(chroma_subsampling)?;
    check_rgba_destination(rgba, rgba_stride, image.width, image.height, channels)?;

    let chroma_range = get_yuv_range(BIT_DEPTH as u32, range);
    let kr_kb = matrix.get_kr_kb();
    const PRECISION: i32 = 13;
    let i_transform = if let Some(stored) =
        get_built_inverse_transform(PRECISION as u32, BIT_DEPTH as u32, range, matrix)
    {
        stored
    } else {
        let transform = get_inverse_transform(
            BIT_DEPTH as u32,
            chroma_range.range_y,
            chroma_range.range_uv,
            kr_kb.kr,
            kr_kb.kb,
        );
        transform.to_integers(PRECISION as u32)
    };
    let cr_coef = i_transform.cr_coef;
    let cb_coef = i_transform.cb_coef;
    let y_coef = i_transform.y_coef;
    let g_coef_1 = i_transform.g_coeff_1;
    let g_coef_2 = i_transform.g_coeff_2;

    let bias_y = chroma_range.bias_y as i32;
    let bias_uv = chroma_range.bias_uv as i32;

    let msb_shift = (16 - BIT_DEPTH) as i32;

    #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "sse"))]
    let use_sse = std::arch::is_x86_feature_detected!("sse4.1");
    #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "avx"))]
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
    #[cfg(all(
        any(target_arch = "x86", target_arch = "x86_64"),
        feature = "nightly_avx512"
    ))]
    let avx512_wide_row_handler = if use_vbmi {
        avx512_yuv_p16_to_rgba8_row::<
            DESTINATION_CHANNELS,
            SAMPLING,
            ENDIANNESS,
            BYTES_POSITION,
            BIT_DEPTH,
            PRECISION,
            true,
        >
    } else {
        avx512_yuv_p16_to_rgba8_row::<
            DESTINATION_CHANNELS,
            SAMPLING,
            ENDIANNESS,
            BYTES_POSITION,
            BIT_DEPTH,
            PRECISION,
            false,
        >
    };

    #[inline(always)]
    /// Saturating rounding shift right against bit depth
    fn qrshr<const BIT_DEPTH: usize>(val: i32) -> i32 {
        let total_shift = PRECISION + (BIT_DEPTH as i32 - 8);
        let rounding: i32 = 1 << (total_shift - 1);
        let max_value: i32 = (1 << BIT_DEPTH) - 1;
        ((val + rounding) >> total_shift).min(max_value).max(0)
    }

    let process_wide_row =
        |_y_plane: &[u16], _u_plane: &[u16], _v_plane: &[u16], _rgba: &mut [u8]| {
            let mut _cx = 0usize;
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            {
                let mut _v_offset = ProcessedOffset { cx: 0, ux: 0 };
                {
                    #[cfg(feature = "nightly_avx512")]
                    if use_avx512 {
                        unsafe {
                            let offset = avx512_wide_row_handler(
                                _y_plane,
                                _u_plane,
                                _v_plane,
                                _rgba,
                                image.width,
                                &chroma_range,
                                &i_transform,
                                _v_offset.cx,
                                _v_offset.ux,
                            );
                            _v_offset = offset;
                        }
                    }
                    #[cfg(feature = "avx")]
                    if use_avx {
                        use crate::avx2::avx_yuv_p16_to_rgba8_row;
                        unsafe {
                            let offset = avx_yuv_p16_to_rgba8_row::<
                                DESTINATION_CHANNELS,
                                SAMPLING,
                                ENDIANNESS,
                                BYTES_POSITION,
                                BIT_DEPTH,
                                PRECISION,
                            >(
                                _y_plane,
                                _u_plane,
                                _v_plane,
                                _rgba,
                                image.width,
                                &chroma_range,
                                &i_transform,
                                _v_offset.cx,
                                _v_offset.ux,
                            );
                            _v_offset = offset;
                        }
                    }
                    #[cfg(feature = "sse")]
                    if use_sse {
                        use crate::sse::sse_yuv_p16_to_rgba8_row;
                        unsafe {
                            let offset = sse_yuv_p16_to_rgba8_row::<
                                DESTINATION_CHANNELS,
                                SAMPLING,
                                ENDIANNESS,
                                BYTES_POSITION,
                                BIT_DEPTH,
                                PRECISION,
                            >(
                                _y_plane,
                                _u_plane,
                                _v_plane,
                                _rgba,
                                image.width,
                                &chroma_range,
                                &i_transform,
                                _v_offset.cx,
                                _v_offset.ux,
                            );
                            _v_offset = offset;
                        }
                    }
                }
                _cx = _v_offset.cx;
            }
            #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
            {
                unsafe {
                    let offset = neon_yuv_p16_to_rgba_row::<
                        DESTINATION_CHANNELS,
                        SAMPLING,
                        ENDIANNESS,
                        BYTES_POSITION,
                        BIT_DEPTH,
                        PRECISION,
                    >(
                        _y_plane,
                        _u_plane,
                        _v_plane,
                        _rgba,
                        0,
                        image.width,
                        &chroma_range,
                        &i_transform,
                        0,
                        0,
                    );
                    _cx = offset.cx;
                }
            }
            _cx
        };

    let process_halved_chroma_row = |y_plane: &[u16],
                                     u_plane: &[u16],
                                     v_plane: &[u16],
                                     rgba: &mut [u8]| {
        let cx = process_wide_row(y_plane, u_plane, v_plane, rgba);

        for (((rgba, y_src), &u_src), &v_src) in rgba
            .chunks_exact_mut(channels * 2)
            .zip(y_plane.chunks_exact(2))
            .zip(u_plane.iter())
            .zip(v_plane.iter())
            .skip(cx / 2)
        {
            let y_value0 =
                (to_ne::<ENDIANNESS, BYTES_POSITION>(y_src[0], msb_shift) as i32 - bias_y) * y_coef;
            let cb_value = to_ne::<ENDIANNESS, BYTES_POSITION>(u_src, msb_shift) as i32 - bias_uv;
            let cr_value = to_ne::<ENDIANNESS, BYTES_POSITION>(v_src, msb_shift) as i32 - bias_uv;

            let r0 = qrshr::<BIT_DEPTH>(y_value0 + cr_coef * cr_value);
            let b0 = qrshr::<BIT_DEPTH>(y_value0 + cb_coef * cb_value);
            let g0 = qrshr::<BIT_DEPTH>(y_value0 - g_coef_1 * cr_value - g_coef_2 * cb_value);

            let rgba0 = &mut rgba[0..channels];

            rgba0[dst_chans.get_r_channel_offset()] = r0 as u8;
            rgba0[dst_chans.get_g_channel_offset()] = g0 as u8;
            rgba0[dst_chans.get_b_channel_offset()] = b0 as u8;
            if dst_chans.has_alpha() {
                rgba0[dst_chans.get_a_channel_offset()] = 255;
            }

            let y_value1 =
                (to_ne::<ENDIANNESS, BYTES_POSITION>(y_src[1], msb_shift) as i32 - bias_y) * y_coef;

            let r1 = qrshr::<BIT_DEPTH>(y_value1 + cr_coef * cr_value);
            let b1 = qrshr::<BIT_DEPTH>(y_value1 + cb_coef * cb_value);
            let g1 = qrshr::<BIT_DEPTH>(y_value1 - g_coef_1 * cr_value - g_coef_2 * cb_value);

            let rgba1 = &mut rgba[channels..channels * 2];

            rgba1[dst_chans.get_r_channel_offset()] = r1 as u8;
            rgba1[dst_chans.get_g_channel_offset()] = g1 as u8;
            rgba1[dst_chans.get_b_channel_offset()] = b1 as u8;
            if dst_chans.has_alpha() {
                rgba1[dst_chans.get_a_channel_offset()] = 255;
            }
        }

        if image.width & 1 != 0 {
            let y_value0 = (to_ne::<ENDIANNESS, BYTES_POSITION>(*y_plane.last().unwrap(), msb_shift)
                as i32
                - bias_y)
                * y_coef;
            let cb_value = to_ne::<ENDIANNESS, BYTES_POSITION>(*u_plane.last().unwrap(), msb_shift)
                as i32
                - bias_uv;
            let cr_value = to_ne::<ENDIANNESS, BYTES_POSITION>(*v_plane.last().unwrap(), msb_shift)
                as i32
                - bias_uv;
            let rgba = rgba.chunks_exact_mut(channels).last().unwrap();
            let rgba0 = &mut rgba[0..channels];

            let r0 = qrshr::<BIT_DEPTH>(y_value0 + cr_coef * cr_value);
            let b0 = qrshr::<BIT_DEPTH>(y_value0 + cb_coef * cb_value);
            let g0 = qrshr::<BIT_DEPTH>(y_value0 - g_coef_1 * cr_value - g_coef_2 * cb_value);
            rgba0[dst_chans.get_r_channel_offset()] = r0 as u8;
            rgba0[dst_chans.get_g_channel_offset()] = g0 as u8;
            rgba0[dst_chans.get_b_channel_offset()] = b0 as u8;
            if dst_chans.has_alpha() {
                rgba0[dst_chans.get_a_channel_offset()] = 255;
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
            let cx = process_wide_row(y_plane, u_plane, v_plane, rgba);

            for (((rgba, &y_src), &u_src), &v_src) in rgba
                .chunks_exact_mut(channels)
                .zip(y_plane.iter())
                .zip(u_plane.iter())
                .zip(v_plane.iter())
                .skip(cx)
            {
                let y_value = (to_ne::<ENDIANNESS, BYTES_POSITION>(y_src, msb_shift) as i32
                    - bias_y)
                    * y_coef;
                let cb_value =
                    to_ne::<ENDIANNESS, BYTES_POSITION>(u_src, msb_shift) as i32 - bias_uv;
                let cr_value =
                    to_ne::<ENDIANNESS, BYTES_POSITION>(v_src, msb_shift) as i32 - bias_uv;

                let r = qrshr::<BIT_DEPTH>(y_value + cr_coef * cr_value);
                let b = qrshr::<BIT_DEPTH>(y_value + cb_coef * cb_value);
                let g = qrshr::<BIT_DEPTH>(y_value - g_coef_1 * cr_value - g_coef_2 * cb_value);

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
            for (rgba, y_plane) in rgba
                .chunks_exact_mut(rgba_stride as usize)
                .zip(y_plane.chunks_exact(image.y_stride as usize))
            {
                process_halved_chroma_row(
                    &y_plane[0..image.width as usize],
                    &u_plane[0..(image.width as usize).div_ceil(2)],
                    &v_plane[0..(image.width as usize).div_ceil(2)],
                    &mut rgba[0..image.width as usize * channels],
                );
            }
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

macro_rules! build_cnv {
    ($method: ident, $px_fmt: expr, $sampling: expr,$bit_depth: expr, $sampling_written: expr, $px_written: expr, $px_written_small: expr, $endian: expr) => {
        #[doc = concat!("
Convert ",$sampling_written, " planar format with ", $bit_depth," bit pixel format to ", $px_written," 8-bit format.

This function takes ", $sampling_written, " planar data with ",$bit_depth," bit precision.
and converts it to ", $px_written," format with 8 bit-depth precision per channel

# Arguments

* `planar_image` - Source ",$sampling_written," planar image.
* `", $px_written_small, "` - A mutable slice to store the converted ", $px_written," 8 bit-depth format.
* `", $px_written_small, "_stride` - The stride (components per row) for ", $px_written," 8 bit-depth format.
* `range` - The YUV range (limited or full).
* `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).

# Panics

This function panics if the lengths of the planes or the input ", $px_written," data are not valid based
on the specified width, height, and strides, or if invalid YUV range or matrix is provided.")]
        pub fn $method(
            planar_image: &YuvPlanarImage<u16>,
            dst: &mut [u8],
            dst_stride: u32,
            range: YuvRange,
            matrix: YuvStandardMatrix,
        ) -> Result<(), YuvError> {
             yuv_p16_to_image_ant::<{ $px_fmt as u8 }, { $sampling as u8 }, { $endian as u8 }, { YuvBytesPacking::LeastSignificantBytes as u8 }, $bit_depth>(
                planar_image,
                dst,
                dst_stride,
                range,
                matrix,
             )
        }
    };
}

build_cnv!(
    i010_to_rgba,
    YuvChromaSubsampling::Yuv420,
    YuvSourceChannels::Rgba,
    10,
    "YUV 420 10-bit",
    "RGBA",
    "rgba",
    YuvEndianness::LittleEndian
);
#[cfg(feature = "big_endian")]
build_cnv!(
    i010_be_to_rgba,
    YuvChromaSubsampling::Yuv420,
    YuvSourceChannels::Rgba,
    10,
    "YUV 420 10-bit",
    "RGBA",
    "rgba",
    YuvEndianness::BigEndian
);

build_cnv!(
    i010_to_bgra,
    YuvChromaSubsampling::Yuv420,
    YuvSourceChannels::Bgra,
    10,
    "YUV 420 10-bit",
    "BGRA",
    "bgra",
    YuvEndianness::LittleEndian
);
#[cfg(feature = "big_endian")]
build_cnv!(
    i010_be_to_bgra,
    YuvChromaSubsampling::Yuv420,
    YuvSourceChannels::Rgba,
    10,
    "YUV 420 10-bit",
    "BGRA",
    "bgra",
    YuvEndianness::BigEndian
);

build_cnv!(
    i010_to_rgb,
    YuvChromaSubsampling::Yuv420,
    YuvSourceChannels::Rgb,
    10,
    "YUV 420 10-bit",
    "RGB",
    "rgb",
    YuvEndianness::LittleEndian
);
#[cfg(feature = "big_endian")]
build_cnv!(
    i010_be_to_rgb,
    YuvChromaSubsampling::Yuv420,
    YuvSourceChannels::Rgb,
    10,
    "YUV 420 10-bit",
    "RGB",
    "rgb",
    YuvEndianness::BigEndian
);

build_cnv!(
    i010_to_bgr,
    YuvChromaSubsampling::Yuv420,
    YuvSourceChannels::Bgr,
    10,
    "YUV 420 10-bit",
    "BGR",
    "bgr",
    YuvEndianness::LittleEndian
);
#[cfg(feature = "big_endian")]
build_cnv!(
    i010_be_to_bgr,
    YuvChromaSubsampling::Yuv420,
    YuvSourceChannels::Bgr,
    10,
    "YUV 420 10-bit",
    "BGR",
    "bgr",
    YuvEndianness::BigEndian
);

build_cnv!(
    i210_to_rgba,
    YuvChromaSubsampling::Yuv422,
    YuvSourceChannels::Rgba,
    10,
    "YUV 420 10-bit",
    "RGBA",
    "rgba",
    YuvEndianness::LittleEndian
);
#[cfg(feature = "big_endian")]
build_cnv!(
    i210_be_to_rgba,
    YuvChromaSubsampling::Yuv422,
    YuvSourceChannels::Rgba,
    10,
    "YUV 420 10-bit",
    "RGBA",
    "rgba",
    YuvEndianness::BigEndian
);

build_cnv!(
    i210_to_bgra,
    YuvChromaSubsampling::Yuv422,
    YuvSourceChannels::Bgra,
    10,
    "YUV 420 10-bit",
    "BGRA",
    "bgra",
    YuvEndianness::LittleEndian
);
#[cfg(feature = "big_endian")]
build_cnv!(
    i210_be_to_bgra,
    YuvChromaSubsampling::Yuv422,
    YuvSourceChannels::Rgba,
    10,
    "YUV 420 10-bit",
    "BGRA",
    "bgra",
    YuvEndianness::BigEndian
);

build_cnv!(
    i210_to_rgb,
    YuvChromaSubsampling::Yuv422,
    YuvSourceChannels::Rgb,
    10,
    "YUV 420 10-bit",
    "RGB",
    "rgb",
    YuvEndianness::LittleEndian
);
#[cfg(feature = "big_endian")]
build_cnv!(
    i210_be_to_rgb,
    YuvChromaSubsampling::Yuv422,
    YuvSourceChannels::Rgb,
    10,
    "YUV 420 10-bit",
    "RGB",
    "rgb",
    YuvEndianness::BigEndian
);

build_cnv!(
    i210_to_bgr,
    YuvChromaSubsampling::Yuv422,
    YuvSourceChannels::Bgr,
    10,
    "YUV 420 10-bit",
    "BGR",
    "bgr",
    YuvEndianness::LittleEndian
);
#[cfg(feature = "big_endian")]
build_cnv!(
    i210_be_to_bgr,
    YuvChromaSubsampling::Yuv422,
    YuvSourceChannels::Bgr,
    10,
    "YUV 420 10-bit",
    "BGR",
    "bgr",
    YuvEndianness::BigEndian
);

build_cnv!(
    i012_to_rgba,
    YuvChromaSubsampling::Yuv420,
    YuvSourceChannels::Rgba,
    12,
    "YUV 420 12-bit",
    "RGBA",
    "rgba",
    YuvEndianness::LittleEndian
);
#[cfg(feature = "big_endian")]
build_cnv!(
    i012_be_to_rgba,
    YuvChromaSubsampling::Yuv420,
    YuvSourceChannels::Rgba,
    12,
    "YUV 420 12-bit",
    "RGBA",
    "rgba",
    YuvEndianness::BigEndian
);

build_cnv!(
    i012_to_bgra,
    YuvChromaSubsampling::Yuv420,
    YuvSourceChannels::Bgra,
    12,
    "YUV 420 12-bit",
    "BGRA",
    "bgra",
    YuvEndianness::LittleEndian
);
#[cfg(feature = "big_endian")]
build_cnv!(
    i012_be_to_bgra,
    YuvChromaSubsampling::Yuv420,
    YuvSourceChannels::Rgba,
    12,
    "YUV 420 12-bit",
    "BGRA",
    "bgra",
    YuvEndianness::BigEndian
);

build_cnv!(
    i012_to_rgb,
    YuvChromaSubsampling::Yuv420,
    YuvSourceChannels::Rgb,
    12,
    "YUV 420 12-bit",
    "RGB",
    "rgb",
    YuvEndianness::LittleEndian
);
#[cfg(feature = "big_endian")]
build_cnv!(
    i012_be_to_rgb,
    YuvChromaSubsampling::Yuv420,
    YuvSourceChannels::Rgb,
    12,
    "YUV 420 12-bit",
    "RGB",
    "rgb",
    YuvEndianness::BigEndian
);

build_cnv!(
    i012_to_bgr,
    YuvChromaSubsampling::Yuv420,
    YuvSourceChannels::Bgr,
    12,
    "YUV 420 12-bit",
    "BGR",
    "bgr",
    YuvEndianness::LittleEndian
);
#[cfg(feature = "big_endian")]
build_cnv!(
    i012_be_to_bgr,
    YuvChromaSubsampling::Yuv420,
    YuvSourceChannels::Bgr,
    12,
    "YUV 420 12-bit",
    "BGR",
    "bgr",
    YuvEndianness::BigEndian
);

build_cnv!(
    i212_to_rgba,
    YuvChromaSubsampling::Yuv422,
    YuvSourceChannels::Rgba,
    12,
    "YUV 420 12-bit",
    "RGBA",
    "rgba",
    YuvEndianness::LittleEndian
);
#[cfg(feature = "big_endian")]
build_cnv!(
    i212_be_to_rgba,
    YuvChromaSubsampling::Yuv422,
    YuvSourceChannels::Rgba,
    12,
    "YUV 420 12-bit",
    "RGBA",
    "rgba",
    YuvEndianness::BigEndian
);

build_cnv!(
    i212_to_bgra,
    YuvChromaSubsampling::Yuv422,
    YuvSourceChannels::Bgra,
    12,
    "YUV 420 12-bit",
    "BGRA",
    "bgra",
    YuvEndianness::LittleEndian
);
#[cfg(feature = "big_endian")]
build_cnv!(
    i212_be_to_bgra,
    YuvChromaSubsampling::Yuv422,
    YuvSourceChannels::Rgba,
    12,
    "YUV 420 12-bit",
    "BGRA",
    "bgra",
    YuvEndianness::BigEndian
);

build_cnv!(
    i212_to_rgb,
    YuvChromaSubsampling::Yuv422,
    YuvSourceChannels::Rgb,
    12,
    "YUV 420 12-bit",
    "RGB",
    "rgb",
    YuvEndianness::LittleEndian
);
#[cfg(feature = "big_endian")]
build_cnv!(
    i212_be_to_rgb,
    YuvChromaSubsampling::Yuv422,
    YuvSourceChannels::Rgb,
    12,
    "YUV 420 12-bit",
    "RGB",
    "rgb",
    YuvEndianness::BigEndian
);

build_cnv!(
    i212_to_bgr,
    YuvChromaSubsampling::Yuv422,
    YuvSourceChannels::Bgr,
    12,
    "YUV 420 12-bit",
    "BGR",
    "bgr",
    YuvEndianness::LittleEndian
);
#[cfg(feature = "big_endian")]
build_cnv!(
    i212_be_to_bgr,
    YuvChromaSubsampling::Yuv422,
    YuvSourceChannels::Bgr,
    12,
    "YUV 420 12-bit",
    "BGR",
    "bgr",
    YuvEndianness::BigEndian
);
