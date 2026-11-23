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
use crate::images::{projected_rgba_plane_mut, YuvPlanarProjectionAlpha};
#[allow(unused_imports)]
use crate::internals::ProcessedOffset;
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use crate::neon::neon_yuv_p16_to_rgba_alpha_row;
use crate::numerics::to_ne;
use crate::yuv_error::check_rgba_destination;
use crate::yuv_support::{
    get_yuv_range, search_inverse_transform, YuvBytesPacking, YuvChromaSubsampling, YuvEndianness,
    YuvRange, YuvSourceChannels, YuvStandardMatrix,
};
use crate::{YuvError, YuvPlanarImageWithAlpha};
#[cfg(feature = "rayon")]
use rayon::iter::{IndexedParallelIterator, ParallelIterator};
#[cfg(feature = "rayon")]
use rayon::prelude::{ParallelSlice, ParallelSliceMut};

fn yuv_p16_to_image_alpha_ant<
    const DESTINATION_CHANNELS: u8,
    const SAMPLING: u8,
    const ENDIANNESS: u8,
    const BYTES_POSITION: u8,
    const BIT_DEPTH: usize,
>(
    image: &YuvPlanarImageWithAlpha<u16>,
    rgba: &mut [u8],
    rgba_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) -> Result<(), YuvError> {
    let dst_chans: YuvSourceChannels = DESTINATION_CHANNELS.into();
    let channels = dst_chans.get_channels_count();

    assert!(
        dst_chans != YuvSourceChannels::Rgb && dst_chans != YuvSourceChannels::Bgr,
        "Cannot call YUV p16 to Rgb8 with alpha without real alpha"
    );
    assert!(
        BIT_DEPTH == 10 || BIT_DEPTH == 12,
        "YUV16 -> RGB8 implemented only 10 and 12 bit depth"
    );

    let chroma_subsampling: YuvChromaSubsampling = SAMPLING.into();

    image.check_constraints(chroma_subsampling)?;
    check_rgba_destination(rgba, rgba_stride, image.width, image.height, channels)?;

    let chroma_range = get_yuv_range(BIT_DEPTH as u32, range);
    let kr_kb = matrix.get_kr_kb();
    const PRECISION: i32 = 13;
    let i_transform = search_inverse_transform(
        PRECISION,
        BIT_DEPTH as u32,
        range,
        matrix,
        chroma_range,
        kr_kb,
    );
    let cr_coef = i_transform.cr_coef;
    let cb_coef = i_transform.cb_coef;
    let y_coef = i_transform.y_coef;
    let g_coef_1 = i_transform.g_coeff_1;
    let g_coef_2 = i_transform.g_coeff_2;

    let bias_y = chroma_range.bias_y as i32;
    let bias_uv = chroma_range.bias_uv as i32;

    #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "sse"))]
    let use_sse = std::arch::is_x86_feature_detected!("sse4.1");
    #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), feature = "avx"))]
    let use_avx = std::arch::is_x86_feature_detected!("avx2");

    let msb_shift = (16 - BIT_DEPTH) as i32;
    let store_shift = BIT_DEPTH - 8;

    #[inline(always)]
    /// Saturating rounding shift right against bit depth
    fn qrshr<const BIT_DEPTH: usize>(val: i32) -> i32 {
        let total_shift = PRECISION + (BIT_DEPTH as i32 - 8);
        let rounding: i32 = 1 << (total_shift - 1);
        let max_value: i32 = (1 << BIT_DEPTH) - 1;
        ((val + rounding) >> total_shift).min(max_value).max(0)
    }

    let process_wide_row = |_y_plane: &[u16],
                            _u_plane: &[u16],
                            _v_plane: &[u16],
                            _a_plane: &[u16],
                            _rgba: &mut [u8]| {
        let mut _cx = 0usize;
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            let mut _v_offset = ProcessedOffset { cx: 0, ux: 0 };
            {
                #[cfg(feature = "avx")]
                if use_avx {
                    use crate::avx2::avx_yuv_p16_to_rgba8_alpha_row;
                    unsafe {
                        let offset = avx_yuv_p16_to_rgba8_alpha_row::<
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
                            _a_plane,
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
                    use crate::sse::sse_yuv_p16_to_rgba8_alpha_row;
                    unsafe {
                        let offset = sse_yuv_p16_to_rgba8_alpha_row::<
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
                            _a_plane,
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
                let offset = neon_yuv_p16_to_rgba_alpha_row::<
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
                    _a_plane,
                    _rgba,
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
                                     a_plane: &[u16],
                                     rgba: &mut [u8]| {
        let cx = process_wide_row(y_plane, u_plane, v_plane, a_plane, rgba);

        for ((((rgba, y_src), &u_src), &v_src), a_src) in rgba
            .chunks_exact_mut(channels * 2)
            .zip(y_plane.chunks_exact(2))
            .zip(u_plane.iter())
            .zip(v_plane.iter())
            .zip(a_plane.chunks_exact(2))
            .skip(cx / 2)
        {
            let y_value0 =
                (to_ne::<ENDIANNESS, BYTES_POSITION>(y_src[0], msb_shift) as i32 - bias_y) * y_coef;
            let cb_value = to_ne::<ENDIANNESS, BYTES_POSITION>(u_src, msb_shift) as i32 - bias_uv;
            let cr_value = to_ne::<ENDIANNESS, BYTES_POSITION>(v_src, msb_shift) as i32 - bias_uv;

            let r0 = qrshr::<BIT_DEPTH>(y_value0 + cr_coef * cr_value);
            let b0 = qrshr::<BIT_DEPTH>(y_value0 + cb_coef * cb_value);
            let g0 = qrshr::<BIT_DEPTH>(y_value0 - g_coef_1 * cr_value - g_coef_2 * cb_value);

            let rgba0 = &mut rgba[..channels];

            rgba0[dst_chans.get_r_channel_offset()] = r0 as u8;
            rgba0[dst_chans.get_g_channel_offset()] = g0 as u8;
            rgba0[dst_chans.get_b_channel_offset()] = b0 as u8;
            rgba0[dst_chans.get_a_channel_offset()] = (a_src[0] >> store_shift).min(255) as u8;

            let y_value1 =
                (to_ne::<ENDIANNESS, BYTES_POSITION>(y_src[1], msb_shift) as i32 - bias_y) * y_coef;

            let r1 = qrshr::<BIT_DEPTH>(y_value1 + cr_coef * cr_value);
            let b1 = qrshr::<BIT_DEPTH>(y_value1 + cb_coef * cb_value);
            let g1 = qrshr::<BIT_DEPTH>(y_value1 - g_coef_1 * cr_value - g_coef_2 * cb_value);

            let rgba1 = &mut rgba[channels..channels * 2];

            rgba1[dst_chans.get_r_channel_offset()] = r1 as u8;
            rgba1[dst_chans.get_g_channel_offset()] = g1 as u8;
            rgba1[dst_chans.get_b_channel_offset()] = b1 as u8;
            rgba1[dst_chans.get_a_channel_offset()] = (a_src[1] >> store_shift).min(255) as u8;
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
            let a_value = *a_plane.last().unwrap();
            let rgba = rgba.chunks_exact_mut(channels).last().unwrap();
            let rgba0 = &mut rgba[..channels];

            let r0 = qrshr::<BIT_DEPTH>(y_value0 + cr_coef * cr_value);
            let b0 = qrshr::<BIT_DEPTH>(y_value0 + cb_coef * cb_value);
            let g0 = qrshr::<BIT_DEPTH>(y_value0 - g_coef_1 * cr_value - g_coef_2 * cb_value);
            rgba0[dst_chans.get_r_channel_offset()] = r0 as u8;
            rgba0[dst_chans.get_g_channel_offset()] = g0 as u8;
            rgba0[dst_chans.get_b_channel_offset()] = b0 as u8;
            rgba0[dst_chans.get_a_channel_offset()] = (a_value >> store_shift).min(255) as u8;
        }
    };

    // Destination and source buffers only have to hold `stride * (height - 1) + row`
    // elements, so the last row may be shorter than a stride. Iterate with `chunks` over
    // buffers trimmed to exactly `height` rows so the last row is neither dropped nor
    // followed by extra rows from oversized buffers.
    let YuvPlanarProjectionAlpha {
        y: y_plane,
        u: u_plane,
        v: v_plane,
        a: a_plane,
    } = image.projected_planes(chroma_subsampling);
    let rgba = projected_rgba_plane_mut(rgba, image.width, image.height, rgba_stride, dst_chans);

    if chroma_subsampling == YuvChromaSubsampling::Yuv444 {
        let iter;
        #[cfg(feature = "rayon")]
        {
            iter = rgba
                .par_chunks_mut(rgba_stride as usize)
                .zip(y_plane.par_chunks(image.y_stride as usize))
                .zip(a_plane.par_chunks(image.a_stride as usize))
                .zip(u_plane.par_chunks(image.u_stride as usize))
                .zip(v_plane.par_chunks(image.v_stride as usize));
        }
        #[cfg(not(feature = "rayon"))]
        {
            iter = rgba
                .chunks_mut(rgba_stride as usize)
                .zip(y_plane.chunks(image.y_stride as usize))
                .zip(a_plane.chunks(image.a_stride as usize))
                .zip(u_plane.chunks(image.u_stride as usize))
                .zip(v_plane.chunks(image.v_stride as usize));
        }
        iter.for_each(|((((rgba, y_plane), a_plane), u_plane), v_plane)| {
            let y_plane = &y_plane[..image.width as usize];
            let cx = process_wide_row(y_plane, u_plane, v_plane, a_plane, rgba);

            for ((((rgba, &y_src), &u_src), &v_src), &a_src) in rgba
                .chunks_exact_mut(channels)
                .zip(y_plane.iter())
                .zip(u_plane.iter())
                .zip(v_plane.iter())
                .zip(a_plane.iter())
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
                rgba[dst_chans.get_a_channel_offset()] = (a_src >> store_shift).min(255) as u8;
            }
        });
    } else if chroma_subsampling == YuvChromaSubsampling::Yuv422 {
        let iter;
        #[cfg(feature = "rayon")]
        {
            iter = rgba
                .par_chunks_mut(rgba_stride as usize)
                .zip(y_plane.par_chunks(image.y_stride as usize))
                .zip(a_plane.par_chunks(image.a_stride as usize))
                .zip(u_plane.par_chunks(image.u_stride as usize))
                .zip(v_plane.par_chunks(image.v_stride as usize));
        }
        #[cfg(not(feature = "rayon"))]
        {
            iter = rgba
                .chunks_mut(rgba_stride as usize)
                .zip(y_plane.chunks(image.y_stride as usize))
                .zip(a_plane.chunks(image.a_stride as usize))
                .zip(u_plane.chunks(image.u_stride as usize))
                .zip(v_plane.chunks(image.v_stride as usize));
        }
        iter.for_each(|((((rgba, y_plane), a_plane), u_plane), v_plane)| {
            process_halved_chroma_row(
                &y_plane[..image.width as usize],
                &u_plane[..(image.width as usize).div_ceil(2)],
                &v_plane[..(image.width as usize).div_ceil(2)],
                &a_plane[..image.width as usize],
                &mut rgba[..image.width as usize * channels],
            );
        });
    } else if chroma_subsampling == YuvChromaSubsampling::Yuv420 {
        let iter;
        #[cfg(feature = "rayon")]
        {
            iter = rgba
                .par_chunks_mut(rgba_stride as usize * 2)
                .zip(y_plane.par_chunks(image.y_stride as usize * 2))
                .zip(a_plane.par_chunks(image.a_stride as usize * 2))
                .zip(u_plane.par_chunks(image.u_stride as usize))
                .zip(v_plane.par_chunks(image.v_stride as usize));
        }
        #[cfg(not(feature = "rayon"))]
        {
            iter = rgba
                .chunks_mut(rgba_stride as usize * 2)
                .zip(y_plane.chunks(image.y_stride as usize * 2))
                .zip(a_plane.chunks(image.a_stride as usize * 2))
                .zip(u_plane.chunks(image.u_stride as usize))
                .zip(v_plane.chunks(image.v_stride as usize));
        }
        iter.take(image.height as usize / 2).for_each(
            |((((rgba, y_plane), a_plane), u_plane), v_plane)| {
                for ((rgba, y_plane), a_plane) in rgba
                    .chunks_mut(rgba_stride as usize)
                    .zip(y_plane.chunks(image.y_stride as usize))
                    .zip(a_plane.chunks(image.a_stride as usize))
                {
                    process_halved_chroma_row(
                        &y_plane[0..image.width as usize],
                        &u_plane[0..(image.width as usize).div_ceil(2)],
                        &v_plane[0..(image.width as usize).div_ceil(2)],
                        &a_plane[0..image.width as usize],
                        &mut rgba[0..image.width as usize * channels],
                    );
                }
            },
        );

        if image.height & 1 != 0 {
            let width = image.width as usize;
            let chroma_width = width.div_ceil(2);
            let last_y_row = image.height as usize - 1;
            let last_chroma_row = (image.height as usize).div_ceil(2) - 1;
            let rgba_start = last_y_row * rgba_stride as usize;
            let y_start = last_y_row * image.y_stride as usize;
            let a_start = last_y_row * image.a_stride as usize;
            let u_start = last_chroma_row * image.u_stride as usize;
            let v_start = last_chroma_row * image.v_stride as usize;
            process_halved_chroma_row(
                &y_plane[y_start..y_start + width],
                &u_plane[u_start..u_start + chroma_width],
                &v_plane[v_start..v_start + chroma_width],
                &a_plane[a_start..a_start + width],
                &mut rgba[rgba_start..rgba_start + width * channels],
            );
        }
    } else {
        unreachable!();
    }

    Ok(())
}

macro_rules! d_cnv {
    ($method: ident, $px_fmt: expr, $sampling: expr, $endian: expr, $sampling_written: expr, $px_written: expr, $px_written_small: expr, $bit_depth: expr) => {
        #[doc = concat!("
Convert ",$sampling_written, " planar format with ", stringify!($bit_depth), " bit pixel format to ", $px_written," 8 bit-depth format with interleaving alpha.

This function takes ", $sampling_written, " planar data with ", stringify!($bit_depth), " bit precision and interleaved provided alpha channel,
and converts it to ", $px_written," format with 8 bit-depth precision per channel.

# Arguments

* `planar_image` - Source ",$sampling_written," planar image.
* `", $px_written_small, "` - A mutable slice to store the converted ", $px_written," 8 bit-depth data.
* `", $px_written_small, "_stride` - The stride (components per row) for ", $px_written," 8 bit-depth data.
* `range` - The YUV range (limited or full).
* `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
* `bit_depth` - Bit depth of source YUV planes, only 10 and 12 is supported.

# Panics

This function panics if the lengths of the planes or the input ", $px_written," data are not valid based
on the specified width, height, and strides, or if invalid YUV range or matrix is provided.")]
        pub fn $method(
            planar_image_with_alpha: &YuvPlanarImageWithAlpha<u16>,
            dst: &mut [u8],
            dst_stride: u32,
            range: YuvRange,
            matrix: YuvStandardMatrix,
        ) -> Result<(), YuvError> {
            yuv_p16_to_image_alpha_ant::<{ $px_fmt as u8 },
                            { $sampling as u8 },
                            { $endian as u8 },
                            { YuvBytesPacking::LeastSignificantBytes as u8 }, $bit_depth>(
                planar_image_with_alpha, dst, dst_stride, range, matrix)
        }
    };
}

d_cnv!(
    i010_alpha_to_rgba,
    YuvSourceChannels::Rgba,
    YuvChromaSubsampling::Yuv420,
    YuvEndianness::LittleEndian,
    "I010A",
    "RGBA",
    "rgba",
    10
);
#[cfg(feature = "big_endian")]
d_cnv!(
    i010_be_alpha_to_rgba,
    YuvSourceChannels::Rgba,
    YuvChromaSubsampling::Yuv420,
    YuvEndianness::BigEndian,
    "I010ABE",
    "RGBA",
    "rgba",
    10
);

d_cnv!(
    i210_alpha_to_rgba,
    YuvSourceChannels::Rgba,
    YuvChromaSubsampling::Yuv422,
    YuvEndianness::LittleEndian,
    "I210A",
    "RGBA",
    "rgba",
    10
);
#[cfg(feature = "big_endian")]
d_cnv!(
    i210_alpha_be_to_rgba,
    YuvSourceChannels::Rgba,
    YuvChromaSubsampling::Yuv422,
    YuvEndianness::BigEndian,
    "I210ABE",
    "RGBA",
    "rgba",
    10
);
d_cnv!(
    i410_alpha_to_rgba,
    YuvSourceChannels::Rgba,
    YuvChromaSubsampling::Yuv444,
    YuvEndianness::LittleEndian,
    "I410A",
    "RGBA",
    "rgba",
    10
);
#[cfg(feature = "big_endian")]
d_cnv!(
    i410_be_alpha_to_rgba,
    YuvSourceChannels::Rgba,
    YuvChromaSubsampling::Yuv444,
    YuvEndianness::BigEndian,
    "I410ABE",
    "RGBA",
    "rgba",
    10
);

d_cnv!(
    i012_alpha_to_rgba,
    YuvSourceChannels::Rgba,
    YuvChromaSubsampling::Yuv420,
    YuvEndianness::LittleEndian,
    "I012A",
    "RGBA",
    "rgba",
    12
);
#[cfg(feature = "big_endian")]
d_cnv!(
    i012_be_alpha_to_rgba,
    YuvSourceChannels::Rgba,
    YuvChromaSubsampling::Yuv420,
    YuvEndianness::BigEndian,
    "I012ABE",
    "RGBA",
    "rgba",
    12
);
d_cnv!(
    i212_alpha_to_rgba,
    YuvSourceChannels::Rgba,
    YuvChromaSubsampling::Yuv422,
    YuvEndianness::LittleEndian,
    "I212A",
    "RGBA",
    "rgba",
    12
);
#[cfg(feature = "big_endian")]
d_cnv!(
    i212_be_alpha_to_rgba,
    YuvSourceChannels::Rgba,
    YuvChromaSubsampling::Yuv422,
    YuvEndianness::BigEndian,
    "I212ABE",
    "RGBA",
    "rgba",
    12
);
d_cnv!(
    i412_alpha_to_rgba,
    YuvSourceChannels::Rgba,
    YuvChromaSubsampling::Yuv444,
    YuvEndianness::LittleEndian,
    "I412A",
    "RGBA",
    "rgba",
    12
);
#[cfg(feature = "big_endian")]
d_cnv!(
    i412_be_alpha_to_rgba,
    YuvSourceChannels::Rgba,
    YuvChromaSubsampling::Yuv444,
    YuvEndianness::BigEndian,
    "I412ABE",
    "RGBA",
    "rgba",
    12
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alpha_to_rgba_preserves_row_padding() {
        let width = 9;
        let y = [0; 9];
        let uv = [512; 9];
        let alpha = [1023; 9];
        let image = YuvPlanarImageWithAlpha {
            y_plane: &y,
            y_stride: width,
            u_plane: &uv,
            u_stride: width,
            v_plane: &uv,
            v_stride: width,
            a_plane: &alpha,
            a_stride: width,
            width,
            height: 1,
        };
        let mut rgba = [0xA5; 64];
        let stride = rgba.len() as u32;

        i410_alpha_to_rgba(
            &image,
            &mut rgba,
            stride,
            YuvRange::Full,
            YuvStandardMatrix::Bt709,
        )
        .unwrap();

        let (pixels, padding) = rgba.split_at(width as usize * 4);
        assert_eq!(pixels, [0, 0, 0, 255].repeat(width as usize));
        assert_eq!(padding, [0xA5; 28]);
    }

    type Convert = fn(
        &YuvPlanarImageWithAlpha<u16>,
        &mut [u8],
        u32,
        YuvRange,
        YuvStandardMatrix,
    ) -> Result<(), YuvError>;

    const CONVERTERS: [(Convert, u32, YuvChromaSubsampling); 6] = [
        (i010_alpha_to_rgba, 10, YuvChromaSubsampling::Yuv420),
        (i210_alpha_to_rgba, 10, YuvChromaSubsampling::Yuv422),
        (i410_alpha_to_rgba, 10, YuvChromaSubsampling::Yuv444),
        (i012_alpha_to_rgba, 12, YuvChromaSubsampling::Yuv420),
        (i212_alpha_to_rgba, 12, YuvChromaSubsampling::Yuv422),
        (i412_alpha_to_rgba, 12, YuvChromaSubsampling::Yuv444),
    ];

    /// Converts a black image whose alpha encodes `row * 16 + column + 1` and checks every
    /// pixel plus that destination padding is never touched. Destination is sized to the exact
    /// minimum `stride * (height - 1) + width * 4`, the same as `check_rgba_destination` allows.
    fn check_converter(
        convert: Convert,
        depth: u32,
        sampling: YuvChromaSubsampling,
        width: u32,
        height: u32,
        src_padding: u32,
        dst_padding: u32,
    ) {
        let (chroma_width, chroma_height) = match sampling {
            YuvChromaSubsampling::Yuv420 => (width.div_ceil(2), height.div_ceil(2)),
            YuvChromaSubsampling::Yuv422 => (width.div_ceil(2), height),
            YuvChromaSubsampling::Yuv444 => (width, height),
        };
        let y_stride = width + src_padding;
        let a_stride = width + src_padding;
        let uv_stride = chroma_width + src_padding;
        let expected_alpha = |x: u32, y: u32| ((y * 16 + x) % 255 + 1) as u8;

        let y = vec![0u16; (y_stride * (height - 1) + width) as usize];
        let uv =
            vec![1u16 << (depth - 1); (uv_stride * (chroma_height - 1) + chroma_width) as usize];
        let mut alpha = vec![0u16; (a_stride * (height - 1) + width) as usize];
        for row in 0..height {
            for x in 0..width {
                alpha[(row * a_stride + x) as usize] =
                    (expected_alpha(x, row) as u16) << (depth - 8);
            }
        }
        let image = YuvPlanarImageWithAlpha {
            y_plane: &y,
            y_stride,
            u_plane: &uv,
            u_stride: uv_stride,
            v_plane: &uv,
            v_stride: uv_stride,
            a_plane: &alpha,
            a_stride,
            width,
            height,
        };
        let row_len = width as usize * 4;
        let stride = row_len + dst_padding as usize;
        let mut rgba = vec![0xA5; stride * (height as usize - 1) + row_len];

        convert(
            &image,
            &mut rgba,
            stride as u32,
            YuvRange::Full,
            YuvStandardMatrix::Bt709,
        )
        .unwrap();

        let ctx = format!(
            "depth {depth}, {sampling:?}, {width}x{height}, src pad {src_padding}, dst pad {dst_padding}"
        );
        for (row, line) in rgba.chunks(stride).enumerate() {
            let (pixels, padding) = line.split_at(row_len);
            assert!(
                padding.iter().all(|&x| x == 0xA5),
                "{ctx}: padding of row {row} overwritten"
            );
            for (x, pixel) in pixels.chunks_exact(4).enumerate() {
                assert_eq!(
                    pixel,
                    [0, 0, 0, expected_alpha(x as u32, row as u32)],
                    "{ctx}: row {row}, pixel {x}"
                );
            }
        }
    }

    #[test]
    fn alpha_to_rgba_tightly_packed_rows() {
        for (convert, depth, sampling) in CONVERTERS {
            for width in 1..=65 {
                check_converter(convert, depth, sampling, width, 1, 0, 0);
            }
        }
    }

    #[test]
    fn alpha_to_rgba_strided_rows() {
        for (convert, depth, sampling) in CONVERTERS {
            for width in [1, 2, 7, 8, 9, 15, 16, 17, 31, 33, 63, 65] {
                for height in 1..=5 {
                    for (src_padding, dst_padding) in [(0, 0), (0, 3), (2, 0), (3, 4), (1, 7)] {
                        check_converter(
                            convert,
                            depth,
                            sampling,
                            width,
                            height,
                            src_padding,
                            dst_padding,
                        );
                    }
                }
            }
        }
    }
}
