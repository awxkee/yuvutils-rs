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
use crate::built_coefficients::get_built_inverse_transform;
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use crate::neon::neon_yuv_p16_to_rgba_alpha_row;
use crate::numerics::to_ne;
use crate::yuv_error::check_rgba_destination;
use crate::yuv_support::{
    get_inverse_transform, get_yuv_range, YuvBytesPacking, YuvChromaSubsampling, YuvEndianness,
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

            let rgba0 = &mut rgba[0..channels];

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
            let rgba0 = &mut rgba[0..channels];

            let r0 = qrshr::<BIT_DEPTH>(y_value0 + cr_coef * cr_value);
            let b0 = qrshr::<BIT_DEPTH>(y_value0 + cb_coef * cb_value);
            let g0 = qrshr::<BIT_DEPTH>(y_value0 - g_coef_1 * cr_value - g_coef_2 * cb_value);
            rgba0[dst_chans.get_r_channel_offset()] = r0 as u8;
            rgba0[dst_chans.get_g_channel_offset()] = g0 as u8;
            rgba0[dst_chans.get_b_channel_offset()] = b0 as u8;
            rgba0[dst_chans.get_a_channel_offset()] = (a_value >> store_shift).min(255) as u8;
        }
    };

    if chroma_subsampling == YuvChromaSubsampling::Yuv444 {
        let iter;
        #[cfg(feature = "rayon")]
        {
            iter = rgba
                .par_chunks_exact_mut(rgba_stride as usize)
                .zip(image.y_plane.par_chunks_exact(image.y_stride as usize))
                .zip(image.a_plane.par_chunks_exact(image.a_stride as usize))
                .zip(image.u_plane.par_chunks_exact(image.u_stride as usize))
                .zip(image.v_plane.par_chunks_exact(image.v_stride as usize));
        }
        #[cfg(not(feature = "rayon"))]
        {
            iter = rgba
                .chunks_exact_mut(rgba_stride as usize)
                .zip(image.y_plane.chunks_exact(image.y_stride as usize))
                .zip(image.a_plane.chunks_exact(image.a_stride as usize))
                .zip(image.u_plane.chunks_exact(image.u_stride as usize))
                .zip(image.v_plane.chunks_exact(image.v_stride as usize));
        }
        iter.for_each(|((((rgba, y_plane), a_plane), u_plane), v_plane)| {
            let y_plane = &y_plane[0..image.width as usize];
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
                .par_chunks_exact_mut(rgba_stride as usize)
                .zip(image.y_plane.par_chunks_exact(image.y_stride as usize))
                .zip(image.a_plane.par_chunks_exact(image.a_stride as usize))
                .zip(image.u_plane.par_chunks_exact(image.u_stride as usize))
                .zip(image.v_plane.par_chunks_exact(image.v_stride as usize));
        }
        #[cfg(not(feature = "rayon"))]
        {
            iter = rgba
                .chunks_exact_mut(rgba_stride as usize)
                .zip(image.y_plane.chunks_exact(image.y_stride as usize))
                .zip(image.a_plane.chunks_exact(image.a_stride as usize))
                .zip(image.u_plane.chunks_exact(image.u_stride as usize))
                .zip(image.v_plane.chunks_exact(image.v_stride as usize));
        }
        iter.for_each(|((((rgba, y_plane), a_plane), u_plane), v_plane)| {
            process_halved_chroma_row(y_plane, u_plane, v_plane, a_plane, rgba);
        });
    } else if chroma_subsampling == YuvChromaSubsampling::Yuv420 {
        let iter;
        #[cfg(feature = "rayon")]
        {
            iter = rgba
                .par_chunks_exact_mut(rgba_stride as usize * 2)
                .zip(image.y_plane.par_chunks_exact(image.y_stride as usize * 2))
                .zip(image.a_plane.par_chunks_exact(image.a_stride as usize * 2))
                .zip(image.u_plane.par_chunks_exact(image.u_stride as usize))
                .zip(image.v_plane.par_chunks_exact(image.v_stride as usize));
        }
        #[cfg(not(feature = "rayon"))]
        {
            iter = rgba
                .chunks_exact_mut(rgba_stride as usize * 2)
                .zip(image.y_plane.chunks_exact(image.y_stride as usize * 2))
                .zip(image.a_plane.chunks_exact(image.a_stride as usize * 2))
                .zip(image.u_plane.chunks_exact(image.u_stride as usize))
                .zip(image.v_plane.chunks_exact(image.v_stride as usize));
        }
        iter.for_each(|((((rgba, y_plane), a_plane), u_plane), v_plane)| {
            for ((rgba, y_plane), a_plane) in rgba
                .chunks_exact_mut(rgba_stride as usize)
                .zip(y_plane.chunks_exact(image.y_stride as usize))
                .zip(a_plane.chunks_exact(image.a_stride as usize))
            {
                process_halved_chroma_row(
                    &y_plane[0..image.width as usize],
                    &u_plane[0..(image.width as usize).div_ceil(2)],
                    &v_plane[0..(image.width as usize).div_ceil(2)],
                    &a_plane[0..image.width as usize],
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
            let a_plane = image
                .a_plane
                .chunks_exact(image.a_stride as usize)
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
                &a_plane[0..image.width as usize],
                &mut rgba[0..image.width as usize * channels],
            );
        }
    } else {
        unreachable!();
    }

    Ok(())
}

pub(crate) fn yuv_p16_to_image_alpha_impl<
    const DESTINATION_CHANNELS: u8,
    const SAMPLING: u8,
    const ENDIANNESS: u8,
    const BYTES_POSITION: u8,
>(
    image: &YuvPlanarImageWithAlpha<u16>,
    rgba: &mut [u8],
    rgba_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    bit_depth: usize,
) -> Result<(), YuvError> {
    if bit_depth == 10 {
        yuv_p16_to_image_alpha_ant::<DESTINATION_CHANNELS, SAMPLING, ENDIANNESS, BYTES_POSITION, 10>(
            image,
            rgba,
            rgba_stride,
            range,
            matrix,
        )
    } else if bit_depth == 12 {
        yuv_p16_to_image_alpha_ant::<DESTINATION_CHANNELS, SAMPLING, ENDIANNESS, BYTES_POSITION, 12>(
            image,
            rgba,
            rgba_stride,
            range,
            matrix,
        )
    } else {
        unimplemented!("YUV16 -> RGB8 implemented only for 10 and 12 bit")
    }
}

/// Convert YUV 420 planar format with 8+ bit pixel format to BGRA format with interleaving alpha.
///
/// This function takes YUV 420 planar data with 8+ bit precision.
/// and converts it to BGRA format with 8-bit precision per channel
///
/// # Arguments
///
/// * `planar_image_with_alpha` - Source planar image with alpha.
/// * `bgra` - A mutable slice to store the converted BGRA data.
/// * `bgra_stride` - The stride (components per row) for BGRA data.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
/// * `endianness` - The endianness of stored bytes
/// * `bytes_packing` - position of significant bytes ( most significant or least significant ) if it in most significant it should be stated as per Apple *kCVPixelFormatType_422YpCbCr10BiPlanarFullRange/kCVPixelFormatType_422YpCbCr10BiPlanarVideoRange*
/// * `bit_depth` - Bit depth of source YUV planes
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGRA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv420_p16_with_alpha_to_bgra(
    planar_image_with_alpha: &YuvPlanarImageWithAlpha<u16>,
    bgra: &mut [u8],
    bgra_stride: u32,
    bit_depth: usize,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    endianness: YuvEndianness,
    bytes_packing: YuvBytesPacking,
) -> Result<(), YuvError> {
    let dispatcher = match endianness {
        YuvEndianness::BigEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_p16_to_image_alpha_impl::<
                    { YuvSourceChannels::Bgra as u8 },
                    { YuvChromaSubsampling::Yuv420 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_p16_to_image_alpha_impl::<
                    { YuvSourceChannels::Bgra as u8 },
                    { YuvChromaSubsampling::Yuv420 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
        YuvEndianness::LittleEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_p16_to_image_alpha_impl::<
                    { YuvSourceChannels::Bgra as u8 },
                    { YuvChromaSubsampling::Yuv420 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_p16_to_image_alpha_impl::<
                    { YuvSourceChannels::Bgra as u8 },
                    { YuvChromaSubsampling::Yuv420 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
    };
    dispatcher(
        planar_image_with_alpha,
        bgra,
        bgra_stride,
        range,
        matrix,
        bit_depth,
    )
}

/// Convert YUV 422 format with 8+ bit pixel format to BGRA format with interleaving alpha.
///
/// This function takes YUV 422 data with 8+ bit precision.
/// and converts it to BGRA format with 8-bit precision per channel.
///
/// # Arguments
///
/// * `planar_image_with_alpha` -  Source planar image with alpha.
/// * `bgra` - A mutable slice to store the converted BGRA data.
/// * `bgra_stride` - The stride (components per row) for BGRA data.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
/// * `endianness` - The endianness of stored bytes
/// * `bytes_packing` - position of significant bytes ( most significant or least significant ) if it in most significant it should be stated as per Apple *kCVPixelFormatType_422YpCbCr10BiPlanarFullRange/kCVPixelFormatType_422YpCbCr10BiPlanarVideoRange*
/// * `bit_depth` - Bit depth of source YUV planes
///
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGRA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv422_p16_with_alpha_to_bgra(
    planar_image_with_alpha: &YuvPlanarImageWithAlpha<u16>,
    bgra: &mut [u8],
    bgra_stride: u32,
    bit_depth: usize,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    endianness: YuvEndianness,
    bytes_packing: YuvBytesPacking,
) -> Result<(), YuvError> {
    let dispatcher = match endianness {
        YuvEndianness::BigEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_p16_to_image_alpha_impl::<
                    { YuvSourceChannels::Bgra as u8 },
                    { YuvChromaSubsampling::Yuv422 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_p16_to_image_alpha_impl::<
                    { YuvSourceChannels::Bgra as u8 },
                    { YuvChromaSubsampling::Yuv422 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
        YuvEndianness::LittleEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_p16_to_image_alpha_impl::<
                    { YuvSourceChannels::Bgra as u8 },
                    { YuvChromaSubsampling::Yuv422 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_p16_to_image_alpha_impl::<
                    { YuvSourceChannels::Bgra as u8 },
                    { YuvChromaSubsampling::Yuv422 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
    };
    dispatcher(
        planar_image_with_alpha,
        bgra,
        bgra_stride,
        range,
        matrix,
        bit_depth,
    )
}

/// Convert YUV 420 planar format with 8+ bit pixel format to RGBA format with interleaving alpha.
///
/// This function takes YUV 420 planar data with 8+ bit precision.
/// and converts it to RGBA format with 8-bit precision per channel
///
/// # Arguments
///
/// * `planar_image_with_alpha` -  Source planar image with alpha.
/// * `rgba` - A mutable slice to store the converted RGBA data.
/// * `rgba_stride` - The stride (components per row) for RGBA data.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
/// * `endianness` - The endianness of stored bytes
/// * `bytes_packing` - position of significant bytes ( most significant or least significant ) if it in most significant it should be stated as per Apple *kCVPixelFormatType_422YpCbCr10BiPlanarFullRange/kCVPixelFormatType_422YpCbCr10BiPlanarVideoRange*
/// * `bit_depth` - Bit depth of source YUV planes
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input RGBA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv420_p16_with_alpha_to_rgba(
    planar_image_with_alpha: &YuvPlanarImageWithAlpha<u16>,
    rgba: &mut [u8],
    rgba_stride: u32,
    bit_depth: usize,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    endianness: YuvEndianness,
    bytes_packing: YuvBytesPacking,
) -> Result<(), YuvError> {
    let dispatcher = match endianness {
        YuvEndianness::BigEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_p16_to_image_alpha_impl::<
                    { YuvSourceChannels::Rgba as u8 },
                    { YuvChromaSubsampling::Yuv420 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_p16_to_image_alpha_impl::<
                    { YuvSourceChannels::Rgba as u8 },
                    { YuvChromaSubsampling::Yuv420 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
        YuvEndianness::LittleEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_p16_to_image_alpha_impl::<
                    { YuvSourceChannels::Rgba as u8 },
                    { YuvChromaSubsampling::Yuv420 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_p16_to_image_alpha_impl::<
                    { YuvSourceChannels::Rgba as u8 },
                    { YuvChromaSubsampling::Yuv420 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
    };
    dispatcher(
        planar_image_with_alpha,
        rgba,
        rgba_stride,
        range,
        matrix,
        bit_depth,
    )
}

/// Convert YUV 422 format with 8+ bit pixel format to RGBA format with interleaving alpha.
///
/// This function takes YUV 422 data with 8+ bit precision stored.
/// and converts it to RGBA format with 8-bit precision per channel.
///
/// # Arguments
///
/// * `planar_image_with_alpha` -  Source planar image with alpha.
/// * `rgba_data` - A mutable slice to store the converted RGBA data.
/// * `rgba_stride` - The stride (components per row) for RGBA data.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
/// * `endianness` - The endianness of stored bytes
/// * `bytes_packing` - position of significant bytes ( most significant or least significant ) if it in most significant it should be stated as per Apple *kCVPixelFormatType_422YpCbCr10BiPlanarFullRange/kCVPixelFormatType_422YpCbCr10BiPlanarVideoRange*
/// * `bit_depth` - Bit depth of source YUV planes
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input RGBA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv422_p16_with_alpha_to_rgba(
    planar_image_with_alpha: &YuvPlanarImageWithAlpha<u16>,
    rgba: &mut [u8],
    rgba_stride: u32,
    bit_depth: usize,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    endianness: YuvEndianness,
    bytes_packing: YuvBytesPacking,
) -> Result<(), YuvError> {
    let dispatcher = match endianness {
        YuvEndianness::BigEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_p16_to_image_alpha_impl::<
                    { YuvSourceChannels::Rgba as u8 },
                    { YuvChromaSubsampling::Yuv422 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_p16_to_image_alpha_impl::<
                    { YuvSourceChannels::Rgba as u8 },
                    { YuvChromaSubsampling::Yuv422 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
        YuvEndianness::LittleEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_p16_to_image_alpha_impl::<
                    { YuvSourceChannels::Rgba as u8 },
                    { YuvChromaSubsampling::Yuv422 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_p16_to_image_alpha_impl::<
                    { YuvSourceChannels::Rgba as u8 },
                    { YuvChromaSubsampling::Yuv422 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
    };
    dispatcher(
        planar_image_with_alpha,
        rgba,
        rgba_stride,
        range,
        matrix,
        bit_depth,
    )
}

/// Convert YUV 444 planar format with 8+ bit pixel format to RGBA format with interleaving alpha.
///
/// This function takes YUV 444 planar data with 8+ bit precision.
/// and converts it to RGBA format with 8-bit precision per channel
///
/// # Arguments
///
/// * `planar_image_with_alpha` -  Source planar image with alpha.
/// * `rgba_data` - A mutable slice to store the converted RGBA data.
/// * `rgba_stride` - The stride (components per row) for RGBA data.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
/// * `endianness` - The endianness of stored bytes
/// * `bytes_packing` - position of significant bytes ( most significant or least significant ) if it in most significant it should be stated as per Apple *kCVPixelFormatType_422YpCbCr10BiPlanarFullRange/kCVPixelFormatType_422YpCbCr10BiPlanarVideoRange*
/// * `bit_depth` - Bit depth of source YUV planes
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input RGBA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv444_p16_with_alpha_to_rgba(
    planar_image_with_alpha: &YuvPlanarImageWithAlpha<u16>,
    rgba: &mut [u8],
    rgba_stride: u32,
    bit_depth: usize,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    endianness: YuvEndianness,
    bytes_packing: YuvBytesPacking,
) -> Result<(), YuvError> {
    let dispatcher = match endianness {
        YuvEndianness::BigEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_p16_to_image_alpha_impl::<
                    { YuvSourceChannels::Rgba as u8 },
                    { YuvChromaSubsampling::Yuv444 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_p16_to_image_alpha_impl::<
                    { YuvSourceChannels::Rgba as u8 },
                    { YuvChromaSubsampling::Yuv444 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
        YuvEndianness::LittleEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_p16_to_image_alpha_impl::<
                    { YuvSourceChannels::Rgba as u8 },
                    { YuvChromaSubsampling::Yuv444 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_p16_to_image_alpha_impl::<
                    { YuvSourceChannels::Rgba as u8 },
                    { YuvChromaSubsampling::Yuv444 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
    };
    dispatcher(
        planar_image_with_alpha,
        rgba,
        rgba_stride,
        range,
        matrix,
        bit_depth,
    )
}

/// Convert YUV 444 planar format with 8+ bit pixel format to BGRA format with interleaving alpha.
///
/// This function takes YUV 444 planar data with 8+ bit precision.
/// and converts it to BGRA format with 8-bit precision per channel
///
/// # Arguments
///
/// * `planar_image_with_alpha` -  Source planar image with alpha.
/// * `bgra` - A mutable slice to store the converted BGRA data.
/// * `bgra_stride` - The stride (components per row) for BGRA data.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
/// * `endianness` - The endianness of stored bytes
/// * `bytes_packing` - position of significant bytes ( most significant or least significant ) if it in most significant it should be stated as per Apple *kCVPixelFormatType_422YpCbCr10BiPlanarFullRange/kCVPixelFormatType_422YpCbCr10BiPlanarVideoRange*
/// * `bit_depth` - Bit depth of source YUV planes
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGRA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv444_p16_with_alpha_to_bgra(
    planar_image_with_alpha: &YuvPlanarImageWithAlpha<u16>,
    bgra: &mut [u8],
    bgra_stride: u32,
    bit_depth: usize,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    endianness: YuvEndianness,
    bytes_packing: YuvBytesPacking,
) -> Result<(), YuvError> {
    let dispatcher = match endianness {
        YuvEndianness::BigEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_p16_to_image_alpha_impl::<
                    { YuvSourceChannels::Bgra as u8 },
                    { YuvChromaSubsampling::Yuv444 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_p16_to_image_alpha_impl::<
                    { YuvSourceChannels::Bgra as u8 },
                    { YuvChromaSubsampling::Yuv444 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
        YuvEndianness::LittleEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_p16_to_image_alpha_impl::<
                    { YuvSourceChannels::Bgra as u8 },
                    { YuvChromaSubsampling::Yuv444 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_p16_to_image_alpha_impl::<
                    { YuvSourceChannels::Bgra as u8 },
                    { YuvChromaSubsampling::Yuv444 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
    };
    dispatcher(
        planar_image_with_alpha,
        bgra,
        bgra_stride,
        range,
        matrix,
        bit_depth,
    )
}
