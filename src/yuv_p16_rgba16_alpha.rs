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
use crate::neon::neon_yuv_p16_to_rgba16_alpha_row;
use crate::numerics::{qrshr_n, to_ne};
use crate::yuv_support::{
    get_inverse_transform, get_yuv_range, YuvBytesPacking, YuvChromaSample, YuvEndianness,
    YuvRange, YuvSourceChannels, YuvStandardMatrix,
};
use crate::{YuvError, YuvPlanarImageWithAlpha};
#[cfg(feature = "rayon")]
use rayon::iter::{IndexedParallelIterator, ParallelIterator};
#[cfg(feature = "rayon")]
use rayon::prelude::{ParallelSlice, ParallelSliceMut};

pub(crate) fn yuv_p16_to_image_alpha_impl<
    const DESTINATION_CHANNELS: u8,
    const SAMPLING: u8,
    const ENDIANNESS: u8,
    const BYTES_POSITION: u8,
>(
    planar_image_with_alpha: &YuvPlanarImageWithAlpha<u16>,
    rgba16: &mut [u16],
    rgba_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    bit_depth: usize,
) -> Result<(), YuvError> {
    let dst_chans: YuvSourceChannels = DESTINATION_CHANNELS.into();
    let channels = dst_chans.get_channels_count();

    assert!(
        dst_chans != YuvSourceChannels::Rgb && dst_chans != YuvSourceChannels::Bgr,
        "Cannot call YUV p16 to Rgb8 with alpha without real alpha"
    );

    let chroma_subsampling: YuvChromaSample = SAMPLING.into();
    let range = get_yuv_range(bit_depth as u32, range);
    let kr_kb = matrix.get_kr_kb();

    planar_image_with_alpha.check_constraints(chroma_subsampling)?;

    let max_range_p16 = ((1u32 << bit_depth as u32) - 1) as i32;
    const PRECISION: i32 = 6;
    let transform = get_inverse_transform(
        max_range_p16 as u32,
        range.range_y,
        range.range_uv,
        kr_kb.kr,
        kr_kb.kb,
    );
    let i_transform = transform.to_integers(PRECISION as u32);
    let cr_coef = i_transform.cr_coef;
    let cb_coef = i_transform.cb_coef;
    let y_coef = i_transform.y_coef;
    let g_coef_1 = i_transform.g_coeff_1;
    let g_coef_2 = i_transform.g_coeff_2;

    let bias_y = range.bias_y as i32;
    let bias_uv = range.bias_uv as i32;

    let msb_shift = (16 - bit_depth) as i32;

    let iter;
    #[cfg(feature = "rayon")]
    {
        iter = rgba16
            .par_chunks_exact_mut(rgba_stride as usize)
            .zip(
                planar_image_with_alpha
                    .y_plane
                    .par_chunks_exact(planar_image_with_alpha.y_stride as usize),
            )
            .zip(
                planar_image_with_alpha
                    .a_plane
                    .par_chunks_exact(planar_image_with_alpha.a_stride as usize),
            );
    }
    #[cfg(not(feature = "rayon"))]
    {
        iter = rgba16
            .chunks_exact_mut(rgba_stride as usize)
            .zip(
                planar_image_with_alpha
                    .y_plane
                    .chunks_exact(planar_image_with_alpha.y_stride as usize),
            )
            .zip(
                planar_image_with_alpha
                    .a_plane
                    .chunks_exact(planar_image_with_alpha.a_stride as usize),
            );
    }

    let width = planar_image_with_alpha.width;

    let u_plane = planar_image_with_alpha.u_plane;
    let v_plane = planar_image_with_alpha.v_plane;

    iter.enumerate()
        .for_each(|(y, ((rgba16, y_plane), a_plane))| {
            let u_offset = if chroma_subsampling == YuvChromaSample::Yuv420 {
                (y >> 1) * (planar_image_with_alpha.u_stride as usize)
            } else {
                y * (planar_image_with_alpha.u_stride as usize)
            };
            let v_offset = if chroma_subsampling == YuvChromaSample::Yuv420 {
                (y >> 1) * (planar_image_with_alpha.v_stride as usize)
            } else {
                y * (planar_image_with_alpha.v_stride as usize)
            };

            let mut _cx = 0usize;

            let u_plane = match chroma_subsampling {
                YuvChromaSample::Yuv420 | YuvChromaSample::Yuv422 => {
                    let chroma_width = (width as usize + 1) / 2;
                    &u_plane[u_offset..(u_offset + chroma_width)]
                }
                YuvChromaSample::Yuv444 => &u_plane[u_offset..(u_offset + width as usize)],
            };
            let v_plane = match chroma_subsampling {
                YuvChromaSample::Yuv420 | YuvChromaSample::Yuv422 => {
                    let chroma_width = (width as usize + 1) / 2;
                    &v_plane[v_offset..(v_offset + chroma_width)]
                }
                YuvChromaSample::Yuv444 => &v_plane[v_offset..(v_offset + width as usize)],
            };

            #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
            {
                unsafe {
                    let offset = neon_yuv_p16_to_rgba16_alpha_row::<
                        DESTINATION_CHANNELS,
                        SAMPLING,
                        ENDIANNESS,
                        BYTES_POSITION,
                    >(
                        y_plane.as_ptr(),
                        u_plane.as_ptr(),
                        v_plane.as_ptr(),
                        a_plane.as_ptr(),
                        rgba16.as_mut_ptr(),
                        planar_image_with_alpha.width,
                        &range,
                        &i_transform,
                        0,
                        _cx,
                        bit_depth,
                    );
                    _cx = offset.cx;
                }
            }

            if chroma_subsampling == YuvChromaSample::Yuv444 && _cx < width as usize {
                for ((((rgba, &y_src), &u_src), &v_src), &a_src) in rgba16
                    .chunks_exact_mut(channels)
                    .zip(y_plane.iter())
                    .zip(u_plane.iter())
                    .zip(v_plane.iter())
                    .zip(a_plane.iter())
                    .skip(_cx)
                {
                    let y_value = (to_ne::<ENDIANNESS, BYTES_POSITION>(y_src, msb_shift) as i32
                        - bias_y)
                        * y_coef;
                    let cb_value =
                        to_ne::<ENDIANNESS, BYTES_POSITION>(u_src, msb_shift) as i32 - bias_uv;
                    let cr_value =
                        to_ne::<ENDIANNESS, BYTES_POSITION>(v_src, msb_shift) as i32 - bias_uv;

                    let r = qrshr_n::<PRECISION>(y_value + cr_coef * cr_value, max_range_p16);
                    let b = qrshr_n::<PRECISION>(y_value + cb_coef * cb_value, max_range_p16);
                    let g = qrshr_n::<PRECISION>(
                        y_value - g_coef_1 * cr_value - g_coef_2 * cb_value,
                        max_range_p16,
                    );

                    rgba[dst_chans.get_r_channel_offset()] = r as u16;
                    rgba[dst_chans.get_g_channel_offset()] = g as u16;
                    rgba[dst_chans.get_b_channel_offset()] = b as u16;
                    rgba[dst_chans.get_a_channel_offset()] = a_src;
                }
            } else if _cx < width as usize {
                for ((((rgba, y_src), &u_src), &v_src), a_src) in rgba16
                    .chunks_exact_mut(channels * 2)
                    .zip(y_plane.chunks_exact(2))
                    .zip(u_plane.iter())
                    .zip(v_plane.iter())
                    .zip(a_plane.chunks_exact(2))
                    .skip(_cx / 2)
                {
                    let y0_value =
                        (to_ne::<ENDIANNESS, BYTES_POSITION>(y_src[0], msb_shift) as i32 - bias_y)
                            * y_coef;
                    let cb_value =
                        to_ne::<ENDIANNESS, BYTES_POSITION>(u_src, msb_shift) as i32 - bias_uv;
                    let cr_value =
                        to_ne::<ENDIANNESS, BYTES_POSITION>(v_src, msb_shift) as i32 - bias_uv;

                    let r0 = qrshr_n::<PRECISION>(y0_value + cr_coef * cr_value, max_range_p16);
                    let b0 = qrshr_n::<PRECISION>(y0_value + cb_coef * cb_value, max_range_p16);
                    let g0 = qrshr_n::<PRECISION>(
                        y0_value - g_coef_1 * cr_value - g_coef_2 * cb_value,
                        max_range_p16,
                    );

                    rgba[dst_chans.get_r_channel_offset()] = r0 as u16;
                    rgba[dst_chans.get_g_channel_offset()] = g0 as u16;
                    rgba[dst_chans.get_b_channel_offset()] = b0 as u16;
                    rgba[dst_chans.get_a_channel_offset()] = a_src[0];

                    let y1_value =
                        (to_ne::<ENDIANNESS, BYTES_POSITION>(y_src[1], msb_shift) as i32 - bias_y)
                            * y_coef;

                    let r1 = qrshr_n::<PRECISION>(y1_value + cr_coef * cr_value, max_range_p16);
                    let b1 = qrshr_n::<PRECISION>(y1_value + cb_coef * cb_value, max_range_p16);
                    let g1 = qrshr_n::<PRECISION>(
                        y1_value - g_coef_1 * cr_value - g_coef_2 * cb_value,
                        max_range_p16,
                    );

                    rgba[dst_chans.get_r_channel_offset() + channels] = r1 as u16;
                    rgba[dst_chans.get_g_channel_offset() + channels] = g1 as u16;
                    rgba[dst_chans.get_b_channel_offset() + channels] = b1 as u16;
                    if dst_chans.has_alpha() {
                        rgba[dst_chans.get_a_channel_offset() + channels] = a_src[1];
                    }
                }

                if width & 1 != 0 {
                    let y0_value =
                        (to_ne::<ENDIANNESS, BYTES_POSITION>(*y_plane.last().unwrap(), msb_shift)
                            as i32
                            - bias_y)
                            * y_coef;
                    let cb_value =
                        to_ne::<ENDIANNESS, BYTES_POSITION>(*u_plane.last().unwrap(), msb_shift)
                            as i32
                            - bias_uv;
                    let cr_value =
                        to_ne::<ENDIANNESS, BYTES_POSITION>(*v_plane.last().unwrap(), msb_shift)
                            as i32
                            - bias_uv;
                    let last_a = a_plane.last().unwrap();
                    let rgba = rgba16.chunks_exact_mut(channels * 2).into_remainder();

                    let r0 = qrshr_n::<PRECISION>(y0_value + cr_coef * cr_value, max_range_p16);
                    let b0 = qrshr_n::<PRECISION>(y0_value + cb_coef * cb_value, max_range_p16);
                    let g0 = qrshr_n::<PRECISION>(
                        y0_value - g_coef_1 * cr_value - g_coef_2 * cb_value,
                        max_range_p16,
                    );

                    rgba[dst_chans.get_r_channel_offset()] = r0 as u16;
                    rgba[dst_chans.get_g_channel_offset()] = g0 as u16;
                    rgba[dst_chans.get_b_channel_offset()] = b0 as u16;
                    if dst_chans.has_alpha() {
                        rgba[dst_chans.get_a_channel_offset()] = *last_a;
                    }
                }
            }
        });

    Ok(())
}

/// Convert YUV 420 planar format with 8+ bit pixel format to BGRA 8+ bit-depth format with interleaving alpha.
///
/// This function takes YUV 420 planar data with 8+ bit precision.
/// and converts it to BGRA format with 8-bit precision per channel
///
/// # Arguments
///
/// * `planar_image_with_alpha` -  Source planar image with alpha.
/// * `bgra` - A mutable slice to store the converted BGRA data.
/// * `bgra_stride` - The stride (elements per row) for BGRA data.
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
pub fn yuv420_p16_with_alpha_to_bgra16(
    planar_image_with_alpha: &YuvPlanarImageWithAlpha<u16>,
    bgra: &mut [u16],
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
                    { YuvChromaSample::Yuv420 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_p16_to_image_alpha_impl::<
                    { YuvSourceChannels::Bgra as u8 },
                    { YuvChromaSample::Yuv420 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
        YuvEndianness::LittleEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_p16_to_image_alpha_impl::<
                    { YuvSourceChannels::Bgra as u8 },
                    { YuvChromaSample::Yuv420 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_p16_to_image_alpha_impl::<
                    { YuvSourceChannels::Bgra as u8 },
                    { YuvChromaSample::Yuv420 as u8 },
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

/// Convert YUV 422 format with 8+ bit pixel format to BGRA 8+ bit-depth format with interleaving alpha.
///
/// This function takes YUV 422 data with 8+ bit precision.
/// and converts it to BGRA format with 8-bit precision per channel.
///
/// # Arguments
///
/// * `planar_image_with_alpha` -  Source planar image with alpha.
/// * `bgra` - A mutable slice to store the converted BGRA data.
/// * `bgra_stride` - The stride (elements per row) for BGRA data.
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
pub fn yuv422_p16_with_alpha_to_bgra16(
    planar_image_with_alpha: &YuvPlanarImageWithAlpha<u16>,
    bgra: &mut [u16],
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
                    { YuvChromaSample::Yuv422 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_p16_to_image_alpha_impl::<
                    { YuvSourceChannels::Bgra as u8 },
                    { YuvChromaSample::Yuv422 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
        YuvEndianness::LittleEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_p16_to_image_alpha_impl::<
                    { YuvSourceChannels::Bgra as u8 },
                    { YuvChromaSample::Yuv422 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_p16_to_image_alpha_impl::<
                    { YuvSourceChannels::Bgra as u8 },
                    { YuvChromaSample::Yuv422 as u8 },
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

/// Convert YUV 420 planar format with 8+ bit pixel format to RGBA 8+ bit-depth format with interleaving alpha.
///
/// This function takes YUV 420 planar data with 8+ bit precision.
/// and converts it to RGBA format with 8-bit precision per channel
///
/// # Arguments
///
/// * `planar_image_with_alpha` -  Source planar image with alpha.
/// * `rgba` - A mutable slice to store the converted RGBA data.
/// * `rgba_stride` - The stride (elements per row) for RGBA data.
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
pub fn yuv420_p16_with_alpha_to_rgba16(
    planar_image_with_alpha: &YuvPlanarImageWithAlpha<u16>,
    rgba: &mut [u16],
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
                    { YuvChromaSample::Yuv420 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_p16_to_image_alpha_impl::<
                    { YuvSourceChannels::Rgba as u8 },
                    { YuvChromaSample::Yuv420 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
        YuvEndianness::LittleEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_p16_to_image_alpha_impl::<
                    { YuvSourceChannels::Rgba as u8 },
                    { YuvChromaSample::Yuv420 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_p16_to_image_alpha_impl::<
                    { YuvSourceChannels::Rgba as u8 },
                    { YuvChromaSample::Yuv420 as u8 },
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

/// Convert YUV 422 format with 8+ bit pixel format to RGBA 8+ bit-depth format with interleaving alpha.
///
/// This function takes YUV 422 data with 8+ bit precision stored.
/// and converts it to RGBA format with 8-bit precision per channel.
///
/// # Arguments
///
/// * `planar_image_with_alpha` -  Source planar image with alpha.
/// * `rgba` - A mutable slice to store the converted RGBA data.
/// * `rgba_stride` - The stride (elements per row) for RGBA data.
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
pub fn yuv422_p16_with_alpha_to_rgba16(
    planar_image_with_alpha: &YuvPlanarImageWithAlpha<u16>,
    rgba: &mut [u16],
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
                    { YuvChromaSample::Yuv422 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_p16_to_image_alpha_impl::<
                    { YuvSourceChannels::Rgba as u8 },
                    { YuvChromaSample::Yuv422 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
        YuvEndianness::LittleEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_p16_to_image_alpha_impl::<
                    { YuvSourceChannels::Rgba as u8 },
                    { YuvChromaSample::Yuv422 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_p16_to_image_alpha_impl::<
                    { YuvSourceChannels::Rgba as u8 },
                    { YuvChromaSample::Yuv422 as u8 },
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

/// Convert YUV 444 planar format with 8+ bit pixel format to RGBA 8+ bit-depth format with interleaving alpha.
///
/// This function takes YUV 444 planar data with 8+ bit precision.
/// and converts it to RGBA format with 8-bit precision per channel
///
/// # Arguments
///
/// * `planar_image_with_alpha` -  Source planar image with alpha.
/// * `rgba` - A mutable slice to store the converted RGBA data.
/// * `rgba_stride` - The stride (elements per row) for RGBA data.
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
pub fn yuv444_p16_with_alpha_to_rgba16(
    planar_image_with_alpha: &YuvPlanarImageWithAlpha<u16>,
    rgba: &mut [u16],
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
                    { YuvChromaSample::Yuv444 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_p16_to_image_alpha_impl::<
                    { YuvSourceChannels::Rgba as u8 },
                    { YuvChromaSample::Yuv444 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
        YuvEndianness::LittleEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_p16_to_image_alpha_impl::<
                    { YuvSourceChannels::Rgba as u8 },
                    { YuvChromaSample::Yuv444 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_p16_to_image_alpha_impl::<
                    { YuvSourceChannels::Rgba as u8 },
                    { YuvChromaSample::Yuv444 as u8 },
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

/// Convert YUV 444 planar format with 8+ bit pixel format to BGRA 8+ bit-depth format with interleaving alpha.
///
/// This function takes YUV 444 planar data with 8+ bit precision.
/// and converts it to BGRA format with 8-bit precision per channel
///
/// # Arguments
///
/// * `planar_image_with_alpha` -  Source planar image with alpha.
/// * `bgra` - A mutable slice to store the converted BGRA data.
/// * `bgra_stride` - The stride (elements per row) for BGRA data.
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
pub fn yuv444_p16_with_alpha_to_bgra16(
    planar_image_with_alpha: &YuvPlanarImageWithAlpha<u16>,
    bgra: &mut [u16],
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
                    { YuvChromaSample::Yuv444 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_p16_to_image_alpha_impl::<
                    { YuvSourceChannels::Bgra as u8 },
                    { YuvChromaSample::Yuv444 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
        YuvEndianness::LittleEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_p16_to_image_alpha_impl::<
                    { YuvSourceChannels::Bgra as u8 },
                    { YuvChromaSample::Yuv444 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_p16_to_image_alpha_impl::<
                    { YuvSourceChannels::Bgra as u8 },
                    { YuvChromaSample::Yuv444 as u8 },
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
