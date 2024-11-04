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
use crate::neon::neon_yuv_nv12_p10_to_rgba_row;
use crate::yuv_support::*;
use crate::{YuvBiPlanarImage, YuvError};
#[cfg(feature = "rayon")]
use rayon::iter::{IndexedParallelIterator, ParallelIterator};
#[cfg(feature = "rayon")]
use rayon::prelude::ParallelSliceMut;

fn yuv_nv_p10_to_image_impl<
    const DESTINATION_CHANNELS: u8,
    const NV_ORDER: u8,
    const SAMPLING: u8,
    const ENDIANNESS: u8,
    const BYTES_POSITION: u8,
>(
    bi_planar_image: &YuvBiPlanarImage<u16>,
    bgra: &mut [u8],
    bgra_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) -> Result<(), YuvError> {
    let dst_chans: YuvSourceChannels = DESTINATION_CHANNELS.into();
    let channels = dst_chans.get_channels_count();
    let uv_order: YuvNVOrder = NV_ORDER.into();
    let chroma_subsampling: YuvChromaSample = SAMPLING.into();
    let endianness: YuvEndianness = ENDIANNESS.into();
    let bytes_position: YuvBytesPacking = BYTES_POSITION.into();

    bi_planar_image.check_constraints(chroma_subsampling)?;

    let range = get_yuv_range(10, range);
    let kr_kb = matrix.get_kr_kb();
    let max_range_p10 = (1u32 << 10u32) - 1u32;
    let transform = get_inverse_transform(
        max_range_p10,
        range.range_y,
        range.range_uv,
        kr_kb.kr,
        kr_kb.kb,
    );
    const ROUNDING_CONST: i32 = 1 << 5;
    let i_transform = transform.to_integers(6u32);
    let cr_coef = i_transform.cr_coef;
    let cb_coef = i_transform.cb_coef;
    let y_coef = i_transform.y_coef;
    let g_coef_1 = i_transform.g_coeff_1;
    let g_coef_2 = i_transform.g_coeff_2;

    let bias_y = range.bias_y as i32;
    let bias_uv = range.bias_uv as i32;

    let iterator_step = match chroma_subsampling {
        YuvChromaSample::Yuv420 => 2usize,
        YuvChromaSample::Yuv422 => 2usize,
        YuvChromaSample::Yuv444 => 1usize,
    };

    let iter;
    #[cfg(feature = "rayon")]
    {
        iter = bgra.par_chunks_exact_mut(bgra_stride as usize);
    }
    #[cfg(not(feature = "rayon"))]
    {
        iter = bgra.chunks_exact_mut(bgra_stride as usize);
    }

    let y_plane = bi_planar_image.y_plane;
    let uv_plane = bi_planar_image.uv_plane;
    let y_stride = bi_planar_image.y_stride * 2;
    let uv_stride = bi_planar_image.uv_stride * 2;
    let width = bi_planar_image.width;

    iter.enumerate().for_each(|(y, bgra)| unsafe {
        let y_offset = y * (y_stride as usize);
        let uv_offset = if chroma_subsampling == YuvChromaSample::Yuv420 {
            (y >> 1) * (uv_stride as usize)
        } else {
            y * (uv_stride as usize)
        };
        let dst_offset = 0usize;

        let y_src_ptr = y_plane.as_ptr() as *const u8;
        let uv_src_ptr = uv_plane.as_ptr() as *const u8;

        let mut _cx = 0usize;

        let mut _ux = 0usize;

        let y_ld_ptr = y_src_ptr.add(y_offset) as *const u16;
        let uv_ld_ptr = uv_src_ptr.add(uv_offset) as *const u16;

        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        {
            let offset = neon_yuv_nv12_p10_to_rgba_row::<
                DESTINATION_CHANNELS,
                NV_ORDER,
                SAMPLING,
                ENDIANNESS,
                BYTES_POSITION,
            >(
                y_ld_ptr,
                uv_ld_ptr,
                bgra,
                dst_offset,
                width,
                &range,
                &i_transform,
                _cx,
                _ux,
            );
            _cx = offset.cx;
            _ux = offset.ux;
        }

        for x in (_cx..width as usize).step_by(iterator_step) {
            let y_value: i32;
            let mut cb_value: i32;
            let mut cr_value: i32;
            match endianness {
                YuvEndianness::BigEndian => {
                    let mut y_vl = u16::from_be(y_ld_ptr.add(x).read_unaligned()) as i32;
                    let mut cb_vl = u16::from_be(
                        uv_ld_ptr
                            .add(_ux + uv_order.get_u_position())
                            .read_unaligned(),
                    ) as i32;
                    let mut cr_vl = u16::from_be(
                        uv_ld_ptr
                            .add(_ux + uv_order.get_v_position())
                            .read_unaligned(),
                    ) as i32;
                    if bytes_position == YuvBytesPacking::MostSignificantBytes {
                        y_vl >>= 6;
                        cb_vl >>= 6;
                        cr_vl >>= 6;
                    }
                    y_value = (y_vl - bias_y) * y_coef;

                    cb_value = cb_vl;
                    cr_value = cr_vl;
                }
                YuvEndianness::LittleEndian => {
                    let mut y_vl = u16::from_le(y_ld_ptr.add(x).read_unaligned()) as i32;
                    let mut cb_vl = u16::from_le(
                        uv_ld_ptr
                            .add(_ux + uv_order.get_u_position())
                            .read_unaligned(),
                    ) as i32;
                    let mut cr_vl = u16::from_le(
                        uv_ld_ptr
                            .add(_ux + uv_order.get_v_position())
                            .read_unaligned(),
                    ) as i32;
                    if bytes_position == YuvBytesPacking::MostSignificantBytes {
                        y_vl >>= 6;
                        cb_vl >>= 6;
                        cr_vl >>= 6;
                    }
                    y_value = (y_vl - bias_y) * y_coef;

                    cb_value = cb_vl;
                    cr_value = cr_vl;
                }
            }

            match uv_order {
                YuvNVOrder::UV => {
                    cb_value -= bias_uv;
                    cr_value -= bias_uv;
                }
                YuvNVOrder::VU => {
                    cr_value = cb_value - bias_uv;
                    cb_value = cr_value - bias_uv;
                }
            }

            // shift right 8 due to we want to make it 8 bit instead of 10

            let r_u16 = (y_value + cr_coef * cr_value + ROUNDING_CONST) >> 8;
            let b_u16 = (y_value + cb_coef * cb_value + ROUNDING_CONST) >> 8;
            let g_u16 = (y_value - g_coef_1 * cr_value - g_coef_2 * cb_value + ROUNDING_CONST) >> 8;

            let r = r_u16.min(255).max(0);
            let b = b_u16.min(255).max(0);
            let g = g_u16.min(255).max(0);

            let px = x * channels;

            let rgb_offset = dst_offset + px;

            let dst_slice = bgra.get_unchecked_mut(rgb_offset..);
            *dst_slice.get_unchecked_mut(dst_chans.get_b_channel_offset()) = b as u8;
            *dst_slice.get_unchecked_mut(dst_chans.get_g_channel_offset()) = g as u8;
            *dst_slice.get_unchecked_mut(dst_chans.get_r_channel_offset()) = r as u8;
            if dst_chans.has_alpha() {
                *bgra.get_unchecked_mut(dst_chans.get_a_channel_offset()) = 255;
            }

            if chroma_subsampling == YuvChromaSample::Yuv422
                || chroma_subsampling == YuvChromaSample::Yuv420
            {
                let next_px = x + 1;
                if next_px < width as usize {
                    let y_value: i32 = match endianness {
                        YuvEndianness::BigEndian => {
                            let mut y_vl =
                                u16::from_be(y_ld_ptr.add(next_px).read_unaligned()) as i32;
                            if bytes_position == YuvBytesPacking::MostSignificantBytes {
                                y_vl >>= 6;
                            }
                            (y_vl - bias_y) * y_coef
                        }
                        YuvEndianness::LittleEndian => {
                            let mut y_vl =
                                u16::from_le(y_ld_ptr.add(next_px).read_unaligned()) as i32;
                            if bytes_position == YuvBytesPacking::MostSignificantBytes {
                                y_vl >>= 6;
                            }
                            (y_vl - bias_y) * y_coef
                        }
                    };

                    let r_u16 = (y_value + cr_coef * cr_value + ROUNDING_CONST) >> 8;
                    let b_u16 = (y_value + cb_coef * cb_value + ROUNDING_CONST) >> 8;
                    let g_u16 =
                        (y_value - g_coef_1 * cr_value - g_coef_2 * cb_value + ROUNDING_CONST) >> 8;

                    let r = r_u16.min(255).max(0);
                    let b = b_u16.min(255).max(0);
                    let g = g_u16.min(255).max(0);

                    let px = next_px * channels;
                    let rgb_offset = dst_offset + px;
                    let dst_slice = bgra.get_unchecked_mut(rgb_offset..);
                    *dst_slice.get_unchecked_mut(dst_chans.get_b_channel_offset()) = b as u8;
                    *dst_slice.get_unchecked_mut(dst_chans.get_g_channel_offset()) = g as u8;
                    *dst_slice.get_unchecked_mut(dst_chans.get_r_channel_offset()) = r as u8;
                    if dst_chans.has_alpha() {
                        *dst_slice.get_unchecked_mut(dst_chans.get_a_channel_offset()) = 255;
                    }
                }
            }

            _ux += 2;
        }
    });

    Ok(())
}

/// Convert YUV NV12 format with 10-bit pixel format to BGRA format.
///
/// This function takes YUV NV12 data with 10-bit precision
/// and converts it to BGRA format with 8-bit precision.
///
/// # Arguments
///
/// * `bi_planar_image` - Source Bi-Planar 10-bit image.
/// * `bgra` - A mutable slice to store the converted BGRA data.
/// * `bgra_stride` - The stride (bytes per row) for the BGRA image data.
/// * `endianness` - The endianness of stored bytes
/// * `bytes_packing` - position of significant bytes ( most significant or least significant ) if it in most significant it should be stated as per Apple *kCVPixelFormatType_422YpCbCr10BiPlanarFullRange/kCVPixelFormatType_422YpCbCr10BiPlanarVideoRange*
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGRA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv_nv12_p10_to_bgra(
    bi_planar_image: &YuvBiPlanarImage<u16>,
    bgra: &mut [u8],
    bgra_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    endianness: YuvEndianness,
    bytes_packing: YuvBytesPacking,
) -> Result<(), YuvError> {
    let dispatcher = match endianness {
        YuvEndianness::BigEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Bgra as u8 },
                    { YuvNVOrder::UV as u8 },
                    { YuvChromaSample::Yuv420 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Bgra as u8 },
                    { YuvNVOrder::UV as u8 },
                    { YuvChromaSample::Yuv420 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
        YuvEndianness::LittleEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Bgra as u8 },
                    { YuvNVOrder::UV as u8 },
                    { YuvChromaSample::Yuv420 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Bgra as u8 },
                    { YuvNVOrder::UV as u8 },
                    { YuvChromaSample::Yuv420 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
    };
    dispatcher(bi_planar_image, bgra, bgra_stride, range, matrix)
}

/// Convert YUV NV12 format with 10-bit pixel format to RGBA format.
///
/// This function takes YUV NV12 data with 10-bit precision
/// and converts it to RGBA format with 8-bit precision.
///
/// # Arguments
///
/// * `bi_planar_image` - Source Bi-Planar 10-bit image.
/// * `rgba` - A mutable slice to store the converted RGBA data.
/// * `rgba_stride` - The stride (bytes per row) for the RGBA image data.
/// * `endianness` - The endianness of stored bytes
/// * `bytes_packing` - position of significant bytes ( most significant or least significant ) if it in most significant it should be stated as per Apple *kCVPixelFormatType_422YpCbCr10BiPlanarFullRange/kCVPixelFormatType_422YpCbCr10BiPlanarVideoRange*
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input RGBA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv_nv12_p10_to_rgba(
    bi_planar_image: &YuvBiPlanarImage<u16>,
    rgba: &mut [u8],
    rgba_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    endianness: YuvEndianness,
    bytes_packing: YuvBytesPacking,
) -> Result<(), YuvError> {
    let dispatcher = match endianness {
        YuvEndianness::BigEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Rgba as u8 },
                    { YuvNVOrder::UV as u8 },
                    { YuvChromaSample::Yuv420 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Rgba as u8 },
                    { YuvNVOrder::UV as u8 },
                    { YuvChromaSample::Yuv420 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
        },
        YuvEndianness::LittleEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Rgba as u8 },
                    { YuvNVOrder::UV as u8 },
                    { YuvChromaSample::Yuv420 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Rgba as u8 },
                    { YuvNVOrder::UV as u8 },
                    { YuvChromaSample::Yuv420 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
    };
    dispatcher(bi_planar_image, rgba, rgba_stride, range, matrix)
}

/// Convert YUV NV12 format with 10-bit pixel format to BGR format.
///
/// This function takes YUV NV12 data with 10-bit precision
/// and converts it to BGR format with 8-bit precision.
///
/// # Arguments
///
/// * `bi_planar_image` - Source Bi-Planar 10-bit image.
/// * `bgr` - A mutable slice to store the converted BGR data.
/// * `bgr_stride` - The stride (bytes per row) for the BGR image data.
/// * `endianness` - The endianness of stored bytes
/// * `bytes_packing` - position of significant bytes ( most significant or least significant ) if it in most significant it should be stated as per Apple *kCVPixelFormatType_422YpCbCr10BiPlanarFullRange/kCVPixelFormatType_422YpCbCr10BiPlanarVideoRange*
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGR data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv_nv12_p10_to_bgr(
    bi_planar_image: &YuvBiPlanarImage<u16>,
    bgr: &mut [u8],
    bgr_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    endianness: YuvEndianness,
    bytes_packing: YuvBytesPacking,
) -> Result<(), YuvError> {
    let dispatcher = match endianness {
        YuvEndianness::BigEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Bgr as u8 },
                    { YuvNVOrder::UV as u8 },
                    { YuvChromaSample::Yuv420 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Bgr as u8 },
                    { YuvNVOrder::UV as u8 },
                    { YuvChromaSample::Yuv420 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
        YuvEndianness::LittleEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Bgr as u8 },
                    { YuvNVOrder::UV as u8 },
                    { YuvChromaSample::Yuv420 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Bgr as u8 },
                    { YuvNVOrder::UV as u8 },
                    { YuvChromaSample::Yuv420 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
    };
    dispatcher(bi_planar_image, bgr, bgr_stride, range, matrix)
}

/// Convert YUV NV12 format with 10-bit pixel format to RGB format.
///
/// This function takes YUV NV12 data with 10-bit precision
/// and converts it to RGB format with 8-bit precision.
///
/// # Arguments
///
/// * `bi_planar_image` - Source Bi-Planar 10-bit image.
/// * `rgb` - A mutable slice to store the converted RGB data.
/// * `rgb_stride` - The stride (bytes per row) for the RGB image data.
/// * `endianness` - The endianness of stored bytes
/// * `bytes_packing` - position of significant bytes ( most significant or least significant ) if it in most significant it should be stated as per Apple *kCVPixelFormatType_422YpCbCr10BiPlanarFullRange/kCVPixelFormatType_422YpCbCr10BiPlanarVideoRange*
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input RGB data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv_nv12_p10_to_rgb(
    bi_planar_image: &YuvBiPlanarImage<u16>,
    rgb: &mut [u8],
    rgb_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    endianness: YuvEndianness,
    bytes_packing: YuvBytesPacking,
) -> Result<(), YuvError> {
    let dispatcher = match endianness {
        YuvEndianness::BigEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Rgb as u8 },
                    { YuvNVOrder::UV as u8 },
                    { YuvChromaSample::Yuv420 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Rgb as u8 },
                    { YuvNVOrder::UV as u8 },
                    { YuvChromaSample::Yuv420 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
        YuvEndianness::LittleEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Rgb as u8 },
                    { YuvNVOrder::UV as u8 },
                    { YuvChromaSample::Yuv420 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Rgb as u8 },
                    { YuvNVOrder::UV as u8 },
                    { YuvChromaSample::Yuv420 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
    };
    dispatcher(bi_planar_image, rgb, rgb_stride, range, matrix)
}

/// Convert YUV NV16 format with 10-bit pixel format to BGRA format.
///
/// This function takes YUV NV16 data with 10-bit precision.
/// and converts it to BGRA format with 8-bit precision.
///
/// # Arguments
///
/// * `bi_planar_image` - Source Bi-Planar 10-bit image.
/// * `bgra` - A mutable slice to store the converted BGRA data.
/// * `bgra_stride` - The stride (bytes per row) for the BGRA image data.
/// * `endianness` - The endianness of stored bytes
/// * `bytes_packing` - position of significant bytes ( most significant or least significant ) if it in most significant it should be stated as per Apple *kCVPixelFormatType_422YpCbCr10BiPlanarFullRange/kCVPixelFormatType_422YpCbCr10BiPlanarVideoRange*
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGRA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv_nv16_p10_to_bgra(
    bi_planar_image: &YuvBiPlanarImage<u16>,
    bgra: &mut [u8],
    bgra_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    endianness: YuvEndianness,
    bytes_packing: YuvBytesPacking,
) -> Result<(), YuvError> {
    let dispatcher = match endianness {
        YuvEndianness::BigEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Bgra as u8 },
                    { YuvNVOrder::UV as u8 },
                    { YuvChromaSample::Yuv422 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Bgra as u8 },
                    { YuvNVOrder::UV as u8 },
                    { YuvChromaSample::Yuv422 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
        YuvEndianness::LittleEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Bgra as u8 },
                    { YuvNVOrder::UV as u8 },
                    { YuvChromaSample::Yuv422 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Bgra as u8 },
                    { YuvNVOrder::UV as u8 },
                    { YuvChromaSample::Yuv422 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
    };
    dispatcher(bi_planar_image, bgra, bgra_stride, range, matrix)
}

/// Convert YUV NV61 format with 10-bit pixel format to BGRA format.
///
/// This function takes YUV NV61 data with 10-bit precision.
/// and converts it to BGRA format with 8-bit precision.
///
/// # Arguments
///
/// * `bi_planar_image` - Source Bi-Planar 10-bit image.
/// * `bgra` - A mutable slice to store the converted BGRA data.
/// * `bgra_stride` - The stride (bytes per row) for the BGRA image data.
/// * `endianness` - The endianness of stored bytes
/// * `bytes_packing` - position of significant bytes ( most significant or least significant ) if it in most significant it should be stated as per Apple *kCVPixelFormatType_422YpCbCr10BiPlanarFullRange/kCVPixelFormatType_422YpCbCr10BiPlanarVideoRange*
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGRA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv_nv61_p10_to_bgra(
    bi_planar_image: &YuvBiPlanarImage<u16>,
    bgra: &mut [u8],
    bgra_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    endianness: YuvEndianness,
    bytes_packing: YuvBytesPacking,
) -> Result<(), YuvError> {
    let dispatcher = match endianness {
        YuvEndianness::BigEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Bgra as u8 },
                    { YuvNVOrder::VU as u8 },
                    { YuvChromaSample::Yuv422 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Bgra as u8 },
                    { YuvNVOrder::VU as u8 },
                    { YuvChromaSample::Yuv422 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
        YuvEndianness::LittleEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Bgra as u8 },
                    { YuvNVOrder::VU as u8 },
                    { YuvChromaSample::Yuv422 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Bgra as u8 },
                    { YuvNVOrder::VU as u8 },
                    { YuvChromaSample::Yuv422 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
    };
    dispatcher(bi_planar_image, bgra, bgra_stride, range, matrix)
}

/// Convert YUV NV16 format with 10-bit pixel format to BGR format.
///
/// This function takes YUV NV16 data with 10-bit precision.
/// and converts it to BGR format with 8-bit precision.
///
/// # Arguments
///
/// * `bi_planar_image` - Source Bi-Planar 10-bit image.
/// * `bgra` - A mutable slice to store the converted BGR data.
/// * `bgra_stride` - The stride (bytes per row) for the BGR image data.
/// * `endianness` - The endianness of stored bytes
/// * `bytes_packing` - position of significant bytes ( most significant or least significant ) if it in most significant it should be stated as per Apple *kCVPixelFormatType_422YpCbCr10BiPlanarFullRange/kCVPixelFormatType_422YpCbCr10BiPlanarVideoRange*
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGR data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv_nv16_p10_to_bgr(
    bi_planar_image: &YuvBiPlanarImage<u16>,
    bgra: &mut [u8],
    bgra_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    endianness: YuvEndianness,
    bytes_packing: YuvBytesPacking,
) -> Result<(), YuvError> {
    let dispatcher = match endianness {
        YuvEndianness::BigEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Bgr as u8 },
                    { YuvNVOrder::UV as u8 },
                    { YuvChromaSample::Yuv422 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Bgr as u8 },
                    { YuvNVOrder::UV as u8 },
                    { YuvChromaSample::Yuv422 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
        YuvEndianness::LittleEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Bgr as u8 },
                    { YuvNVOrder::UV as u8 },
                    { YuvChromaSample::Yuv422 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Bgr as u8 },
                    { YuvNVOrder::UV as u8 },
                    { YuvChromaSample::Yuv422 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
    };
    dispatcher(bi_planar_image, bgra, bgra_stride, range, matrix)
}

/// Convert YUV NV61 format with 10-bit pixel format to BGR format.
///
/// This function takes YUV NV61 data with 10-bit precision.
/// and converts it to BGR format with 8-bit precision.
///
/// # Arguments
///
/// * `bi_planar_image` - Source Bi-Planar 10-bit image.
/// * `bgra` - A mutable slice to store the converted BGR data.
/// * `bgra_stride` - The stride (bytes per row) for the BGR image data.
/// * `endianness` - The endianness of stored bytes
/// * `bytes_packing` - position of significant bytes ( most significant or least significant ) if it in most significant it should be stated as per Apple *kCVPixelFormatType_422YpCbCr10BiPlanarFullRange/kCVPixelFormatType_422YpCbCr10BiPlanarVideoRange*
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGR data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv_nv61_p10_to_bgr(
    bi_planar_image: &YuvBiPlanarImage<u16>,
    bgra: &mut [u8],
    bgra_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    endianness: YuvEndianness,
    bytes_packing: YuvBytesPacking,
) -> Result<(), YuvError> {
    let dispatcher = match endianness {
        YuvEndianness::BigEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Bgr as u8 },
                    { YuvNVOrder::VU as u8 },
                    { YuvChromaSample::Yuv422 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Bgr as u8 },
                    { YuvNVOrder::VU as u8 },
                    { YuvChromaSample::Yuv422 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
        YuvEndianness::LittleEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Bgr as u8 },
                    { YuvNVOrder::VU as u8 },
                    { YuvChromaSample::Yuv422 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Bgr as u8 },
                    { YuvNVOrder::VU as u8 },
                    { YuvChromaSample::Yuv422 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
    };
    dispatcher(bi_planar_image, bgra, bgra_stride, range, matrix)
}

/// Convert YUV NV16 format with 10-bit pixel format to RGB format.
///
/// This function takes YUV NV16 data with 10-bit precision.
/// and converts it to RGB format with 8-bit precision.
///
/// # Arguments
///
/// * `bi_planar_image` - Source Bi-Planar 10-bit image.
/// * `bgra` - A mutable slice to store the converted RGB data.
/// * `bgra_stride` - The stride (bytes per row) for the RGB image data.
/// * `endianness` - The endianness of stored bytes
/// * `bytes_packing` - position of significant bytes ( most significant or least significant ) if it in most significant it should be stated as per Apple *kCVPixelFormatType_422YpCbCr10BiPlanarFullRange/kCVPixelFormatType_422YpCbCr10BiPlanarVideoRange*
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input RGB data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv_nv16_p10_to_rgb(
    bi_planar_image: &YuvBiPlanarImage<u16>,
    bgra: &mut [u8],
    bgra_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    endianness: YuvEndianness,
    bytes_packing: YuvBytesPacking,
) -> Result<(), YuvError> {
    let dispatcher = match endianness {
        YuvEndianness::BigEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Rgb as u8 },
                    { YuvNVOrder::UV as u8 },
                    { YuvChromaSample::Yuv422 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Rgb as u8 },
                    { YuvNVOrder::UV as u8 },
                    { YuvChromaSample::Yuv422 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
        YuvEndianness::LittleEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Rgb as u8 },
                    { YuvNVOrder::UV as u8 },
                    { YuvChromaSample::Yuv422 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Rgb as u8 },
                    { YuvNVOrder::UV as u8 },
                    { YuvChromaSample::Yuv422 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
    };
    dispatcher(bi_planar_image, bgra, bgra_stride, range, matrix)
}

/// Convert YUV NV61 format with 10-bit pixel format to RGB format.
///
/// This function takes YUV NV61 data with 10-bit precision.
/// and converts it to RGB format with 8-bit precision.
///
/// # Arguments
///
/// * `bi_planar_image` - Source Bi-Planar 10-bit image.
/// * `bgra` - A mutable slice to store the converted RGB data.
/// * `bgra_stride` - The stride (bytes per row) for the RGB image data.
/// * `endianness` - The endianness of stored bytes
/// * `bytes_packing` - position of significant bytes ( most significant or least significant ) if it in most significant it should be stated as per Apple *kCVPixelFormatType_422YpCbCr10BiPlanarFullRange/kCVPixelFormatType_422YpCbCr10BiPlanarVideoRange*
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input RGB data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv_nv61_p10_to_rgb(
    bi_planar_image: &YuvBiPlanarImage<u16>,
    bgra: &mut [u8],
    bgra_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    endianness: YuvEndianness,
    bytes_packing: YuvBytesPacking,
) -> Result<(), YuvError> {
    let dispatcher = match endianness {
        YuvEndianness::BigEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Rgb as u8 },
                    { YuvNVOrder::VU as u8 },
                    { YuvChromaSample::Yuv422 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Rgb as u8 },
                    { YuvNVOrder::VU as u8 },
                    { YuvChromaSample::Yuv422 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
        YuvEndianness::LittleEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Rgb as u8 },
                    { YuvNVOrder::VU as u8 },
                    { YuvChromaSample::Yuv422 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Rgb as u8 },
                    { YuvNVOrder::VU as u8 },
                    { YuvChromaSample::Yuv422 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
    };
    dispatcher(bi_planar_image, bgra, bgra_stride, range, matrix)
}

/// Convert YUV NV16 format with 10-bit pixel format to RGB format.
///
/// This function takes YUV NV16 data with 10-bit precision.
/// and converts it to RGBA format with 8-bit precision.
///
/// # Arguments
///
/// * `bi_planar_image` - Source Bi-Planar 10-bit image.
/// * `bgra` - A mutable slice to store the converted RGBA data.
/// * `bgra_stride` - The stride (bytes per row) for the RGBA image data.
/// * `endianness` - The endianness of stored bytes
/// * `bytes_packing` - position of significant bytes ( most significant or least significant ) if it in most significant it should be stated as per Apple *kCVPixelFormatType_422YpCbCr10BiPlanarFullRange/kCVPixelFormatType_422YpCbCr10BiPlanarVideoRange*
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input RGBA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv_nv16_p10_to_rgba(
    bi_planar_image: &YuvBiPlanarImage<u16>,
    bgra: &mut [u8],
    bgra_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    endianness: YuvEndianness,
    bytes_packing: YuvBytesPacking,
) -> Result<(), YuvError> {
    let dispatcher = match endianness {
        YuvEndianness::BigEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Rgba as u8 },
                    { YuvNVOrder::UV as u8 },
                    { YuvChromaSample::Yuv422 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Rgba as u8 },
                    { YuvNVOrder::UV as u8 },
                    { YuvChromaSample::Yuv422 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
        YuvEndianness::LittleEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Rgba as u8 },
                    { YuvNVOrder::UV as u8 },
                    { YuvChromaSample::Yuv422 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Rgba as u8 },
                    { YuvNVOrder::UV as u8 },
                    { YuvChromaSample::Yuv422 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
    };
    dispatcher(bi_planar_image, bgra, bgra_stride, range, matrix)
}

/// Convert YUV NV61 format with 10-bit pixel format to RGB format.
///
/// This function takes YUV NV61 data with 10-bit precision.
/// and converts it to RGBA format with 8-bit precision.
///
/// # Arguments
///
/// * `bi_planar_image` - Source Bi-Planar 10-bit image.
/// * `bgra` - A mutable slice to store the converted RGBA data.
/// * `bgra_stride` - The stride (bytes per row) for the RGBA image data.
/// * `endianness` - The endianness of stored bytes
/// * `bytes_packing` - position of significant bytes ( most significant or least significant ) if it in most significant it should be stated as per Apple *kCVPixelFormatType_422YpCbCr10BiPlanarFullRange/kCVPixelFormatType_422YpCbCr10BiPlanarVideoRange*
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input RGBA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv_nv61_p10_to_rgba(
    bi_planar_image: &YuvBiPlanarImage<u16>,
    bgra: &mut [u8],
    bgra_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    endianness: YuvEndianness,
    bytes_packing: YuvBytesPacking,
) -> Result<(), YuvError> {
    let dispatcher = match endianness {
        YuvEndianness::BigEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Rgba as u8 },
                    { YuvNVOrder::VU as u8 },
                    { YuvChromaSample::Yuv422 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Rgba as u8 },
                    { YuvNVOrder::VU as u8 },
                    { YuvChromaSample::Yuv422 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
        YuvEndianness::LittleEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Rgba as u8 },
                    { YuvNVOrder::VU as u8 },
                    { YuvChromaSample::Yuv422 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Rgba as u8 },
                    { YuvNVOrder::VU as u8 },
                    { YuvChromaSample::Yuv422 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
    };
    dispatcher(bi_planar_image, bgra, bgra_stride, range, matrix)
}

/// Convert YUV NV21 format with 10-bit pixel format to BGR format.
///
/// This function takes YUV NV21 data with 10-bit precision
/// and converts it to BGR format with 8-bit precision.
///
/// # Arguments
///
/// * `bi_planar_image` - Source Bi-Planar 10-bit image.
/// * `bgr` - A mutable slice to store the converted BGR data.
/// * `bgr_stride` - The stride (bytes per row) for the BGR image data.
/// * `endianness` - The endianness of stored bytes
/// * `bytes_packing` - position of significant bytes ( most significant or least significant ) if it in most significant it should be stated as per Apple *kCVPixelFormatType_422YpCbCr10BiPlanarFullRange/kCVPixelFormatType_422YpCbCr10BiPlanarVideoRange*
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGR data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv_nv21_p10_to_bgr(
    bi_planar_image: &YuvBiPlanarImage<u16>,
    bgr: &mut [u8],
    bgr_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    endianness: YuvEndianness,
    bytes_packing: YuvBytesPacking,
) -> Result<(), YuvError> {
    let dispatcher = match endianness {
        YuvEndianness::BigEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Bgr as u8 },
                    { YuvNVOrder::VU as u8 },
                    { YuvChromaSample::Yuv420 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Bgr as u8 },
                    { YuvNVOrder::VU as u8 },
                    { YuvChromaSample::Yuv420 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
        YuvEndianness::LittleEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Bgr as u8 },
                    { YuvNVOrder::VU as u8 },
                    { YuvChromaSample::Yuv420 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Bgr as u8 },
                    { YuvNVOrder::VU as u8 },
                    { YuvChromaSample::Yuv420 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
    };
    dispatcher(bi_planar_image, bgr, bgr_stride, range, matrix)
}

/// Convert YUV NV21 format with 10-bit pixel format to BGRA format.
///
/// This function takes YUV NV21 data with 10-bit precision
/// and converts it to BGRA format with 8-bit precision.
///
/// # Arguments
///
/// * `bi_planar_image` - Source Bi-Planar 10-bit image.
/// * `bgra` - A mutable slice to store the converted BGRA data.
/// * `bgra_stride` - The stride (bytes per row) for the BGRA image data.
/// * `endianness` - The endianness of stored bytes
/// * `bytes_packing` - position of significant bytes ( most significant or least significant ) if it in most significant it should be stated as per Apple *kCVPixelFormatType_422YpCbCr10BiPlanarFullRange/kCVPixelFormatType_422YpCbCr10BiPlanarVideoRange*
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGRA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv_nv21_p10_to_bgra(
    bi_planar_image: &YuvBiPlanarImage<u16>,
    bgra: &mut [u8],
    bgra_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    endianness: YuvEndianness,
    bytes_packing: YuvBytesPacking,
) -> Result<(), YuvError> {
    let dispatcher = match endianness {
        YuvEndianness::BigEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Bgra as u8 },
                    { YuvNVOrder::VU as u8 },
                    { YuvChromaSample::Yuv420 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Bgra as u8 },
                    { YuvNVOrder::VU as u8 },
                    { YuvChromaSample::Yuv420 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
        YuvEndianness::LittleEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Bgra as u8 },
                    { YuvNVOrder::VU as u8 },
                    { YuvChromaSample::Yuv420 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Bgra as u8 },
                    { YuvNVOrder::VU as u8 },
                    { YuvChromaSample::Yuv420 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
    };
    dispatcher(bi_planar_image, bgra, bgra_stride, range, matrix)
}

/// Convert YUV NV21 format with 10-bit pixel format to RGB format.
///
/// This function takes YUV NV21 data with 10-bit precision
/// and converts it to RGB format with 8-bit precision.
///
/// # Arguments
///
/// * `bi_planar_image` - Source Bi-Planar 10-bit image.
/// * `rgb` - A mutable slice to store the converted RGB data.
/// * `rgb_stride` - The stride (bytes per row) for the RGB image data.
/// * `endianness` - The endianness of stored bytes
/// * `bytes_packing` - position of significant bytes ( most significant or least significant ) if it in most significant it should be stated as per Apple *kCVPixelFormatType_422YpCbCr10BiPlanarFullRange/kCVPixelFormatType_422YpCbCr10BiPlanarVideoRange*
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input RGB data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv_nv21_p10_to_rgb(
    bi_planar_image: &YuvBiPlanarImage<u16>,
    rgb: &mut [u8],
    rgb_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    endianness: YuvEndianness,
    bytes_packing: YuvBytesPacking,
) -> Result<(), YuvError> {
    let dispatcher = match endianness {
        YuvEndianness::BigEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Rgb as u8 },
                    { YuvNVOrder::VU as u8 },
                    { YuvChromaSample::Yuv420 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Rgb as u8 },
                    { YuvNVOrder::VU as u8 },
                    { YuvChromaSample::Yuv420 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
        YuvEndianness::LittleEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Rgb as u8 },
                    { YuvNVOrder::VU as u8 },
                    { YuvChromaSample::Yuv420 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Rgb as u8 },
                    { YuvNVOrder::VU as u8 },
                    { YuvChromaSample::Yuv420 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
    };
    dispatcher(bi_planar_image, rgb, rgb_stride, range, matrix)
}

/// Convert YUV NV21 format with 10-bit pixel format to RGBA format.
///
/// This function takes YUV NV21 data with 10-bit precision
/// and converts it to RGBA format with 8-bit precision.
///
/// # Arguments
///
/// * `bi_planar_image` - Source Bi-Planar 10-bit image.
/// * `rgba` - A mutable slice to store the converted RGBA data.
/// * `rgba_stride` - The stride (bytes per row) for the RGBA image data.
/// * `endianness` - The endianness of stored bytes
/// * `bytes_packing` - position of significant bytes ( most significant or least significant ) if it in most significant it should be stated as per Apple *kCVPixelFormatType_422YpCbCr10BiPlanarFullRange/kCVPixelFormatType_422YpCbCr10BiPlanarVideoRange*
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input RGBA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv_nv21_p10_to_rgba(
    bi_planar_image: &YuvBiPlanarImage<u16>,
    rgba: &mut [u8],
    rgba_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    endianness: YuvEndianness,
    bytes_packing: YuvBytesPacking,
) -> Result<(), YuvError> {
    let dispatcher = match endianness {
        YuvEndianness::BigEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Rgba as u8 },
                    { YuvNVOrder::VU as u8 },
                    { YuvChromaSample::Yuv420 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Rgba as u8 },
                    { YuvNVOrder::VU as u8 },
                    { YuvChromaSample::Yuv420 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
        YuvEndianness::LittleEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Rgba as u8 },
                    { YuvNVOrder::VU as u8 },
                    { YuvChromaSample::Yuv420 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Rgba as u8 },
                    { YuvNVOrder::VU as u8 },
                    { YuvChromaSample::Yuv420 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
    };
    dispatcher(bi_planar_image, rgba, rgba_stride, range, matrix)
}
