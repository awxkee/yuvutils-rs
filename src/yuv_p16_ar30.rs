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
use crate::numerics::{qrshr, to_ne};
use crate::yuv_error::check_rgba_destination;
use crate::yuv_support::{
    get_inverse_transform, get_yuv_range, Rgb30, YuvBytesPacking, YuvChromaSubsampling,
    YuvEndianness, YuvRange, YuvStandardMatrix,
};
use crate::{Rgb30ByteOrder, YuvError, YuvPlanarImage};
#[cfg(feature = "rayon")]
use rayon::iter::{IndexedParallelIterator, ParallelIterator};
#[cfg(feature = "rayon")]
use rayon::prelude::{ParallelSlice, ParallelSliceMut};

fn yuv_p16_to_image_ar30<
    const AR30_LAYOUT: usize,
    const AR30_STORE: usize,
    const SAMPLING: u8,
    const ENDIANNESS: u8,
    const BYTES_POSITION: u8,
    const BIT_DEPTH: usize,
>(
    image: &YuvPlanarImage<u16>,
    rgba: &mut [u32],
    rgba_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) -> Result<(), YuvError> {
    let ar30_layout: Rgb30 = AR30_LAYOUT.into();

    let chroma_subsampling: YuvChromaSubsampling = SAMPLING.into();
    let range = get_yuv_range(BIT_DEPTH as u32, range);

    image.check_constraints(chroma_subsampling)?;
    check_rgba_destination(rgba, rgba_stride, image.width, image.height, 1)?;

    let kr_kb = matrix.get_kr_kb();
    const AR30_DEPTH: usize = 10;
    let max_range_p10 = ((1u32 << AR30_DEPTH as u32) - 1) as i32;
    const PRECISION: i32 = 12;
    let transform = get_inverse_transform(
        max_range_p10 as u32,
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

    let msb_shift = (16 - BIT_DEPTH) as i32;

    let process_halved_chroma_row = |y_plane: &[u16],
                                     u_plane: &[u16],
                                     v_plane: &[u16],
                                     rgba: &mut [u32]| {
        for (((rgba, y_src), &u_src), &v_src) in rgba
            .chunks_exact_mut(2)
            .zip(y_plane.chunks_exact(2))
            .zip(u_plane.iter())
            .zip(v_plane.iter())
        {
            let y_value0 =
                (to_ne::<ENDIANNESS, BYTES_POSITION>(y_src[0], msb_shift) as i32 - bias_y) * y_coef;
            let cb_value = to_ne::<ENDIANNESS, BYTES_POSITION>(u_src, msb_shift) as i32 - bias_uv;
            let cr_value = to_ne::<ENDIANNESS, BYTES_POSITION>(v_src, msb_shift) as i32 - bias_uv;

            let r0 = qrshr::<PRECISION, AR30_DEPTH>(y_value0 + cr_coef * cr_value);
            let b0 = qrshr::<PRECISION, AR30_DEPTH>(y_value0 + cb_coef * cb_value);
            let g0 = qrshr::<PRECISION, AR30_DEPTH>(
                y_value0 - g_coef_1 * cr_value - g_coef_2 * cb_value,
            );

            let rgba_2 = &mut rgba[0..2];

            let pixel0 = ar30_layout.pack::<AR30_STORE>(r0, g0, b0);
            rgba_2[0] = pixel0;

            let y_value1 =
                (to_ne::<ENDIANNESS, BYTES_POSITION>(y_src[1], msb_shift) as i32 - bias_y) * y_coef;

            let r1 = qrshr::<PRECISION, BIT_DEPTH>(y_value1 + cr_coef * cr_value);
            let b1 = qrshr::<PRECISION, BIT_DEPTH>(y_value1 + cb_coef * cb_value);
            let g1 =
                qrshr::<PRECISION, BIT_DEPTH>(y_value1 - g_coef_1 * cr_value - g_coef_2 * cb_value);

            let pixel1 = ar30_layout.pack::<AR30_STORE>(r1, g1, b1);
            rgba_2[1] = pixel1;
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
            let rgba = rgba.chunks_exact_mut(2).last().unwrap();

            let r0 = qrshr::<PRECISION, BIT_DEPTH>(y_value0 + cr_coef * cr_value);
            let b0 = qrshr::<PRECISION, BIT_DEPTH>(y_value0 + cb_coef * cb_value);
            let g0 =
                qrshr::<PRECISION, BIT_DEPTH>(y_value0 - g_coef_1 * cr_value - g_coef_2 * cb_value);
            let pixel0 = ar30_layout.pack::<AR30_STORE>(r0, g0, b0);
            rgba[0] = pixel0;
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
            for (((rgba, &y_src), &u_src), &v_src) in rgba
                .iter_mut()
                .zip(y_plane.iter())
                .zip(u_plane.iter())
                .zip(v_plane.iter())
            {
                let y_value = (to_ne::<ENDIANNESS, BYTES_POSITION>(y_src, msb_shift) as i32
                    - bias_y)
                    * y_coef;
                let cb_value =
                    to_ne::<ENDIANNESS, BYTES_POSITION>(u_src, msb_shift) as i32 - bias_uv;
                let cr_value =
                    to_ne::<ENDIANNESS, BYTES_POSITION>(v_src, msb_shift) as i32 - bias_uv;

                let r = qrshr::<PRECISION, BIT_DEPTH>(y_value + cr_coef * cr_value);
                let b = qrshr::<PRECISION, BIT_DEPTH>(y_value + cb_coef * cb_value);
                let g = qrshr::<PRECISION, BIT_DEPTH>(
                    y_value - g_coef_1 * cr_value - g_coef_2 * cb_value,
                );

                let pixel0 = ar30_layout.pack::<AR30_STORE>(r, g, b);
                *rgba = pixel0;
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
            process_halved_chroma_row(y_plane, u_plane, v_plane, rgba);
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
                process_halved_chroma_row(y_plane, u_plane, v_plane, rgba);
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
            process_halved_chroma_row(y_plane, u_plane, v_plane, rgba);
        }
    } else {
        unreachable!();
    }

    Ok(())
}

pub(crate) fn yuv_p16_to_image_ar30_impl<
    const AR30_LAYOUT: usize,
    const SAMPLING: u8,
    const ENDIANNESS: u8,
    const BYTES_POSITION: u8,
>(
    planar_image: &YuvPlanarImage<u16>,
    rgba: &mut [u32],
    rgba_stride: u32,
    store_type: Rgb30ByteOrder,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    bit_depth: usize,
) -> Result<(), YuvError> {
    if bit_depth == 10 {
        match store_type {
            Rgb30ByteOrder::Host => yuv_p16_to_image_ar30::<
                AR30_LAYOUT,
                { Rgb30ByteOrder::Host as usize },
                SAMPLING,
                ENDIANNESS,
                BYTES_POSITION,
                10,
            >(planar_image, rgba, rgba_stride, range, matrix),
            Rgb30ByteOrder::Network => yuv_p16_to_image_ar30::<
                AR30_LAYOUT,
                { Rgb30ByteOrder::Network as usize },
                SAMPLING,
                ENDIANNESS,
                BYTES_POSITION,
                10,
            >(planar_image, rgba, rgba_stride, range, matrix),
        }
    } else if bit_depth == 12 {
        match store_type {
            Rgb30ByteOrder::Host => yuv_p16_to_image_ar30::<
                AR30_LAYOUT,
                { Rgb30ByteOrder::Host as usize },
                SAMPLING,
                ENDIANNESS,
                BYTES_POSITION,
                12,
            >(planar_image, rgba, rgba_stride, range, matrix),
            Rgb30ByteOrder::Network => yuv_p16_to_image_ar30::<
                AR30_LAYOUT,
                { Rgb30ByteOrder::Network as usize },
                SAMPLING,
                ENDIANNESS,
                BYTES_POSITION,
                12,
            >(planar_image, rgba, rgba_stride, range, matrix),
        }
    } else {
        unimplemented!("Only 10 and 12 bit is implemented on YUV16 -> AR30")
    }
}

/// Convert YUV 420 planar format with 8+ bit pixel format to AR30 (RGBA2101010) format
///
/// This function takes YUV 420 planar data with 8+ bit precision.
/// and converts it to AR30 image format
///
/// # Arguments
///
/// * `planar_image` - Source YUV planar image.
/// * `ar30` - A mutable slice to store the converted AR30 data.
/// * `ar30_stride` - The stride (components per row) for AR30 data.
/// * `byte_order` - see [Rgb30ByteOrder] for more info
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
/// * `endianness` - The endianness of stored bytes
/// * `bytes_packing` - position of significant bytes ( most significant or least significant ) if it in most significant it should be stated as per Apple *kCVPixelFormatType_422YpCbCr10BiPlanarFullRange/kCVPixelFormatType_422YpCbCr10BiPlanarVideoRange*
/// * `bit_depth` - Bit depth of source YUV planes
///
/// # Error
///
/// This function panics if the lengths of the planes or the input RGBX1010102 data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv420_p16_to_ar30(
    planar_image: &YuvPlanarImage<u16>,
    ar30: &mut [u32],
    ar30_stride: u32,
    byte_order: Rgb30ByteOrder,
    bit_depth: usize,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    endianness: YuvEndianness,
    bytes_packing: YuvBytesPacking,
) -> Result<(), YuvError> {
    let dispatcher = match endianness {
        YuvEndianness::BigEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_p16_to_image_ar30_impl::<
                    { Rgb30::Ar30 as usize },
                    { YuvChromaSubsampling::Yuv420 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_p16_to_image_ar30_impl::<
                    { Rgb30::Ar30 as usize },
                    { YuvChromaSubsampling::Yuv420 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
        YuvEndianness::LittleEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_p16_to_image_ar30_impl::<
                    { Rgb30::Ar30 as usize },
                    { YuvChromaSubsampling::Yuv420 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_p16_to_image_ar30_impl::<
                    { Rgb30::Ar30 as usize },
                    { YuvChromaSubsampling::Yuv420 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
    };
    dispatcher(
        planar_image,
        ar30,
        ar30_stride,
        byte_order,
        range,
        matrix,
        bit_depth,
    )
}

/// Convert YUV 422 planar format with 8+ bit pixel format to AR30 (RGBA2101010) format
///
/// This function takes YUV 422 planar data with 8+ bit precision.
/// and converts it to AR30 image format
///
/// # Arguments
///
/// * `planar_image` - Source YUV planar image.
/// * `ar30` - A mutable slice to store the converted AR30 data.
/// * `ar30_stride` - The stride (components per row) for AR30 data.
/// * `byte_order` - see [Rgb30ByteOrder] for more info
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
/// * `endianness` - The endianness of stored bytes
/// * `bytes_packing` - position of significant bytes ( most significant or least significant ) if it in most significant it should be stated as per Apple *kCVPixelFormatType_422YpCbCr10BiPlanarFullRange/kCVPixelFormatType_422YpCbCr10BiPlanarVideoRange*
/// * `bit_depth` - Bit depth of source YUV planes
///
/// # Error
///
/// This function panics if the lengths of the planes or the input RGBX1010102 data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv422_p16_to_ar30(
    planar_image: &YuvPlanarImage<u16>,
    ar30: &mut [u32],
    ar30_stride: u32,
    byte_order: Rgb30ByteOrder,
    bit_depth: usize,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    endianness: YuvEndianness,
    bytes_packing: YuvBytesPacking,
) -> Result<(), YuvError> {
    let dispatcher = match endianness {
        YuvEndianness::BigEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_p16_to_image_ar30_impl::<
                    { Rgb30::Ar30 as usize },
                    { YuvChromaSubsampling::Yuv422 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_p16_to_image_ar30_impl::<
                    { Rgb30::Ar30 as usize },
                    { YuvChromaSubsampling::Yuv422 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
        YuvEndianness::LittleEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_p16_to_image_ar30_impl::<
                    { Rgb30::Ar30 as usize },
                    { YuvChromaSubsampling::Yuv422 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_p16_to_image_ar30_impl::<
                    { Rgb30::Ar30 as usize },
                    { YuvChromaSubsampling::Yuv422 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
    };
    dispatcher(
        planar_image,
        ar30,
        ar30_stride,
        byte_order,
        range,
        matrix,
        bit_depth,
    )
}

/// Convert YUV 444 planar format with 8+ bit pixel format to AR30 (RGBA2101010) format
///
/// This function takes YUV 444 planar data with 8+ bit precision.
/// and converts it to AR30 image format
///
/// # Arguments
///
/// * `planar_image` - Source YUV 4:4:4 planar image.
/// * `ar30` - A mutable slice to store the converted AR30 data.
/// * `ar30_stride` - The stride (components per row) for AR30 data.
/// * `byte_order` - see [Rgb30ByteOrder] for more info
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
/// * `endianness` - The endianness of stored bytes
/// * `bytes_packing` - position of significant bytes ( most significant or least significant ) if it in most significant it should be stated as per Apple *kCVPixelFormatType_422YpCbCr10BiPlanarFullRange/kCVPixelFormatType_422YpCbCr10BiPlanarVideoRange*
/// * `bit_depth` - Bit depth of source YUV planes
///
/// # Error
///
/// This function panics if the lengths of the planes or the input RGBX1010102 data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv444_p16_to_ar30(
    planar_image: &YuvPlanarImage<u16>,
    ar30: &mut [u32],
    ar30_stride: u32,
    byte_order: Rgb30ByteOrder,
    bit_depth: usize,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    endianness: YuvEndianness,
    bytes_packing: YuvBytesPacking,
) -> Result<(), YuvError> {
    let dispatcher = match endianness {
        YuvEndianness::BigEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_p16_to_image_ar30_impl::<
                    { Rgb30::Ar30 as usize },
                    { YuvChromaSubsampling::Yuv444 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_p16_to_image_ar30_impl::<
                    { Rgb30::Ar30 as usize },
                    { YuvChromaSubsampling::Yuv444 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
        YuvEndianness::LittleEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_p16_to_image_ar30_impl::<
                    { Rgb30::Ar30 as usize },
                    { YuvChromaSubsampling::Yuv444 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_p16_to_image_ar30_impl::<
                    { Rgb30::Ar30 as usize },
                    { YuvChromaSubsampling::Yuv444 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
    };
    dispatcher(
        planar_image,
        ar30,
        ar30_stride,
        byte_order,
        range,
        matrix,
        bit_depth,
    )
}

/// Convert YUV 420 planar format with 8+ bit pixel format to AB30 (BGRA2101010) format
///
/// This function takes YUV 420 planar data with 8+ bit precision.
/// and converts it to AB30 image format
///
/// # Arguments
///
/// * `planar_image` - Source YUV planar image.
/// * `ab30` - A mutable slice to store the converted AB30 data.
/// * `ab30_stride` - The stride (components per row) for AB30 data.
/// * `byte_order` - see [Rgb30ByteOrder] for more info
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
/// * `endianness` - The endianness of stored bytes
/// * `bytes_packing` - position of significant bytes ( most significant or least significant ) if it in most significant it should be stated as per Apple *kCVPixelFormatType_422YpCbCr10BiPlanarFullRange/kCVPixelFormatType_422YpCbCr10BiPlanarVideoRange*
/// * `bit_depth` - Bit depth of source YUV planes
///
/// # Error
///
/// This function panics if the lengths of the planes or the input BGRA2101010 data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv420_p16_to_ab30(
    planar_image: &YuvPlanarImage<u16>,
    ab30: &mut [u32],
    ab30_stride: u32,
    byte_order: Rgb30ByteOrder,
    bit_depth: usize,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    endianness: YuvEndianness,
    bytes_packing: YuvBytesPacking,
) -> Result<(), YuvError> {
    let dispatcher = match endianness {
        YuvEndianness::BigEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_p16_to_image_ar30_impl::<
                    { Rgb30::Ab30 as usize },
                    { YuvChromaSubsampling::Yuv420 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_p16_to_image_ar30_impl::<
                    { Rgb30::Ab30 as usize },
                    { YuvChromaSubsampling::Yuv420 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
        YuvEndianness::LittleEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_p16_to_image_ar30_impl::<
                    { Rgb30::Ab30 as usize },
                    { YuvChromaSubsampling::Yuv420 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_p16_to_image_ar30_impl::<
                    { Rgb30::Ab30 as usize },
                    { YuvChromaSubsampling::Yuv420 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
    };
    dispatcher(
        planar_image,
        ab30,
        ab30_stride,
        byte_order,
        range,
        matrix,
        bit_depth,
    )
}

/// Convert YUV 422 planar format with 8+ bit pixel format to AB30 (BGRA2101010) format
///
/// This function takes YUV 422 planar data with 8+ bit precision.
/// and converts it to AB30 image format
///
/// # Arguments
///
/// * `planar_image` - Source YUV planar image.
/// * `ab30` - A mutable slice to store the converted AB30 data.
/// * `ab30_stride` - The stride (components per row) for AB30 data.
/// * `byte_order` - see [Rgb30ByteOrder] for more info
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
/// * `endianness` - The endianness of stored bytes
/// * `bytes_packing` - position of significant bytes ( most significant or least significant ) if it in most significant it should be stated as per Apple *kCVPixelFormatType_422YpCbCr10BiPlanarFullRange/kCVPixelFormatType_422YpCbCr10BiPlanarVideoRange*
/// * `bit_depth` - Bit depth of source YUV planes
///
/// # Error
///
/// This function panics if the lengths of the planes or the input BGRA2101010 data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv422_p16_to_ab30(
    planar_image: &YuvPlanarImage<u16>,
    ab30: &mut [u32],
    ab30_stride: u32,
    byte_order: Rgb30ByteOrder,
    bit_depth: usize,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    endianness: YuvEndianness,
    bytes_packing: YuvBytesPacking,
) -> Result<(), YuvError> {
    let dispatcher = match endianness {
        YuvEndianness::BigEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_p16_to_image_ar30_impl::<
                    { Rgb30::Ab30 as usize },
                    { YuvChromaSubsampling::Yuv422 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_p16_to_image_ar30_impl::<
                    { Rgb30::Ab30 as usize },
                    { YuvChromaSubsampling::Yuv422 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
        YuvEndianness::LittleEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_p16_to_image_ar30_impl::<
                    { Rgb30::Ab30 as usize },
                    { YuvChromaSubsampling::Yuv422 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_p16_to_image_ar30_impl::<
                    { Rgb30::Ab30 as usize },
                    { YuvChromaSubsampling::Yuv422 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
    };
    dispatcher(
        planar_image,
        ab30,
        ab30_stride,
        byte_order,
        range,
        matrix,
        bit_depth,
    )
}

/// Convert YUV 444 planar format with 8+ bit pixel format to AB30 (BGRA2101010) format
///
/// This function takes YUV 444 planar data with 8+ bit precision.
/// and converts it to AB30 image format
///
/// # Arguments
///
/// * `planar_image` - Source YUV planar image.
/// * `ab30` - A mutable slice to store the converted AB30 data.
/// * `ab30_stride` - The stride (components per row) for AB30 data.
/// * `byte_order` - see [Rgb30ByteOrder] for more info
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
/// * `endianness` - The endianness of stored bytes
/// * `bytes_packing` - position of significant bytes ( most significant or least significant ) if it in most significant it should be stated as per Apple *kCVPixelFormatType_422YpCbCr10BiPlanarFullRange/kCVPixelFormatType_422YpCbCr10BiPlanarVideoRange*
/// * `bit_depth` - Bit depth of source YUV planes
///
/// # Error
///
/// This function panics if the lengths of the planes or the input BGRA2101010 data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv444_p16_to_ab30(
    planar_image: &YuvPlanarImage<u16>,
    ab30: &mut [u32],
    ab30_stride: u32,
    byte_order: Rgb30ByteOrder,
    bit_depth: usize,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    endianness: YuvEndianness,
    bytes_packing: YuvBytesPacking,
) -> Result<(), YuvError> {
    let dispatcher = match endianness {
        YuvEndianness::BigEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_p16_to_image_ar30_impl::<
                    { Rgb30::Ab30 as usize },
                    { YuvChromaSubsampling::Yuv444 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_p16_to_image_ar30_impl::<
                    { Rgb30::Ab30 as usize },
                    { YuvChromaSubsampling::Yuv444 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
        YuvEndianness::LittleEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_p16_to_image_ar30_impl::<
                    { Rgb30::Ab30 as usize },
                    { YuvChromaSubsampling::Yuv444 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_p16_to_image_ar30_impl::<
                    { Rgb30::Ab30 as usize },
                    { YuvChromaSubsampling::Yuv444 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
    };
    dispatcher(
        planar_image,
        ab30,
        ab30_stride,
        byte_order,
        range,
        matrix,
        bit_depth,
    )
}

/// Convert YUV 420 planar format with 8+ bit pixel format to AR30 (RGBA1010102) format
///
/// This function takes YUV 420 planar data with 8+ bit precision.
/// and converts it to RA30 image format
///
/// # Arguments
///
/// * `planar_image` - Source YUV planar image.
/// * `ra30` - A mutable slice to store the converted RA30 data.
/// * `ra30_stride` - The stride (components per row) for RA30 data.
/// * `byte_order` - see [Rgb30ByteOrder] for more info
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
/// * `endianness` - The endianness of stored bytes
/// * `bytes_packing` - position of significant bytes ( most significant or least significant ) if it in most significant it should be stated as per Apple *kCVPixelFormatType_422YpCbCr10BiPlanarFullRange/kCVPixelFormatType_422YpCbCr10BiPlanarVideoRange*
/// * `bit_depth` - Bit depth of source YUV planes
///
/// # Error
///
/// This function panics if the lengths of the planes or the input RGBA1010102 data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv420_p16_to_ra30(
    planar_image: &YuvPlanarImage<u16>,
    ra30: &mut [u32],
    ra30_stride: u32,
    byte_order: Rgb30ByteOrder,
    bit_depth: usize,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    endianness: YuvEndianness,
    bytes_packing: YuvBytesPacking,
) -> Result<(), YuvError> {
    let dispatcher = match endianness {
        YuvEndianness::BigEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_p16_to_image_ar30_impl::<
                    { Rgb30::Ra30 as usize },
                    { YuvChromaSubsampling::Yuv420 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_p16_to_image_ar30_impl::<
                    { Rgb30::Ra30 as usize },
                    { YuvChromaSubsampling::Yuv420 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
        YuvEndianness::LittleEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_p16_to_image_ar30_impl::<
                    { Rgb30::Ra30 as usize },
                    { YuvChromaSubsampling::Yuv420 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_p16_to_image_ar30_impl::<
                    { Rgb30::Ra30 as usize },
                    { YuvChromaSubsampling::Yuv420 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
    };
    dispatcher(
        planar_image,
        ra30,
        ra30_stride,
        byte_order,
        range,
        matrix,
        bit_depth,
    )
}

/// Convert YUV 422 planar format with 8+ bit pixel format to AR30 (RGBA1010102) format
///
/// This function takes YUV 422 planar data with 8+ bit precision.
/// and converts it to RA30 image format
///
/// # Arguments
///
/// * `planar_image` - Source YUV planar image.
/// * `ra30` - A mutable slice to store the converted RA30 data.
/// * `ra30_stride` - The stride (components per row) for RA30 data.
/// * `byte_order` - see [Rgb30ByteOrder] for more info
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
/// * `endianness` - The endianness of stored bytes
/// * `bytes_packing` - position of significant bytes ( most significant or least significant ) if it in most significant it should be stated as per Apple *kCVPixelFormatType_422YpCbCr10BiPlanarFullRange/kCVPixelFormatType_422YpCbCr10BiPlanarVideoRange*
/// * `bit_depth` - Bit depth of source YUV planes
///
/// # Error
///
/// This function panics if the lengths of the planes or the input RGBA1010102 data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv422_p16_to_ra30(
    planar_image: &YuvPlanarImage<u16>,
    ra30: &mut [u32],
    ra30_stride: u32,
    byte_order: Rgb30ByteOrder,
    bit_depth: usize,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    endianness: YuvEndianness,
    bytes_packing: YuvBytesPacking,
) -> Result<(), YuvError> {
    let dispatcher = match endianness {
        YuvEndianness::BigEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_p16_to_image_ar30_impl::<
                    { Rgb30::Ra30 as usize },
                    { YuvChromaSubsampling::Yuv422 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_p16_to_image_ar30_impl::<
                    { Rgb30::Ra30 as usize },
                    { YuvChromaSubsampling::Yuv422 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
        YuvEndianness::LittleEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_p16_to_image_ar30_impl::<
                    { Rgb30::Ra30 as usize },
                    { YuvChromaSubsampling::Yuv422 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_p16_to_image_ar30_impl::<
                    { Rgb30::Ra30 as usize },
                    { YuvChromaSubsampling::Yuv422 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
    };
    dispatcher(
        planar_image,
        ra30,
        ra30_stride,
        byte_order,
        range,
        matrix,
        bit_depth,
    )
}

/// Convert YUV 444 planar format with 8+ bit pixel format to AR30 (RGBA1010102) format
///
/// This function takes YUV 444 planar data with 8+ bit precision.
/// and converts it to RA30 image format
///
/// # Arguments
///
/// * `planar_image` - Source YUV planar image.
/// * `ra30` - A mutable slice to store the converted RA30 data.
/// * `ra30_stride` - The stride (components per row) for RA30 data.
/// * `byte_order` - see [Rgb30ByteOrder] for more info
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
/// * `endianness` - The endianness of stored bytes
/// * `bytes_packing` - position of significant bytes ( most significant or least significant ) if it in most significant it should be stated as per Apple *kCVPixelFormatType_422YpCbCr10BiPlanarFullRange/kCVPixelFormatType_422YpCbCr10BiPlanarVideoRange*
/// * `bit_depth` - Bit depth of source YUV planes
///
/// # Error
///
/// This function panics if the lengths of the planes or the input RGBA1010102 data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv444_p16_to_ra30(
    planar_image: &YuvPlanarImage<u16>,
    ra30: &mut [u32],
    ra30_stride: u32,
    byte_order: Rgb30ByteOrder,
    bit_depth: usize,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    endianness: YuvEndianness,
    bytes_packing: YuvBytesPacking,
) -> Result<(), YuvError> {
    let dispatcher = match endianness {
        YuvEndianness::BigEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_p16_to_image_ar30_impl::<
                    { Rgb30::Ra30 as usize },
                    { YuvChromaSubsampling::Yuv444 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_p16_to_image_ar30_impl::<
                    { Rgb30::Ra30 as usize },
                    { YuvChromaSubsampling::Yuv444 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
        YuvEndianness::LittleEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_p16_to_image_ar30_impl::<
                    { Rgb30::Ra30 as usize },
                    { YuvChromaSubsampling::Yuv444 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_p16_to_image_ar30_impl::<
                    { Rgb30::Ra30 as usize },
                    { YuvChromaSubsampling::Yuv444 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
    };
    dispatcher(
        planar_image,
        ra30,
        ra30_stride,
        byte_order,
        range,
        matrix,
        bit_depth,
    )
}
