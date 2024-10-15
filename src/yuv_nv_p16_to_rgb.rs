/*
 * // Copyright (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use crate::neon::neon_yuv_nv_p16_to_rgba_row;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use crate::sse::sse_yuv_nv_p16_to_rgba_row;
use crate::yuv_support::*;
#[cfg(feature = "rayon")]
use rayon::iter::{IndexedParallelIterator, ParallelIterator};
#[cfg(feature = "rayon")]
use rayon::prelude::ParallelSliceMut;
use std::slice;

fn yuv_nv_p16_to_image_impl<
    const DESTINATION_CHANNELS: u8,
    const NV_ORDER: u8,
    const SAMPLING: u8,
    const ENDIANNESS: u8,
    const BYTES_POSITION: u8,
    const BIT_DEPTH: u8,
>(
    y_plane: &[u16],
    y_stride: u32,
    uv_plane: &[u16],
    uv_stride: u32,
    bgra: &mut [u16],
    bgra_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) {
    let dst_chans: YuvSourceChannels = DESTINATION_CHANNELS.into();
    let channels = dst_chans.get_channels_count();
    let uv_order: YuvNVOrder = NV_ORDER.into();
    let chroma_subsampling: YuvChromaSample = SAMPLING.into();
    let endianness: YuvEndianness = ENDIANNESS.into();
    let bytes_position: YuvBytesPacking = BYTES_POSITION.into();
    let range = get_yuv_range(BIT_DEPTH as u32, range);
    let kr_kb = get_kr_kb(matrix);
    let max_range = ((1u32 << (BIT_DEPTH as u32)) - 1u32) as i32;
    let transform = get_inverse_transform(
        max_range as u32,
        range.range_y,
        range.range_uv,
        kr_kb.kr,
        kr_kb.kb,
    );
    const PRECISION: i32 = 6;
    const ROUNDING_CONST: i32 = 1 << (PRECISION - 1);
    let i_transform = transform.to_integers(PRECISION as u32);
    let cr_coef = i_transform.cr_coef;
    let cb_coef = i_transform.cb_coef;
    let y_coef = i_transform.y_coef;
    let g_coef_1 = i_transform.g_coeff_1;
    let g_coef_2 = i_transform.g_coeff_2;

    let bias_y = range.bias_y as i32;
    let bias_uv = range.bias_uv as i32;

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    let mut _use_sse = std::arch::is_x86_feature_detected!("sse4.1");

    let casted_slice = unsafe {
        slice::from_raw_parts_mut(
            bgra.as_mut_ptr() as *mut u8,
            height as usize * bgra_stride as usize,
        )
    };

    let iterator_step = match chroma_subsampling {
        YuvChromaSample::YUV420 => 2usize,
        YuvChromaSample::YUV422 => 2usize,
        YuvChromaSample::YUV444 => 1usize,
    };

    let iter;
    #[cfg(feature = "rayon")]
    {
        iter = casted_slice.par_chunks_exact_mut(bgra_stride as usize);
    }
    #[cfg(not(feature = "rayon"))]
    {
        iter = casted_slice.chunks_exact_mut(bgra_stride as usize);
    }

    iter.enumerate().for_each(|(y, bgra)| unsafe {
        let y_offset = y * (y_stride as usize);
        let uv_offset = if chroma_subsampling == YuvChromaSample::YUV420 {
            (y >> 1) * (uv_stride as usize)
        } else {
            y * (uv_stride as usize)
        };

        let mut _cx = 0usize;
        let mut _ux = 0usize;

        let msb_shift = 16 - BIT_DEPTH;

        let y_src_ptr = y_plane.as_ptr() as *const u8;
        let uv_src_ptr = uv_plane.as_ptr() as *const u8;

        let y_ld_ptr = y_src_ptr.add(y_offset) as *const u16;
        let uv_ld_ptr = uv_src_ptr.add(uv_offset) as *const u16;
        let dst_st_ptr = bgra.as_mut_ptr() as *mut u16;

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if _use_sse {
                let processed = sse_yuv_nv_p16_to_rgba_row::<
                    DESTINATION_CHANNELS,
                    NV_ORDER,
                    SAMPLING,
                    ENDIANNESS,
                    BYTES_POSITION,
                    BIT_DEPTH,
                >(
                    y_ld_ptr,
                    uv_ld_ptr,
                    dst_st_ptr,
                    width,
                    &range,
                    &i_transform,
                    _cx,
                    _ux,
                );
                _cx = processed.cx;
                _ux = processed.ux;
            }
        }

        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        {
            let processed = neon_yuv_nv_p16_to_rgba_row::<
                DESTINATION_CHANNELS,
                NV_ORDER,
                SAMPLING,
                ENDIANNESS,
                BYTES_POSITION,
                BIT_DEPTH,
            >(
                y_ld_ptr,
                uv_ld_ptr,
                dst_st_ptr,
                width,
                &range,
                &i_transform,
                _cx,
                _ux,
            );
            _cx = processed.cx;
            _ux = processed.ux;
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
                        y_vl >>= msb_shift;
                        cb_vl >>= msb_shift;
                        cr_vl >>= msb_shift;
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
                        y_vl >>= msb_shift;
                        cb_vl >>= msb_shift;
                        cr_vl >>= msb_shift;
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
            let r_p16 = (y_value + cr_coef * cr_value + ROUNDING_CONST) >> PRECISION;
            let b_p16 = (y_value + cb_coef * cb_value + ROUNDING_CONST) >> PRECISION;
            let g_p16 =
                (y_value - g_coef_1 * cr_value - g_coef_2 * cb_value + ROUNDING_CONST) >> PRECISION;

            let r = r_p16.clamp(0, max_range);
            let b = b_p16.clamp(0, max_range);
            let g = g_p16.clamp(0, max_range);

            let px = x * channels;

            let dst_store = dst_st_ptr.add(px);
            dst_store
                .add(dst_chans.get_b_channel_offset())
                .write_unaligned(b as u16);
            dst_store
                .add(dst_chans.get_g_channel_offset())
                .write_unaligned(g as u16);
            dst_store
                .add(dst_chans.get_r_channel_offset())
                .write_unaligned(r as u16);
            if dst_chans.has_alpha() {
                dst_store
                    .add(dst_chans.get_a_channel_offset())
                    .write_unaligned(max_range as u16);
            }

            if chroma_subsampling == YuvChromaSample::YUV422
                || chroma_subsampling == YuvChromaSample::YUV420
            {
                let next_px = x + 1;
                if next_px < width as usize {
                    let y_value: i32 = match endianness {
                        YuvEndianness::BigEndian => {
                            let mut y_vl =
                                u16::from_be(y_ld_ptr.add(next_px).read_unaligned()) as i32;
                            if bytes_position == YuvBytesPacking::MostSignificantBytes {
                                y_vl >>= msb_shift;
                            }
                            (y_vl - bias_y) * y_coef
                        }
                        YuvEndianness::LittleEndian => {
                            let mut y_vl =
                                u16::from_le(y_ld_ptr.add(next_px).read_unaligned()) as i32;
                            if bytes_position == YuvBytesPacking::MostSignificantBytes {
                                y_vl >>= msb_shift;
                            }
                            (y_vl - bias_y) * y_coef
                        }
                    };

                    let r_p16 = (y_value + cr_coef * cr_value + ROUNDING_CONST) >> PRECISION;
                    let b_p16 = (y_value + cb_coef * cb_value + ROUNDING_CONST) >> PRECISION;
                    let g_p16 = (y_value - g_coef_1 * cr_value - g_coef_2 * cb_value
                        + ROUNDING_CONST)
                        >> PRECISION;

                    let r = r_p16.clamp(0, max_range);
                    let b = b_p16.clamp(0, max_range);
                    let g = g_p16.clamp(0, max_range);

                    let px = next_px * channels;
                    let dst_store = dst_st_ptr.add(px);
                    dst_store
                        .add(dst_chans.get_b_channel_offset())
                        .write_unaligned(b as u16);
                    dst_store
                        .add(dst_chans.get_g_channel_offset())
                        .write_unaligned(g as u16);
                    dst_store
                        .add(dst_chans.get_r_channel_offset())
                        .write_unaligned(r as u16);
                    if dst_chans.has_alpha() {
                        dst_store
                            .add(dst_chans.get_a_channel_offset())
                            .write_unaligned(max_range as u16);
                    }
                }
            }

            _ux += 2;
        }
    });
}

fn yuv_nv_p16_to_image<
    const DESTINATION_CHANNELS: u8,
    const NV_ORDER: u8,
    const SAMPLING: u8,
    const ENDIANNESS: u8,
    const BYTES_POSITION: u8,
>(
    y_plane: &[u16],
    y_stride: u32,
    uv_plane: &[u16],
    uv_stride: u32,
    bgra: &mut [u16],
    bgra_stride: u32,
    bit_depth: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) {
    if bit_depth == 10 {
        yuv_nv_p16_to_image_impl::<
            DESTINATION_CHANNELS,
            NV_ORDER,
            SAMPLING,
            ENDIANNESS,
            BYTES_POSITION,
            10,
        >(
            y_plane,
            y_stride,
            uv_plane,
            uv_stride,
            bgra,
            bgra_stride,
            width,
            height,
            range,
            matrix,
        );
    } else if bit_depth == 12 {
        yuv_nv_p16_to_image_impl::<
            DESTINATION_CHANNELS,
            NV_ORDER,
            SAMPLING,
            ENDIANNESS,
            BYTES_POSITION,
            12,
        >(
            y_plane,
            y_stride,
            uv_plane,
            uv_stride,
            bgra,
            bgra_stride,
            width,
            height,
            range,
            matrix,
        );
    } else {
        panic!("Bit depth {} is not implemented", bit_depth);
    }
}

/// Convert YUV NV12 format with 10/12-bit pixel format to BGRA format.
///
/// This function takes YUV NV12 data with 10/12-bit precision
/// and converts it to BGRA format.
///
/// # Arguments
///
/// * `y_plane` -  A slice containing Y (luminance) with 10 bit depth.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `uv_plane` - A slice to load the UV (chrominance) with 10 bit depth.
/// * `uv_stride` - The stride (bytes per row) for the UV plane.
/// * `bgra` - A mutable slice to store the converted BGRA data.
/// * `bgra_stride` - The stride (bytes per row) for the BGRA image data.
/// * `bit_depth` - Only 10 or 12 bit-depth is supported
/// * `width` - The width of the YUV image.
/// * `height` - The height of the YUV image.
/// * `endianness` - The endianness of stored bytes
/// * `bytes_packing` - position of significant bytes ( most significant or least significant ) if it in most significant it should be stated as per Apple *kCVPixelFormatType_422YpCbCr10BiPlanarFullRange/kCVPixelFormatType_422YpCbCr10BiPlanarVideoRange*
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGRA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv_nv12_to_bgra_p16(
    y_plane: &[u16],
    y_stride: u32,
    uv_plane: &[u16],
    uv_stride: u32,
    bgra: &mut [u16],
    bgra_stride: u32,
    bit_depth: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    endianness: YuvEndianness,
    bytes_packing: YuvBytesPacking,
) {
    let dispatcher = match endianness {
        YuvEndianness::BigEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_nv_p16_to_image::<
                    { YuvSourceChannels::Bgra as u8 },
                    { YuvNVOrder::UV as u8 },
                    { YuvChromaSample::YUV420 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_nv_p16_to_image::<
                    { YuvSourceChannels::Bgra as u8 },
                    { YuvNVOrder::UV as u8 },
                    { YuvChromaSample::YUV420 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
        YuvEndianness::LittleEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_nv_p16_to_image::<
                    { YuvSourceChannels::Bgra as u8 },
                    { YuvNVOrder::UV as u8 },
                    { YuvChromaSample::YUV420 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_nv_p16_to_image::<
                    { YuvSourceChannels::Bgra as u8 },
                    { YuvNVOrder::UV as u8 },
                    { YuvChromaSample::YUV420 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
    };
    dispatcher(
        y_plane,
        y_stride,
        uv_plane,
        uv_stride,
        bgra,
        bgra_stride,
        bit_depth,
        width,
        height,
        range,
        matrix,
    );
}

/// Convert YUV NV12 format with 10/12-bit pixel format to RGBA format.
///
/// This function takes YUV NV12 data with 10/12-bit precision
/// and converts it to RGBA format.
///
/// # Arguments
///
/// * `y_plane` -  A slice containing Y (luminance) with 10 bit depth.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `uv_plane` - A slice to load the UV (chrominance) with 10 bit depth.
/// * `uv_stride` - The stride (bytes per row) for the UV plane.
/// * `rgba` - A mutable slice to store the converted RGBA data.
/// * `rgba_stride` - The stride (bytes per row) for the RGBA image data.
/// * `bit_depth` - Only 10 or 12 bit-depth is supported
/// * `width` - The width of the YUV image.
/// * `height` - The height of the YUV image.
/// * `endianness` - The endianness of stored bytes
/// * `bytes_packing` - position of significant bytes ( most significant or least significant ) if it in most significant it should be stated as per Apple *kCVPixelFormatType_422YpCbCr10BiPlanarFullRange/kCVPixelFormatType_422YpCbCr10BiPlanarVideoRange*
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input RGBA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv_nv12_to_rgba_p16(
    y_plane: &[u16],
    y_stride: u32,
    uv_plane: &[u16],
    uv_stride: u32,
    rgba: &mut [u16],
    rgba_stride: u32,
    bit_depth: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    endianness: YuvEndianness,
    bytes_packing: YuvBytesPacking,
) {
    let dispatcher = match endianness {
        YuvEndianness::BigEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_nv_p16_to_image::<
                    { YuvSourceChannels::Rgba as u8 },
                    { YuvNVOrder::UV as u8 },
                    { YuvChromaSample::YUV420 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_nv_p16_to_image::<
                    { YuvSourceChannels::Rgba as u8 },
                    { YuvNVOrder::UV as u8 },
                    { YuvChromaSample::YUV420 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
        YuvEndianness::LittleEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_nv_p16_to_image::<
                    { YuvSourceChannels::Rgba as u8 },
                    { YuvNVOrder::UV as u8 },
                    { YuvChromaSample::YUV420 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_nv_p16_to_image::<
                    { YuvSourceChannels::Rgba as u8 },
                    { YuvNVOrder::UV as u8 },
                    { YuvChromaSample::YUV420 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
    };
    dispatcher(
        y_plane,
        y_stride,
        uv_plane,
        uv_stride,
        rgba,
        rgba_stride,
        bit_depth,
        width,
        height,
        range,
        matrix,
    );
}

/// Convert YUV NV12 format with 10/12-bit pixel format to BGR format.
///
/// This function takes YUV NV12 data with 10/12-bit precision
/// and converts it to BGR format.
///
/// # Arguments
///
/// * `y_plane` -  A slice containing Y (luminance) with 10 bit depth.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `uv_plane` - A slice to load the UV (chrominance) with 10 bit depth.
/// * `uv_stride` - The stride (bytes per row) for the UV plane.
/// * `bgr` - A mutable slice to store the converted BGR data.
/// * `bgr_stride` - The stride (bytes per row) for the BGR image data.
/// * `bit_depth` - Only 10 or 12 bit-depth is supported
/// * `width` - The width of the YUV image.
/// * `height` - The height of the YUV image.
/// * `endianness` - The endianness of stored bytes
/// * `bytes_packing` - position of significant bytes ( most significant or least significant ) if it in most significant it should be stated as per Apple *kCVPixelFormatType_422YpCbCr10BiPlanarFullRange/kCVPixelFormatType_422YpCbCr10BiPlanarVideoRange*
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGR data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv_nv12_to_bgr_p16(
    y_plane: &[u16],
    y_stride: u32,
    uv_plane: &[u16],
    uv_stride: u32,
    bgr: &mut [u16],
    bgr_stride: u32,
    bit_depth: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    endianness: YuvEndianness,
    bytes_packing: YuvBytesPacking,
) {
    let dispatcher = match endianness {
        YuvEndianness::BigEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_nv_p16_to_image::<
                    { YuvSourceChannels::Bgr as u8 },
                    { YuvNVOrder::UV as u8 },
                    { YuvChromaSample::YUV420 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_nv_p16_to_image::<
                    { YuvSourceChannels::Bgr as u8 },
                    { YuvNVOrder::UV as u8 },
                    { YuvChromaSample::YUV420 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
        YuvEndianness::LittleEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_nv_p16_to_image::<
                    { YuvSourceChannels::Bgr as u8 },
                    { YuvNVOrder::UV as u8 },
                    { YuvChromaSample::YUV420 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_nv_p16_to_image::<
                    { YuvSourceChannels::Bgr as u8 },
                    { YuvNVOrder::UV as u8 },
                    { YuvChromaSample::YUV420 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
    };
    dispatcher(
        y_plane, y_stride, uv_plane, uv_stride, bgr, bgr_stride, bit_depth, width, height, range,
        matrix,
    );
}

/// Convert YUV NV12 format with 10/12-bit pixel format to RGB format.
///
/// This function takes YUV NV12 data with 10/12-bit precision
/// and converts it to RGB format.
///
/// # Arguments
///
/// * `y_plane` -  A slice containing Y (luminance) with 10 bit depth.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `uv_plane` - A slice to load the UV (chrominance) with 10 bit depth.
/// * `uv_stride` - The stride (bytes per row) for the UV plane.
/// * `rgb` - A mutable slice to store the converted RGB data.
/// * `rgb_stride` - The stride (bytes per row) for the RGB image data.
/// * `bit_depth` - Only 10 or 12 bit-depth is supported
/// * `width` - The width of the YUV image.
/// * `height` - The height of the YUV image.
/// * `endianness` - The endianness of stored bytes
/// * `bytes_packing` - position of significant bytes ( most significant or least significant ) if it in most significant it should be stated as per Apple *kCVPixelFormatType_422YpCbCr10BiPlanarFullRange/kCVPixelFormatType_422YpCbCr10BiPlanarVideoRange*
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input RGB data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv_nv12_to_rgb_p16(
    y_plane: &[u16],
    y_stride: u32,
    uv_plane: &[u16],
    uv_stride: u32,
    rgb: &mut [u16],
    rgb_stride: u32,
    bit_depth: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    endianness: YuvEndianness,
    bytes_packing: YuvBytesPacking,
) {
    let dispatcher = match endianness {
        YuvEndianness::BigEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_nv_p16_to_image::<
                    { YuvSourceChannels::Rgb as u8 },
                    { YuvNVOrder::UV as u8 },
                    { YuvChromaSample::YUV420 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_nv_p16_to_image::<
                    { YuvSourceChannels::Rgb as u8 },
                    { YuvNVOrder::UV as u8 },
                    { YuvChromaSample::YUV420 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
        YuvEndianness::LittleEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_nv_p16_to_image::<
                    { YuvSourceChannels::Rgb as u8 },
                    { YuvNVOrder::UV as u8 },
                    { YuvChromaSample::YUV420 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_nv_p16_to_image::<
                    { YuvSourceChannels::Rgb as u8 },
                    { YuvNVOrder::UV as u8 },
                    { YuvChromaSample::YUV420 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
    };
    dispatcher(
        y_plane, y_stride, uv_plane, uv_stride, rgb, rgb_stride, bit_depth, width, height, range,
        matrix,
    );
}

/// Convert YUV NV16 format with 10/12-bit pixel format to BGRA format.
///
/// This function takes YUV NV16 data with 10/12-bit precision.
/// and converts it to BGRA format.
///
/// # Arguments
///
/// * `y_plane` -  A slice containing Y (luminance) with 10 bit depth.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `uv_plane` - A slice to load the UV (chrominance) with 10 bit depth.
/// * `uv_stride` - The stride (bytes per row) for the UV plane.
/// * `bgra` - A mutable slice to store the converted BGRA data.
/// * `bgra_stride` - The stride (bytes per row) for the BGRA image data.
/// * `bit_depth` - Only 10 or 12 bit-depth is supported
/// * `width` - The width of the YUV image.
/// * `height` - The height of the YUV image.
/// * `endianness` - The endianness of stored bytes
/// * `bytes_packing` - position of significant bytes ( most significant or least significant ) if it in most significant it should be stated as per Apple *kCVPixelFormatType_422YpCbCr10BiPlanarFullRange/kCVPixelFormatType_422YpCbCr10BiPlanarVideoRange*
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGRA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv_nv16_to_bgra_p16(
    y_plane: &[u16],
    y_stride: u32,
    uv_plane: &[u16],
    uv_stride: u32,
    bgra: &mut [u16],
    bgra_stride: u32,
    bit_depth: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    endianness: YuvEndianness,
    bytes_packing: YuvBytesPacking,
) {
    let dispatcher = match endianness {
        YuvEndianness::BigEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_nv_p16_to_image::<
                    { YuvSourceChannels::Bgra as u8 },
                    { YuvNVOrder::UV as u8 },
                    { YuvChromaSample::YUV422 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_nv_p16_to_image::<
                    { YuvSourceChannels::Bgra as u8 },
                    { YuvNVOrder::UV as u8 },
                    { YuvChromaSample::YUV422 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
        YuvEndianness::LittleEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_nv_p16_to_image::<
                    { YuvSourceChannels::Bgra as u8 },
                    { YuvNVOrder::UV as u8 },
                    { YuvChromaSample::YUV422 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_nv_p16_to_image::<
                    { YuvSourceChannels::Bgra as u8 },
                    { YuvNVOrder::UV as u8 },
                    { YuvChromaSample::YUV422 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
    };
    dispatcher(
        y_plane,
        y_stride,
        uv_plane,
        uv_stride,
        bgra,
        bgra_stride,
        bit_depth,
        width,
        height,
        range,
        matrix,
    );
}

/// Convert YUV NV61 format with 10/12-bit pixel format to BGRA format.
///
/// This function takes YUV NV61 data with 10/12-bit precision.
/// and converts it to BGRA format.
///
/// # Arguments
///
/// * `y_plane` -  A slice containing Y (luminance) with 10 bit depth.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `uv_plane` - A slice to load the UV (chrominance) with 10 bit depth.
/// * `uv_stride` - The stride (bytes per row) for the UV plane.
/// * `bgra` - A mutable slice to store the converted BGRA data.
/// * `bgra_stride` - The stride (bytes per row) for the BGRA image data.
/// * `bit_depth` - Only 10 or 12 bit-depth is supported
/// * `width` - The width of the YUV image.
/// * `height` - The height of the YUV image.
/// * `endianness` - The endianness of stored bytes
/// * `bytes_packing` - position of significant bytes ( most significant or least significant ) if it in most significant it should be stated as per Apple *kCVPixelFormatType_422YpCbCr10BiPlanarFullRange/kCVPixelFormatType_422YpCbCr10BiPlanarVideoRange*
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGRA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv_nv61_to_bgra_p16(
    y_plane: &[u16],
    y_stride: u32,
    uv_plane: &[u16],
    uv_stride: u32,
    bgra: &mut [u16],
    bgra_stride: u32,
    bit_depth: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    endianness: YuvEndianness,
    bytes_packing: YuvBytesPacking,
) {
    let dispatcher = match endianness {
        YuvEndianness::BigEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_nv_p16_to_image::<
                    { YuvSourceChannels::Bgra as u8 },
                    { YuvNVOrder::VU as u8 },
                    { YuvChromaSample::YUV422 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_nv_p16_to_image::<
                    { YuvSourceChannels::Bgra as u8 },
                    { YuvNVOrder::VU as u8 },
                    { YuvChromaSample::YUV422 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
        YuvEndianness::LittleEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_nv_p16_to_image::<
                    { YuvSourceChannels::Bgra as u8 },
                    { YuvNVOrder::VU as u8 },
                    { YuvChromaSample::YUV422 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_nv_p16_to_image::<
                    { YuvSourceChannels::Bgra as u8 },
                    { YuvNVOrder::VU as u8 },
                    { YuvChromaSample::YUV422 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
    };
    dispatcher(
        y_plane,
        y_stride,
        uv_plane,
        uv_stride,
        bgra,
        bgra_stride,
        bit_depth,
        width,
        height,
        range,
        matrix,
    );
}

/// Convert YUV NV16 format with 10/12-bit pixel format to BGR format.
///
/// This function takes YUV NV16 data with 10/12-bit precision.
/// and converts it to BGR format.
///
/// # Arguments
///
/// * `y_plane` -  A slice containing Y (luminance) with 10 bit depth.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `uv_plane` - A slice to load the UV (chrominance) with 10 bit depth.
/// * `uv_stride` - The stride (bytes per row) for the UV plane.
/// * `bgra` - A mutable slice to store the converted BGR data.
/// * `bgra_stride` - The stride (bytes per row) for the BGR image data.
/// * `bit_depth` - Only 10 or 12 bit-depth is supported
/// * `width` - The width of the YUV image.
/// * `height` - The height of the YUV image.
/// * `endianness` - The endianness of stored bytes
/// * `bytes_packing` - position of significant bytes ( most significant or least significant ) if it in most significant it should be stated as per Apple *kCVPixelFormatType_422YpCbCr10BiPlanarFullRange/kCVPixelFormatType_422YpCbCr10BiPlanarVideoRange*
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGR data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv_nv16_to_bgr_p16(
    y_plane: &[u16],
    y_stride: u32,
    uv_plane: &[u16],
    uv_stride: u32,
    bgra: &mut [u16],
    bgra_stride: u32,
    bit_depth: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    endianness: YuvEndianness,
    bytes_packing: YuvBytesPacking,
) {
    let dispatcher = match endianness {
        YuvEndianness::BigEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_nv_p16_to_image::<
                    { YuvSourceChannels::Bgr as u8 },
                    { YuvNVOrder::UV as u8 },
                    { YuvChromaSample::YUV422 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_nv_p16_to_image::<
                    { YuvSourceChannels::Bgr as u8 },
                    { YuvNVOrder::UV as u8 },
                    { YuvChromaSample::YUV422 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
        YuvEndianness::LittleEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_nv_p16_to_image::<
                    { YuvSourceChannels::Bgr as u8 },
                    { YuvNVOrder::UV as u8 },
                    { YuvChromaSample::YUV422 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_nv_p16_to_image::<
                    { YuvSourceChannels::Bgr as u8 },
                    { YuvNVOrder::UV as u8 },
                    { YuvChromaSample::YUV422 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
    };
    dispatcher(
        y_plane,
        y_stride,
        uv_plane,
        uv_stride,
        bgra,
        bgra_stride,
        bit_depth,
        width,
        height,
        range,
        matrix,
    );
}

/// Convert YUV NV61 format with 10/12-bit pixel format to BGR format.
///
/// This function takes YUV NV61 data with 10/12-bit precision.
/// and converts it to BGR format.
///
/// # Arguments
///
/// * `y_plane` -  A slice containing Y (luminance) with 10 bit depth.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `uv_plane` - A slice to load the UV (chrominance) with 10 bit depth.
/// * `uv_stride` - The stride (bytes per row) for the UV plane.
/// * `bgra` - A mutable slice to store the converted BGR data.
/// * `bgra_stride` - The stride (bytes per row) for the BGR image data.
/// * `bit_depth` - Only 10 or 12 bit-depth is supported
/// * `width` - The width of the YUV image.
/// * `height` - The height of the YUV image.
/// * `endianness` - The endianness of stored bytes
/// * `bytes_packing` - position of significant bytes ( most significant or least significant ) if it in most significant it should be stated as per Apple *kCVPixelFormatType_422YpCbCr10BiPlanarFullRange/kCVPixelFormatType_422YpCbCr10BiPlanarVideoRange*
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGR data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv_nv61_to_bgr_p16(
    y_plane: &[u16],
    y_stride: u32,
    uv_plane: &[u16],
    uv_stride: u32,
    bgra: &mut [u16],
    bgra_stride: u32,
    bit_depth: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    endianness: YuvEndianness,
    bytes_packing: YuvBytesPacking,
) {
    let dispatcher = match endianness {
        YuvEndianness::BigEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_nv_p16_to_image::<
                    { YuvSourceChannels::Bgr as u8 },
                    { YuvNVOrder::VU as u8 },
                    { YuvChromaSample::YUV422 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_nv_p16_to_image::<
                    { YuvSourceChannels::Bgr as u8 },
                    { YuvNVOrder::VU as u8 },
                    { YuvChromaSample::YUV422 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
        YuvEndianness::LittleEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_nv_p16_to_image::<
                    { YuvSourceChannels::Bgr as u8 },
                    { YuvNVOrder::VU as u8 },
                    { YuvChromaSample::YUV422 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_nv_p16_to_image::<
                    { YuvSourceChannels::Bgr as u8 },
                    { YuvNVOrder::VU as u8 },
                    { YuvChromaSample::YUV422 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
    };
    dispatcher(
        y_plane,
        y_stride,
        uv_plane,
        uv_stride,
        bgra,
        bgra_stride,
        bit_depth,
        width,
        height,
        range,
        matrix,
    );
}

/// Convert YUV NV16 format with 10/12-bit pixel format to RGB format.
///
/// This function takes YUV NV16 data with 10/12-bit precision.
/// and converts it to RGB format.
///
/// # Arguments
///
/// * `y_plane` -  A slice containing Y (luminance) with 10 bit depth.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `uv_plane` - A slice to load the UV (chrominance) with 10 bit depth.
/// * `uv_stride` - The stride (bytes per row) for the UV plane.
/// * `bgra` - A mutable slice to store the converted RGB data.
/// * `bgra_stride` - The stride (bytes per row) for the RGB image data.
/// * `bit_depth` - Only 10 or 12 bit-depth is supported
/// * `width` - The width of the YUV image.
/// * `height` - The height of the YUV image.
/// * `endianness` - The endianness of stored bytes
/// * `bytes_packing` - position of significant bytes ( most significant or least significant ) if it in most significant it should be stated as per Apple *kCVPixelFormatType_422YpCbCr10BiPlanarFullRange/kCVPixelFormatType_422YpCbCr10BiPlanarVideoRange*
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input RGB data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv_nv16_to_rgb_p16(
    y_plane: &[u16],
    y_stride: u32,
    uv_plane: &[u16],
    uv_stride: u32,
    bgra: &mut [u16],
    bgra_stride: u32,
    bit_depth: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    endianness: YuvEndianness,
    bytes_packing: YuvBytesPacking,
) {
    let dispatcher = match endianness {
        YuvEndianness::BigEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_nv_p16_to_image::<
                    { YuvSourceChannels::Rgb as u8 },
                    { YuvNVOrder::UV as u8 },
                    { YuvChromaSample::YUV422 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_nv_p16_to_image::<
                    { YuvSourceChannels::Rgb as u8 },
                    { YuvNVOrder::UV as u8 },
                    { YuvChromaSample::YUV422 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
        YuvEndianness::LittleEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_nv_p16_to_image::<
                    { YuvSourceChannels::Rgb as u8 },
                    { YuvNVOrder::UV as u8 },
                    { YuvChromaSample::YUV422 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_nv_p16_to_image::<
                    { YuvSourceChannels::Rgb as u8 },
                    { YuvNVOrder::UV as u8 },
                    { YuvChromaSample::YUV422 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
    };
    dispatcher(
        y_plane,
        y_stride,
        uv_plane,
        uv_stride,
        bgra,
        bgra_stride,
        bit_depth,
        width,
        height,
        range,
        matrix,
    );
}

/// Convert YUV NV61 format with 10/12-bit pixel format to RGB format.
///
/// This function takes YUV NV61 data with 10/12-bit precision.
/// and converts it to RGB format.
///
/// # Arguments
///
/// * `y_plane` -  A slice containing Y (luminance) with 10 bit depth.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `uv_plane` - A slice to load the UV (chrominance) with 10 bit depth.
/// * `uv_stride` - The stride (bytes per row) for the UV plane.
/// * `bgra` - A mutable slice to store the converted RGB data.
/// * `bgra_stride` - The stride (bytes per row) for the RGB image data.
/// * `bit_depth` - Only 10 or 12 bit-depth is supported
/// * `width` - The width of the YUV image.
/// * `height` - The height of the YUV image.
/// * `endianness` - The endianness of stored bytes
/// * `bytes_packing` - position of significant bytes ( most significant or least significant ) if it in most significant it should be stated as per Apple *kCVPixelFormatType_422YpCbCr10BiPlanarFullRange/kCVPixelFormatType_422YpCbCr10BiPlanarVideoRange*
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input RGB data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv_nv61_to_rgb_p16(
    y_plane: &[u16],
    y_stride: u32,
    uv_plane: &[u16],
    uv_stride: u32,
    bgra: &mut [u16],
    bgra_stride: u32,
    bit_depth: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    endianness: YuvEndianness,
    bytes_packing: YuvBytesPacking,
) {
    let dispatcher = match endianness {
        YuvEndianness::BigEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_nv_p16_to_image::<
                    { YuvSourceChannels::Rgb as u8 },
                    { YuvNVOrder::VU as u8 },
                    { YuvChromaSample::YUV422 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_nv_p16_to_image::<
                    { YuvSourceChannels::Rgb as u8 },
                    { YuvNVOrder::VU as u8 },
                    { YuvChromaSample::YUV422 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
        YuvEndianness::LittleEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_nv_p16_to_image::<
                    { YuvSourceChannels::Rgb as u8 },
                    { YuvNVOrder::VU as u8 },
                    { YuvChromaSample::YUV422 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_nv_p16_to_image::<
                    { YuvSourceChannels::Rgb as u8 },
                    { YuvNVOrder::VU as u8 },
                    { YuvChromaSample::YUV422 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
    };
    dispatcher(
        y_plane,
        y_stride,
        uv_plane,
        uv_stride,
        bgra,
        bgra_stride,
        bit_depth,
        width,
        height,
        range,
        matrix,
    );
}

/// Convert YUV NV16 format with 10/12-bit pixel format to RGB format.
///
/// This function takes YUV NV16 data with 10/12-bit precision.
/// and converts it to RGBA format.
///
/// # Arguments
///
/// * `y_plane` -  A slice containing Y (luminance) with 10 bit depth.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `uv_plane` - A slice to load the UV (chrominance) with 10 bit depth.
/// * `uv_stride` - The stride (bytes per row) for the UV plane.
/// * `bgra` - A mutable slice to store the converted RGBA data.
/// * `bgra_stride` - The stride (bytes per row) for the RGBA image data.
/// * `bit_depth` - Only 10 or 12 bit-depth is supported
/// * `width` - The width of the YUV image.
/// * `height` - The height of the YUV image.
/// * `endianness` - The endianness of stored bytes
/// * `bytes_packing` - position of significant bytes ( most significant or least significant ) if it in most significant it should be stated as per Apple *kCVPixelFormatType_422YpCbCr10BiPlanarFullRange/kCVPixelFormatType_422YpCbCr10BiPlanarVideoRange*
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input RGBA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv_nv16_to_rgba_p16(
    y_plane: &[u16],
    y_stride: u32,
    uv_plane: &[u16],
    uv_stride: u32,
    bgra: &mut [u16],
    bgra_stride: u32,
    bit_depth: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    endianness: YuvEndianness,
    bytes_packing: YuvBytesPacking,
) {
    let dispatcher = match endianness {
        YuvEndianness::BigEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_nv_p16_to_image::<
                    { YuvSourceChannels::Rgba as u8 },
                    { YuvNVOrder::UV as u8 },
                    { YuvChromaSample::YUV422 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_nv_p16_to_image::<
                    { YuvSourceChannels::Rgba as u8 },
                    { YuvNVOrder::UV as u8 },
                    { YuvChromaSample::YUV422 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
        YuvEndianness::LittleEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_nv_p16_to_image::<
                    { YuvSourceChannels::Rgba as u8 },
                    { YuvNVOrder::UV as u8 },
                    { YuvChromaSample::YUV422 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_nv_p16_to_image::<
                    { YuvSourceChannels::Rgba as u8 },
                    { YuvNVOrder::UV as u8 },
                    { YuvChromaSample::YUV422 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
    };
    dispatcher(
        y_plane,
        y_stride,
        uv_plane,
        uv_stride,
        bgra,
        bgra_stride,
        bit_depth,
        width,
        height,
        range,
        matrix,
    );
}

/// Convert YUV NV61 format with 10/12-bit pixel format to RGB format.
///
/// This function takes YUV NV61 data with 10/12-bit precision.
/// and converts it to RGBA format.
///
/// # Arguments
///
/// * `y_plane` -  A slice containing Y (luminance) with 10 bit depth.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `uv_plane` - A slice to load the UV (chrominance) with 10 bit depth.
/// * `uv_stride` - The stride (bytes per row) for the UV plane.
/// * `bgra` - A mutable slice to store the converted RGBA data.
/// * `bgra_stride` - The stride (bytes per row) for the RGBA image data.
/// * `bit_depth` - Only 10 or 12 bit-depth is supported
/// * `width` - The width of the YUV image.
/// * `height` - The height of the YUV image.
/// * `endianness` - The endianness of stored bytes
/// * `bytes_packing` - position of significant bytes ( most significant or least significant ) if it in most significant it should be stated as per Apple *kCVPixelFormatType_422YpCbCr10BiPlanarFullRange/kCVPixelFormatType_422YpCbCr10BiPlanarVideoRange*
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input RGBA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv_nv61_to_rgba_p16(
    y_plane: &[u16],
    y_stride: u32,
    uv_plane: &[u16],
    uv_stride: u32,
    bgra: &mut [u16],
    bgra_stride: u32,
    bit_depth: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    endianness: YuvEndianness,
    bytes_packing: YuvBytesPacking,
) {
    let dispatcher = match endianness {
        YuvEndianness::BigEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_nv_p16_to_image::<
                    { YuvSourceChannels::Rgba as u8 },
                    { YuvNVOrder::VU as u8 },
                    { YuvChromaSample::YUV422 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_nv_p16_to_image::<
                    { YuvSourceChannels::Rgba as u8 },
                    { YuvNVOrder::VU as u8 },
                    { YuvChromaSample::YUV422 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
        YuvEndianness::LittleEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_nv_p16_to_image::<
                    { YuvSourceChannels::Rgba as u8 },
                    { YuvNVOrder::VU as u8 },
                    { YuvChromaSample::YUV422 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_nv_p16_to_image::<
                    { YuvSourceChannels::Rgba as u8 },
                    { YuvNVOrder::VU as u8 },
                    { YuvChromaSample::YUV422 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
    };
    dispatcher(
        y_plane,
        y_stride,
        uv_plane,
        uv_stride,
        bgra,
        bgra_stride,
        bit_depth,
        width,
        height,
        range,
        matrix,
    );
}

/// Convert YUV NV21 format with 10/12-bit pixel format to BGR format.
///
/// This function takes YUV NV21 data with 10/12-bit precision
/// and converts it to BGR format.
///
/// # Arguments
///
/// * `y_plane` -  A slice containing Y (luminance) with 10 bit depth.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `uv_plane` - A slice to load the UV (chrominance) with 10 bit depth.
/// * `uv_stride` - The stride (bytes per row) for the UV plane.
/// * `bgr` - A mutable slice to store the converted BGR data.
/// * `bgr_stride` - The stride (bytes per row) for the BGR image data.
/// * `bit_depth` - Only 10 or 12 bit-depth is supported
/// * `width` - The width of the YUV image.
/// * `height` - The height of the YUV image.
/// * `endianness` - The endianness of stored bytes
/// * `bytes_packing` - position of significant bytes ( most significant or least significant ) if it in most significant it should be stated as per Apple *kCVPixelFormatType_422YpCbCr10BiPlanarFullRange/kCVPixelFormatType_422YpCbCr10BiPlanarVideoRange*
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGR data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv_nv21_to_bgr_p16(
    y_plane: &[u16],
    y_stride: u32,
    uv_plane: &[u16],
    uv_stride: u32,
    bgr: &mut [u16],
    bgr_stride: u32,
    bit_depth: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    endianness: YuvEndianness,
    bytes_packing: YuvBytesPacking,
) {
    let dispatcher = match endianness {
        YuvEndianness::BigEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_nv_p16_to_image::<
                    { YuvSourceChannels::Bgr as u8 },
                    { YuvNVOrder::VU as u8 },
                    { YuvChromaSample::YUV420 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_nv_p16_to_image::<
                    { YuvSourceChannels::Bgr as u8 },
                    { YuvNVOrder::VU as u8 },
                    { YuvChromaSample::YUV420 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
        YuvEndianness::LittleEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_nv_p16_to_image::<
                    { YuvSourceChannels::Bgr as u8 },
                    { YuvNVOrder::VU as u8 },
                    { YuvChromaSample::YUV420 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_nv_p16_to_image::<
                    { YuvSourceChannels::Bgr as u8 },
                    { YuvNVOrder::VU as u8 },
                    { YuvChromaSample::YUV420 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
    };
    dispatcher(
        y_plane, y_stride, uv_plane, uv_stride, bgr, bgr_stride, bit_depth, width, height, range,
        matrix,
    );
}

/// Convert YUV NV21 format with 10/12-bit pixel format to BGRA format.
///
/// This function takes YUV NV21 data with 10/12-bit precision
/// and converts it to BGRA format.
///
/// # Arguments
///
/// * `y_plane` -  A slice containing Y (luminance) with 10 bit depth.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `uv_plane` - A slice to load the UV (chrominance) with 10 bit depth.
/// * `uv_stride` - The stride (bytes per row) for the UV plane.
/// * `bgra` - A mutable slice to store the converted BGRA data.
/// * `bgra_stride` - The stride (bytes per row) for the BGRA image data.
/// * `bit_depth` - Only 10 or 12 bit-depth is supported
/// * `width` - The width of the YUV image.
/// * `height` - The height of the YUV image.
/// * `endianness` - The endianness of stored bytes
/// * `bytes_packing` - position of significant bytes ( most significant or least significant ) if it in most significant it should be stated as per Apple *kCVPixelFormatType_422YpCbCr10BiPlanarFullRange/kCVPixelFormatType_422YpCbCr10BiPlanarVideoRange*
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGRA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv_nv21_to_bgra_p16(
    y_plane: &[u16],
    y_stride: u32,
    uv_plane: &[u16],
    uv_stride: u32,
    bgra: &mut [u16],
    bgra_stride: u32,
    bit_depth: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    endianness: YuvEndianness,
    bytes_packing: YuvBytesPacking,
) {
    let dispatcher = match endianness {
        YuvEndianness::BigEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_nv_p16_to_image::<
                    { YuvSourceChannels::Bgra as u8 },
                    { YuvNVOrder::VU as u8 },
                    { YuvChromaSample::YUV420 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_nv_p16_to_image::<
                    { YuvSourceChannels::Bgra as u8 },
                    { YuvNVOrder::VU as u8 },
                    { YuvChromaSample::YUV420 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
        YuvEndianness::LittleEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_nv_p16_to_image::<
                    { YuvSourceChannels::Bgra as u8 },
                    { YuvNVOrder::VU as u8 },
                    { YuvChromaSample::YUV420 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_nv_p16_to_image::<
                    { YuvSourceChannels::Bgra as u8 },
                    { YuvNVOrder::VU as u8 },
                    { YuvChromaSample::YUV420 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
    };
    dispatcher(
        y_plane,
        y_stride,
        uv_plane,
        uv_stride,
        bgra,
        bgra_stride,
        bit_depth,
        width,
        height,
        range,
        matrix,
    );
}

/// Convert YUV NV21 format with 10/12-bit pixel format to RGB format.
///
/// This function takes YUV NV21 data with 10/12-bit precision
/// and converts it to RGB format.
///
/// # Arguments
///
/// * `y_plane` -  A slice containing Y (luminance) with 10 bit depth.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `uv_plane` - A slice to load the UV (chrominance) with 10 bit depth.
/// * `uv_stride` - The stride (bytes per row) for the UV plane.
/// * `rgb` - A mutable slice to store the converted RGB data.
/// * `rgb_stride` - The stride (bytes per row) for the RGB image data.
/// * `bit_depth` - Only 10 or 12 bit-depth is supported
/// * `width` - The width of the YUV image.
/// * `height` - The height of the YUV image.
/// * `endianness` - The endianness of stored bytes
/// * `bytes_packing` - position of significant bytes ( most significant or least significant ) if it in most significant it should be stated as per Apple *kCVPixelFormatType_422YpCbCr10BiPlanarFullRange/kCVPixelFormatType_422YpCbCr10BiPlanarVideoRange*
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input RGB data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv_nv21_to_rgb_p16(
    y_plane: &[u16],
    y_stride: u32,
    uv_plane: &[u16],
    uv_stride: u32,
    rgb: &mut [u16],
    rgb_stride: u32,
    bit_depth: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    endianness: YuvEndianness,
    bytes_packing: YuvBytesPacking,
) {
    let dispatcher = match endianness {
        YuvEndianness::BigEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_nv_p16_to_image::<
                    { YuvSourceChannels::Rgb as u8 },
                    { YuvNVOrder::VU as u8 },
                    { YuvChromaSample::YUV420 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_nv_p16_to_image::<
                    { YuvSourceChannels::Rgb as u8 },
                    { YuvNVOrder::VU as u8 },
                    { YuvChromaSample::YUV420 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
        YuvEndianness::LittleEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_nv_p16_to_image::<
                    { YuvSourceChannels::Rgb as u8 },
                    { YuvNVOrder::VU as u8 },
                    { YuvChromaSample::YUV420 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_nv_p16_to_image::<
                    { YuvSourceChannels::Rgb as u8 },
                    { YuvNVOrder::VU as u8 },
                    { YuvChromaSample::YUV420 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
    };
    dispatcher(
        y_plane, y_stride, uv_plane, uv_stride, rgb, rgb_stride, bit_depth, width, height, range,
        matrix,
    );
}

/// Convert YUV NV21 format with 10/12-bit pixel format to RGBA format.
///
/// This function takes YUV NV21 data with 10/12-bit precision
/// and converts it to RGBA format.
///
/// # Arguments
///
/// * `y_plane` -  A slice containing Y (luminance) with 10 bit depth.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `uv_plane` - A slice to load the UV (chrominance) with 10 bit depth.
/// * `uv_stride` - The stride (bytes per row) for the UV plane.
/// * `rgba` - A mutable slice to store the converted RGBA data.
/// * `rgba_stride` - The stride (bytes per row) for the RGBA image data.
/// * `bit_depth` - Only 10 or 12 bit-depth is supported
/// * `width` - The width of the YUV image.
/// * `height` - The height of the YUV image.
/// * `endianness` - The endianness of stored bytes
/// * `bytes_packing` - position of significant bytes ( most significant or least significant ) if it in most significant it should be stated as per Apple *kCVPixelFormatType_422YpCbCr10BiPlanarFullRange/kCVPixelFormatType_422YpCbCr10BiPlanarVideoRange*
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input RGBA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv_nv21_to_rgba_p16(
    y_plane: &[u16],
    y_stride: u32,
    uv_plane: &[u16],
    uv_stride: u32,
    rgba: &mut [u16],
    rgba_stride: u32,
    bit_depth: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    endianness: YuvEndianness,
    bytes_packing: YuvBytesPacking,
) {
    let dispatcher = match endianness {
        YuvEndianness::BigEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_nv_p16_to_image::<
                    { YuvSourceChannels::Rgba as u8 },
                    { YuvNVOrder::VU as u8 },
                    { YuvChromaSample::YUV420 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_nv_p16_to_image::<
                    { YuvSourceChannels::Rgba as u8 },
                    { YuvNVOrder::VU as u8 },
                    { YuvChromaSample::YUV420 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
        YuvEndianness::LittleEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_nv_p16_to_image::<
                    { YuvSourceChannels::Rgba as u8 },
                    { YuvNVOrder::VU as u8 },
                    { YuvChromaSample::YUV420 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_nv_p16_to_image::<
                    { YuvSourceChannels::Rgba as u8 },
                    { YuvNVOrder::VU as u8 },
                    { YuvChromaSample::YUV420 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
    };
    dispatcher(
        y_plane,
        y_stride,
        uv_plane,
        uv_stride,
        rgba,
        rgba_stride,
        bit_depth,
        width,
        height,
        range,
        matrix,
    );
}
