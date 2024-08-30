/*
 * // Copyright (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use crate::neon::neon_yuv_nv12_p10_to_rgba_row;
use std::slice;

use crate::yuv_support::*;

fn yuv_nv_p10_to_image_impl<
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
    bgra: &mut [u8],
    bgra_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) {
    let destination_channels: YuvSourceChannels = DESTINATION_CHANNELS.into();
    let channels = destination_channels.get_channels_count();
    let uv_order: YuvNVOrder = NV_ORDER.into();
    let chroma_subsampling: YuvChromaSample = SAMPLING.into();
    let endianness: YuvEndiannes = ENDIANNESS.into();
    let bytes_position: YuvBytesPacking = BYTES_POSITION.into();
    let range = get_yuv_range(10, range);
    let kr_kb = get_kr_kb(matrix);
    let max_range_p10 = (1u32 << 10u32) - 1u32;
    let transform = get_inverse_transform(
        max_range_p10,
        range.range_y,
        range.range_uv,
        kr_kb.kr,
        kr_kb.kb,
    );
    let i_transform = transform.to_integers(6u32);
    let cr_coef = i_transform.cr_coef;
    let cb_coef = i_transform.cb_coef;
    let y_coef = i_transform.y_coef;
    let g_coef_1 = i_transform.g_coeff_1;
    let g_coef_2 = i_transform.g_coeff_2;

    let bias_y = range.bias_y as i32;
    let bias_uv = range.bias_uv as i32;

    let mut y_offset = 0usize;
    let mut uv_offset = 0usize;
    let mut dst_offset = 0usize;

    let y_src_ptr = y_plane.as_ptr() as *const u8;
    let uv_src_ptr = uv_plane.as_ptr() as *const u8;

    let iterator_step = match chroma_subsampling {
        YuvChromaSample::YUV420 => 2usize,
        YuvChromaSample::YUV422 => 2usize,
        YuvChromaSample::YUV444 => 1usize,
    };

    for y in 0..height as usize {
        #[allow(unused_variables)]
        #[allow(unused_mut)]
        let mut cx = 0usize;

        #[allow(unused_variables)]
        #[allow(unused_mut)]
        let mut ux = 0usize;

        let y_ld_ptr = unsafe { y_src_ptr.offset(y_offset as isize) as *const u16 };
        let y_ld = unsafe { slice::from_raw_parts(y_ld_ptr, width as usize) };
        let uv_ld_ptr = unsafe { uv_src_ptr.offset(uv_offset as isize) as *const u16 };
        let uv_length = if chroma_subsampling == YuvChromaSample::YUV444 {
            width as usize * 2usize
        } else {
            width as usize
        };
        let uv_ld = unsafe { slice::from_raw_parts(uv_ld_ptr, uv_length) };

        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        unsafe {
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
                cx,
                ux,
            );
            cx = offset.cx;
            ux = offset.ux;
        }

        for x in (cx..width as usize).step_by(iterator_step) {
            let y_value: i32;
            let mut cb_value: i32;
            let mut cr_value: i32;
            match endianness {
                YuvEndiannes::BigEndian => {
                    let mut y_vl = u16::from_be(unsafe { *y_ld.get_unchecked(x) }) as i32;
                    let mut cb_vl = u16::from_be(unsafe { *uv_ld.get_unchecked(ux) }) as i32;
                    let mut cr_vl = u16::from_be(unsafe { *uv_ld.get_unchecked(ux + 1) }) as i32;
                    if bytes_position == YuvBytesPacking::MostSignificantBytes {
                        y_vl = y_vl >> 6;
                        cb_vl = cb_vl >> 6;
                        cr_vl = cr_vl >> 6;
                    }
                    y_value = (y_vl - bias_y) * y_coef;

                    cb_value = cb_vl;
                    cr_value = cr_vl;
                }
                YuvEndiannes::LittleEndian => {
                    let mut y_vl = u16::from_le(unsafe { *y_ld.get_unchecked(x) }) as i32;
                    let mut cb_vl = u16::from_le(unsafe { *uv_ld.get_unchecked(ux) }) as i32;
                    let mut cr_vl = u16::from_le(unsafe { *uv_ld.get_unchecked(ux + 1) }) as i32;
                    if bytes_position == YuvBytesPacking::MostSignificantBytes {
                        y_vl = y_vl >> 6;
                        cb_vl = cb_vl >> 6;
                        cr_vl = cr_vl >> 6;
                    }
                    y_value = (y_vl - bias_y) * y_coef;

                    cb_value = cb_vl;
                    cr_value = cr_vl;
                }
            }

            match uv_order {
                YuvNVOrder::UV => {
                    cb_value = cb_value - bias_uv;
                    cr_value = cr_value - bias_uv;
                }
                YuvNVOrder::VU => {
                    cr_value = cb_value - bias_uv;
                    cb_value = cr_value - bias_uv;
                }
            }

            // shift right 8 due to we want to make it 8 bit instead of 10

            let r_u16 = (y_value + cr_coef * cr_value) >> 8;
            let b_u16 = (y_value + cb_coef * cb_value) >> 8;
            let g_u16 = (y_value - g_coef_1 * cr_value - g_coef_2 * cb_value) >> 8;

            let r = r_u16.min(255).max(0);
            let b = b_u16.min(255).max(0);
            let g = g_u16.min(255).max(0);

            let px = x * channels;

            let rgb_offset = dst_offset + px;

            unsafe {
                *bgra.get_unchecked_mut(rgb_offset + destination_channels.get_b_channel_offset()) =
                    b as u8;
                *bgra.get_unchecked_mut(rgb_offset + destination_channels.get_g_channel_offset()) =
                    g as u8;
                *bgra.get_unchecked_mut(rgb_offset + destination_channels.get_r_channel_offset()) =
                    r as u8;
                if destination_channels.has_alpha() {
                    *bgra.get_unchecked_mut(
                        rgb_offset + destination_channels.get_a_channel_offset(),
                    ) = 255;
                }
            }

            if chroma_subsampling == YuvChromaSample::YUV422
                || chroma_subsampling == YuvChromaSample::YUV420
            {
                let next_px = x + 1;
                if next_px < width as usize {
                    let y_value: i32;
                    match endianness {
                        YuvEndiannes::BigEndian => {
                            let mut y_vl =
                                u16::from_be(unsafe { *y_ld.get_unchecked(next_px) }) as i32;
                            if bytes_position == YuvBytesPacking::MostSignificantBytes {
                                y_vl = y_vl >> 6;
                            }
                            y_value = (y_vl - bias_y) * y_coef;
                        }
                        YuvEndiannes::LittleEndian => {
                            let mut y_vl =
                                u16::from_le(unsafe { *y_ld.get_unchecked(next_px) }) as i32;
                            if bytes_position == YuvBytesPacking::MostSignificantBytes {
                                y_vl = y_vl >> 6;
                            }
                            y_value = (y_vl - bias_y) * y_coef;
                        }
                    }

                    let r_u16 = (y_value + cr_coef * cr_value) >> 8;
                    let b_u16 = (y_value + cb_coef * cb_value) >> 8;
                    let g_u16 = (y_value - g_coef_1 * cr_value - g_coef_2 * cb_value) >> 8;

                    let r = r_u16.min(255).max(0);
                    let b = b_u16.min(255).max(0);
                    let g = g_u16.min(255).max(0);

                    let px = next_px * channels;
                    let rgb_offset = dst_offset + px;
                    unsafe {
                        *bgra.get_unchecked_mut(
                            rgb_offset + destination_channels.get_b_channel_offset(),
                        ) = b as u8;
                        *bgra.get_unchecked_mut(
                            rgb_offset + destination_channels.get_g_channel_offset(),
                        ) = g as u8;
                        *bgra.get_unchecked_mut(
                            rgb_offset + destination_channels.get_r_channel_offset(),
                        ) = r as u8;
                        if destination_channels.has_alpha() {
                            *bgra.get_unchecked_mut(
                                rgb_offset + destination_channels.get_a_channel_offset(),
                            ) = 255;
                        }
                    }
                }
            }

            ux += 2;
        }

        match chroma_subsampling {
            YuvChromaSample::YUV420 => {
                if y & 1 == 1 {
                    uv_offset += uv_stride as usize;
                }
            }
            YuvChromaSample::YUV422 | YuvChromaSample::YUV444 => {
                uv_offset += uv_stride as usize;
            }
        }

        dst_offset += bgra_stride as usize;
        y_offset += y_stride as usize;
    }
}

/// Convert YUV NV12 format with 10-bit pixel format to BGRA format.
///
/// This function takes YUV NV12 data with 10-bit precision
/// and converts it to BGRA format with 8-bit precision.
///
/// # Arguments
///
/// * `y_plane` -  A slice containing Y (luminance) with 10 bit depth.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `uv_plane` - A slice to load the UV (chrominance) with 10 bit depth.
/// * `uv_stride` - The stride (bytes per row) for the UV plane.
/// * `bgra` - A mutable slice to store the converted BGRA data.
/// * `bgra_stride` - The stride (bytes per row) for the BGRA image data.
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
pub fn yuv_nv12_p10_to_bgra(
    y_plane: &[u16],
    y_stride: u32,
    uv_plane: &[u16],
    uv_stride: u32,
    bgra: &mut [u8],
    bgra_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    endianness: YuvEndiannes,
    bytes_packing: YuvBytesPacking,
) {
    let dispatcher = match endianness {
        YuvEndiannes::BigEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Bgra as u8 },
                    { YuvNVOrder::UV as u8 },
                    { YuvChromaSample::YUV420 as u8 },
                    { YuvEndiannes::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Bgra as u8 },
                    { YuvNVOrder::UV as u8 },
                    { YuvChromaSample::YUV420 as u8 },
                    { YuvEndiannes::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
        },
        YuvEndiannes::LittleEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Bgra as u8 },
                    { YuvNVOrder::UV as u8 },
                    { YuvChromaSample::YUV420 as u8 },
                    { YuvEndiannes::LittleEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Bgra as u8 },
                    { YuvNVOrder::UV as u8 },
                    { YuvChromaSample::YUV420 as u8 },
                    { YuvEndiannes::LittleEndian as u8 },
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
        width,
        height,
        range,
        matrix,
    );
}

/// Convert YUV NV12 format with 10-bit pixel format to RGBA format.
///
/// This function takes YUV NV12 data with 10-bit precision
/// and converts it to RGBA format with 8-bit precision.
///
/// # Arguments
///
/// * `y_plane` -  A slice containing Y (luminance) with 10 bit depth.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `uv_plane` - A slice to load the UV (chrominance) with 10 bit depth.
/// * `uv_stride` - The stride (bytes per row) for the UV plane.
/// * `rgba` - A mutable slice to store the converted RGBA data.
/// * `rgba_stride` - The stride (bytes per row) for the RGBA image data.
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
pub fn yuv_nv12_p10_to_rgba(
    y_plane: &[u16],
    y_stride: u32,
    uv_plane: &[u16],
    uv_stride: u32,
    rgba: &mut [u8],
    rgba_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    endianness: YuvEndiannes,
    bytes_packing: YuvBytesPacking,
) {
    let dispatcher = match endianness {
        YuvEndiannes::BigEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Rgba as u8 },
                    { YuvNVOrder::UV as u8 },
                    { YuvChromaSample::YUV420 as u8 },
                    { YuvEndiannes::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Rgba as u8 },
                    { YuvNVOrder::UV as u8 },
                    { YuvChromaSample::YUV420 as u8 },
                    { YuvEndiannes::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
        },
        YuvEndiannes::LittleEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Rgba as u8 },
                    { YuvNVOrder::UV as u8 },
                    { YuvChromaSample::YUV420 as u8 },
                    { YuvEndiannes::LittleEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Rgba as u8 },
                    { YuvNVOrder::UV as u8 },
                    { YuvChromaSample::YUV420 as u8 },
                    { YuvEndiannes::LittleEndian as u8 },
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
        width,
        height,
        range,
        matrix,
    );
}

/// Convert YUV NV12 format with 10-bit pixel format to BGR format.
///
/// This function takes YUV NV12 data with 10-bit precision
/// and converts it to BGR format with 8-bit precision.
///
/// # Arguments
///
/// * `y_plane` -  A slice containing Y (luminance) with 10 bit depth.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `uv_plane` - A slice to load the UV (chrominance) with 10 bit depth.
/// * `uv_stride` - The stride (bytes per row) for the UV plane.
/// * `bgr` - A mutable slice to store the converted BGR data.
/// * `bgr_stride` - The stride (bytes per row) for the BGR image data.
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
pub fn yuv_nv12_p10_to_bgr(
    y_plane: &[u16],
    y_stride: u32,
    uv_plane: &[u16],
    uv_stride: u32,
    bgr: &mut [u8],
    bgr_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    endianness: YuvEndiannes,
    bytes_packing: YuvBytesPacking,
) {
    let dispatcher = match endianness {
        YuvEndiannes::BigEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Bgr as u8 },
                    { YuvNVOrder::UV as u8 },
                    { YuvChromaSample::YUV420 as u8 },
                    { YuvEndiannes::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Bgr as u8 },
                    { YuvNVOrder::UV as u8 },
                    { YuvChromaSample::YUV420 as u8 },
                    { YuvEndiannes::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
        },
        YuvEndiannes::LittleEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Bgr as u8 },
                    { YuvNVOrder::UV as u8 },
                    { YuvChromaSample::YUV420 as u8 },
                    { YuvEndiannes::LittleEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Bgr as u8 },
                    { YuvNVOrder::UV as u8 },
                    { YuvChromaSample::YUV420 as u8 },
                    { YuvEndiannes::LittleEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
    };
    dispatcher(
        y_plane, y_stride, uv_plane, uv_stride, bgr, bgr_stride, width, height, range, matrix,
    );
}

/// Convert YUV NV12 format with 10-bit pixel format to RGB format.
///
/// This function takes YUV NV12 data with 10-bit precision
/// and converts it to RGB format with 8-bit precision.
///
/// # Arguments
///
/// * `y_plane` -  A slice containing Y (luminance) with 10 bit depth.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `uv_plane` - A slice to load the UV (chrominance) with 10 bit depth.
/// * `uv_stride` - The stride (bytes per row) for the UV plane.
/// * `rgb` - A mutable slice to store the converted RGB data.
/// * `rgb_stride` - The stride (bytes per row) for the RGB image data.
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
pub fn yuv_nv12_p10_to_rgb(
    y_plane: &[u16],
    y_stride: u32,
    uv_plane: &[u16],
    uv_stride: u32,
    rgb: &mut [u8],
    rgb_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    endianness: YuvEndiannes,
    bytes_packing: YuvBytesPacking,
) {
    let dispatcher = match endianness {
        YuvEndiannes::BigEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Rgb as u8 },
                    { YuvNVOrder::UV as u8 },
                    { YuvChromaSample::YUV420 as u8 },
                    { YuvEndiannes::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Rgb as u8 },
                    { YuvNVOrder::UV as u8 },
                    { YuvChromaSample::YUV420 as u8 },
                    { YuvEndiannes::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
        },
        YuvEndiannes::LittleEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Rgb as u8 },
                    { YuvNVOrder::UV as u8 },
                    { YuvChromaSample::YUV420 as u8 },
                    { YuvEndiannes::LittleEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Rgb as u8 },
                    { YuvNVOrder::UV as u8 },
                    { YuvChromaSample::YUV420 as u8 },
                    { YuvEndiannes::LittleEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
    };
    dispatcher(
        y_plane, y_stride, uv_plane, uv_stride, rgb, rgb_stride, width, height, range, matrix,
    );
}

/// Convert YUV NV16 format with 10-bit pixel format to BGRA format.
///
/// This function takes YUV NV16 data with 10-bit precision.
/// and converts it to BGRA format with 8-bit precision.
///
/// # Arguments
///
/// * `y_plane` -  A slice containing Y (luminance) with 10 bit depth.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `uv_plane` - A slice to load the UV (chrominance) with 10 bit depth.
/// * `uv_stride` - The stride (bytes per row) for the UV plane.
/// * `bgra` - A mutable slice to store the converted BGRA data.
/// * `bgra_stride` - The stride (bytes per row) for the BGRA image data.
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
pub fn yuv_nv16_p10_to_bgra(
    y_plane: &[u16],
    y_stride: u32,
    uv_plane: &[u16],
    uv_stride: u32,
    bgra: &mut [u8],
    bgra_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    endianness: YuvEndiannes,
    bytes_packing: YuvBytesPacking,
) {
    let dispatcher = match endianness {
        YuvEndiannes::BigEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Bgra as u8 },
                    { YuvNVOrder::UV as u8 },
                    { YuvChromaSample::YUV422 as u8 },
                    { YuvEndiannes::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Bgra as u8 },
                    { YuvNVOrder::UV as u8 },
                    { YuvChromaSample::YUV422 as u8 },
                    { YuvEndiannes::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
        },
        YuvEndiannes::LittleEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Bgra as u8 },
                    { YuvNVOrder::UV as u8 },
                    { YuvChromaSample::YUV422 as u8 },
                    { YuvEndiannes::LittleEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Bgra as u8 },
                    { YuvNVOrder::UV as u8 },
                    { YuvChromaSample::YUV422 as u8 },
                    { YuvEndiannes::LittleEndian as u8 },
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
        width,
        height,
        range,
        matrix,
    );
}

/// Convert YUV NV61 format with 10-bit pixel format to BGRA format.
///
/// This function takes YUV NV61 data with 10-bit precision.
/// and converts it to BGRA format with 8-bit precision.
///
/// # Arguments
///
/// * `y_plane` -  A slice containing Y (luminance) with 10 bit depth.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `uv_plane` - A slice to load the UV (chrominance) with 10 bit depth.
/// * `uv_stride` - The stride (bytes per row) for the UV plane.
/// * `bgra` - A mutable slice to store the converted BGRA data.
/// * `bgra_stride` - The stride (bytes per row) for the BGRA image data.
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
pub fn yuv_nv61_p10_to_bgra(
    y_plane: &[u16],
    y_stride: u32,
    uv_plane: &[u16],
    uv_stride: u32,
    bgra: &mut [u8],
    bgra_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    endianness: YuvEndiannes,
    bytes_packing: YuvBytesPacking,
) {
    let dispatcher = match endianness {
        YuvEndiannes::BigEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Bgra as u8 },
                    { YuvNVOrder::VU as u8 },
                    { YuvChromaSample::YUV422 as u8 },
                    { YuvEndiannes::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Bgra as u8 },
                    { YuvNVOrder::VU as u8 },
                    { YuvChromaSample::YUV422 as u8 },
                    { YuvEndiannes::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
        },
        YuvEndiannes::LittleEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Bgra as u8 },
                    { YuvNVOrder::VU as u8 },
                    { YuvChromaSample::YUV422 as u8 },
                    { YuvEndiannes::LittleEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Bgra as u8 },
                    { YuvNVOrder::VU as u8 },
                    { YuvChromaSample::YUV422 as u8 },
                    { YuvEndiannes::LittleEndian as u8 },
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
        width,
        height,
        range,
        matrix,
    );
}

/// Convert YUV NV16 format with 10-bit pixel format to BGR format.
///
/// This function takes YUV NV16 data with 10-bit precision.
/// and converts it to BGR format with 8-bit precision.
///
/// # Arguments
///
/// * `y_plane` -  A slice containing Y (luminance) with 10 bit depth.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `uv_plane` - A slice to load the UV (chrominance) with 10 bit depth.
/// * `uv_stride` - The stride (bytes per row) for the UV plane.
/// * `bgra` - A mutable slice to store the converted BGR data.
/// * `bgra_stride` - The stride (bytes per row) for the BGR image data.
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
pub fn yuv_nv16_p10_to_bgr(
    y_plane: &[u16],
    y_stride: u32,
    uv_plane: &[u16],
    uv_stride: u32,
    bgra: &mut [u8],
    bgra_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    endianness: YuvEndiannes,
    bytes_packing: YuvBytesPacking,
) {
    let dispatcher = match endianness {
        YuvEndiannes::BigEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Bgr as u8 },
                    { YuvNVOrder::UV as u8 },
                    { YuvChromaSample::YUV422 as u8 },
                    { YuvEndiannes::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Bgr as u8 },
                    { YuvNVOrder::UV as u8 },
                    { YuvChromaSample::YUV422 as u8 },
                    { YuvEndiannes::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
        },
        YuvEndiannes::LittleEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Bgr as u8 },
                    { YuvNVOrder::UV as u8 },
                    { YuvChromaSample::YUV422 as u8 },
                    { YuvEndiannes::LittleEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Bgr as u8 },
                    { YuvNVOrder::UV as u8 },
                    { YuvChromaSample::YUV422 as u8 },
                    { YuvEndiannes::LittleEndian as u8 },
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
        width,
        height,
        range,
        matrix,
    );
}

/// Convert YUV NV61 format with 10-bit pixel format to BGR format.
///
/// This function takes YUV NV61 data with 10-bit precision.
/// and converts it to BGR format with 8-bit precision.
///
/// # Arguments
///
/// * `y_plane` -  A slice containing Y (luminance) with 10 bit depth.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `uv_plane` - A slice to load the UV (chrominance) with 10 bit depth.
/// * `uv_stride` - The stride (bytes per row) for the UV plane.
/// * `bgra` - A mutable slice to store the converted BGR data.
/// * `bgra_stride` - The stride (bytes per row) for the BGR image data.
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
pub fn yuv_nv61_p10_to_bgr(
    y_plane: &[u16],
    y_stride: u32,
    uv_plane: &[u16],
    uv_stride: u32,
    bgra: &mut [u8],
    bgra_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    endianness: YuvEndiannes,
    bytes_packing: YuvBytesPacking,
) {
    let dispatcher = match endianness {
        YuvEndiannes::BigEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Bgr as u8 },
                    { YuvNVOrder::VU as u8 },
                    { YuvChromaSample::YUV422 as u8 },
                    { YuvEndiannes::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Bgr as u8 },
                    { YuvNVOrder::VU as u8 },
                    { YuvChromaSample::YUV422 as u8 },
                    { YuvEndiannes::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
        },
        YuvEndiannes::LittleEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Bgr as u8 },
                    { YuvNVOrder::VU as u8 },
                    { YuvChromaSample::YUV422 as u8 },
                    { YuvEndiannes::LittleEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Bgr as u8 },
                    { YuvNVOrder::VU as u8 },
                    { YuvChromaSample::YUV422 as u8 },
                    { YuvEndiannes::LittleEndian as u8 },
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
        width,
        height,
        range,
        matrix,
    );
}

/// Convert YUV NV16 format with 10-bit pixel format to RGB format.
///
/// This function takes YUV NV16 data with 10-bit precision.
/// and converts it to RGB format with 8-bit precision.
///
/// # Arguments
///
/// * `y_plane` -  A slice containing Y (luminance) with 10 bit depth.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `uv_plane` - A slice to load the UV (chrominance) with 10 bit depth.
/// * `uv_stride` - The stride (bytes per row) for the UV plane.
/// * `bgra` - A mutable slice to store the converted RGB data.
/// * `bgra_stride` - The stride (bytes per row) for the RGB image data.
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
pub fn yuv_nv16_p10_to_rgb(
    y_plane: &[u16],
    y_stride: u32,
    uv_plane: &[u16],
    uv_stride: u32,
    bgra: &mut [u8],
    bgra_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    endianness: YuvEndiannes,
    bytes_packing: YuvBytesPacking,
) {
    let dispatcher = match endianness {
        YuvEndiannes::BigEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Rgb as u8 },
                    { YuvNVOrder::UV as u8 },
                    { YuvChromaSample::YUV422 as u8 },
                    { YuvEndiannes::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Rgb as u8 },
                    { YuvNVOrder::UV as u8 },
                    { YuvChromaSample::YUV422 as u8 },
                    { YuvEndiannes::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
        },
        YuvEndiannes::LittleEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Rgb as u8 },
                    { YuvNVOrder::UV as u8 },
                    { YuvChromaSample::YUV422 as u8 },
                    { YuvEndiannes::LittleEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Rgb as u8 },
                    { YuvNVOrder::UV as u8 },
                    { YuvChromaSample::YUV422 as u8 },
                    { YuvEndiannes::LittleEndian as u8 },
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
        width,
        height,
        range,
        matrix,
    );
}

/// Convert YUV NV61 format with 10-bit pixel format to RGB format.
///
/// This function takes YUV NV61 data with 10-bit precision.
/// and converts it to RGB format with 8-bit precision.
///
/// # Arguments
///
/// * `y_plane` -  A slice containing Y (luminance) with 10 bit depth.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `uv_plane` - A slice to load the UV (chrominance) with 10 bit depth.
/// * `uv_stride` - The stride (bytes per row) for the UV plane.
/// * `bgra` - A mutable slice to store the converted RGB data.
/// * `bgra_stride` - The stride (bytes per row) for the RGB image data.
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
pub fn yuv_nv61_p10_to_rgb(
    y_plane: &[u16],
    y_stride: u32,
    uv_plane: &[u16],
    uv_stride: u32,
    bgra: &mut [u8],
    bgra_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    endianness: YuvEndiannes,
    bytes_packing: YuvBytesPacking,
) {
    let dispatcher = match endianness {
        YuvEndiannes::BigEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Rgb as u8 },
                    { YuvNVOrder::VU as u8 },
                    { YuvChromaSample::YUV422 as u8 },
                    { YuvEndiannes::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Rgb as u8 },
                    { YuvNVOrder::VU as u8 },
                    { YuvChromaSample::YUV422 as u8 },
                    { YuvEndiannes::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
        },
        YuvEndiannes::LittleEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Rgb as u8 },
                    { YuvNVOrder::VU as u8 },
                    { YuvChromaSample::YUV422 as u8 },
                    { YuvEndiannes::LittleEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Rgb as u8 },
                    { YuvNVOrder::VU as u8 },
                    { YuvChromaSample::YUV422 as u8 },
                    { YuvEndiannes::LittleEndian as u8 },
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
        width,
        height,
        range,
        matrix,
    );
}

/// Convert YUV NV16 format with 10-bit pixel format to RGB format.
///
/// This function takes YUV NV16 data with 10-bit precision.
/// and converts it to RGBA format with 8-bit precision.
///
/// # Arguments
///
/// * `y_plane` -  A slice containing Y (luminance) with 10 bit depth.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `uv_plane` - A slice to load the UV (chrominance) with 10 bit depth.
/// * `uv_stride` - The stride (bytes per row) for the UV plane.
/// * `bgra` - A mutable slice to store the converted RGBA data.
/// * `bgra_stride` - The stride (bytes per row) for the RGBA image data.
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
pub fn yuv_nv16_p10_to_rgba(
    y_plane: &[u16],
    y_stride: u32,
    uv_plane: &[u16],
    uv_stride: u32,
    bgra: &mut [u8],
    bgra_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    endianness: YuvEndiannes,
    bytes_packing: YuvBytesPacking,
) {
    let dispatcher = match endianness {
        YuvEndiannes::BigEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Rgba as u8 },
                    { YuvNVOrder::UV as u8 },
                    { YuvChromaSample::YUV422 as u8 },
                    { YuvEndiannes::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Rgba as u8 },
                    { YuvNVOrder::UV as u8 },
                    { YuvChromaSample::YUV422 as u8 },
                    { YuvEndiannes::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
        },
        YuvEndiannes::LittleEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Rgba as u8 },
                    { YuvNVOrder::UV as u8 },
                    { YuvChromaSample::YUV422 as u8 },
                    { YuvEndiannes::LittleEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Rgba as u8 },
                    { YuvNVOrder::UV as u8 },
                    { YuvChromaSample::YUV422 as u8 },
                    { YuvEndiannes::LittleEndian as u8 },
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
        width,
        height,
        range,
        matrix,
    );
}

/// Convert YUV NV61 format with 10-bit pixel format to RGB format.
///
/// This function takes YUV NV61 data with 10-bit precision.
/// and converts it to RGBA format with 8-bit precision.
///
/// # Arguments
///
/// * `y_plane` -  A slice containing Y (luminance) with 10 bit depth.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `uv_plane` - A slice to load the UV (chrominance) with 10 bit depth.
/// * `uv_stride` - The stride (bytes per row) for the UV plane.
/// * `bgra` - A mutable slice to store the converted RGBA data.
/// * `bgra_stride` - The stride (bytes per row) for the RGBA image data.
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
pub fn yuv_nv61_p10_to_rgba(
    y_plane: &[u16],
    y_stride: u32,
    uv_plane: &[u16],
    uv_stride: u32,
    bgra: &mut [u8],
    bgra_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    endianness: YuvEndiannes,
    bytes_packing: YuvBytesPacking,
) {
    let dispatcher = match endianness {
        YuvEndiannes::BigEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Rgba as u8 },
                    { YuvNVOrder::VU as u8 },
                    { YuvChromaSample::YUV422 as u8 },
                    { YuvEndiannes::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Rgba as u8 },
                    { YuvNVOrder::VU as u8 },
                    { YuvChromaSample::YUV422 as u8 },
                    { YuvEndiannes::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
        },
        YuvEndiannes::LittleEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Rgba as u8 },
                    { YuvNVOrder::VU as u8 },
                    { YuvChromaSample::YUV422 as u8 },
                    { YuvEndiannes::LittleEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Rgba as u8 },
                    { YuvNVOrder::VU as u8 },
                    { YuvChromaSample::YUV422 as u8 },
                    { YuvEndiannes::LittleEndian as u8 },
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
        width,
        height,
        range,
        matrix,
    );
}

/// Convert YUV NV21 format with 10-bit pixel format to BGR format.
///
/// This function takes YUV NV21 data with 10-bit precision
/// and converts it to BGR format with 8-bit precision.
///
/// # Arguments
///
/// * `y_plane` -  A slice containing Y (luminance) with 10 bit depth.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `uv_plane` - A slice to load the UV (chrominance) with 10 bit depth.
/// * `uv_stride` - The stride (bytes per row) for the UV plane.
/// * `bgr` - A mutable slice to store the converted BGR data.
/// * `bgr_stride` - The stride (bytes per row) for the BGR image data.
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
pub fn yuv_nv21_p10_to_bgr(
    y_plane: &[u16],
    y_stride: u32,
    uv_plane: &[u16],
    uv_stride: u32,
    bgr: &mut [u8],
    bgr_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    endianness: YuvEndiannes,
    bytes_packing: YuvBytesPacking,
) {
    let dispatcher = match endianness {
        YuvEndiannes::BigEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Bgr as u8 },
                    { YuvNVOrder::VU as u8 },
                    { YuvChromaSample::YUV420 as u8 },
                    { YuvEndiannes::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Bgr as u8 },
                    { YuvNVOrder::VU as u8 },
                    { YuvChromaSample::YUV420 as u8 },
                    { YuvEndiannes::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
        },
        YuvEndiannes::LittleEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Bgr as u8 },
                    { YuvNVOrder::VU as u8 },
                    { YuvChromaSample::YUV420 as u8 },
                    { YuvEndiannes::LittleEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Bgr as u8 },
                    { YuvNVOrder::VU as u8 },
                    { YuvChromaSample::YUV420 as u8 },
                    { YuvEndiannes::LittleEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
    };
    dispatcher(
        y_plane, y_stride, uv_plane, uv_stride, bgr, bgr_stride, width, height, range, matrix,
    );
}

/// Convert YUV NV21 format with 10-bit pixel format to BGRA format.
///
/// This function takes YUV NV21 data with 10-bit precision
/// and converts it to BGRA format with 8-bit precision.
///
/// # Arguments
///
/// * `y_plane` -  A slice containing Y (luminance) with 10 bit depth.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `uv_plane` - A slice to load the UV (chrominance) with 10 bit depth.
/// * `uv_stride` - The stride (bytes per row) for the UV plane.
/// * `bgra` - A mutable slice to store the converted BGRA data.
/// * `bgra_stride` - The stride (bytes per row) for the BGRA image data.
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
pub fn yuv_nv21_p10_to_bgra(
    y_plane: &[u16],
    y_stride: u32,
    uv_plane: &[u16],
    uv_stride: u32,
    bgra: &mut [u8],
    bgra_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    endianness: YuvEndiannes,
    bytes_packing: YuvBytesPacking,
) {
    let dispatcher = match endianness {
        YuvEndiannes::BigEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Bgra as u8 },
                    { YuvNVOrder::VU as u8 },
                    { YuvChromaSample::YUV420 as u8 },
                    { YuvEndiannes::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Bgra as u8 },
                    { YuvNVOrder::VU as u8 },
                    { YuvChromaSample::YUV420 as u8 },
                    { YuvEndiannes::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
        },
        YuvEndiannes::LittleEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Bgra as u8 },
                    { YuvNVOrder::VU as u8 },
                    { YuvChromaSample::YUV420 as u8 },
                    { YuvEndiannes::LittleEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Bgra as u8 },
                    { YuvNVOrder::VU as u8 },
                    { YuvChromaSample::YUV420 as u8 },
                    { YuvEndiannes::LittleEndian as u8 },
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
        width,
        height,
        range,
        matrix,
    );
}

/// Convert YUV NV21 format with 10-bit pixel format to RGB format.
///
/// This function takes YUV NV21 data with 10-bit precision
/// and converts it to RGB format with 8-bit precision.
///
/// # Arguments
///
/// * `y_plane` -  A slice containing Y (luminance) with 10 bit depth.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `uv_plane` - A slice to load the UV (chrominance) with 10 bit depth.
/// * `uv_stride` - The stride (bytes per row) for the UV plane.
/// * `rgb` - A mutable slice to store the converted RGB data.
/// * `rgb_stride` - The stride (bytes per row) for the RGB image data.
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
pub fn yuv_nv21_p10_to_rgb(
    y_plane: &[u16],
    y_stride: u32,
    uv_plane: &[u16],
    uv_stride: u32,
    rgb: &mut [u8],
    rgb_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    endianness: YuvEndiannes,
    bytes_packing: YuvBytesPacking,
) {
    let dispatcher = match endianness {
        YuvEndiannes::BigEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Rgb as u8 },
                    { YuvNVOrder::VU as u8 },
                    { YuvChromaSample::YUV420 as u8 },
                    { YuvEndiannes::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Rgb as u8 },
                    { YuvNVOrder::VU as u8 },
                    { YuvChromaSample::YUV420 as u8 },
                    { YuvEndiannes::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
        },
        YuvEndiannes::LittleEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Rgb as u8 },
                    { YuvNVOrder::VU as u8 },
                    { YuvChromaSample::YUV420 as u8 },
                    { YuvEndiannes::LittleEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Rgb as u8 },
                    { YuvNVOrder::VU as u8 },
                    { YuvChromaSample::YUV420 as u8 },
                    { YuvEndiannes::LittleEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
    };
    dispatcher(
        y_plane, y_stride, uv_plane, uv_stride, rgb, rgb_stride, width, height, range, matrix,
    );
}

/// Convert YUV NV21 format with 10-bit pixel format to RGBA format.
///
/// This function takes YUV NV21 data with 10-bit precision
/// and converts it to RGBA format with 8-bit precision.
///
/// # Arguments
///
/// * `y_plane` -  A slice containing Y (luminance) with 10 bit depth.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `uv_plane` - A slice to load the UV (chrominance) with 10 bit depth.
/// * `uv_stride` - The stride (bytes per row) for the UV plane.
/// * `rgba` - A mutable slice to store the converted RGBA data.
/// * `rgba_stride` - The stride (bytes per row) for the RGBA image data.
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
pub fn yuv_nv21_p10_to_rgba(
    y_plane: &[u16],
    y_stride: u32,
    uv_plane: &[u16],
    uv_stride: u32,
    rgba: &mut [u8],
    rgba_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    endianness: YuvEndiannes,
    bytes_packing: YuvBytesPacking,
) {
    let dispatcher = match endianness {
        YuvEndiannes::BigEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Rgba as u8 },
                    { YuvNVOrder::VU as u8 },
                    { YuvChromaSample::YUV420 as u8 },
                    { YuvEndiannes::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Rgba as u8 },
                    { YuvNVOrder::VU as u8 },
                    { YuvChromaSample::YUV420 as u8 },
                    { YuvEndiannes::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
        },
        YuvEndiannes::LittleEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Rgba as u8 },
                    { YuvNVOrder::VU as u8 },
                    { YuvChromaSample::YUV420 as u8 },
                    { YuvEndiannes::LittleEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Rgba as u8 },
                    { YuvNVOrder::VU as u8 },
                    { YuvChromaSample::YUV420 as u8 },
                    { YuvEndiannes::LittleEndian as u8 },
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
        width,
        height,
        range,
        matrix,
    );
}
