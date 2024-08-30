/*
 * // Copyright (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use crate::neon::neon_yuv_p10_to_rgba_row;

use crate::yuv_support::{
    get_inverse_transform, get_kr_kb, get_yuv_range, YuvBytesPacking, YuvChromaSample,
    YuvEndiannes, YuvRange, YuvSourceChannels, YuvStandardMatrix,
};

fn yuv_p10_to_image_impl<
    const DESTINATION_CHANNELS: u8,
    const SAMPLING: u8,
    const ENDIANNESS: u8,
    const BYTES_POSITION: u8,
>(
    y_plane: &[u16],
    y_stride: u32,
    u_plane: &[u16],
    u_stride: u32,
    v_plane: &[u16],
    v_stride: u32,
    rgba: &mut [u8],
    rgba_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) {
    let dst_chans: YuvSourceChannels = DESTINATION_CHANNELS.into();
    let channels = dst_chans.get_channels_count();
    let chroma_subsampling: YuvChromaSample = SAMPLING.into();
    let endianness: YuvEndiannes = ENDIANNESS.into();
    let bytes_position: YuvBytesPacking = BYTES_POSITION.into();
    let range = get_yuv_range(10, range);
    let kr_kb = get_kr_kb(matrix);
    let max_range_p10 = (1u32 << 10u32) - 1;
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
    let mut u_offset = 0usize;
    let mut v_offset = 0usize;
    let mut dst_offset = 0usize;

    let y_src_ptr = y_plane.as_ptr() as *const u8;
    let u_src_ptr = u_plane.as_ptr() as *const u8;
    let v_src_ptr = v_plane.as_ptr() as *const u8;

    for y in 0..height as usize {
        let mut x = 0usize;
        let mut cx = 0usize;

        let y_ld_ptr = unsafe { y_src_ptr.offset(y_offset as isize) as *const u16 };
        let u_ld_ptr = unsafe { u_src_ptr.offset(u_offset as isize) as *const u16 };
        let v_ld_ptr = unsafe { v_src_ptr.offset(v_offset as isize) as *const u16 };

        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        unsafe {
            let offset = neon_yuv_p10_to_rgba_row::<
                DESTINATION_CHANNELS,
                SAMPLING,
                ENDIANNESS,
                BYTES_POSITION,
            >(
                y_ld_ptr,
                u_ld_ptr,
                v_ld_ptr,
                rgba,
                dst_offset,
                width,
                &range,
                &i_transform,
                x,
                cx,
            );
            x = offset.cx;
            cx = offset.ux;
        }

        while x < width as usize {
            let y_value: i32;
            let cb_value: i32;
            let cr_value: i32;
            match endianness {
                YuvEndiannes::BigEndian => {
                    let mut y_vl = u16::from_be(unsafe { y_ld_ptr.add(x).read_unaligned() }) as i32;
                    let mut cb_vl =
                        u16::from_be(unsafe { u_ld_ptr.add(cx).read_unaligned() }) as i32;
                    let mut cr_vl =
                        u16::from_be(unsafe { v_ld_ptr.add(cx).read_unaligned() }) as i32;
                    if bytes_position == YuvBytesPacking::MostSignificantBytes {
                        y_vl = y_vl >> 6;
                        cb_vl = cb_vl >> 6;
                        cr_vl = cr_vl >> 6;
                    }
                    y_value = (y_vl - bias_y) * y_coef;

                    cb_value = cb_vl - bias_uv;
                    cr_value = cr_vl - bias_uv;
                }
                YuvEndiannes::LittleEndian => {
                    let mut y_vl = u16::from_le(unsafe { y_ld_ptr.add(x).read_unaligned() }) as i32;
                    let mut cb_vl =
                        u16::from_le(unsafe { u_ld_ptr.add(cx).read_unaligned() }) as i32;
                    let mut cr_vl =
                        u16::from_le(unsafe { v_ld_ptr.add(cx).read_unaligned() }) as i32;
                    if bytes_position == YuvBytesPacking::MostSignificantBytes {
                        y_vl = y_vl >> 6;
                        cb_vl = cb_vl >> 6;
                        cr_vl = cr_vl >> 6;
                    }
                    y_value = (y_vl - bias_y) * y_coef;

                    cb_value = cb_vl - bias_uv;
                    cr_value = cr_vl - bias_uv;
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
                let dst_slice = rgba.get_unchecked_mut(rgb_offset..);
                *dst_slice.get_unchecked_mut(dst_chans.get_b_channel_offset()) = b as u8;
                *dst_slice.get_unchecked_mut(dst_chans.get_g_channel_offset()) = g as u8;
                *dst_slice.get_unchecked_mut(dst_chans.get_r_channel_offset()) = r as u8;
                if dst_chans.has_alpha() {
                    *dst_slice.get_unchecked_mut(dst_chans.get_a_channel_offset()) = 255;
                }
            }

            x += 1;

            if x + 1 < width as usize {
                let y_value: i32;
                match endianness {
                    YuvEndiannes::BigEndian => {
                        let mut y_vl =
                            u16::from_be(unsafe { y_ld_ptr.add(x).read_unaligned() }) as i32;
                        if bytes_position == YuvBytesPacking::MostSignificantBytes {
                            y_vl = y_vl >> 6;
                        }
                        y_value = (y_vl - bias_y) * y_coef;
                    }
                    YuvEndiannes::LittleEndian => {
                        let mut y_vl =
                            u16::from_le(unsafe { y_ld_ptr.add(x).read_unaligned() }) as i32;
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

                let px = x * channels;
                let rgb_offset = dst_offset + px;
                unsafe {
                    let dst_slice = rgba.get_unchecked_mut(rgb_offset..);
                    *dst_slice.get_unchecked_mut(dst_chans.get_b_channel_offset()) = b as u8;
                    *dst_slice.get_unchecked_mut(dst_chans.get_g_channel_offset()) = g as u8;
                    *dst_slice.get_unchecked_mut(dst_chans.get_r_channel_offset()) = r as u8;
                    if dst_chans.has_alpha() {
                        *dst_slice.get_unchecked_mut(dst_chans.get_a_channel_offset()) = 255;
                    }
                }
            }

            x += 1;
            cx += 1;
        }

        match chroma_subsampling {
            YuvChromaSample::YUV420 => {
                if y & 1 == 1 {
                    u_offset += u_stride as usize;
                    v_offset += v_stride as usize;
                }
            }
            YuvChromaSample::YUV422 | YuvChromaSample::YUV444 => {
                u_offset += u_stride as usize;
                v_offset += v_stride as usize;
            }
        }

        dst_offset += rgba_stride as usize;
        y_offset += y_stride as usize;
    }
}

/// Convert YUV 420 planar format with 10-bit pixel format to BGRA format.
///
/// This function takes YUV 420 planar data with 10-bit precision.
/// and converts it to BGRA format with 8-bit precision per channel
///
/// # Arguments
///
/// * `y_plane` -  A slice containing Y (luminance) with 10 bit depth.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `u_plane` - A slice to load the U (chrominance) with 10 bit depth.
/// * `u_stride` - The stride (bytes per row) for the U plane.
/// * `v_plane` - A slice to load the V (chrominance) with 10 bit depth.
/// * `v_stride` - The stride (bytes per row) for the U plane.
/// * `width` - The width of the YUV image.
/// * `height` - The height of the YUV image.
/// * `bgra` - A mutable slice to store the converted BGRA data.
/// * `bgra_stride` - The stride (bytes per row) for BGRA data.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
/// * `endianness` - The endianness of stored bytes
/// * `bytes_packing` - position of significant bytes ( most significant or least significant ) if it in most significant it should be stated as per Apple *kCVPixelFormatType_422YpCbCr10BiPlanarFullRange/kCVPixelFormatType_422YpCbCr10BiPlanarVideoRange*
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGRA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv420_p10_to_bgra(
    y_plane: &[u16],
    y_stride: u32,
    u_plane: &[u16],
    u_stride: u32,
    v_plane: &[u16],
    v_stride: u32,
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
                yuv_p10_to_image_impl::<
                    { YuvSourceChannels::Bgra as u8 },
                    { YuvChromaSample::YUV420 as u8 },
                    { YuvEndiannes::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_p10_to_image_impl::<
                    { YuvSourceChannels::Bgra as u8 },
                    { YuvChromaSample::YUV420 as u8 },
                    { YuvEndiannes::BigEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
        YuvEndiannes::LittleEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_p10_to_image_impl::<
                    { YuvSourceChannels::Bgra as u8 },
                    { YuvChromaSample::YUV420 as u8 },
                    { YuvEndiannes::LittleEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_p10_to_image_impl::<
                    { YuvSourceChannels::Bgra as u8 },
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
        u_plane,
        u_stride,
        v_plane,
        v_stride,
        bgra,
        bgra_stride,
        width,
        height,
        range,
        matrix,
    );
}

/// Convert YUV 420 planar format with 10-bit pixel format to BGRA format.
///
/// This function takes YUV 420 planar data with 10-bit precision.
/// and converts it to BGR format with 8-bit precision per channel
///
/// # Arguments
///
/// * `y_plane` -  A slice containing Y (luminance) with 10 bit depth.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `u_plane` - A slice to load the U (chrominance) with 10 bit depth.
/// * `u_stride` - The stride (bytes per row) for the U plane.
/// * `v_plane` - A slice to load the V (chrominance) with 10 bit depth.
/// * `v_stride` - The stride (bytes per row) for the U plane.
/// * `width` - The width of the YUV image.
/// * `height` - The height of the YUV image.
/// * `bgra` - A mutable slice to store the converted BGR data.
/// * `bgra_stride` - The stride (bytes per row) for BGR data.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
/// * `endianness` - The endianness of stored bytes
/// * `bytes_packing` - position of significant bytes ( most significant or least significant ) if it in most significant it should be stated as per Apple *kCVPixelFormatType_422YpCbCr10BiPlanarFullRange/kCVPixelFormatType_422YpCbCr10BiPlanarVideoRange*
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGR data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv420_p10_to_bgr(
    y_plane: &[u16],
    y_stride: u32,
    u_plane: &[u16],
    u_stride: u32,
    v_plane: &[u16],
    v_stride: u32,
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
                yuv_p10_to_image_impl::<
                    { YuvSourceChannels::Bgr as u8 },
                    { YuvChromaSample::YUV420 as u8 },
                    { YuvEndiannes::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_p10_to_image_impl::<
                    { YuvSourceChannels::Bgr as u8 },
                    { YuvChromaSample::YUV420 as u8 },
                    { YuvEndiannes::BigEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
        YuvEndiannes::LittleEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_p10_to_image_impl::<
                    { YuvSourceChannels::Bgr as u8 },
                    { YuvChromaSample::YUV420 as u8 },
                    { YuvEndiannes::LittleEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_p10_to_image_impl::<
                    { YuvSourceChannels::Bgr as u8 },
                    { YuvChromaSample::YUV420 as u8 },
                    { YuvEndiannes::LittleEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
    };
    dispatcher(
        y_plane, y_stride, u_plane, u_stride, v_plane, v_stride, bgr, bgr_stride, width, height,
        range, matrix,
    );
}

/// Convert YUV 422 format with 10-bit pixel format to BGRA format .
///
/// This function takes YUV 422 data with 10-bit precision.
/// and converts it to BGRA format with 8-bit precision per channel.
///
/// # Arguments
///
/// * `y_plane` -  A slice containing Y (luminance) with 10 bit depth.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `u_plane` - A slice to load the U (chrominance) with 10 bit depth.
/// * `u_stride` - The stride (bytes per row) for the U plane.
/// * `v_plane` - A slice to load the V (chrominance) with 10 bit depth.
/// * `v_stride` - The stride (bytes per row) for the U plane
/// * `width` - The width of the YUV image.
/// * `height` - The height of the YUV image.
/// * `bgra` - A mutable slice to store the converted BGRA data.
/// * `bgra_stride` - The stride (bytes per row) for BGRA data.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
/// * `endianness` - The endianness of stored bytes
/// * `bytes_packing` - position of significant bytes ( most significant or least significant ) if it in most significant it should be stated as per Apple *kCVPixelFormatType_422YpCbCr10BiPlanarFullRange/kCVPixelFormatType_422YpCbCr10BiPlanarVideoRange*
///
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGRA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv422_p10_to_bgra(
    y_plane: &[u16],
    y_stride: u32,
    u_plane: &[u16],
    u_stride: u32,
    v_plane: &[u16],
    v_stride: u32,
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
                yuv_p10_to_image_impl::<
                    { YuvSourceChannels::Bgra as u8 },
                    { YuvChromaSample::YUV422 as u8 },
                    { YuvEndiannes::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_p10_to_image_impl::<
                    { YuvSourceChannels::Bgra as u8 },
                    { YuvChromaSample::YUV422 as u8 },
                    { YuvEndiannes::BigEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
        YuvEndiannes::LittleEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_p10_to_image_impl::<
                    { YuvSourceChannels::Bgra as u8 },
                    { YuvChromaSample::YUV422 as u8 },
                    { YuvEndiannes::LittleEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_p10_to_image_impl::<
                    { YuvSourceChannels::Bgra as u8 },
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
        u_plane,
        u_stride,
        v_plane,
        v_stride,
        bgra,
        bgra_stride,
        width,
        height,
        range,
        matrix,
    );
}

/// Convert YUV 422 format with 10-bit pixel format to BGR format.
///
/// This function takes YUV 422 data with 10-bit precision.
/// and converts it to BGR format with 8-bit precision per channel.
///
/// # Arguments
///
/// * `y_plane` -  A slice containing Y (luminance) with 10 bit depth.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `u_plane` - A slice to load the U (chrominance) with 10 bit depth.
/// * `u_stride` - The stride (bytes per row) for the U plane.
/// * `v_plane` - A slice to load the V (chrominance) with 10 bit depth.
/// * `v_stride` - The stride (bytes per row) for the U plane
/// * `width` - The width of the YUV image.
/// * `height` - The height of the YUV image.
/// * `bgr` - A mutable slice to store the converted BGR data.
/// * `bgr_stride` - The stride (bytes per row) for BGR data.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
/// * `endianness` - The endianness of stored bytes
/// * `bytes_packing` - position of significant bytes ( most significant or least significant ) if it in most significant it should be stated as per Apple *kCVPixelFormatType_422YpCbCr10BiPlanarFullRange/kCVPixelFormatType_422YpCbCr10BiPlanarVideoRange*
///
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGR data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv422_p10_to_bgr(
    y_plane: &[u16],
    y_stride: u32,
    u_plane: &[u16],
    u_stride: u32,
    v_plane: &[u16],
    v_stride: u32,
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
                yuv_p10_to_image_impl::<
                    { YuvSourceChannels::Bgr as u8 },
                    { YuvChromaSample::YUV422 as u8 },
                    { YuvEndiannes::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_p10_to_image_impl::<
                    { YuvSourceChannels::Bgr as u8 },
                    { YuvChromaSample::YUV422 as u8 },
                    { YuvEndiannes::BigEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
        YuvEndiannes::LittleEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_p10_to_image_impl::<
                    { YuvSourceChannels::Bgr as u8 },
                    { YuvChromaSample::YUV422 as u8 },
                    { YuvEndiannes::LittleEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_p10_to_image_impl::<
                    { YuvSourceChannels::Bgr as u8 },
                    { YuvChromaSample::YUV422 as u8 },
                    { YuvEndiannes::LittleEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
    };
    dispatcher(
        y_plane, y_stride, u_plane, u_stride, v_plane, v_stride, bgr, bgr_stride, width, height,
        range, matrix,
    );
}

/// Convert YUV 420 planar format with 10-bit pixel format to RGBA format.
///
/// This function takes YUV 420 planar data with 10-bit precision.
/// and converts it to RGBA format with 8-bit precision per channel
///
/// # Arguments
///
/// * `y_plane` -  A slice containing Y (luminance) with 10 bit depth.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `u_plane` - A slice to load the U (chrominance) with 10 bit depth.
/// * `u_stride` - The stride (bytes per row) for the U plane.
/// * `v_plane` - A slice to load the V (chrominance) with 10 bit depth.
/// * `v_stride` - The stride (bytes per row) for the U plane.
/// * `width` - The width of the YUV image.
/// * `height` - The height of the YUV image.
/// * `rgba` - A mutable slice to store the converted RGBA data.
/// * `rgba_stride` - The stride (bytes per row) for RGBA data.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
/// * `endianness` - The endianness of stored bytes
/// * `bytes_packing` - position of significant bytes ( most significant or least significant ) if it in most significant it should be stated as per Apple *kCVPixelFormatType_422YpCbCr10BiPlanarFullRange/kCVPixelFormatType_422YpCbCr10BiPlanarVideoRange*
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input RGBA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv420_p10_to_rgba(
    y_plane: &[u16],
    y_stride: u32,
    u_plane: &[u16],
    u_stride: u32,
    v_plane: &[u16],
    v_stride: u32,
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
                yuv_p10_to_image_impl::<
                    { YuvSourceChannels::Rgba as u8 },
                    { YuvChromaSample::YUV420 as u8 },
                    { YuvEndiannes::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_p10_to_image_impl::<
                    { YuvSourceChannels::Rgba as u8 },
                    { YuvChromaSample::YUV420 as u8 },
                    { YuvEndiannes::BigEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
        YuvEndiannes::LittleEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_p10_to_image_impl::<
                    { YuvSourceChannels::Rgba as u8 },
                    { YuvChromaSample::YUV420 as u8 },
                    { YuvEndiannes::LittleEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_p10_to_image_impl::<
                    { YuvSourceChannels::Rgba as u8 },
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
        u_plane,
        u_stride,
        v_plane,
        v_stride,
        rgba,
        rgba_stride,
        width,
        height,
        range,
        matrix,
    );
}

/// Convert YUV 420 planar format with 10-bit pixel format to RGB format.
///
/// This function takes YUV 420 planar data with 10-bit precision.
/// and converts it to RGB format with 8-bit precision per channel
///
/// # Arguments
///
/// * `y_plane` -  A slice containing Y (luminance) with 10 bit depth.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `u_plane` - A slice to load the U (chrominance) with 10 bit depth.
/// * `u_stride` - The stride (bytes per row) for the U plane.
/// * `v_plane` - A slice to load the V (chrominance) with 10 bit depth.
/// * `v_stride` - The stride (bytes per row) for the U plane.
/// * `width` - The width of the YUV image.
/// * `height` - The height of the YUV image.
/// * `rgb` - A mutable slice to store the converted RGB data.
/// * `rgb_stride` - The stride (bytes per row) for RGB data.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
/// * `endianness` - The endianness of stored bytes
/// * `bytes_packing` - position of significant bytes ( most significant or least significant ) if it in most significant it should be stated as per Apple *kCVPixelFormatType_422YpCbCr10BiPlanarFullRange/kCVPixelFormatType_422YpCbCr10BiPlanarVideoRange*
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input RGB data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv420_p10_to_rgb(
    y_plane: &[u16],
    y_stride: u32,
    u_plane: &[u16],
    u_stride: u32,
    v_plane: &[u16],
    v_stride: u32,
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
                yuv_p10_to_image_impl::<
                    { YuvSourceChannels::Rgb as u8 },
                    { YuvChromaSample::YUV420 as u8 },
                    { YuvEndiannes::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_p10_to_image_impl::<
                    { YuvSourceChannels::Rgb as u8 },
                    { YuvChromaSample::YUV420 as u8 },
                    { YuvEndiannes::BigEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
        YuvEndiannes::LittleEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_p10_to_image_impl::<
                    { YuvSourceChannels::Rgb as u8 },
                    { YuvChromaSample::YUV420 as u8 },
                    { YuvEndiannes::LittleEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_p10_to_image_impl::<
                    { YuvSourceChannels::Rgb as u8 },
                    { YuvChromaSample::YUV420 as u8 },
                    { YuvEndiannes::LittleEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
    };
    dispatcher(
        y_plane, y_stride, u_plane, u_stride, v_plane, v_stride, rgb, rgb_stride, width, height,
        range, matrix,
    );
}

/// Convert YUV 422 format with 10-bit pixel format to RGBA format.
///
/// This function takes YUV 422 data with 10-bit precision stored.
/// and converts it to RGBA format with 8-bit precision per channel.
///
/// # Arguments
///
/// * `y_plane` -  A slice containing Y (luminance) with 10 bit depth.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `u_plane` - A slice to load the U (chrominance) with 10 bit depth.
/// * `u_stride` - The stride (bytes per row) for the U plane.
/// * `v_plane` - A slice to load the V (chrominance) with 10 bit depth.
/// * `v_stride` - The stride (bytes per row) for the U plane
/// * `width` - The width of the YUV image.
/// * `height` - The height of the YUV image.
/// * `rgba_data` - A mutable slice to store the converted RGBA data.
/// * `rgba_stride` - The stride (bytes per row) for RGBA data.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
/// * `endianness` - The endianness of stored bytes
/// * `bytes_packing` - position of significant bytes ( most significant or least significant ) if it in most significant it should be stated as per Apple *kCVPixelFormatType_422YpCbCr10BiPlanarFullRange/kCVPixelFormatType_422YpCbCr10BiPlanarVideoRange*
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input RGBA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv422_p10_to_rgba(
    y_plane: &[u16],
    y_stride: u32,
    u_plane: &[u16],
    u_stride: u32,
    v_plane: &[u16],
    v_stride: u32,
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
                yuv_p10_to_image_impl::<
                    { YuvSourceChannels::Rgba as u8 },
                    { YuvChromaSample::YUV422 as u8 },
                    { YuvEndiannes::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_p10_to_image_impl::<
                    { YuvSourceChannels::Rgba as u8 },
                    { YuvChromaSample::YUV422 as u8 },
                    { YuvEndiannes::BigEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
        YuvEndiannes::LittleEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_p10_to_image_impl::<
                    { YuvSourceChannels::Rgba as u8 },
                    { YuvChromaSample::YUV422 as u8 },
                    { YuvEndiannes::LittleEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_p10_to_image_impl::<
                    { YuvSourceChannels::Rgba as u8 },
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
        u_plane,
        u_stride,
        v_plane,
        v_stride,
        rgba,
        rgba_stride,
        width,
        height,
        range,
        matrix,
    );
}

/// Convert YUV 422 format with 10-bit pixel format to RGB format.
///
/// This function takes YUV 422 data with 10-bit precision stored.
/// and converts it to RGB format with 8-bit precision per channel.
///
/// # Arguments
///
/// * `y_plane` -  A slice containing Y (luminance) with 10 bit depth.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `u_plane` - A slice to load the U (chrominance) with 10 bit depth.
/// * `u_stride` - The stride (bytes per row) for the U plane.
/// * `v_plane` - A slice to load the V (chrominance) with 10 bit depth.
/// * `v_stride` - The stride (bytes per row) for the U plane
/// * `width` - The width of the YUV image.
/// * `height` - The height of the YUV image.
/// * `rgb_data` - A mutable slice to store the converted RGB data.
/// * `rgb_stride` - The stride (bytes per row) for RGB data.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
/// * `endianness` - The endianness of stored bytes
/// * `bytes_packing` - position of significant bytes ( most significant or least significant ) if it in most significant it should be stated as per Apple *kCVPixelFormatType_422YpCbCr10BiPlanarFullRange/kCVPixelFormatType_422YpCbCr10BiPlanarVideoRange*
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input RGB data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv422_p10_to_rgb(
    y_plane: &[u16],
    y_stride: u32,
    u_plane: &[u16],
    u_stride: u32,
    v_plane: &[u16],
    v_stride: u32,
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
                yuv_p10_to_image_impl::<
                    { YuvSourceChannels::Rgb as u8 },
                    { YuvChromaSample::YUV422 as u8 },
                    { YuvEndiannes::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_p10_to_image_impl::<
                    { YuvSourceChannels::Rgb as u8 },
                    { YuvChromaSample::YUV422 as u8 },
                    { YuvEndiannes::BigEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
        YuvEndiannes::LittleEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_p10_to_image_impl::<
                    { YuvSourceChannels::Rgb as u8 },
                    { YuvChromaSample::YUV422 as u8 },
                    { YuvEndiannes::LittleEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_p10_to_image_impl::<
                    { YuvSourceChannels::Rgb as u8 },
                    { YuvChromaSample::YUV422 as u8 },
                    { YuvEndiannes::LittleEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
    };
    dispatcher(
        y_plane, y_stride, u_plane, u_stride, v_plane, v_stride, rgb, rgb_stride, width, height,
        range, matrix,
    );
}

/// Convert YUV 444 planar format with 10-bit pixel format to RGBA format.
///
/// This function takes YUV 444 planar data with 10-bit precision.
/// and converts it to RGBA format with 8-bit precision per channel
///
/// # Arguments
///
/// * `y_plane` -  A slice containing Y (luminance) with 10 bit depth.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `u_plane` - A slice to load the U (chrominance) with 10 bit depth.
/// * `u_stride` - The stride (bytes per row) for the U plane.
/// * `v_plane` - A slice to load the V (chrominance) with 10 bit depth.
/// * `v_stride` - The stride (bytes per row) for the U plane.
/// * `width` - The width of the YUV image.
/// * `height` - The height of the YUV image.
/// * `rgba_data` - A mutable slice to store the converted RGBA data.
/// * `rgba_stride` - The stride (bytes per row) for RGBA data.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
/// * `endianness` - The endianness of stored bytes
/// * `bytes_packing` - position of significant bytes ( most significant or least significant ) if it in most significant it should be stated as per Apple *kCVPixelFormatType_422YpCbCr10BiPlanarFullRange/kCVPixelFormatType_422YpCbCr10BiPlanarVideoRange*
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input RGBA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv444_p10_to_rgba(
    y_plane: &[u16],
    y_stride: u32,
    u_plane: &[u16],
    u_stride: u32,
    v_plane: &[u16],
    v_stride: u32,
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
                yuv_p10_to_image_impl::<
                    { YuvSourceChannels::Rgba as u8 },
                    { YuvChromaSample::YUV444 as u8 },
                    { YuvEndiannes::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_p10_to_image_impl::<
                    { YuvSourceChannels::Rgba as u8 },
                    { YuvChromaSample::YUV444 as u8 },
                    { YuvEndiannes::BigEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
        YuvEndiannes::LittleEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_p10_to_image_impl::<
                    { YuvSourceChannels::Rgba as u8 },
                    { YuvChromaSample::YUV444 as u8 },
                    { YuvEndiannes::LittleEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_p10_to_image_impl::<
                    { YuvSourceChannels::Rgba as u8 },
                    { YuvChromaSample::YUV444 as u8 },
                    { YuvEndiannes::LittleEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
    };
    dispatcher(
        y_plane,
        y_stride,
        u_plane,
        u_stride,
        v_plane,
        v_stride,
        rgba,
        rgba_stride,
        width,
        height,
        range,
        matrix,
    );
}

/// Convert YUV 444 planar format with 10-bit pixel format to RGB format.
///
/// This function takes YUV 444 planar data with 10-bit precision.
/// and converts it to RGB format with 8-bit precision per channel
///
/// # Arguments
///
/// * `y_plane` -  A slice containing Y (luminance) with 10 bit depth.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `u_plane` - A slice to load the U (chrominance) with 10 bit depth.
/// * `u_stride` - The stride (bytes per row) for the U plane.
/// * `v_plane` - A slice to load the V (chrominance) with 10 bit depth.
/// * `v_stride` - The stride (bytes per row) for the U plane.
/// * `width` - The width of the YUV image.
/// * `height` - The height of the YUV image.
/// * `rgb_data` - A mutable slice to store the converted RGB data.
/// * `rgb_stride` - The stride (bytes per row) for RGB data.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
/// * `endianness` - The endianness of stored bytes
/// * `bytes_packing` - position of significant bytes ( most significant or least significant ) if it in most significant it should be stated as per Apple *kCVPixelFormatType_422YpCbCr10BiPlanarFullRange/kCVPixelFormatType_422YpCbCr10BiPlanarVideoRange*
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input RGB data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv444_p10_to_rgb(
    y_plane: &[u16],
    y_stride: u32,
    u_plane: &[u16],
    u_stride: u32,
    v_plane: &[u16],
    v_stride: u32,
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
                yuv_p10_to_image_impl::<
                    { YuvSourceChannels::Rgb as u8 },
                    { YuvChromaSample::YUV444 as u8 },
                    { YuvEndiannes::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_p10_to_image_impl::<
                    { YuvSourceChannels::Rgb as u8 },
                    { YuvChromaSample::YUV444 as u8 },
                    { YuvEndiannes::BigEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
        YuvEndiannes::LittleEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_p10_to_image_impl::<
                    { YuvSourceChannels::Rgb as u8 },
                    { YuvChromaSample::YUV444 as u8 },
                    { YuvEndiannes::LittleEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_p10_to_image_impl::<
                    { YuvSourceChannels::Rgb as u8 },
                    { YuvChromaSample::YUV444 as u8 },
                    { YuvEndiannes::LittleEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
    };
    dispatcher(
        y_plane, y_stride, u_plane, u_stride, v_plane, v_stride, rgb, rgb_stride, width, height,
        range, matrix,
    );
}

/// Convert YUV 444 planar format with 10-bit pixel format to BGRA format.
///
/// This function takes YUV 444 planar data with 10-bit precision.
/// and converts it to BGRA format with 8-bit precision per channel
///
/// # Arguments
///
/// * `y_plane` -  A slice containing Y (luminance) with 10 bit depth.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `u_plane` - A slice to load the U (chrominance) with 10 bit depth.
/// * `u_stride` - The stride (bytes per row) for the U plane.
/// * `v_plane` - A slice to load the V (chrominance) with 10 bit depth.
/// * `v_stride` - The stride (bytes per row) for the U plane.
/// * `width` - The width of the YUV image.
/// * `height` - The height of the YUV image.
/// * `bgra` - A mutable slice to store the converted BGRA data.
/// * `bgra_stride` - The stride (bytes per row) for BGRA data.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
/// * `endianness` - The endianness of stored bytes
/// * `bytes_packing` - position of significant bytes ( most significant or least significant ) if it in most significant it should be stated as per Apple *kCVPixelFormatType_422YpCbCr10BiPlanarFullRange/kCVPixelFormatType_422YpCbCr10BiPlanarVideoRange*
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGRA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv444_p10_to_bgra(
    y_plane: &[u16],
    y_stride: u32,
    u_plane: &[u16],
    u_stride: u32,
    v_plane: &[u16],
    v_stride: u32,
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
                yuv_p10_to_image_impl::<
                    { YuvSourceChannels::Bgra as u8 },
                    { YuvChromaSample::YUV444 as u8 },
                    { YuvEndiannes::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_p10_to_image_impl::<
                    { YuvSourceChannels::Bgra as u8 },
                    { YuvChromaSample::YUV444 as u8 },
                    { YuvEndiannes::BigEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
        YuvEndiannes::LittleEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_p10_to_image_impl::<
                    { YuvSourceChannels::Bgra as u8 },
                    { YuvChromaSample::YUV444 as u8 },
                    { YuvEndiannes::LittleEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_p10_to_image_impl::<
                    { YuvSourceChannels::Bgra as u8 },
                    { YuvChromaSample::YUV444 as u8 },
                    { YuvEndiannes::LittleEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
    };
    dispatcher(
        y_plane,
        y_stride,
        u_plane,
        u_stride,
        v_plane,
        v_stride,
        bgra,
        bgra_stride,
        width,
        height,
        range,
        matrix,
    );
}

/// Convert YUV 444 planar format with 10-bit pixel format to BGR format.
///
/// This function takes YUV 444 planar data with 10-bit precision.
/// and converts it to BGR format with 8-bit precision per channel
///
/// # Arguments
///
/// * `y_plane` -  A slice containing Y (luminance) with 10 bit depth.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `u_plane` - A slice to load the U (chrominance) with 10 bit depth.
/// * `u_stride` - The stride (bytes per row) for the U plane.
/// * `v_plane` - A slice to load the V (chrominance) with 10 bit depth.
/// * `v_stride` - The stride (bytes per row) for the U plane.
/// * `width` - The width of the YUV image.
/// * `height` - The height of the YUV image.
/// * `bgr` - A mutable slice to store the converted BGR data.
/// * `bgr_stride` - The stride (bytes per row) for BGR data.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
/// * `endianness` - The endianness of stored bytes
/// * `bytes_packing` - position of significant bytes ( most significant or least significant ) if it in most significant it should be stated as per Apple *kCVPixelFormatType_422YpCbCr10BiPlanarFullRange/kCVPixelFormatType_422YpCbCr10BiPlanarVideoRange*
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGR data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv444_p10_to_bgr(
    y_plane: &[u16],
    y_stride: u32,
    u_plane: &[u16],
    u_stride: u32,
    v_plane: &[u16],
    v_stride: u32,
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
                yuv_p10_to_image_impl::<
                    { YuvSourceChannels::Bgr as u8 },
                    { YuvChromaSample::YUV444 as u8 },
                    { YuvEndiannes::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_p10_to_image_impl::<
                    { YuvSourceChannels::Bgr as u8 },
                    { YuvChromaSample::YUV444 as u8 },
                    { YuvEndiannes::BigEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
        YuvEndiannes::LittleEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_p10_to_image_impl::<
                    { YuvSourceChannels::Bgr as u8 },
                    { YuvChromaSample::YUV444 as u8 },
                    { YuvEndiannes::LittleEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_p10_to_image_impl::<
                    { YuvSourceChannels::Bgr as u8 },
                    { YuvChromaSample::YUV444 as u8 },
                    { YuvEndiannes::LittleEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
    };
    dispatcher(
        y_plane, y_stride, u_plane, u_stride, v_plane, v_stride, bgr, bgr_stride, width, height,
        range, matrix,
    );
}
