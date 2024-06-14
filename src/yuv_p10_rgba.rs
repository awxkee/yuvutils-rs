/*
 * // Copyright (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use crate::neon::neon_yuv_p10_to_rgba_row;
use std::slice;

use crate::yuv_support::{
    get_inverse_transform, get_kr_kb, get_yuv_range, YuvBytesPosition, YuvChromaSample, YuvEndian,
    YuvRange, YuvSourceChannels, YuvStandardMatrix,
};

fn yuv_p10_to_rgbx_impl<
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
    let destination_channels: YuvSourceChannels = DESTINATION_CHANNELS.into();
    let channels = destination_channels.get_channels_count();
    let chroma_subsampling: YuvChromaSample = SAMPLING.into();
    let endianness: YuvEndian = ENDIANNESS.into();
    let bytes_position: YuvBytesPosition = BYTES_POSITION.into();
    let range = get_yuv_range(10, range);
    let kr_kb = get_kr_kb(matrix);
    let max_range_p10 = (2f32.powi(10) - 1f32) as u32;
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
        let y_ld = unsafe { slice::from_raw_parts(y_ld_ptr, width as usize) };
        let u_ld_ptr = unsafe { u_src_ptr.offset(u_offset as isize) as *const u16 };
        let u_ld = unsafe { slice::from_raw_parts(u_ld_ptr, width as usize) };
        let v_ld_ptr = unsafe { v_src_ptr.offset(v_offset as isize) as *const u16 };
        let v_ld = unsafe { slice::from_raw_parts(v_ld_ptr, width as usize) };

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
                YuvEndian::BigEndian => {
                    let mut y_vl = u16::from_be(unsafe { *y_ld.get_unchecked(x) }) as i32;
                    let mut cb_vl = u16::from_be(unsafe { *u_ld.get_unchecked(cx) }) as i32;
                    let mut cr_vl = u16::from_be(unsafe { *v_ld.get_unchecked(cx) }) as i32;
                    if bytes_position == YuvBytesPosition::MostSignificantBytes {
                        y_vl = y_vl >> 6;
                        cb_vl = cb_vl >> 6;
                        cr_vl = cr_vl >> 6;
                    }
                    y_value = (y_vl - bias_y) * y_coef;

                    cb_value = cb_vl - bias_uv;
                    cr_value = cr_vl - bias_uv;
                }
                YuvEndian::LittleEndian => {
                    let mut y_vl = u16::from_le(unsafe { *y_ld.get_unchecked(x) }) as i32;
                    let mut cb_vl = u16::from_le(unsafe { *u_ld.get_unchecked(cx) }) as i32;
                    let mut cr_vl = u16::from_le(unsafe { *v_ld.get_unchecked(cx) }) as i32;
                    if bytes_position == YuvBytesPosition::MostSignificantBytes {
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
                *rgba.get_unchecked_mut(rgb_offset + destination_channels.get_b_channel_offset()) =
                    b as u8;
                *rgba.get_unchecked_mut(rgb_offset + destination_channels.get_g_channel_offset()) =
                    g as u8;
                *rgba.get_unchecked_mut(rgb_offset + destination_channels.get_r_channel_offset()) =
                    r as u8;
                if destination_channels.has_alpha() {
                    *rgba.get_unchecked_mut(
                        rgb_offset + destination_channels.get_a_channel_offset(),
                    ) = 255;
                }
            }

            x += 1;

            if x + 1 < width as usize {
                let y_value: i32;
                match endianness {
                    YuvEndian::BigEndian => {
                        let mut y_vl = u16::from_be(unsafe { *y_ld.get_unchecked(x) }) as i32;
                        if bytes_position == YuvBytesPosition::MostSignificantBytes {
                            y_vl = y_vl >> 6;
                        }
                        y_value = (y_vl - bias_y) * y_coef;
                    }
                    YuvEndian::LittleEndian => {
                        let mut y_vl = u16::from_le(unsafe { *y_ld.get_unchecked(x) }) as i32;
                        if bytes_position == YuvBytesPosition::MostSignificantBytes {
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
                    *rgba.get_unchecked_mut(
                        rgb_offset + destination_channels.get_b_channel_offset(),
                    ) = b as u8;
                    *rgba.get_unchecked_mut(
                        rgb_offset + destination_channels.get_g_channel_offset(),
                    ) = g as u8;
                    *rgba.get_unchecked_mut(
                        rgb_offset + destination_channels.get_r_channel_offset(),
                    ) = r as u8;
                    if destination_channels.has_alpha() {
                        *rgba.get_unchecked_mut(
                            rgb_offset + destination_channels.get_a_channel_offset(),
                        ) = 255;
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

/// Convert YUV 420 planar format with 10-bit (Little-Endian) pixel format to BGRA format.
///
/// This function takes YUV 420 planar data with 10-bit precision stored in Little-Endian.
/// and converts it to BGRA format with 8-bit precision per channel
///
/// # Arguments
///
/// * `y_plane` -  A slice containing Y (luminance) with 10 bit depth (Little-Endian).
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `u_plane` - A slice to load the U (chrominance) with 10 bit depth (Little-Endian).
/// * `u_stride` - The stride (bytes per row) for the U plane.
/// * `v_plane` - A slice to load the V (chrominance) with 10 bit depth (Little-Endian).
/// * `v_stride` - The stride (bytes per row) for the U plane.
/// * `width` - The width of the YUV image.
/// * `height` - The height of the YUV image.
/// * `bgra_data` - A mutable slice to store the converted BGRA data.
/// * `bgra_stride` - The stride (bytes per row) for BGRA data.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
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
) {
    yuv_p10_to_rgbx_impl::<
        { YuvSourceChannels::Bgra as u8 },
        { YuvChromaSample::YUV420 as u8 },
        { YuvEndian::LittleEndian as u8 },
        { YuvBytesPosition::LeastSignificantBytes as u8 },
    >(
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

/// Convert YUV 422 format with 10-bit (Little-Endian) pixel format to BGRA format .
///
/// This function takes YUV 422 data with 10-bit precision stored in Little-Endian.
/// and converts it to BGRA format with 8-bit precision per channel.
///
/// # Arguments
///
/// * `y_plane` -  A slice containing Y (luminance) with 10 bit depth (Little-Endian).
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `u_plane` - A slice to load the U (chrominance) with 10 bit depth (Little-Endian).
/// * `u_stride` - The stride (bytes per row) for the U plane.
/// * `v_plane` - A slice to load the V (chrominance) with 10 bit depth (Little-Endian).
/// * `v_stride` - The stride (bytes per row) for the U plane
/// * `width` - The width of the YUV image.
/// * `height` - The height of the YUV image.
/// * `bgra_data` - A mutable slice to store the converted BGRA data.
/// * `bgra_stride` - The stride (bytes per row) for BGRA data.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
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
) {
    yuv_p10_to_rgbx_impl::<
        { YuvSourceChannels::Bgra as u8 },
        { YuvChromaSample::YUV422 as u8 },
        { YuvEndian::LittleEndian as u8 },
        { YuvBytesPosition::LeastSignificantBytes as u8 },
    >(
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

/// Convert YUV 420 format with 10-bit (Big-Endian) pixel format to BGRA format.
///
/// This function takes YUV 420 data with 10-bit precision stored in Big-Endian.
/// and converts it to BGRA format with 8-bit precision per channel.
///
/// # Arguments
///
/// * `y_plane` -  A slice containing Y (luminance) with 10 bit depth (Big-Endian).
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `u_plane` - A slice to load the U (chrominance) with 10 bit depth (Big-Endian).
/// * `u_stride` - The stride (bytes per row) for the U plane.
/// * `v_plane` - A slice to load the V (chrominance) with 10 bit depth (Big-Endian).
/// * `v_stride` - The stride (bytes per row) for the U plane
/// * `width` - The width of the YUV image.
/// * `height` - The height of the YUV image.
/// * `bgra_data` - A mutable slice to store the converted BGRA data.
/// * `bgra_stride` - The stride (bytes per row) for BGRA data.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGRA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv420_p10_be_to_bgra(
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
) {
    yuv_p10_to_rgbx_impl::<
        { YuvSourceChannels::Bgra as u8 },
        { YuvChromaSample::YUV420 as u8 },
        { YuvEndian::BigEndian as u8 },
        { YuvBytesPosition::LeastSignificantBytes as u8 },
    >(
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

/// Convert YUV 422 format with 10-bit (Big-Endian) pixel format to BGRA format.
///
/// This function takes YUV 422 data with 10-bit precision stored in Big-Endian.
/// and converts it to BGRA format with 8-bit precision per channel.
///
/// # Arguments
///
/// * `y_plane` -  A slice containing Y (luminance) with 10 bit depth (Big-Endian).
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `u_plane` - A slice to load the U (chrominance) with 10 bit depth (Big-Endian).
/// * `u_stride` - The stride (bytes per row) for the U plane.
/// * `v_plane` - A slice to load the V (chrominance) with 10 bit depth (Big-Endian).
/// * `v_stride` - The stride (bytes per row) for the U plane
/// * `width` - The width of the YUV image.
/// * `height` - The height of the YUV image.
/// * `bgra_data` - A mutable slice to store the converted BGRA data.
/// * `bgra_stride` - The stride (bytes per row) for BGRA data.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGRA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv422_p10_be_to_bgra(
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
) {
    yuv_p10_to_rgbx_impl::<
        { YuvSourceChannels::Bgra as u8 },
        { YuvChromaSample::YUV422 as u8 },
        { YuvEndian::BigEndian as u8 },
        { YuvBytesPosition::LeastSignificantBytes as u8 },
    >(
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

/// Convert YUV 420 planar format with 10-bit (Little-Endian) pixel format to RGBA format.
///
/// This function takes YUV 420 planar data with 10-bit precision stored in Little-Endian.
/// and converts it to RGBA format with 8-bit precision per channel
///
/// # Arguments
///
/// * `y_plane` -  A slice containing Y (luminance) with 10 bit depth (Little-Endian).
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `u_plane` - A slice to load the U (chrominance) with 10 bit depth (Little-Endian).
/// * `u_stride` - The stride (bytes per row) for the U plane.
/// * `v_plane` - A slice to load the V (chrominance) with 10 bit depth (Little-Endian).
/// * `v_stride` - The stride (bytes per row) for the U plane.
/// * `width` - The width of the YUV image.
/// * `height` - The height of the YUV image.
/// * `rgba_data` - A mutable slice to store the converted BGRA data.
/// * `rgba_stride` - The stride (bytes per row) for RGBA data.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
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
) {
    yuv_p10_to_rgbx_impl::<
        { YuvSourceChannels::Rgba as u8 },
        { YuvChromaSample::YUV420 as u8 },
        { YuvEndian::LittleEndian as u8 },
        { YuvBytesPosition::LeastSignificantBytes as u8 },
    >(
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

/// Convert YUV 422 format with 10-bit (Little-Endian) pixel format to RGBA format .
///
/// This function takes YUV 422 data with 10-bit precision stored in Little-Endian.
/// and converts it to RGBA format with 8-bit precision per channel.
///
/// # Arguments
///
/// * `y_plane` -  A slice containing Y (luminance) with 10 bit depth (Little-Endian).
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `u_plane` - A slice to load the U (chrominance) with 10 bit depth (Little-Endian).
/// * `u_stride` - The stride (bytes per row) for the U plane.
/// * `v_plane` - A slice to load the V (chrominance) with 10 bit depth (Little-Endian).
/// * `v_stride` - The stride (bytes per row) for the U plane
/// * `width` - The width of the YUV image.
/// * `height` - The height of the YUV image.
/// * `rgba_data` - A mutable slice to store the converted RGBA data.
/// * `rgba_stride` - The stride (bytes per row) for RGBA data.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
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
) {
    yuv_p10_to_rgbx_impl::<
        { YuvSourceChannels::Rgba as u8 },
        { YuvChromaSample::YUV422 as u8 },
        { YuvEndian::LittleEndian as u8 },
        { YuvBytesPosition::LeastSignificantBytes as u8 },
    >(
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

/// Convert YUV 420 planar format with 10-bit (Big-Endian) pixel format to RGBA format.
///
/// This function takes YUV 420 planar data with 10-bit precision stored in Big-Endian.
/// and converts it to RGBA format with 8-bit precision per channel
///
/// # Arguments
///
/// * `y_plane` -  A slice containing Y (luminance) with 10 bit depth (Big-Endian).
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `u_plane` - A slice to load the U (chrominance) with 10 bit depth (Big-Endian).
/// * `u_stride` - The stride (bytes per row) for the U plane.
/// * `v_plane` - A slice to load the V (chrominance) with 10 bit depth (Big-Endian).
/// * `v_stride` - The stride (bytes per row) for the U plane.
/// * `width` - The width of the YUV image.
/// * `height` - The height of the YUV image.
/// * `rgba_data` - A mutable slice to store the converted RGBA data.
/// * `rgba_stride` - The stride (bytes per row) for RGBA data.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input RGBA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv420_p10_be_to_rgba(
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
    yuv_p10_to_rgbx_impl::<
        { YuvSourceChannels::Rgba as u8 },
        { YuvChromaSample::YUV420 as u8 },
        { YuvEndian::BigEndian as u8 },
        { YuvBytesPosition::LeastSignificantBytes as u8 },
    >(
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

/// Convert YUV 422 format with 10-bit (Big-Endian) pixel format to RGBA format .
///
/// This function takes YUV 422 data with 10-bit precision stored in Big-Endian.
/// and converts it to RGBA format with 8-bit precision per channel.
///
/// # Arguments
///
/// * `y_plane` -  A slice containing Y (luminance) with 10 bit depth (Big-Endian).
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `u_plane` - A slice to load the U (chrominance) with 10 bit depth (Big-Endian).
/// * `u_stride` - The stride (bytes per row) for the U plane.
/// * `v_plane` - A slice to load the V (chrominance) with 10 bit depth (Big-Endian).
/// * `v_stride` - The stride (bytes per row) for the U plane
/// * `width` - The width of the YUV image.
/// * `height` - The height of the YUV image.
/// * `rgba_data` - A mutable slice to store the converted RGBA data.
/// * `rgba_stride` - The stride (bytes per row) for RGBA data.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input RGBA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv422_p10_be_to_rgba(
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
    yuv_p10_to_rgbx_impl::<
        { YuvSourceChannels::Rgba as u8 },
        { YuvChromaSample::YUV422 as u8 },
        { YuvEndian::BigEndian as u8 },
        { YuvBytesPosition::LeastSignificantBytes as u8 },
    >(
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

/// Convert YUV 444 planar format with 10-bit (Little-Endian) pixel format to RGBA format.
///
/// This function takes YUV 444 planar data with 10-bit precision stored in Little-Endian.
/// and converts it to RGBA format with 8-bit precision per channel
///
/// # Arguments
///
/// * `y_plane` -  A slice containing Y (luminance) with 10 bit depth (Little-Endian).
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `u_plane` - A slice to load the U (chrominance) with 10 bit depth (Little-Endian).
/// * `u_stride` - The stride (bytes per row) for the U plane.
/// * `v_plane` - A slice to load the V (chrominance) with 10 bit depth (Little-Endian).
/// * `v_stride` - The stride (bytes per row) for the U plane.
/// * `width` - The width of the YUV image.
/// * `height` - The height of the YUV image.
/// * `rgba_data` - A mutable slice to store the converted RGBA data.
/// * `rgba_stride` - The stride (bytes per row) for RGBA data.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
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
) {
    yuv_p10_to_rgbx_impl::<
        { YuvSourceChannels::Rgba as u8 },
        { YuvChromaSample::YUV444 as u8 },
        { YuvEndian::LittleEndian as u8 },
        { YuvBytesPosition::LeastSignificantBytes as u8 },
    >(
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

/// Convert YUV 444 format with 10-bit (Big-Endian) pixel format to RGBA format .
///
/// This function takes YUV 444 data with 10-bit precision stored in Big-Endian.
/// and converts it to RGBA format with 8-bit precision per channel.
///
/// # Arguments
///
/// * `y_plane` -  A slice containing Y (luminance) with 10 bit depth (Big-Endian).
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `u_plane` - A slice to load the U (chrominance) with 10 bit depth (Big-Endian).
/// * `u_stride` - The stride (bytes per row) for the U plane.
/// * `v_plane` - A slice to load the V (chrominance) with 10 bit depth (Big-Endian).
/// * `v_stride` - The stride (bytes per row) for the U plane
/// * `width` - The width of the YUV image.
/// * `height` - The height of the YUV image.
/// * `rgba_data` - A mutable slice to store the converted RGBA data.
/// * `rgba_stride` - The stride (bytes per row) for RGBA data.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input RGBA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv444_p10_be_to_rgba(
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
    yuv_p10_to_rgbx_impl::<
        { YuvSourceChannels::Rgba as u8 },
        { YuvChromaSample::YUV444 as u8 },
        { YuvEndian::BigEndian as u8 },
        { YuvBytesPosition::LeastSignificantBytes as u8 },
    >(
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

/// Convert YUV 444 planar format with 10-bit (Little-Endian) pixel format to BGRA format.
///
/// This function takes YUV 444 planar data with 10-bit precision stored in Little-Endian.
/// and converts it to BGRA format with 8-bit precision per channel
///
/// # Arguments
///
/// * `y_plane` -  A slice containing Y (luminance) with 10 bit depth (Little-Endian).
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `u_plane` - A slice to load the U (chrominance) with 10 bit depth (Little-Endian).
/// * `u_stride` - The stride (bytes per row) for the U plane.
/// * `v_plane` - A slice to load the V (chrominance) with 10 bit depth (Little-Endian).
/// * `v_stride` - The stride (bytes per row) for the U plane.
/// * `width` - The width of the YUV image.
/// * `height` - The height of the YUV image.
/// * `rgba_data` - A mutable slice to store the converted BGRA data.
/// * `rgba_stride` - The stride (bytes per row) for BGRA data.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
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
    rgba: &mut [u8],
    rgba_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) {
    yuv_p10_to_rgbx_impl::<
        { YuvSourceChannels::Bgra as u8 },
        { YuvChromaSample::YUV444 as u8 },
        { YuvEndian::LittleEndian as u8 },
        { YuvBytesPosition::LeastSignificantBytes as u8 },
    >(
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

/// Convert YUV 444 format with 10-bit (Big-Endian) pixel format to BGRA format .
///
/// This function takes YUV 444 data with 10-bit precision stored in Big-Endian.
/// and converts it to BGRA format with 8-bit precision per channel.
///
/// # Arguments
///
/// * `y_plane` -  A slice containing Y (luminance) with 10 bit depth (Big-Endian).
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `u_plane` - A slice to load the U (chrominance) with 10 bit depth (Big-Endian).
/// * `u_stride` - The stride (bytes per row) for the U plane.
/// * `v_plane` - A slice to load the V (chrominance) with 10 bit depth (Big-Endian).
/// * `v_stride` - The stride (bytes per row) for the U plane
/// * `width` - The width of the YUV image.
/// * `height` - The height of the YUV image.
/// * `rgba_data` - A mutable slice to store the converted BGRA data.
/// * `rgba_stride` - The stride (bytes per row) for BGRA data.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGRA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv444_p10_be_to_bgra(
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
    yuv_p10_to_rgbx_impl::<
        { YuvSourceChannels::Bgra as u8 },
        { YuvChromaSample::YUV444 as u8 },
        { YuvEndian::BigEndian as u8 },
        { YuvBytesPosition::LeastSignificantBytes as u8 },
    >(
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
