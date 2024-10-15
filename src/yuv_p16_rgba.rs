/*
 * // Copyright (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */
use crate::yuv_p10_rgba::yuv_p16_to_image_impl;
use crate::yuv_support::{
    YuvBytesPacking, YuvChromaSample, YuvEndianness, YuvRange, YuvSourceChannels, YuvStandardMatrix,
};

/// Convert YUV 420 planar format with 8+ bit pixel format to BGRA format.
///
/// This function takes YUV 420 planar data with 8+ bit precision.
/// and converts it to BGRA format with 8-bit precision per channel
///
/// # Arguments
///
/// * `y_plane` -  A slice containing Y (luminance) with 8+ bit depth.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `u_plane` - A slice to load the U (chrominance) with 8+ bit depth.
/// * `u_stride` - The stride (bytes per row) for the U plane.
/// * `v_plane` - A slice to load the V (chrominance) with 8+ bit depth.
/// * `v_stride` - The stride (bytes per row) for the U plane.
/// * `width` - The width of the YUV image.
/// * `height` - The height of the YUV image.
/// * `bgra` - A mutable slice to store the converted BGRA data.
/// * `bgra_stride` - The stride (bytes per row) for BGRA data.
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
pub fn yuv420_p16_to_bgra(
    y_plane: &[u16],
    y_stride: u32,
    u_plane: &[u16],
    u_stride: u32,
    v_plane: &[u16],
    v_stride: u32,
    bgra: &mut [u8],
    bgra_stride: u32,
    bit_depth: usize,
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
                yuv_p16_to_image_impl::<
                    { YuvSourceChannels::Bgra as u8 },
                    { YuvChromaSample::YUV420 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_p16_to_image_impl::<
                    { YuvSourceChannels::Bgra as u8 },
                    { YuvChromaSample::YUV420 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
        YuvEndianness::LittleEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_p16_to_image_impl::<
                    { YuvSourceChannels::Bgra as u8 },
                    { YuvChromaSample::YUV420 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_p16_to_image_impl::<
                    { YuvSourceChannels::Bgra as u8 },
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
        bit_depth,
    );
}

/// Convert YUV 420 planar format with 8+ bit pixel format to BGR format.
///
/// This function takes YUV 420 planar data with 8+ bit precision.
/// and converts it to BGR format with 8-bit precision per channel
///
/// # Arguments
///
/// * `y_plane` -  A slice containing Y (luminance) with 8+ bit depth.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `u_plane` - A slice to load the U (chrominance) with 8+ bit depth.
/// * `u_stride` - The stride (bytes per row) for the U plane.
/// * `v_plane` - A slice to load the V (chrominance) with 8+ bit depth.
/// * `v_stride` - The stride (bytes per row) for the U plane.
/// * `width` - The width of the YUV image.
/// * `height` - The height of the YUV image.
/// * `bgra` - A mutable slice to store the converted BGR data.
/// * `bgra_stride` - The stride (bytes per row) for BGR data.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
/// * `endianness` - The endianness of stored bytes
/// * `bytes_packing` - position of significant bytes ( most significant or least significant ) if it in most significant it should be stated as per Apple *kCVPixelFormatType_422YpCbCr10BiPlanarFullRange/kCVPixelFormatType_422YpCbCr10BiPlanarVideoRange*
/// * `bit_depth` - Bit depth of source YUV planes
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGR data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv420_p16_to_bgr(
    y_plane: &[u16],
    y_stride: u32,
    u_plane: &[u16],
    u_stride: u32,
    v_plane: &[u16],
    v_stride: u32,
    bgr: &mut [u8],
    bgr_stride: u32,
    bit_depth: usize,
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
                yuv_p16_to_image_impl::<
                    { YuvSourceChannels::Bgr as u8 },
                    { YuvChromaSample::YUV420 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_p16_to_image_impl::<
                    { YuvSourceChannels::Bgr as u8 },
                    { YuvChromaSample::YUV420 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
        YuvEndianness::LittleEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_p16_to_image_impl::<
                    { YuvSourceChannels::Bgr as u8 },
                    { YuvChromaSample::YUV420 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_p16_to_image_impl::<
                    { YuvSourceChannels::Bgr as u8 },
                    { YuvChromaSample::YUV420 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
    };
    dispatcher(
        y_plane, y_stride, u_plane, u_stride, v_plane, v_stride, bgr, bgr_stride, width, height,
        range, matrix, bit_depth,
    );
}

/// Convert YUV 422 format with 8+ bit pixel format to BGRA format .
///
/// This function takes YUV 422 data with 8+ bit precision.
/// and converts it to BGRA format with 8-bit precision per channel.
///
/// # Arguments
///
/// * `y_plane` -  A slice containing Y (luminance) with 8+ bit depth.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `u_plane` - A slice to load the U (chrominance) with 8+ bit depth.
/// * `u_stride` - The stride (bytes per row) for the U plane.
/// * `v_plane` - A slice to load the V (chrominance) with 8+ bit depth.
/// * `v_stride` - The stride (bytes per row) for the U plane
/// * `width` - The width of the YUV image.
/// * `height` - The height of the YUV image.
/// * `bgra` - A mutable slice to store the converted BGRA data.
/// * `bgra_stride` - The stride (bytes per row) for BGRA data.
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
pub fn yuv422_p16_to_bgra(
    y_plane: &[u16],
    y_stride: u32,
    u_plane: &[u16],
    u_stride: u32,
    v_plane: &[u16],
    v_stride: u32,
    bgra: &mut [u8],
    bgra_stride: u32,
    bit_depth: usize,
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
                yuv_p16_to_image_impl::<
                    { YuvSourceChannels::Bgra as u8 },
                    { YuvChromaSample::YUV422 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_p16_to_image_impl::<
                    { YuvSourceChannels::Bgra as u8 },
                    { YuvChromaSample::YUV422 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
        YuvEndianness::LittleEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_p16_to_image_impl::<
                    { YuvSourceChannels::Bgra as u8 },
                    { YuvChromaSample::YUV422 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_p16_to_image_impl::<
                    { YuvSourceChannels::Bgra as u8 },
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
        bit_depth,
    );
}

/// Convert YUV 422 format with 8+ bit pixel format to BGR format.
///
/// This function takes YUV 422 data with 8+ bit precision.
/// and converts it to BGR format with 8-bit precision per channel.
///
/// # Arguments
///
/// * `y_plane` -  A slice containing Y (luminance) with 8+ bit depth.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `u_plane` - A slice to load the U (chrominance) with 8+ bit depth.
/// * `u_stride` - The stride (bytes per row) for the U plane.
/// * `v_plane` - A slice to load the V (chrominance) with 8+ bit depth.
/// * `v_stride` - The stride (bytes per row) for the U plane
/// * `width` - The width of the YUV image.
/// * `height` - The height of the YUV image.
/// * `bgr` - A mutable slice to store the converted BGR data.
/// * `bgr_stride` - The stride (bytes per row) for BGR data.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
/// * `endianness` - The endianness of stored bytes
/// * `bytes_packing` - position of significant bytes ( most significant or least significant ) if it in most significant it should be stated as per Apple *kCVPixelFormatType_422YpCbCr10BiPlanarFullRange/kCVPixelFormatType_422YpCbCr10BiPlanarVideoRange*
/// * `bit_depth` - Bit depth of source YUV planes
///
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGR data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv422_p16_to_bgr(
    y_plane: &[u16],
    y_stride: u32,
    u_plane: &[u16],
    u_stride: u32,
    v_plane: &[u16],
    v_stride: u32,
    bgr: &mut [u8],
    bgr_stride: u32,
    bit_depth: usize,
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
                yuv_p16_to_image_impl::<
                    { YuvSourceChannels::Bgr as u8 },
                    { YuvChromaSample::YUV422 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_p16_to_image_impl::<
                    { YuvSourceChannels::Bgr as u8 },
                    { YuvChromaSample::YUV422 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
        YuvEndianness::LittleEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_p16_to_image_impl::<
                    { YuvSourceChannels::Bgr as u8 },
                    { YuvChromaSample::YUV422 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_p16_to_image_impl::<
                    { YuvSourceChannels::Bgr as u8 },
                    { YuvChromaSample::YUV422 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
    };
    dispatcher(
        y_plane, y_stride, u_plane, u_stride, v_plane, v_stride, bgr, bgr_stride, width, height,
        range, matrix, bit_depth,
    );
}

/// Convert YUV 420 planar format with 8+ bit pixel format to RGBA format.
///
/// This function takes YUV 420 planar data with 8+ bit precision.
/// and converts it to RGBA format with 8-bit precision per channel
///
/// # Arguments
///
/// * `y_plane` -  A slice containing Y (luminance) with 8+ bit depth.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `u_plane` - A slice to load the U (chrominance) with 8+ bit depth.
/// * `u_stride` - The stride (bytes per row) for the U plane.
/// * `v_plane` - A slice to load the V (chrominance) with 8+ bit depth.
/// * `v_stride` - The stride (bytes per row) for the U plane.
/// * `width` - The width of the YUV image.
/// * `height` - The height of the YUV image.
/// * `rgba` - A mutable slice to store the converted RGBA data.
/// * `rgba_stride` - The stride (bytes per row) for RGBA data.
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
pub fn yuv420_p16_to_rgba(
    y_plane: &[u16],
    y_stride: u32,
    u_plane: &[u16],
    u_stride: u32,
    v_plane: &[u16],
    v_stride: u32,
    rgba: &mut [u8],
    rgba_stride: u32,
    bit_depth: usize,
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
                yuv_p16_to_image_impl::<
                    { YuvSourceChannels::Rgba as u8 },
                    { YuvChromaSample::YUV420 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_p16_to_image_impl::<
                    { YuvSourceChannels::Rgba as u8 },
                    { YuvChromaSample::YUV420 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
        YuvEndianness::LittleEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_p16_to_image_impl::<
                    { YuvSourceChannels::Rgba as u8 },
                    { YuvChromaSample::YUV420 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_p16_to_image_impl::<
                    { YuvSourceChannels::Rgba as u8 },
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
        bit_depth,
    );
}

/// Convert YUV 420 planar format with 8+ bit pixel format to RGB format.
///
/// This function takes YUV 420 planar data with 8+ bit precision.
/// and converts it to RGB format with 8-bit precision per channel
///
/// # Arguments
///
/// * `y_plane` -  A slice containing Y (luminance) with 8+ bit depth.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `u_plane` - A slice to load the U (chrominance) with 8+ bit depth.
/// * `u_stride` - The stride (bytes per row) for the U plane.
/// * `v_plane` - A slice to load the V (chrominance) with 8+ bit depth.
/// * `v_stride` - The stride (bytes per row) for the U plane.
/// * `width` - The width of the YUV image.
/// * `height` - The height of the YUV image.
/// * `rgb` - A mutable slice to store the converted RGB data.
/// * `rgb_stride` - The stride (bytes per row) for RGB data.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
/// * `endianness` - The endianness of stored bytes
/// * `bytes_packing` - position of significant bytes ( most significant or least significant ) if it in most significant it should be stated as per Apple *kCVPixelFormatType_422YpCbCr10BiPlanarFullRange/kCVPixelFormatType_422YpCbCr10BiPlanarVideoRange*
/// * `bit_depth` - Bit depth of source YUV planes
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input RGB data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv420_p16_to_rgb(
    y_plane: &[u16],
    y_stride: u32,
    u_plane: &[u16],
    u_stride: u32,
    v_plane: &[u16],
    v_stride: u32,
    rgb: &mut [u8],
    rgb_stride: u32,
    bit_depth: usize,
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
                yuv_p16_to_image_impl::<
                    { YuvSourceChannels::Rgb as u8 },
                    { YuvChromaSample::YUV420 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_p16_to_image_impl::<
                    { YuvSourceChannels::Rgb as u8 },
                    { YuvChromaSample::YUV420 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
        YuvEndianness::LittleEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_p16_to_image_impl::<
                    { YuvSourceChannels::Rgb as u8 },
                    { YuvChromaSample::YUV420 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_p16_to_image_impl::<
                    { YuvSourceChannels::Rgb as u8 },
                    { YuvChromaSample::YUV420 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
    };
    dispatcher(
        y_plane, y_stride, u_plane, u_stride, v_plane, v_stride, rgb, rgb_stride, width, height,
        range, matrix, bit_depth,
    );
}

/// Convert YUV 422 format with 8+ bit pixel format to RGBA format.
///
/// This function takes YUV 422 data with 8+ bit precision stored.
/// and converts it to RGBA format with 8-bit precision per channel.
///
/// # Arguments
///
/// * `y_plane` -  A slice containing Y (luminance) with 8+ bit depth.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `u_plane` - A slice to load the U (chrominance) with 8+ bit depth.
/// * `u_stride` - The stride (bytes per row) for the U plane.
/// * `v_plane` - A slice to load the V (chrominance) with 8+ bit depth.
/// * `v_stride` - The stride (bytes per row) for the U plane
/// * `width` - The width of the YUV image.
/// * `height` - The height of the YUV image.
/// * `rgba_data` - A mutable slice to store the converted RGBA data.
/// * `rgba_stride` - The stride (bytes per row) for RGBA data.
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
pub fn yuv422_p16_to_rgba(
    y_plane: &[u16],
    y_stride: u32,
    u_plane: &[u16],
    u_stride: u32,
    v_plane: &[u16],
    v_stride: u32,
    rgba: &mut [u8],
    rgba_stride: u32,
    bit_depth: usize,
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
                yuv_p16_to_image_impl::<
                    { YuvSourceChannels::Rgba as u8 },
                    { YuvChromaSample::YUV422 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_p16_to_image_impl::<
                    { YuvSourceChannels::Rgba as u8 },
                    { YuvChromaSample::YUV422 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
        YuvEndianness::LittleEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_p16_to_image_impl::<
                    { YuvSourceChannels::Rgba as u8 },
                    { YuvChromaSample::YUV422 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_p16_to_image_impl::<
                    { YuvSourceChannels::Rgba as u8 },
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
        bit_depth,
    );
}

/// Convert YUV 422 format with 8+ bit pixel format to RGB format.
///
/// This function takes YUV 422 data with 8+ bit precision stored.
/// and converts it to RGB format with 8-bit precision per channel.
///
/// # Arguments
///
/// * `y_plane` -  A slice containing Y (luminance) with 8+ bit depth.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `u_plane` - A slice to load the U (chrominance) with 8+ bit depth.
/// * `u_stride` - The stride (bytes per row) for the U plane.
/// * `v_plane` - A slice to load the V (chrominance) with 8+ bit depth.
/// * `v_stride` - The stride (bytes per row) for the U plane
/// * `width` - The width of the YUV image.
/// * `height` - The height of the YUV image.
/// * `rgb_data` - A mutable slice to store the converted RGB data.
/// * `rgb_stride` - The stride (bytes per row) for RGB data.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
/// * `endianness` - The endianness of stored bytes
/// * `bytes_packing` - position of significant bytes ( most significant or least significant ) if it in most significant it should be stated as per Apple *kCVPixelFormatType_422YpCbCr10BiPlanarFullRange/kCVPixelFormatType_422YpCbCr10BiPlanarVideoRange*
/// * `bit_depth` - Bit depth of source YUV planes
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input RGB data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv422_p16_to_rgb(
    y_plane: &[u16],
    y_stride: u32,
    u_plane: &[u16],
    u_stride: u32,
    v_plane: &[u16],
    v_stride: u32,
    rgb: &mut [u8],
    rgb_stride: u32,
    bit_depth: usize,
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
                yuv_p16_to_image_impl::<
                    { YuvSourceChannels::Rgb as u8 },
                    { YuvChromaSample::YUV422 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_p16_to_image_impl::<
                    { YuvSourceChannels::Rgb as u8 },
                    { YuvChromaSample::YUV422 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
        YuvEndianness::LittleEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_p16_to_image_impl::<
                    { YuvSourceChannels::Rgb as u8 },
                    { YuvChromaSample::YUV422 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_p16_to_image_impl::<
                    { YuvSourceChannels::Rgb as u8 },
                    { YuvChromaSample::YUV422 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
    };
    dispatcher(
        y_plane, y_stride, u_plane, u_stride, v_plane, v_stride, rgb, rgb_stride, width, height,
        range, matrix, bit_depth,
    );
}

/// Convert YUV 444 planar format with 8+ bit pixel format to RGBA format.
///
/// This function takes YUV 444 planar data with 8+ bit precision.
/// and converts it to RGBA format with 8-bit precision per channel
///
/// # Arguments
///
/// * `y_plane` -  A slice containing Y (luminance) with 8+ bit depth.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `u_plane` - A slice to load the U (chrominance) with 8+ bit depth.
/// * `u_stride` - The stride (bytes per row) for the U plane.
/// * `v_plane` - A slice to load the V (chrominance) with 8+ bit depth.
/// * `v_stride` - The stride (bytes per row) for the U plane.
/// * `width` - The width of the YUV image.
/// * `height` - The height of the YUV image.
/// * `rgba_data` - A mutable slice to store the converted RGBA data.
/// * `rgba_stride` - The stride (bytes per row) for RGBA data.
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
pub fn yuv444_p16_to_rgba(
    y_plane: &[u16],
    y_stride: u32,
    u_plane: &[u16],
    u_stride: u32,
    v_plane: &[u16],
    v_stride: u32,
    rgba: &mut [u8],
    rgba_stride: u32,
    bit_depth: usize,
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
                yuv_p16_to_image_impl::<
                    { YuvSourceChannels::Rgba as u8 },
                    { YuvChromaSample::YUV444 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_p16_to_image_impl::<
                    { YuvSourceChannels::Rgba as u8 },
                    { YuvChromaSample::YUV444 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
        YuvEndianness::LittleEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_p16_to_image_impl::<
                    { YuvSourceChannels::Rgba as u8 },
                    { YuvChromaSample::YUV444 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_p16_to_image_impl::<
                    { YuvSourceChannels::Rgba as u8 },
                    { YuvChromaSample::YUV444 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
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
        bit_depth,
    );
}

/// Convert YUV 444 planar format with 8+ bit pixel format to RGB format.
///
/// This function takes YUV 444 planar data with 8+ bit precision.
/// and converts it to RGB format with 8-bit precision per channel
///
/// # Arguments
///
/// * `y_plane` -  A slice containing Y (luminance) with 8+ bit depth.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `u_plane` - A slice to load the U (chrominance) with 8+ bit depth.
/// * `u_stride` - The stride (bytes per row) for the U plane.
/// * `v_plane` - A slice to load the V (chrominance) with 8+ bit depth.
/// * `v_stride` - The stride (bytes per row) for the U plane.
/// * `width` - The width of the YUV image.
/// * `height` - The height of the YUV image.
/// * `rgb_data` - A mutable slice to store the converted RGB data.
/// * `rgb_stride` - The stride (bytes per row) for RGB data.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
/// * `endianness` - The endianness of stored bytes
/// * `bytes_packing` - position of significant bytes ( most significant or least significant ) if it in most significant it should be stated as per Apple *kCVPixelFormatType_422YpCbCr10BiPlanarFullRange/kCVPixelFormatType_422YpCbCr10BiPlanarVideoRange*
/// * `bit_depth` - Bit depth of source YUV planes
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input RGB data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv444_p16_to_rgb(
    y_plane: &[u16],
    y_stride: u32,
    u_plane: &[u16],
    u_stride: u32,
    v_plane: &[u16],
    v_stride: u32,
    rgb: &mut [u8],
    rgb_stride: u32,
    bit_depth: usize,
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
                yuv_p16_to_image_impl::<
                    { YuvSourceChannels::Rgb as u8 },
                    { YuvChromaSample::YUV444 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_p16_to_image_impl::<
                    { YuvSourceChannels::Rgb as u8 },
                    { YuvChromaSample::YUV444 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
        YuvEndianness::LittleEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_p16_to_image_impl::<
                    { YuvSourceChannels::Rgb as u8 },
                    { YuvChromaSample::YUV444 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_p16_to_image_impl::<
                    { YuvSourceChannels::Rgb as u8 },
                    { YuvChromaSample::YUV444 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
    };
    dispatcher(
        y_plane, y_stride, u_plane, u_stride, v_plane, v_stride, rgb, rgb_stride, width, height,
        range, matrix, bit_depth,
    );
}

/// Convert YUV 444 planar format with 8+ bit pixel format to BGRA format.
///
/// This function takes YUV 444 planar data with 8+ bit precision.
/// and converts it to BGRA format with 8-bit precision per channel
///
/// # Arguments
///
/// * `y_plane` -  A slice containing Y (luminance) with 8+ bit depth.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `u_plane` - A slice to load the U (chrominance) with 8+ bit depth.
/// * `u_stride` - The stride (bytes per row) for the U plane.
/// * `v_plane` - A slice to load the V (chrominance) with 8+ bit depth.
/// * `v_stride` - The stride (bytes per row) for the U plane.
/// * `width` - The width of the YUV image.
/// * `height` - The height of the YUV image.
/// * `bgra` - A mutable slice to store the converted BGRA data.
/// * `bgra_stride` - The stride (bytes per row) for BGRA data.
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
pub fn yuv444_p16_to_bgra(
    y_plane: &[u16],
    y_stride: u32,
    u_plane: &[u16],
    u_stride: u32,
    v_plane: &[u16],
    v_stride: u32,
    bgra: &mut [u8],
    bgra_stride: u32,
    bit_depth: usize,
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
                yuv_p16_to_image_impl::<
                    { YuvSourceChannels::Bgra as u8 },
                    { YuvChromaSample::YUV444 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_p16_to_image_impl::<
                    { YuvSourceChannels::Bgra as u8 },
                    { YuvChromaSample::YUV444 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
        YuvEndianness::LittleEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_p16_to_image_impl::<
                    { YuvSourceChannels::Bgra as u8 },
                    { YuvChromaSample::YUV444 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_p16_to_image_impl::<
                    { YuvSourceChannels::Bgra as u8 },
                    { YuvChromaSample::YUV444 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
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
        bit_depth,
    );
}

/// Convert YUV 444 planar format with 8+ bit pixel format to BGR format.
///
/// This function takes YUV 444 planar data with 8+ bit precision.
/// and converts it to BGR format with 8-bit precision per channel
///
/// # Arguments
///
/// * `y_plane` -  A slice containing Y (luminance) with 8+ bit depth.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `u_plane` - A slice to load the U (chrominance) with 8+ bit depth.
/// * `u_stride` - The stride (bytes per row) for the U plane.
/// * `v_plane` - A slice to load the V (chrominance) with 8+ bit depth.
/// * `v_stride` - The stride (bytes per row) for the U plane.
/// * `width` - The width of the YUV image.
/// * `height` - The height of the YUV image.
/// * `bgr` - A mutable slice to store the converted BGR data.
/// * `bgr_stride` - The stride (bytes per row) for BGR data.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
/// * `endianness` - The endianness of stored bytes
/// * `bytes_packing` - position of significant bytes ( most significant or least significant ) if it in most significant it should be stated as per Apple *kCVPixelFormatType_422YpCbCr10BiPlanarFullRange/kCVPixelFormatType_422YpCbCr10BiPlanarVideoRange*
/// * `bit_depth` - Bit depth of source YUV planes
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGR data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv444_p16_to_bgr(
    y_plane: &[u16],
    y_stride: u32,
    u_plane: &[u16],
    u_stride: u32,
    v_plane: &[u16],
    v_stride: u32,
    bgr: &mut [u8],
    bgr_stride: u32,
    bit_depth: usize,
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
                yuv_p16_to_image_impl::<
                    { YuvSourceChannels::Bgr as u8 },
                    { YuvChromaSample::YUV444 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_p16_to_image_impl::<
                    { YuvSourceChannels::Bgr as u8 },
                    { YuvChromaSample::YUV444 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
        YuvEndianness::LittleEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_p16_to_image_impl::<
                    { YuvSourceChannels::Bgr as u8 },
                    { YuvChromaSample::YUV444 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_p16_to_image_impl::<
                    { YuvSourceChannels::Bgr as u8 },
                    { YuvChromaSample::YUV444 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
    };
    dispatcher(
        y_plane, y_stride, u_plane, u_stride, v_plane, v_stride, bgr, bgr_stride, width, height,
        range, matrix, bit_depth,
    );
}
