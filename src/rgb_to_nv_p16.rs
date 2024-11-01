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
use crate::yuv_support::{
    get_forward_transform, get_yuv_range, ToIntegerTransform, YuvChromaSample, YuvNVOrder,
    YuvSourceChannels,
};
use crate::{YuvBytesPacking, YuvEndianness, YuvRange, YuvStandardMatrix};

#[inline(always)]
fn transform_integer<const ENDIANNESS: u8, const BYTES_POSITION: u8, const BIT_DEPTH: u8>(
    v: i32,
) -> u16 {
    let endianness: YuvEndianness = ENDIANNESS.into();
    let bytes_position: YuvBytesPacking = BYTES_POSITION.into();
    let packing: i32 = 16 - BIT_DEPTH as i32;
    let packed_bytes = match bytes_position {
        YuvBytesPacking::MostSignificantBytes => v << packing,
        YuvBytesPacking::LeastSignificantBytes => v,
    } as u16;
    match endianness {
        YuvEndianness::BigEndian => packed_bytes.to_be(),
        YuvEndianness::LittleEndian => packed_bytes.to_le(),
    }
}

fn rgbx_to_yuv_bi_planar_10_impl<
    const ORIGIN_CHANNELS: u8,
    const NV_ORDER: u8,
    const SAMPLING: u8,
    const ENDIANNESS: u8,
    const BYTES_POSITION: u8,
    const BIT_DEPTH: u8,
>(
    y_plane: &mut [u16],
    y_stride: u32,
    uv_plane: &mut [u16],
    uv_stride: u32,
    rgba: &[u16],
    rgba_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) {
    let nv_order: YuvNVOrder = NV_ORDER.into();
    let chroma_subsampling: YuvChromaSample = SAMPLING.into();
    let src_chans: YuvSourceChannels = ORIGIN_CHANNELS.into();
    let channels = src_chans.get_channels_count();
    let range = get_yuv_range(BIT_DEPTH as u32, range);
    let kr_kb = matrix.get_kr_kb();
    let max_range = (1u32 << BIT_DEPTH as u32) - 1u32;
    let transform_precise =
        get_forward_transform(max_range, range.range_y, range.range_uv, kr_kb.kr, kr_kb.kb);
    let transform = transform_precise.to_integers(8);
    const PRECISION: i32 = 8;
    const ROUNDING_CONST_BIAS: i32 = 1 << (PRECISION - 1);
    let bias_y = range.bias_y as i32 * (1 << PRECISION) + ROUNDING_CONST_BIAS;
    let bias_uv = range.bias_uv as i32 * (1 << PRECISION) + ROUNDING_CONST_BIAS;

    let iterator_step = match chroma_subsampling {
        YuvChromaSample::YUV420 => 2usize,
        YuvChromaSample::YUV422 => 2usize,
        YuvChromaSample::YUV444 => 1usize,
    };

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    let mut _use_sse = std::arch::is_x86_feature_detected!("sse4.1");

    let mut y_offset = 0usize;
    let mut uv_offset = 0usize;
    let mut rgba_offset = 0usize;

    let y_dst_ptr = y_plane.as_mut_ptr() as *mut u8;
    let uv_dst_ptr = uv_plane.as_mut_ptr() as *mut u8;
    let rgb_src_ptr = rgba.as_ptr() as *const u8;

    for y in 0..height as usize {
        let mut _cx = 0usize;
        let mut _ux = 0usize;

        let compute_uv_row = chroma_subsampling == YuvChromaSample::YUV444
            || chroma_subsampling == YuvChromaSample::YUV422
            || y & 1 == 0;

        let y_st_ptr = unsafe { y_dst_ptr.add(y_offset) as *mut u16 };
        let uv_st_ptr = unsafe { uv_dst_ptr.add(uv_offset) as *mut u16 };
        let rgb_ld_ptr = unsafe { rgb_src_ptr.add(rgba_offset) as *const u16 };

        for x in (_cx..width as usize).step_by(iterator_step) {
            let px = x * channels;
            let src = unsafe { rgb_ld_ptr.add(px) };
            let r0 = unsafe { src.add(src_chans.get_r_channel_offset()).read_unaligned() } as i32;
            let g0 = unsafe { src.add(src_chans.get_g_channel_offset()).read_unaligned() } as i32;
            let b0 = unsafe { src.add(src_chans.get_b_channel_offset()).read_unaligned() } as i32;

            let mut r1 = r0;
            let mut g1 = g0;
            let mut b1 = b0;

            let y_0 =
                (r0 * transform.yr + g0 * transform.yg + b0 * transform.yb + bias_y) >> PRECISION;
            unsafe {
                y_st_ptr.add(x).write_unaligned(transform_integer::<
                    ENDIANNESS,
                    BYTES_POSITION,
                    BIT_DEPTH,
                >(y_0));
            }
            match chroma_subsampling {
                YuvChromaSample::YUV420 | YuvChromaSample::YUV422 => {
                    if x + 1 < width as usize {
                        let next_px = (x + 1) * channels;
                        let src = unsafe { rgb_ld_ptr.add(next_px) };
                        r1 = unsafe { src.add(src_chans.get_r_channel_offset()).read_unaligned() }
                            as i32;
                        g1 = unsafe { src.add(src_chans.get_g_channel_offset()).read_unaligned() }
                            as i32;
                        b1 = unsafe { src.add(src_chans.get_b_channel_offset()).read_unaligned() }
                            as i32;
                        let y_1 =
                            (r1 * transform.yr + g1 * transform.yg + b1 * transform.yb + bias_y)
                                >> PRECISION;
                        unsafe {
                            y_st_ptr.add(x + 1).write_unaligned(transform_integer::<
                                ENDIANNESS,
                                BYTES_POSITION,
                                BIT_DEPTH,
                            >(y_1));
                        }
                    }
                }
                _ => {}
            }

            if compute_uv_row {
                let r = if chroma_subsampling == YuvChromaSample::YUV444 {
                    r0
                } else {
                    (r0 + r1 + 1) >> 1
                };
                let g = if chroma_subsampling == YuvChromaSample::YUV444 {
                    g0
                } else {
                    (g0 + g1 + 1) >> 1
                };
                let b = if chroma_subsampling == YuvChromaSample::YUV444 {
                    b0
                } else {
                    (b0 + b1 + 1) >> 1
                };
                let cb = (r * transform.cb_r + g * transform.cb_g + b * transform.cb_b + bias_uv)
                    >> PRECISION;
                let cr = (r * transform.cr_r + g * transform.cr_g + b * transform.cr_b + bias_uv)
                    >> PRECISION;
                let u_pos = match chroma_subsampling {
                    YuvChromaSample::YUV420 | YuvChromaSample::YUV422 => _ux,
                    YuvChromaSample::YUV444 => _ux,
                };
                unsafe {
                    let dst_ptr = uv_st_ptr.add(u_pos);
                    dst_ptr
                        .add(nv_order.get_u_position())
                        .write_unaligned(
                            transform_integer::<ENDIANNESS, BYTES_POSITION, BIT_DEPTH>(cb),
                        );
                    dst_ptr
                        .add(nv_order.get_v_position())
                        .write_unaligned(
                            transform_integer::<ENDIANNESS, BYTES_POSITION, BIT_DEPTH>(cr),
                        );
                }
            }

            _ux += 2;
        }

        y_offset += y_stride as usize;
        rgba_offset += rgba_stride as usize;
        match chroma_subsampling {
            YuvChromaSample::YUV420 => {
                if y & 1 == 1 {
                    uv_offset += uv_stride as usize;
                }
            }
            YuvChromaSample::YUV444 | YuvChromaSample::YUV422 => {
                uv_offset += uv_stride as usize;
            }
        }
    }
}

fn rgbx_to_yuv_bi_planar_10<
    const ORIGIN_CHANNELS: u8,
    const NV_ORDER: u8,
    const SAMPLING: u8,
    const ENDIANNESS: u8,
    const BYTES_POSITION: u8,
>(
    y_plane: &mut [u16],
    y_stride: u32,
    uv_plane: &mut [u16],
    uv_stride: u32,
    rgba: &[u16],
    rgba_stride: u32,
    bit_depth: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) {
    if bit_depth == 10 {
        rgbx_to_yuv_bi_planar_10_impl::<
            ORIGIN_CHANNELS,
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
            rgba,
            rgba_stride,
            width,
            height,
            range,
            matrix,
        );
    } else if bit_depth == 12 {
        rgbx_to_yuv_bi_planar_10_impl::<
            ORIGIN_CHANNELS,
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
            rgba,
            rgba_stride,
            width,
            height,
            range,
            matrix,
        );
    } else {
        panic!("Bit depth {} is not implemented", bit_depth);
    }
}

/// Convert RGB image data to YUV 420 bi-planar (NV12 10-bit) format with 10 or 12 bit depth.
///
/// This function performs RGB to YUV conversion and stores the result in YUV420 bi-planar format,
/// with separate planes for Y (luminance), UV (chrominance) components.
///
/// # Arguments
///
/// * `y_plane` - A mutable slice to store the Y (luminance) plane data.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `uv_plane` - A mutable slice to store the UV (chrominance) plane data.
/// * `uv_stride` - The stride (bytes per row) for the UV plane.
/// * `bgr` - The input RGB image data slice.
/// * `bgr_stride` - The stride (bytes per row) for the RGB image data.
/// * `bit_depth` - Only 10 or 12 bit-depth is supported
/// * `width` - The width of the image in pixels.
/// * `height` - The height of the image in pixels.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
/// * `endianness` - The endianness of final YUV
/// * `bytes_packing` - position of significant bytes for YUV ( most significant or least significant ) if it in most significant it should be stated as per Apple *kCVPixelFormatType_422YpCbCr10BiPlanarFullRange/kCVPixelFormatType_422YpCbCr10BiPlanarVideoRange*
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input RGB data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn rgb_to_yuv_nv12_p16(
    y_plane: &mut [u16],
    y_stride: u32,
    uv_plane: &mut [u16],
    uv_stride: u32,
    rgb: &[u16],
    rgb_stride: u32,
    bit_depth: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    endianness: YuvEndianness,
    bytes_packing: YuvBytesPacking,
) {
    if bit_depth != 10 && bit_depth != 12 {
        panic!("Only 10 and 12 bit depth is supported");
    }
    let dispatcher = match endianness {
        YuvEndianness::BigEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                rgbx_to_yuv_bi_planar_10::<
                    { YuvSourceChannels::Rgb as u8 },
                    { YuvNVOrder::UV as u8 },
                    { YuvChromaSample::YUV420 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                rgbx_to_yuv_bi_planar_10::<
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
                rgbx_to_yuv_bi_planar_10::<
                    { YuvSourceChannels::Rgb as u8 },
                    { YuvNVOrder::UV as u8 },
                    { YuvChromaSample::YUV420 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                rgbx_to_yuv_bi_planar_10::<
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

/// Convert RGB image data to YUV 420 bi-planar (NV21 10-bit) format with 10 or 12 bit depth.
///
/// This function performs RGB to YUV conversion and stores the result in YUV420 bi-planar format,
/// with separate planes for Y (luminance), VU (chrominance) components.
///
/// # Arguments
///
/// * `y_plane` - A mutable slice to store the Y (luminance) plane data.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `uv_plane` - A mutable slice to store the UV (chrominance) plane data.
/// * `uv_stride` - The stride (bytes per row) for the VU plane.
/// * `bgr` - The input RGB image data slice.
/// * `bgr_stride` - The stride (bytes per row) for the RGB image data.
/// * `bit_depth` - Only 10 or 12 bit-depth is supported
/// * `width` - The width of the image in pixels.
/// * `height` - The height of the image in pixels.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
/// * `endianness` - The endianness of final YUV
/// * `bytes_packing` - position of significant bytes for YUV ( most significant or least significant ) if it in most significant it should be stated as per Apple *kCVPixelFormatType_422YpCbCr10BiPlanarFullRange/kCVPixelFormatType_422YpCbCr10BiPlanarVideoRange*
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input RGB data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn rgb_to_yuv_nv21_p16(
    y_plane: &mut [u16],
    y_stride: u32,
    uv_plane: &mut [u16],
    uv_stride: u32,
    rgb: &[u16],
    rgb_stride: u32,
    bit_depth: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    endianness: YuvEndianness,
    bytes_packing: YuvBytesPacking,
) {
    if bit_depth != 10 && bit_depth != 12 {
        panic!("Only 10 and 12 bit depth is supported");
    }
    let dispatcher = match endianness {
        YuvEndianness::BigEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                rgbx_to_yuv_bi_planar_10::<
                    { YuvSourceChannels::Rgb as u8 },
                    { YuvNVOrder::VU as u8 },
                    { YuvChromaSample::YUV420 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                rgbx_to_yuv_bi_planar_10::<
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
                rgbx_to_yuv_bi_planar_10::<
                    { YuvSourceChannels::Rgb as u8 },
                    { YuvNVOrder::VU as u8 },
                    { YuvChromaSample::YUV420 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                rgbx_to_yuv_bi_planar_10::<
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

/// Convert RGBA image data to YUV 420 bi-planar (NV12 10-bit) format with 10 or 12 bit depth.
///
/// This function performs RGBA to YUV conversion and stores the result in YUV422 bi-planar format,
/// with separate planes for Y (luminance), UV (chrominance) components.
///
/// # Arguments
///
/// * `y_plane` - A mutable slice to store the Y (luminance) plane data.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `uv_plane` - A mutable slice to store the UV (chrominance) plane data.
/// * `uv_stride` - The stride (bytes per row) for the UV plane.
/// * `rgba` - The input RGBA image data slice.
/// * `rgba_stride` - The stride (bytes per row) for the RGBA image data.
/// * `bit_depth` - Only 10 or 12 bit-depth is supported
/// * `width` - The width of the image in pixels.
/// * `height` - The height of the image in pixels.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
/// * `endianness` - The endianness of final YUV
/// * `bytes_packing` - position of significant bytes for YUV ( most significant or least significant ) if it in most significant it should be stated as per Apple *kCVPixelFormatType_422YpCbCr10BiPlanarFullRange/kCVPixelFormatType_422YpCbCr10BiPlanarVideoRange*
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input RGB data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn rgba_to_yuv_nv12_p16(
    y_plane: &mut [u16],
    y_stride: u32,
    uv_plane: &mut [u16],
    uv_stride: u32,
    rgba: &[u16],
    rgba_stride: u32,
    bit_depth: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    endianness: YuvEndianness,
    bytes_packing: YuvBytesPacking,
) {
    if bit_depth != 10 && bit_depth != 12 {
        panic!("Only 10 and 12 bit depth is supported");
    }
    let dispatcher = match endianness {
        YuvEndianness::BigEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                rgbx_to_yuv_bi_planar_10::<
                    { YuvSourceChannels::Rgba as u8 },
                    { YuvNVOrder::UV as u8 },
                    { YuvChromaSample::YUV420 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                rgbx_to_yuv_bi_planar_10::<
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
                rgbx_to_yuv_bi_planar_10::<
                    { YuvSourceChannels::Rgba as u8 },
                    { YuvNVOrder::UV as u8 },
                    { YuvChromaSample::YUV420 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                rgbx_to_yuv_bi_planar_10::<
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

/// Convert RGBA image data to YUV 420 bi-planar (NV21 10-bit) format with 10 or 12 bit depth.
///
/// This function performs RGBA to YUV conversion and stores the result in YUV422 bi-planar format,
/// with separate planes for Y (luminance), UV (chrominance) components.
///
/// # Arguments
///
/// * `y_plane` - A mutable slice to store the Y (luminance) plane data.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `uv_plane` - A mutable slice to store the UV (chrominance) plane data.
/// * `uv_stride` - The stride (bytes per row) for the VU plane
/// * `rgba` - The input RGBA image data slice.
/// * `rgba_stride` - The stride (bytes per row) for the RGBA image data.
/// * `bit_depth` - Only 10 or 12 bit-depth is supported
/// * `width` - The width of the image in pixels.
/// * `height` - The height of the image in pixels.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
/// * `endianness` - The endianness of final YUV
/// * `bytes_packing` - position of significant bytes for YUV ( most significant or least significant ) if it in most significant it should be stated as per Apple *kCVPixelFormatType_422YpCbCr10BiPlanarFullRange/kCVPixelFormatType_422YpCbCr10BiPlanarVideoRange*
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input RGB data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn rgba_to_yuv_nv21_p16(
    y_plane: &mut [u16],
    y_stride: u32,
    uv_plane: &mut [u16],
    uv_stride: u32,
    rgba: &[u16],
    rgba_stride: u32,
    bit_depth: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    endianness: YuvEndianness,
    bytes_packing: YuvBytesPacking,
) {
    if bit_depth != 10 && bit_depth != 12 {
        panic!("Only 10 and 12 bit depth is supported");
    }
    let dispatcher = match endianness {
        YuvEndianness::BigEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                rgbx_to_yuv_bi_planar_10::<
                    { YuvSourceChannels::Rgba as u8 },
                    { YuvNVOrder::VU as u8 },
                    { YuvChromaSample::YUV420 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                rgbx_to_yuv_bi_planar_10::<
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
                rgbx_to_yuv_bi_planar_10::<
                    { YuvSourceChannels::Rgba as u8 },
                    { YuvNVOrder::VU as u8 },
                    { YuvChromaSample::YUV420 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                rgbx_to_yuv_bi_planar_10::<
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

/// Convert BGR image data to YUV 420 bi-planar (NV12 10-bit) format with 10 or 12 bit depth.
///
/// This function performs BGR to YUV conversion and stores the result in YUV420 bi-planar format,
/// with separate planes for Y (luminance), UV (chrominance) components.
///
/// # Arguments
///
/// * `y_plane` - A mutable slice to store the Y (luminance) plane data.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `uv_plane` - A mutable slice to store the UV (chrominance) plane data.
/// * `uv_stride` - The stride (bytes per row) for the UV plane.
/// * `bgr` - The input BGR image data slice.
/// * `bgr_stride` - The stride (bytes per row) for the BGR image data.
/// * `bit_depth` - Only 10 or 12 bit-depth is supported
/// * `width` - The width of the image in pixels.
/// * `height` - The height of the image in pixels.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
/// * `endianness` - The endianness of final YUV
/// * `bytes_packing` - position of significant bytes for YUV ( most significant or least significant ) if it in most significant it should be stated as per Apple *kCVPixelFormatType_422YpCbCr10BiPlanarFullRange/kCVPixelFormatType_422YpCbCr10BiPlanarVideoRange*
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGR data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn bgr_to_yuv_nv12_p16(
    y_plane: &mut [u16],
    y_stride: u32,
    uv_plane: &mut [u16],
    uv_stride: u32,
    bgr: &[u16],
    bgr_stride: u32,
    bit_depth: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    endianness: YuvEndianness,
    bytes_packing: YuvBytesPacking,
) {
    if bit_depth != 10 && bit_depth != 12 {
        panic!("Only 10 and 12 bit depth is supported");
    }
    let dispatcher = match endianness {
        YuvEndianness::BigEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                rgbx_to_yuv_bi_planar_10::<
                    { YuvSourceChannels::Bgr as u8 },
                    { YuvNVOrder::UV as u8 },
                    { YuvChromaSample::YUV420 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                rgbx_to_yuv_bi_planar_10::<
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
                rgbx_to_yuv_bi_planar_10::<
                    { YuvSourceChannels::Bgr as u8 },
                    { YuvNVOrder::UV as u8 },
                    { YuvChromaSample::YUV420 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                rgbx_to_yuv_bi_planar_10::<
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

/// Convert BGR image data to YUV 420 bi-planar (NV21 10-bit) format with 10 or 12 bit depth.
///
/// This function performs BGR to YUV conversion and stores the result in YUV420 bi-planar format,
/// with separate planes for Y (luminance), VU (chrominance) components.
///
/// # Arguments
///
/// * `y_plane` - A mutable slice to store the Y (luminance) plane data.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `uv_plane` - A mutable slice to store the UV (chrominance) plane data.
/// * `uv_stride` - The stride (bytes per row) for the VU plane.
/// * `bgr` - The input BGR image data slice.
/// * `bgr_stride` - The stride (bytes per row) for the BGR image data.
/// * `bit_depth` - Only 10 or 12 bit-depth is supported
/// * `width` - The width of the image in pixels.
/// * `height` - The height of the image in pixels.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
/// * `endianness` - The endianness of final YUV
/// * `bytes_packing` - position of significant bytes for YUV ( most significant or least significant ) if it in most significant it should be stated as per Apple *kCVPixelFormatType_422YpCbCr10BiPlanarFullRange/kCVPixelFormatType_422YpCbCr10BiPlanarVideoRange*
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGR data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn bgr_to_yuv_nv21_p16(
    y_plane: &mut [u16],
    y_stride: u32,
    uv_plane: &mut [u16],
    uv_stride: u32,
    bgr: &[u16],
    bgr_stride: u32,
    bit_depth: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    endianness: YuvEndianness,
    bytes_packing: YuvBytesPacking,
) {
    if bit_depth != 10 && bit_depth != 12 {
        panic!("Only 10 and 12 bit depth is supported");
    }
    let dispatcher = match endianness {
        YuvEndianness::BigEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                rgbx_to_yuv_bi_planar_10::<
                    { YuvSourceChannels::Bgr as u8 },
                    { YuvNVOrder::VU as u8 },
                    { YuvChromaSample::YUV420 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                rgbx_to_yuv_bi_planar_10::<
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
                rgbx_to_yuv_bi_planar_10::<
                    { YuvSourceChannels::Bgr as u8 },
                    { YuvNVOrder::VU as u8 },
                    { YuvChromaSample::YUV420 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                rgbx_to_yuv_bi_planar_10::<
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

/// Convert BGRA image data to YUV 420 bi-planar (NV12 10-bit) format with 10 or 12 bit depth.
///
/// This function performs BGRA to YUV conversion and stores the result in YUV420 bi-planar format,
/// with separate planes for Y (luminance), UV (chrominance) components.
///
/// # Arguments
///
/// * `y_plane` - A mutable slice to store the Y (luminance) plane data.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `uv_plane` - A mutable slice to store the UV (chrominance) plane data.
/// * `uv_stride` - The stride (bytes per row) for the UV plane.
/// * `bgra` - The input BGRA image data slice.
/// * `bgra_stride` - The stride (bytes per row) for the BGRA image data.
/// * `bit_depth` - Only 10 or 12 bit-depth is supported
/// * `width` - The width of the image in pixels.
/// * `height` - The height of the image in pixels.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
/// * `endianness` - The endianness of final YUV
/// * `bytes_packing` - position of significant bytes for YUV ( most significant or least significant ) if it in most significant it should be stated as per Apple *kCVPixelFormatType_422YpCbCr10BiPlanarFullRange/kCVPixelFormatType_422YpCbCr10BiPlanarVideoRange*
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGRA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn bgra_to_yuv_nv12_p16(
    y_plane: &mut [u16],
    y_stride: u32,
    uv_plane: &mut [u16],
    uv_stride: u32,
    bgra: &[u16],
    bgra_stride: u32,
    bit_depth: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    endianness: YuvEndianness,
    bytes_packing: YuvBytesPacking,
) {
    if bit_depth != 10 && bit_depth != 12 {
        panic!("Only 10 and 12 bit depth is supported");
    }
    let dispatcher = match endianness {
        YuvEndianness::BigEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                rgbx_to_yuv_bi_planar_10::<
                    { YuvSourceChannels::Bgra as u8 },
                    { YuvNVOrder::UV as u8 },
                    { YuvChromaSample::YUV420 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                rgbx_to_yuv_bi_planar_10::<
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
                rgbx_to_yuv_bi_planar_10::<
                    { YuvSourceChannels::Bgra as u8 },
                    { YuvNVOrder::UV as u8 },
                    { YuvChromaSample::YUV420 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                rgbx_to_yuv_bi_planar_10::<
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

/// Convert BGRA image data to YUV 420 bi-planar (NV21 10-bit) format with 10 or 12 bit depth.
///
/// This function performs BGRA to YUV conversion and stores the result in YUV420 bi-planar format,
/// with separate planes for Y (luminance), VU (chrominance) components.
///
/// # Arguments
///
/// * `y_plane` - A mutable slice to store the Y (luminance) plane data.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `uv_plane` - A mutable slice to store the UV (chrominance) plane data.
/// * `uv_stride` - The stride (bytes per row) for the VU plane.
/// * `bgra` - The input BGRA image data slice.
/// * `bgra_stride` - The stride (bytes per row) for the BGRA image data.
/// * `bit_depth` - Only 10 or 12 bit-depth is supported
/// * `width` - The width of the image in pixels.
/// * `height` - The height of the image in pixels.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
/// * `endianness` - The endianness of final YUV
/// * `bytes_packing` - position of significant bytes for YUV ( most significant or least significant ) if it in most significant it should be stated as per Apple *kCVPixelFormatType_422YpCbCr10BiPlanarFullRange/kCVPixelFormatType_422YpCbCr10BiPlanarVideoRange*
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGRA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn bgra_to_yuv_nv21_p16(
    y_plane: &mut [u16],
    y_stride: u32,
    uv_plane: &mut [u16],
    uv_stride: u32,
    bgra: &[u16],
    bgra_stride: u32,
    bit_depth: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    endianness: YuvEndianness,
    bytes_packing: YuvBytesPacking,
) {
    if bit_depth != 10 && bit_depth != 12 {
        panic!("Only 10 and 12 bit depth is supported");
    }
    let dispatcher = match endianness {
        YuvEndianness::BigEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                rgbx_to_yuv_bi_planar_10::<
                    { YuvSourceChannels::Bgra as u8 },
                    { YuvNVOrder::VU as u8 },
                    { YuvChromaSample::YUV420 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                rgbx_to_yuv_bi_planar_10::<
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
                rgbx_to_yuv_bi_planar_10::<
                    { YuvSourceChannels::Bgra as u8 },
                    { YuvNVOrder::VU as u8 },
                    { YuvChromaSample::YUV420 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                rgbx_to_yuv_bi_planar_10::<
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

/// Convert BGR image data to YUV 422 bi-planar (NV16 10-bit) format with 10 or 12 bit depth.
///
/// This function performs BGR to YUV conversion and stores the result in YUV422 bi-planar format,
/// with separate planes for Y (luminance), UV (chrominance) components.
///
/// # Arguments
///
/// * `y_plane` - A mutable slice to store the Y (luminance) plane data.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `uv_plane` - A mutable slice to store the UV (chrominance) plane data.
/// * `uv_stride` - The stride (bytes per row) for the UV plane.
/// * `bgr` - The input BGR image data slice.
/// * `bgr_stride` - The stride (bytes per row) for the BGR image data.
/// * `bit_depth` - Only 10 or 12 bit-depth is supported
/// * `width` - The width of the image in pixels.
/// * `height` - The height of the image in pixels.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
/// * `endianness` - The endianness of final YUV
/// * `bytes_packing` - position of significant bytes for YUV ( most significant or least significant ) if it in most significant it should be stated as per Apple *kCVPixelFormatType_422YpCbCr10BiPlanarFullRange/kCVPixelFormatType_422YpCbCr10BiPlanarVideoRange*
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGR data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn bgr_to_yuv_nv16_p16(
    y_plane: &mut [u16],
    y_stride: u32,
    uv_plane: &mut [u16],
    uv_stride: u32,
    bgr: &[u16],
    bgr_stride: u32,
    bit_depth: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    endianness: YuvEndianness,
    bytes_packing: YuvBytesPacking,
) {
    if bit_depth != 10 && bit_depth != 12 {
        panic!("Only 10 and 12 bit depth is supported");
    }
    let dispatcher = match endianness {
        YuvEndianness::BigEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                rgbx_to_yuv_bi_planar_10::<
                    { YuvSourceChannels::Bgr as u8 },
                    { YuvNVOrder::UV as u8 },
                    { YuvChromaSample::YUV422 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                rgbx_to_yuv_bi_planar_10::<
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
                rgbx_to_yuv_bi_planar_10::<
                    { YuvSourceChannels::Bgr as u8 },
                    { YuvNVOrder::UV as u8 },
                    { YuvChromaSample::YUV422 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                rgbx_to_yuv_bi_planar_10::<
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
        y_plane, y_stride, uv_plane, uv_stride, bgr, bgr_stride, bit_depth, width, height, range,
        matrix,
    );
}

/// Convert BGR image data to YUV 422 bi-planar (NV61 10-bit) format with 10 or 12 bit depth.
///
/// This function performs BGR to YUV conversion and stores the result in YUV422 bi-planar format,
/// with separate planes for Y (luminance), VU (chrominance) components.
///
/// # Arguments
///
/// * `y_plane` - A mutable slice to store the Y (luminance) plane data.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `uv_plane` - A mutable slice to store the UV (chrominance) plane data.
/// * `uv_stride` - The stride (bytes per row) for the VU plane.
/// * `bgr` - The input BGR image data slice.
/// * `bgr_stride` - The stride (bytes per row) for the BGR image data.
/// * `bit_depth` - Only 10 or 12 bit-depth is supported
/// * `width` - The width of the image in pixels.
/// * `height` - The height of the image in pixels.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
/// * `endianness` - The endianness of final YUV
/// * `bytes_packing` - position of significant bytes for YUV ( most significant or least significant ) if it in most significant it should be stated as per Apple *kCVPixelFormatType_422YpCbCr10BiPlanarFullRange/kCVPixelFormatType_422YpCbCr10BiPlanarVideoRange*
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGR data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn bgr_to_yuv_nv61_p16(
    y_plane: &mut [u16],
    y_stride: u32,
    uv_plane: &mut [u16],
    uv_stride: u32,
    bgr: &[u16],
    bgr_stride: u32,
    bit_depth: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    endianness: YuvEndianness,
    bytes_packing: YuvBytesPacking,
) {
    if bit_depth != 10 && bit_depth != 12 {
        panic!("Only 10 and 12 bit depth is supported");
    }
    let dispatcher = match endianness {
        YuvEndianness::BigEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                rgbx_to_yuv_bi_planar_10::<
                    { YuvSourceChannels::Bgr as u8 },
                    { YuvNVOrder::VU as u8 },
                    { YuvChromaSample::YUV422 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                rgbx_to_yuv_bi_planar_10::<
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
                rgbx_to_yuv_bi_planar_10::<
                    { YuvSourceChannels::Bgr as u8 },
                    { YuvNVOrder::VU as u8 },
                    { YuvChromaSample::YUV422 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                rgbx_to_yuv_bi_planar_10::<
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
        y_plane, y_stride, uv_plane, uv_stride, bgr, bgr_stride, bit_depth, width, height, range,
        matrix,
    );
}

/// Convert RGB image data to YUV 422 bi-planar (NV16 10-bit) format with 10 or 12 bit depth.
///
/// This function performs RGB to YUV conversion and stores the result in YUV422 bi-planar format,
/// with separate planes for Y (luminance), UV (chrominance) components.
///
/// # Arguments
///
/// * `y_plane` - A mutable slice to store the Y (luminance) plane data.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `uv_plane` - A mutable slice to store the UV (chrominance) plane data.
/// * `uv_stride` - The stride (bytes per row) for the UV plane.
/// * `rgb` - The input RGB image data slice.
/// * `rgb_stride` - The stride (bytes per row) for the RGB image data.
/// * `bit_depth` - Only 10 or 12 bit-depth is supported
/// * `width` - The width of the image in pixels.
/// * `height` - The height of the image in pixels.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
/// * `endianness` - The endianness of final YUV
/// * `bytes_packing` - position of significant bytes for YUV ( most significant or least significant ) if it in most significant it should be stated as per Apple *kCVPixelFormatType_422YpCbCr10BiPlanarFullRange/kCVPixelFormatType_422YpCbCr10BiPlanarVideoRange*
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input RGB data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn rgb_to_yuv_nv16_p16(
    y_plane: &mut [u16],
    y_stride: u32,
    uv_plane: &mut [u16],
    uv_stride: u32,
    rgb: &[u16],
    rgb_stride: u32,
    bit_depth: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    endianness: YuvEndianness,
    bytes_packing: YuvBytesPacking,
) {
    if bit_depth != 10 && bit_depth != 12 {
        panic!("Only 10 and 12 bit depth is supported");
    }
    let dispatcher = match endianness {
        YuvEndianness::BigEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                rgbx_to_yuv_bi_planar_10::<
                    { YuvSourceChannels::Rgb as u8 },
                    { YuvNVOrder::UV as u8 },
                    { YuvChromaSample::YUV422 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                rgbx_to_yuv_bi_planar_10::<
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
                rgbx_to_yuv_bi_planar_10::<
                    { YuvSourceChannels::Rgb as u8 },
                    { YuvNVOrder::UV as u8 },
                    { YuvChromaSample::YUV422 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                rgbx_to_yuv_bi_planar_10::<
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
        y_plane, y_stride, uv_plane, uv_stride, rgb, rgb_stride, bit_depth, width, height, range,
        matrix,
    );
}

/// Convert RGB image data to YUV 422 bi-planar (NV61 10-bit) format with 10 or 12 bit depth.
///
/// This function performs RGB to YUV conversion and stores the result in YUV422 bi-planar format,
/// with separate planes for Y (luminance), VU (chrominance) components.
///
/// # Arguments
///
/// * `y_plane` - A mutable slice to store the Y (luminance) plane data.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `uv_plane` - A mutable slice to store the UV (chrominance) plane data.
/// * `uv_stride` - The stride (bytes per row) for the VU plane.
/// * `rgb` - The input RGB image data slice.
/// * `rgb_stride` - The stride (bytes per row) for the RGB image data.
/// * `bit_depth` - Only 10 or 12 bit-depth is supported
/// * `width` - The width of the image in pixels.
/// * `height` - The height of the image in pixels.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
/// * `endianness` - The endianness of final YUV
/// * `bytes_packing` - position of significant bytes for YUV ( most significant or least significant ) if it in most significant it should be stated as per Apple *kCVPixelFormatType_422YpCbCr10BiPlanarFullRange/kCVPixelFormatType_422YpCbCr10BiPlanarVideoRange*
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input RGB data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn rgb_to_yuv_nv61_p16(
    y_plane: &mut [u16],
    y_stride: u32,
    uv_plane: &mut [u16],
    uv_stride: u32,
    rgb: &[u16],
    rgb_stride: u32,
    bit_depth: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    endianness: YuvEndianness,
    bytes_packing: YuvBytesPacking,
) {
    if bit_depth != 10 && bit_depth != 12 {
        panic!("Only 10 and 12 bit depth is supported");
    }
    let dispatcher = match endianness {
        YuvEndianness::BigEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                rgbx_to_yuv_bi_planar_10::<
                    { YuvSourceChannels::Rgb as u8 },
                    { YuvNVOrder::VU as u8 },
                    { YuvChromaSample::YUV422 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                rgbx_to_yuv_bi_planar_10::<
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
                rgbx_to_yuv_bi_planar_10::<
                    { YuvSourceChannels::Rgb as u8 },
                    { YuvNVOrder::VU as u8 },
                    { YuvChromaSample::YUV422 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                rgbx_to_yuv_bi_planar_10::<
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
        y_plane, y_stride, uv_plane, uv_stride, rgb, rgb_stride, bit_depth, width, height, range,
        matrix,
    );
}

/// Convert RGBA image data to YUV 422 bi-planar (NV16 10-bit) format with 10 or 12 bit depth.
///
/// This function performs RGBA to YUV conversion and stores the result in YUV422 bi-planar format,
/// with separate planes for Y (luminance), UV (chrominance) components.
///
/// # Arguments
///
/// * `y_plane` - A mutable slice to store the Y (luminance) plane data.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `uv_plane` - A mutable slice to store the UV (chrominance) plane data.
/// * `uv_stride` - The stride (bytes per row) for the UV plane.
/// * `rgba` - The input RGBA image data slice.
/// * `rgba_stride` - The stride (bytes per row) for the RGBA image data.
/// * `bit_depth` - Only 10 or 12 bit-depth is supported
/// * `width` - The width of the image in pixels.
/// * `height` - The height of the image in pixels.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
/// * `endianness` - The endianness of final YUV
/// * `bytes_packing` - position of significant bytes for YUV ( most significant or least significant ) if it in most significant it should be stated as per Apple *kCVPixelFormatType_422YpCbCr10BiPlanarFullRange/kCVPixelFormatType_422YpCbCr10BiPlanarVideoRange*
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input RGBA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn rgba_to_yuv_nv16_p16(
    y_plane: &mut [u16],
    y_stride: u32,
    uv_plane: &mut [u16],
    uv_stride: u32,
    rgba: &[u16],
    rgba_stride: u32,
    bit_depth: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    endianness: YuvEndianness,
    bytes_packing: YuvBytesPacking,
) {
    if bit_depth != 10 && bit_depth != 12 {
        panic!("Only 10 and 12 bit depth is supported");
    }
    let dispatcher = match endianness {
        YuvEndianness::BigEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                rgbx_to_yuv_bi_planar_10::<
                    { YuvSourceChannels::Rgba as u8 },
                    { YuvNVOrder::UV as u8 },
                    { YuvChromaSample::YUV422 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                rgbx_to_yuv_bi_planar_10::<
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
                rgbx_to_yuv_bi_planar_10::<
                    { YuvSourceChannels::Rgba as u8 },
                    { YuvNVOrder::UV as u8 },
                    { YuvChromaSample::YUV422 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                rgbx_to_yuv_bi_planar_10::<
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
        rgba,
        rgba_stride,
        bit_depth,
        width,
        height,
        range,
        matrix,
    );
}

/// Convert RGBA image data to YUV 422 bi-planar (NV61 10-bit) format with 10 or 12 bit depth.
///
/// This function performs RGBA to YUV conversion and stores the result in YUV422 bi-planar format,
/// with separate planes for Y (luminance), VU (chrominance) components.
///
/// # Arguments
///
/// * `y_plane` - A mutable slice to store the Y (luminance) plane data.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `uv_plane` - A mutable slice to store the VU (chrominance) plane data.
/// * `uv_stride` - The stride (bytes per row) for the VU plane.
/// * `rgba` - The input RGBA image data slice.
/// * `rgba_stride` - The stride (bytes per row) for the RGBA image data.
/// * `bit_depth` - Only 10 or 12 bit-depth is supported
/// * `width` - The width of the image in pixels.
/// * `height` - The height of the image in pixels.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
/// * `endianness` - The endianness of final YUV
/// * `bytes_packing` - position of significant bytes for YUV ( most significant or least significant ) if it in most significant it should be stated as per Apple *kCVPixelFormatType_422YpCbCr10BiPlanarFullRange/kCVPixelFormatType_422YpCbCr10BiPlanarVideoRange*
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input RGBA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn rgba_to_yuv_nv61_p16(
    y_plane: &mut [u16],
    y_stride: u32,
    uv_plane: &mut [u16],
    uv_stride: u32,
    rgba: &[u16],
    rgba_stride: u32,
    bit_depth: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    endianness: YuvEndianness,
    bytes_packing: YuvBytesPacking,
) {
    if bit_depth != 10 && bit_depth != 12 {
        panic!("Only 10 and 12 bit depth is supported");
    }
    let dispatcher = match endianness {
        YuvEndianness::BigEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                rgbx_to_yuv_bi_planar_10::<
                    { YuvSourceChannels::Rgba as u8 },
                    { YuvNVOrder::VU as u8 },
                    { YuvChromaSample::YUV422 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                rgbx_to_yuv_bi_planar_10::<
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
                rgbx_to_yuv_bi_planar_10::<
                    { YuvSourceChannels::Rgba as u8 },
                    { YuvNVOrder::VU as u8 },
                    { YuvChromaSample::YUV422 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                rgbx_to_yuv_bi_planar_10::<
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
        rgba,
        rgba_stride,
        bit_depth,
        width,
        height,
        range,
        matrix,
    );
}

/// Convert BGRA image data to YUV 422 bi-planar (NV16 10-bit) format with 10 or 12 bit depth.
///
/// This function performs BGRA to YUV conversion and stores the result in YUV422 bi-planar format,
/// with separate planes for Y (luminance), UV (chrominance) components.
///
/// # Arguments
///
/// * `y_plane` - A mutable slice to store the Y (luminance) plane data.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `uv_plane` - A mutable slice to store the UV (chrominance) plane data.
/// * `uv_stride` - The stride (bytes per row) for the UV plane.
/// * `bgra` - The input BGRA image data slice.
/// * `bgra_stride` - The stride (bytes per row) for the BGRA image data.
/// * `bit_depth` - Only 10 or 12 bit-depth is supported
/// * `width` - The width of the image in pixels.
/// * `height` - The height of the image in pixels.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
/// * `endianness` - The endianness of final YUV
/// * `bytes_packing` - position of significant bytes for YUV ( most significant or least significant ) if it in most significant it should be stated as per Apple *kCVPixelFormatType_422YpCbCr10BiPlanarFullRange/kCVPixelFormatType_422YpCbCr10BiPlanarVideoRange*
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGRA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn bgra_to_yuv_nv16_p16(
    y_plane: &mut [u16],
    y_stride: u32,
    uv_plane: &mut [u16],
    uv_stride: u32,
    bgra: &[u16],
    bgra_stride: u32,
    bit_depth: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    endianness: YuvEndianness,
    bytes_packing: YuvBytesPacking,
) {
    if bit_depth != 10 && bit_depth != 12 {
        panic!("Only 10 and 12 bit depth is supported");
    }
    let dispatcher = match endianness {
        YuvEndianness::BigEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                rgbx_to_yuv_bi_planar_10::<
                    { YuvSourceChannels::Bgra as u8 },
                    { YuvNVOrder::UV as u8 },
                    { YuvChromaSample::YUV422 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                rgbx_to_yuv_bi_planar_10::<
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
                rgbx_to_yuv_bi_planar_10::<
                    { YuvSourceChannels::Bgra as u8 },
                    { YuvNVOrder::UV as u8 },
                    { YuvChromaSample::YUV422 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                rgbx_to_yuv_bi_planar_10::<
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

/// Convert BGRA image data to YUV 422 bi-planar (NV61 10-bit) format with 10 or 12 bit depth.
///
/// This function performs BGRA to YUV conversion and stores the result in YUV422 bi-planar format,
/// with separate planes for Y (luminance), VU (chrominance) components.
///
/// # Arguments
///
/// * `y_plane` - A mutable slice to store the Y (luminance) plane data.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `uv_plane` - A mutable slice to store the UV (chrominance) plane data.
/// * `uv_stride` - The stride (bytes per row) for the VU plane.
/// * `bgra` - The input BGRA image data slice.
/// * `bgra_stride` - The stride (bytes per row) for the BGRA image data.
/// * `bit_depth` - Only 10 or 12 bit-depth is supported
/// * `width` - The width of the image in pixels.
/// * `height` - The height of the image in pixels.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
/// * `endianness` - The endianness of final YUV
/// * `bytes_packing` - position of significant bytes for YUV ( most significant or least significant ) if it in most significant it should be stated as per Apple *kCVPixelFormatType_422YpCbCr10BiPlanarFullRange/kCVPixelFormatType_422YpCbCr10BiPlanarVideoRange*
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGRA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn bgra_to_yuv_nv61_p16(
    y_plane: &mut [u16],
    y_stride: u32,
    uv_plane: &mut [u16],
    uv_stride: u32,
    bgra: &[u16],
    bgra_stride: u32,
    bit_depth: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    endianness: YuvEndianness,
    bytes_packing: YuvBytesPacking,
) {
    if bit_depth != 10 && bit_depth != 12 {
        panic!("Only 10 and 12 bit depth is supported");
    }
    let dispatcher = match endianness {
        YuvEndianness::BigEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                rgbx_to_yuv_bi_planar_10::<
                    { YuvSourceChannels::Bgra as u8 },
                    { YuvNVOrder::VU as u8 },
                    { YuvChromaSample::YUV422 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                rgbx_to_yuv_bi_planar_10::<
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
                rgbx_to_yuv_bi_planar_10::<
                    { YuvSourceChannels::Bgra as u8 },
                    { YuvNVOrder::VU as u8 },
                    { YuvChromaSample::YUV422 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                rgbx_to_yuv_bi_planar_10::<
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

/// Convert RGB image data to YUV 444 bi-planar (NV24 10-bit) format with 10 or 12 bit depth.
///
/// This function performs RGB to YUV conversion and stores the result in YUV444 bi-planar format,
/// with separate planes for Y (luminance), UV (chrominance) components.
///
/// # Arguments
///
/// * `y_plane` - A mutable slice to store the Y (luminance) plane data.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `uv_plane` - A mutable slice to store the UV (chrominance) plane data.
/// * `uv_stride` - The stride (bytes per row) for the UV plane.
/// * `bgr` - The input RGB image data slice.
/// * `bgr_stride` - The stride (bytes per row) for the RGB image data.
/// * `bit_depth` - Only 10 or 12 bit-depth is supported
/// * `width` - The width of the image in pixels.
/// * `height` - The height of the image in pixels.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
/// * `endianness` - The endianness of final YUV
/// * `bytes_packing` - position of significant bytes for YUV ( most significant or least significant ) if it in most significant it should be stated as per Apple *kCVPixelFormatType_422YpCbCr10BiPlanarFullRange/kCVPixelFormatType_422YpCbCr10BiPlanarVideoRange*
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input RGB data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn rgb_to_yuv_nv24_p16(
    y_plane: &mut [u16],
    y_stride: u32,
    uv_plane: &mut [u16],
    uv_stride: u32,
    rgb: &[u16],
    rgb_stride: u32,
    bit_depth: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    endianness: YuvEndianness,
    bytes_packing: YuvBytesPacking,
) {
    if bit_depth != 10 && bit_depth != 12 {
        panic!("Only 10 and 12 bit depth is supported");
    }
    let dispatcher = match endianness {
        YuvEndianness::BigEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                rgbx_to_yuv_bi_planar_10::<
                    { YuvSourceChannels::Rgb as u8 },
                    { YuvNVOrder::UV as u8 },
                    { YuvChromaSample::YUV444 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                rgbx_to_yuv_bi_planar_10::<
                    { YuvSourceChannels::Rgb as u8 },
                    { YuvNVOrder::UV as u8 },
                    { YuvChromaSample::YUV444 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
        YuvEndianness::LittleEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                rgbx_to_yuv_bi_planar_10::<
                    { YuvSourceChannels::Rgb as u8 },
                    { YuvNVOrder::UV as u8 },
                    { YuvChromaSample::YUV444 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                rgbx_to_yuv_bi_planar_10::<
                    { YuvSourceChannels::Rgb as u8 },
                    { YuvNVOrder::UV as u8 },
                    { YuvChromaSample::YUV444 as u8 },
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

/// Convert RGB image data to YUV 444 bi-planar (NV42 10-bit) format with 10 or 12 bit depth.
///
/// This function performs RGB to YUV conversion and stores the result in YUV444 bi-planar format,
/// with separate planes for Y (luminance), VU (chrominance) components.
///
/// # Arguments
///
/// * `y_plane` - A mutable slice to store the Y (luminance) plane data.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `uv_plane` - A mutable slice to store the UV (chrominance) plane data.
/// * `uv_stride` - The stride (bytes per row) for the VU plane.
/// * `bgr` - The input RGB image data slice.
/// * `bgr_stride` - The stride (bytes per row) for the RGB image data.
/// * `bit_depth` - Only 10 or 12 bit-depth is supported
/// * `width` - The width of the image in pixels.
/// * `height` - The height of the image in pixels.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
/// * `endianness` - The endianness of final YUV
/// * `bytes_packing` - position of significant bytes for YUV ( most significant or least significant ) if it in most significant it should be stated as per Apple *kCVPixelFormatType_422YpCbCr10BiPlanarFullRange/kCVPixelFormatType_422YpCbCr10BiPlanarVideoRange*
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input RGB data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn rgb_to_yuv_nv42_p16(
    y_plane: &mut [u16],
    y_stride: u32,
    uv_plane: &mut [u16],
    uv_stride: u32,
    rgb: &[u16],
    rgb_stride: u32,
    bit_depth: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    endianness: YuvEndianness,
    bytes_packing: YuvBytesPacking,
) {
    if bit_depth != 10 && bit_depth != 12 {
        panic!("Only 10 and 12 bit depth is supported");
    }
    let dispatcher = match endianness {
        YuvEndianness::BigEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                rgbx_to_yuv_bi_planar_10::<
                    { YuvSourceChannels::Rgb as u8 },
                    { YuvNVOrder::VU as u8 },
                    { YuvChromaSample::YUV444 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                rgbx_to_yuv_bi_planar_10::<
                    { YuvSourceChannels::Rgb as u8 },
                    { YuvNVOrder::VU as u8 },
                    { YuvChromaSample::YUV444 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
        YuvEndianness::LittleEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                rgbx_to_yuv_bi_planar_10::<
                    { YuvSourceChannels::Rgb as u8 },
                    { YuvNVOrder::VU as u8 },
                    { YuvChromaSample::YUV444 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                rgbx_to_yuv_bi_planar_10::<
                    { YuvSourceChannels::Rgb as u8 },
                    { YuvNVOrder::VU as u8 },
                    { YuvChromaSample::YUV444 as u8 },
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

/// Convert BGR image data to YUV 444 bi-planar (NV24 10-bit) format with 10 or 12 bit depth.
///
/// This function performs BGR to YUV conversion and stores the result in YUV444 bi-planar format,
/// with separate planes for Y (luminance), UV (chrominance) components.
///
/// # Arguments
///
/// * `y_plane` - A mutable slice to store the Y (luminance) plane data.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `uv_plane` - A mutable slice to store the UV (chrominance) plane data.
/// * `uv_stride` - The stride (bytes per row) for the UV plane.
/// * `bgr` - The input BGR image data slice.
/// * `bgr_stride` - The stride (bytes per row) for the BGR image data.
/// * `bit_depth` - Only 10 or 12 bit-depth is supported
/// * `width` - The width of the image in pixels.
/// * `height` - The height of the image in pixels.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
/// * `endianness` - The endianness of final YUV
/// * `bytes_packing` - position of significant bytes for YUV ( most significant or least significant ) if it in most significant it should be stated as per Apple *kCVPixelFormatType_422YpCbCr10BiPlanarFullRange/kCVPixelFormatType_422YpCbCr10BiPlanarVideoRange*
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGR data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn bgr_to_yuv_nv24_p16(
    y_plane: &mut [u16],
    y_stride: u32,
    uv_plane: &mut [u16],
    uv_stride: u32,
    bgr: &[u16],
    bgr_stride: u32,
    bit_depth: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    endianness: YuvEndianness,
    bytes_packing: YuvBytesPacking,
) {
    if bit_depth != 10 && bit_depth != 12 {
        panic!("Only 10 and 12 bit depth is supported");
    }
    let dispatcher = match endianness {
        YuvEndianness::BigEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                rgbx_to_yuv_bi_planar_10::<
                    { YuvSourceChannels::Bgr as u8 },
                    { YuvNVOrder::UV as u8 },
                    { YuvChromaSample::YUV444 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                rgbx_to_yuv_bi_planar_10::<
                    { YuvSourceChannels::Bgr as u8 },
                    { YuvNVOrder::UV as u8 },
                    { YuvChromaSample::YUV444 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
        YuvEndianness::LittleEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                rgbx_to_yuv_bi_planar_10::<
                    { YuvSourceChannels::Bgr as u8 },
                    { YuvNVOrder::UV as u8 },
                    { YuvChromaSample::YUV444 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                rgbx_to_yuv_bi_planar_10::<
                    { YuvSourceChannels::Bgr as u8 },
                    { YuvNVOrder::UV as u8 },
                    { YuvChromaSample::YUV444 as u8 },
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

/// Convert BGR image data to YUV 444 bi-planar (NV42 10-bit) format with 10 or 12 bit depth.
///
/// This function performs BGR to YUV conversion and stores the result in YUV444 bi-planar format,
/// with separate planes for Y (luminance), VU (chrominance) components.
///
/// # Arguments
///
/// * `y_plane` - A mutable slice to store the Y (luminance) plane data.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `uv_plane` - A mutable slice to store the UV (chrominance) plane data.
/// * `uv_stride` - The stride (bytes per row) for the VU plane.
/// * `bgr` - The input BGR image data slice.
/// * `bgr_stride` - The stride (bytes per row) for the BGR image data.
/// * `bit_depth` - Only 10 or 12 bit-depth is supported
/// * `width` - The width of the image in pixels.
/// * `height` - The height of the image in pixels.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
/// * `endianness` - The endianness of final YUV
/// * `bytes_packing` - position of significant bytes for YUV ( most significant or least significant ) if it in most significant it should be stated as per Apple *kCVPixelFormatType_422YpCbCr10BiPlanarFullRange/kCVPixelFormatType_422YpCbCr10BiPlanarVideoRange*
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGR data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn bgr_to_yuv_nv42_p16(
    y_plane: &mut [u16],
    y_stride: u32,
    uv_plane: &mut [u16],
    uv_stride: u32,
    bgr: &[u16],
    bgr_stride: u32,
    bit_depth: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    endianness: YuvEndianness,
    bytes_packing: YuvBytesPacking,
) {
    if bit_depth != 10 && bit_depth != 12 {
        panic!("Only 10 and 12 bit depth is supported");
    }
    let dispatcher = match endianness {
        YuvEndianness::BigEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                rgbx_to_yuv_bi_planar_10::<
                    { YuvSourceChannels::Bgr as u8 },
                    { YuvNVOrder::VU as u8 },
                    { YuvChromaSample::YUV444 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                rgbx_to_yuv_bi_planar_10::<
                    { YuvSourceChannels::Bgr as u8 },
                    { YuvNVOrder::VU as u8 },
                    { YuvChromaSample::YUV444 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
        YuvEndianness::LittleEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                rgbx_to_yuv_bi_planar_10::<
                    { YuvSourceChannels::Bgr as u8 },
                    { YuvNVOrder::VU as u8 },
                    { YuvChromaSample::YUV444 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                rgbx_to_yuv_bi_planar_10::<
                    { YuvSourceChannels::Bgr as u8 },
                    { YuvNVOrder::VU as u8 },
                    { YuvChromaSample::YUV444 as u8 },
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

/// Convert BGRA image data to YUV 444 bi-planar (NV24 10-bit) format with 10 or 12 bit depth.
///
/// This function performs BGR to YUV conversion and stores the result in YUV444 bi-planar format,
/// with separate planes for Y (luminance), UV (chrominance) components.
///
/// # Arguments
///
/// * `y_plane` - A mutable slice to store the Y (luminance) plane data.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `uv_plane` - A mutable slice to store the UV (chrominance) plane data.
/// * `uv_stride` - The stride (bytes per row) for the UV plane.
/// * `bgr` - The input BGRA image data slice.
/// * `bgr_stride` - The stride (bytes per row) for the BGRA image data.
/// * `bit_depth` - Only 10 or 12 bit-depth is supported
/// * `width` - The width of the image in pixels.
/// * `height` - The height of the image in pixels.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
/// * `endianness` - The endianness of final YUV
/// * `bytes_packing` - position of significant bytes for YUV ( most significant or least significant ) if it in most significant it should be stated as per Apple *kCVPixelFormatType_422YpCbCr10BiPlanarFullRange/kCVPixelFormatType_422YpCbCr10BiPlanarVideoRange*
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGRA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn bgra_to_yuv_nv24_p16(
    y_plane: &mut [u16],
    y_stride: u32,
    uv_plane: &mut [u16],
    uv_stride: u32,
    bgra: &[u16],
    bgra_stride: u32,
    bit_depth: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    endianness: YuvEndianness,
    bytes_packing: YuvBytesPacking,
) {
    if bit_depth != 10 && bit_depth != 12 {
        panic!("Only 10 and 12 bit depth is supported");
    }
    let dispatcher = match endianness {
        YuvEndianness::BigEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                rgbx_to_yuv_bi_planar_10::<
                    { YuvSourceChannels::Bgra as u8 },
                    { YuvNVOrder::UV as u8 },
                    { YuvChromaSample::YUV444 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                rgbx_to_yuv_bi_planar_10::<
                    { YuvSourceChannels::Bgra as u8 },
                    { YuvNVOrder::UV as u8 },
                    { YuvChromaSample::YUV444 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
        YuvEndianness::LittleEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                rgbx_to_yuv_bi_planar_10::<
                    { YuvSourceChannels::Bgra as u8 },
                    { YuvNVOrder::UV as u8 },
                    { YuvChromaSample::YUV444 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                rgbx_to_yuv_bi_planar_10::<
                    { YuvSourceChannels::Bgra as u8 },
                    { YuvNVOrder::UV as u8 },
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

/// Convert BGRA image data to YUV 444 bi-planar (NV42 10-bit) format with 10 or 12 bit depth.
///
/// This function performs BGR to YUV conversion and stores the result in YUV444 bi-planar format,
/// with separate planes for Y (luminance), VU (chrominance) components.
///
/// # Arguments
///
/// * `y_plane` - A mutable slice to store the Y (luminance) plane data.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `uv_plane` - A mutable slice to store the UV (chrominance) plane data.
/// * `uv_stride` - The stride (bytes per row) for the VU plane.
/// * `bgr` - The input BGRA image data slice.
/// * `bgr_stride` - The stride (bytes per row) for the BGRA image data.
/// * `bit_depth` - Only 10 or 12 bit-depth is supported
/// * `width` - The width of the image in pixels.
/// * `height` - The height of the image in pixels.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
/// * `endianness` - The endianness of final YUV
/// * `bytes_packing` - position of significant bytes for YUV ( most significant or least significant ) if it in most significant it should be stated as per Apple *kCVPixelFormatType_422YpCbCr10BiPlanarFullRange/kCVPixelFormatType_422YpCbCr10BiPlanarVideoRange*
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGRA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn bgra_to_yuv_nv42_p16(
    y_plane: &mut [u16],
    y_stride: u32,
    uv_plane: &mut [u16],
    uv_stride: u32,
    bgra: &[u16],
    bgra_stride: u32,
    bit_depth: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    endianness: YuvEndianness,
    bytes_packing: YuvBytesPacking,
) {
    if bit_depth != 10 && bit_depth != 12 {
        panic!("Only 10 and 12 bit depth is supported");
    }
    let dispatcher = match endianness {
        YuvEndianness::BigEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                rgbx_to_yuv_bi_planar_10::<
                    { YuvSourceChannels::Bgra as u8 },
                    { YuvNVOrder::VU as u8 },
                    { YuvChromaSample::YUV444 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                rgbx_to_yuv_bi_planar_10::<
                    { YuvSourceChannels::Bgra as u8 },
                    { YuvNVOrder::VU as u8 },
                    { YuvChromaSample::YUV444 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
        YuvEndianness::LittleEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                rgbx_to_yuv_bi_planar_10::<
                    { YuvSourceChannels::Bgra as u8 },
                    { YuvNVOrder::VU as u8 },
                    { YuvChromaSample::YUV444 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                rgbx_to_yuv_bi_planar_10::<
                    { YuvSourceChannels::Bgra as u8 },
                    { YuvNVOrder::VU as u8 },
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

/// Convert RGBA image data to YUV 444 bi-planar (NV24 10-bit) format with 10 or 12 bit depth.
///
/// This function performs RGBA to YUV conversion and stores the result in YUV444 bi-planar format,
/// with separate planes for Y (luminance), UV (chrominance) components.
///
/// # Arguments
///
/// * `y_plane` - A mutable slice to store the Y (luminance) plane data.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `uv_plane` - A mutable slice to store the UV (chrominance) plane data.
/// * `uv_stride` - The stride (bytes per row) for the UV plane.
/// * `rgba` - The input RGBA image data slice.
/// * `rgba_stride` - The stride (bytes per row) for the RGBA image data.
/// * `bit_depth` - Only 10 or 12 bit-depth is supported
/// * `width` - The width of the image in pixels.
/// * `height` - The height of the image in pixels.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
/// * `endianness` - The endianness of final YUV
/// * `bytes_packing` - position of significant bytes for YUV ( most significant or least significant ) if it in most significant it should be stated as per Apple *kCVPixelFormatType_422YpCbCr10BiPlanarFullRange/kCVPixelFormatType_422YpCbCr10BiPlanarVideoRange*
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input RGBA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn rgba_to_yuv_nv24_p16(
    y_plane: &mut [u16],
    y_stride: u32,
    uv_plane: &mut [u16],
    uv_stride: u32,
    rgba: &[u16],
    rgba_stride: u32,
    bit_depth: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    endianness: YuvEndianness,
    bytes_packing: YuvBytesPacking,
) {
    if bit_depth != 10 && bit_depth != 12 {
        panic!("Only 10 and 12 bit depth is supported");
    }
    let dispatcher = match endianness {
        YuvEndianness::BigEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                rgbx_to_yuv_bi_planar_10::<
                    { YuvSourceChannels::Rgba as u8 },
                    { YuvNVOrder::UV as u8 },
                    { YuvChromaSample::YUV444 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                rgbx_to_yuv_bi_planar_10::<
                    { YuvSourceChannels::Rgba as u8 },
                    { YuvNVOrder::UV as u8 },
                    { YuvChromaSample::YUV444 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
        YuvEndianness::LittleEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                rgbx_to_yuv_bi_planar_10::<
                    { YuvSourceChannels::Rgba as u8 },
                    { YuvNVOrder::UV as u8 },
                    { YuvChromaSample::YUV444 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                rgbx_to_yuv_bi_planar_10::<
                    { YuvSourceChannels::Rgba as u8 },
                    { YuvNVOrder::UV as u8 },
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

/// Convert RGBA image data to YUV 444 bi-planar (NV42 10-bit) format with 10 or 12 bit depth.
///
/// This function performs RGBA to YUV conversion and stores the result in YUV444 bi-planar format,
/// with separate planes for Y (luminance), VU (chrominance) components.
///
/// # Arguments
///
/// * `y_plane` - A mutable slice to store the Y (luminance) plane data.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `uv_plane` - A mutable slice to store the VU (chrominance) plane data.
/// * `uv_stride` - The stride (bytes per row) for the VU plane.
/// * `rgba` - The input RGBA image data slice.
/// * `rgba_stride` - The stride (bytes per row) for the RGBA image data.
/// * `bit_depth` - Only 10 or 12 bit-depth is supported
/// * `width` - The width of the image in pixels.
/// * `height` - The height of the image in pixels.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
/// * `endianness` - The endianness of final YUV
/// * `bytes_packing` - position of significant bytes for YUV ( most significant or least significant ) if it in most significant it should be stated as per Apple *kCVPixelFormatType_422YpCbCr10BiPlanarFullRange/kCVPixelFormatType_422YpCbCr10BiPlanarVideoRange*
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input RGBA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn rgba_to_yuv_nv42_p16(
    y_plane: &mut [u16],
    y_stride: u32,
    uv_plane: &mut [u16],
    uv_stride: u32,
    rgba: &[u16],
    rgba_stride: u32,
    bit_depth: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    endianness: YuvEndianness,
    bytes_packing: YuvBytesPacking,
) {
    if bit_depth != 10 && bit_depth != 12 {
        panic!("Only 10 and 12 bit depth is supported");
    }
    let dispatcher = match endianness {
        YuvEndianness::BigEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                rgbx_to_yuv_bi_planar_10::<
                    { YuvSourceChannels::Rgba as u8 },
                    { YuvNVOrder::VU as u8 },
                    { YuvChromaSample::YUV444 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                rgbx_to_yuv_bi_planar_10::<
                    { YuvSourceChannels::Rgba as u8 },
                    { YuvNVOrder::VU as u8 },
                    { YuvChromaSample::YUV444 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
        YuvEndianness::LittleEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                rgbx_to_yuv_bi_planar_10::<
                    { YuvSourceChannels::Rgba as u8 },
                    { YuvNVOrder::VU as u8 },
                    { YuvChromaSample::YUV444 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                rgbx_to_yuv_bi_planar_10::<
                    { YuvSourceChannels::Rgba as u8 },
                    { YuvNVOrder::VU as u8 },
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
