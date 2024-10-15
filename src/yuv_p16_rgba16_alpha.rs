/*
 * // Copyright (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use crate::neon::neon_yuv_p16_to_rgba16_alpha_row;
use crate::yuv_support::{
    get_inverse_transform, get_kr_kb, get_yuv_range, YuvBytesPacking, YuvChromaSample,
    YuvEndianness, YuvRange, YuvSourceChannels, YuvStandardMatrix,
};
#[cfg(feature = "rayon")]
use rayon::iter::{IndexedParallelIterator, ParallelIterator};
#[cfg(feature = "rayon")]
use rayon::prelude::ParallelSliceMut;
use std::slice;

pub(crate) fn yuv_p16_to_image_alpha_impl<
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
    a_plane: &[u16],
    a_stride: u32,
    rgba16: &mut [u16],
    rgba_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    bit_depth: usize,
) {
    let dst_chans: YuvSourceChannels = DESTINATION_CHANNELS.into();
    let channels = dst_chans.get_channels_count();

    if dst_chans == YuvSourceChannels::Rgb || dst_chans == YuvSourceChannels::Bgr {
        panic!("Cannot call YUV p16 to Rgb8 with alpha without real alpha");
    }

    let chroma_subsampling: YuvChromaSample = SAMPLING.into();
    let endianness: YuvEndianness = ENDIANNESS.into();
    let bytes_position: YuvBytesPacking = BYTES_POSITION.into();
    let range = get_yuv_range(bit_depth as u32, range);
    let kr_kb = get_kr_kb(matrix);
    let max_range_p16 = (1u32 << bit_depth as u32) - 1;
    const PRECISION: i32 = 6;
    const ROUNDING_CONST: i32 = 1 << (PRECISION - 1);
    let transform = get_inverse_transform(
        max_range_p16,
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

    let msb_shift = 16 - bit_depth;
    let store_shift = PRECISION as usize;

    let dst_offset = 0usize;

    let rgba_safe_align = unsafe {
        slice::from_raw_parts_mut(
            rgba16.as_mut_ptr() as *mut u8,
            height as usize * rgba_stride as usize,
        )
    };

    let iter;
    #[cfg(feature = "rayon")]
    {
        iter = rgba_safe_align.par_chunks_exact_mut(rgba_stride as usize);
    }
    #[cfg(not(feature = "rayon"))]
    {
        iter = rgba_safe_align.chunks_exact_mut(rgba_stride as usize);
    }

    iter.enumerate().for_each(|(y, rgba16)| unsafe {
        let y_offset = y * (y_stride as usize);
        let a_offset = y * (a_stride as usize);
        let u_offset = if chroma_subsampling == YuvChromaSample::YUV420 {
            (y >> 1) * (u_stride as usize)
        } else {
            y * (u_stride as usize)
        };
        let v_offset = if chroma_subsampling == YuvChromaSample::YUV420 {
            (y >> 1) * (v_stride as usize)
        } else {
            y * (v_stride as usize)
        };

        let y_src_ptr = y_plane.as_ptr() as *const u8;
        let u_src_ptr = u_plane.as_ptr() as *const u8;
        let v_src_ptr = v_plane.as_ptr() as *const u8;
        let a_src_ptr = a_plane.as_ptr() as *const u8;

        let mut x = 0usize;
        let mut cx = 0usize;

        let y_ld_ptr = y_src_ptr.add(y_offset) as *const u16;
        let u_ld_ptr = u_src_ptr.add(u_offset) as *const u16;
        let v_ld_ptr = v_src_ptr.add(v_offset) as *const u16;
        let a_ld_ptr = a_src_ptr.add(a_offset) as *const u16;

        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        {
            let offset = neon_yuv_p16_to_rgba16_alpha_row::<
                DESTINATION_CHANNELS,
                SAMPLING,
                ENDIANNESS,
                BYTES_POSITION,
            >(
                y_ld_ptr,
                u_ld_ptr,
                v_ld_ptr,
                a_ld_ptr,
                rgba16.as_mut_ptr() as *mut u16,
                dst_offset,
                width,
                &range,
                &i_transform,
                x,
                cx,
                bit_depth,
            );
            x = offset.cx;
            cx = offset.ux;
        }

        while x < width as usize {
            let y_value: i32;
            let cb_value: i32;
            let cr_value: i32;
            match endianness {
                YuvEndianness::BigEndian => {
                    let mut y_vl = u16::from_be(y_ld_ptr.add(x).read_unaligned()) as i32;
                    let mut cb_vl = u16::from_be(u_ld_ptr.add(cx).read_unaligned()) as i32;
                    let mut cr_vl = u16::from_be(v_ld_ptr.add(cx).read_unaligned()) as i32;
                    if bytes_position == YuvBytesPacking::MostSignificantBytes {
                        y_vl >>= msb_shift;
                        cb_vl >>= msb_shift;
                        cr_vl >>= msb_shift;
                    }
                    y_value = (y_vl - bias_y) * y_coef;

                    cb_value = cb_vl - bias_uv;
                    cr_value = cr_vl - bias_uv;
                }
                YuvEndianness::LittleEndian => {
                    let mut y_vl = u16::from_le(y_ld_ptr.add(x).read_unaligned()) as i32;
                    let mut cb_vl = u16::from_le(u_ld_ptr.add(cx).read_unaligned()) as i32;
                    let mut cr_vl = u16::from_le(v_ld_ptr.add(cx).read_unaligned()) as i32;
                    if bytes_position == YuvBytesPacking::MostSignificantBytes {
                        y_vl >>= msb_shift;
                        cb_vl >>= msb_shift;
                        cr_vl >>= msb_shift;
                    }
                    y_value = (y_vl - bias_y) * y_coef;

                    cb_value = cb_vl - bias_uv;
                    cr_value = cr_vl - bias_uv;
                }
            }

            let r_u16 = (y_value + cr_coef * cr_value + ROUNDING_CONST) >> store_shift;
            let b_u16 = (y_value + cb_coef * cb_value + ROUNDING_CONST) >> store_shift;
            let g_u16 = (y_value - g_coef_1 * cr_value - g_coef_2 * cb_value + ROUNDING_CONST)
                >> store_shift;

            let r = r_u16.min(max_range_p16 as i32).max(0);
            let b = b_u16.min(max_range_p16 as i32).max(0);
            let g = g_u16.min(max_range_p16 as i32).max(0);

            let px = x * channels;

            let rgb_offset = dst_offset + px;

            let dst_slice = (rgba16.as_mut_ptr() as *mut u16).add(rgb_offset);

            dst_slice
                .add(dst_chans.get_b_channel_offset())
                .write_unaligned(b as u16);
            dst_slice
                .add(dst_chans.get_g_channel_offset())
                .write_unaligned(g as u16);
            dst_slice
                .add(dst_chans.get_r_channel_offset())
                .write_unaligned(r as u16);
            let mut a0 = if endianness == YuvEndianness::BigEndian {
                u16::from_be(a_ld_ptr.add(x).read_unaligned())
            } else {
                u16::from_le(a_ld_ptr.add(x).read_unaligned())
            };
            if bytes_position == YuvBytesPacking::MostSignificantBytes {
                a0 >>= msb_shift;
            }
            dst_slice
                .add(dst_chans.get_a_channel_offset())
                .write_unaligned(a0);

            x += 1;

            if x + 1 < width as usize {
                let y_value: i32 = match endianness {
                    YuvEndianness::BigEndian => {
                        let mut y_vl = u16::from_be(y_ld_ptr.add(x).read_unaligned()) as i32;
                        if bytes_position == YuvBytesPacking::MostSignificantBytes {
                            y_vl >>= msb_shift;
                        }
                        (y_vl - bias_y) * y_coef
                    }
                    YuvEndianness::LittleEndian => {
                        let mut y_vl = u16::from_le(y_ld_ptr.add(x).read_unaligned()) as i32;
                        if bytes_position == YuvBytesPacking::MostSignificantBytes {
                            y_vl >>= msb_shift;
                        }
                        (y_vl - bias_y) * y_coef
                    }
                };

                let r_u16 = (y_value + cr_coef * cr_value + ROUNDING_CONST) >> store_shift;
                let b_u16 = (y_value + cb_coef * cb_value + ROUNDING_CONST) >> store_shift;
                let g_u16 = (y_value - g_coef_1 * cr_value - g_coef_2 * cb_value + ROUNDING_CONST)
                    >> store_shift;

                let r = r_u16.min(max_range_p16 as i32).max(0);
                let b = b_u16.min(max_range_p16 as i32).max(0);
                let g = g_u16.min(max_range_p16 as i32).max(0);

                let mut a1 = if endianness == YuvEndianness::BigEndian {
                    u16::from_be(a_ld_ptr.add(x).read_unaligned())
                } else {
                    u16::from_le(a_ld_ptr.add(x).read_unaligned())
                };
                if bytes_position == YuvBytesPacking::MostSignificantBytes {
                    a1 >>= msb_shift;
                }

                let px = x * channels;
                let rgb_offset = dst_offset + px;
                let dst_slice = (rgba16.as_mut_ptr() as *mut u16).add(rgb_offset);
                dst_slice
                    .add(dst_chans.get_b_channel_offset())
                    .write_unaligned(b as u16);
                dst_slice
                    .add(dst_chans.get_g_channel_offset())
                    .write_unaligned(g as u16);
                dst_slice
                    .add(dst_chans.get_r_channel_offset())
                    .write_unaligned(r as u16);
                dst_slice
                    .add(dst_chans.get_a_channel_offset())
                    .write_unaligned(a1);
            }

            x += 1;
            cx += 1;
        }
    });
}

/// Convert YUV 420 planar format with 8+ bit pixel format to BGRA 8+ bit-depth format with interleaving alpha.
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
/// * `a_plane` - A slice to load the alpha with 8+ bit depth.
/// * `a_stride` - The stride (bytes per row) for the Alpha plane.
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
pub fn yuv420_p16_with_alpha_to_bgra16(
    y_plane: &[u16],
    y_stride: u32,
    u_plane: &[u16],
    u_stride: u32,
    v_plane: &[u16],
    v_stride: u32,
    a_plane: &[u16],
    a_stride: u32,
    bgra: &mut [u16],
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
                yuv_p16_to_image_alpha_impl::<
                    { YuvSourceChannels::Bgra as u8 },
                    { YuvChromaSample::YUV420 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_p16_to_image_alpha_impl::<
                    { YuvSourceChannels::Bgra as u8 },
                    { YuvChromaSample::YUV420 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
        YuvEndianness::LittleEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_p16_to_image_alpha_impl::<
                    { YuvSourceChannels::Bgra as u8 },
                    { YuvChromaSample::YUV420 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_p16_to_image_alpha_impl::<
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
        a_plane,
        a_stride,
        bgra,
        bgra_stride,
        width,
        height,
        range,
        matrix,
        bit_depth,
    );
}

/// Convert YUV 422 format with 8+ bit pixel format to BGRA 8+ bit-depth format with interleaving alpha.
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
/// * `v_stride` - The stride (bytes per row) for the U plane.
/// * `a_plane` - A slice to load the alpha with 8+ bit depth.
/// * `a_stride` - The stride (bytes per row) for the Alpha plane.
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
pub fn yuv422_p16_with_alpha_to_bgra16(
    y_plane: &[u16],
    y_stride: u32,
    u_plane: &[u16],
    u_stride: u32,
    v_plane: &[u16],
    v_stride: u32,
    a_plane: &[u16],
    a_stride: u32,
    bgra: &mut [u16],
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
                yuv_p16_to_image_alpha_impl::<
                    { YuvSourceChannels::Bgra as u8 },
                    { YuvChromaSample::YUV422 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_p16_to_image_alpha_impl::<
                    { YuvSourceChannels::Bgra as u8 },
                    { YuvChromaSample::YUV422 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
        YuvEndianness::LittleEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_p16_to_image_alpha_impl::<
                    { YuvSourceChannels::Bgra as u8 },
                    { YuvChromaSample::YUV422 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_p16_to_image_alpha_impl::<
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
        a_plane,
        a_stride,
        bgra,
        bgra_stride,
        width,
        height,
        range,
        matrix,
        bit_depth,
    );
}

/// Convert YUV 420 planar format with 8+ bit pixel format to RGBA 8+ bit-depth format with interleaving alpha.
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
/// * `a_plane` - A slice to load the alpha with 8+ bit depth.
/// * `a_stride` - The stride (bytes per row) for the Alpha plane.
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
pub fn yuv420_p16_with_alpha_to_rgba16(
    y_plane: &[u16],
    y_stride: u32,
    u_plane: &[u16],
    u_stride: u32,
    v_plane: &[u16],
    v_stride: u32,
    a_plane: &[u16],
    a_stride: u32,
    rgba: &mut [u16],
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
                yuv_p16_to_image_alpha_impl::<
                    { YuvSourceChannels::Rgba as u8 },
                    { YuvChromaSample::YUV420 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_p16_to_image_alpha_impl::<
                    { YuvSourceChannels::Rgba as u8 },
                    { YuvChromaSample::YUV420 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
        YuvEndianness::LittleEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_p16_to_image_alpha_impl::<
                    { YuvSourceChannels::Rgba as u8 },
                    { YuvChromaSample::YUV420 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_p16_to_image_alpha_impl::<
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
        a_plane,
        a_stride,
        rgba,
        rgba_stride,
        width,
        height,
        range,
        matrix,
        bit_depth,
    );
}

/// Convert YUV 422 format with 8+ bit pixel format to RGBA 8+ bit-depth format with interleaving alpha.
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
/// * `v_stride` - The stride (bytes per row) for the U plane.
/// * `a_plane` - A slice to load the alpha with 8+ bit depth.
/// * `a_stride` - The stride (bytes per row) for the Alpha plane.
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
pub fn yuv422_p16_with_alpha_to_rgba16(
    y_plane: &[u16],
    y_stride: u32,
    u_plane: &[u16],
    u_stride: u32,
    v_plane: &[u16],
    v_stride: u32,
    a_plane: &[u16],
    a_stride: u32,
    rgba: &mut [u16],
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
                yuv_p16_to_image_alpha_impl::<
                    { YuvSourceChannels::Rgba as u8 },
                    { YuvChromaSample::YUV422 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_p16_to_image_alpha_impl::<
                    { YuvSourceChannels::Rgba as u8 },
                    { YuvChromaSample::YUV422 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
        YuvEndianness::LittleEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_p16_to_image_alpha_impl::<
                    { YuvSourceChannels::Rgba as u8 },
                    { YuvChromaSample::YUV422 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_p16_to_image_alpha_impl::<
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
        a_plane,
        a_stride,
        rgba,
        rgba_stride,
        width,
        height,
        range,
        matrix,
        bit_depth,
    );
}

/// Convert YUV 444 planar format with 8+ bit pixel format to RGBA 8+ bit-depth format with interleaving alpha.
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
/// * `a_plane` - A slice to load the alpha with 8+ bit depth.
/// * `a_stride` - The stride (bytes per row) for the Alpha plane.
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
pub fn yuv444_p16_with_alpha_to_rgba16(
    y_plane: &[u16],
    y_stride: u32,
    u_plane: &[u16],
    u_stride: u32,
    v_plane: &[u16],
    v_stride: u32,
    a_plane: &[u16],
    a_stride: u32,
    rgba: &mut [u16],
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
                yuv_p16_to_image_alpha_impl::<
                    { YuvSourceChannels::Rgba as u8 },
                    { YuvChromaSample::YUV444 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_p16_to_image_alpha_impl::<
                    { YuvSourceChannels::Rgba as u8 },
                    { YuvChromaSample::YUV444 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
        YuvEndianness::LittleEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_p16_to_image_alpha_impl::<
                    { YuvSourceChannels::Rgba as u8 },
                    { YuvChromaSample::YUV444 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_p16_to_image_alpha_impl::<
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
        a_plane,
        a_stride,
        rgba,
        rgba_stride,
        width,
        height,
        range,
        matrix,
        bit_depth,
    );
}

/// Convert YUV 444 planar format with 8+ bit pixel format to BGRA 8+ bit-depth format with interleaving alpha.
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
/// * `a_plane` - A slice to load the alpha with 8+ bit depth.
/// * `a_stride` - The stride (bytes per row) for the Alpha plane.
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
pub fn yuv444_p16_with_alpha_to_bgra16(
    y_plane: &[u16],
    y_stride: u32,
    u_plane: &[u16],
    u_stride: u32,
    v_plane: &[u16],
    v_stride: u32,
    a_plane: &[u16],
    a_stride: u32,
    bgra: &mut [u16],
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
                yuv_p16_to_image_alpha_impl::<
                    { YuvSourceChannels::Bgra as u8 },
                    { YuvChromaSample::YUV444 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_p16_to_image_alpha_impl::<
                    { YuvSourceChannels::Bgra as u8 },
                    { YuvChromaSample::YUV444 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
        YuvEndianness::LittleEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_p16_to_image_alpha_impl::<
                    { YuvSourceChannels::Bgra as u8 },
                    { YuvChromaSample::YUV444 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_p16_to_image_alpha_impl::<
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
        a_plane,
        a_stride,
        bgra,
        bgra_stride,
        width,
        height,
        range,
        matrix,
        bit_depth,
    );
}
