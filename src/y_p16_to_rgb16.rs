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
use crate::neon::neon_y_p16_to_rgba16_row;
use crate::yuv_support::*;
#[cfg(feature = "rayon")]
use rayon::iter::{IndexedParallelIterator, ParallelIterator};
#[cfg(feature = "rayon")]
use rayon::prelude::ParallelSliceMut;
use std::slice;

// Chroma subsampling always assumed as 400
fn yuv400_p16_to_rgbx<
    const DESTINATION_CHANNELS: u8,
    const ENDIANNESS: u8,
    const BYTES_POSITION: u8,
>(
    y_plane16: &[u16],
    y_stride: u32,
    rgba16: &mut [u16],
    rgba_stride: u32,
    bit_depth: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) {
    let destination_channels: YuvSourceChannels = DESTINATION_CHANNELS.into();

    let max_colors = (1 << bit_depth) - 1;

    let channels = destination_channels.get_channels_count();
    let range = get_yuv_range(bit_depth, range);
    let kr_kb = get_kr_kb(matrix);
    let transform = get_inverse_transform(
        max_colors,
        range.range_y,
        range.range_uv,
        kr_kb.kr,
        kr_kb.kb,
    );

    const PRECISION: i32 = 6;
    const ROUNDING_CONST: i32 = 1 << (PRECISION - 1);
    let inverse_transform = transform.to_integers(PRECISION as u32);
    let y_coef = inverse_transform.y_coef;

    let bias_y = range.bias_y as i32;

    let casted_rgba = unsafe {
        slice::from_raw_parts_mut(
            rgba16.as_mut_ptr() as *mut u8,
            rgba_stride as usize * height as usize,
        )
    };

    let iter;
    #[cfg(feature = "rayon")]
    {
        iter = casted_rgba.par_chunks_exact_mut(rgba_stride as usize);
    }
    #[cfg(not(feature = "rayon"))]
    {
        iter = casted_rgba.chunks_exact_mut(rgba_stride as usize);
    }

    iter.enumerate().for_each(|(y, rgba16)| unsafe {
        let y_offset = y * (y_stride as usize);

        let mut _cx = 0usize;

        let dst_ptr = rgba16.as_mut_ptr() as *mut u16;
        let y_ptr = (y_plane16.as_ptr() as *const u8).add(y_offset) as *const u16;

        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        {
            let offset = neon_y_p16_to_rgba16_row::<DESTINATION_CHANNELS, ENDIANNESS, BYTES_POSITION>(
                y_ptr,
                dst_ptr,
                0,
                width,
                &range,
                &inverse_transform,
                0,
                bit_depth as usize,
            );
            _cx = offset.cx;
        }

        for x in _cx..width as usize {
            let y_value =
                (y_ptr.add(x).read_unaligned() as i32 - bias_y) * y_coef;

            let r = ((y_value + ROUNDING_CONST) >> PRECISION)
                .min(max_colors as i32)
                .max(0);

            let px = x * channels;

            let rgba_shift = px;

            let dst = dst_ptr.add(rgba_shift);
            dst.add(destination_channels.get_r_channel_offset()).write_unaligned(r as u16);
            dst.add(destination_channels.get_g_channel_offset()).write_unaligned(r as u16);
            dst.add(destination_channels.get_b_channel_offset()).write_unaligned(r as u16);
            if destination_channels.has_alpha() {
                dst.add(destination_channels.get_a_channel_offset()).write_unaligned(max_colors as u16);
            }
        }
    });
}

/// Convert YUV 400 planar format to RGB 8+-bit format.
///
/// This function takes YUV 400 planar format data with 8+-bit precision,
/// and converts it to RGB format with 8+-bit per channel precision.
///
/// # Arguments
///
/// * `y_plane` - A slice to load the Y (luminance) plane data.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `width` - The width of the YUV image.
/// * `height` - The height of the YUV image.
/// * `rgb_data` - A mutable slice to store the converted RGB data.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input RGB data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv400_p16_to_rgb16(
    y_plane: &[u16],
    y_stride: u32,
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
    let callee = match endianness {
        YuvEndianness::BigEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv400_p16_to_rgbx::<
                    { YuvSourceChannels::Rgb as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv400_p16_to_rgbx::<
                    { YuvSourceChannels::Rgb as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
        YuvEndianness::LittleEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv400_p16_to_rgbx::<
                    { YuvSourceChannels::Rgb as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv400_p16_to_rgbx::<
                    { YuvSourceChannels::Rgb as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
    };
    callee(
        y_plane, y_stride, rgb, rgb_stride, bit_depth, width, height, range, matrix,
    );
}

/// Convert YUV 400 planar format to BGR 8+-bit format.
///
/// This function takes YUV 400 planar format data with 8+-bit precision,
/// and converts it to BGR format with 8+-bit per channel precision.
///
/// # Arguments
///
/// * `y_plane` - A slice to load the Y (luminance) plane data.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `width` - The width of the YUV image.
/// * `height` - The height of the YUV image.
/// * `rgb_data` - A mutable slice to store the converted BGR data.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGR data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv400_p16_to_bgr16(
    y_plane: &[u16],
    y_stride: u32,
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
    let callee = match endianness {
        YuvEndianness::BigEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv400_p16_to_rgbx::<
                    { YuvSourceChannels::Bgr as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv400_p16_to_rgbx::<
                    { YuvSourceChannels::Bgr as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
        YuvEndianness::LittleEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv400_p16_to_rgbx::<
                    { YuvSourceChannels::Bgr as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv400_p16_to_rgbx::<
                    { YuvSourceChannels::Bgr as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
    };
    callee(
        y_plane, y_stride, bgr, bgr_stride, bit_depth, width, height, range, matrix,
    )
}

/// Convert YUV 400 planar format to RGBA 8+-bit format.
///
/// This function takes YUV 400 planar format data with 8+-bit precision,
/// and converts it to RGBA format with 8+-bit per channel precision.
///
/// # Arguments
///
/// * `y_plane` - A slice to load the Y (luminance) plane data.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `width` - The width of the YUV image.
/// * `height` - The height of the YUV image.
/// * `rgba_data` - A mutable slice to store the converted RGBA data.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGRA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv400_p16_to_rgba16(
    y_plane: &[u16],
    y_stride: u32,
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
    let callee = match endianness {
        YuvEndianness::BigEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv400_p16_to_rgbx::<
                    { YuvSourceChannels::Rgba as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv400_p16_to_rgbx::<
                    { YuvSourceChannels::Rgba as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
        YuvEndianness::LittleEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv400_p16_to_rgbx::<
                    { YuvSourceChannels::Rgba as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv400_p16_to_rgbx::<
                    { YuvSourceChannels::Rgba as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
    };
    callee(
        y_plane,
        y_stride,
        rgba,
        rgba_stride,
        bit_depth,
        width,
        height,
        range,
        matrix,
    )
}

/// Convert YUV 400 planar format to BGRA 8+-bit format.
///
/// This function takes YUV 400 planar format data with 8+-bit precision,
/// and converts it to BGRA format with 8+-bit per channel precision.
///
/// # Arguments
///
/// * `y_plane` - A slice to load the Y (luminance) plane data.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `width` - The width of the YUV image.
/// * `height` - The height of the YUV image.
/// * `bgra_data` - A mutable slice to store the converted BGRA data.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGRA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv400_p16_to_bgra16(
    y_plane: &[u16],
    y_stride: u32,
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
    let callee = match endianness {
        YuvEndianness::BigEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv400_p16_to_rgbx::<
                    { YuvSourceChannels::Bgra as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv400_p16_to_rgbx::<
                    { YuvSourceChannels::Bgra as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
        YuvEndianness::LittleEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv400_p16_to_rgbx::<
                    { YuvSourceChannels::Bgra as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv400_p16_to_rgbx::<
                    { YuvSourceChannels::Bgra as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
    };
    callee(
        y_plane,
        y_stride,
        bgra,
        bgra_stride,
        bit_depth,
        width,
        height,
        range,
        matrix,
    )
}
