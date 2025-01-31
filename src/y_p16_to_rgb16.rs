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
use crate::{YuvError, YuvGrayImage};
#[cfg(feature = "rayon")]
use rayon::iter::{IndexedParallelIterator, ParallelIterator};
#[cfg(feature = "rayon")]
use rayon::prelude::{ParallelSlice, ParallelSliceMut};

// Chroma subsampling always assumed as 400
fn yuv400_p16_to_rgbx_impl<
    const DESTINATION_CHANNELS: u8,
    const ENDIANNESS: u8,
    const BYTES_POSITION: u8,
    const BIT_DEPTH: usize,
>(
    image: &YuvGrayImage<u16>,
    rgba16: &mut [u16],
    rgba_stride: u32,
    bit_depth: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) -> Result<(), YuvError> {
    let destination_channels: YuvSourceChannels = DESTINATION_CHANNELS.into();

    let max_colors = (1 << bit_depth) - 1;

    let channels = destination_channels.get_channels_count();
    let chroma_range = get_yuv_range(bit_depth, range);
    let kr_kb = matrix.get_kr_kb();

    const PRECISION: i32 = 13;
    const ROUNDING_CONST: i32 = (1 << (PRECISION - 1)) - 1;

    let inverse_transform =
        search_inverse_transform(PRECISION, bit_depth, range, matrix, chroma_range, kr_kb);
    let y_coef = inverse_transform.y_coef;

    let bias_y = chroma_range.bias_y as i32;

    let iter;
    #[cfg(feature = "rayon")]
    {
        iter = rgba16
            .par_chunks_exact_mut(rgba_stride as usize)
            .zip(image.y_plane.par_chunks_exact(image.y_stride as usize));
    }
    #[cfg(not(feature = "rayon"))]
    {
        iter = rgba16
            .chunks_exact_mut(rgba_stride as usize)
            .zip(image.y_plane.chunks_exact(image.y_stride as usize));
    }
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    let neon_wide_handler = neon_y_p16_to_rgba16_row::<
        DESTINATION_CHANNELS,
        ENDIANNESS,
        BYTES_POSITION,
        PRECISION,
        BIT_DEPTH,
    >;

    match range {
        YuvRange::Limited => {
            iter.for_each(|(rgba16, y_plane)| {
                let y_plane = &y_plane[0..image.width as usize];
                let mut _cx = 0usize;

                #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
                {
                    unsafe {
                        let offset = neon_wide_handler(
                            y_plane,
                            rgba16,
                            image.width,
                            &chroma_range,
                            &inverse_transform,
                            0,
                        );
                        _cx = offset.cx;
                    }
                }

                for (dst, &y_src) in rgba16.chunks_exact_mut(channels).zip(y_plane).skip(_cx) {
                    let y_value = (y_src as i32 - bias_y) * y_coef;

                    let r = ((y_value + ROUNDING_CONST) >> PRECISION)
                        .min(max_colors)
                        .max(0);

                    dst[destination_channels.get_r_channel_offset()] = r as u16;
                    dst[destination_channels.get_g_channel_offset()] = r as u16;
                    dst[destination_channels.get_b_channel_offset()] = r as u16;
                    if destination_channels.has_alpha() {
                        dst[destination_channels.get_a_channel_offset()] = max_colors as u16;
                    }
                }
            });
        }
        YuvRange::Full => {
            iter.for_each(|(rgba16, y_plane)| {
                let y_plane = &y_plane[0..image.width as usize];
                for (dst, &y_src) in rgba16.chunks_exact_mut(channels).zip(y_plane) {
                    let r = y_src;

                    dst[destination_channels.get_r_channel_offset()] = r;
                    dst[destination_channels.get_g_channel_offset()] = r;
                    dst[destination_channels.get_b_channel_offset()] = r;
                    if destination_channels.has_alpha() {
                        dst[destination_channels.get_a_channel_offset()] = max_colors as u16;
                    }
                }
            });
        }
    }

    Ok(())
}

macro_rules! d_cnv {
    ($method: ident, $px_fmt: expr, $yuv_name: expr, $rgb_name: expr, $bit_depth: expr) => {
        #[doc = concat!("Convert ", $yuv_name," format to ", $rgb_name, " ", stringify!($bit_depth),"-bit format.

This function takes ", $yuv_name," format data with ", $rgb_name, " ", stringify!($bit_depth),"-bit precision,
and converts it to ", $rgb_name, " ", stringify!($bit_depth),"-bit per channel precision.

# Arguments

* `gray_image` - Source ", $yuv_name, " gray image.
* `rgb_data` - A mutable slice to store the converted ", $rgb_name, stringify!($bit_depth)," data.
* `rgb_stride` - Elements per row.
* `range` - The YUV range (limited or full).
* `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).

# Panics

This function panics if the lengths of the planes or the input RGB data are not valid based
on the specified width, height, and strides, or if invalid YUV range or matrix is provided.")]
        pub fn $method(
            gray_image: &YuvGrayImage<u16>,
            dst: &mut [u16],
            dst_stride: u32,
            range: YuvRange,
            matrix: YuvStandardMatrix,
        ) -> Result<(), YuvError> {
            yuv400_p16_to_rgbx_impl::<
                { $px_fmt as u8 },
                { YuvEndianness::LittleEndian as u8 },
                { YuvBytesPacking::LeastSignificantBytes as u8 },
                $bit_depth
            >(gray_image, dst, dst_stride, $bit_depth, range, matrix)
        }
    };
}

d_cnv!(y010_to_rgba10, YuvSourceChannels::Rgba, "Y010", "RGBA", 10);
d_cnv!(y010_to_rgb10, YuvSourceChannels::Rgb, "Y010", "RGB", 10);
d_cnv!(y012_to_rgb12, YuvSourceChannels::Rgb, "Y012", "RGBA", 12);
d_cnv!(y012_to_rgba12, YuvSourceChannels::Rgba, "Y012", "RGBA", 12);
