/*
 * Copyright (c) Radzivon Bartoshyk, 06/2026. All rights reserved.
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

use crate::images::projected_rgba_plane;
use crate::yuv_error::check_rgba_destination;
use crate::yuv_support::{
    get_forward_transform, get_yuv_range, ToIntegerTransform, YuvSourceChannels,
};
use crate::{
    YuvBytesPacking, YuvEndianness, YuvError, YuvGrayImageMut, YuvRange, YuvStandardMatrix,
};
#[cfg(feature = "rayon")]
use rayon::iter::{IndexedParallelIterator, ParallelIterator};
#[cfg(feature = "rayon")]
use rayon::prelude::{ParallelSlice, ParallelSliceMut};

#[inline(always)]
fn transform_integer<const ENDIANNESS: u8, const BYTES_POSITION: u8, const BIT_DEPTH: usize>(
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
        #[cfg(feature = "big_endian")]
        YuvEndianness::BigEndian => packed_bytes.to_be(),
        YuvEndianness::LittleEndian => packed_bytes.to_le(),
    }
}

fn rgbx_to_yuv_ant<
    const ORIGIN_CHANNELS: u8,
    const ENDIANNESS: u8,
    const BYTES_POSITION: u8,
    const BIT_DEPTH: usize,
    const PRECISION: i32,
>(
    image: &mut YuvGrayImageMut<u16>,
    rgba: &[u16],
    rgba_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) -> Result<(), YuvError> {
    let src_chans: YuvSourceChannels = ORIGIN_CHANNELS.into();
    let channels = src_chans.get_channels_count();

    image.check_constraints()?;
    check_rgba_destination(rgba, rgba_stride, image.width, image.height, channels)?;

    let range = get_yuv_range(BIT_DEPTH as u32, range);
    let kr_kb = matrix.get_kr_kb();
    let max_range = (1u32 << BIT_DEPTH) - 1u32;
    let transform_precise =
        get_forward_transform(max_range, range.range_y, range.range_uv, kr_kb.kr, kr_kb.kb);

    let transform = transform_precise.to_integers(PRECISION as u32);
    let rnd_const: i32 = (1 << (PRECISION - 1)) - 1;
    let bias_y = range.bias_y as i32 * (1 << PRECISION) + rnd_const;

    let width = image.width;
    let height = image.height;

    let y_stride = image.y_stride as usize;
    let y_plane = image.projected_plane_mut();
    let rgba = projected_rgba_plane(rgba, width, height, rgba_stride, src_chans);

    let iter;
    #[cfg(feature = "rayon")]
    {
        iter = y_plane
            .par_chunks_mut(y_stride)
            .zip(rgba.par_chunks(rgba_stride as usize));
    }
    #[cfg(not(feature = "rayon"))]
    {
        iter = y_plane
            .chunks_mut(y_stride)
            .zip(rgba.chunks(rgba_stride as usize));
    }
    iter.for_each(|(y_dst, rgba)| {
        let y_dst = &mut y_dst[..width as usize];

        for (y_dst, rgba) in y_dst.iter_mut().zip(rgba.chunks_exact(channels)) {
            let r0 = rgba[src_chans.get_r_channel_offset()] as i32;
            let g0 = rgba[src_chans.get_g_channel_offset()] as i32;
            let b0 = rgba[src_chans.get_b_channel_offset()] as i32;
            let y_0 =
                (r0 * transform.yr + g0 * transform.yg + b0 * transform.yb + bias_y) >> PRECISION;
            *y_dst = transform_integer::<ENDIANNESS, BYTES_POSITION, BIT_DEPTH>(y_0);
        }
    });

    Ok(())
}

macro_rules! d_cvn {
    ($method: ident, $px_fmt: expr,
    $yuv_name: expr, $rgb_name: expr,
    $rgb_small: expr, $bit_depth: expr,
    $endianness: expr) => {
        #[doc = concat!("Convert ", $rgb_name, " image data to ", $yuv_name, " format with ", $bit_depth, " bit depth.

This function performs ", $rgb_name, stringify!($bit_depth), " to ",$yuv_name," conversion and stores the result in ", $yuv_name," format,
with Y plane (luminance).

# Arguments

* `planar_image` - Target gray image.
* `",$rgb_small,"` - The input ", $rgb_name," image data slice.
* `",$rgb_small,"_stride` - The stride (components per row) for the ", $rgb_name ," image data.
* `range` - The YUV range (limited or full).
* `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).

# Panics

This function panics if the lengths of the planes or the input RGBA data are not valid based
on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
")]
        pub fn $method(
            planar_image: &mut YuvGrayImageMut<u16>,
            rgba: &[u16],
            rgba_stride: u32,
            range: YuvRange,
            matrix: YuvStandardMatrix,
        ) -> Result<(), YuvError> {
            rgbx_to_yuv_ant::<
                { $px_fmt as u8 },
                { $endianness as u8 },
                { YuvBytesPacking::LeastSignificantBytes as u8 },
                $bit_depth,
                15,
            >(planar_image, rgba, rgba_stride, range, matrix)
        }
    };
}

d_cvn!(
    rgba10_to_y010,
    YuvSourceChannels::Rgba,
    "Y010",
    "RGBA10",
    "rgba10",
    10,
    YuvEndianness::LittleEndian
);

d_cvn!(
    rgb10_to_y010,
    YuvSourceChannels::Rgb,
    "Y010",
    "RGB10",
    "rgb10",
    10,
    YuvEndianness::LittleEndian
);

d_cvn!(
    rgba12_to_y012,
    YuvSourceChannels::Rgba,
    "Y012",
    "RGBA12",
    "rgba12",
    12,
    YuvEndianness::LittleEndian
);
d_cvn!(
    rgb12_to_y012,
    YuvSourceChannels::Rgb,
    "Y012",
    "RGB12",
    "rgb12",
    12,
    YuvEndianness::LittleEndian
);
