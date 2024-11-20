/*
 * Copyright (c) Radzivon Bartoshyk, 11/2024. All rights reserved.
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
use crate::yuv_error::check_rgba_destination;
use crate::yuv_support::{Rgb30, YuvSourceChannels};
use crate::{Rgb30ByteOrder, YuvError};

fn rgb_to_ar30_impl<
    const AR30_LAYOUT: usize,
    const AR30_BYTE_ORDER: usize,
    const RGBA_LAYOUT: u8,
>(
    ar30: &mut [u32],
    ar30_stride: u32,
    rgba: &[u8],
    rgba_stride: u32,
    width: u32,
    height: u32,
) -> Result<(), YuvError> {
    let rgba_layout: YuvSourceChannels = RGBA_LAYOUT.into();
    let ar30_layout: Rgb30 = AR30_LAYOUT.into();
    check_rgba_destination(ar30, ar30_stride, width, height, 1)?;
    check_rgba_destination(
        rgba,
        rgba_stride,
        width,
        height,
        rgba_layout.get_channels_count(),
    )?;

    for (src, dst) in rgba
        .chunks_exact(rgba_stride as usize)
        .zip(ar30.chunks_exact_mut(ar30_stride as usize))
    {
        for (src, dst) in src
            .chunks_exact(rgba_layout.get_channels_count())
            .zip(dst.iter_mut())
        {
            let packed = if rgba_layout.has_alpha() {
                ar30_layout.pack_w_a::<AR30_BYTE_ORDER>(
                    src[0] as i32,
                    src[1] as i32,
                    src[2] as i32,
                    src[3] as i32 >> 6,
                )
            } else {
                ar30_layout.pack::<AR30_BYTE_ORDER>(src[0] as i32, src[1] as i32, src[2] as i32)
            };
            *dst = packed;
        }
    }
    Ok(())
}

/// Converts RGB 8 bit depth to AR30 (RGBA2101010)
///
/// # Arguments
///
/// * `ar30`: Dest AR30 data
/// * `ar30_stride`: Dest AR30 stride
/// * `byte_order`: See [Rgb30ByteOrder] for more info
/// * `rgb`: Destination RGB data
/// * `rgb_stride`: Destination RGB stride
/// * `width`: Image width
/// * `height`: Image height
///
pub fn rgb8_to_ar30(
    ar30: &mut [u32],
    ar30_stride: u32,
    byte_order: Rgb30ByteOrder,
    rgb: &[u8],
    rgb_stride: u32,
    width: u32,
    height: u32,
) -> Result<(), YuvError> {
    match byte_order {
        Rgb30ByteOrder::Host => rgb_to_ar30_impl::<
            { Rgb30::Ar30 as usize },
            { Rgb30ByteOrder::Host as usize },
            { YuvSourceChannels::Rgb as u8 },
        >(ar30, ar30_stride, rgb, rgb_stride, width, height),
        Rgb30ByteOrder::Network => rgb_to_ar30_impl::<
            { Rgb30::Ar30 as usize },
            { Rgb30ByteOrder::Network as usize },
            { YuvSourceChannels::Rgb as u8 },
        >(ar30, ar30_stride, rgb, rgb_stride, width, height),
    }
}

/// Converts RGB 8 bit depth to RA30 (RGBA1010102)
///
/// # Arguments
///
/// * `ra30`: Dest RA30 data
/// * `ra30_stride`: Dest RA30 stride
/// * `byte_order`: See [Rgb30ByteOrder] for more info
/// * `rgb`: Destination RGB data
/// * `rgb_stride`: Destination RGB stride
/// * `width`: Image width
/// * `height`: Image height
///
pub fn rgb8_to_ra30(
    ar30: &mut [u32],
    ar30_stride: u32,
    byte_order: Rgb30ByteOrder,
    rgb: &[u8],
    rgb_stride: u32,
    width: u32,
    height: u32,
) -> Result<(), YuvError> {
    match byte_order {
        Rgb30ByteOrder::Host => rgb_to_ar30_impl::<
            { Rgb30::Ar30 as usize },
            { Rgb30ByteOrder::Host as usize },
            { YuvSourceChannels::Rgb as u8 },
        >(ar30, ar30_stride, rgb, rgb_stride, width, height),
        Rgb30ByteOrder::Network => rgb_to_ar30_impl::<
            { Rgb30::Ar30 as usize },
            { Rgb30ByteOrder::Network as usize },
            { YuvSourceChannels::Rgb as u8 },
        >(ar30, ar30_stride, rgb, rgb_stride, width, height),
    }
}

/// Converts RGB 8 bit depth to AR30 (RGBA2101010)
///
/// # Arguments
///
/// * `ar30`: Dest AR30 data
/// * `ar30_stride`: Dest AR30 stride
/// * `byte_order`: See [Rgb30ByteOrder] for more info
/// * `rgb`: Destination RGBA data
/// * `rgb_stride`: Destination RGBA stride
/// * `width`: Image width
/// * `height`: Image height
///
pub fn rgba8_to_ar30(
    ar30: &mut [u32],
    ar30_stride: u32,
    byte_order: Rgb30ByteOrder,
    rgba: &[u8],
    rgba_stride: u32,
    width: u32,
    height: u32,
) -> Result<(), YuvError> {
    match byte_order {
        Rgb30ByteOrder::Host => rgb_to_ar30_impl::<
            { Rgb30::Ar30 as usize },
            { Rgb30ByteOrder::Host as usize },
            { YuvSourceChannels::Rgba as u8 },
        >(ar30, ar30_stride, rgba, rgba_stride, width, height),
        Rgb30ByteOrder::Network => rgb_to_ar30_impl::<
            { Rgb30::Ar30 as usize },
            { Rgb30ByteOrder::Network as usize },
            { YuvSourceChannels::Rgba as u8 },
        >(ar30, ar30_stride, rgba, rgba_stride, width, height),
    }
}

/// Converts RGBA 8 bit depth to RA30 (RGBA1010102)
///
/// # Arguments
///
/// * `ra30`: Dest RA30 data
/// * `ra30_stride`: Dest RA30 stride
/// * `byte_order`: See [Rgb30ByteOrder] for more info
/// * `rgba`: Destination RGBA data
/// * `rgba_stride`: Destination RGBA stride
/// * `width`: Image width
/// * `height`: Image height
///
pub fn rgba8_to_ra30(
    ar30: &mut [u32],
    ar30_stride: u32,
    byte_order: Rgb30ByteOrder,
    rgba: &[u8],
    rgba_stride: u32,
    width: u32,
    height: u32,
) -> Result<(), YuvError> {
    match byte_order {
        Rgb30ByteOrder::Host => rgb_to_ar30_impl::<
            { Rgb30::Ar30 as usize },
            { Rgb30ByteOrder::Host as usize },
            { YuvSourceChannels::Rgba as u8 },
        >(ar30, ar30_stride, rgba, rgba_stride, width, height),
        Rgb30ByteOrder::Network => rgb_to_ar30_impl::<
            { Rgb30::Ar30 as usize },
            { Rgb30ByteOrder::Network as usize },
            { YuvSourceChannels::Rgba as u8 },
        >(ar30, ar30_stride, rgba, rgba_stride, width, height),
    }
}
