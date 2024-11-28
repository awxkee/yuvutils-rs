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
#![forbid(unsafe_code)]
use crate::numerics::qrshr;
use crate::yuv_error::check_rgba_destination;
use crate::yuv_support::{get_yuv_range, YuvSourceChannels};
use crate::{YuvChromaSubsampling, YuvError, YuvPlanarImageWithAlpha, YuvRange};
use num_traits::AsPrimitive;
#[cfg(feature = "rayon")]
use rayon::iter::{IndexedParallelIterator, ParallelIterator};
#[cfg(feature = "rayon")]
use rayon::prelude::{ParallelSlice, ParallelSliceMut};
use std::fmt::Debug;

#[inline]
fn gbr_to_rgbx_alpha_impl<
    V: Copy + AsPrimitive<i32> + 'static + Sized + Debug + Send + Sync,
    const CHANNELS: u8,
    const BIT_DEPTH: usize,
>(
    image: &YuvPlanarImageWithAlpha<V>,
    rgba: &mut [V],
    rgba_stride: u32,
    yuv_range: YuvRange,
) -> Result<(), YuvError>
where
    i32: AsPrimitive<V>,
{
    let destination_channels: YuvSourceChannels = CHANNELS.into();
    let channels = destination_channels.get_channels_count();
    assert_eq!(
        channels, 4,
        "GBRA -> RGBA is implemented only on 4 channels"
    );
    assert!(
        (8..=16).contains(&BIT_DEPTH),
        "Invalid bit depth is provided"
    );
    assert!(
        if BIT_DEPTH > 8 {
            size_of::<V>() == 2
        } else {
            size_of::<V>() == 1
        },
        "Unsupported bit depth and data type combination"
    );
    let y_plane = image.y_plane;
    let u_plane = image.u_plane;
    let v_plane = image.v_plane;
    let y_stride = image.y_stride as usize;
    let u_stride = image.u_stride as usize;
    let v_stride = image.v_stride as usize;
    let height = image.height;

    image.check_constraints(YuvChromaSubsampling::Yuv444)?;
    check_rgba_destination(rgba, rgba_stride, image.width, height, channels)?;

    let y_iter;
    let rgb_iter;
    let u_iter;
    let v_iter;
    let a_iter;

    #[cfg(feature = "rayon")]
    {
        y_iter = y_plane.par_chunks_exact(y_stride);
        rgb_iter = rgba.par_chunks_exact_mut(rgba_stride as usize);
        u_iter = u_plane.par_chunks_exact(u_stride);
        v_iter = v_plane.par_chunks_exact(v_stride);
        a_iter = image.a_plane.par_chunks_exact(image.a_stride as usize);
    }
    #[cfg(not(feature = "rayon"))]
    {
        y_iter = y_plane.chunks_exact(y_stride);
        rgb_iter = rgba.chunks_exact_mut(rgba_stride as usize);
        u_iter = u_plane.chunks_exact(u_stride);
        v_iter = v_plane.chunks_exact(v_stride);
        a_iter = image.a_plane.chunks_exact(image.a_stride as usize);
    }

    match yuv_range {
        YuvRange::Limited => {
            const PRECISION: i32 = 13;
            // All channels on identity should use Y range
            let range = get_yuv_range(BIT_DEPTH as u32, yuv_range);
            let range_rgba = (1 << BIT_DEPTH) - 1;
            let y_coef =
                ((range_rgba as f32 / range.range_y as f32) * (1 << PRECISION) as f32) as i32;
            let y_bias = range.bias_y as i32;

            let iter = y_iter.zip(u_iter).zip(v_iter).zip(rgb_iter).zip(a_iter);
            iter.for_each(|((((y_src, u_src), v_src), rgb), a_src)| {
                let y_src = &y_src[0..image.width as usize];
                let rgb_chunks = rgb.chunks_exact_mut(channels);

                for ((((&y_src, &u_src), &v_src), rgb_dst), &a_src) in y_src
                    .iter()
                    .zip(u_src)
                    .zip(v_src)
                    .zip(rgb_chunks)
                    .zip(a_src)
                {
                    rgb_dst[destination_channels.get_r_channel_offset()] =
                        qrshr::<PRECISION, BIT_DEPTH>((v_src.as_() - y_bias) * y_coef).as_();
                    rgb_dst[destination_channels.get_g_channel_offset()] =
                        qrshr::<PRECISION, BIT_DEPTH>((y_src.as_() - y_bias) * y_coef).as_();
                    rgb_dst[destination_channels.get_b_channel_offset()] =
                        qrshr::<PRECISION, BIT_DEPTH>((u_src.as_() - y_bias) * y_coef).as_();
                    rgb_dst[destination_channels.get_a_channel_offset()] = a_src;
                }
            });
        }
        YuvRange::Full => {
            let iter = y_iter.zip(u_iter).zip(v_iter).zip(rgb_iter).zip(a_iter);
            iter.for_each(|((((y_src, u_src), v_src), rgb), a_src)| {
                let y_src = &y_src[0..image.width as usize];
                let rgb_chunks = rgb.chunks_exact_mut(channels);

                for ((((&y_src, &u_src), &v_src), rgb_dst), &a_src) in y_src
                    .iter()
                    .zip(u_src)
                    .zip(v_src)
                    .zip(rgb_chunks)
                    .zip(a_src)
                {
                    rgb_dst[destination_channels.get_r_channel_offset()] = v_src;
                    rgb_dst[destination_channels.get_g_channel_offset()] = y_src;
                    rgb_dst[destination_channels.get_b_channel_offset()] = u_src;
                    rgb_dst[destination_channels.get_a_channel_offset()] = a_src;
                }
            });
        }
    }

    Ok(())
}

/// Convert YUV Identity Matrix ( aka 'GBR ) with alpha channel to RGBA
///
/// This function takes GBR interleaved format data with 8-bit precision,
/// and converts it to RGBA format with 8-bit per channel precision.
///
/// # Arguments
///
/// * `image` - Source GBR image.
/// * `rgba` - A slice to store the RGBA plane data.
/// * `rgba_stride` - The stride (components per row) for the RGBA plane.
/// * `range` - Yuv values range.
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input RGBA data are not valid based
/// on the specified width, height, and strides is provided.
///
pub fn gbr_with_alpha_to_rgba(
    image: &YuvPlanarImageWithAlpha<u8>,
    rgb: &mut [u8],
    rgb_stride: u32,
    range: YuvRange,
) -> Result<(), YuvError> {
    gbr_to_rgbx_alpha_impl::<u8, { YuvSourceChannels::Rgba as u8 }, 8>(
        image, rgb, rgb_stride, range,
    )
}

/// Convert YUV Identity Matrix ( aka 'GBR ) with alpha channel to BGRA
///
/// This function takes GBR interleaved format data with 8-bit precision,
/// and converts it to BGRA format with 8-bit per channel precision.
///
/// # Arguments
///
/// * `image` - Source GBR image.
/// * `rgba` - A slice to store the BGRA plane data.
/// * `rgba_stride` - The stride (components per row) for the BGRA plane.
/// * `range` - Yuv values range.
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGRA data are not valid based
/// on the specified width, height, and strides is provided.
///
pub fn gbr_with_alpha_to_bgra(
    image: &YuvPlanarImageWithAlpha<u8>,
    rgb: &mut [u8],
    rgb_stride: u32,
    range: YuvRange,
) -> Result<(), YuvError> {
    gbr_to_rgbx_alpha_impl::<u8, { YuvSourceChannels::Bgra as u8 }, 8>(
        image, rgb, rgb_stride, range,
    )
}

/// Convert YUV Identity Matrix ( aka 'GBR ) with alpha channel to RGBA
///
/// This function takes GBR interleaved format data with 8+ bit precision,
/// and converts it to RGBA format with 8+ bit per channel precision.
///
/// # Arguments
///
/// * `image` - Source GBR image.
/// * `rgba` - A slice to store the RGBA plane data.
/// * `rgba_stride` - The stride (components per row) for the RGBA plane.
/// * `bit_depth` - YUV and RGB bit depth
/// * `range` - Yuv values range.
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input RGBA data are not valid based
/// on the specified width, height, and strides is provided.
///
pub fn gbr_with_alpha_to_rgba_p16(
    image: &YuvPlanarImageWithAlpha<u16>,
    rgba: &mut [u16],
    rgba_stride: u32,
    bit_depth: u32,
    range: YuvRange,
) -> Result<(), YuvError> {
    assert!(
        bit_depth == 10 || bit_depth == 12,
        "Only 10 and 12 bit supported"
    );
    if bit_depth == 10 {
        gbr_to_rgbx_alpha_impl::<u16, { YuvSourceChannels::Rgba as u8 }, 10>(
            image,
            rgba,
            rgba_stride,
            range,
        )
    } else if bit_depth == 12 {
        gbr_to_rgbx_alpha_impl::<u16, { YuvSourceChannels::Rgba as u8 }, 12>(
            image,
            rgba,
            rgba_stride,
            range,
        )
    } else {
        unreachable!();
    }
}

/// Convert YUV Identity Matrix ( aka 'GBR ) with alpha to BGRA
///
/// This function takes GBR interleaved format data with 8+ bit precision,
/// and converts it to BGRA format with 8+ bit per channel precision.
///
/// # Arguments
///
/// * `image` - Source GBR image.
/// * `bgra` - A slice to store the BGRA plane data.
/// * `bgra_stride` - The stride (components per row) for the BGRA plane.
/// * `bit_depth` - YUV and RGB bit depth.
/// * `range` - Yuv values range.
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGRA data are not valid based
/// on the specified width, height, and strides is provided.
///
pub fn gbr_with_alpha_to_bgra_p16(
    image: &YuvPlanarImageWithAlpha<u16>,
    bgra: &mut [u16],
    bgra_stride: u32,
    bit_depth: u32,
    range: YuvRange,
) -> Result<(), YuvError> {
    assert!(
        bit_depth == 10 || bit_depth == 12,
        "Only 10 and 12 bit supported"
    );
    if bit_depth == 10 {
        gbr_to_rgbx_alpha_impl::<u16, { YuvSourceChannels::Bgra as u8 }, 10>(
            image,
            bgra,
            bgra_stride,
            range,
        )
    } else if bit_depth == 12 {
        gbr_to_rgbx_alpha_impl::<u16, { YuvSourceChannels::Bgra as u8 }, 12>(
            image,
            bgra,
            bgra_stride,
            range,
        )
    } else {
        unreachable!();
    }
}
