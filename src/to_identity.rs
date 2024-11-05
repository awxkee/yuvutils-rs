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
use crate::numerics::qrshr;
use crate::yuv_error::check_rgba_destination;
use crate::yuv_support::{get_yuv_range, YuvSourceChannels};
use crate::{YuvChromaSubsample, YuvError, YuvPlanarImageMut, YuvRange};
use num_traits::AsPrimitive;
use std::fmt::Debug;

#[inline]
fn rgbx_to_gbr_impl<
    V: Copy + AsPrimitive<i32> + 'static + Sized + Debug,
    const CHANNELS: u8,
    const BIT_DEPTH: usize,
>(
    image: &mut YuvPlanarImageMut<V>,
    rgba: &[V],
    rgba_stride: u32,
    yuv_range: YuvRange,
) -> Result<(), YuvError>
where
    i32: AsPrimitive<V>,
{
    let destination_channels: YuvSourceChannels = CHANNELS.into();
    let channels = destination_channels.get_channels_count();
    assert!(
        channels == 3 || channels == 4,
        "GBR -> RGB is implemented only on 3 and 4 channels"
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

    image.check_constraints(YuvChromaSubsample::Yuv444)?;
    check_rgba_destination(rgba, rgba_stride, image.width, image.height, channels)?;

    let y_plane = image.y_plane.borrow_mut();
    let u_plane = image.u_plane.borrow_mut();
    let v_plane = image.v_plane.borrow_mut();
    let y_stride = image.y_stride as usize;
    let u_stride = image.u_stride as usize;
    let v_stride = image.v_stride as usize;

    let y_iter = y_plane.chunks_exact_mut(y_stride);
    let rgba_iter = rgba.chunks_exact(rgba_stride as usize);
    let u_iter = u_plane.chunks_exact_mut(u_stride);
    let v_iter = v_plane.chunks_exact_mut(v_stride);

    match yuv_range {
        YuvRange::Limited => {
            const PRECISION: i32 = 11;
            // All channels on identity should use Y range
            let range = get_yuv_range(BIT_DEPTH as u32, yuv_range);
            let range_rgba = (1 << BIT_DEPTH) - 1;
            let y_coef =
                ((range.range_y as f32 / range_rgba as f32) * (1 << PRECISION) as f32) as i32;
            let y_bias = range.bias_y as i32 * (1 << PRECISION);

            for (((y_dst, u_dst), v_dst), rgba) in y_iter.zip(u_iter).zip(v_iter).zip(rgba_iter) {
                let rgb_chunks = rgba.chunks_exact(channels);

                for (((y_dst, u_dst), v_dst), rgb_dst) in y_dst
                    .iter_mut()
                    .zip(u_dst.iter_mut())
                    .zip(v_dst.iter_mut())
                    .zip(rgb_chunks)
                {
                    *v_dst =
                        qrshr::<PRECISION, BIT_DEPTH>(rgb_dst[0].as_() * y_coef + y_bias).as_();
                    *y_dst =
                        qrshr::<PRECISION, BIT_DEPTH>(rgb_dst[1].as_() * y_coef + y_bias).as_();
                    *u_dst =
                        qrshr::<PRECISION, BIT_DEPTH>(rgb_dst[2].as_() * y_coef + y_bias).as_();
                }
            }
        }
        YuvRange::Full => {
            for (((y_dst, u_dst), v_dst), rgba) in y_iter.zip(u_iter).zip(v_iter).zip(rgba_iter) {
                let rgb_chunks = rgba.chunks_exact(channels);

                for (((y_dst, u_dst), v_dst), rgb_dst) in y_dst
                    .iter_mut()
                    .zip(u_dst.iter_mut())
                    .zip(v_dst.iter_mut())
                    .zip(rgb_chunks)
                {
                    *v_dst = rgb_dst[0];
                    *y_dst = rgb_dst[1];
                    *u_dst = rgb_dst[2];
                }
            }
        }
    }

    Ok(())
}

/// Convert RGB to YUV Identity Matrix ( aka 'GBR )
///
/// This function takes RGB image format data with 8-bit precision,
/// and converts it to GBR YUV format with 8-bit per channel precision.
///
/// # Arguments
///
/// * `image` - Target GBR image.
/// * `rgb` - A slice to load RGB data.
/// * `rgb_stride` - The stride (components per row) for the RGB plane.
/// * `range` - Yuv values range.
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input RGB data are not valid based
/// on the specified width, height, and strides is provided.
///
pub fn rgb_to_gbr(
    image: &mut YuvPlanarImageMut<u8>,
    rgb: &[u8],
    rgb_stride: u32,
    range: YuvRange,
) -> Result<(), YuvError> {
    rgbx_to_gbr_impl::<u8, { YuvSourceChannels::Rgb as u8 }, 8>(image, rgb, rgb_stride, range)
}

/// Convert BGR to YUV Identity Matrix ( aka 'GBR )
///
/// This function takes BGR image format data with 8-bit precision,
/// and converts it to GBR YUV format with 8-bit per channel precision.
///
/// # Arguments
///
/// * `image` - Target GBR image.
/// * `bgr` - A slice to load BGR data.
/// * `bgr_stride` - The stride (components per row) for the BGR plane.
/// * `range` - Yuv values range.
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGR data are not valid based
/// on the specified width, height, and strides is provided.
///
pub fn bgr_to_gbr(
    image: &mut YuvPlanarImageMut<u8>,
    bgr: &mut [u8],
    bgr_stride: u32,
    range: YuvRange,
) -> Result<(), YuvError> {
    rgbx_to_gbr_impl::<u8, { YuvSourceChannels::Bgr as u8 }, 8>(image, bgr, bgr_stride, range)
}

/// Convert BGRA to YUV Identity Matrix ( aka 'GBR )
///
/// This function takes BGRA image format data with 8-bit precision,
/// and converts it to GBR YUV format with 8-bit per channel precision.
///
/// # Arguments
///
/// * `image` - Target GBR image.
/// * `bgra` - A slice to load RGBA data.
/// * `bgra_stride` - The stride (components per row) for the RGBA plane.
/// * `range` - Yuv values range.
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGRA data are not valid based
/// on the specified width, height, and strides is provided.
///
pub fn bgra_to_gbr(
    image: &mut YuvPlanarImageMut<u8>,
    bgra: &mut [u8],
    bgra_stride: u32,
    range: YuvRange,
) -> Result<(), YuvError> {
    rgbx_to_gbr_impl::<u8, { YuvSourceChannels::Bgra as u8 }, 8>(image, bgra, bgra_stride, range)
}

/// Convert RGBA to YUV Identity Matrix ( aka 'GBR )
///
/// This function takes BGRA RGBA format data with 8-bit precision,
/// and converts it to GBR YUV format with 8-bit per channel precision.
///
/// # Arguments
///
/// * `image` - Target GBR image.
/// * `rgba` - A slice to load RGBA data.
/// * `rgba_stride` - The stride (components per row) for the RGBA plane.
/// * `range` - Yuv values range.
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input RGBA data are not valid based
/// on the specified width, height, and strides is provided.
///
pub fn rgba_to_gbr(
    image: &mut YuvPlanarImageMut<u8>,
    rgba: &[u8],
    rgba_stride: u32,
    range: YuvRange,
) -> Result<(), YuvError> {
    rgbx_to_gbr_impl::<u8, { YuvSourceChannels::Rgba as u8 }, 8>(image, rgba, rgba_stride, range)
}

/// Convert RGB to YUV Identity Matrix ( aka 'GBR )
///
/// This function takes RGB image format data with 8+-bit precision,
/// and converts it to GBR YUV format with 8+-bit per channel precision.
///
/// # Arguments
///
/// * `image` - Target GBR image.
/// * `rgb16` - A slice with RGB data
/// * `rgb_stride` - The stride (components per row) for the RGB plane.
/// * `bit_depth` - Only 10 and 12 is supported.
/// * `range` - Yuv values range.
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input RGB data are not valid based
/// on the specified width, height, and strides is provided.
///
pub fn rgb16_to_gbr16(
    image: &mut YuvPlanarImageMut<u16>,
    rgb16: &[u16],
    rgb_stride: u32,
    bit_depth: u32,
    range: YuvRange,
) -> Result<(), YuvError> {
    assert!(
        bit_depth == 10 || bit_depth == 12,
        "Only 10 and 12 bit supported"
    );
    if bit_depth == 10 {
        rgbx_to_gbr_impl::<u16, { YuvSourceChannels::Rgb as u8 }, 10>(
            image, rgb16, rgb_stride, range,
        )
    } else if bit_depth == 12 {
        rgbx_to_gbr_impl::<u16, { YuvSourceChannels::Rgb as u8 }, 12>(
            image, rgb16, rgb_stride, range,
        )
    } else {
        unreachable!();
    }
}

/// Convert RGBA to YUV Identity Matrix ( aka 'GBR )
///
/// This function takes RGBA image format data with 8+-bit precision,
/// and converts it to GBR YUV format with 8+-bit per channel precision.
///
/// # Arguments
///
/// * `image` - Target GBR image.
/// * `rgba16` - A slice with RGBA data
/// * `rgba_stride` - The stride (components per row) for the RGBA plane.
/// * `bit_depth` - Only 10 and 12 is supported.
/// * `range` - Yuv values range.
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input RGB data are not valid based
/// on the specified width, height, and strides is provided.
///
pub fn rgba16_to_gbr16(
    image: &mut YuvPlanarImageMut<u16>,
    rgba16: &[u16],
    rgba_stride: u32,
    bit_depth: u32,
    range: YuvRange,
) -> Result<(), YuvError> {
    assert!(
        bit_depth == 10 || bit_depth == 12,
        "Only 10 and 12 bit supported"
    );
    if bit_depth == 10 {
        rgbx_to_gbr_impl::<u16, { YuvSourceChannels::Rgba as u8 }, 10>(
            image,
            rgba16,
            rgba_stride,
            range,
        )
    } else if bit_depth == 12 {
        rgbx_to_gbr_impl::<u16, { YuvSourceChannels::Rgba as u8 }, 12>(
            image,
            rgba16,
            rgba_stride,
            range,
        )
    } else {
        unreachable!();
    }
}

/// Convert BGR to YUV Identity Matrix ( aka 'GBR )
///
/// This function takes BGR image format data with 8+-bit precision,
/// and converts it to GBR YUV format with 8+-bit per channel precision.
///
/// # Arguments
///
/// * `image` - Target GBR image.
/// * `bgr16` - A slice with BGR data
/// * `bgr_stride` - The stride (components per row) for the BGR plane.
/// * `bit_depth` - Only 10 and 12 is supported.
/// * `range` - Yuv values range.
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input RGB data are not valid based
/// on the specified width, height, and strides is provided.
///
pub fn bgr16_to_gbr16(
    image: &mut YuvPlanarImageMut<u16>,
    bgr16: &[u16],
    bgr_stride: u32,
    bit_depth: u32,
    range: YuvRange,
) -> Result<(), YuvError> {
    assert!(
        bit_depth == 10 || bit_depth == 12,
        "Only 10 and 12 bit supported"
    );
    if bit_depth == 10 {
        rgbx_to_gbr_impl::<u16, { YuvSourceChannels::Bgr as u8 }, 10>(
            image, bgr16, bgr_stride, range,
        )
    } else if bit_depth == 12 {
        rgbx_to_gbr_impl::<u16, { YuvSourceChannels::Bgr as u8 }, 12>(
            image, bgr16, bgr_stride, range,
        )
    } else {
        unreachable!();
    }
}

/// Convert BGRA to YUV Identity Matrix ( aka 'GBR )
///
/// This function takes BGRA image format data with 8+-bit precision,
/// and converts it to GBR YUV format with 8+-bit per channel precision.
///
/// # Arguments
///
/// * `image` - Target GBR image.
/// * `bgra16` - A slice with BGRA data
/// * `bgra_stride` - The stride (components per row) for the BGRA plane.
/// * `bit_depth` - Only 10 and 12 is supported.
/// * `range` - Yuv values range.
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input RGB data are not valid based
/// on the specified width, height, and strides is provided.
///
pub fn bgra16_to_gbr16(
    image: &mut YuvPlanarImageMut<u16>,
    bgra16: &[u16],
    bgra_stride: u32,
    bit_depth: u32,
    range: YuvRange,
) -> Result<(), YuvError> {
    assert!(
        bit_depth == 10 || bit_depth == 12,
        "Only 10 and 12 bit supported"
    );
    if bit_depth == 10 {
        rgbx_to_gbr_impl::<u16, { YuvSourceChannels::Bgra as u8 }, 10>(
            image,
            bgra16,
            bgra_stride,
            range,
        )
    } else if bit_depth == 12 {
        rgbx_to_gbr_impl::<u16, { YuvSourceChannels::Bgra as u8 }, 12>(
            image,
            bgra16,
            bgra_stride,
            range,
        )
    } else {
        unreachable!();
    }
}
