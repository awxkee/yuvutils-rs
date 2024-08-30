/*
 * // Copyright (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use crate::avx2::image_to_gbr_avx;
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use crate::neon::image_to_gbr_neon;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use crate::sse::image_to_gbr_sse;
use crate::yuv_support::YuvSourceChannels;

fn image_to_gbr<const SOURCE_CHANNELS: u8>(
    rgba: &[u8],
    rgba_stride: u32,
    gbr: &mut [u8],
    gbr_stride: u32,
    width: u32,
    height: u32,
) {
    let source_channels: YuvSourceChannels = SOURCE_CHANNELS.into();
    let channels = source_channels.get_channels_count();

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    let _use_sse = std::arch::is_x86_feature_detected!("sse4.1");
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    let _use_avx2 = std::arch::is_x86_feature_detected!("avx2");

    let mut gbr_offset = 0usize;
    let mut rgba_offset = 0usize;

    for _ in 0..height as usize {
        let mut _cx = 0usize;

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        unsafe {
            if _use_avx2 {
                _cx = image_to_gbr_avx::<SOURCE_CHANNELS>(
                    rgba,
                    rgba_offset,
                    gbr,
                    gbr_offset,
                    width,
                    _cx,
                );
            }
            if _use_sse {
                _cx = image_to_gbr_sse::<SOURCE_CHANNELS>(
                    rgba,
                    rgba_offset,
                    gbr,
                    gbr_offset,
                    width,
                    _cx,
                );
            }
        }

        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        unsafe {
            _cx = image_to_gbr_neon::<SOURCE_CHANNELS>(
                rgba,
                rgba_offset,
                gbr,
                gbr_offset,
                width,
                _cx,
            );
        }

        for x in _cx..width as usize {
            let px = x * channels;

            let rgba_shift = rgba_offset + px;

            let dst_local = unsafe { rgba.get_unchecked(rgba_shift..) };

            let r = unsafe { *dst_local.get_unchecked(source_channels.get_r_channel_offset()) };
            let g = unsafe { *dst_local.get_unchecked(source_channels.get_g_channel_offset()) };
            let b = unsafe { *dst_local.get_unchecked(source_channels.get_b_channel_offset()) };

            let gbr_local = unsafe { gbr.get_unchecked_mut((gbr_offset + x * 3)..) };

            unsafe {
                *gbr_local.get_unchecked_mut(0) = g;
                *gbr_local.get_unchecked_mut(1) = b;
                *gbr_local.get_unchecked_mut(2) = r;
            }
        }

        gbr_offset += gbr_stride as usize;
        rgba_offset += rgba_stride as usize;
    }
}

/// Convert RGB to YUV Identity Matrix ( aka 'GBR )
///
/// This function takes RGB image format data with 8-bit precision,
/// and converts it to GBR YUV format with 8-bit per channel precision.
///
/// # Arguments
///
/// * `rgb` - A slice to load the RGB plane data.
/// * `rgb_stride` - The stride (bytes per row) for the RGB plane.
/// * `gbr` - A slice to store the GBR data.
/// * `gbr_stride` - The stride (bytes per row) for the GBR plane.
/// * `width` - The width of the image.
/// * `height` - The height of the image.
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input RGB data are not valid based
/// on the specified width, height, and strides is provided.
///
pub fn rgb_to_gbr(
    rgb: &[u8],
    rgb_stride: u32,
    gbr: &mut [u8],
    gbr_stride: u32,
    width: u32,
    height: u32,
) {
    image_to_gbr::<{ YuvSourceChannels::Rgb as u8 }>(
        rgb, rgb_stride, gbr, gbr_stride, width, height,
    )
}

/// Convert BGR to YUV Identity Matrix ( aka 'GBR )
///
/// This function takes BGR image format data with 8-bit precision,
/// and converts it to GBR YUV format with 8-bit per channel precision.
///
/// # Arguments
///
/// * `bgr` - A slice to load the BGR plane data.
/// * `bgr_stride` - The stride (bytes per row) for the BGR plane.
/// * `gbr` - A slice to store the GBR data.
/// * `gbr_stride` - The stride (bytes per row) for the GBR plane.
/// * `width` - The width of the image.
/// * `height` - The height of the image.
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGR data are not valid based
/// on the specified width, height, and strides is provided.
///
pub fn bgr_to_gbr(
    bgr: &[u8],
    bgr_stride: u32,
    gbr: &mut [u8],
    gbr_stride: u32,
    width: u32,
    height: u32,
) {
    image_to_gbr::<{ YuvSourceChannels::Bgr as u8 }>(
        bgr, bgr_stride, gbr, gbr_stride, width, height,
    )
}

/// Convert BGRA to YUV Identity Matrix ( aka 'GBR )
///
/// This function takes BGRA image format data with 8-bit precision,
/// and converts it to GBR YUV format with 8-bit per channel precision.
///
/// # Arguments
///
/// * `bgra` - A slice to load the BGRA plane data.
/// * `bgra_stride` - The stride (bytes per row) for the BGRA plane.
/// * `gbr` - A slice to store the GBR data.
/// * `gbr_stride` - The stride (bytes per row) for the GBR plane.
/// * `width` - The width of the image.
/// * `height` - The height of the image.
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGRA data are not valid based
/// on the specified width, height, and strides is provided.
///
pub fn bgra_to_gbr(
    bgra: &[u8],
    bgra_stride: u32,
    gbr: &mut [u8],
    gbr_stride: u32,
    width: u32,
    height: u32,
) {
    image_to_gbr::<{ YuvSourceChannels::Bgra as u8 }>(
        bgra,
        bgra_stride,
        gbr,
        gbr_stride,
        width,
        height,
    )
}

/// Convert RGBA to YUV Identity Matrix ( aka 'GBR )
///
/// This function takes BGRA RGBA format data with 8-bit precision,
/// and converts it to GBR YUV format with 8-bit per channel precision.
///
/// # Arguments
///
/// * `rgba` - A slice to load the RGBA plane data.
/// * `rgba_stride` - The stride (bytes per row) for the RGBA plane.
/// * `gbr` - A slice to store the GBR data.
/// * `gbr_stride` - The stride (bytes per row) for the GBR plane.
/// * `width` - The width of the image.
/// * `height` - The height of the image.
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input RGBA data are not valid based
/// on the specified width, height, and strides is provided.
///
pub fn rgba_to_gbr(
    rgba: &[u8],
    rgba_stride: u32,
    gbr: &mut [u8],
    gbr_stride: u32,
    width: u32,
    height: u32,
) {
    image_to_gbr::<{ YuvSourceChannels::Rgba as u8 }>(
        rgba,
        rgba_stride,
        gbr,
        gbr_stride,
        width,
        height,
    )
}
