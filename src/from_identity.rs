/*
 * // Copyright (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use crate::avx2::gbr_to_image_avx;
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use crate::neon::gbr_to_image_neon;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use crate::sse::gbr_to_image_sse;
use crate::yuv_support::YuvSourceChannels;

fn gbr_to_image_impl<const DESTINATION_CHANNELS: u8>(
    gbr: &[u8],
    gbr_stride: u32,
    rgba: &mut [u8],
    rgba_stride: u32,
    width: u32,
    height: u32,
) {
    let destination_channels: YuvSourceChannels = DESTINATION_CHANNELS.into();
    let channels = destination_channels.get_channels_count();

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    let _use_sse = std::arch::is_x86_feature_detected!("sse4.1");
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    let _use_avx = std::arch::is_x86_feature_detected!("avx2");

    let mut gbr_offset = 0usize;
    let mut rgba_offset = 0usize;

    for _ in 0..height as usize {
        let mut _cx = 0usize;

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        unsafe {
            if _use_avx {
                _cx = gbr_to_image_avx::<DESTINATION_CHANNELS>(
                    gbr,
                    gbr_offset,
                    rgba,
                    rgba_offset,
                    width,
                    _cx,
                );
            }
            if _use_sse {
                _cx = gbr_to_image_sse::<DESTINATION_CHANNELS>(
                    gbr,
                    gbr_offset,
                    rgba,
                    rgba_offset,
                    width,
                    _cx,
                );
            }
        }

        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        unsafe {
            _cx = gbr_to_image_neon::<DESTINATION_CHANNELS>(
                gbr,
                gbr_offset,
                rgba,
                rgba_offset,
                width,
                _cx,
            );
        }

        for x in _cx..width as usize {
            let gbr_local = unsafe { gbr.get_unchecked((gbr_offset + x * 3)..) };

            let g = unsafe { *gbr_local.get_unchecked(0) };
            let b = unsafe { *gbr_local.get_unchecked(1) };
            let r = unsafe { *gbr_local.get_unchecked(2) };

            let px = x * channels;

            let rgba_shift = rgba_offset + px;

            let dst_local = unsafe { rgba.get_unchecked_mut(rgba_shift..) };

            unsafe {
                *dst_local.get_unchecked_mut(destination_channels.get_r_channel_offset()) = r
            };
            unsafe {
                *dst_local.get_unchecked_mut(destination_channels.get_g_channel_offset()) = g
            };
            unsafe {
                *dst_local.get_unchecked_mut(destination_channels.get_b_channel_offset()) = b
            };
            if destination_channels.has_alpha() {
                unsafe {
                    *dst_local.get_unchecked_mut(destination_channels.get_a_channel_offset()) = 255
                };
            }
        }

        gbr_offset += gbr_stride as usize;
        rgba_offset += rgba_stride as usize;
    }
}

/// Convert YUV Identity Matrix ( aka 'GBR ) to RGB
///
/// This function takes GBR interleaved format data with 8-bit precision,
/// and converts it to RGB format with 8-bit per channel precision.
///
/// # Arguments
///
/// * `gbr` - A slice to load the GBR data.
/// * `gbr_stride` - The stride (bytes per row) for the GBR plane.
/// * `rgb` - A slice to store the RGB plane data.
/// * `rgb_stride` - The stride (bytes per row) for the RGB plane.
/// * `width` - The width of the image.
/// * `height` - The height of the image.
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input RGB data are not valid based
/// on the specified width, height, and strides is provided.
///
pub fn gbr_to_rgb(
    gbr: &[u8],
    gbr_stride: u32,
    rgb: &mut [u8],
    rgb_stride: u32,
    width: u32,
    height: u32,
) {
    gbr_to_image_impl::<{ YuvSourceChannels::Rgb as u8 }>(
        gbr, gbr_stride, rgb, rgb_stride, width, height,
    )
}

/// Convert YUV Identity Matrix ( aka 'GBR ) to BGR
///
/// This function takes GBR interleaved format data with 8-bit precision,
/// and converts it to BGR format with 8-bit per channel precision.
///
/// # Arguments
///
/// * `gbr` - A slice to load the GBR data.
/// * `gbr_stride` - The stride (bytes per row) for the GBR plane.
/// * `bgr` - A slice to store the BGR plane data.
/// * `bgr_stride` - The stride (bytes per row) for the BGR plane.
/// * `width` - The width of the image.
/// * `height` - The height of the image.
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGR data are not valid based
/// on the specified width, height, and strides is provided.
///
pub fn gbr_to_bgr(
    gbr: &[u8],
    gbr_stride: u32,
    bgr: &mut [u8],
    bgr_stride: u32,
    width: u32,
    height: u32,
) {
    gbr_to_image_impl::<{ YuvSourceChannels::Bgr as u8 }>(
        gbr, gbr_stride, bgr, bgr_stride, width, height,
    )
}

/// Convert YUV Identity Matrix ( aka 'GBR ) to RGBA
///
/// This function takes GBR interleaved format data with 8-bit precision,
/// and converts it to RGBA format with 8-bit per channel precision.
///
/// # Arguments
///
/// * `gbr` - A slice to load the GBR data.
/// * `gbr_stride` - The stride (bytes per row) for the GBR plane.
/// * `rgba` - A slice to store the RGBA plane data.
/// * `rgba_stride` - The stride (bytes per row) for the RGBA plane.
/// * `width` - The width of the image.
/// * `height` - The height of the image.
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input RGBA data are not valid based
/// on the specified width, height, and strides is provided.
///
pub fn gbr_to_rgba(
    gbr: &[u8],
    gbr_stride: u32,
    rgb: &mut [u8],
    rgb_stride: u32,
    width: u32,
    height: u32,
) {
    gbr_to_image_impl::<{ YuvSourceChannels::Rgba as u8 }>(
        gbr, gbr_stride, rgb, rgb_stride, width, height,
    )
}

/// Convert YUV Identity Matrix ( aka 'GBR ) to BGRA
///
/// This function takes GBR interleaved format data with 8-bit precision,
/// and converts it to BGRA format with 8-bit per channel precision.
///
/// # Arguments
///
/// * `gbr` - A slice to load the GBR data.
/// * `gbr_stride` - The stride (bytes per row) for the GBR plane.
/// * `rgba` - A slice to store the BGRA plane data.
/// * `rgba_stride` - The stride (bytes per row) for the BGRA plane.
/// * `width` - The width of the image.
/// * `height` - The height of the image.
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGRA data are not valid based
/// on the specified width, height, and strides is provided.
///
pub fn gbr_to_bgra(
    gbr: &[u8],
    gbr_stride: u32,
    rgb: &mut [u8],
    rgb_stride: u32,
    width: u32,
    height: u32,
) {
    gbr_to_image_impl::<{ YuvSourceChannels::Rgba as u8 }>(
        gbr, gbr_stride, rgb, rgb_stride, width, height,
    )
}
