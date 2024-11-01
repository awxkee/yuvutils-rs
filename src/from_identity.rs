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
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use crate::avx2::gbr_to_image_avx;
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use crate::neon::gbr_to_image_neon;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use crate::sse::gbr_to_image_sse;
use crate::yuv_support::YuvSourceChannels;

fn gbr_to_image_impl<const DESTINATION_CHANNELS: u8>(
    source_gbr: &[u8],
    gbr_stride: u32,
    rgba: &mut [u8],
    rgba_stride: u32,
    width: u32,
    _: u32,
) {
    let destination_channels: YuvSourceChannels = DESTINATION_CHANNELS.into();
    let channels = destination_channels.get_channels_count();

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    let _use_sse = std::arch::is_x86_feature_detected!("sse4.1");
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    let _use_avx = std::arch::is_x86_feature_detected!("avx2");

    for (dst_row, src_row) in rgba
        .chunks_exact_mut(rgba_stride as usize)
        .zip(source_gbr.chunks_exact(gbr_stride as usize))
    {
        let mut _cx = 0usize;

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        unsafe {
            if _use_avx {
                _cx = gbr_to_image_avx::<DESTINATION_CHANNELS>(src_row, 0, dst_row, 0, width, _cx);
            }
            if _use_sse {
                _cx = gbr_to_image_sse::<DESTINATION_CHANNELS>(src_row, 0, dst_row, 0, width, _cx);
            }
        }

        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        unsafe {
            _cx = gbr_to_image_neon::<DESTINATION_CHANNELS>(src_row, 0, dst_row, 0, width, _cx);
        }

        for (dst, src) in dst_row
            .chunks_exact_mut(channels)
            .zip(src_row.chunks_exact(3))
            .skip(_cx)
        {
            let g = src[0];
            let b = src[1];
            let r = src[2];
            dst[destination_channels.get_r_channel_offset()] = r;
            dst[destination_channels.get_g_channel_offset()] = g;
            dst[destination_channels.get_b_channel_offset()] = b;
            if destination_channels.has_alpha() {
                dst[destination_channels.get_a_channel_offset()] = 255;
            }
        }
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
