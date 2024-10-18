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
use crate::neon::gbr_to_image_neon_p16;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use crate::sse::gbr_to_image_sse_p16;
use crate::yuv_support::YuvSourceChannels;

fn gbr_to_image_impl_p16<const DESTINATION_CHANNELS: u8>(
    gbr: &[u16],
    gbr_stride: u32,
    rgba: &mut [u16],
    rgba_stride: u32,
    bit_depth: u32,
    width: u32,
    height: u32,
) {
    let destination_channels: YuvSourceChannels = DESTINATION_CHANNELS.into();
    let channels = destination_channels.get_channels_count();

    let mut gbr_offset = 0usize;
    let mut rgba_offset = 0usize;

    let max_colors = (1 << bit_depth) as u16 - 1;

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    let _use_sse = std::arch::is_x86_feature_detected!("sse4.1");

    for _ in 0..height as usize {
        let mut _cx = 0usize;

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        unsafe {
            if _use_sse {
                _cx = gbr_to_image_sse_p16::<DESTINATION_CHANNELS>(
                    gbr.as_ptr(),
                    rgba.as_mut_ptr(),
                    bit_depth,
                    width,
                    _cx,
                );
            }
        }

        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        unsafe {
            _cx = gbr_to_image_neon_p16::<DESTINATION_CHANNELS>(
                gbr.as_ptr(),
                rgba.as_mut_ptr(),
                bit_depth,
                width,
                _cx,
            );
        }

        let gbr_src_ptr = unsafe { (gbr.as_ptr() as *const u8).add(gbr_offset) as *const u16 };
        let dst_ptr = unsafe { (rgba.as_ptr() as *mut u8).add(rgba_offset) as *mut u16 };

        for x in _cx..width as usize {
            unsafe {
                let gbr_local = gbr_src_ptr.add(x * 3);

                let g = gbr_local.add(0).read_unaligned();
                let b = gbr_local.add(1).read_unaligned();
                let r = gbr_local.add(2).read_unaligned();

                let px = x * channels;

                let rgba_shift = px;
                let dst_local = dst_ptr.add(rgba_shift);

                dst_local
                    .add(destination_channels.get_r_channel_offset())
                    .write_unaligned(r);
                dst_local
                    .add(destination_channels.get_g_channel_offset())
                    .write_unaligned(g);
                dst_local
                    .add(destination_channels.get_b_channel_offset())
                    .write_unaligned(b);
                if destination_channels.has_alpha() {
                    dst_local
                        .add(destination_channels.get_a_channel_offset())
                        .write_unaligned(max_colors);
                }
            }
        }

        gbr_offset += gbr_stride as usize;
        rgba_offset += rgba_stride as usize;
    }
}

/// Convert YUV Identity Matrix ( aka 'GBR ) to RGB
///
/// This function takes GBR interleaved format data with 8+ bit precision,
/// and converts it to RGB format with 8+ bit per channel precision.
///
/// # Arguments
///
/// * `gbr` - A slice to load the GBR data.
/// * `gbr_stride` - The stride (bytes per row) for the GBR plane.
/// * `rgb` - A slice to store the RGB plane data.
/// * `rgb_stride` - The stride (bytes per row) for the RGB plane.
/// * `bit_depth` - YUV and RGB bit depth
/// * `width` - The width of the image.
/// * `height` - The height of the image.
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input RGB data are not valid based
/// on the specified width, height, and strides is provided.
///
pub fn gbr_to_rgb_p16(
    gbr: &[u16],
    gbr_stride: u32,
    rgb: &mut [u16],
    rgb_stride: u32,
    bit_depth: u32,
    width: u32,
    height: u32,
) {
    gbr_to_image_impl_p16::<{ YuvSourceChannels::Rgb as u8 }>(
        gbr, gbr_stride, rgb, rgb_stride, bit_depth, width, height,
    )
}

/// Convert YUV Identity Matrix ( aka 'GBR ) to BGR
///
/// This function takes GBR interleaved format data with 8+ bit precision,
/// and converts it to BGR format with 8+ bit per channel precision.
///
/// # Arguments
///
/// * `gbr` - A slice to load the GBR data.
/// * `gbr_stride` - The stride (bytes per row) for the GBR plane.
/// * `bgr` - A slice to store the BGR plane data.
/// * `bgr_stride` - The stride (bytes per row) for the BGR plane.
/// * `bit_depth` - YUV and RGB bit depth
/// * `width` - The width of the image.
/// * `height` - The height of the image.
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGR data are not valid based
/// on the specified width, height, and strides is provided.
///
pub fn gbr_to_bgr_p16(
    gbr: &[u16],
    gbr_stride: u32,
    bgr: &mut [u16],
    bgr_stride: u32,
    bit_depth: u32,
    width: u32,
    height: u32,
) {
    gbr_to_image_impl_p16::<{ YuvSourceChannels::Bgr as u8 }>(
        gbr, gbr_stride, bgr, bgr_stride, bit_depth, width, height,
    )
}

/// Convert YUV Identity Matrix ( aka 'GBR ) to RGBA
///
/// This function takes GBR interleaved format data with 8+ bit precision,
/// and converts it to RGBA format with 8+ bit per channel precision.
///
/// # Arguments
///
/// * `gbr` - A slice to load the GBR data.
/// * `gbr_stride` - The stride (bytes per row) for the GBR plane.
/// * `rgba` - A slice to store the RGBA plane data.
/// * `rgba_stride` - The stride (bytes per row) for the RGBA plane.
/// * `bit_depth` - YUV and RGB bit depth
/// * `width` - The width of the image.
/// * `height` - The height of the image.
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input RGBA data are not valid based
/// on the specified width, height, and strides is provided.
///
pub fn gbr_to_rgba_p16(
    gbr: &[u16],
    gbr_stride: u32,
    rgb: &mut [u16],
    rgb_stride: u32,
    bit_depth: u32,
    width: u32,
    height: u32,
) {
    gbr_to_image_impl_p16::<{ YuvSourceChannels::Rgba as u8 }>(
        gbr, gbr_stride, rgb, rgb_stride, bit_depth, width, height,
    )
}

/// Convert YUV Identity Matrix ( aka 'GBR ) to BGRA
///
/// This function takes GBR interleaved format data with 8+ bit precision,
/// and converts it to BGRA format with 8+ bit per channel precision.
///
/// # Arguments
///
/// * `gbr` - A slice to load the GBR data.
/// * `gbr_stride` - The stride (bytes per row) for the GBR plane.
/// * `rgba` - A slice to store the BGRA plane data.
/// * `rgba_stride` - The stride (bytes per row) for the BGRA plane.
/// * `bit_depth` - YUV and RGB bit depth
/// * `width` - The width of the image.
/// * `height` - The height of the image.
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGRA data are not valid based
/// on the specified width, height, and strides is provided.
///
pub fn gbr_to_bgra_p16(
    gbr: &[u16],
    gbr_stride: u32,
    rgb: &mut [u16],
    rgb_stride: u32,
    bit_depth: u32,
    width: u32,
    height: u32,
) {
    gbr_to_image_impl_p16::<{ YuvSourceChannels::Rgba as u8 }>(
        gbr, gbr_stride, rgb, rgb_stride, bit_depth, width, height,
    )
}
