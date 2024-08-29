/*
 * // Copyright (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */
use crate::yuv_support::YuvSourceChannels;
use std::arch::aarch64::*;

pub unsafe fn gbr_to_image_neon<const DESTINATION_CHANNELS: u8>(
    gbr: &[u8],
    gbr_offset: usize,
    rgb: &mut [u8],
    rgb_offset: usize,
    width: u32,
    start_cx: usize,
) -> usize {
    let mut _cx = start_cx;

    let destination_channels: YuvSourceChannels = DESTINATION_CHANNELS.into();
    let channels = destination_channels.get_channels_count();

    let mut gbr_start_ptr = gbr.as_ptr().add(gbr_offset);
    let mut rgb_start_ptr = rgb.as_mut_ptr().add(rgb_offset);

    let rgb_part_size = channels * 16;

    while _cx + 16 < width as usize {
        let gbr_pixel = vld3q_u8(gbr_start_ptr);

        let r_pixel = gbr_pixel.2;
        let g_pixel = gbr_pixel.0;
        let b_pixel = gbr_pixel.1;

        match destination_channels {
            YuvSourceChannels::Rgb => {
                vst3q_u8(rgb_start_ptr, uint8x16x3_t(r_pixel, g_pixel, b_pixel));
            }
            YuvSourceChannels::Rgba => {
                let a_pixel = vdupq_n_u8(255);
                vst4q_u8(
                    rgb_start_ptr,
                    uint8x16x4_t(r_pixel, g_pixel, b_pixel, a_pixel),
                );
            }
            YuvSourceChannels::Bgra => {
                let a_pixel = vdupq_n_u8(255);
                vst4q_u8(
                    rgb_start_ptr,
                    uint8x16x4_t(b_pixel, g_pixel, r_pixel, a_pixel),
                );
            }
            YuvSourceChannels::Bgr => {
                vst3q_u8(rgb_start_ptr, uint8x16x3_t(b_pixel, g_pixel, r_pixel));
            }
        }

        gbr_start_ptr = gbr_start_ptr.add(3 * 16);
        rgb_start_ptr = rgb_start_ptr.add(rgb_part_size);
        _cx += 16;
    }

    let rgb_part_size_small = channels * 8;

    while _cx + 8 < width as usize {
        let gbr_pixel = vld3_u8(gbr_start_ptr);

        let r_pixel = gbr_pixel.2;
        let g_pixel = gbr_pixel.0;
        let b_pixel = gbr_pixel.1;

        match destination_channels {
            YuvSourceChannels::Rgb => {
                vst3_u8(rgb_start_ptr, uint8x8x3_t(r_pixel, g_pixel, b_pixel));
            }
            YuvSourceChannels::Rgba => {
                let a_pixel = vdup_n_u8(255);
                vst4_u8(
                    rgb_start_ptr,
                    uint8x8x4_t(r_pixel, g_pixel, b_pixel, a_pixel),
                );
            }
            YuvSourceChannels::Bgra => {
                let a_pixel = vdup_n_u8(255);
                vst4_u8(
                    rgb_start_ptr,
                    uint8x8x4_t(b_pixel, g_pixel, r_pixel, a_pixel),
                );
            }
            YuvSourceChannels::Bgr => {
                vst3_u8(rgb_start_ptr, uint8x8x3_t(b_pixel, g_pixel, r_pixel));
            }
        }

        gbr_start_ptr = gbr_start_ptr.add(3 * 8);
        rgb_start_ptr = rgb_start_ptr.add(rgb_part_size_small);
        _cx += 8;
    }

    _cx
}
