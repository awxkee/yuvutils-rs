/*
 * // Copyright (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */
use crate::yuv_support::YuvSourceChannels;
use std::arch::aarch64::*;

pub unsafe fn gbr_to_image_neon_p16<const DESTINATION_CHANNELS: u8>(
    gbr: *const u16,
    rgb: *mut u16,
    bit_depth: u32,
    width: u32,
    start_cx: usize,
) -> usize {
    let mut _cx = start_cx;

    let destination_channels: YuvSourceChannels = DESTINATION_CHANNELS.into();
    let channels = destination_channels.get_channels_count();

    let mut gbr_start_ptr = gbr.add(3 * _cx);
    let mut rgb_start_ptr = rgb.add(channels * _cx);

    let v_max_colors = vdupq_n_u16(1 << bit_depth - 1);

    let rgb_part_size = channels * 8;

    while _cx + 8 < width as usize {
        let gbr_pixel = vld3q_u16(gbr_start_ptr);

        let r_pixel = gbr_pixel.2;
        let g_pixel = gbr_pixel.0;
        let b_pixel = gbr_pixel.1;

        match destination_channels {
            YuvSourceChannels::Rgb => {
                vst3q_u16(rgb_start_ptr, uint16x8x3_t(r_pixel, g_pixel, b_pixel));
            }
            YuvSourceChannels::Rgba => {
                vst4q_u16(
                    rgb_start_ptr,
                    uint16x8x4_t(r_pixel, g_pixel, b_pixel, v_max_colors),
                );
            }
            YuvSourceChannels::Bgra => {
                vst4q_u16(
                    rgb_start_ptr,
                    uint16x8x4_t(b_pixel, g_pixel, r_pixel, v_max_colors),
                );
            }
            YuvSourceChannels::Bgr => {
                vst3q_u16(rgb_start_ptr, uint16x8x3_t(b_pixel, g_pixel, r_pixel));
            }
        }

        gbr_start_ptr = gbr_start_ptr.add(3 * 8);
        rgb_start_ptr = rgb_start_ptr.add(rgb_part_size);
        _cx += 8;
    }

    let rgb_part_size_small = channels * 4;

    while _cx + 4 < width as usize {
        let gbr_pixel = vld3_u16(gbr_start_ptr);

        let r_pixel = gbr_pixel.2;
        let g_pixel = gbr_pixel.0;
        let b_pixel = gbr_pixel.1;

        match destination_channels {
            YuvSourceChannels::Rgb => {
                vst3_u16(rgb_start_ptr, uint16x4x3_t(r_pixel, g_pixel, b_pixel));
            }
            YuvSourceChannels::Rgba => {
                vst4_u16(
                    rgb_start_ptr,
                    uint16x4x4_t(r_pixel, g_pixel, b_pixel, vget_low_u16(v_max_colors)),
                );
            }
            YuvSourceChannels::Bgra => {
                vst4_u16(
                    rgb_start_ptr,
                    uint16x4x4_t(b_pixel, g_pixel, r_pixel, vget_low_u16(v_max_colors)),
                );
            }
            YuvSourceChannels::Bgr => {
                vst3_u16(rgb_start_ptr, uint16x4x3_t(b_pixel, g_pixel, r_pixel));
            }
        }

        gbr_start_ptr = gbr_start_ptr.add(3 * 4);
        rgb_start_ptr = rgb_start_ptr.add(rgb_part_size_small);
        _cx += 4;
    }

    _cx
}
