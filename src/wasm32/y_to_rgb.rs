/*
 * // Copyright (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

use crate::wasm32::transpose::{wasm_store_interleave_u8x3, wasm_store_interleave_u8x4};
use crate::wasm32::utils::u16x8_pack_sat_u8x16;
use crate::yuv_support::{CbCrInverseTransform, YuvChromaRange, YuvSourceChannels};
use std::arch::wasm32::*;

#[target_feature(enable = "simd128")]
pub unsafe fn wasm_y_to_rgb_row<const DESTINATION_CHANNELS: u8>(
    range: &YuvChromaRange,
    transform: &CbCrInverseTransform<i32>,
    y_plane: &[u8],
    rgba: &mut [u8],
    start_cx: usize,
    y_offset: usize,
    rgba_offset: usize,
    width: usize,
) -> usize {
    let destination_channels: YuvSourceChannels = DESTINATION_CHANNELS.into();
    let channels = destination_channels.get_channels_count();

    let y_ptr = y_plane.as_ptr();
    let rgba_ptr = rgba.as_mut_ptr();

    let y_corr = u8x16_splat(range.bias_y as u8);
    let v_luma_coeff = u8x16_splat(transform.y_coef as u8);
    let v_min_values = i16x8_splat(0i16);
    let v_alpha = u8x16_splat(255u8);

    let mut cx = start_cx;

    while cx + 16 < width as usize {
        let y_values = u8x16_sub(v128_load(y_ptr.add(y_offset + cx) as *const v128), y_corr);

        let y_high = u16x8_extmul_high_u8x16(y_values, v_luma_coeff);

        let r_high = i16x8_shr(i16x8_max(y_high, v_min_values), 6);

        let y_low = u16x8_extmul_low_u8x16(y_values, v_luma_coeff);

        let r_low = i16x8_shr(i16x8_max(y_low, v_min_values), 6);

        let r_values = u16x8_pack_sat_u8x16(r_low, r_high);

        let dst_shift = rgba_offset + cx * channels;

        match destination_channels {
            YuvSourceChannels::Rgb => {
                let dst_pack = (r_values, r_values, r_values);
                wasm_store_interleave_u8x3(rgba_ptr.add(dst_shift), dst_pack);
            }
            YuvSourceChannels::Rgba => {
                let dst_pack = (r_values, r_values, r_values, v_alpha);
                wasm_store_interleave_u8x4(rgba_ptr.add(dst_shift), dst_pack);
            }
            YuvSourceChannels::Bgra => {
                let dst_pack = (r_values, r_values, r_values, v_alpha);
                wasm_store_interleave_u8x4(rgba_ptr.add(dst_shift), dst_pack);
            }
        }

        cx += 16;
    }

    cx
}
