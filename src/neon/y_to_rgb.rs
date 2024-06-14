/*
 * // Copyright (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

use crate::yuv_support::{CbCrInverseTransform, YuvChromaRange, YuvSourceChannels};
use std::arch::aarch64::{
    uint8x16x3_t, uint8x16x4_t, vcombine_u8, vdup_n_u8, vdupq_n_s16, vdupq_n_u8, vget_low_u8,
    vld1q_u8, vmaxq_s16, vmull_high_u8, vmull_u8, vqshrun_n_s16, vreinterpretq_s16_u16, vst3q_u8,
    vst4q_u8, vsubq_u8,
};

#[inline(always)]
pub unsafe fn neon_y_to_rgb_row<const DESTINATION_CHANNELS: u8>(
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

    let y_corr = vdupq_n_u8(range.bias_y as u8);
    let v_luma_coeff = vdupq_n_u8(transform.y_coef as u8);
    let v_luma_coeff_8 = vdup_n_u8(transform.y_coef as u8);
    let v_min_values = vdupq_n_s16(0i16);
    let v_alpha = vdupq_n_u8(255u8);

    let mut cx = start_cx;

    while cx + 16 < width as usize {
        let y_values = vsubq_u8(vld1q_u8(y_ptr.add(y_offset + cx)), y_corr);

        let y_high = vreinterpretq_s16_u16(vmull_high_u8(y_values, v_luma_coeff));

        let r_high = vqshrun_n_s16::<6>(vmaxq_s16(y_high, v_min_values));

        let y_low = vreinterpretq_s16_u16(vmull_u8(vget_low_u8(y_values), v_luma_coeff_8));

        let r_low = vqshrun_n_s16::<6>(vmaxq_s16(y_low, v_min_values));

        let r_values = vcombine_u8(r_low, r_high);

        let dst_shift = rgba_offset + cx * channels;

        match destination_channels {
            YuvSourceChannels::Rgb => {
                let dst_pack: uint8x16x3_t = uint8x16x3_t(r_values, r_values, r_values);
                vst3q_u8(rgba_ptr.add(dst_shift), dst_pack);
            }
            YuvSourceChannels::Rgba => {
                let dst_pack: uint8x16x4_t = uint8x16x4_t(r_values, r_values, r_values, v_alpha);
                vst4q_u8(rgba_ptr.add(dst_shift), dst_pack);
            }
            YuvSourceChannels::Bgra => {
                let dst_pack: uint8x16x4_t = uint8x16x4_t(r_values, r_values, r_values, v_alpha);
                vst4q_u8(rgba_ptr.add(dst_shift), dst_pack);
            }
        }

        cx += 16;
    }

    cx
}
