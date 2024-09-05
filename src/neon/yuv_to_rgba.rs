/*
 * // Copyright (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

use crate::internals::ProcessedOffset;
use crate::yuv_support::{
    CbCrInverseTransform, YuvChromaRange, YuvChromaSample, YuvSourceChannels,
};
use std::arch::aarch64::*;

#[inline(always)]
pub unsafe fn neon_yuv_to_rgba_row<const DESTINATION_CHANNELS: u8, const SAMPLING: u8>(
    range: &YuvChromaRange,
    transform: &CbCrInverseTransform<i32>,
    y_plane: &[u8],
    u_plane: &[u8],
    v_plane: &[u8],
    rgba: &mut [u8],
    start_cx: usize,
    start_ux: usize,
    y_offset: usize,
    u_offset: usize,
    v_offset: usize,
    rgba_offset: usize,
    width: usize,
) -> ProcessedOffset {
    let chroma_subsampling: YuvChromaSample = SAMPLING.into();
    let destination_channels: YuvSourceChannels = DESTINATION_CHANNELS.into();
    let channels = destination_channels.get_channels_count();

    let mut cx = start_cx;
    let mut uv_x = start_ux;

    let y_ptr = y_plane.as_ptr();
    let u_ptr = u_plane.as_ptr();
    let v_ptr = v_plane.as_ptr();
    let rgba_ptr = rgba.as_mut_ptr();

    let y_corr = vdupq_n_u8(range.bias_y as u8);
    let uv_corr = vdupq_n_s16(range.bias_uv as i16);
    let v_luma_coeff = vdupq_n_u8(transform.y_coef as u8);
    let v_cr_coeff = vdupq_n_s16(transform.cr_coef as i16);
    let v_cb_coeff = vdupq_n_s16(transform.cb_coef as i16);
    let v_min_values = vdupq_n_s16(0i16);
    let v_g_coeff_1 = vdupq_n_s16(-1i16 * transform.g_coeff_1 as i16);
    let v_g_coeff_2 = vdupq_n_s16(-1i16 * transform.g_coeff_2 as i16);
    let v_alpha = vdupq_n_u8(255u8);

    while cx + 16 < width {
        let y_values = vqsubq_u8(vld1q_u8(y_ptr.add(y_offset + cx)), y_corr);

        let u_high_u8: uint8x8_t;
        let v_high_u8: uint8x8_t;
        let u_low_u8: uint8x8_t;
        let v_low_u8: uint8x8_t;

        match chroma_subsampling {
            YuvChromaSample::YUV420 | YuvChromaSample::YUV422 => {
                let u_values = vld1_u8(u_ptr.add(u_offset + uv_x));
                let v_values = vld1_u8(v_ptr.add(v_offset + uv_x));

                u_high_u8 = vzip2_u8(u_values, u_values);
                v_high_u8 = vzip2_u8(v_values, v_values);
                u_low_u8 = vzip1_u8(u_values, u_values);
                v_low_u8 = vzip1_u8(v_values, v_values);
            }
            YuvChromaSample::YUV444 => {
                let u_values = vld1q_u8(u_ptr.add(u_offset + uv_x));
                let v_values = vld1q_u8(v_ptr.add(v_offset + uv_x));

                u_high_u8 = vget_high_u8(u_values);
                v_high_u8 = vget_high_u8(v_values);
                u_low_u8 = vget_low_u8(u_values);
                v_low_u8 = vget_low_u8(v_values);
            }
        }

        let u_high = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(u_high_u8)), uv_corr);
        let v_high = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(v_high_u8)), uv_corr);
        let y_high = vreinterpretq_s16_u16(vmull_high_u8(y_values, v_luma_coeff));

        let r_high = vqrshrun_n_s16::<6>(vmaxq_s16(
            vqaddq_s16(y_high, vmulq_s16(v_high, v_cr_coeff)),
            v_min_values,
        ));
        let b_high = vqrshrun_n_s16::<6>(vmaxq_s16(
            vqaddq_s16(y_high, vmulq_s16(u_high, v_cb_coeff)),
            v_min_values,
        ));
        let g_high = vqrshrun_n_s16::<6>(vmaxq_s16(
            vqaddq_s16(
                y_high,
                vqaddq_s16(
                    vmulq_s16(v_high, v_g_coeff_1),
                    vmulq_s16(u_high, v_g_coeff_2),
                ),
            ),
            v_min_values,
        ));

        let u_low = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(u_low_u8)), uv_corr);
        let v_low = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(v_low_u8)), uv_corr);
        let y_low =
            vreinterpretq_s16_u16(vmull_u8(vget_low_u8(y_values), vget_low_u8(v_luma_coeff)));

        let r_low = vqrshrun_n_s16::<6>(vmaxq_s16(
            vqaddq_s16(y_low, vmulq_s16(v_low, v_cr_coeff)),
            v_min_values,
        ));
        let b_low = vqrshrun_n_s16::<6>(vmaxq_s16(
            vqaddq_s16(y_low, vmulq_s16(u_low, v_cb_coeff)),
            v_min_values,
        ));
        let g_low = vqrshrun_n_s16::<6>(vmaxq_s16(
            vqaddq_s16(
                y_low,
                vqaddq_s16(vmulq_s16(v_low, v_g_coeff_1), vmulq_s16(u_low, v_g_coeff_2)),
            ),
            v_min_values,
        ));

        let r_values = vcombine_u8(r_low, r_high);
        let g_values = vcombine_u8(g_low, g_high);
        let b_values = vcombine_u8(b_low, b_high);

        let dst_shift = rgba_offset + cx * channels;

        match destination_channels {
            YuvSourceChannels::Rgb => {
                let dst_pack: uint8x16x3_t = uint8x16x3_t(r_values, g_values, b_values);
                vst3q_u8(rgba_ptr.add(dst_shift), dst_pack);
            }
            YuvSourceChannels::Bgr => {
                let dst_pack: uint8x16x3_t = uint8x16x3_t(b_values, g_values, r_values);
                vst3q_u8(rgba_ptr.add(dst_shift), dst_pack);
            }
            YuvSourceChannels::Rgba => {
                let dst_pack: uint8x16x4_t = uint8x16x4_t(r_values, g_values, b_values, v_alpha);
                vst4q_u8(rgba_ptr.add(dst_shift), dst_pack);
            }
            YuvSourceChannels::Bgra => {
                let dst_pack: uint8x16x4_t = uint8x16x4_t(b_values, g_values, r_values, v_alpha);
                vst4q_u8(rgba_ptr.add(dst_shift), dst_pack);
            }
        }

        cx += 16;

        match chroma_subsampling {
            YuvChromaSample::YUV420 | YuvChromaSample::YUV422 => {
                uv_x += 8;
            }
            YuvChromaSample::YUV444 => {
                uv_x += 16;
            }
        }
    }

    while cx + 8 < width {
        let y_values = vqsub_u8(vld1_u8(y_ptr.add(y_offset + cx)), vget_low_u8(y_corr));

        let u_low_u8: uint8x8_t;
        let v_low_u8: uint8x8_t;

        match chroma_subsampling {
            YuvChromaSample::YUV420 | YuvChromaSample::YUV422 => {
                let u_values =
                    vreinterpret_u8_u32(vld1_dup_u32(u_ptr.add(u_offset + uv_x) as *const u32));
                let v_values =
                    vreinterpret_u8_u32(vld1_dup_u32(v_ptr.add(v_offset + uv_x) as *const u32));

                u_low_u8 = vzip1_u8(u_values, u_values);
                v_low_u8 = vzip1_u8(v_values, v_values);
            }
            YuvChromaSample::YUV444 => {
                let u_values = vld1_u8(u_ptr.add(u_offset + uv_x));
                let v_values = vld1_u8(v_ptr.add(v_offset + uv_x));

                u_low_u8 = u_values;
                v_low_u8 = v_values;
            }
        }

        let u_low = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(u_low_u8)), uv_corr);
        let v_low = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(v_low_u8)), uv_corr);
        let y_low = vreinterpretq_s16_u16(vmull_u8(y_values, vget_low_u8(v_luma_coeff)));

        let r_low = vqrshrun_n_s16::<6>(vmaxq_s16(
            vqaddq_s16(y_low, vmulq_s16(v_low, v_cr_coeff)),
            v_min_values,
        ));
        let b_low = vqrshrun_n_s16::<6>(vmaxq_s16(
            vqaddq_s16(y_low, vmulq_s16(u_low, v_cb_coeff)),
            v_min_values,
        ));
        let g_low = vqrshrun_n_s16::<6>(vmaxq_s16(
            vqaddq_s16(
                y_low,
                vqaddq_s16(vmulq_s16(v_low, v_g_coeff_1), vmulq_s16(u_low, v_g_coeff_2)),
            ),
            v_min_values,
        ));

        let r_values = r_low;
        let g_values = g_low;
        let b_values = b_low;

        let dst_shift = rgba_offset + cx * channels;

        match destination_channels {
            YuvSourceChannels::Rgb => {
                let dst_pack: uint8x8x3_t = uint8x8x3_t(r_values, g_values, b_values);
                vst3_u8(rgba_ptr.add(dst_shift), dst_pack);
            }
            YuvSourceChannels::Bgr => {
                let dst_pack: uint8x8x3_t = uint8x8x3_t(b_values, g_values, r_values);
                vst3_u8(rgba_ptr.add(dst_shift), dst_pack);
            }
            YuvSourceChannels::Rgba => {
                let dst_pack: uint8x8x4_t =
                    uint8x8x4_t(r_values, g_values, b_values, vget_low_u8(v_alpha));
                vst4_u8(rgba_ptr.add(dst_shift), dst_pack);
            }
            YuvSourceChannels::Bgra => {
                let dst_pack: uint8x8x4_t =
                    uint8x8x4_t(b_values, g_values, r_values, vget_low_u8(v_alpha));
                vst4_u8(rgba_ptr.add(dst_shift), dst_pack);
            }
        }

        cx += 8;

        match chroma_subsampling {
            YuvChromaSample::YUV420 | YuvChromaSample::YUV422 => {
                uv_x += 4;
            }
            YuvChromaSample::YUV444 => {
                uv_x += 8;
            }
        }
    }

    ProcessedOffset { cx, ux: uv_x }
}
