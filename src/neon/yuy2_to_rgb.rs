/*
 * // Copyright (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */
use crate::yuv_support::{
    CbCrInverseTransform, YuvChromaRange, YuvSourceChannels, Yuy2Description,
};
use crate::yuv_to_yuy2::YuvToYuy2Navigation;
use std::arch::aarch64::*;

pub fn yuy2_to_rgb_neon<const DST_CHANNELS: u8, const YUY2_TARGET: usize>(
    range: &YuvChromaRange,
    transform: &CbCrInverseTransform<i32>,
    yuy2_store: &[u8],
    yuy2_offset: usize,
    rgb: &mut [u8],
    rgb_offset: usize,
    width: u32,
    nav: YuvToYuy2Navigation,
) -> YuvToYuy2Navigation {
    let yuy2_source: Yuy2Description = YUY2_TARGET.into();
    let dst_chans: YuvSourceChannels = DST_CHANNELS.into();

    let mut _cx = nav.cx;
    let mut _yuy2_x = nav.x;

    unsafe {
        let max_x_16 = (width.saturating_sub(1) as usize / 2).saturating_sub(16);
        let max_x_8 = (width.saturating_sub(1) as usize / 2).saturating_sub(8);

        let y_corr = vdupq_n_u8(range.bias_y as u8);
        let uv_corr = vdupq_n_s16(range.bias_uv as i16);
        let v_luma_coeff = vdupq_n_u8(transform.y_coef as u8);
        let v_cr_coeff = vdupq_n_s16(transform.cr_coef as i16);
        let v_cb_coeff = vdupq_n_s16(transform.cb_coef as i16);
        let v_min_values = vdupq_n_s16(0i16);
        let v_g_coeff_1 = vdupq_n_s16(-1i16 * transform.g_coeff_1 as i16);
        let v_g_coeff_2 = vdupq_n_s16(-1i16 * transform.g_coeff_2 as i16);
        let v_alpha = vdupq_n_u8(255u8);

        for x in (_yuy2_x..max_x_16).step_by(16) {
            let dst_offset = yuy2_offset + x * 4;
            let dst_pos = rgb_offset + _cx * dst_chans.get_channels_count();
            let dst_ptr = rgb.as_mut_ptr().add(dst_pos);

            let pixel_set = vld4q_u8(yuy2_store.as_ptr().add(dst_offset));
            let mut y_first = match yuy2_source {
                Yuy2Description::YUYV | Yuy2Description::YVYU => pixel_set.0,
                Yuy2Description::UYVY | Yuy2Description::VYUY => pixel_set.1,
            };
            let mut y_second = match yuy2_source {
                Yuy2Description::YUYV | Yuy2Description::YVYU => pixel_set.2,
                Yuy2Description::UYVY | Yuy2Description::VYUY => pixel_set.3,
            };

            let y_first_reconstructed = vzip1q_u8(y_first, y_second);
            let y_second_reconstructed = vzip2q_u8(y_first, y_second);
            y_first = y_first_reconstructed;
            y_second = y_second_reconstructed;

            let u_value = match yuy2_source {
                Yuy2Description::YUYV => pixel_set.1,
                Yuy2Description::UYVY => pixel_set.0,
                Yuy2Description::YVYU => pixel_set.3,
                Yuy2Description::VYUY => pixel_set.2,
            };
            let v_value = match yuy2_source {
                Yuy2Description::YUYV => pixel_set.3,
                Yuy2Description::UYVY => pixel_set.2,
                Yuy2Description::YVYU => pixel_set.1,
                Yuy2Description::VYUY => pixel_set.0,
            };

            let low_u_value = vzip1q_u8(u_value, u_value);
            let high_u_value = vzip2q_u8(u_value, u_value);
            let low_v_value = vzip1q_u8(v_value, v_value);
            let high_v_value = vzip2q_u8(v_value, v_value);

            y_first = vsubq_u8(y_first, y_corr);
            y_second = vsubq_u8(y_second, y_corr);

            let u_l_h = vsubq_s16(vreinterpretq_s16_u16(vmovl_high_u8(low_u_value)), uv_corr);
            let v_l_h = vsubq_s16(vreinterpretq_s16_u16(vmovl_high_u8(low_v_value)), uv_corr);
            let y_l_h = vreinterpretq_s16_u16(vmull_high_u8(y_first, v_luma_coeff));

            let r_l_h = vqshrun_n_s16::<6>(vmaxq_s16(
                vqaddq_s16(y_l_h, vmulq_s16(v_l_h, v_cr_coeff)),
                v_min_values,
            ));
            let b_l_h = vqshrun_n_s16::<6>(vmaxq_s16(
                vqaddq_s16(y_l_h, vmulq_s16(u_l_h, v_cb_coeff)),
                v_min_values,
            ));
            let g_l_h = vqshrun_n_s16::<6>(vmaxq_s16(
                vqaddq_s16(
                    y_l_h,
                    vqaddq_s16(vmulq_s16(v_l_h, v_g_coeff_1), vmulq_s16(u_l_h, v_g_coeff_2)),
                ),
                v_min_values,
            ));

            let u_low = vsubq_s16(
                vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(low_u_value))),
                uv_corr,
            );
            let v_low = vsubq_s16(
                vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(low_v_value))),
                uv_corr,
            );
            let y_low =
                vreinterpretq_s16_u16(vmull_u8(vget_low_u8(y_first), vget_low_u8(v_luma_coeff)));

            let r_l_l = vqshrun_n_s16::<6>(vmaxq_s16(
                vqaddq_s16(y_low, vmulq_s16(v_low, v_cr_coeff)),
                v_min_values,
            ));
            let b_l_l = vqshrun_n_s16::<6>(vmaxq_s16(
                vqaddq_s16(y_low, vmulq_s16(u_low, v_cb_coeff)),
                v_min_values,
            ));
            let g_l_l = vqshrun_n_s16::<6>(vmaxq_s16(
                vqaddq_s16(
                    y_low,
                    vqaddq_s16(vmulq_s16(v_low, v_g_coeff_1), vmulq_s16(u_low, v_g_coeff_2)),
                ),
                v_min_values,
            ));

            let r_l = vcombine_u8(r_l_l, r_l_h);
            let g_l = vcombine_u8(g_l_l, g_l_h);
            let b_l = vcombine_u8(b_l_l, b_l_h);

            match dst_chans {
                YuvSourceChannels::Rgb => {
                    let packed = uint8x16x3_t(r_l, g_l, b_l);
                    vst3q_u8(dst_ptr, packed);
                }
                YuvSourceChannels::Rgba => {
                    let packed = uint8x16x4_t(r_l, g_l, b_l, v_alpha);
                    vst4q_u8(dst_ptr, packed);
                }
                YuvSourceChannels::Bgra => {
                    let packed = uint8x16x4_t(b_l, g_l, r_l, v_alpha);
                    vst4q_u8(dst_ptr, packed);
                }
                YuvSourceChannels::Bgr => {
                    let packed = uint8x16x3_t(b_l, g_l, r_l);
                    vst3q_u8(dst_ptr, packed);
                }
            }

            let u_h_h = vsubq_s16(vreinterpretq_s16_u16(vmovl_high_u8(high_u_value)), uv_corr);
            let v_h_h = vsubq_s16(vreinterpretq_s16_u16(vmovl_high_u8(high_v_value)), uv_corr);
            let y_h_h = vreinterpretq_s16_u16(vmull_high_u8(y_second, v_luma_coeff));

            let r_h_h = vqshrun_n_s16::<6>(vmaxq_s16(
                vqaddq_s16(y_h_h, vmulq_s16(v_h_h, v_cr_coeff)),
                v_min_values,
            ));
            let b_h_h = vqshrun_n_s16::<6>(vmaxq_s16(
                vqaddq_s16(y_h_h, vmulq_s16(u_h_h, v_cb_coeff)),
                v_min_values,
            ));
            let g_h_h = vqshrun_n_s16::<6>(vmaxq_s16(
                vqaddq_s16(
                    y_h_h,
                    vqaddq_s16(vmulq_s16(v_h_h, v_g_coeff_1), vmulq_s16(u_h_h, v_g_coeff_2)),
                ),
                v_min_values,
            ));

            let u_h_l = vsubq_s16(
                vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(high_u_value))),
                uv_corr,
            );
            let v_h_l = vsubq_s16(
                vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(high_v_value))),
                uv_corr,
            );
            let y_h_l =
                vreinterpretq_s16_u16(vmull_u8(vget_low_u8(y_second), vget_low_u8(v_luma_coeff)));

            let r_h_l = vqshrun_n_s16::<6>(vmaxq_s16(
                vqaddq_s16(y_h_l, vmulq_s16(v_h_l, v_cr_coeff)),
                v_min_values,
            ));
            let b_h_l = vqshrun_n_s16::<6>(vmaxq_s16(
                vqaddq_s16(y_h_l, vmulq_s16(u_h_l, v_cb_coeff)),
                v_min_values,
            ));
            let g_h_l = vqshrun_n_s16::<6>(vmaxq_s16(
                vqaddq_s16(
                    y_h_l,
                    vqaddq_s16(vmulq_s16(v_h_l, v_g_coeff_1), vmulq_s16(u_h_l, v_g_coeff_2)),
                ),
                v_min_values,
            ));

            let r_h = vcombine_u8(r_h_l, r_h_h);
            let g_h = vcombine_u8(g_h_l, g_h_h);
            let b_h = vcombine_u8(b_h_l, b_h_h);

            match dst_chans {
                YuvSourceChannels::Rgb => {
                    let packed = uint8x16x3_t(r_h, g_h, b_h);
                    vst3q_u8(dst_ptr.add(16 * dst_chans.get_channels_count()), packed);
                }
                YuvSourceChannels::Rgba => {
                    let packed = uint8x16x4_t(r_h, g_h, b_h, v_alpha);
                    vst4q_u8(dst_ptr.add(16 * dst_chans.get_channels_count()), packed);
                }
                YuvSourceChannels::Bgra => {
                    let packed = uint8x16x4_t(b_h, g_h, r_h, v_alpha);
                    vst4q_u8(dst_ptr.add(16 * dst_chans.get_channels_count()), packed);
                }
                YuvSourceChannels::Bgr => {
                    let packed = uint8x16x3_t(b_h, g_h, r_h);
                    vst3q_u8(dst_ptr.add(16 * dst_chans.get_channels_count()), packed);
                }
            }

            _yuy2_x = x;
            if x + 16 < max_x_16 {
                _cx += 32;
            }
        }

        for x in (_yuy2_x..max_x_8).step_by(8) {
            let dst_offset = yuy2_offset + x * 4;
            let dst_pos = rgb_offset + _cx * dst_chans.get_channels_count();
            let dst_ptr = rgb.as_mut_ptr().add(dst_pos);

            let pixel_set = vld4_u8(yuy2_store.as_ptr().add(dst_offset));
            let mut y_first = match yuy2_source {
                Yuy2Description::YUYV | Yuy2Description::YVYU => pixel_set.0,
                Yuy2Description::UYVY | Yuy2Description::VYUY => pixel_set.1,
            };
            let mut y_second = match yuy2_source {
                Yuy2Description::YUYV | Yuy2Description::YVYU => pixel_set.2,
                Yuy2Description::UYVY | Yuy2Description::VYUY => pixel_set.3,
            };

            let y_first_reconstructed = vzip1_u8(y_first, y_second);
            let y_second_reconstructed = vzip2_u8(y_first, y_second);
            y_first = y_first_reconstructed;
            y_second = y_second_reconstructed;

            let u_value = match yuy2_source {
                Yuy2Description::YUYV => pixel_set.1,
                Yuy2Description::UYVY => pixel_set.0,
                Yuy2Description::YVYU => pixel_set.3,
                Yuy2Description::VYUY => pixel_set.2,
            };
            let v_value = match yuy2_source {
                Yuy2Description::YUYV => pixel_set.3,
                Yuy2Description::UYVY => pixel_set.2,
                Yuy2Description::YVYU => pixel_set.1,
                Yuy2Description::VYUY => pixel_set.0,
            };

            y_first = vsub_u8(y_first, vget_low_u8(y_corr));
            y_second = vsub_u8(y_second, vget_low_u8(y_corr));

            let low_u_value = vzip1_u8(u_value, u_value);
            let high_u_value = vzip2_u8(u_value, u_value);
            let low_v_value = vzip1_u8(v_value, v_value);
            let high_v_value = vzip2_u8(v_value, v_value);

            let u_h = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(high_u_value)), uv_corr);
            let v_h = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(high_v_value)), uv_corr);
            let y_h = vreinterpretq_s16_u16(vmull_u8(y_second, vget_low_u8(v_luma_coeff)));

            let r_h = vqshrun_n_s16::<6>(vmaxq_s16(
                vqaddq_s16(y_h, vmulq_s16(v_h, v_cr_coeff)),
                v_min_values,
            ));
            let b_h = vqshrun_n_s16::<6>(vmaxq_s16(
                vqaddq_s16(y_h, vmulq_s16(u_h, v_cb_coeff)),
                v_min_values,
            ));
            let g_h = vqshrun_n_s16::<6>(vmaxq_s16(
                vqaddq_s16(
                    y_h,
                    vqaddq_s16(vmulq_s16(v_h, v_g_coeff_1), vmulq_s16(u_h, v_g_coeff_2)),
                ),
                v_min_values,
            ));

            let u_low = vsubq_s16(
                vreinterpretq_s16_u16(vmovl_u8(low_u_value)),
                uv_corr,
            );
            let v_low = vsubq_s16(
                vreinterpretq_s16_u16(vmovl_u8(low_v_value)),
                uv_corr,
            );
            let y_low =
                vreinterpretq_s16_u16(vmull_u8(y_first, vget_low_u8(v_luma_coeff)));

            let r_l = vqshrun_n_s16::<6>(vmaxq_s16(
                vqaddq_s16(y_low, vmulq_s16(v_low, v_cr_coeff)),
                v_min_values,
            ));
            let b_l = vqshrun_n_s16::<6>(vmaxq_s16(
                vqaddq_s16(y_low, vmulq_s16(u_low, v_cb_coeff)),
                v_min_values,
            ));
            let g_l = vqshrun_n_s16::<6>(vmaxq_s16(
                vqaddq_s16(
                    y_low,
                    vqaddq_s16(vmulq_s16(v_low, v_g_coeff_1), vmulq_s16(u_low, v_g_coeff_2)),
                ),
                v_min_values,
            ));

            let r_l = vcombine_u8(r_l, r_h);
            let g_l = vcombine_u8(g_l, g_h);
            let b_l = vcombine_u8(b_l, b_h);

            match dst_chans {
                YuvSourceChannels::Rgb => {
                    let packed = uint8x16x3_t(r_l, g_l, b_l);
                    vst3q_u8(dst_ptr, packed);
                }
                YuvSourceChannels::Rgba => {
                    let packed = uint8x16x4_t(r_l, g_l, b_l, v_alpha);
                    vst4q_u8(dst_ptr, packed);
                }
                YuvSourceChannels::Bgra => {
                    let packed = uint8x16x4_t(b_l, g_l, r_l, v_alpha);
                    vst4q_u8(dst_ptr, packed);
                }
                YuvSourceChannels::Bgr => {
                    let packed = uint8x16x3_t(b_l, g_l, r_l);
                    vst3q_u8(dst_ptr, packed);
                }
            }

            _yuy2_x = x;
            if x + 8 < max_x_8 {
                _cx += 16;
            }
        }
    }

    YuvToYuy2Navigation {
        cx: _cx,
        uv_x: 0,
        x: _yuy2_x,
    }
}
