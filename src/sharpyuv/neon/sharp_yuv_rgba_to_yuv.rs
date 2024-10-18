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

use crate::internals::ProcessedOffset;
use crate::yuv_support::{CbCrForwardTransform, YuvChromaRange, YuvSourceChannels};
use std::arch::aarch64::*;

#[inline]
unsafe fn vsquash_row(v0: uint16x8_t, v1: uint16x8_t, gamma_map: &[u8]) -> int16x4_t {
    let c0 = vuzp1_u16(vget_low_u16(v0), vget_high_u16(v0));
    let c1 = vuzp2_u16(vget_low_u16(v0), vget_high_u16(v0));
    let mut acc = vmull_n_u16(c0, 9);

    let c2 = vuzp1_u16(vget_low_u16(v1), vget_high_u16(v1));
    let c3 = vuzp2_u16(vget_low_u16(v1), vget_high_u16(v1));
    acc = vmlal_n_u16(acc, c1, 3);
    acc = vmlal_n_u16(acc, c2, 3);

    acc = vaddw_u16(acc, c3);

    let mut accumulated_row = vrshrn_n_u32::<4>(acc);
    accumulated_row = vset_lane_u16::<0>(
        *gamma_map.get_unchecked(vget_lane_u16::<0>(accumulated_row) as usize) as u16,
        accumulated_row,
    );
    accumulated_row = vset_lane_u16::<1>(
        *gamma_map.get_unchecked(vget_lane_u16::<1>(accumulated_row) as usize) as u16,
        accumulated_row,
    );
    accumulated_row = vset_lane_u16::<2>(
        *gamma_map.get_unchecked(vget_lane_u16::<2>(accumulated_row) as usize) as u16,
        accumulated_row,
    );
    accumulated_row = vset_lane_u16::<3>(
        *gamma_map.get_unchecked(vget_lane_u16::<3>(accumulated_row) as usize) as u16,
        accumulated_row,
    );
    vreinterpret_s16_u16(accumulated_row)
}

#[inline]
unsafe fn vinterpolate_row<const ORIGIN_CHANNELS: u8>(
    linearized_row: *const u16,
    linearized_row_1: *const u16,
    gamma_map: &[u8],
) -> (int16x4_t, int16x4_t, int16x4_t) {
    let pixels_current_0 = vld3q_u16(linearized_row);
    let pixels_current_0_next = vld3q_u16(linearized_row_1);

    let source_channels: YuvSourceChannels = ORIGIN_CHANNELS.into();

    match source_channels {
        YuvSourceChannels::Rgb | YuvSourceChannels::Rgba => {
            let r = vsquash_row(pixels_current_0.0, pixels_current_0_next.0, gamma_map);
            let g = vsquash_row(pixels_current_0.1, pixels_current_0_next.1, gamma_map);
            let b = vsquash_row(pixels_current_0.2, pixels_current_0_next.2, gamma_map);
            (r, g, b)
        }
        YuvSourceChannels::Bgra | YuvSourceChannels::Bgr => {
            let r = vsquash_row(pixels_current_0.2, pixels_current_0_next.2, gamma_map);
            let g = vsquash_row(pixels_current_0.1, pixels_current_0_next.1, gamma_map);
            let b = vsquash_row(pixels_current_0.0, pixels_current_0_next.0, gamma_map);
            (r, g, b)
        }
    }
}

#[inline(always)]
pub unsafe fn neon_rgba_to_sharp_yuv<
    const ORIGIN_CHANNELS: u8,
    const SAMPLING: u8,
    const PRECISION: i32,
>(
    transform: &CbCrForwardTransform<i32>,
    range: &YuvChromaRange,
    y_plane: *mut u8,
    u_plane: *mut u8,
    v_plane: *mut u8,
    rgba: &[u8],
    rgba_offset: usize,
    linearized_row: &[u16],
    linearized_row_1: &[u16],
    gamma_map: &[u8],
    start_cx: usize,
    start_ux: usize,
    width: usize,
    compute_uv_row: bool,
) -> ProcessedOffset {
    let source_channels: YuvSourceChannels = ORIGIN_CHANNELS.into();
    let channels = source_channels.get_channels_count();

    let rounding_const_bias: i32 = 1 << (PRECISION - 1);
    let bias_y = range.bias_y as i32 * (1 << PRECISION) + rounding_const_bias;
    let bias_uv = range.bias_uv as i32 * (1 << PRECISION) + rounding_const_bias;

    let y_ptr = y_plane;
    let u_ptr = u_plane;
    let v_ptr = v_plane;
    let rgba_ptr = rgba.as_ptr();

    let i_bias_y = vdupq_n_s16(range.bias_y as i16);
    let i_cap_y = vdupq_n_u16(range.range_y as u16 + range.bias_y as u16);
    let i_cap_uv = vdupq_n_u16(range.bias_y as u16 + range.range_uv as u16);

    let y_bias = vdupq_n_s32(bias_y);
    let uv_bias = vdupq_n_s32(bias_uv);
    let v_yr = vdupq_n_s16(transform.yr as i16);
    let v_yg = vdupq_n_s16(transform.yg as i16);
    let v_yb = vdupq_n_s16(transform.yb as i16);
    let v_cb_r = vdupq_n_s16(transform.cb_r as i16);
    let v_cb_g = vdupq_n_s16(transform.cb_g as i16);
    let v_cb_b = vdupq_n_s16(transform.cb_b as i16);
    let v_cr_r = vdupq_n_s16(transform.cr_r as i16);
    let v_cr_g = vdupq_n_s16(transform.cr_g as i16);
    let v_cr_b = vdupq_n_s16(transform.cr_b as i16);

    let mut cx = start_cx;
    let mut ux = start_ux;

    let v_zeros = vdupq_n_s32(0i32);

    while cx + 16 < width {
        let r_values_u8: uint8x16_t;
        let g_values_u8: uint8x16_t;
        let b_values_u8: uint8x16_t;

        match source_channels {
            YuvSourceChannels::Rgb | YuvSourceChannels::Bgr => {
                let rgb_values = vld3q_u8(rgba_ptr.add(rgba_offset + cx * channels));
                if source_channels == YuvSourceChannels::Rgb {
                    r_values_u8 = rgb_values.0;
                    g_values_u8 = rgb_values.1;
                    b_values_u8 = rgb_values.2;
                } else {
                    r_values_u8 = rgb_values.2;
                    g_values_u8 = rgb_values.1;
                    b_values_u8 = rgb_values.0;
                }
            }
            YuvSourceChannels::Rgba => {
                let rgb_values = vld4q_u8(rgba_ptr.add(rgba_offset + cx * channels));
                r_values_u8 = rgb_values.0;
                g_values_u8 = rgb_values.1;
                b_values_u8 = rgb_values.2;
            }
            YuvSourceChannels::Bgra => {
                let rgb_values = vld4q_u8(rgba_ptr.add(rgba_offset + cx * channels));
                r_values_u8 = rgb_values.2;
                g_values_u8 = rgb_values.1;
                b_values_u8 = rgb_values.0;
            }
        }

        let r_high = vreinterpretq_s16_u16(vmovl_high_u8(r_values_u8));
        let g_high = vreinterpretq_s16_u16(vmovl_high_u8(g_values_u8));
        let b_high = vreinterpretq_s16_u16(vmovl_high_u8(b_values_u8));

        let r_h_low = vget_low_s16(r_high);
        let g_h_low = vget_low_s16(g_high);
        let b_h_low = vget_low_s16(b_high);

        let mut y_h_high = vmlal_high_s16(y_bias, r_high, v_yr);
        y_h_high = vmlal_high_s16(y_h_high, g_high, v_yg);
        y_h_high = vmlal_high_s16(y_h_high, b_high, v_yb);
        y_h_high = vmaxq_s32(y_h_high, v_zeros);

        let mut y_h_low = vmlal_s16(y_bias, r_h_low, vget_low_s16(v_yr));
        y_h_low = vmlal_s16(y_h_low, g_h_low, vget_low_s16(v_yg));
        y_h_low = vmlal_s16(y_h_low, b_h_low, vget_low_s16(v_yb));
        y_h_low = vmaxq_s32(y_h_low, v_zeros);

        let y_high = vminq_u16(
            vreinterpretq_u16_s16(vmaxq_s16(
                vcombine_s16(
                    vshrn_n_s32::<PRECISION>(y_h_low),
                    vshrn_n_s32::<PRECISION>(y_h_high),
                ),
                i_bias_y,
            )),
            i_cap_y,
        );

        let r_low = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(r_values_u8)));
        let g_low = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(g_values_u8)));
        let b_low = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(b_values_u8)));

        let r_l_low = vget_low_s16(r_low);
        let g_l_low = vget_low_s16(g_low);
        let b_l_low = vget_low_s16(b_low);

        let mut y_l_high = vmlal_high_s16(y_bias, r_low, v_yr);
        y_l_high = vmlal_high_s16(y_l_high, g_low, v_yg);
        y_l_high = vmlal_high_s16(y_l_high, b_low, v_yb);
        y_l_high = vmaxq_s32(y_l_high, v_zeros);

        let mut y_l_low = vmlal_s16(y_bias, r_l_low, vget_low_s16(v_yr));
        y_l_low = vmlal_s16(y_l_low, g_l_low, vget_low_s16(v_yg));
        y_l_low = vmlal_s16(y_l_low, b_l_low, vget_low_s16(v_yb));
        y_l_low = vmaxq_s32(y_l_low, v_zeros);

        let y_low = vminq_u16(
            vreinterpretq_u16_s16(vmaxq_s16(
                vcombine_s16(
                    vshrn_n_s32::<PRECISION>(y_l_low),
                    vshrn_n_s32::<PRECISION>(y_l_high),
                ),
                i_bias_y,
            )),
            i_cap_y,
        );

        let y = vcombine_u8(vqmovn_u16(y_low), vqmovn_u16(y_high));
        vst1q_u8(y_ptr.add(cx), y);

        if compute_uv_row {
            let row0 = linearized_row.as_ptr().add(cx * 3);
            let row1 = linearized_row_1.as_ptr().add(cx * 3);
            let (r_low, g_low, b_low) = vinterpolate_row::<ORIGIN_CHANNELS>(row0, row1, gamma_map);
            let (r_high, g_high, b_high) =
                vinterpolate_row::<ORIGIN_CHANNELS>(row0.add(8 * 3), row1.add(8 * 3), gamma_map);

            let mut cb_h_low = vmlal_s16(uv_bias, r_high, vget_low_s16(v_cb_r));
            cb_h_low = vmlal_s16(cb_h_low, g_high, vget_low_s16(v_cb_g));
            cb_h_low = vmlal_s16(cb_h_low, b_high, vget_low_s16(v_cb_b));

            let cb_high = vmin_u16(
                vreinterpret_u16_s16(vmax_s16(
                    vshrn_n_s32::<PRECISION>(cb_h_low),
                    vget_low_s16(i_bias_y),
                )),
                vget_low_u16(i_cap_uv),
            );

            let mut cr_h_low = vmlal_s16(uv_bias, r_high, vget_low_s16(v_cr_r));
            cr_h_low = vmlal_s16(cr_h_low, g_high, vget_low_s16(v_cr_g));
            cr_h_low = vmlal_s16(cr_h_low, b_high, vget_low_s16(v_cr_b));

            let cr_high = vmin_u16(
                vreinterpret_u16_s16(vmax_s16(
                    vshrn_n_s32::<PRECISION>(cr_h_low),
                    vget_low_s16(i_bias_y),
                )),
                vget_low_u16(i_cap_uv),
            );

            let mut cb_l_low = vmlal_s16(uv_bias, r_low, vget_low_s16(v_cb_r));
            cb_l_low = vmlal_s16(cb_l_low, g_low, vget_low_s16(v_cb_g));
            cb_l_low = vmlal_s16(cb_l_low, b_low, vget_low_s16(v_cb_b));

            let cb_low = vmin_u16(
                vreinterpret_u16_s16(vmax_s16(
                    vshrn_n_s32::<PRECISION>(cb_l_low),
                    vget_low_s16(i_bias_y),
                )),
                vget_low_u16(i_cap_uv),
            );

            let mut cr_l_low = vmlal_s16(uv_bias, r_low, vget_low_s16(v_cr_r));
            cr_l_low = vmlal_s16(cr_l_low, g_low, vget_low_s16(v_cr_g));
            cr_l_low = vmlal_s16(cr_l_low, b_low, vget_low_s16(v_cr_b));

            let cr_low = vmin_u16(
                vreinterpret_u16_s16(vmax_s16(
                    vshrn_n_s32::<PRECISION>(cr_l_low),
                    vget_low_s16(i_bias_y),
                )),
                vget_low_u16(i_cap_uv),
            );
            let cb = vqmovn_u16(vcombine_u16(cb_low, cb_high));
            let cr = vqmovn_u16(vcombine_u16(cr_low, cr_high));

            vst1_u8(u_ptr.add(ux), cb);
            vst1_u8(v_ptr.add(ux), cr);

            ux += 8;
        }

        cx += 16;
    }

    ProcessedOffset { cx, ux }
}
