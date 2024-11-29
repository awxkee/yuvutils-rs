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

#[target_feature(enable = "rdm")]
pub(crate) unsafe fn neon_rgba_to_yuv_rdm420<const ORIGIN_CHANNELS: u8, const PRECISION: i32>(
    transform: &CbCrForwardTransform<i32>,
    range: &YuvChromaRange,
    y_plane0: &mut [u8],
    y_plane1: &mut [u8],
    u_plane: &mut [u8],
    v_plane: &mut [u8],
    rgba0: &[u8],
    rgba1: &[u8],
    start_cx: usize,
    start_ux: usize,
    width: usize,
) -> ProcessedOffset {
    let source_channels: YuvSourceChannels = ORIGIN_CHANNELS.into();
    let channels = source_channels.get_channels_count();

    const V_SCALE: i32 = 2;
    let bias_y = range.bias_y as i16;
    let bias_uv = range.bias_uv as i16;

    let u_ptr = u_plane;
    let v_ptr = v_plane;

    let i_bias_y = vdupq_n_s16(range.bias_y as i16);
    let i_cap_y = vdupq_n_u16(range.range_y as u16 + range.bias_y as u16);
    let i_cap_uv = vdupq_n_u16(range.bias_y as u16 + range.range_uv as u16);

    let y_bias = vdupq_n_s16(bias_y);
    let uv_bias = vdupq_n_s16(bias_uv);

    let weights_arr: [i16; 8] = [
        transform.yr as i16,
        transform.yg as i16,
        transform.yb as i16,
        transform.cb_r as i16,
        transform.cb_g as i16,
        transform.cb_b as i16,
        transform.cr_r as i16,
        transform.cr_g as i16,
    ];
    let v_weights = vld1q_s16(weights_arr.as_ptr());
    let v_cr_b = vdupq_n_s16(transform.cr_b as i16);

    let mut cx = start_cx;
    let mut ux = start_ux;

    while cx + 16 < width {
        let r_values0: uint8x16_t;
        let g_values0: uint8x16_t;
        let b_values0: uint8x16_t;

        let r_values1: uint8x16_t;
        let g_values1: uint8x16_t;
        let b_values1: uint8x16_t;

        match source_channels {
            YuvSourceChannels::Rgb | YuvSourceChannels::Bgr => {
                let rgb_values0 = vld3q_u8(rgba0.get_unchecked(cx * channels..).as_ptr());
                if source_channels == YuvSourceChannels::Rgb {
                    r_values0 = rgb_values0.0;
                    g_values0 = rgb_values0.1;
                    b_values0 = rgb_values0.2;
                } else {
                    r_values0 = rgb_values0.2;
                    g_values0 = rgb_values0.1;
                    b_values0 = rgb_values0.0;
                }

                let rgb_values1 = vld3q_u8(rgba1.get_unchecked(cx * channels..).as_ptr());
                if source_channels == YuvSourceChannels::Rgb {
                    r_values1 = rgb_values1.0;
                    g_values1 = rgb_values1.1;
                    b_values1 = rgb_values1.2;
                } else {
                    r_values1 = rgb_values1.2;
                    g_values1 = rgb_values1.1;
                    b_values1 = rgb_values1.0;
                }
            }
            YuvSourceChannels::Rgba => {
                let rgb_values0 = vld4q_u8(rgba0.get_unchecked(cx * channels..).as_ptr());
                r_values0 = rgb_values0.0;
                g_values0 = rgb_values0.1;
                b_values0 = rgb_values0.2;

                let rgb_values1 = vld4q_u8(rgba1.get_unchecked(cx * channels..).as_ptr());
                r_values1 = rgb_values1.0;
                g_values1 = rgb_values1.1;
                b_values1 = rgb_values1.2;
            }
            YuvSourceChannels::Bgra => {
                let rgb_values0 = vld4q_u8(rgba0.get_unchecked(cx * channels..).as_ptr());
                r_values0 = rgb_values0.2;
                g_values0 = rgb_values0.1;
                b_values0 = rgb_values0.0;

                let rgb_values1 = vld4q_u8(rgba1.get_unchecked(cx * channels..).as_ptr());
                r_values1 = rgb_values1.2;
                g_values1 = rgb_values1.1;
                b_values1 = rgb_values1.0;
            }
        }

        let r0hi = vreinterpretq_s16_u16(vshll_high_n_u8::<V_SCALE>(r_values0));
        let g0hi = vreinterpretq_s16_u16(vshll_high_n_u8::<V_SCALE>(g_values0));
        let b0hi = vreinterpretq_s16_u16(vshll_high_n_u8::<V_SCALE>(b_values0));

        let r1hi = vreinterpretq_s16_u16(vshll_high_n_u8::<V_SCALE>(r_values1));
        let g1hi = vreinterpretq_s16_u16(vshll_high_n_u8::<V_SCALE>(g_values1));
        let b1hi = vreinterpretq_s16_u16(vshll_high_n_u8::<V_SCALE>(b_values1));

        let mut y0_high = vqrdmlahq_laneq_s16::<0>(y_bias, r0hi, v_weights);
        y0_high = vqrdmlahq_laneq_s16::<1>(y0_high, g0hi, v_weights);
        y0_high = vqrdmlahq_laneq_s16::<2>(y0_high, b0hi, v_weights);

        let y0_high = vminq_u16(vreinterpretq_u16_s16(vmaxq_s16(y0_high, i_bias_y)), i_cap_y);

        let mut y1_high = vqrdmlahq_laneq_s16::<0>(y_bias, r1hi, v_weights);
        y1_high = vqrdmlahq_laneq_s16::<1>(y1_high, g1hi, v_weights);
        y1_high = vqrdmlahq_laneq_s16::<2>(y1_high, b1hi, v_weights);

        let y1_high = vminq_u16(vreinterpretq_u16_s16(vmaxq_s16(y1_high, i_bias_y)), i_cap_y);

        let r0_low = vreinterpretq_s16_u16(vshll_n_u8::<V_SCALE>(vget_low_u8(r_values0)));
        let g0_low = vreinterpretq_s16_u16(vshll_n_u8::<V_SCALE>(vget_low_u8(g_values0)));
        let b0_low = vreinterpretq_s16_u16(vshll_n_u8::<V_SCALE>(vget_low_u8(b_values0)));

        let r1_low = vreinterpretq_s16_u16(vshll_n_u8::<V_SCALE>(vget_low_u8(r_values1)));
        let g1_low = vreinterpretq_s16_u16(vshll_n_u8::<V_SCALE>(vget_low_u8(g_values1)));
        let b1_low = vreinterpretq_s16_u16(vshll_n_u8::<V_SCALE>(vget_low_u8(b_values1)));

        let mut y0_low = vqrdmlahq_laneq_s16::<0>(y_bias, r0_low, v_weights);
        y0_low = vqrdmlahq_laneq_s16::<1>(y0_low, g0_low, v_weights);
        y0_low = vqrdmlahq_laneq_s16::<2>(y0_low, b0_low, v_weights);

        let y0_low = vminq_u16(vreinterpretq_u16_s16(vmaxq_s16(y0_low, i_bias_y)), i_cap_y);

        let mut y1_low = vqrdmlahq_laneq_s16::<0>(y_bias, r1_low, v_weights);
        y1_low = vqrdmlahq_laneq_s16::<1>(y1_low, g1_low, v_weights);
        y1_low = vqrdmlahq_laneq_s16::<2>(y1_low, b1_low, v_weights);

        let y1_low = vminq_u16(vreinterpretq_u16_s16(vmaxq_s16(y1_low, i_bias_y)), i_cap_y);

        let y0 = vcombine_u8(vmovn_u16(y0_low), vmovn_u16(y0_high));
        vst1q_u8(y_plane0.get_unchecked_mut(cx..).as_mut_ptr(), y0);

        let y1 = vcombine_u8(vmovn_u16(y1_low), vmovn_u16(y1_high));
        vst1q_u8(y_plane1.get_unchecked_mut(cx..).as_mut_ptr(), y1);

        let box_r_values = vpaddlq_u8(r_values0);
        let r1 = vreinterpretq_s16_u16(vshlq_n_u16::<V_SCALE>(vrshrq_n_u16::<1>(box_r_values)));
        let box_g_values = vpaddlq_u8(g_values0);
        let g1 = vreinterpretq_s16_u16(vshlq_n_u16::<V_SCALE>(vrshrq_n_u16::<1>(box_g_values)));
        let box_b_values = vpaddlq_u8(b_values0);
        let b1 = vreinterpretq_s16_u16(vshlq_n_u16::<V_SCALE>(vrshrq_n_u16::<1>(box_b_values)));

        let mut cbl = vqrdmlahq_laneq_s16::<3>(uv_bias, r1, v_weights);
        cbl = vqrdmlahq_laneq_s16::<4>(cbl, g1, v_weights);
        cbl = vqrdmlahq_laneq_s16::<5>(cbl, b1, v_weights);

        let cb = vmovn_u16(vminq_u16(
            vreinterpretq_u16_s16(vmaxq_s16(cbl, i_bias_y)),
            i_cap_uv,
        ));

        let mut crl = vqrdmlahq_laneq_s16::<6>(uv_bias, r1, v_weights);
        crl = vqrdmlahq_laneq_s16::<7>(crl, g1, v_weights);
        crl = vqrdmlahq_laneq_s16::<0>(crl, b1, v_cr_b);

        let cr = vmovn_u16(vminq_u16(
            vreinterpretq_u16_s16(vmaxq_s16(crl, i_bias_y)),
            i_cap_uv,
        ));

        vst1_u8(u_ptr.get_unchecked_mut(ux..).as_mut_ptr(), cb);
        vst1_u8(v_ptr.get_unchecked_mut(ux..).as_mut_ptr(), cr);

        ux += 8;

        cx += 16;
    }

    ProcessedOffset { cx, ux }
}

pub(crate) unsafe fn neon_rgba_to_yuv420<const ORIGIN_CHANNELS: u8, const PRECISION: i32>(
    transform: &CbCrForwardTransform<i32>,
    range: &YuvChromaRange,
    y_plane0: &mut [u8],
    y_plane1: &mut [u8],
    u_plane: &mut [u8],
    v_plane: &mut [u8],
    rgba0: &[u8],
    rgba1: &[u8],
    start_cx: usize,
    start_ux: usize,
    width: usize,
) -> ProcessedOffset {
    let source_channels: YuvSourceChannels = ORIGIN_CHANNELS.into();
    let channels = source_channels.get_channels_count();

    let rounding_const_bias: i32 = 1 << (PRECISION - 1);
    let bias_y = range.bias_y as i32 * (1 << PRECISION) + rounding_const_bias;
    let bias_uv = range.bias_uv as i32 * (1 << PRECISION) + rounding_const_bias;

    let u_ptr = u_plane;
    let v_ptr = v_plane;

    let i_bias_y = vdupq_n_s16(range.bias_y as i16);
    let i_cap_y = vdupq_n_u16(range.range_y as u16 + range.bias_y as u16);
    let i_cap_uv = vdupq_n_u16(range.bias_y as u16 + range.range_uv as u16);

    let y_bias = vdupq_n_s32(bias_y);
    let uv_bias = vdupq_n_s32(bias_uv);

    let weights_arr: [i16; 8] = [
        transform.yr as i16,
        transform.yg as i16,
        transform.yb as i16,
        transform.cb_r as i16,
        transform.cb_g as i16,
        transform.cb_b as i16,
        transform.cr_r as i16,
        transform.cr_g as i16,
    ];
    let v_weights = vld1q_s16(weights_arr.as_ptr());
    let v_cr_b = vdupq_n_s16(transform.cr_b as i16);

    let mut cx = start_cx;
    let mut ux = start_ux;

    while cx + 16 < width {
        let r_values0: uint8x16_t;
        let g_values0: uint8x16_t;
        let b_values0: uint8x16_t;

        let r_values1: uint8x16_t;
        let g_values1: uint8x16_t;
        let b_values1: uint8x16_t;

        match source_channels {
            YuvSourceChannels::Rgb | YuvSourceChannels::Bgr => {
                let rgb_values0 = vld3q_u8(rgba0.get_unchecked(cx * channels..).as_ptr());
                if source_channels == YuvSourceChannels::Rgb {
                    r_values0 = rgb_values0.0;
                    g_values0 = rgb_values0.1;
                    b_values0 = rgb_values0.2;
                } else {
                    r_values0 = rgb_values0.2;
                    g_values0 = rgb_values0.1;
                    b_values0 = rgb_values0.0;
                }

                let rgb_values1 = vld3q_u8(rgba1.get_unchecked(cx * channels..).as_ptr());
                if source_channels == YuvSourceChannels::Rgb {
                    r_values1 = rgb_values1.0;
                    g_values1 = rgb_values1.1;
                    b_values1 = rgb_values1.2;
                } else {
                    r_values1 = rgb_values1.2;
                    g_values1 = rgb_values1.1;
                    b_values1 = rgb_values1.0;
                }
            }
            YuvSourceChannels::Rgba => {
                let rgb_values0 = vld4q_u8(rgba0.get_unchecked(cx * channels..).as_ptr());
                r_values0 = rgb_values0.0;
                g_values0 = rgb_values0.1;
                b_values0 = rgb_values0.2;

                let rgb_values1 = vld4q_u8(rgba1.get_unchecked(cx * channels..).as_ptr());
                r_values1 = rgb_values1.0;
                g_values1 = rgb_values1.1;
                b_values1 = rgb_values1.2;
            }
            YuvSourceChannels::Bgra => {
                let rgb_values0 = vld4q_u8(rgba0.get_unchecked(cx * channels..).as_ptr());
                r_values0 = rgb_values0.2;
                g_values0 = rgb_values0.1;
                b_values0 = rgb_values0.0;

                let rgb_values1 = vld4q_u8(rgba1.get_unchecked(cx * channels..).as_ptr());
                r_values1 = rgb_values1.2;
                g_values1 = rgb_values1.1;
                b_values1 = rgb_values1.0;
            }
        }

        let r_high0 = vreinterpretq_s16_u16(vmovl_high_u8(r_values0));
        let g_high0 = vreinterpretq_s16_u16(vmovl_high_u8(g_values0));
        let b_high0 = vreinterpretq_s16_u16(vmovl_high_u8(b_values0));

        let r_h_low0 = vget_low_s16(r_high0);
        let g_h_low0 = vget_low_s16(g_high0);
        let b_h_low0 = vget_low_s16(b_high0);

        let mut y0_h_high = vmlal_high_laneq_s16::<0>(y_bias, r_high0, v_weights);
        y0_h_high = vmlal_high_laneq_s16::<1>(y0_h_high, g_high0, v_weights);
        y0_h_high = vmlal_high_laneq_s16::<2>(y0_h_high, b_high0, v_weights);

        let mut y0_h_low = vmlal_laneq_s16::<0>(y_bias, r_h_low0, v_weights);
        y0_h_low = vmlal_laneq_s16::<1>(y0_h_low, g_h_low0, v_weights);
        y0_h_low = vmlal_laneq_s16::<2>(y0_h_low, b_h_low0, v_weights);

        let y0_high = vminq_u16(
            vreinterpretq_u16_s16(vmaxq_s16(
                vcombine_s16(
                    vshrn_n_s32::<PRECISION>(y0_h_low),
                    vshrn_n_s32::<PRECISION>(y0_h_high),
                ),
                i_bias_y,
            )),
            i_cap_y,
        );

        let r_high1 = vreinterpretq_s16_u16(vmovl_high_u8(r_values1));
        let g_high1 = vreinterpretq_s16_u16(vmovl_high_u8(g_values1));
        let b_high1 = vreinterpretq_s16_u16(vmovl_high_u8(b_values1));

        let r_h_low1 = vget_low_s16(r_high1);
        let g_h_low1 = vget_low_s16(g_high1);
        let b_h_low1 = vget_low_s16(b_high1);

        let mut y1_h_high = vmlal_high_laneq_s16::<0>(y_bias, r_high1, v_weights);
        y1_h_high = vmlal_high_laneq_s16::<1>(y1_h_high, g_high1, v_weights);
        y1_h_high = vmlal_high_laneq_s16::<2>(y1_h_high, b_high1, v_weights);

        let mut y1_h_low = vmlal_laneq_s16::<0>(y_bias, r_h_low1, v_weights);
        y1_h_low = vmlal_laneq_s16::<1>(y1_h_low, g_h_low1, v_weights);
        y1_h_low = vmlal_laneq_s16::<2>(y1_h_low, b_h_low1, v_weights);

        let y1_high = vminq_u16(
            vreinterpretq_u16_s16(vmaxq_s16(
                vcombine_s16(
                    vshrn_n_s32::<PRECISION>(y1_h_low),
                    vshrn_n_s32::<PRECISION>(y1_h_high),
                ),
                i_bias_y,
            )),
            i_cap_y,
        );

        let r_low0 = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(r_values0)));
        let g_low0 = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(g_values0)));
        let b_low0 = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(b_values0)));

        let r_l_low0 = vget_low_s16(r_low0);
        let g_l_low0 = vget_low_s16(g_low0);
        let b_l_low0 = vget_low_s16(b_low0);

        let mut y0_l_high = vmlal_high_laneq_s16::<0>(y_bias, r_low0, v_weights);
        y0_l_high = vmlal_high_laneq_s16::<1>(y0_l_high, g_low0, v_weights);
        y0_l_high = vmlal_high_laneq_s16::<2>(y0_l_high, b_low0, v_weights);

        let mut y0_l_low = vmlal_laneq_s16::<0>(y_bias, r_l_low0, v_weights);
        y0_l_low = vmlal_laneq_s16::<1>(y0_l_low, g_l_low0, v_weights);
        y0_l_low = vmlal_laneq_s16::<2>(y0_l_low, b_l_low0, v_weights);

        let y0_low = vminq_u16(
            vreinterpretq_u16_s16(vmaxq_s16(
                vcombine_s16(
                    vshrn_n_s32::<PRECISION>(y0_l_low),
                    vshrn_n_s32::<PRECISION>(y0_l_high),
                ),
                i_bias_y,
            )),
            i_cap_y,
        );

        let r_low1 = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(r_values1)));
        let g_low1 = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(g_values1)));
        let b_low1 = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(b_values1)));

        let r_l_low1 = vget_low_s16(r_low1);
        let g_l_low1 = vget_low_s16(g_low1);
        let b_l_low1 = vget_low_s16(b_low1);

        let mut y1_l_high = vmlal_high_laneq_s16::<0>(y_bias, r_low1, v_weights);
        y1_l_high = vmlal_high_laneq_s16::<1>(y1_l_high, g_low1, v_weights);
        y1_l_high = vmlal_high_laneq_s16::<2>(y1_l_high, b_low1, v_weights);

        let mut y1_l_low = vmlal_laneq_s16::<0>(y_bias, r_l_low1, v_weights);
        y1_l_low = vmlal_laneq_s16::<1>(y1_l_low, g_l_low1, v_weights);
        y1_l_low = vmlal_laneq_s16::<2>(y1_l_low, b_l_low1, v_weights);

        let y1_low = vminq_u16(
            vreinterpretq_u16_s16(vmaxq_s16(
                vcombine_s16(
                    vshrn_n_s32::<PRECISION>(y1_l_low),
                    vshrn_n_s32::<PRECISION>(y1_l_high),
                ),
                i_bias_y,
            )),
            i_cap_y,
        );

        let y0 = vcombine_u8(vmovn_u16(y0_low), vmovn_u16(y0_high));
        vst1q_u8(y_plane0.get_unchecked_mut(cx..).as_mut_ptr(), y0);
        let y1 = vcombine_u8(vmovn_u16(y1_low), vmovn_u16(y1_high));
        vst1q_u8(y_plane1.get_unchecked_mut(cx..).as_mut_ptr(), y1);

        let r1 = vreinterpretq_s16_u16(vrshrq_n_u16::<1>(vpaddlq_u8(r_values0)));
        let g1 = vreinterpretq_s16_u16(vrshrq_n_u16::<1>(vpaddlq_u8(g_values0)));
        let b1 = vreinterpretq_s16_u16(vrshrq_n_u16::<1>(vpaddlq_u8(b_values0)));

        let mut cb_h = vmlal_high_laneq_s16::<3>(uv_bias, r1, v_weights);
        cb_h = vmlal_high_laneq_s16::<4>(cb_h, g1, v_weights);
        cb_h = vmlal_high_laneq_s16::<5>(cb_h, b1, v_weights);

        let mut cb_l = vmlal_laneq_s16::<3>(uv_bias, vget_low_s16(r1), v_weights);
        cb_l = vmlal_laneq_s16::<4>(cb_l, vget_low_s16(g1), v_weights);
        cb_l = vmlal_laneq_s16::<5>(cb_l, vget_low_s16(b1), v_weights);

        let cb = vmovn_u16(vminq_u16(
            vreinterpretq_u16_s16(vmaxq_s16(
                vcombine_s16(
                    vshrn_n_s32::<PRECISION>(cb_l),
                    vshrn_n_s32::<PRECISION>(cb_h),
                ),
                i_bias_y,
            )),
            i_cap_uv,
        ));

        let mut cr_h = vmlal_high_laneq_s16::<6>(uv_bias, r1, v_weights);
        cr_h = vmlal_high_laneq_s16::<7>(cr_h, g1, v_weights);
        cr_h = vmlal_high_laneq_s16::<0>(cr_h, b1, v_cr_b);

        let mut cr_l = vmlal_laneq_s16::<6>(uv_bias, vget_low_s16(r1), v_weights);
        cr_l = vmlal_laneq_s16::<7>(cr_l, vget_low_s16(g1), v_weights);
        cr_l = vmlal_laneq_s16::<0>(cr_l, vget_low_s16(b1), v_cr_b);

        let cr = vmovn_u16(vminq_u16(
            vreinterpretq_u16_s16(vmaxq_s16(
                vcombine_s16(
                    vshrn_n_s32::<PRECISION>(cr_l),
                    vshrn_n_s32::<PRECISION>(cr_h),
                ),
                i_bias_y,
            )),
            i_cap_uv,
        ));

        vst1_u8(u_ptr.get_unchecked_mut(ux..).as_mut_ptr(), cb);
        vst1_u8(v_ptr.get_unchecked_mut(ux..).as_mut_ptr(), cr);

        ux += 8;
        cx += 16;
    }

    ProcessedOffset { cx, ux }
}