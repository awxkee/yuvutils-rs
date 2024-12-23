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

use std::arch::aarch64::*;

use crate::internals::ProcessedOffset;
use crate::neon::neon_simd_support::{neon_vld_rgb_for_yuv, vdotl_laneq_u16_x3};
use crate::yuv_support::{CbCrForwardTransform, YuvChromaRange, YuvNVOrder, YuvSourceChannels};

#[target_feature(enable = "rdm")]
pub(crate) unsafe fn neon_rgbx_to_nv_row_rdm420<
    const ORIGIN_CHANNELS: u8,
    const UV_ORDER: u8,
    const PRECISION: i32,
>(
    y_plane0: &mut [u8],
    y_plane1: &mut [u8],
    uv_plane: &mut [u8],
    rgba0: &[u8],
    rgba1: &[u8],
    width: u32,
    range: &YuvChromaRange,
    transform: &CbCrForwardTransform<i32>,
    start_cx: usize,
    start_ux: usize,
) -> ProcessedOffset {
    let order: YuvNVOrder = UV_ORDER.into();
    let source_channels: YuvSourceChannels = ORIGIN_CHANNELS.into();
    let channels = source_channels.get_channels_count();

    const V_SCALE: i32 = 2;
    let bias_y = range.bias_y as i16;
    let bias_uv = range.bias_uv as i16;

    let uv_ptr = uv_plane.as_mut_ptr();

    let i_bias_y = vdupq_n_s16(range.bias_y as i16);
    let i_cap_uv = vdupq_n_u16(range.bias_y as u16 + range.range_uv as u16);

    let y_base = vdupq_n_u32(bias_y as u32 * (1 << PRECISION) + (1 << (PRECISION - 1)) - 1);
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

    while cx + 16 < width as usize {
        let (r_values0, g_values0, b_values0) =
            neon_vld_rgb_for_yuv::<ORIGIN_CHANNELS>(rgba0.get_unchecked(cx * channels..).as_ptr());
        let (r_values1, g_values1, b_values1) =
            neon_vld_rgb_for_yuv::<ORIGIN_CHANNELS>(rgba1.get_unchecked(cx * channels..).as_ptr());

        let y0_high = vdotl_laneq_u16_x3::<PRECISION, 0, 1, 2>(
            y_base,
            vmovl_high_u8(r_values0),
            vmovl_high_u8(g_values0),
            vmovl_high_u8(b_values0),
            vreinterpretq_u16_s16(v_weights),
        );

        let y0_low = vdotl_laneq_u16_x3::<PRECISION, 0, 1, 2>(
            y_base,
            vmovl_u8(vget_low_u8(r_values0)),
            vmovl_u8(vget_low_u8(g_values0)),
            vmovl_u8(vget_low_u8(b_values0)),
            vreinterpretq_u16_s16(v_weights),
        );

        let y1_low = vdotl_laneq_u16_x3::<PRECISION, 0, 1, 2>(
            y_base,
            vmovl_u8(vget_low_u8(r_values1)),
            vmovl_u8(vget_low_u8(g_values1)),
            vmovl_u8(vget_low_u8(b_values1)),
            vreinterpretq_u16_s16(v_weights),
        );

        let y1_high = vdotl_laneq_u16_x3::<PRECISION, 0, 1, 2>(
            y_base,
            vmovl_high_u8(r_values1),
            vmovl_high_u8(g_values1),
            vmovl_high_u8(b_values1),
            vreinterpretq_u16_s16(v_weights),
        );

        let y0 = vcombine_u8(vqmovn_u16(y0_low), vqmovn_u16(y0_high));
        vst1q_u8(y_plane0.get_unchecked_mut(cx..).as_mut_ptr(), y0);
        let y1 = vcombine_u8(vqmovn_u16(y1_low), vqmovn_u16(y1_high));
        vst1q_u8(y_plane1.get_unchecked_mut(cx..).as_mut_ptr(), y1);

        let r1 = vreinterpretq_s16_u16(vshlq_n_u16::<V_SCALE>(vrshrq_n_u16::<1>(vhaddq_u16(
            vpaddlq_u8(r_values0),
            vpaddlq_u8(r_values1),
        ))));
        let g1 = vreinterpretq_s16_u16(vshlq_n_u16::<V_SCALE>(vrshrq_n_u16::<1>(vhaddq_u16(
            vpaddlq_u8(g_values0),
            vpaddlq_u8(g_values1),
        ))));
        let b1 = vreinterpretq_s16_u16(vshlq_n_u16::<V_SCALE>(vrshrq_n_u16::<1>(vhaddq_u16(
            vpaddlq_u8(b_values0),
            vpaddlq_u8(b_values1),
        ))));

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

        match order {
            YuvNVOrder::UV => {
                let store: uint8x8x2_t = uint8x8x2_t(cb, cr);
                vst2_u8(uv_ptr.add(ux), store);
            }
            YuvNVOrder::VU => {
                let store: uint8x8x2_t = uint8x8x2_t(cr, cb);
                vst2_u8(uv_ptr.add(ux), store);
            }
        }

        ux += 16;
        cx += 16;
    }

    ProcessedOffset { cx, ux }
}

pub(crate) unsafe fn neon_rgbx_to_nv_row420<
    const ORIGIN_CHANNELS: u8,
    const UV_ORDER: u8,
    const PRECISION: i32,
>(
    y_plane0: &mut [u8],
    y_plane1: &mut [u8],
    uv_plane: &mut [u8],
    rgba0: &[u8],
    rgba1: &[u8],
    width: u32,
    range: &YuvChromaRange,
    transform: &CbCrForwardTransform<i32>,
    start_cx: usize,
    start_ux: usize,
) -> ProcessedOffset {
    let order: YuvNVOrder = UV_ORDER.into();
    let source_channels: YuvSourceChannels = ORIGIN_CHANNELS.into();
    let channels = source_channels.get_channels_count();
    let rounding_const_bias: i32 = (1 << (PRECISION - 1)) - 1;
    let bias_y = range.bias_y as i32 * (1 << PRECISION) + rounding_const_bias;
    let bias_uv = range.bias_uv as i32 * (1 << PRECISION) + rounding_const_bias;

    let uv_ptr = uv_plane.as_mut_ptr();

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

    while cx + 16 < width as usize {
        let (r_values_0, g_values_0, b_values_0) =
            neon_vld_rgb_for_yuv::<ORIGIN_CHANNELS>(rgba0.get_unchecked(cx * channels..).as_ptr());
        let (r_values_1, g_values_1, b_values_1) =
            neon_vld_rgb_for_yuv::<ORIGIN_CHANNELS>(rgba1.get_unchecked(cx * channels..).as_ptr());

        let r_high0 = vreinterpretq_s16_u16(vmovl_high_u8(r_values_0));
        let g_high0 = vreinterpretq_s16_u16(vmovl_high_u8(g_values_0));
        let b_high0 = vreinterpretq_s16_u16(vmovl_high_u8(b_values_0));

        let r_high1 = vreinterpretq_s16_u16(vmovl_high_u8(r_values_1));
        let g_high1 = vreinterpretq_s16_u16(vmovl_high_u8(g_values_1));
        let b_high1 = vreinterpretq_s16_u16(vmovl_high_u8(b_values_1));

        let mut y_h_high = vmlal_high_laneq_s16::<0>(y_bias, r_high0, v_weights);
        y_h_high = vmlal_high_laneq_s16::<1>(y_h_high, g_high0, v_weights);
        y_h_high = vmlal_high_laneq_s16::<2>(y_h_high, b_high0, v_weights);

        let mut y_h_low = vmlal_laneq_s16::<0>(y_bias, vget_low_s16(r_high0), v_weights);
        y_h_low = vmlal_laneq_s16::<1>(y_h_low, vget_low_s16(g_high0), v_weights);
        y_h_low = vmlal_laneq_s16::<2>(y_h_low, vget_low_s16(b_high0), v_weights);

        let y_high0 = vminq_u16(
            vreinterpretq_u16_s16(vcombine_s16(
                vshrn_n_s32::<PRECISION>(y_h_low),
                vshrn_n_s32::<PRECISION>(y_h_high),
            )),
            i_cap_y,
        );

        let mut y_h_high = vmlal_high_laneq_s16::<0>(y_bias, r_high1, v_weights);
        y_h_high = vmlal_high_laneq_s16::<1>(y_h_high, g_high1, v_weights);
        y_h_high = vmlal_high_laneq_s16::<2>(y_h_high, b_high1, v_weights);

        let mut y_h_low = vmlal_laneq_s16::<0>(y_bias, vget_low_s16(r_high1), v_weights);
        y_h_low = vmlal_laneq_s16::<1>(y_h_low, vget_low_s16(g_high1), v_weights);
        y_h_low = vmlal_laneq_s16::<2>(y_h_low, vget_low_s16(b_high1), v_weights);

        let y_high1 = vminq_u16(
            vreinterpretq_u16_s16(vcombine_s16(
                vshrn_n_s32::<PRECISION>(y_h_low),
                vshrn_n_s32::<PRECISION>(y_h_high),
            )),
            i_cap_y,
        );

        let r_low0 = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(r_values_0)));
        let g_low0 = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(g_values_0)));
        let b_low0 = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(b_values_0)));

        let r_low1 = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(r_values_1)));
        let g_low1 = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(g_values_1)));
        let b_low1 = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(b_values_1)));

        let mut y_l_high = vmlal_high_laneq_s16::<0>(y_bias, r_low0, v_weights);
        y_l_high = vmlal_high_laneq_s16::<1>(y_l_high, g_low0, v_weights);
        y_l_high = vmlal_high_laneq_s16::<2>(y_l_high, b_low0, v_weights);

        let mut y_l_low = vmlal_laneq_s16::<0>(y_bias, vget_low_s16(r_low0), v_weights);
        y_l_low = vmlal_laneq_s16::<1>(y_l_low, vget_low_s16(g_low0), v_weights);
        y_l_low = vmlal_laneq_s16::<2>(y_l_low, vget_low_s16(b_low0), v_weights);

        let y_low0 = vminq_u16(
            vreinterpretq_u16_s16(vcombine_s16(
                vshrn_n_s32::<PRECISION>(y_l_low),
                vshrn_n_s32::<PRECISION>(y_l_high),
            )),
            i_cap_y,
        );

        let mut y_l_high = vmlal_high_laneq_s16::<0>(y_bias, r_low1, v_weights);
        y_l_high = vmlal_high_laneq_s16::<1>(y_l_high, g_low1, v_weights);
        y_l_high = vmlal_high_laneq_s16::<2>(y_l_high, b_low1, v_weights);

        let mut y_l_low = vmlal_laneq_s16::<0>(y_bias, vget_low_s16(r_low1), v_weights);
        y_l_low = vmlal_laneq_s16::<1>(y_l_low, vget_low_s16(g_low1), v_weights);
        y_l_low = vmlal_laneq_s16::<2>(y_l_low, vget_low_s16(b_low1), v_weights);

        let y_low1 = vminq_u16(
            vreinterpretq_u16_s16(vcombine_s16(
                vshrn_n_s32::<PRECISION>(y_l_low),
                vshrn_n_s32::<PRECISION>(y_l_high),
            )),
            i_cap_y,
        );

        let y0 = vcombine_u8(vmovn_u16(y_low0), vmovn_u16(y_high0));
        vst1q_u8(y_plane0.get_unchecked_mut(cx..).as_mut_ptr(), y0);

        let y1 = vcombine_u8(vmovn_u16(y_low1), vmovn_u16(y_high1));
        vst1q_u8(y_plane1.get_unchecked_mut(cx..).as_mut_ptr(), y1);

        let r1 = vreinterpretq_s16_u16(vrshrq_n_u16::<1>(vhaddq_u16(
            vpaddlq_u8(r_values_0),
            vpaddlq_u8(r_values_1),
        )));
        let g1 = vreinterpretq_s16_u16(vrshrq_n_u16::<1>(vhaddq_u16(
            vpaddlq_u8(g_values_0),
            vpaddlq_u8(g_values_1),
        )));
        let b1 = vreinterpretq_s16_u16(vrshrq_n_u16::<1>(vhaddq_u16(
            vpaddlq_u8(b_values_0),
            vpaddlq_u8(b_values_1),
        )));

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

        match order {
            YuvNVOrder::UV => {
                let store: uint8x8x2_t = uint8x8x2_t(cb, cr);
                vst2_u8(uv_ptr.add(ux), store);
            }
            YuvNVOrder::VU => {
                let store: uint8x8x2_t = uint8x8x2_t(cr, cb);
                vst2_u8(uv_ptr.add(ux), store);
            }
        }
        ux += 16;
        cx += 16;
    }

    ProcessedOffset { cx, ux }
}
