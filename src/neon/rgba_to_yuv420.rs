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
use crate::neon::utils::{neon_vld_h_rgb_for_yuv, neon_vld_rgb_for_yuv};
use crate::yuv_support::{CbCrForwardTransform, YuvChromaRange, YuvSourceChannels};
use std::arch::aarch64::*;

#[target_feature(enable = "rdm")]
/// Special path for Planar YUV 4:2:0 for aarch64 with RDM available
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

    const V_SCALE: i32 = 4;
    const V_HALF_SCALE: i32 = V_SCALE - 2;
    const A_E: i32 = 2;

    let u_ptr = u_plane;
    let v_ptr = v_plane;

    let y_bias = vdupq_n_s16(range.bias_y as i16 * (1 << A_E));
    let uv_bias = vdupq_n_s16(range.bias_uv as i16 * (1 << A_E) + (1 << (A_E - 1)) - 1);

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
        let (r_values0, g_values0, b_values0) =
            neon_vld_rgb_for_yuv::<ORIGIN_CHANNELS>(rgba0.get_unchecked(cx * channels..).as_ptr());
        let (r_values1, g_values1, b_values1) =
            neon_vld_rgb_for_yuv::<ORIGIN_CHANNELS>(rgba1.get_unchecked(cx * channels..).as_ptr());

        let r_high0 = vreinterpretq_s16_u16(vshll_high_n_u8::<V_SCALE>(r_values0));
        let g_high0 = vreinterpretq_s16_u16(vshll_high_n_u8::<V_SCALE>(g_values0));
        let b_high0 = vreinterpretq_s16_u16(vshll_high_n_u8::<V_SCALE>(b_values0));

        let r_high1 = vreinterpretq_s16_u16(vshll_high_n_u8::<V_SCALE>(r_values1));
        let g_high1 = vreinterpretq_s16_u16(vshll_high_n_u8::<V_SCALE>(g_values1));
        let b_high1 = vreinterpretq_s16_u16(vshll_high_n_u8::<V_SCALE>(b_values1));

        let mut y_high0 = vqrdmlahq_laneq_s16::<0>(y_bias, r_high0, v_weights);
        y_high0 = vqrdmlahq_laneq_s16::<1>(y_high0, g_high0, v_weights);
        y_high0 = vqrdmlahq_laneq_s16::<2>(y_high0, b_high0, v_weights);

        let y0_high = vqshrn_n_u16::<A_E>(vreinterpretq_u16_s16(y_high0));

        let mut y_high1 = vqrdmlahq_laneq_s16::<0>(y_bias, r_high1, v_weights);
        y_high1 = vqrdmlahq_laneq_s16::<1>(y_high1, g_high1, v_weights);
        y_high1 = vqrdmlahq_laneq_s16::<2>(y_high1, b_high1, v_weights);

        let y1_high = vqshrn_n_u16::<A_E>(vreinterpretq_u16_s16(y_high1));

        let r_low0 = vreinterpretq_s16_u16(vshll_n_u8::<V_SCALE>(vget_low_u8(r_values0)));
        let g_low0 = vreinterpretq_s16_u16(vshll_n_u8::<V_SCALE>(vget_low_u8(g_values0)));
        let b_low0 = vreinterpretq_s16_u16(vshll_n_u8::<V_SCALE>(vget_low_u8(b_values0)));

        let r_low1 = vreinterpretq_s16_u16(vshll_n_u8::<V_SCALE>(vget_low_u8(r_values1)));
        let g_low1 = vreinterpretq_s16_u16(vshll_n_u8::<V_SCALE>(vget_low_u8(g_values1)));
        let b_low1 = vreinterpretq_s16_u16(vshll_n_u8::<V_SCALE>(vget_low_u8(b_values1)));

        let mut y_low0 = vqrdmlahq_laneq_s16::<0>(y_bias, r_low0, v_weights);
        y_low0 = vqrdmlahq_laneq_s16::<1>(y_low0, g_low0, v_weights);
        y_low0 = vqrdmlahq_laneq_s16::<2>(y_low0, b_low0, v_weights);

        let y0_low = vqshrn_n_u16::<A_E>(vreinterpretq_u16_s16(y_low0));

        let mut y_low1 = vqrdmlahq_laneq_s16::<0>(y_bias, r_low1, v_weights);
        y_low1 = vqrdmlahq_laneq_s16::<1>(y_low1, g_low1, v_weights);
        y_low1 = vqrdmlahq_laneq_s16::<2>(y_low1, b_low1, v_weights);

        let y1_low = vqshrn_n_u16::<A_E>(vreinterpretq_u16_s16(y_low1));

        let y0 = vcombine_u8(y0_low, y0_high);
        vst1q_u8(y_plane0.get_unchecked_mut(cx..).as_mut_ptr(), y0);
        let y1 = vcombine_u8(y1_low, y1_high);
        vst1q_u8(y_plane1.get_unchecked_mut(cx..).as_mut_ptr(), y1);

        let box_r_values = vaddq_u16(vpaddlq_u8(r_values0), vpaddlq_u8(r_values1));
        let r1 = vreinterpretq_s16_u16(vshlq_n_u16::<V_HALF_SCALE>(box_r_values));
        let box_g_values = vaddq_u16(vpaddlq_u8(g_values0), vpaddlq_u8(g_values1));
        let g1 = vreinterpretq_s16_u16(vshlq_n_u16::<V_HALF_SCALE>(box_g_values));
        let box_b_values = vaddq_u16(vpaddlq_u8(b_values0), vpaddlq_u8(b_values1));
        let b1 = vreinterpretq_s16_u16(vshlq_n_u16::<V_HALF_SCALE>(box_b_values));

        let mut cbl = vqrdmlahq_laneq_s16::<3>(uv_bias, r1, v_weights);
        cbl = vqrdmlahq_laneq_s16::<4>(cbl, g1, v_weights);
        cbl = vqrdmlahq_laneq_s16::<5>(cbl, b1, v_weights);

        let cb = vqshrn_n_u16::<A_E>(vreinterpretq_u16_s16(cbl));

        let mut crl = vqrdmlahq_laneq_s16::<6>(uv_bias, r1, v_weights);
        crl = vqrdmlahq_laneq_s16::<7>(crl, g1, v_weights);
        crl = vqrdmlahq_laneq_s16::<0>(crl, b1, v_cr_b);

        let cr = vqshrn_n_u16::<A_E>(vreinterpretq_u16_s16(crl));

        vst1_u8(u_ptr.get_unchecked_mut(ux..).as_mut_ptr(), cb);
        vst1_u8(v_ptr.get_unchecked_mut(ux..).as_mut_ptr(), cr);

        ux += 8;
        cx += 16;
    }

    while cx + 8 < width {
        let (r_values0, g_values0, b_values0) = neon_vld_h_rgb_for_yuv::<ORIGIN_CHANNELS>(
            rgba0.get_unchecked(cx * channels..).as_ptr(),
        );
        let (r_values1, g_values1, b_values1) = neon_vld_h_rgb_for_yuv::<ORIGIN_CHANNELS>(
            rgba1.get_unchecked(cx * channels..).as_ptr(),
        );

        let r_low0 = vreinterpretq_s16_u16(vshll_n_u8::<V_SCALE>(r_values0));
        let g_low0 = vreinterpretq_s16_u16(vshll_n_u8::<V_SCALE>(g_values0));
        let b_low0 = vreinterpretq_s16_u16(vshll_n_u8::<V_SCALE>(b_values0));

        let r_low1 = vreinterpretq_s16_u16(vshll_n_u8::<V_SCALE>(r_values1));
        let g_low1 = vreinterpretq_s16_u16(vshll_n_u8::<V_SCALE>(g_values1));
        let b_low1 = vreinterpretq_s16_u16(vshll_n_u8::<V_SCALE>(b_values1));

        let mut y_low0 = vqrdmlahq_laneq_s16::<0>(y_bias, r_low0, v_weights);
        y_low0 = vqrdmlahq_laneq_s16::<1>(y_low0, g_low0, v_weights);
        y_low0 = vqrdmlahq_laneq_s16::<2>(y_low0, b_low0, v_weights);

        let y0_low = vqshrn_n_u16::<A_E>(vreinterpretq_u16_s16(y_low0));

        let mut y_low1 = vqrdmlahq_laneq_s16::<0>(y_bias, r_low1, v_weights);
        y_low1 = vqrdmlahq_laneq_s16::<1>(y_low1, g_low1, v_weights);
        y_low1 = vqrdmlahq_laneq_s16::<2>(y_low1, b_low1, v_weights);

        let y1_low = vqshrn_n_u16::<A_E>(vreinterpretq_u16_s16(y_low1));

        vst1_u8(y_plane0.get_unchecked_mut(cx..).as_mut_ptr(), y0_low);
        vst1_u8(y_plane1.get_unchecked_mut(cx..).as_mut_ptr(), y1_low);

        let box_r_values = vadd_u16(vpaddl_u8(r_values0), vpaddl_u8(r_values1));
        let r1 = vreinterpret_s16_u16(vshl_n_u16::<V_HALF_SCALE>(box_r_values));
        let box_g_values = vadd_u16(vpaddl_u8(g_values0), vpaddl_u8(g_values1));
        let g1 = vreinterpret_s16_u16(vshl_n_u16::<V_HALF_SCALE>(box_g_values));
        let box_b_values = vadd_u16(vpaddl_u8(b_values0), vpaddl_u8(b_values1));
        let b1 = vreinterpret_s16_u16(vshl_n_u16::<V_HALF_SCALE>(box_b_values));

        let mut cbl = vqrdmlah_laneq_s16::<3>(vget_low_s16(uv_bias), r1, v_weights);
        cbl = vqrdmlah_laneq_s16::<4>(cbl, g1, v_weights);
        cbl = vqrdmlah_laneq_s16::<5>(cbl, b1, v_weights);

        let cb = vqshrn_n_u16::<A_E>(vcombine_u16(
            vreinterpret_u16_s16(cbl),
            vreinterpret_u16_s16(cbl),
        ));

        let mut crl = vqrdmlah_laneq_s16::<6>(vget_low_s16(uv_bias), r1, v_weights);
        crl = vqrdmlah_laneq_s16::<7>(crl, g1, v_weights);
        crl = vqrdmlah_laneq_s16::<0>(crl, b1, v_cr_b);

        let cr = vqshrn_n_u16::<A_E>(vcombine_u16(
            vreinterpret_u16_s16(crl),
            vreinterpret_u16_s16(crl),
        ));

        vst1_lane_u32::<0>(
            u_ptr.get_unchecked_mut(ux..).as_mut_ptr() as *mut u32,
            vreinterpret_u32_u8(cb),
        );
        vst1_lane_u32::<0>(
            v_ptr.get_unchecked_mut(ux..).as_mut_ptr() as *mut u32,
            vreinterpret_u32_u8(cr),
        );

        ux += 4;
        cx += 8;
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
        let (r_values0, g_values0, b_values0) =
            neon_vld_rgb_for_yuv::<ORIGIN_CHANNELS>(rgba0.get_unchecked(cx * channels..).as_ptr());
        let (r_values1, g_values1, b_values1) =
            neon_vld_rgb_for_yuv::<ORIGIN_CHANNELS>(rgba1.get_unchecked(cx * channels..).as_ptr());

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

        let y0_high = vreinterpretq_u16_s16(vcombine_s16(
            vshrn_n_s32::<PRECISION>(y0_h_low),
            vshrn_n_s32::<PRECISION>(y0_h_high),
        ));

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

        let y1_high = vreinterpretq_u16_s16(vcombine_s16(
            vshrn_n_s32::<PRECISION>(y1_h_low),
            vshrn_n_s32::<PRECISION>(y1_h_high),
        ));

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

        let y0_low = vreinterpretq_u16_s16(vcombine_s16(
            vshrn_n_s32::<PRECISION>(y0_l_low),
            vshrn_n_s32::<PRECISION>(y0_l_high),
        ));

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

        let y1_low = vreinterpretq_u16_s16(vcombine_s16(
            vshrn_n_s32::<PRECISION>(y1_l_low),
            vshrn_n_s32::<PRECISION>(y1_l_high),
        ));

        let y0 = vcombine_u8(vmovn_u16(y0_low), vmovn_u16(y0_high));
        vst1q_u8(y_plane0.get_unchecked_mut(cx..).as_mut_ptr(), y0);
        let y1 = vcombine_u8(vmovn_u16(y1_low), vmovn_u16(y1_high));
        vst1q_u8(y_plane1.get_unchecked_mut(cx..).as_mut_ptr(), y1);

        let r1 = vreinterpretq_s16_u16(vrshrq_n_u16::<1>(vhaddq_u16(
            vpaddlq_u8(r_values0),
            vpaddlq_u8(r_values1),
        )));
        let g1 = vreinterpretq_s16_u16(vrshrq_n_u16::<1>(vhaddq_u16(
            vpaddlq_u8(g_values0),
            vpaddlq_u8(g_values1),
        )));
        let b1 = vreinterpretq_s16_u16(vrshrq_n_u16::<1>(vhaddq_u16(
            vpaddlq_u8(b_values0),
            vpaddlq_u8(b_values1),
        )));

        let mut cb_h = vmlal_high_laneq_s16::<3>(uv_bias, r1, v_weights);
        cb_h = vmlal_high_laneq_s16::<4>(cb_h, g1, v_weights);
        cb_h = vmlal_high_laneq_s16::<5>(cb_h, b1, v_weights);

        let mut cb_l = vmlal_laneq_s16::<3>(uv_bias, vget_low_s16(r1), v_weights);
        cb_l = vmlal_laneq_s16::<4>(cb_l, vget_low_s16(g1), v_weights);
        cb_l = vmlal_laneq_s16::<5>(cb_l, vget_low_s16(b1), v_weights);

        let cb = vmovn_u16(vreinterpretq_u16_s16(vcombine_s16(
            vshrn_n_s32::<PRECISION>(cb_l),
            vshrn_n_s32::<PRECISION>(cb_h),
        )));

        let mut cr_h = vmlal_high_laneq_s16::<6>(uv_bias, r1, v_weights);
        cr_h = vmlal_high_laneq_s16::<7>(cr_h, g1, v_weights);
        cr_h = vmlal_high_laneq_s16::<0>(cr_h, b1, v_cr_b);

        let mut cr_l = vmlal_laneq_s16::<6>(uv_bias, vget_low_s16(r1), v_weights);
        cr_l = vmlal_laneq_s16::<7>(cr_l, vget_low_s16(g1), v_weights);
        cr_l = vmlal_laneq_s16::<0>(cr_l, vget_low_s16(b1), v_cr_b);

        let cr = vmovn_u16(vreinterpretq_u16_s16(vcombine_s16(
            vshrn_n_s32::<PRECISION>(cr_l),
            vshrn_n_s32::<PRECISION>(cr_h),
        )));

        vst1_u8(u_ptr.get_unchecked_mut(ux..).as_mut_ptr(), cb);
        vst1_u8(v_ptr.get_unchecked_mut(ux..).as_mut_ptr(), cr);

        ux += 8;
        cx += 16;
    }

    ProcessedOffset { cx, ux }
}
