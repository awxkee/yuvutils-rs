/*
 * Copyright (c) Radzivon Bartoshyk, 1/2025. All rights reserved.
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
use crate::internals::{tail_420, ProcessedOffset};
use crate::neon::utils::neon_vld_rgb_for_yuv;
use crate::yuv_support::{CbCrForwardTransform, YuvChromaRange, YuvSourceChannels};
use std::arch::aarch64::*;

#[inline(always)]
unsafe fn encode_16_part_prof420<const ORIGIN_CHANNELS: u8>(
    rgba0: &[u8],
    rgba1: &[u8],
    y_plane0: &mut [u8],
    y_plane1: &mut [u8],
    u_plane: &mut [u8],
    v_plane: &mut [u8],
    y_bias: int32x4_t,
    uv_bias: int32x4_t,
    v_weights_a: int16x8_t,
    v_weights_b: int16x8_t,
    v_cr_b: int16x8_t,
) {
    const PRECISION: i32 = 16;
    let (r_values0, g_values0, b_values0) = neon_vld_rgb_for_yuv::<ORIGIN_CHANNELS>(rgba0.as_ptr());
    let (r_values1, g_values1, b_values1) = neon_vld_rgb_for_yuv::<ORIGIN_CHANNELS>(rgba1.as_ptr());

    // Row 0, high 8 pixels: Y = (yr*R + (yg_a+yg_b)*G + yb*B + bias) >> 16
    let r_high0 = vreinterpretq_s16_u16(vmovl_high_u8(r_values0));
    let g_high0 = vreinterpretq_s16_u16(vmovl_high_u8(g_values0));
    let b_high0 = vreinterpretq_s16_u16(vmovl_high_u8(b_values0));

    let r_h_low0 = vget_low_s16(r_high0);
    let g_h_low0 = vget_low_s16(g_high0);
    let b_h_low0 = vget_low_s16(b_high0);

    let mut y0_h_high = vmlal_high_laneq_s16::<0>(y_bias, r_high0, v_weights_a);
    y0_h_high = vmlal_high_laneq_s16::<1>(y0_h_high, g_high0, v_weights_a);
    y0_h_high = vmlal_high_laneq_s16::<1>(y0_h_high, g_high0, v_weights_b);
    y0_h_high = vmlal_high_laneq_s16::<2>(y0_h_high, b_high0, v_weights_a);

    let mut y0_h_low = vmlal_laneq_s16::<0>(y_bias, r_h_low0, v_weights_a);
    y0_h_low = vmlal_laneq_s16::<1>(y0_h_low, g_h_low0, v_weights_a);
    y0_h_low = vmlal_laneq_s16::<1>(y0_h_low, g_h_low0, v_weights_b);
    y0_h_low = vmlal_laneq_s16::<2>(y0_h_low, b_h_low0, v_weights_a);

    let y0_h_low = vqshrn_n_s32::<PRECISION>(y0_h_low);
    let y0_h_high = vqshrn_n_s32::<PRECISION>(y0_h_high);
    let y0_high = vreinterpretq_u16_s16(vcombine_s16(y0_h_low, y0_h_high));

    // Row 1, high 8 pixels
    let r_high1 = vreinterpretq_s16_u16(vmovl_high_u8(r_values1));
    let g_high1 = vreinterpretq_s16_u16(vmovl_high_u8(g_values1));
    let b_high1 = vreinterpretq_s16_u16(vmovl_high_u8(b_values1));

    let r_h_low1 = vget_low_s16(r_high1);
    let g_h_low1 = vget_low_s16(g_high1);
    let b_h_low1 = vget_low_s16(b_high1);

    let mut y1_h_high = vmlal_high_laneq_s16::<0>(y_bias, r_high1, v_weights_a);
    y1_h_high = vmlal_high_laneq_s16::<1>(y1_h_high, g_high1, v_weights_a);
    y1_h_high = vmlal_high_laneq_s16::<1>(y1_h_high, g_high1, v_weights_b);
    y1_h_high = vmlal_high_laneq_s16::<2>(y1_h_high, b_high1, v_weights_a);

    let mut y1_h_low = vmlal_laneq_s16::<0>(y_bias, r_h_low1, v_weights_a);
    y1_h_low = vmlal_laneq_s16::<1>(y1_h_low, g_h_low1, v_weights_a);
    y1_h_low = vmlal_laneq_s16::<1>(y1_h_low, g_h_low1, v_weights_b);
    y1_h_low = vmlal_laneq_s16::<2>(y1_h_low, b_h_low1, v_weights_a);

    let y1_h_low = vqshrn_n_s32::<PRECISION>(y1_h_low);
    let y1_h_high = vqshrn_n_s32::<PRECISION>(y1_h_high);
    let y1_high = vreinterpretq_u16_s16(vcombine_s16(y1_h_low, y1_h_high));

    // Row 0, low 8 pixels
    let r_low0 = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(r_values0)));
    let g_low0 = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(g_values0)));
    let b_low0 = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(b_values0)));

    let r_l_low0 = vget_low_s16(r_low0);
    let g_l_low0 = vget_low_s16(g_low0);
    let b_l_low0 = vget_low_s16(b_low0);

    let mut y0_l_high = vmlal_high_laneq_s16::<0>(y_bias, r_low0, v_weights_a);
    y0_l_high = vmlal_high_laneq_s16::<1>(y0_l_high, g_low0, v_weights_a);
    y0_l_high = vmlal_high_laneq_s16::<1>(y0_l_high, g_low0, v_weights_b);
    y0_l_high = vmlal_high_laneq_s16::<2>(y0_l_high, b_low0, v_weights_a);

    let mut y0_l_low = vmlal_laneq_s16::<0>(y_bias, r_l_low0, v_weights_a);
    y0_l_low = vmlal_laneq_s16::<1>(y0_l_low, g_l_low0, v_weights_a);
    y0_l_low = vmlal_laneq_s16::<1>(y0_l_low, g_l_low0, v_weights_b);
    y0_l_low = vmlal_laneq_s16::<2>(y0_l_low, b_l_low0, v_weights_a);

    let y0_l_low = vqshrn_n_s32::<PRECISION>(y0_l_low);
    let y0_l_high = vqshrn_n_s32::<PRECISION>(y0_l_high);
    let y0_low = vreinterpretq_u16_s16(vcombine_s16(y0_l_low, y0_l_high));

    // Row 1, low 8 pixels
    let r_low1 = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(r_values1)));
    let g_low1 = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(g_values1)));
    let b_low1 = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(b_values1)));

    let r_l_low1 = vget_low_s16(r_low1);
    let g_l_low1 = vget_low_s16(g_low1);
    let b_l_low1 = vget_low_s16(b_low1);

    let mut y1_l_high = vmlal_high_laneq_s16::<0>(y_bias, r_low1, v_weights_a);
    y1_l_high = vmlal_high_laneq_s16::<1>(y1_l_high, g_low1, v_weights_a);
    y1_l_high = vmlal_high_laneq_s16::<1>(y1_l_high, g_low1, v_weights_b);
    y1_l_high = vmlal_high_laneq_s16::<2>(y1_l_high, b_low1, v_weights_a);

    let mut y1_l_low = vmlal_laneq_s16::<0>(y_bias, r_l_low1, v_weights_a);
    y1_l_low = vmlal_laneq_s16::<1>(y1_l_low, g_l_low1, v_weights_a);
    y1_l_low = vmlal_laneq_s16::<1>(y1_l_low, g_l_low1, v_weights_b);
    y1_l_low = vmlal_laneq_s16::<2>(y1_l_low, b_l_low1, v_weights_a);

    let y1_l_low = vqshrn_n_s32::<PRECISION>(y1_l_low);
    let y1_l_high = vqshrn_n_s32::<PRECISION>(y1_l_high);
    let y1_low = vreinterpretq_u16_s16(vcombine_s16(y1_l_low, y1_l_high));

    let y0 = vcombine_u8(vmovn_u16(y0_low), vmovn_u16(y0_high));
    let y1 = vcombine_u8(vmovn_u16(y1_low), vmovn_u16(y1_high));
    vst1q_u8(y_plane0.as_mut_ptr(), y0);
    vst1q_u8(y_plane1.as_mut_ptr(), y1);

    // Chroma: 4:2:0 subsampling - rounding halving average of 2x2 pixel blocks
    let rhv = vrhaddq_u8(r_values0, r_values1);
    let ghv = vrhaddq_u8(g_values0, g_values1);
    let bhv = vrhaddq_u8(b_values0, b_values1);

    let r1hv = vpaddlq_u8(rhv);
    let g1hv = vpaddlq_u8(ghv);
    let b1hv = vpaddlq_u8(bhv);

    let r1 = vreinterpretq_s16_u16(vrshrq_n_u16::<1>(r1hv));
    let g1 = vreinterpretq_s16_u16(vrshrq_n_u16::<1>(g1hv));
    let b1 = vreinterpretq_s16_u16(vrshrq_n_u16::<1>(b1hv));

    // UV: split cb_b (lane 5) and cr_r (lane 6) via v_weights_b for full-range safety
    let mut cb_h = vmlal_high_laneq_s16::<5>(uv_bias, b1, v_weights_a);
    let mut cb_l = vmlal_laneq_s16::<5>(uv_bias, vget_low_s16(b1), v_weights_a);
    cb_h = vmlal_high_laneq_s16::<5>(cb_h, b1, v_weights_b);
    cb_l = vmlal_laneq_s16::<5>(cb_l, vget_low_s16(b1), v_weights_b);
    let mut cr_h = vmlal_high_laneq_s16::<6>(uv_bias, r1, v_weights_a);
    let mut cr_l = vmlal_laneq_s16::<6>(uv_bias, vget_low_s16(r1), v_weights_a);
    cr_h = vmlal_high_laneq_s16::<6>(cr_h, r1, v_weights_b);
    cr_l = vmlal_laneq_s16::<6>(cr_l, vget_low_s16(r1), v_weights_b);

    cb_h = vmlal_high_laneq_s16::<4>(cb_h, g1, v_weights_a);
    cb_l = vmlal_laneq_s16::<4>(cb_l, vget_low_s16(g1), v_weights_a);
    cr_h = vmlal_high_laneq_s16::<7>(cr_h, g1, v_weights_a);
    cr_l = vmlal_laneq_s16::<7>(cr_l, vget_low_s16(g1), v_weights_a);

    cb_h = vmlal_high_laneq_s16::<3>(cb_h, r1, v_weights_a);
    cb_l = vmlal_laneq_s16::<3>(cb_l, vget_low_s16(r1), v_weights_a);
    cr_h = vmlal_high_laneq_s16::<0>(cr_h, b1, v_cr_b);
    cr_l = vmlal_laneq_s16::<0>(cr_l, vget_low_s16(b1), v_cr_b);

    let cb_l = vqshrn_n_s32::<PRECISION>(cb_l);
    let cb_h = vqshrn_n_s32::<PRECISION>(cb_h);
    let cr_l = vqshrn_n_s32::<PRECISION>(cr_l);
    let cr_h = vqshrn_n_s32::<PRECISION>(cr_h);

    let cb = vmovn_u16(vreinterpretq_u16_s16(vcombine_s16(cb_l, cb_h)));
    let cr = vmovn_u16(vreinterpretq_u16_s16(vcombine_s16(cr_l, cr_h)));

    vst1_u8(u_plane.as_mut_ptr(), cb);
    vst1_u8(v_plane.as_mut_ptr(), cr);
}

pub(crate) unsafe fn neon_rgba_to_yuv_prof420<const ORIGIN_CHANNELS: u8>(
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

    const PRECISION: i32 = 16;

    let rounding_const_bias: i32 = (1 << (PRECISION - 1)) - 1;
    let bias_y = range.bias_y as i32 * (1 << PRECISION) + rounding_const_bias;
    let bias_uv = range.bias_uv as i32 * (1 << PRECISION) + rounding_const_bias;

    let u_ptr = u_plane;
    let v_ptr = v_plane;

    let y_bias = vdupq_n_s32(bias_y);
    let uv_bias = vdupq_n_s32(bias_uv);

    // Split yg into two i16-safe halves: yg_a + yg_b = yg
    let yg_a = (transform.yg / 2) as i16;
    let yg_b = (transform.yg - transform.yg / 2) as i16;

    // Split cb_b and cr_r for full-range safety (they can exceed i16::MAX at P16)
    let cb_b_a = (transform.cb_b / 2) as i16;
    let cb_b_b = (transform.cb_b - transform.cb_b / 2) as i16;
    let cr_r_a = (transform.cr_r / 2) as i16;
    let cr_r_b = (transform.cr_r - transform.cr_r / 2) as i16;

    // v_weights_a: [yr, yg_a, yb, cb_r, cb_g, cb_b_a, cr_r_a, cr_g]
    let weights_a_arr: [i16; 8] = [
        transform.yr as i16,
        yg_a,
        transform.yb as i16,
        transform.cb_r as i16,
        transform.cb_g as i16,
        cb_b_a,
        cr_r_a,
        transform.cr_g as i16,
    ];
    let v_weights_a = vld1q_s16(weights_a_arr.as_ptr());

    // v_weights_b: [0, yg_b, 0, 0, 0, cb_b_b, cr_r_b, 0]
    let weights_b_arr: [i16; 8] = [0, yg_b, 0, 0, 0, cb_b_b, cr_r_b, 0];
    let v_weights_b = vld1q_s16(weights_b_arr.as_ptr());

    let v_cr_b = vdupq_n_s16(transform.cr_b as i16);

    let mut cx = start_cx;
    let mut ux = start_ux;

    while cx + 16 <= width {
        encode_16_part_prof420::<ORIGIN_CHANNELS>(
            rgba0.get_unchecked(cx * channels..),
            rgba1.get_unchecked(cx * channels..),
            y_plane0.get_unchecked_mut(cx..),
            y_plane1.get_unchecked_mut(cx..),
            u_ptr.get_unchecked_mut(ux..),
            v_ptr.get_unchecked_mut(ux..),
            y_bias,
            uv_bias,
            v_weights_a,
            v_weights_b,
            v_cr_b,
        );

        ux += 8;
        cx += 16;
    }

    tail_420!(
        16,
        cx,
        ux,
        width,
        channels,
        rgba0,
        rgba1,
        y_plane0,
        y_plane1,
        u_ptr,
        v_ptr,
        |sb0, sb1, yb0, yb1, ub, vb| {
            encode_16_part_prof420::<ORIGIN_CHANNELS>(
                sb0,
                sb1,
                yb0,
                yb1,
                ub,
                vb,
                y_bias,
                uv_bias,
                v_weights_a,
                v_weights_b,
                v_cr_b,
            );
        }
    );

    ProcessedOffset { cx, ux }
}
