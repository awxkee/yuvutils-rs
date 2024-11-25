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
use crate::neon::neon_simd_support::{vaddn_dot, vdotl_s16, vmullq_s16, vweight_x2};
use crate::yuv_support::{CbCrInverseTransform, YuvChromaRange, YuvNVOrder, YuvSourceChannels};
use std::arch::aarch64::*;

#[target_feature(enable = "rdm")]
pub(crate) unsafe fn neon_yuv_nv_to_rgba_row_rdm420<
    const UV_ORDER: u8,
    const DESTINATION_CHANNELS: u8,
>(
    range: &YuvChromaRange,
    transform: &CbCrInverseTransform<i32>,
    y_plane0: &[u8],
    y_plane1: &[u8],
    uv_plane: &[u8],
    rgba0: &mut [u8],
    rgba1: &mut [u8],
    start_cx: usize,
    start_ux: usize,
    width: usize,
) -> ProcessedOffset {
    let order: YuvNVOrder = UV_ORDER.into();
    let destination_channels: YuvSourceChannels = DESTINATION_CHANNELS.into();
    let channels = destination_channels.get_channels_count();

    let uv_ptr = uv_plane.as_ptr();

    let y_corr = vdupq_n_u8(range.bias_y as u8);
    let uv_corr = vdupq_n_s16(range.bias_uv as i16);
    let v_alpha = vdupq_n_u8(255u8);

    let mut cx = start_cx;
    let mut ux = start_ux;

    const SCALE: i32 = 7;
    const V_SHR: i32 = 4;

    let weights_arr: [i16; 8] = [
        transform.y_coef as i16,
        transform.cr_coef as i16,
        transform.cb_coef as i16,
        transform.g_coeff_1 as i16,
        transform.g_coeff_2 as i16,
        0,
        0,
        0,
    ];

    let v_weights = vld1q_s16(weights_arr.as_ptr());

    while cx + 16 < width {
        let y_values0 = vqsubq_u8(vld1q_u8(y_plane0.get_unchecked(cx..).as_ptr()), y_corr);
        let y_values1 = vqsubq_u8(vld1q_u8(y_plane1.get_unchecked(cx..).as_ptr()), y_corr);

        let mut uv_values = vld2_u8(uv_ptr.add(ux));
        if order == YuvNVOrder::VU {
            uv_values = uint8x8x2_t(uv_values.1, uv_values.0);
        }

        let u_high_u8 = vzip2_u8(uv_values.0, uv_values.0);
        let v_high_u8 = vzip2_u8(uv_values.1, uv_values.1);
        let u_low_u8 = vzip1_u8(uv_values.0, uv_values.0);
        let v_low_u8 = vzip1_u8(uv_values.1, uv_values.1);

        let u_high = vshlq_n_s16::<SCALE>(vsubq_s16(
            vreinterpretq_s16_u16(vmovl_u8(u_high_u8)),
            uv_corr,
        ));
        let v_high = vshlq_n_s16::<SCALE>(vsubq_s16(
            vreinterpretq_s16_u16(vmovl_u8(v_high_u8)),
            uv_corr,
        ));
        let y_v_shl0 = vshll_high_n_u8::<SCALE>(y_values0);
        let y_v_shl1 = vshll_high_n_u8::<SCALE>(y_values1);
        let y_high0 = vqrdmulhq_laneq_s16::<0>(vreinterpretq_s16_u16(y_v_shl0), v_weights);
        let y_high1 = vqrdmulhq_laneq_s16::<0>(vreinterpretq_s16_u16(y_v_shl1), v_weights);

        let g_coeff_hi = vqrdmlahq_laneq_s16::<4>(
            vqrdmulhq_laneq_s16::<3>(v_high, v_weights),
            u_high,
            v_weights,
        );

        let r_high0 = vqrshrun_n_s16::<V_SHR>(vqrdmlahq_laneq_s16::<1>(y_high0, v_high, v_weights));
        let b_high0 = vqrshrun_n_s16::<V_SHR>(vqrdmlahq_laneq_s16::<2>(y_high0, u_high, v_weights));
        let g_high0 = vqrshrun_n_s16::<V_SHR>(vsubq_s16(y_high0, g_coeff_hi));

        let r_high1 = vqrshrun_n_s16::<V_SHR>(vqrdmlahq_laneq_s16::<1>(y_high1, v_high, v_weights));
        let b_high1 = vqrshrun_n_s16::<V_SHR>(vqrdmlahq_laneq_s16::<2>(y_high1, u_high, v_weights));
        let g_high1 = vqrshrun_n_s16::<V_SHR>(vsubq_s16(y_high1, g_coeff_hi));

        let u_low = vshlq_n_s16::<SCALE>(vsubq_s16(
            vreinterpretq_s16_u16(vmovl_u8(u_low_u8)),
            uv_corr,
        ));
        let v_low = vshlq_n_s16::<SCALE>(vsubq_s16(
            vreinterpretq_s16_u16(vmovl_u8(v_low_u8)),
            uv_corr,
        ));
        let y_v_shl0 = vshll_n_u8::<SCALE>(vget_low_u8(y_values0));
        let y_v_shl1 = vshll_n_u8::<SCALE>(vget_low_u8(y_values1));
        let y_low0 = vqrdmulhq_laneq_s16::<0>(vreinterpretq_s16_u16(y_v_shl0), v_weights);
        let y_low1 = vqrdmulhq_laneq_s16::<0>(vreinterpretq_s16_u16(y_v_shl1), v_weights);

        let g_coeff_lo =
            vqrdmlahq_laneq_s16::<4>(vqrdmulhq_laneq_s16::<3>(v_low, v_weights), u_low, v_weights);

        let r_low0 = vqrshrun_n_s16::<V_SHR>(vqrdmlahq_laneq_s16::<1>(y_low0, v_low, v_weights));
        let b_low0 = vqrshrun_n_s16::<V_SHR>(vqrdmlahq_laneq_s16::<2>(y_low0, u_low, v_weights));
        let g_low0 = vqrshrun_n_s16::<V_SHR>(vsubq_s16(y_low0, g_coeff_lo));

        let r_low1 = vqrshrun_n_s16::<V_SHR>(vqrdmlahq_laneq_s16::<1>(y_low1, v_low, v_weights));
        let b_low1 = vqrshrun_n_s16::<V_SHR>(vqrdmlahq_laneq_s16::<2>(y_low1, u_low, v_weights));
        let g_low1 = vqrshrun_n_s16::<V_SHR>(vsubq_s16(y_low1, g_coeff_lo));

        let r_values0 = vcombine_u8(r_low0, r_high0);
        let g_values0 = vcombine_u8(g_low0, g_high0);
        let b_values0 = vcombine_u8(b_low0, b_high0);

        let r_values1 = vcombine_u8(r_low1, r_high1);
        let g_values1 = vcombine_u8(g_low1, g_high1);
        let b_values1 = vcombine_u8(b_low1, b_high1);

        let dst_shift = cx * channels;

        match destination_channels {
            YuvSourceChannels::Rgb => {
                let dst_pack0: uint8x16x3_t = uint8x16x3_t(r_values0, g_values0, b_values0);
                vst3q_u8(rgba0.get_unchecked_mut(dst_shift..).as_mut_ptr(), dst_pack0);
                let dst_pack1: uint8x16x3_t = uint8x16x3_t(r_values1, g_values1, b_values1);
                vst3q_u8(rgba1.get_unchecked_mut(dst_shift..).as_mut_ptr(), dst_pack1);
            }
            YuvSourceChannels::Bgr => {
                let dst_pack0: uint8x16x3_t = uint8x16x3_t(b_values0, g_values0, r_values0);
                vst3q_u8(rgba0.get_unchecked_mut(dst_shift..).as_mut_ptr(), dst_pack0);
                let dst_pack1: uint8x16x3_t = uint8x16x3_t(b_values1, g_values1, r_values1);
                vst3q_u8(rgba1.get_unchecked_mut(dst_shift..).as_mut_ptr(), dst_pack1);
            }
            YuvSourceChannels::Rgba => {
                let dst_pack0: uint8x16x4_t =
                    uint8x16x4_t(r_values0, g_values0, b_values0, v_alpha);
                vst4q_u8(rgba0.get_unchecked_mut(dst_shift..).as_mut_ptr(), dst_pack0);
                let dst_pack1: uint8x16x4_t =
                    uint8x16x4_t(r_values1, g_values1, b_values1, v_alpha);
                vst4q_u8(rgba1.get_unchecked_mut(dst_shift..).as_mut_ptr(), dst_pack1);
            }
            YuvSourceChannels::Bgra => {
                let dst_pack0: uint8x16x4_t =
                    uint8x16x4_t(b_values0, g_values0, r_values0, v_alpha);
                vst4q_u8(rgba0.get_unchecked_mut(dst_shift..).as_mut_ptr(), dst_pack0);
                let dst_pack1: uint8x16x4_t =
                    uint8x16x4_t(b_values1, g_values1, r_values1, v_alpha);
                vst4q_u8(rgba1.get_unchecked_mut(dst_shift..).as_mut_ptr(), dst_pack1);
            }
        }

        cx += 16;
        ux += 16;
    }

    let shuffle_u = vld1_u8([0, 0, 2, 2, 4, 4, 6, 6].as_ptr());
    let shuffle_v = vld1_u8([1, 1, 3, 3, 5, 5, 7, 7].as_ptr());

    while cx + 8 < width {
        let y_values0 = vqsub_u8(
            vld1_u8(y_plane0.get_unchecked(cx..).as_ptr()),
            vget_low_u8(y_corr),
        );
        let y_values1 = vqsub_u8(
            vld1_u8(y_plane0.get_unchecked(cx..).as_ptr()),
            vget_low_u8(y_corr),
        );

        let mut u_low_u8: uint8x8_t;
        let mut v_low_u8: uint8x8_t;

        let uv_values = vld1_u8(uv_ptr.add(ux));

        u_low_u8 = vtbl1_u8(uv_values, shuffle_u);
        v_low_u8 = vtbl1_u8(uv_values, shuffle_v);

        #[allow(clippy::manual_swap)]
        if order == YuvNVOrder::VU {
            let new_v = u_low_u8;
            u_low_u8 = v_low_u8;
            v_low_u8 = new_v;
        }

        let u_low = vshlq_n_s16::<SCALE>(vsubq_s16(
            vreinterpretq_s16_u16(vmovl_u8(u_low_u8)),
            uv_corr,
        ));
        let v_low = vshlq_n_s16::<SCALE>(vsubq_s16(
            vreinterpretq_s16_u16(vmovl_u8(v_low_u8)),
            uv_corr,
        ));
        let y_v_shl0 = vshll_n_u8::<SCALE>(y_values0);
        let y_v_shl1 = vshll_n_u8::<SCALE>(y_values1);
        let y_low0 = vqrdmulhq_laneq_s16::<0>(vreinterpretq_s16_u16(y_v_shl0), v_weights);
        let y_low1 = vqrdmulhq_laneq_s16::<0>(vreinterpretq_s16_u16(y_v_shl1), v_weights);

        let g_coeff_lo =
            vqrdmlahq_laneq_s16::<4>(vqrdmulhq_laneq_s16::<3>(v_low, v_weights), u_low, v_weights);

        let r_low0 = vqrshrun_n_s16::<V_SHR>(vqrdmlahq_laneq_s16::<1>(y_low0, v_low, v_weights));
        let b_low0 = vqrshrun_n_s16::<V_SHR>(vqrdmlahq_laneq_s16::<2>(y_low0, u_low, v_weights));
        let g_low0 = vqrshrun_n_s16::<V_SHR>(vsubq_s16(y_low0, g_coeff_lo));

        let r_low1 = vqrshrun_n_s16::<V_SHR>(vqrdmlahq_laneq_s16::<1>(y_low1, v_low, v_weights));
        let b_low1 = vqrshrun_n_s16::<V_SHR>(vqrdmlahq_laneq_s16::<2>(y_low1, u_low, v_weights));
        let g_low1 = vqrshrun_n_s16::<V_SHR>(vsubq_s16(y_low1, g_coeff_lo));

        let r_values0 = r_low0;
        let g_values0 = g_low0;
        let b_values0 = b_low0;

        let r_values1 = r_low1;
        let g_values1 = g_low1;
        let b_values1 = b_low1;

        let dst_shift = cx * channels;

        match destination_channels {
            YuvSourceChannels::Rgb => {
                let dst_pack0: uint8x8x3_t = uint8x8x3_t(r_values0, g_values0, b_values0);
                vst3_u8(rgba0.get_unchecked_mut(dst_shift..).as_mut_ptr(), dst_pack0);
                let dst_pack1: uint8x8x3_t = uint8x8x3_t(r_values1, g_values1, b_values1);
                vst3_u8(rgba1.get_unchecked_mut(dst_shift..).as_mut_ptr(), dst_pack1);
            }
            YuvSourceChannels::Bgr => {
                let dst_pack0: uint8x8x3_t = uint8x8x3_t(b_values0, g_values0, r_values0);
                vst3_u8(rgba0.get_unchecked_mut(dst_shift..).as_mut_ptr(), dst_pack0);
                let dst_pack1: uint8x8x3_t = uint8x8x3_t(b_values1, g_values1, r_values1);
                vst3_u8(rgba1.get_unchecked_mut(dst_shift..).as_mut_ptr(), dst_pack1);
            }
            YuvSourceChannels::Rgba => {
                let dst_pack0: uint8x8x4_t =
                    uint8x8x4_t(r_values0, g_values0, b_values0, vget_low_u8(v_alpha));
                vst4_u8(rgba0.get_unchecked_mut(dst_shift..).as_mut_ptr(), dst_pack0);
                let dst_pack1: uint8x8x4_t =
                    uint8x8x4_t(r_values1, g_values1, b_values1, vget_low_u8(v_alpha));
                vst4_u8(rgba1.get_unchecked_mut(dst_shift..).as_mut_ptr(), dst_pack1);
            }
            YuvSourceChannels::Bgra => {
                let dst_pack0: uint8x8x4_t =
                    uint8x8x4_t(b_values0, g_values0, r_values0, vget_low_u8(v_alpha));
                vst4_u8(rgba0.get_unchecked_mut(dst_shift..).as_mut_ptr(), dst_pack0);
                let dst_pack1: uint8x8x4_t =
                    uint8x8x4_t(b_values1, g_values1, r_values1, vget_low_u8(v_alpha));
                vst4_u8(rgba1.get_unchecked_mut(dst_shift..).as_mut_ptr(), dst_pack1);
            }
        }

        cx += 8;
        ux += 8;
    }

    ProcessedOffset { cx, ux }
}

pub(crate) unsafe fn neon_yuv_nv_to_rgba_row420<
    const PRECISION: i32,
    const UV_ORDER: u8,
    const DESTINATION_CHANNELS: u8,
>(
    range: &YuvChromaRange,
    transform: &CbCrInverseTransform<i32>,
    y_plane0: &[u8],
    y_plane1: &[u8],
    uv_plane: &[u8],
    rgba0: &mut [u8],
    rgba1: &mut [u8],
    start_cx: usize,
    start_ux: usize,
    width: usize,
) -> ProcessedOffset {
    let order: YuvNVOrder = UV_ORDER.into();
    let destination_channels: YuvSourceChannels = DESTINATION_CHANNELS.into();
    let channels = destination_channels.get_channels_count();

    let uv_ptr = uv_plane.as_ptr();

    let y_corr = vdupq_n_u8(range.bias_y as u8);
    let uv_corr = vdupq_n_s16(range.bias_uv as i16);
    let v_luma_coeff = vdupq_n_s16(transform.y_coef as i16);
    let v_cr_coeff = vdupq_n_s16(transform.cr_coef as i16);
    let v_cb_coeff = vdupq_n_s16(transform.cb_coef as i16);
    let v_g_coeff_1 = vdupq_n_s16(-(transform.g_coeff_1 as i16));
    let v_g_coeff_2 = vdupq_n_s16(-(transform.g_coeff_2 as i16));
    let v_alpha = vdupq_n_u8(255u8);

    let mut cx = start_cx;
    let mut ux = start_ux;

    while cx + 16 < width {
        let y_values0 = vqsubq_u8(vld1q_u8(y_plane0.get_unchecked(cx..).as_ptr()), y_corr);
        let y_values1 = vqsubq_u8(vld1q_u8(y_plane1.get_unchecked(cx..).as_ptr()), y_corr);

        let mut uv_values = vld2_u8(uv_ptr.add(ux));
        if order == YuvNVOrder::VU {
            uv_values = uint8x8x2_t(uv_values.1, uv_values.0);
        }

        let u_high_u8 = vzip2_u8(uv_values.0, uv_values.0);
        let v_high_u8 = vzip2_u8(uv_values.1, uv_values.1);
        let u_low_u8 = vzip1_u8(uv_values.0, uv_values.0);
        let v_low_u8 = vzip1_u8(uv_values.1, uv_values.1);

        let u_high = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(u_high_u8)), uv_corr);
        let v_high = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(v_high_u8)), uv_corr);
        let y_high0 = vmullq_s16(
            vreinterpretq_s16_u16(vmovl_high_u8(y_values0)),
            v_luma_coeff,
        );
        let y_high1 = vmullq_s16(
            vreinterpretq_s16_u16(vmovl_high_u8(y_values1)),
            v_luma_coeff,
        );

        let g_coeff_hi = vweight_x2(v_high, v_g_coeff_1, u_high, v_g_coeff_2);

        let r_high0 = vdotl_s16::<PRECISION>(y_high0, v_high, v_cr_coeff);
        let b_high0 = vdotl_s16::<PRECISION>(y_high0, u_high, v_cb_coeff);
        let g_high0 = vaddn_dot::<PRECISION>(y_high0, g_coeff_hi);

        let r_high1 = vdotl_s16::<PRECISION>(y_high1, v_high, v_cr_coeff);
        let b_high1 = vdotl_s16::<PRECISION>(y_high1, u_high, v_cb_coeff);
        let g_high1 = vaddn_dot::<PRECISION>(y_high1, g_coeff_hi);

        let u_low = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(u_low_u8)), uv_corr);
        let v_low = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(v_low_u8)), uv_corr);
        let y_low0 = vmullq_s16(
            vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(y_values0))),
            v_luma_coeff,
        );
        let y_low1 = vmullq_s16(
            vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(y_values1))),
            v_luma_coeff,
        );

        let g_coeff_lo = vweight_x2(v_low, v_g_coeff_1, u_low, v_g_coeff_2);

        let r_low0 = vdotl_s16::<PRECISION>(y_low0, v_low, v_cr_coeff);
        let b_low0 = vdotl_s16::<PRECISION>(y_low0, u_low, v_cb_coeff);
        let g_low0 = vaddn_dot::<PRECISION>(y_low0, g_coeff_lo);

        let r_low1 = vdotl_s16::<PRECISION>(y_low1, v_low, v_cr_coeff);
        let b_low1 = vdotl_s16::<PRECISION>(y_low1, u_low, v_cb_coeff);
        let g_low1 = vaddn_dot::<PRECISION>(y_low1, g_coeff_lo);

        let r_values0 = vcombine_u8(vqmovun_s16(r_low0), vqmovun_s16(r_high0));
        let g_values0 = vcombine_u8(vqmovun_s16(g_low0), vqmovun_s16(g_high0));
        let b_values0 = vcombine_u8(vqmovun_s16(b_low0), vqmovun_s16(b_high0));

        let r_values1 = vcombine_u8(vqmovun_s16(r_low1), vqmovun_s16(r_high1));
        let g_values1 = vcombine_u8(vqmovun_s16(g_low1), vqmovun_s16(g_high1));
        let b_values1 = vcombine_u8(vqmovun_s16(b_low1), vqmovun_s16(b_high1));

        let dst_shift = cx * channels;

        match destination_channels {
            YuvSourceChannels::Rgb => {
                let dst_pack0: uint8x16x3_t = uint8x16x3_t(r_values0, g_values0, b_values0);
                vst3q_u8(rgba0.get_unchecked_mut(dst_shift..).as_mut_ptr(), dst_pack0);
                let dst_pack1: uint8x16x3_t = uint8x16x3_t(r_values1, g_values1, b_values1);
                vst3q_u8(rgba1.get_unchecked_mut(dst_shift..).as_mut_ptr(), dst_pack1);
            }
            YuvSourceChannels::Bgr => {
                let dst_pack0: uint8x16x3_t = uint8x16x3_t(b_values0, g_values0, r_values0);
                vst3q_u8(rgba0.get_unchecked_mut(dst_shift..).as_mut_ptr(), dst_pack0);
                let dst_pack1: uint8x16x3_t = uint8x16x3_t(b_values1, g_values1, r_values1);
                vst3q_u8(rgba1.get_unchecked_mut(dst_shift..).as_mut_ptr(), dst_pack1);
            }
            YuvSourceChannels::Rgba => {
                let dst_pack0: uint8x16x4_t =
                    uint8x16x4_t(r_values0, g_values0, b_values0, v_alpha);
                vst4q_u8(rgba0.get_unchecked_mut(dst_shift..).as_mut_ptr(), dst_pack0);
                let dst_pack1: uint8x16x4_t =
                    uint8x16x4_t(r_values1, g_values1, b_values1, v_alpha);
                vst4q_u8(rgba1.get_unchecked_mut(dst_shift..).as_mut_ptr(), dst_pack1);
            }
            YuvSourceChannels::Bgra => {
                let dst_pack0: uint8x16x4_t =
                    uint8x16x4_t(b_values0, g_values0, r_values0, v_alpha);
                vst4q_u8(rgba0.get_unchecked_mut(dst_shift..).as_mut_ptr(), dst_pack0);
                let dst_pack1: uint8x16x4_t =
                    uint8x16x4_t(b_values1, g_values1, r_values1, v_alpha);
                vst4q_u8(rgba1.get_unchecked_mut(dst_shift..).as_mut_ptr(), dst_pack1);
            }
        }

        cx += 16;
        ux += 16;
    }

    let shuffle_u = vld1_u8([0, 0, 2, 2, 4, 4, 6, 6].as_ptr());
    let shuffle_v = vld1_u8([1, 1, 3, 3, 5, 5, 7, 7].as_ptr());

    while cx + 8 < width {
        let y_values0 = vqsub_u8(
            vld1_u8(y_plane0.get_unchecked(cx..).as_ptr()),
            vget_low_u8(y_corr),
        );
        let y_values1 = vqsub_u8(
            vld1_u8(y_plane1.get_unchecked(cx..).as_ptr()),
            vget_low_u8(y_corr),
        );

        let mut u_low_u8: uint8x8_t;
        let mut v_low_u8: uint8x8_t;

        let uv_values = vld1_u8(uv_ptr.add(ux));

        u_low_u8 = vtbl1_u8(uv_values, shuffle_u);
        v_low_u8 = vtbl1_u8(uv_values, shuffle_v);

        #[allow(clippy::manual_swap)]
        if order == YuvNVOrder::VU {
            let new_v = u_low_u8;
            u_low_u8 = v_low_u8;
            v_low_u8 = new_v;
        }

        let u_low = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(u_low_u8)), uv_corr);
        let v_low = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(v_low_u8)), uv_corr);
        let y_low0 = vmullq_s16(vreinterpretq_s16_u16(vmovl_u8(y_values0)), v_luma_coeff);
        let y_low1 = vmullq_s16(vreinterpretq_s16_u16(vmovl_u8(y_values1)), v_luma_coeff);

        let g_coeff_lo = vweight_x2(v_low, v_g_coeff_1, u_low, v_g_coeff_2);

        let r_low0 = vdotl_s16::<PRECISION>(y_low0, v_low, v_cr_coeff);
        let b_low0 = vdotl_s16::<PRECISION>(y_low0, u_low, v_cb_coeff);
        let g_low0 = vaddn_dot::<PRECISION>(y_low0, g_coeff_lo);

        let r_low1 = vdotl_s16::<PRECISION>(y_low1, v_low, v_cr_coeff);
        let b_low1 = vdotl_s16::<PRECISION>(y_low1, u_low, v_cb_coeff);
        let g_low1 = vaddn_dot::<PRECISION>(y_low1, g_coeff_lo);

        let r_values0 = vqmovun_s16(r_low0);
        let g_values0 = vqmovun_s16(g_low0);
        let b_values0 = vqmovun_s16(b_low0);

        let r_values1 = vqmovun_s16(r_low1);
        let g_values1 = vqmovun_s16(g_low1);
        let b_values1 = vqmovun_s16(b_low1);

        let dst_shift = cx * channels;

        match destination_channels {
            YuvSourceChannels::Rgb => {
                let dst_pack0: uint8x8x3_t = uint8x8x3_t(r_values0, g_values0, b_values0);
                vst3_u8(rgba0.get_unchecked_mut(dst_shift..).as_mut_ptr(), dst_pack0);
                let dst_pack1: uint8x8x3_t = uint8x8x3_t(r_values1, g_values1, b_values1);
                vst3_u8(rgba1.get_unchecked_mut(dst_shift..).as_mut_ptr(), dst_pack1);
            }
            YuvSourceChannels::Bgr => {
                let dst_pack0: uint8x8x3_t = uint8x8x3_t(b_values0, g_values0, r_values0);
                vst3_u8(rgba0.get_unchecked_mut(dst_shift..).as_mut_ptr(), dst_pack0);
                let dst_pack1: uint8x8x3_t = uint8x8x3_t(b_values1, g_values1, r_values1);
                vst3_u8(rgba1.get_unchecked_mut(dst_shift..).as_mut_ptr(), dst_pack1);
            }
            YuvSourceChannels::Rgba => {
                let dst_pack0: uint8x8x4_t =
                    uint8x8x4_t(r_values0, g_values0, b_values0, vget_low_u8(v_alpha));
                vst4_u8(rgba0.get_unchecked_mut(dst_shift..).as_mut_ptr(), dst_pack0);
                let dst_pack1: uint8x8x4_t =
                    uint8x8x4_t(r_values1, g_values1, b_values1, vget_low_u8(v_alpha));
                vst4_u8(rgba1.get_unchecked_mut(dst_shift..).as_mut_ptr(), dst_pack1);
            }
            YuvSourceChannels::Bgra => {
                let dst_pack0: uint8x8x4_t =
                    uint8x8x4_t(b_values0, g_values0, r_values0, vget_low_u8(v_alpha));
                vst4_u8(rgba0.get_unchecked_mut(dst_shift..).as_mut_ptr(), dst_pack0);
                let dst_pack1: uint8x8x4_t =
                    uint8x8x4_t(b_values1, g_values1, r_values1, vget_low_u8(v_alpha));
                vst4_u8(rgba1.get_unchecked_mut(dst_shift..).as_mut_ptr(), dst_pack1);
            }
        }

        cx += 8;
        ux += 8;
    }

    ProcessedOffset { cx, ux }
}
