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

use crate::neon::utils::{
    neon_store_half_rgb8, neon_store_rgb8, vexpand8_to_10, vexpand_high_8_to_10, vmullq_laneq_s16,
    xvld1q_u8_x2,
};
use crate::yuv_support::{CbCrInverseTransform, YuvChromaRange, YuvSourceChannels};
use std::arch::aarch64::*;

#[target_feature(enable = "rdm")]
pub(crate) unsafe fn neon_y_to_rgb_row_rdm<const DESTINATION_CHANNELS: u8>(
    range: &YuvChromaRange,
    transform: &CbCrInverseTransform<i32>,
    y_plane: &[u8],
    rgba: &mut [u8],
    start_cx: usize,
    width: usize,
) -> usize {
    let destination_channels: YuvSourceChannels = DESTINATION_CHANNELS.into();
    let channels = destination_channels.get_channels_count();

    let rgba_ptr = rgba.as_mut_ptr();

    let y_corr = vdupq_n_u8(range.bias_y as u8);
    let v_alpha = vdupq_n_u8(255u8);

    let mut cx = start_cx;

    const V_SCALE: i32 = 2;

    while cx + 32 < width {
        let y_vals = xvld1q_u8_x2(y_plane.get_unchecked(cx..).as_ptr());
        let y_values0 = vqsubq_u8(y_vals.0, y_corr);
        let y_values1 = vqsubq_u8(y_vals.1, y_corr);

        let y_high0 = vqrdmulhq_n_s16(
            vreinterpretq_s16_u16(vexpand_high_8_to_10(y_values0)),
            transform.y_coef as i16,
        );

        let y_high1 = vqrdmulhq_n_s16(
            vreinterpretq_s16_u16(vexpand_high_8_to_10(y_values1)),
            transform.y_coef as i16,
        );

        let r_high0 = vqmovun_s16(y_high0);
        let r_high1 = vqmovun_s16(y_high1);

        let y_low0 = vqrdmulhq_n_s16(
            vreinterpretq_s16_u16(vexpand8_to_10(vget_low_u8(y_values0))),
            transform.y_coef as i16,
        );

        let y_low1 = vqrdmulhq_n_s16(
            vreinterpretq_s16_u16(vexpand8_to_10(vget_low_u8(y_values1))),
            transform.y_coef as i16,
        );

        let r_low0 = vqmovun_s16(y_low0);
        let r_low1 = vqmovun_s16(y_low1);

        let r_values0 = vcombine_u8(r_low0, r_high0);
        let r_values1 = vcombine_u8(r_low1, r_high1);

        let dst_shift = cx * channels;

        neon_store_rgb8::<DESTINATION_CHANNELS>(
            rgba_ptr.add(dst_shift),
            r_values0,
            r_values0,
            r_values0,
            v_alpha,
        );

        neon_store_rgb8::<DESTINATION_CHANNELS>(
            rgba_ptr.add(dst_shift + channels * 16),
            r_values1,
            r_values1,
            r_values1,
            v_alpha,
        );

        cx += 32;
    }

    while cx + 16 < width {
        let y_values = vqsubq_u8(vld1q_u8(y_plane.get_unchecked(cx..).as_ptr()), y_corr);

        let y_high = vqrdmulhq_n_s16(
            vreinterpretq_s16_u16(vshll_high_n_u8::<V_SCALE>(y_values)),
            transform.y_coef as i16,
        );

        let r_high = vqmovun_s16(y_high);

        let y_low = vqrdmulhq_n_s16(
            vreinterpretq_s16_u16(vexpand8_to_10(vget_low_u8(y_values))),
            transform.y_coef as i16,
        );

        let r_low = vqmovun_s16(y_low);

        let r_values = vcombine_u8(r_low, r_high);

        let dst_shift = cx * channels;

        neon_store_rgb8::<DESTINATION_CHANNELS>(
            rgba_ptr.add(dst_shift),
            r_values,
            r_values,
            r_values,
            v_alpha,
        );

        cx += 16;
    }

    while cx + 8 < width {
        let y_values = vqsub_u8(
            vld1_u8(y_plane.get_unchecked(cx..).as_ptr()),
            vget_low_u8(y_corr),
        );

        let y_high = vqrdmulhq_n_s16(
            vreinterpretq_s16_u16(vexpand8_to_10(y_values)),
            transform.y_coef as i16,
        );

        let r_vl = vqmovun_s16(y_high);

        let dst_shift = cx * channels;

        neon_store_half_rgb8::<DESTINATION_CHANNELS>(
            rgba_ptr.add(dst_shift),
            r_vl,
            r_vl,
            r_vl,
            vget_low_u8(v_alpha),
        );

        cx += 8;
    }

    cx
}

pub(crate) unsafe fn neon_y_to_rgb_row<const PRECISION: i32, const DESTINATION_CHANNELS: u8>(
    range: &YuvChromaRange,
    transform: &CbCrInverseTransform<i32>,
    y_plane: &[u8],
    rgba: &mut [u8],
    start_cx: usize,
    width: usize,
) -> usize {
    let destination_channels: YuvSourceChannels = DESTINATION_CHANNELS.into();
    let channels = destination_channels.get_channels_count();

    let rgba_ptr = rgba.as_mut_ptr();

    let y_corr = vdupq_n_u8(range.bias_y as u8);
    let v_alpha = vdupq_n_u8(255u8);
    let v_luma_coeff = vdupq_n_s16(transform.y_coef as i16);

    let mut cx = start_cx;

    while cx + 32 < width {
        let y_vals = xvld1q_u8_x2(y_plane.get_unchecked(cx..).as_ptr());
        let y_values0 = vqsubq_u8(y_vals.0, y_corr);
        let y_values1 = vqsubq_u8(y_vals.1, y_corr);

        let y_high0 = vmullq_laneq_s16::<0>(
            vreinterpretq_s16_u16(vmovl_high_u8(y_values0)),
            v_luma_coeff,
        );

        let y_high1 = vmullq_laneq_s16::<0>(
            vreinterpretq_s16_u16(vmovl_high_u8(y_values1)),
            v_luma_coeff,
        );

        let r_high0 = vqmovun_s16(vcombine_s16(
            vrshrn_n_s32::<PRECISION>(y_high0.0),
            vrshrn_n_s32::<PRECISION>(y_high0.1),
        ));

        let r_high1 = vqmovun_s16(vcombine_s16(
            vrshrn_n_s32::<PRECISION>(y_high1.0),
            vrshrn_n_s32::<PRECISION>(y_high1.1),
        ));

        let y_low0 = vmullq_laneq_s16::<0>(
            vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(y_values0))),
            v_luma_coeff,
        );

        let y_low1 = vmullq_laneq_s16::<0>(
            vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(y_values1))),
            v_luma_coeff,
        );

        let r_low0 = vqmovun_s16(vcombine_s16(
            vrshrn_n_s32::<PRECISION>(y_low0.0),
            vrshrn_n_s32::<PRECISION>(y_low0.1),
        ));

        let r_low1 = vqmovun_s16(vcombine_s16(
            vrshrn_n_s32::<PRECISION>(y_low1.0),
            vrshrn_n_s32::<PRECISION>(y_low1.1),
        ));

        let r_values0 = vcombine_u8(r_low0, r_high0);
        let r_values1 = vcombine_u8(r_low1, r_high1);

        let dst_shift = cx * channels;

        neon_store_rgb8::<DESTINATION_CHANNELS>(
            rgba_ptr.add(dst_shift),
            r_values0,
            r_values0,
            r_values0,
            v_alpha,
        );

        neon_store_rgb8::<DESTINATION_CHANNELS>(
            rgba_ptr.add(dst_shift + 16 * channels),
            r_values1,
            r_values1,
            r_values1,
            v_alpha,
        );

        cx += 32;
    }

    while cx + 16 < width {
        let y_values = vqsubq_u8(vld1q_u8(y_plane.get_unchecked(cx..).as_ptr()), y_corr);

        let y_high =
            vmullq_laneq_s16::<0>(vreinterpretq_s16_u16(vmovl_high_u8(y_values)), v_luma_coeff);

        let r_high = vqmovun_s16(vcombine_s16(
            vrshrn_n_s32::<PRECISION>(y_high.0),
            vrshrn_n_s32::<PRECISION>(y_high.1),
        ));

        let y_low = vmullq_laneq_s16::<0>(
            vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(y_values))),
            v_luma_coeff,
        );

        let r_low = vqmovun_s16(vcombine_s16(
            vrshrn_n_s32::<PRECISION>(y_low.0),
            vrshrn_n_s32::<PRECISION>(y_low.1),
        ));

        let r_values = vcombine_u8(r_low, r_high);

        let dst_shift = cx * channels;

        neon_store_rgb8::<DESTINATION_CHANNELS>(
            rgba_ptr.add(dst_shift),
            r_values,
            r_values,
            r_values,
            v_alpha,
        );

        cx += 16;
    }

    while cx + 8 < width {
        let y_values = vqsub_u8(
            vld1_u8(y_plane.get_unchecked(cx..).as_ptr()),
            vget_low_u8(y_corr),
        );

        let y_low = vmullq_laneq_s16::<0>(vreinterpretq_s16_u16(vmovl_u8(y_values)), v_luma_coeff);

        let r_vl = vqmovun_s16(vcombine_s16(
            vrshrn_n_s32::<PRECISION>(y_low.0),
            vrshrn_n_s32::<PRECISION>(y_low.1),
        ));

        let dst_shift = cx * channels;

        neon_store_half_rgb8::<DESTINATION_CHANNELS>(
            rgba_ptr.add(dst_shift),
            r_vl,
            r_vl,
            r_vl,
            vget_low_u8(v_alpha),
        );

        cx += 8;
    }

    cx
}
