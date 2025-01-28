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
use crate::neon::utils::*;
use crate::yuv_support::{CbCrInverseTransform, YuvChromaRange, YuvNVOrder, YuvSourceChannels};
use std::arch::aarch64::*;

#[cfg(feature = "rdm")]
#[target_feature(enable = "rdm")]
/// Special path for BiPlanar YUV 4:2:0 for aarch64 with RDM available
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

    const SCALE: i32 = 2;

    let y_corr = vdupq_n_u8(range.bias_y as u8);
    let uv_corr = vdupq_n_s16((range.bias_uv as i16) << SCALE);
    let v_alpha = vdupq_n_u8(255u8);

    let mut cx = start_cx;
    let mut ux = start_ux;

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

    while cx + 32 < width {
        let y_values_0 = xvld1q_u8_x2(y_plane0.get_unchecked(cx..).as_ptr());
        let y_values_1 = xvld1q_u8_x2(y_plane1.get_unchecked(cx..).as_ptr());
        let mut uv_values = vld2q_u8(uv_ptr.add(ux));

        let y_values00 = vqsubq_u8(y_values_0.0, y_corr);
        let y_values01 = vqsubq_u8(y_values_0.1, y_corr);
        let y_values10 = vqsubq_u8(y_values_1.0, y_corr);
        let y_values11 = vqsubq_u8(y_values_1.1, y_corr);

        if order == YuvNVOrder::VU {
            uv_values = uint8x16x2_t(uv_values.1, uv_values.0);
        }

        let u_high_u8 = vzip2q_u8(uv_values.0, uv_values.0);
        let v_high_u8 = vzip2q_u8(uv_values.1, uv_values.1);

        let u_low_u8 = vzip1q_u8(uv_values.0, uv_values.0);
        let v_low_u8 = vzip1q_u8(uv_values.1, uv_values.1);

        let uhw00 = vreinterpretq_s16_u16(vshll_high_n_u8::<SCALE>(u_low_u8));
        let vhw00 = vreinterpretq_s16_u16(vshll_high_n_u8::<SCALE>(v_low_u8));

        let u_high00 = vsubq_s16(uhw00, uv_corr);
        let v_high00 = vsubq_s16(vhw00, uv_corr);

        let y_v_shl00 = vexpand_high_8_to_10(y_values00);
        let y_v_shl10 = vexpand_high_8_to_10(y_values10);

        let y_high00 = vqrdmulhq_laneq_s16::<0>(vreinterpretq_s16_u16(y_v_shl00), v_weights);
        let y_high10 = vqrdmulhq_laneq_s16::<0>(vreinterpretq_s16_u16(y_v_shl10), v_weights);

        let g_coeff_hi00 = vqrdmlahq_laneq_s16::<4>(
            vqrdmulhq_laneq_s16::<3>(v_high00, v_weights),
            u_high00,
            v_weights,
        );

        let rh00 = vqrdmlahq_laneq_s16::<1>(y_high00, v_high00, v_weights);
        let bh00 = vqrdmlahq_laneq_s16::<2>(y_high00, u_high00, v_weights);
        let gh00 = vsubq_s16(y_high00, g_coeff_hi00);
        let rh10 = vqrdmlahq_laneq_s16::<1>(y_high10, v_high00, v_weights);
        let bh10 = vqrdmlahq_laneq_s16::<2>(y_high10, u_high00, v_weights);
        let gh10 = vsubq_s16(y_high10, g_coeff_hi00);

        let r_high00 = vqmovun_s16(rh00);
        let b_high00 = vqmovun_s16(bh00);
        let g_high00 = vqmovun_s16(gh00);

        let r_high10 = vqmovun_s16(rh10);
        let b_high10 = vqmovun_s16(bh10);
        let g_high10 = vqmovun_s16(gh10);

        let uw0 = vshll_n_u8::<SCALE>(vget_low_u8(u_low_u8));
        let vw0 = vshll_n_u8::<SCALE>(vget_low_u8(v_low_u8));

        let u_low00 = vsubq_s16(vreinterpretq_s16_u16(uw0), uv_corr);
        let v_low00 = vsubq_s16(vreinterpretq_s16_u16(vw0), uv_corr);

        let y_v_shl00 = vexpand8_to_10(vget_low_u8(y_values00));
        let y_v_shl10 = vexpand8_to_10(vget_low_u8(y_values10));

        let y_low00 = vqrdmulhq_laneq_s16::<0>(vreinterpretq_s16_u16(y_v_shl00), v_weights);
        let y_low10 = vqrdmulhq_laneq_s16::<0>(vreinterpretq_s16_u16(y_v_shl10), v_weights);

        let g_coeff_lo00 = vqrdmlahq_laneq_s16::<4>(
            vqrdmulhq_laneq_s16::<3>(v_low00, v_weights),
            u_low00,
            v_weights,
        );

        let rl00 = vqrdmlahq_laneq_s16::<1>(y_low00, v_low00, v_weights);
        let bl00 = vqrdmlahq_laneq_s16::<2>(y_low00, u_low00, v_weights);
        let gl00 = vsubq_s16(y_low00, g_coeff_lo00);
        let rl10 = vqrdmlahq_laneq_s16::<1>(y_low10, v_low00, v_weights);
        let bl10 = vqrdmlahq_laneq_s16::<2>(y_low10, u_low00, v_weights);
        let gl10 = vsubq_s16(y_low10, g_coeff_lo00);

        let r_low00 = vqmovun_s16(rl00);
        let b_low00 = vqmovun_s16(bl00);
        let g_low00 = vqmovun_s16(gl00);

        let r_low10 = vqmovun_s16(rl10);
        let b_low10 = vqmovun_s16(bl10);
        let g_low10 = vqmovun_s16(gl10);

        let r_values0 = vcombine_u8(r_low00, r_high00);
        let g_values0 = vcombine_u8(g_low00, g_high00);
        let b_values0 = vcombine_u8(b_low00, b_high00);

        let r_values1 = vcombine_u8(r_low10, r_high10);
        let g_values1 = vcombine_u8(g_low10, g_high10);
        let b_values1 = vcombine_u8(b_low10, b_high10);

        let u_high01 = vsubq_s16(
            vreinterpretq_s16_u16(vshll_high_n_u8::<SCALE>(u_high_u8)),
            uv_corr,
        );
        let v_high01 = vsubq_s16(
            vreinterpretq_s16_u16(vshll_high_n_u8::<SCALE>(v_high_u8)),
            uv_corr,
        );

        let y_v_shl01 = vexpand_high_8_to_10(y_values01);
        let y_v_shl11 = vexpand_high_8_to_10(y_values11);

        let y_high01 = vqrdmulhq_laneq_s16::<0>(vreinterpretq_s16_u16(y_v_shl01), v_weights);
        let y_high11 = vqrdmulhq_laneq_s16::<0>(vreinterpretq_s16_u16(y_v_shl11), v_weights);

        let g_coeff_hi01 = vqrdmlahq_laneq_s16::<4>(
            vqrdmulhq_laneq_s16::<3>(v_high01, v_weights),
            u_high00,
            v_weights,
        );

        let rh01 = vqrdmlahq_laneq_s16::<1>(y_high01, v_high01, v_weights);
        let bh01 = vqrdmlahq_laneq_s16::<2>(y_high01, u_high01, v_weights);
        let gh01 = vsubq_s16(y_high01, g_coeff_hi01);
        let rh11 = vqrdmlahq_laneq_s16::<1>(y_high11, v_high01, v_weights);
        let bh11 = vqrdmlahq_laneq_s16::<2>(y_high11, u_high01, v_weights);
        let gh11 = vsubq_s16(y_high11, g_coeff_hi01);

        let r_high01 = vqmovun_s16(rh01);
        let b_high01 = vqmovun_s16(bh01);
        let g_high01 = vqmovun_s16(gh01);

        let r_high11 = vqmovun_s16(rh11);
        let b_high11 = vqmovun_s16(bh11);
        let g_high11 = vqmovun_s16(gh11);

        let uw01 = vshll_n_u8::<SCALE>(vget_low_u8(u_high_u8));
        let vw01 = vshll_n_u8::<SCALE>(vget_low_u8(v_high_u8));

        let u_low01 = vsubq_s16(vreinterpretq_s16_u16(uw01), uv_corr);
        let v_low01 = vsubq_s16(vreinterpretq_s16_u16(vw01), uv_corr);

        let y_v_shl01 = vexpand8_to_10(vget_low_u8(y_values01));
        let y_v_shl11 = vexpand8_to_10(vget_low_u8(y_values11));

        let y_low01 = vqrdmulhq_laneq_s16::<0>(vreinterpretq_s16_u16(y_v_shl01), v_weights);
        let y_low11 = vqrdmulhq_laneq_s16::<0>(vreinterpretq_s16_u16(y_v_shl11), v_weights);

        let g_coeff_lo01 = vqrdmlahq_laneq_s16::<4>(
            vqrdmulhq_laneq_s16::<3>(v_low01, v_weights),
            u_low01,
            v_weights,
        );

        let rl01 = vqrdmlahq_laneq_s16::<1>(y_low01, v_low01, v_weights);
        let bl01 = vqrdmlahq_laneq_s16::<2>(y_low01, u_low01, v_weights);
        let gl01 = vsubq_s16(y_low01, g_coeff_lo01);
        let rl11 = vqrdmlahq_laneq_s16::<1>(y_low11, v_low01, v_weights);
        let bl11 = vqrdmlahq_laneq_s16::<2>(y_low11, u_low01, v_weights);
        let gl11 = vsubq_s16(y_low11, g_coeff_lo01);

        let r_low01 = vqmovun_s16(rl01);
        let b_low01 = vqmovun_s16(bl01);
        let g_low01 = vqmovun_s16(gl01);

        let r_low11 = vqmovun_s16(rl11);
        let b_low11 = vqmovun_s16(bl11);
        let g_low11 = vqmovun_s16(gl11);

        let r_values01 = vcombine_u8(r_low01, r_high01);
        let g_values01 = vcombine_u8(g_low01, g_high01);
        let b_values01 = vcombine_u8(b_low01, b_high01);

        let r_values11 = vcombine_u8(r_low11, r_high11);
        let g_values11 = vcombine_u8(g_low11, g_high11);
        let b_values11 = vcombine_u8(b_low11, b_high11);

        let dst_shift = cx * channels;

        neon_store_rgb8::<DESTINATION_CHANNELS>(
            rgba0.get_unchecked_mut(dst_shift..).as_mut_ptr(),
            r_values0,
            g_values0,
            b_values0,
            v_alpha,
        );

        neon_store_rgb8::<DESTINATION_CHANNELS>(
            rgba1.get_unchecked_mut(dst_shift..).as_mut_ptr(),
            r_values1,
            g_values1,
            b_values1,
            v_alpha,
        );

        neon_store_rgb8::<DESTINATION_CHANNELS>(
            rgba0
                .get_unchecked_mut((dst_shift + channels * 16)..)
                .as_mut_ptr(),
            r_values01,
            g_values01,
            b_values01,
            v_alpha,
        );

        neon_store_rgb8::<DESTINATION_CHANNELS>(
            rgba1
                .get_unchecked_mut((dst_shift + channels * 16)..)
                .as_mut_ptr(),
            r_values11,
            g_values11,
            b_values11,
            v_alpha,
        );

        cx += 32;
        ux += 32;
    }

    while cx + 16 < width {
        let vl0 = vld1q_u8(y_plane0.get_unchecked(cx..).as_ptr());
        let vl1 = vld1q_u8(y_plane1.get_unchecked(cx..).as_ptr());
        let mut uv_values = vld2_u8(uv_ptr.add(ux));

        let y_values0 = vqsubq_u8(vl0, y_corr);
        let y_values1 = vqsubq_u8(vl1, y_corr);
        if order == YuvNVOrder::VU {
            uv_values = uint8x8x2_t(uv_values.1, uv_values.0);
        }

        let u_high_u8 = vzip2_u8(uv_values.0, uv_values.0);
        let v_high_u8 = vzip2_u8(uv_values.1, uv_values.1);
        let u_low_u8 = vzip1_u8(uv_values.0, uv_values.0);
        let v_low_u8 = vzip1_u8(uv_values.1, uv_values.1);

        let uhw = vreinterpretq_s16_u16(vshll_n_u8::<SCALE>(u_high_u8));
        let vhw = vreinterpretq_s16_u16(vshll_n_u8::<SCALE>(v_high_u8));

        let u_high = vsubq_s16(uhw, uv_corr);
        let v_high = vsubq_s16(vhw, uv_corr);
        let y_v_shl0 = vexpand_high_8_to_10(y_values0);
        let y_v_shl1 = vexpand_high_8_to_10(y_values1);
        let y_high0 = vqrdmulhq_laneq_s16::<0>(vreinterpretq_s16_u16(y_v_shl0), v_weights);
        let y_high1 = vqrdmulhq_laneq_s16::<0>(vreinterpretq_s16_u16(y_v_shl1), v_weights);

        let g_coeff_hi = vqrdmlahq_laneq_s16::<4>(
            vqrdmulhq_laneq_s16::<3>(v_high, v_weights),
            u_high,
            v_weights,
        );

        let r_high0 = vqmovun_s16(vqrdmlahq_laneq_s16::<1>(y_high0, v_high, v_weights));
        let b_high0 = vqmovun_s16(vqrdmlahq_laneq_s16::<2>(y_high0, u_high, v_weights));
        let g_high0 = vqmovun_s16(vsubq_s16(y_high0, g_coeff_hi));

        let r_high1 = vqmovun_s16(vqrdmlahq_laneq_s16::<1>(y_high1, v_high, v_weights));
        let b_high1 = vqmovun_s16(vqrdmlahq_laneq_s16::<2>(y_high1, u_high, v_weights));
        let g_high1 = vqmovun_s16(vsubq_s16(y_high1, g_coeff_hi));

        let u_low = vsubq_s16(
            vreinterpretq_s16_u16(vshll_n_u8::<SCALE>(u_low_u8)),
            uv_corr,
        );
        let v_low = vsubq_s16(
            vreinterpretq_s16_u16(vshll_n_u8::<SCALE>(v_low_u8)),
            uv_corr,
        );
        let y_v_shl0 = vexpand8_to_10(vget_low_u8(y_values0));
        let y_v_shl1 = vexpand8_to_10(vget_low_u8(y_values1));
        let y_low0 = vqrdmulhq_laneq_s16::<0>(vreinterpretq_s16_u16(y_v_shl0), v_weights);
        let y_low1 = vqrdmulhq_laneq_s16::<0>(vreinterpretq_s16_u16(y_v_shl1), v_weights);

        let g_coeff_lo =
            vqrdmlahq_laneq_s16::<4>(vqrdmulhq_laneq_s16::<3>(v_low, v_weights), u_low, v_weights);

        let r_low0 = vqmovun_s16(vqrdmlahq_laneq_s16::<1>(y_low0, v_low, v_weights));
        let b_low0 = vqmovun_s16(vqrdmlahq_laneq_s16::<2>(y_low0, u_low, v_weights));
        let g_low0 = vqmovun_s16(vsubq_s16(y_low0, g_coeff_lo));

        let r_low1 = vqmovun_s16(vqrdmlahq_laneq_s16::<1>(y_low1, v_low, v_weights));
        let b_low1 = vqmovun_s16(vqrdmlahq_laneq_s16::<2>(y_low1, u_low, v_weights));
        let g_low1 = vqmovun_s16(vsubq_s16(y_low1, g_coeff_lo));

        let r_values0 = vcombine_u8(r_low0, r_high0);
        let g_values0 = vcombine_u8(g_low0, g_high0);
        let b_values0 = vcombine_u8(b_low0, b_high0);

        let r_values1 = vcombine_u8(r_low1, r_high1);
        let g_values1 = vcombine_u8(g_low1, g_high1);
        let b_values1 = vcombine_u8(b_low1, b_high1);

        let dst_shift = cx * channels;

        neon_store_rgb8::<DESTINATION_CHANNELS>(
            rgba0.get_unchecked_mut(dst_shift..).as_mut_ptr(),
            r_values0,
            g_values0,
            b_values0,
            v_alpha,
        );
        neon_store_rgb8::<DESTINATION_CHANNELS>(
            rgba1.get_unchecked_mut(dst_shift..).as_mut_ptr(),
            r_values1,
            g_values1,
            b_values1,
            v_alpha,
        );

        cx += 16;
        ux += 16;
    }

    let shuffle_u = vld1_u8([0, 0, 2, 2, 4, 4, 6, 6].as_ptr());
    let shuffle_v = vld1_u8([1, 1, 3, 3, 5, 5, 7, 7].as_ptr());

    while cx + 8 < width {
        let dst_shift = cx * channels;

        let vl0 = vld1_u8(y_plane0.get_unchecked(cx..).as_ptr());
        let vl1 = vld1_u8(y_plane1.get_unchecked(cx..).as_ptr());
        let y_values0 = vqsub_u8(vl0, vget_low_u8(y_corr));
        let y_values1 = vqsub_u8(vl1, vget_low_u8(y_corr));

        let mut u_low_u8: uint8x8_t;
        let mut v_low_u8: uint8x8_t;

        let uv_values = vld1_u8(uv_plane.get_unchecked(ux..).as_ptr());

        u_low_u8 = vtbl1_u8(uv_values, shuffle_u);
        v_low_u8 = vtbl1_u8(uv_values, shuffle_v);

        if order == YuvNVOrder::VU {
            std::mem::swap(&mut u_low_u8, &mut v_low_u8);
        }

        let u_low = vsubq_s16(
            vreinterpretq_s16_u16(vshll_n_u8::<SCALE>(u_low_u8)),
            uv_corr,
        );
        let v_low = vsubq_s16(
            vreinterpretq_s16_u16(vshll_n_u8::<SCALE>(v_low_u8)),
            uv_corr,
        );
        let y_v_shl0 = vexpand8_to_10(y_values0);
        let y_v_shl1 = vexpand8_to_10(y_values1);
        let y_low0 = vqrdmulhq_laneq_s16::<0>(vreinterpretq_s16_u16(y_v_shl0), v_weights);
        let y_low1 = vqrdmulhq_laneq_s16::<0>(vreinterpretq_s16_u16(y_v_shl1), v_weights);

        let g_coeff_lo =
            vqrdmlahq_laneq_s16::<4>(vqrdmulhq_laneq_s16::<3>(v_low, v_weights), u_low, v_weights);

        let r_low0 = vqmovun_s16(vqrdmlahq_laneq_s16::<1>(y_low0, v_low, v_weights));
        let b_low0 = vqmovun_s16(vqrdmlahq_laneq_s16::<2>(y_low0, u_low, v_weights));
        let g_low0 = vqmovun_s16(vsubq_s16(y_low0, g_coeff_lo));

        let r_low1 = vqmovun_s16(vqrdmlahq_laneq_s16::<1>(y_low1, v_low, v_weights));
        let b_low1 = vqmovun_s16(vqrdmlahq_laneq_s16::<2>(y_low1, u_low, v_weights));
        let g_low1 = vqmovun_s16(vsubq_s16(y_low1, g_coeff_lo));

        let r_values0 = r_low0;
        let g_values0 = g_low0;
        let b_values0 = b_low0;

        let r_values1 = r_low1;
        let g_values1 = g_low1;
        let b_values1 = b_low1;

        neon_store_half_rgb8::<DESTINATION_CHANNELS>(
            rgba0.get_unchecked_mut(dst_shift..).as_mut_ptr(),
            r_values0,
            g_values0,
            b_values0,
            vget_low_u8(v_alpha),
        );
        neon_store_half_rgb8::<DESTINATION_CHANNELS>(
            rgba1.get_unchecked_mut(dst_shift..).as_mut_ptr(),
            r_values1,
            g_values1,
            b_values1,
            vget_low_u8(v_alpha),
        );

        cx += 8;
        ux += 8;
    }

    if cx < width {
        let diff = width - cx;

        assert!(diff <= 8);

        let mut dst_buffer0: [u8; 8 * 4] = [0; 8 * 4];
        let mut dst_buffer1: [u8; 8 * 4] = [0; 8 * 4];
        let mut y_buffer0: [u8; 8] = [0; 8];
        let mut y_buffer1: [u8; 8] = [0; 8];
        let mut uv_buffer: [u8; 8] = [0; 8];

        std::ptr::copy_nonoverlapping(
            y_plane0.get_unchecked(cx..).as_ptr(),
            y_buffer0.as_mut_ptr(),
            diff,
        );

        std::ptr::copy_nonoverlapping(
            y_plane1.get_unchecked(cx..).as_ptr(),
            y_buffer1.as_mut_ptr(),
            diff,
        );

        let hv = diff.div_ceil(2) * 2;

        std::ptr::copy_nonoverlapping(
            uv_plane.get_unchecked(ux..).as_ptr(),
            uv_buffer.as_mut_ptr(),
            hv,
        );

        let vl0 = vld1_u8(y_buffer0.as_ptr());
        let vl1 = vld1_u8(y_buffer1.as_ptr());
        let y_values0 = vqsub_u8(vl0, vget_low_u8(y_corr));
        let y_values1 = vqsub_u8(vl1, vget_low_u8(y_corr));

        let mut u_low_u8: uint8x8_t;
        let mut v_low_u8: uint8x8_t;

        let uv_values = vld1_u8(uv_buffer.as_ptr());

        u_low_u8 = vtbl1_u8(uv_values, shuffle_u);
        v_low_u8 = vtbl1_u8(uv_values, shuffle_v);

        if order == YuvNVOrder::VU {
            std::mem::swap(&mut u_low_u8, &mut v_low_u8);
        }

        let u_low = vsubq_s16(
            vreinterpretq_s16_u16(vshll_n_u8::<SCALE>(u_low_u8)),
            uv_corr,
        );
        let v_low = vsubq_s16(
            vreinterpretq_s16_u16(vshll_n_u8::<SCALE>(v_low_u8)),
            uv_corr,
        );
        let y_v_shl0 = vexpand8_to_10(y_values0);
        let y_v_shl1 = vexpand8_to_10(y_values1);
        let y_low0 = vqrdmulhq_laneq_s16::<0>(vreinterpretq_s16_u16(y_v_shl0), v_weights);
        let y_low1 = vqrdmulhq_laneq_s16::<0>(vreinterpretq_s16_u16(y_v_shl1), v_weights);

        let g_coeff_lo =
            vqrdmlahq_laneq_s16::<4>(vqrdmulhq_laneq_s16::<3>(v_low, v_weights), u_low, v_weights);

        let r_low0 = vqmovun_s16(vqrdmlahq_laneq_s16::<1>(y_low0, v_low, v_weights));
        let b_low0 = vqmovun_s16(vqrdmlahq_laneq_s16::<2>(y_low0, u_low, v_weights));
        let g_low0 = vqmovun_s16(vsubq_s16(y_low0, g_coeff_lo));

        let r_low1 = vqmovun_s16(vqrdmlahq_laneq_s16::<1>(y_low1, v_low, v_weights));
        let b_low1 = vqmovun_s16(vqrdmlahq_laneq_s16::<2>(y_low1, u_low, v_weights));
        let g_low1 = vqmovun_s16(vsubq_s16(y_low1, g_coeff_lo));

        let r_values0 = r_low0;
        let g_values0 = g_low0;
        let b_values0 = b_low0;

        let r_values1 = r_low1;
        let g_values1 = g_low1;
        let b_values1 = b_low1;

        neon_store_half_rgb8::<DESTINATION_CHANNELS>(
            dst_buffer0.as_mut_ptr(),
            r_values0,
            g_values0,
            b_values0,
            vget_low_u8(v_alpha),
        );
        neon_store_half_rgb8::<DESTINATION_CHANNELS>(
            dst_buffer1.as_mut_ptr(),
            r_values1,
            g_values1,
            b_values1,
            vget_low_u8(v_alpha),
        );

        let dst_shift = cx * channels;

        std::ptr::copy_nonoverlapping(
            dst_buffer0.as_mut_ptr(),
            rgba0.get_unchecked_mut(dst_shift..).as_mut_ptr(),
            diff * channels,
        );

        std::ptr::copy_nonoverlapping(
            dst_buffer1.as_mut_ptr(),
            rgba1.get_unchecked_mut(dst_shift..).as_mut_ptr(),
            diff * channels,
        );

        cx += diff;
        ux += hv;
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

    let weights_arr: [i16; 8] = [
        transform.y_coef as i16,
        transform.cr_coef as i16,
        transform.cb_coef as i16,
        -transform.g_coeff_1 as i16,
        -transform.g_coeff_2 as i16,
        0,
        0,
        0,
    ];

    let v_weights = vld1q_s16(weights_arr.as_ptr());

    let v_alpha = vdupq_n_u8(255u8);

    let mut cx = start_cx;
    let mut ux = start_ux;

    while cx + 16 < width {
        let vl0 = vld1q_u8(y_plane0.get_unchecked(cx..).as_ptr());
        let vl1 = vld1q_u8(y_plane1.get_unchecked(cx..).as_ptr());
        let mut uv_values = vld2_u8(uv_ptr.add(ux));

        let y_values0 = vqsubq_u8(vl0, y_corr);
        let y_values1 = vqsubq_u8(vl1, y_corr);

        if order == YuvNVOrder::VU {
            uv_values = uint8x8x2_t(uv_values.1, uv_values.0);
        }

        let u_high_u8 = vzip2_u8(uv_values.0, uv_values.0);
        let v_high_u8 = vzip2_u8(uv_values.1, uv_values.1);
        let u_low_u8 = vzip1_u8(uv_values.0, uv_values.0);
        let v_low_u8 = vzip1_u8(uv_values.1, uv_values.1);

        let uhw = vreinterpretq_s16_u16(vmovl_u8(u_high_u8));
        let vhw = vreinterpretq_s16_u16(vmovl_u8(v_high_u8));
        let yhw0 = vreinterpretq_s16_u16(vmovl_high_u8(y_values0));
        let yhw1 = vreinterpretq_s16_u16(vmovl_high_u8(y_values1));

        let u_high = vsubq_s16(uhw, uv_corr);
        let v_high = vsubq_s16(vhw, uv_corr);
        let y_high0 = vmullq_laneq_s16::<0>(yhw0, v_weights);
        let y_high1 = vmullq_laneq_s16::<0>(yhw1, v_weights);

        let g_coeff_hi = vweight_laneq_x2::<3, 4>(v_high, u_high, v_weights);

        let r_high0 = vdotl_laneq_s16::<PRECISION, 1>(y_high0, v_high, v_weights);
        let b_high0 = vdotl_laneq_s16::<PRECISION, 2>(y_high0, u_high, v_weights);
        let g_high0 = vraddn_dot::<PRECISION>(y_high0, g_coeff_hi);

        let r_high1 = vdotl_laneq_s16::<PRECISION, 1>(y_high1, v_high, v_weights);
        let b_high1 = vdotl_laneq_s16::<PRECISION, 2>(y_high1, u_high, v_weights);
        let g_high1 = vraddn_dot::<PRECISION>(y_high1, g_coeff_hi);

        let ulw = vreinterpretq_s16_u16(vmovl_u8(u_low_u8));
        let vlw = vreinterpretq_s16_u16(vmovl_u8(v_low_u8));
        let ylw0 = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(y_values0)));
        let ylw1 = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(y_values1)));

        let u_low = vsubq_s16(ulw, uv_corr);
        let v_low = vsubq_s16(vlw, uv_corr);
        let y_low0 = vmullq_laneq_s16::<0>(ylw0, v_weights);
        let y_low1 = vmullq_laneq_s16::<0>(ylw1, v_weights);

        let g_coeff_lo = vweight_laneq_x2::<3, 4>(v_low, u_low, v_weights);

        let r_low0 = vdotl_laneq_s16::<PRECISION, 1>(y_low0, v_low, v_weights);
        let b_low0 = vdotl_laneq_s16::<PRECISION, 2>(y_low0, u_low, v_weights);
        let g_low0 = vraddn_dot::<PRECISION>(y_low0, g_coeff_lo);

        let r_low1 = vdotl_laneq_s16::<PRECISION, 1>(y_low1, v_low, v_weights);
        let b_low1 = vdotl_laneq_s16::<PRECISION, 1>(y_low1, u_low, v_weights);
        let g_low1 = vraddn_dot::<PRECISION>(y_low1, g_coeff_lo);

        let r_values0 = vcombine_u8(vqmovun_s16(r_low0), vqmovun_s16(r_high0));
        let g_values0 = vcombine_u8(vqmovun_s16(g_low0), vqmovun_s16(g_high0));
        let b_values0 = vcombine_u8(vqmovun_s16(b_low0), vqmovun_s16(b_high0));

        let r_values1 = vcombine_u8(vqmovun_s16(r_low1), vqmovun_s16(r_high1));
        let g_values1 = vcombine_u8(vqmovun_s16(g_low1), vqmovun_s16(g_high1));
        let b_values1 = vcombine_u8(vqmovun_s16(b_low1), vqmovun_s16(b_high1));

        let dst_shift = cx * channels;

        neon_store_rgb8::<DESTINATION_CHANNELS>(
            rgba0.get_unchecked_mut(dst_shift..).as_mut_ptr(),
            r_values0,
            g_values0,
            b_values0,
            v_alpha,
        );
        neon_store_rgb8::<DESTINATION_CHANNELS>(
            rgba1.get_unchecked_mut(dst_shift..).as_mut_ptr(),
            r_values1,
            g_values1,
            b_values1,
            v_alpha,
        );

        cx += 16;
        ux += 16;
    }

    let shuffle_u = vld1_u8([0, 0, 2, 2, 4, 4, 6, 6].as_ptr());
    let shuffle_v = vld1_u8([1, 1, 3, 3, 5, 5, 7, 7].as_ptr());

    let decode_8_part =
        |dst0: &mut [u8], dst1: &mut [u8], y_src0: &[u8], y_src1: &[u8], uv_src: &[u8]| {
            let vl0 = vld1_u8(y_src0.as_ptr());
            let vl1 = vld1_u8(y_src1.as_ptr());
            let y_values0 = vqsub_u8(vl0, vget_low_u8(y_corr));
            let y_values1 = vqsub_u8(vl1, vget_low_u8(y_corr));

            let mut u_low_u8: uint8x8_t;
            let mut v_low_u8: uint8x8_t;

            let uv_values = vld1_u8(uv_src.as_ptr());

            u_low_u8 = vtbl1_u8(uv_values, shuffle_u);
            v_low_u8 = vtbl1_u8(uv_values, shuffle_v);

            if order == YuvNVOrder::VU {
                std::mem::swap(&mut u_low_u8, &mut v_low_u8);
            }

            let ulw = vreinterpretq_s16_u16(vmovl_u8(u_low_u8));
            let vlw = vreinterpretq_s16_u16(vmovl_u8(v_low_u8));
            let y0lw = vreinterpretq_s16_u16(vmovl_u8(y_values0));
            let y1lw = vreinterpretq_s16_u16(vmovl_u8(y_values1));

            let u_low = vsubq_s16(ulw, uv_corr);
            let v_low = vsubq_s16(vlw, uv_corr);

            let y_low0 = vmullq_laneq_s16::<0>(y0lw, v_weights);
            let y_low1 = vmullq_laneq_s16::<0>(y1lw, v_weights);

            let g_coeff_lo = vweight_laneq_x2::<3, 4>(v_low, u_low, v_weights);

            let r_low0 = vdotl_laneq_s16::<PRECISION, 1>(y_low0, v_low, v_weights);
            let b_low0 = vdotl_laneq_s16::<PRECISION, 2>(y_low0, u_low, v_weights);
            let g_low0 = vraddn_dot::<PRECISION>(y_low0, g_coeff_lo);

            let r_low1 = vdotl_laneq_s16::<PRECISION, 1>(y_low1, v_low, v_weights);
            let b_low1 = vdotl_laneq_s16::<PRECISION, 2>(y_low1, u_low, v_weights);
            let g_low1 = vraddn_dot::<PRECISION>(y_low1, g_coeff_lo);

            let r_values0 = vqmovun_s16(r_low0);
            let g_values0 = vqmovun_s16(g_low0);
            let b_values0 = vqmovun_s16(b_low0);

            let r_values1 = vqmovun_s16(r_low1);
            let g_values1 = vqmovun_s16(g_low1);
            let b_values1 = vqmovun_s16(b_low1);

            neon_store_half_rgb8::<DESTINATION_CHANNELS>(
                dst0.as_mut_ptr(),
                r_values0,
                g_values0,
                b_values0,
                vget_low_u8(v_alpha),
            );
            neon_store_half_rgb8::<DESTINATION_CHANNELS>(
                dst1.as_mut_ptr(),
                r_values1,
                g_values1,
                b_values1,
                vget_low_u8(v_alpha),
            );
        };

    while cx + 8 < width {
        let dst_shift = cx * channels;
        decode_8_part(
            rgba0.get_unchecked_mut(dst_shift..),
            rgba1.get_unchecked_mut(dst_shift..),
            y_plane0.get_unchecked(cx..),
            y_plane1.get_unchecked(cx..),
            uv_plane.get_unchecked(ux..),
        );
        cx += 8;
        ux += 8;
    }

    if cx < width {
        let diff = width - cx;

        assert!(diff <= 8);

        let mut dst_buffer0: [u8; 8 * 4] = [0; 8 * 4];
        let mut dst_buffer1: [u8; 8 * 4] = [0; 8 * 4];
        let mut y_buffer0: [u8; 8] = [0; 8];
        let mut y_buffer1: [u8; 8] = [0; 8];
        let mut uv_buffer: [u8; 8 * 2] = [0; 8 * 2];

        std::ptr::copy_nonoverlapping(
            y_plane0.get_unchecked(cx..).as_ptr(),
            y_buffer0.as_mut_ptr(),
            diff,
        );

        std::ptr::copy_nonoverlapping(
            y_plane1.get_unchecked(cx..).as_ptr(),
            y_buffer1.as_mut_ptr(),
            diff,
        );

        let hv = diff.div_ceil(2) * 2;

        std::ptr::copy_nonoverlapping(
            uv_plane.get_unchecked(ux..).as_ptr(),
            uv_buffer.as_mut_ptr(),
            hv,
        );

        decode_8_part(
            dst_buffer0.as_mut_slice(),
            dst_buffer1.as_mut_slice(),
            y_buffer0.as_slice(),
            y_buffer1.as_slice(),
            uv_buffer.as_slice(),
        );

        let dst_shift = cx * channels;

        std::ptr::copy_nonoverlapping(
            dst_buffer0.as_mut_ptr(),
            rgba0.get_unchecked_mut(dst_shift..).as_mut_ptr(),
            diff * channels,
        );

        std::ptr::copy_nonoverlapping(
            dst_buffer1.as_mut_ptr(),
            rgba1.get_unchecked_mut(dst_shift..).as_mut_ptr(),
            diff * channels,
        );

        cx += diff;
        ux += hv;
    }

    ProcessedOffset { cx, ux }
}
