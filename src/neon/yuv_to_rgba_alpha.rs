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
use crate::neon::neon_simd_support::{
    neon_premultiply_alpha, neon_store_rgb8, vdotl_laneq_s16, vdotl_laneq_s16_x2, vmullq_laneq_s16,
};
use crate::yuv_support::{
    CbCrInverseTransform, YuvChromaRange, YuvChromaSubsampling, YuvSourceChannels,
};
use std::arch::aarch64::*;

#[target_feature(enable = "rdm")]
pub(crate) unsafe fn neon_yuv_to_rgba_alpha_rdm<
    const DESTINATION_CHANNELS: u8,
    const SAMPLING: u8,
>(
    range: &YuvChromaRange,
    transform: &CbCrInverseTransform<i32>,
    y_plane: &[u8],
    u_plane: &[u8],
    v_plane: &[u8],
    a_plane: &[u8],
    rgba: &mut [u8],
    start_cx: usize,
    start_ux: usize,
    width: usize,
    use_premultiply: bool,
) -> ProcessedOffset {
    let chroma_subsampling: YuvChromaSubsampling = SAMPLING.into();
    let destination_channels: YuvSourceChannels = DESTINATION_CHANNELS.into();
    let channels = destination_channels.get_channels_count();

    let mut cx = start_cx;
    let mut uv_x = start_ux;
    let y_ptr = y_plane.as_ptr();
    let u_ptr = u_plane.as_ptr();
    let v_ptr = v_plane.as_ptr();
    let a_ptr = a_plane.as_ptr();
    let rgba_ptr = rgba.as_mut_ptr();

    let y_corr = vdupq_n_u8(range.bias_y as u8);
    let uv_corr = vdupq_n_s16(range.bias_uv as i16);

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

    const SCALE: i32 = 2;

    while cx + 16 < width {
        let y_values = vqsubq_u8(vld1q_u8(y_ptr.add(cx)), y_corr);
        let a_values = vld1q_u8(a_ptr.add(cx));

        let u_high_u8: uint8x8_t;
        let v_high_u8: uint8x8_t;
        let u_low_u8: uint8x8_t;
        let v_low_u8: uint8x8_t;

        match chroma_subsampling {
            YuvChromaSubsampling::Yuv420 | YuvChromaSubsampling::Yuv422 => {
                let u_values = vld1_u8(u_ptr.add(uv_x));
                let v_values = vld1_u8(v_ptr.add(uv_x));

                u_high_u8 = vzip2_u8(u_values, u_values);
                v_high_u8 = vzip2_u8(v_values, v_values);
                u_low_u8 = vzip1_u8(u_values, u_values);
                v_low_u8 = vzip1_u8(v_values, v_values);
            }
            YuvChromaSubsampling::Yuv444 => {
                let u_values = vld1q_u8(u_ptr.add(uv_x));
                let v_values = vld1q_u8(v_ptr.add(uv_x));

                u_high_u8 = vget_high_u8(u_values);
                v_high_u8 = vget_high_u8(v_values);
                u_low_u8 = vget_low_u8(u_values);
                v_low_u8 = vget_low_u8(v_values);
            }
        }

        let u_high = vshlq_n_s16::<SCALE>(vsubq_s16(
            vreinterpretq_s16_u16(vmovl_u8(u_high_u8)),
            uv_corr,
        ));
        let v_high = vshlq_n_s16::<SCALE>(vsubq_s16(
            vreinterpretq_s16_u16(vmovl_u8(v_high_u8)),
            uv_corr,
        ));
        let y_high = vqrdmulhq_laneq_s16::<0>(
            vreinterpretq_s16_u16(vshll_high_n_u8::<SCALE>(y_values)),
            v_weights,
        );

        let r_high = vqmovun_s16(vqrdmlahq_laneq_s16::<1>(y_high, v_high, v_weights));
        let b_high = vqmovun_s16(vqrdmlahq_laneq_s16::<2>(y_high, u_high, v_weights));
        let g_high = vqmovun_s16(vsubq_s16(
            y_high,
            vqrdmlahq_laneq_s16::<4>(
                vqrdmulhq_laneq_s16::<3>(v_high, v_weights),
                u_high,
                v_weights,
            ),
        ));

        let u_low = vshlq_n_s16::<SCALE>(vsubq_s16(
            vreinterpretq_s16_u16(vmovl_u8(u_low_u8)),
            uv_corr,
        ));
        let v_low = vshlq_n_s16::<SCALE>(vsubq_s16(
            vreinterpretq_s16_u16(vmovl_u8(v_low_u8)),
            uv_corr,
        ));
        let y_v_shl = vshll_n_u8::<SCALE>(vget_low_u8(y_values));
        let y_low = vqrdmulhq_laneq_s16::<0>(vreinterpretq_s16_u16(y_v_shl), v_weights);

        let r_low = vqmovun_s16(vqrdmlahq_laneq_s16::<1>(y_low, v_low, v_weights));
        let b_low = vqmovun_s16(vqrdmlahq_laneq_s16::<2>(y_low, u_low, v_weights));
        let g_low = vqmovun_s16(vsubq_s16(
            y_low,
            vqrdmlahq_laneq_s16::<4>(vqrdmulhq_laneq_s16::<3>(v_low, v_weights), u_low, v_weights),
        ));

        let mut r_values = vcombine_u8(r_low, r_high);
        let mut g_values = vcombine_u8(g_low, g_high);
        let mut b_values = vcombine_u8(b_low, b_high);

        let dst_shift = cx * channels;

        if use_premultiply {
            r_values = neon_premultiply_alpha(r_values, a_values);
            g_values = neon_premultiply_alpha(g_values, a_values);
            b_values = neon_premultiply_alpha(b_values, a_values);
        }

        neon_store_rgb8::<DESTINATION_CHANNELS>(
            rgba_ptr.add(dst_shift),
            r_values,
            g_values,
            b_values,
            a_values,
        );

        cx += 16;

        match chroma_subsampling {
            YuvChromaSubsampling::Yuv420 | YuvChromaSubsampling::Yuv422 => {
                uv_x += 8;
            }
            YuvChromaSubsampling::Yuv444 => {
                uv_x += 16;
            }
        }
    }

    ProcessedOffset { cx, ux: uv_x }
}

pub(crate) unsafe fn neon_yuv_to_rgba_alpha<
    const PRECISION: i32,
    const DESTINATION_CHANNELS: u8,
    const SAMPLING: u8,
>(
    range: &YuvChromaRange,
    transform: &CbCrInverseTransform<i32>,
    y_plane: &[u8],
    u_plane: &[u8],
    v_plane: &[u8],
    a_plane: &[u8],
    rgba: &mut [u8],
    start_cx: usize,
    start_ux: usize,
    width: usize,
    use_premultiply: bool,
) -> ProcessedOffset {
    let chroma_subsampling: YuvChromaSubsampling = SAMPLING.into();
    let destination_channels: YuvSourceChannels = DESTINATION_CHANNELS.into();
    let channels = destination_channels.get_channels_count();

    let mut cx = start_cx;
    let mut uv_x = start_ux;
    let y_ptr = y_plane.as_ptr();
    let u_ptr = u_plane.as_ptr();
    let v_ptr = v_plane.as_ptr();
    let a_ptr = a_plane.as_ptr();
    let rgba_ptr = rgba.as_mut_ptr();

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

    while cx + 16 < width {
        let y_values = vqsubq_u8(vld1q_u8(y_ptr.add(cx)), y_corr);
        let a_values = vld1q_u8(a_ptr.add(cx));

        let u_high_u8: uint8x8_t;
        let v_high_u8: uint8x8_t;
        let u_low_u8: uint8x8_t;
        let v_low_u8: uint8x8_t;

        match chroma_subsampling {
            YuvChromaSubsampling::Yuv420 | YuvChromaSubsampling::Yuv422 => {
                let u_values = vld1_u8(u_ptr.add(uv_x));
                let v_values = vld1_u8(v_ptr.add(uv_x));

                u_high_u8 = vzip2_u8(u_values, u_values);
                v_high_u8 = vzip2_u8(v_values, v_values);
                u_low_u8 = vzip1_u8(u_values, u_values);
                v_low_u8 = vzip1_u8(v_values, v_values);
            }
            YuvChromaSubsampling::Yuv444 => {
                let u_values = vld1q_u8(u_ptr.add(uv_x));
                let v_values = vld1q_u8(v_ptr.add(uv_x));

                u_high_u8 = vget_high_u8(u_values);
                v_high_u8 = vget_high_u8(v_values);
                u_low_u8 = vget_low_u8(u_values);
                v_low_u8 = vget_low_u8(v_values);
            }
        }

        let u_high = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(u_high_u8)), uv_corr);
        let v_high = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(v_high_u8)), uv_corr);
        let y_high =
            vmullq_laneq_s16::<0>(vreinterpretq_s16_u16(vmovl_high_u8(y_values)), v_weights);

        let r_high = vdotl_laneq_s16::<PRECISION, 1>(y_high, v_high, v_weights);
        let b_high = vdotl_laneq_s16::<PRECISION, 2>(y_high, u_high, v_weights);
        let g_high = vdotl_laneq_s16_x2::<PRECISION, 3, 4>(y_high, v_high, u_high, v_weights);

        let u_low = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(u_low_u8)), uv_corr);
        let v_low = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(v_low_u8)), uv_corr);
        let y_low = vmullq_laneq_s16::<0>(
            vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(y_values))),
            v_weights,
        );

        let r_low = vdotl_laneq_s16::<PRECISION, 1>(y_low, v_low, v_weights);
        let b_low = vdotl_laneq_s16::<PRECISION, 2>(y_low, u_low, v_weights);
        let g_low = vdotl_laneq_s16_x2::<PRECISION, 3, 4>(y_low, v_low, u_low, v_weights);

        let mut r_values = vcombine_u8(vqmovun_s16(r_low), vqmovun_s16(r_high));
        let mut g_values = vcombine_u8(vqmovun_s16(g_low), vqmovun_s16(g_high));
        let mut b_values = vcombine_u8(vqmovun_s16(b_low), vqmovun_s16(b_high));

        let dst_shift = cx * channels;

        if use_premultiply {
            r_values = neon_premultiply_alpha(r_values, a_values);
            g_values = neon_premultiply_alpha(g_values, a_values);
            b_values = neon_premultiply_alpha(b_values, a_values);
        }

        neon_store_rgb8::<DESTINATION_CHANNELS>(
            rgba_ptr.add(dst_shift),
            r_values,
            g_values,
            b_values,
            a_values,
        );

        cx += 16;

        match chroma_subsampling {
            YuvChromaSubsampling::Yuv420 | YuvChromaSubsampling::Yuv422 => {
                uv_x += 8;
            }
            YuvChromaSubsampling::Yuv444 => {
                uv_x += 16;
            }
        }
    }

    ProcessedOffset { cx, ux: uv_x }
}
