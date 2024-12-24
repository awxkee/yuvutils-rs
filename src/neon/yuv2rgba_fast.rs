/*
 * Copyright (c) Radzivon Bartoshyk, 12/2024. All rights reserved.
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
use crate::neon::utils::{
    neon_store_half_rgb8, neon_store_rgb8, vexpand8_to_16, vexpand_high_8_to_16, vumulhiq_u16,
    xvld1q_u8_x2,
};
use crate::yuv_support::{CbCrInverseTransform, YuvChromaRange, YuvSourceChannels};
use crate::YuvChromaSubsampling;
use std::arch::aarch64::*;

pub(crate) unsafe fn neon_yuv_to_rgba_fast_row<
    const DESTINATION_CHANNELS: u8,
    const SAMPLING: u8,
>(
    range: &YuvChromaRange,
    transform: &CbCrInverseTransform<i32>,
    y_plane: &[u8],
    u_plane: &[u8],
    v_plane: &[u8],
    rgba: &mut [u8],
    start_cx: usize,
    start_ux: usize,
    width: usize,
) -> ProcessedOffset {
    const PRECISION: i32 = 6;
    let chroma_subsampling: YuvChromaSubsampling = SAMPLING.into();
    let destination_channels: YuvSourceChannels = DESTINATION_CHANNELS.into();
    let channels = destination_channels.get_channels_count();

    let mut cx = start_cx;
    let mut uv_x = start_ux;

    let y_ptr = y_plane.as_ptr();
    let u_ptr = u_plane.as_ptr();
    let v_ptr = v_plane.as_ptr();
    let rgba_ptr = rgba.as_mut_ptr();

    let y_corr = vdupq_n_u8(range.bias_y as u8);
    let uv_corr = vdupq_n_s16(range.bias_uv as i16);

    let cb_coeff = vdupq_n_s16(transform.cb_coef as i16);
    let cr_coeff = vdupq_n_s16(transform.cr_coef as i16);
    let y_coeff = vdupq_n_u16(transform.y_coef as u16);
    let g_coeff1 = vdupq_n_s16(-transform.g_coeff_1 as i16);
    let g_coeff2 = vdupq_n_s16(-transform.g_coeff_2 as i16);

    let v_alpha = vdupq_n_u8(255u8);

    while cx + 32 < width {
        let mut y_set = xvld1q_u8_x2(y_ptr.add(cx));
        y_set.0 = vqsubq_u8(y_set.0, y_corr);
        y_set.1 = vqsubq_u8(y_set.1, y_corr);

        let u_high_u8: uint8x16_t;
        let v_high_u8: uint8x16_t;
        let u_low_u8: uint8x16_t;
        let v_low_u8: uint8x16_t;

        match chroma_subsampling {
            YuvChromaSubsampling::Yuv420 | YuvChromaSubsampling::Yuv422 => {
                let u_values = vld1q_u8(u_ptr.add(uv_x));
                let v_values = vld1q_u8(v_ptr.add(uv_x));

                u_high_u8 = vzip2q_u8(u_values, u_values);
                v_high_u8 = vzip2q_u8(v_values, v_values);
                u_low_u8 = vzip1q_u8(u_values, u_values);
                v_low_u8 = vzip1q_u8(v_values, v_values);
            }
            YuvChromaSubsampling::Yuv444 => {
                let u_values = xvld1q_u8_x2(u_ptr.add(uv_x));
                let v_values = xvld1q_u8_x2(v_ptr.add(uv_x));

                u_high_u8 = u_values.1;
                v_high_u8 = v_values.1;
                u_low_u8 = u_values.0;
                v_low_u8 = v_values.0;
            }
        }

        let u_high1 = vsubq_s16(vreinterpretq_s16_u16(vmovl_high_u8(u_high_u8)), uv_corr);
        let v_high1 = vsubq_s16(vreinterpretq_s16_u16(vmovl_high_u8(v_high_u8)), uv_corr);
        let y_high1 = vumulhiq_u16(vexpand_high_8_to_16(y_set.1), y_coeff);

        let r_high1 = vqaddq_s16(y_high1, vmulq_s16(v_high1, cr_coeff));
        let b_high1 = vqaddq_s16(y_high1, vmulq_s16(u_high1, cb_coeff));
        let g_high1 = vqaddq_s16(
            vqaddq_s16(y_high1, vmulq_s16(v_high1, g_coeff1)),
            vmulq_s16(u_high1, g_coeff2),
        );

        let u_high0 = vsubq_s16(
            vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(u_high_u8))),
            uv_corr,
        );
        let v_high0 = vsubq_s16(
            vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(v_high_u8))),
            uv_corr,
        );
        let y_high0 = vumulhiq_u16(vexpand8_to_16(vget_low_u8(y_set.1)), y_coeff);

        let r_high0 = vqaddq_s16(y_high0, vmulq_s16(v_high0, cr_coeff));
        let b_high0 = vqaddq_s16(y_high0, vmulq_s16(u_high0, cb_coeff));
        let g_high0 = vqaddq_s16(
            vqaddq_s16(y_high0, vmulq_s16(v_high0, g_coeff1)),
            vmulq_s16(u_high0, g_coeff2),
        );

        let u_low1 = vsubq_s16(vreinterpretq_s16_u16(vmovl_high_u8(u_low_u8)), uv_corr);
        let v_low1 = vsubq_s16(vreinterpretq_s16_u16(vmovl_high_u8(v_low_u8)), uv_corr);
        let y_low1 = vumulhiq_u16(vexpand_high_8_to_16(y_set.0), y_coeff);

        let r_low1 = vqaddq_s16(y_low1, vmulq_s16(v_low1, cr_coeff));
        let b_low1 = vqaddq_s16(y_low1, vmulq_s16(u_low1, cb_coeff));
        let g_low1 = vqaddq_s16(
            vqaddq_s16(y_low1, vmulq_s16(v_low1, g_coeff1)),
            vmulq_s16(u_low1, g_coeff2),
        );

        let u_low0 = vsubq_s16(
            vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(u_low_u8))),
            uv_corr,
        );
        let v_low0 = vsubq_s16(
            vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(v_low_u8))),
            uv_corr,
        );
        let y_low0 = vumulhiq_u16(vexpand8_to_16(vget_low_u8(y_set.0)), y_coeff);

        let r_low0 = vqaddq_s16(y_low0, vmulq_s16(v_low0, cr_coeff));
        let b_low0 = vqaddq_s16(y_low0, vmulq_s16(u_low0, cb_coeff));
        let g_low0 = vqaddq_s16(
            vqaddq_s16(y_low0, vmulq_s16(v_low0, g_coeff1)),
            vmulq_s16(u_low0, g_coeff2),
        );

        let r_values0 = vcombine_u8(
            vqrshrun_n_s16::<PRECISION>(r_low0),
            vqrshrun_n_s16::<PRECISION>(r_low1),
        );
        let g_values0 = vcombine_u8(
            vqrshrun_n_s16::<PRECISION>(g_low0),
            vqrshrun_n_s16::<PRECISION>(g_low1),
        );
        let b_values0 = vcombine_u8(
            vqrshrun_n_s16::<PRECISION>(b_low0),
            vqrshrun_n_s16::<PRECISION>(b_low1),
        );

        let r_values1 = vcombine_u8(
            vqrshrun_n_s16::<PRECISION>(r_high0),
            vqrshrun_n_s16::<PRECISION>(r_high1),
        );
        let g_values1 = vcombine_u8(
            vqrshrun_n_s16::<PRECISION>(g_high0),
            vqrshrun_n_s16::<PRECISION>(g_high1),
        );
        let b_values1 = vcombine_u8(
            vqrshrun_n_s16::<PRECISION>(b_high0),
            vqrshrun_n_s16::<PRECISION>(b_high1),
        );

        let dst_shift = cx * channels;

        neon_store_rgb8::<DESTINATION_CHANNELS>(
            rgba_ptr.add(dst_shift),
            r_values0,
            g_values0,
            b_values0,
            v_alpha,
        );

        neon_store_rgb8::<DESTINATION_CHANNELS>(
            rgba_ptr.add(dst_shift + 16 * channels),
            r_values1,
            g_values1,
            b_values1,
            v_alpha,
        );

        cx += 32;

        match chroma_subsampling {
            YuvChromaSubsampling::Yuv420 | YuvChromaSubsampling::Yuv422 => {
                uv_x += 16;
            }
            YuvChromaSubsampling::Yuv444 => {
                uv_x += 32;
            }
        }
    }

    while cx + 16 < width {
        let y_values = vqsubq_u8(vld1q_u8(y_ptr.add(cx)), y_corr);

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
        let y_high = vumulhiq_u16(vexpand_high_8_to_16(y_values), y_coeff);

        let r_high = vqaddq_s16(y_high, vmulq_s16(v_high, cr_coeff));
        let b_high = vqaddq_s16(y_high, vmulq_s16(u_high, cb_coeff));
        let g_high = vqaddq_s16(
            vqaddq_s16(y_high, vmulq_s16(v_high, g_coeff1)),
            vmulq_s16(u_high, g_coeff2),
        );

        let u_low = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(u_low_u8)), uv_corr);
        let v_low = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(v_low_u8)), uv_corr);
        let y_low = vumulhiq_u16(vexpand8_to_16(vget_low_u8(y_values)), y_coeff);

        let r_low = vqaddq_s16(y_low, vmulq_s16(v_low, cr_coeff));
        let b_low = vqaddq_s16(y_low, vmulq_s16(u_low, cb_coeff));
        let g_low = vqaddq_s16(
            vqaddq_s16(y_low, vmulq_s16(v_low, g_coeff1)),
            vmulq_s16(u_low, g_coeff2),
        );

        let r_values = vcombine_u8(
            vqrshrun_n_s16::<PRECISION>(r_low),
            vqrshrun_n_s16::<PRECISION>(r_high),
        );
        let g_values = vcombine_u8(
            vqrshrun_n_s16::<PRECISION>(g_low),
            vqrshrun_n_s16::<PRECISION>(g_high),
        );
        let b_values = vcombine_u8(
            vqrshrun_n_s16::<PRECISION>(b_low),
            vqrshrun_n_s16::<PRECISION>(b_high),
        );

        let dst_shift = cx * channels;

        neon_store_rgb8::<DESTINATION_CHANNELS>(
            rgba_ptr.add(dst_shift),
            r_values,
            g_values,
            b_values,
            v_alpha,
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

    while cx + 8 < width {
        let y_values = vqsub_u8(vld1_u8(y_ptr.add(cx)), vget_low_u8(y_corr));

        let u_low_u8: uint8x8_t;
        let v_low_u8: uint8x8_t;

        match chroma_subsampling {
            YuvChromaSubsampling::Yuv420 | YuvChromaSubsampling::Yuv422 => {
                let u_values = vreinterpret_u8_u32(vld1_dup_u32(u_ptr.add(uv_x) as *const u32));
                let v_values = vreinterpret_u8_u32(vld1_dup_u32(v_ptr.add(uv_x) as *const u32));

                u_low_u8 = vzip1_u8(u_values, u_values);
                v_low_u8 = vzip1_u8(v_values, v_values);
            }
            YuvChromaSubsampling::Yuv444 => {
                let u_values = vld1_u8(u_ptr.add(uv_x));
                let v_values = vld1_u8(v_ptr.add(uv_x));

                u_low_u8 = u_values;
                v_low_u8 = v_values;
            }
        }

        let u_low = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(u_low_u8)), uv_corr);
        let v_low = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(v_low_u8)), uv_corr);
        let y_low = vumulhiq_u16(vexpand8_to_16(y_values), y_coeff);

        let r_low = vqaddq_s16(y_low, vmulq_s16(v_low, cr_coeff));
        let b_low = vqaddq_s16(y_low, vmulq_s16(u_low, cb_coeff));
        let g_low = vqaddq_s16(
            vqaddq_s16(y_low, vmulq_s16(v_low, g_coeff1)),
            vmulq_s16(u_low, g_coeff2),
        );

        let r_values = vqrshrun_n_s16::<PRECISION>(r_low);
        let g_values = vqrshrun_n_s16::<PRECISION>(g_low);
        let b_values = vqrshrun_n_s16::<PRECISION>(b_low);

        let dst_shift = cx * channels;

        neon_store_half_rgb8::<DESTINATION_CHANNELS>(
            rgba_ptr.add(dst_shift),
            r_values,
            g_values,
            b_values,
            vget_low_u8(v_alpha),
        );

        cx += 8;

        match chroma_subsampling {
            YuvChromaSubsampling::Yuv420 | YuvChromaSubsampling::Yuv422 => {
                uv_x += 4;
            }
            YuvChromaSubsampling::Yuv444 => {
                uv_x += 8;
            }
        }
    }

    ProcessedOffset { cx, ux: uv_x }
}
