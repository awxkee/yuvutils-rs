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
use crate::neon::neon_simd_support::{neon_store_rgb16, vld_s16_endian, vldq_s16_endian};
use crate::yuv_support::{
    CbCrInverseTransform, YuvChromaRange, YuvChromaSubsampling, YuvSourceChannels,
};

pub(crate) unsafe fn neon_yuv_p16_to_rgba16_alpha_row<
    const DESTINATION_CHANNELS: u8,
    const SAMPLING: u8,
    const ENDIANNESS: u8,
    const BYTES_POSITION: u8,
    const PRECISION: i32,
    const BIT_DEPTH: usize,
>(
    y_ld_ptr: &[u16],
    u_ld_ptr: &[u16],
    v_ld_ptr: &[u16],
    a_ld_ptr: &[u16],
    rgba: &mut [u16],
    width: u32,
    range: &YuvChromaRange,
    transform: &CbCrInverseTransform<i32>,
    start_cx: usize,
    start_ux: usize,
) -> ProcessedOffset {
    let destination_channels: YuvSourceChannels = DESTINATION_CHANNELS.into();
    if destination_channels == YuvSourceChannels::Rgb
        || destination_channels == YuvSourceChannels::Bgr
    {
        unreachable!("Cannot call YUV p16 to Rgb8 with alpha without real alpha");
    }
    let channels = destination_channels.get_channels_count();
    let chroma_subsampling: YuvChromaSubsampling = SAMPLING.into();

    let y_corr = vdupq_n_u16(range.bias_y as u16);
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

    let v_msb_shift = vdupq_n_s16(BIT_DEPTH as i16 - 16);

    let mut cx = start_cx;
    let mut ux = start_ux;

    while cx + 8 < width as usize {
        let a_values_l = vld1q_u16(a_ld_ptr.get_unchecked(cx..).as_ptr());

        let y_values: int16x8_t = vreinterpretq_s16_u16(vqsubq_u16(
            vreinterpretq_u16_s16(vldq_s16_endian::<ENDIANNESS, BYTES_POSITION>(
                y_ld_ptr.get_unchecked(cx..).as_ptr(),
                v_msb_shift,
            )),
            y_corr,
        ));

        let u_high: int16x4_t;
        let v_high: int16x4_t;
        let u_low: int16x4_t;
        let v_low: int16x4_t;

        if chroma_subsampling == YuvChromaSubsampling::Yuv444 {
            let mut u_values_l = vldq_s16_endian::<ENDIANNESS, BYTES_POSITION>(
                u_ld_ptr.get_unchecked(ux..).as_ptr(),
                v_msb_shift,
            );
            let mut v_values_l = vldq_s16_endian::<ENDIANNESS, BYTES_POSITION>(
                v_ld_ptr.get_unchecked(ux..).as_ptr(),
                v_msb_shift,
            );

            u_values_l = vsubq_s16(u_values_l, uv_corr);
            v_values_l = vsubq_s16(v_values_l, uv_corr);

            u_high = vget_high_s16(u_values_l);
            u_low = vget_low_s16(u_values_l);
            v_high = vget_high_s16(v_values_l);
            v_low = vget_low_s16(v_values_l);
        } else {
            let mut u_values_l = vld_s16_endian::<ENDIANNESS, BYTES_POSITION>(
                u_ld_ptr.get_unchecked(ux..).as_ptr(),
                vget_low_s16(v_msb_shift),
            );
            let mut v_values_l = vld_s16_endian::<ENDIANNESS, BYTES_POSITION>(
                v_ld_ptr.get_unchecked(ux..).as_ptr(),
                vget_low_s16(v_msb_shift),
            );
            u_values_l = vsub_s16(u_values_l, vget_low_s16(uv_corr));
            v_values_l = vsub_s16(v_values_l, vget_low_s16(uv_corr));

            u_high = vzip2_s16(u_values_l, u_values_l);
            v_high = vzip2_s16(v_values_l, v_values_l);

            u_low = vzip1_s16(u_values_l, u_values_l);
            v_low = vzip1_s16(v_values_l, v_values_l);
        }

        let y_high = vmull_high_laneq_s16::<0>(y_values, v_weights);

        let r_high = vqrshrun_n_s32::<PRECISION>(vmlal_laneq_s16::<1>(y_high, v_high, v_weights));
        let b_high = vqrshrun_n_s32::<PRECISION>(vmlal_laneq_s16::<2>(y_high, u_high, v_weights));
        let g_high = vqrshrun_n_s32::<PRECISION>(vmlal_laneq_s16::<4>(
            vmlal_laneq_s16::<3>(y_high, v_high, v_weights),
            u_high,
            v_weights,
        ));

        let y_low = vmull_laneq_s16::<0>(vget_low_s16(y_values), v_weights);

        let r_low = vqrshrun_n_s32::<PRECISION>(vmlal_laneq_s16::<1>(y_low, v_low, v_weights));
        let b_low = vqrshrun_n_s32::<PRECISION>(vmlal_laneq_s16::<2>(y_low, u_low, v_weights));
        let g_low = vqrshrun_n_s32::<PRECISION>(vmlal_laneq_s16::<4>(
            vmlal_laneq_s16::<3>(y_low, v_low, v_weights),
            u_low,
            v_weights,
        ));

        let r_values = vcombine_u16(r_low, r_high);
        let g_values = vcombine_u16(g_low, g_high);
        let b_values = vcombine_u16(b_low, b_high);

        let v_alpha = a_values_l;

        neon_store_rgb16::<DESTINATION_CHANNELS>(
            rgba.get_unchecked_mut(cx * channels..).as_mut_ptr(),
            r_values,
            g_values,
            b_values,
            v_alpha,
        );

        cx += 8;

        match chroma_subsampling {
            YuvChromaSubsampling::Yuv420 | YuvChromaSubsampling::Yuv422 => {
                ux += 4;
            }
            YuvChromaSubsampling::Yuv444 => {
                ux += 8;
            }
        }
    }

    ProcessedOffset { cx, ux }
}

#[target_feature(enable = "rdm")]
pub(crate) unsafe fn neon_yuv_p16_to_rgba16_alpha_row_rdm<
    const DESTINATION_CHANNELS: u8,
    const SAMPLING: u8,
    const ENDIANNESS: u8,
    const BYTES_POSITION: u8,
    const PRECISION: i32,
    const BIT_DEPTH: usize,
>(
    y_ld_ptr: &[u16],
    u_ld_ptr: &[u16],
    v_ld_ptr: &[u16],
    a_ld_ptr: &[u16],
    rgba: &mut [u16],
    width: u32,
    range: &YuvChromaRange,
    transform: &CbCrInverseTransform<i32>,
    start_cx: usize,
    start_ux: usize,
) -> ProcessedOffset {
    let destination_channels: YuvSourceChannels = DESTINATION_CHANNELS.into();
    if destination_channels == YuvSourceChannels::Rgb
        || destination_channels == YuvSourceChannels::Bgr
    {
        unreachable!("Cannot call YUV p16 to Rgb8 with alpha without real alpha");
    }
    let channels = destination_channels.get_channels_count();
    let chroma_subsampling: YuvChromaSubsampling = SAMPLING.into();

    let y_corr = vdupq_n_u16(range.bias_y as u16);
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

    let v_msb_shift = vdupq_n_s16(BIT_DEPTH as i16 - 16);
    let zeros = vdupq_n_s16(0);
    let v_max_values = vdupq_n_u16((1 << BIT_DEPTH) - 1);

    let mut cx = start_cx;
    let mut ux = start_ux;

    const SCALE: i32 = 2;

    while cx + 8 < width as usize {
        let a_values_l = vld1q_u16(a_ld_ptr.get_unchecked(cx..).as_ptr());

        let y_values: int16x8_t = vreinterpretq_s16_u16(vqsubq_u16(
            vreinterpretq_u16_s16(vldq_s16_endian::<ENDIANNESS, BYTES_POSITION>(
                y_ld_ptr.get_unchecked(cx..).as_ptr(),
                v_msb_shift,
            )),
            y_corr,
        ));

        let u_values: int16x8_t;
        let v_values: int16x8_t;

        if chroma_subsampling == YuvChromaSubsampling::Yuv444 {
            let mut u_values_l = vldq_s16_endian::<ENDIANNESS, BYTES_POSITION>(
                u_ld_ptr.get_unchecked(ux..).as_ptr(),
                v_msb_shift,
            );
            let mut v_values_l = vldq_s16_endian::<ENDIANNESS, BYTES_POSITION>(
                v_ld_ptr.get_unchecked(ux..).as_ptr(),
                v_msb_shift,
            );

            u_values_l = vsubq_s16(u_values_l, uv_corr);
            v_values_l = vsubq_s16(v_values_l, uv_corr);

            u_values = vshlq_n_s16::<SCALE>(u_values_l);
            v_values = vshlq_n_s16::<SCALE>(v_values_l);
        } else {
            let mut u_values_l = vld_s16_endian::<ENDIANNESS, BYTES_POSITION>(
                u_ld_ptr.get_unchecked(ux..).as_ptr(),
                vget_low_s16(v_msb_shift),
            );
            let mut v_values_l = vld_s16_endian::<ENDIANNESS, BYTES_POSITION>(
                v_ld_ptr.get_unchecked(ux..).as_ptr(),
                vget_low_s16(v_msb_shift),
            );
            u_values_l = vsub_s16(u_values_l, vget_low_s16(uv_corr));
            v_values_l = vsub_s16(v_values_l, vget_low_s16(uv_corr));

            let u_high = vzip2_s16(u_values_l, u_values_l);
            let v_high = vzip2_s16(v_values_l, v_values_l);

            let u_low = vzip1_s16(u_values_l, u_values_l);
            let v_low = vzip1_s16(v_values_l, v_values_l);

            u_values = vshlq_n_s16::<SCALE>(vcombine_s16(u_low, u_high));
            v_values = vshlq_n_s16::<SCALE>(vcombine_s16(v_low, v_high));
        }

        let y_high = vqrdmulhq_laneq_s16::<0>(vshlq_n_s16::<SCALE>(y_values), v_weights);

        let r_vals = vqrdmlahq_laneq_s16::<1>(y_high, v_values, v_weights);
        let b_vals = vqrdmlahq_laneq_s16::<2>(y_high, u_values, v_weights);
        let g_vals = vqrdmlahq_laneq_s16::<4>(
            vqrdmlahq_laneq_s16::<3>(y_high, v_values, v_weights),
            u_values,
            v_weights,
        );

        let r_values = vminq_u16(
            vreinterpretq_u16_s16(vmaxq_s16(r_vals, zeros)),
            v_max_values,
        );
        let g_values = vminq_u16(
            vreinterpretq_u16_s16(vmaxq_s16(g_vals, zeros)),
            v_max_values,
        );
        let b_values = vminq_u16(
            vreinterpretq_u16_s16(vmaxq_s16(b_vals, zeros)),
            v_max_values,
        );

        let v_alpha = a_values_l;

        neon_store_rgb16::<DESTINATION_CHANNELS>(
            rgba.get_unchecked_mut(cx * channels..).as_mut_ptr(),
            r_values,
            g_values,
            b_values,
            v_alpha,
        );

        cx += 8;

        match chroma_subsampling {
            YuvChromaSubsampling::Yuv420 | YuvChromaSubsampling::Yuv422 => {
                ux += 4;
            }
            YuvChromaSubsampling::Yuv444 => {
                ux += 8;
            }
        }
    }

    ProcessedOffset { cx, ux }
}
