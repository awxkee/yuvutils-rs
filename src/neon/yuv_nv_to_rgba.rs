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
use crate::yuv_support::{
    CbCrInverseTransform, YuvChromaRange, YuvChromaSubsampling, YuvNVOrder, YuvSourceChannels,
};
use std::arch::aarch64::*;

#[cfg(feature = "rdm")]
#[target_feature(enable = "rdm")]
pub(crate) unsafe fn neon_yuv_nv_to_rgba_row_rdm<
    const UV_ORDER: u8,
    const DESTINATION_CHANNELS: u8,
    const YUV_CHROMA_SAMPLING: u8,
>(
    range: &YuvChromaRange,
    transform: &CbCrInverseTransform<i32>,
    y_plane: &[u8],
    uv_plane: &[u8],
    rgba: &mut [u8],
    start_cx: usize,
    start_ux: usize,
    width: usize,
) -> ProcessedOffset {
    let order: YuvNVOrder = UV_ORDER.into();
    let destination_channels: YuvSourceChannels = DESTINATION_CHANNELS.into();
    let chroma_subsampling: YuvChromaSubsampling = YUV_CHROMA_SAMPLING.into();
    let channels = destination_channels.get_channels_count();

    let y_ptr = y_plane.as_ptr();
    let uv_ptr = uv_plane.as_ptr();
    let bgra_ptr = rgba.as_mut_ptr();

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
        -transform.g_coeff_1 as i16,
        -transform.g_coeff_2 as i16,
        0,
        0,
        0,
    ];

    let v_weights = vld1q_s16(weights_arr.as_ptr());

    while cx + 32 < width {
        let y_vals = xvld1q_u8_x2(y_ptr.add(cx));

        let u_high_u8: uint8x16_t;
        let v_high_u8: uint8x16_t;
        let u_low_u8: uint8x16_t;
        let v_low_u8: uint8x16_t;

        match chroma_subsampling {
            YuvChromaSubsampling::Yuv420 | YuvChromaSubsampling::Yuv422 => {
                let mut uv_values = vld2q_u8(uv_ptr.add(ux));
                if order == YuvNVOrder::VU {
                    uv_values = uint8x16x2_t(uv_values.1, uv_values.0);
                }

                u_high_u8 = vzip2q_u8(uv_values.0, uv_values.0);
                v_high_u8 = vzip2q_u8(uv_values.1, uv_values.1);
                u_low_u8 = vzip1q_u8(uv_values.0, uv_values.0);
                v_low_u8 = vzip1q_u8(uv_values.1, uv_values.1);
            }
            YuvChromaSubsampling::Yuv444 => {
                let mut uv_values0 = vld2q_u8(uv_ptr.add(ux));
                let mut uv_values1 = vld2q_u8(uv_ptr.add(ux + 16 * 2));
                if order == YuvNVOrder::VU {
                    uv_values0 = uint8x16x2_t(uv_values0.1, uv_values0.0);
                    uv_values1 = uint8x16x2_t(uv_values1.1, uv_values1.0);
                }
                u_high_u8 = uv_values1.0;
                v_high_u8 = uv_values1.1;
                u_low_u8 = uv_values0.0;
                v_low_u8 = uv_values0.1;
            }
        }

        let y_values0 = vqsubq_u8(y_vals.0, y_corr);
        let y_values1 = vqsubq_u8(y_vals.1, y_corr);

        let uh0 = vshll_n_u8::<SCALE>(vget_low_u8(u_high_u8));
        let vh0 = vshll_n_u8::<SCALE>(vget_low_u8(v_high_u8));
        let yh0 = vexpand8_to_10(vget_low_u8(y_values1));

        let u_high0 = vsubq_s16(vreinterpretq_s16_u16(uh0), uv_corr);
        let v_high0 = vsubq_s16(vreinterpretq_s16_u16(vh0), uv_corr);
        let y_high0 = vqrdmulhq_laneq_s16::<0>(vreinterpretq_s16_u16(yh0), v_weights);

        let uh1 = vshll_high_n_u8::<SCALE>(u_high_u8);
        let vh1 = vshll_high_n_u8::<SCALE>(v_high_u8);
        let yh1 = vexpand_high_8_to_10(y_values1);

        let u_high1 = vsubq_s16(vreinterpretq_s16_u16(uh1), uv_corr);
        let v_high1 = vsubq_s16(vreinterpretq_s16_u16(vh1), uv_corr);
        let y_high1 = vqrdmulhq_laneq_s16::<0>(vreinterpretq_s16_u16(yh1), v_weights);

        let rhc1 = vqrdmlahq_laneq_s16::<1>(y_high1, v_high1, v_weights);
        let bhc1 = vqrdmlahq_laneq_s16::<2>(y_high1, u_high1, v_weights);
        let ghc1 = vqrdmlahq_laneq_s16::<3>(y_high1, v_high1, v_weights);
        let rhc0 = vqrdmlahq_laneq_s16::<1>(y_high0, v_high0, v_weights);
        let bhc0 = vqrdmlahq_laneq_s16::<2>(y_high0, u_high0, v_weights);
        let ghc0 = vqrdmlahq_laneq_s16::<3>(y_high0, v_high0, v_weights);

        let r_high1 = vqmovun_s16(rhc1);
        let b_high1 = vqmovun_s16(bhc1);
        let g_high1 = vqmovun_s16(vqrdmlahq_laneq_s16::<4>(ghc1, u_high1, v_weights));

        let r_high0 = vqmovun_s16(rhc0);
        let b_high0 = vqmovun_s16(bhc0);
        let g_high0 = vqmovun_s16(vqrdmlahq_laneq_s16::<4>(ghc0, u_high0, v_weights));

        let uh0 = vshll_n_u8::<SCALE>(vget_low_u8(u_low_u8));
        let vh0 = vshll_n_u8::<SCALE>(vget_low_u8(v_low_u8));
        let yh0 = vexpand8_to_10(vget_low_u8(y_values0));

        let u_low0 = vsubq_s16(vreinterpretq_s16_u16(uh0), uv_corr);
        let v_low0 = vsubq_s16(vreinterpretq_s16_u16(vh0), uv_corr);
        let y_low0 = vqrdmulhq_laneq_s16::<0>(vreinterpretq_s16_u16(yh0), v_weights);

        let ul1 = vshll_high_n_u8::<SCALE>(u_low_u8);
        let vl1 = vshll_high_n_u8::<SCALE>(v_low_u8);
        let yl1 = vexpand_high_8_to_10(y_values0);

        let u_low1 = vsubq_s16(vreinterpretq_s16_u16(ul1), uv_corr);
        let v_low1 = vsubq_s16(vreinterpretq_s16_u16(vl1), uv_corr);
        let y_low1 = vqrdmulhq_laneq_s16::<0>(vreinterpretq_s16_u16(yl1), v_weights);

        let rlc0 = vqrdmlahq_laneq_s16::<1>(y_low0, v_low0, v_weights);
        let bhc0 = vqrdmlahq_laneq_s16::<2>(y_low0, u_low0, v_weights);
        let ghc0 = vqrdmlahq_laneq_s16::<3>(y_low0, v_low0, v_weights);
        let rlc1 = vqrdmlahq_laneq_s16::<1>(y_low1, v_low1, v_weights);
        let blc1 = vqrdmlahq_laneq_s16::<2>(y_low1, u_low1, v_weights);
        let glc1 = vqrdmlahq_laneq_s16::<3>(y_low1, v_low1, v_weights);

        let r_low0 = vqmovun_s16(rlc0);
        let b_low0 = vqmovun_s16(bhc0);
        let g_low0 = vqmovun_s16(vqrdmlahq_laneq_s16::<4>(ghc0, u_low0, v_weights));

        let r_low1 = vqmovun_s16(rlc1);
        let b_low1 = vqmovun_s16(blc1);
        let g_low1 = vqmovun_s16(vqrdmlahq_laneq_s16::<4>(glc1, u_low1, v_weights));

        let r_high = vcombine_u8(r_high0, r_high1);
        let g_high = vcombine_u8(g_high0, g_high1);
        let b_high = vcombine_u8(b_high0, b_high1);

        let r_low = vcombine_u8(r_low0, r_low1);
        let g_low = vcombine_u8(g_low0, g_low1);
        let b_low = vcombine_u8(b_low0, b_low1);

        let dst_shift = cx * channels;

        neon_store_rgb8::<DESTINATION_CHANNELS>(
            bgra_ptr.add(dst_shift),
            r_low,
            g_low,
            b_low,
            v_alpha,
        );

        neon_store_rgb8::<DESTINATION_CHANNELS>(
            bgra_ptr.add(dst_shift + 16 * channels),
            r_high,
            g_high,
            b_high,
            v_alpha,
        );

        cx += 32;

        match chroma_subsampling {
            YuvChromaSubsampling::Yuv420 | YuvChromaSubsampling::Yuv422 => {
                ux += 32;
            }
            YuvChromaSubsampling::Yuv444 => {
                ux += 64;
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
                let mut uv_values = vld2_u8(uv_ptr.add(ux));
                if order == YuvNVOrder::VU {
                    uv_values = uint8x8x2_t(uv_values.1, uv_values.0);
                }

                u_high_u8 = vzip2_u8(uv_values.0, uv_values.0);
                v_high_u8 = vzip2_u8(uv_values.1, uv_values.1);
                u_low_u8 = vzip1_u8(uv_values.0, uv_values.0);
                v_low_u8 = vzip1_u8(uv_values.1, uv_values.1);
            }
            YuvChromaSubsampling::Yuv444 => {
                let mut uv_values = vld2q_u8(uv_ptr.add(ux));
                if order == YuvNVOrder::VU {
                    uv_values = uint8x16x2_t(uv_values.1, uv_values.0);
                }
                u_high_u8 = vget_high_u8(uv_values.0);
                v_high_u8 = vget_high_u8(uv_values.1);
                u_low_u8 = vget_low_u8(uv_values.0);
                v_low_u8 = vget_low_u8(uv_values.1);
            }
        }

        let u_high = vsubq_s16(
            vreinterpretq_s16_u16(vshll_n_u8::<SCALE>(u_high_u8)),
            uv_corr,
        );
        let v_high = vsubq_s16(
            vreinterpretq_s16_u16(vshll_n_u8::<SCALE>(v_high_u8)),
            uv_corr,
        );
        let y_high = vqrdmulhq_laneq_s16::<0>(
            vreinterpretq_s16_u16(vexpand_high_8_to_10(y_values)),
            v_weights,
        );

        let r_high = vqmovun_s16(vqrdmlahq_laneq_s16::<1>(y_high, v_high, v_weights));
        let b_high = vqmovun_s16(vqrdmlahq_laneq_s16::<2>(y_high, u_high, v_weights));
        let g_high = vqmovun_s16(vqrdmlahq_laneq_s16::<4>(
            vqrdmlahq_laneq_s16::<3>(y_high, v_high, v_weights),
            u_high,
            v_weights,
        ));
        let u_low = vsubq_s16(
            vreinterpretq_s16_u16(vshll_n_u8::<SCALE>(u_low_u8)),
            uv_corr,
        );
        let v_low = vsubq_s16(
            vreinterpretq_s16_u16(vshll_n_u8::<SCALE>(v_low_u8)),
            uv_corr,
        );
        let y_v_shl = vexpand8_to_10(vget_low_u8(y_values));
        let y_low = vqrdmulhq_laneq_s16::<0>(vreinterpretq_s16_u16(y_v_shl), v_weights);

        let r_low = vqmovun_s16(vqrdmlahq_laneq_s16::<1>(y_low, v_low, v_weights));
        let b_low = vqmovun_s16(vqrdmlahq_laneq_s16::<2>(y_low, u_low, v_weights));
        let g_low = vqmovun_s16(vqrdmlahq_laneq_s16::<4>(
            vqrdmlahq_laneq_s16::<3>(y_low, v_low, v_weights),
            u_low,
            v_weights,
        ));

        let r_values = vcombine_u8(r_low, r_high);
        let g_values = vcombine_u8(g_low, g_high);
        let b_values = vcombine_u8(b_low, b_high);

        let dst_shift = cx * channels;

        neon_store_rgb8::<DESTINATION_CHANNELS>(
            bgra_ptr.add(dst_shift),
            r_values,
            g_values,
            b_values,
            v_alpha,
        );

        cx += 16;

        match chroma_subsampling {
            YuvChromaSubsampling::Yuv420 | YuvChromaSubsampling::Yuv422 => {
                ux += 16;
            }
            YuvChromaSubsampling::Yuv444 => {
                ux += 32;
            }
        }
    }

    let shuffle_u = vld1_u8([0, 0, 2, 2, 4, 4, 6, 6].as_ptr());
    let shuffle_v = vld1_u8([1, 1, 3, 3, 5, 5, 7, 7].as_ptr());

    while cx + 8 < width {
        let dst_shift = cx * channels;

        let y_values = vqsub_u8(
            vld1_u8(y_plane.get_unchecked(cx..).as_ptr()),
            vget_low_u8(y_corr),
        );

        let mut u_low_u8: uint8x8_t;
        let mut v_low_u8: uint8x8_t;

        match chroma_subsampling {
            YuvChromaSubsampling::Yuv420 | YuvChromaSubsampling::Yuv422 => {
                let uv_values = vld1_u8(uv_plane.get_unchecked(ux..).as_ptr());

                u_low_u8 = vtbl1_u8(uv_values, shuffle_u);
                v_low_u8 = vtbl1_u8(uv_values, shuffle_v);

                if order == YuvNVOrder::VU {
                    std::mem::swap(&mut u_low_u8, &mut v_low_u8);
                }
            }
            YuvChromaSubsampling::Yuv444 => {
                let mut uv_values = vld2_u8(uv_plane.get_unchecked(ux..).as_ptr());
                if order == YuvNVOrder::VU {
                    uv_values = uint8x8x2_t(uv_values.1, uv_values.0);
                }
                u_low_u8 = uv_values.0;
                v_low_u8 = uv_values.1;
            }
        }

        let u_low = vsubq_s16(
            vreinterpretq_s16_u16(vshll_n_u8::<SCALE>(u_low_u8)),
            uv_corr,
        );
        let v_low = vsubq_s16(
            vreinterpretq_s16_u16(vshll_n_u8::<SCALE>(v_low_u8)),
            uv_corr,
        );
        let y_low =
            vqrdmulhq_laneq_s16::<0>(vreinterpretq_s16_u16(vexpand8_to_10(y_values)), v_weights);

        let r_low = vqmovun_s16(vqrdmlahq_laneq_s16::<1>(y_low, v_low, v_weights));
        let b_low = vqmovun_s16(vqrdmlahq_laneq_s16::<2>(y_low, u_low, v_weights));
        let g_low = vqmovun_s16(vqrdmlahq_laneq_s16::<4>(
            vqrdmlahq_laneq_s16::<3>(y_low, v_low, v_weights),
            u_low,
            v_weights,
        ));

        let r_values = r_low;
        let g_values = g_low;
        let b_values = b_low;

        neon_store_half_rgb8::<DESTINATION_CHANNELS>(
            rgba.get_unchecked_mut(dst_shift..).as_mut_ptr(),
            r_values,
            g_values,
            b_values,
            vget_low_u8(v_alpha),
        );

        cx += 8;

        match chroma_subsampling {
            YuvChromaSubsampling::Yuv420 | YuvChromaSubsampling::Yuv422 => {
                ux += 8;
            }
            YuvChromaSubsampling::Yuv444 => {
                ux += 16;
            }
        }
    }

    if cx < width {
        let diff = width - cx;

        assert!(diff <= 8);

        let mut dst_buffer: [u8; 8 * 4] = [0; 8 * 4];
        let mut y_buffer: [u8; 8] = [0; 8];
        let mut uv_buffer: [u8; 8 * 2] = [0; 8 * 2];

        std::ptr::copy_nonoverlapping(
            y_plane.get_unchecked(cx..).as_ptr(),
            y_buffer.as_mut_ptr(),
            diff,
        );

        let ux_size = match chroma_subsampling {
            YuvChromaSubsampling::Yuv420 | YuvChromaSubsampling::Yuv422 => diff.div_ceil(2) * 2,
            YuvChromaSubsampling::Yuv444 => diff * 2,
        };

        std::ptr::copy_nonoverlapping(
            uv_plane.get_unchecked(ux..).as_ptr(),
            uv_buffer.as_mut_ptr(),
            ux_size,
        );

        let y_values = vqsub_u8(vld1_u8(y_buffer.as_ptr()), vget_low_u8(y_corr));

        let mut u_low_u8: uint8x8_t;
        let mut v_low_u8: uint8x8_t;

        match chroma_subsampling {
            YuvChromaSubsampling::Yuv420 | YuvChromaSubsampling::Yuv422 => {
                let uv_values = vld1_u8(uv_buffer.as_ptr());

                u_low_u8 = vtbl1_u8(uv_values, shuffle_u);
                v_low_u8 = vtbl1_u8(uv_values, shuffle_v);

                if order == YuvNVOrder::VU {
                    std::mem::swap(&mut u_low_u8, &mut v_low_u8);
                }
            }
            YuvChromaSubsampling::Yuv444 => {
                let mut uv_values = vld2_u8(uv_buffer.as_ptr());
                if order == YuvNVOrder::VU {
                    uv_values = uint8x8x2_t(uv_values.1, uv_values.0);
                }
                u_low_u8 = uv_values.0;
                v_low_u8 = uv_values.1;
            }
        }

        let u_low = vsubq_s16(
            vreinterpretq_s16_u16(vshll_n_u8::<SCALE>(u_low_u8)),
            uv_corr,
        );
        let v_low = vsubq_s16(
            vreinterpretq_s16_u16(vshll_n_u8::<SCALE>(v_low_u8)),
            uv_corr,
        );
        let y_low =
            vqrdmulhq_laneq_s16::<0>(vreinterpretq_s16_u16(vexpand8_to_10(y_values)), v_weights);

        let r_low = vqmovun_s16(vqrdmlahq_laneq_s16::<1>(y_low, v_low, v_weights));
        let b_low = vqmovun_s16(vqrdmlahq_laneq_s16::<2>(y_low, u_low, v_weights));
        let g_low = vqmovun_s16(vqrdmlahq_laneq_s16::<4>(
            vqrdmlahq_laneq_s16::<3>(y_low, v_low, v_weights),
            u_low,
            v_weights,
        ));

        let r_values = r_low;
        let g_values = g_low;
        let b_values = b_low;

        neon_store_half_rgb8::<DESTINATION_CHANNELS>(
            dst_buffer.as_mut_ptr(),
            r_values,
            g_values,
            b_values,
            vget_low_u8(v_alpha),
        );

        let dst_shift = cx * channels;

        std::ptr::copy_nonoverlapping(
            dst_buffer.as_mut_ptr(),
            rgba.get_unchecked_mut(dst_shift..).as_mut_ptr(),
            diff * channels,
        );

        cx += diff;
        ux += ux_size;
    }

    ProcessedOffset { cx, ux }
}

pub(crate) unsafe fn neon_yuv_nv_to_rgba_row<
    const PRECISION: i32,
    const UV_ORDER: u8,
    const DESTINATION_CHANNELS: u8,
    const YUV_CHROMA_SAMPLING: u8,
>(
    range: &YuvChromaRange,
    transform: &CbCrInverseTransform<i32>,
    y_plane: &[u8],
    uv_plane: &[u8],
    rgba: &mut [u8],
    start_cx: usize,
    start_ux: usize,
    width: usize,
) -> ProcessedOffset {
    let order: YuvNVOrder = UV_ORDER.into();
    let destination_channels: YuvSourceChannels = DESTINATION_CHANNELS.into();
    let chroma_subsampling: YuvChromaSubsampling = YUV_CHROMA_SAMPLING.into();
    let channels = destination_channels.get_channels_count();

    let y_ptr = y_plane.as_ptr();
    let uv_ptr = uv_plane.as_ptr();
    let bgra_ptr = rgba.as_mut_ptr();

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
        let y_values = vqsubq_u8(vld1q_u8(y_ptr.add(cx)), y_corr);

        let u_high_u8: uint8x8_t;
        let v_high_u8: uint8x8_t;
        let u_low_u8: uint8x8_t;
        let v_low_u8: uint8x8_t;

        match chroma_subsampling {
            YuvChromaSubsampling::Yuv420 | YuvChromaSubsampling::Yuv422 => {
                let mut uv_values = vld2_u8(uv_ptr.add(ux));
                if order == YuvNVOrder::VU {
                    uv_values = uint8x8x2_t(uv_values.1, uv_values.0);
                }

                u_high_u8 = vzip2_u8(uv_values.0, uv_values.0);
                v_high_u8 = vzip2_u8(uv_values.1, uv_values.1);
                u_low_u8 = vzip1_u8(uv_values.0, uv_values.0);
                v_low_u8 = vzip1_u8(uv_values.1, uv_values.1);
            }
            YuvChromaSubsampling::Yuv444 => {
                let mut uv_values = vld2q_u8(uv_ptr.add(ux));
                if order == YuvNVOrder::VU {
                    uv_values = uint8x16x2_t(uv_values.1, uv_values.0);
                }
                u_high_u8 = vget_high_u8(uv_values.0);
                v_high_u8 = vget_high_u8(uv_values.1);
                u_low_u8 = vget_low_u8(uv_values.0);
                v_low_u8 = vget_low_u8(uv_values.1);
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

        let r_values = vcombine_u8(vqmovun_s16(r_low), vqmovun_s16(r_high));
        let g_values = vcombine_u8(vqmovun_s16(g_low), vqmovun_s16(g_high));
        let b_values = vcombine_u8(vqmovun_s16(b_low), vqmovun_s16(b_high));

        let dst_shift = cx * channels;

        neon_store_rgb8::<DESTINATION_CHANNELS>(
            bgra_ptr.add(dst_shift),
            r_values,
            g_values,
            b_values,
            v_alpha,
        );

        cx += 16;

        match chroma_subsampling {
            YuvChromaSubsampling::Yuv420 | YuvChromaSubsampling::Yuv422 => {
                ux += 16;
            }
            YuvChromaSubsampling::Yuv444 => {
                ux += 32;
            }
        }
    }

    let shuffle_u = vld1_u8([0, 0, 2, 2, 4, 4, 6, 6].as_ptr());
    let shuffle_v = vld1_u8([1, 1, 3, 3, 5, 5, 7, 7].as_ptr());

    let decode_8_part = |dst: &mut [u8], y_src: &[u8], uv_src: &[u8]| {
        let y_values = vqsub_u8(vld1_u8(y_src.as_ptr()), vget_low_u8(y_corr));

        let mut u_low_u8: uint8x8_t;
        let mut v_low_u8: uint8x8_t;

        match chroma_subsampling {
            YuvChromaSubsampling::Yuv420 | YuvChromaSubsampling::Yuv422 => {
                let uv_values = vld1_u8(uv_src.as_ptr());

                u_low_u8 = vtbl1_u8(uv_values, shuffle_u);
                v_low_u8 = vtbl1_u8(uv_values, shuffle_v);

                #[allow(clippy::manual_swap)]
                if order == YuvNVOrder::VU {
                    let new_v = u_low_u8;
                    u_low_u8 = v_low_u8;
                    v_low_u8 = new_v;
                }
            }
            YuvChromaSubsampling::Yuv444 => {
                let mut uv_values = vld2_u8(uv_src.as_ptr());
                if order == YuvNVOrder::VU {
                    uv_values = uint8x8x2_t(uv_values.1, uv_values.0);
                }
                u_low_u8 = uv_values.0;
                v_low_u8 = uv_values.1;
            }
        }

        let u_low = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(u_low_u8)), uv_corr);
        let v_low = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(v_low_u8)), uv_corr);

        let y_low = vmullq_laneq_s16::<0>(vreinterpretq_s16_u16(vmovl_u8(y_values)), v_weights);

        let r_low = vdotl_laneq_s16::<PRECISION, 1>(y_low, v_low, v_weights);
        let b_low = vdotl_laneq_s16::<PRECISION, 2>(y_low, u_low, v_weights);
        let g_low = vdotl_laneq_s16_x2::<PRECISION, 3, 4>(y_low, v_low, u_low, v_weights);

        let r_values = vqmovun_s16(r_low);
        let g_values = vqmovun_s16(g_low);
        let b_values = vqmovun_s16(b_low);

        neon_store_half_rgb8::<DESTINATION_CHANNELS>(
            dst.as_mut_ptr(),
            r_values,
            g_values,
            b_values,
            vget_low_u8(v_alpha),
        );
    };

    while cx + 8 < width {
        let dst_shift = cx * channels;
        decode_8_part(
            rgba.get_unchecked_mut(dst_shift..),
            y_plane.get_unchecked(cx..),
            uv_plane.get_unchecked(ux..),
        );

        cx += 8;

        match chroma_subsampling {
            YuvChromaSubsampling::Yuv420 | YuvChromaSubsampling::Yuv422 => {
                ux += 8;
            }
            YuvChromaSubsampling::Yuv444 => {
                ux += 16;
            }
        }
    }

    if cx < width {
        let diff = width - cx;

        assert!(diff <= 8);

        let mut dst_buffer: [u8; 8 * 4] = [0; 8 * 4];
        let mut y_buffer: [u8; 8] = [0; 8];
        let mut uv_buffer: [u8; 8 * 2] = [0; 8 * 2];

        std::ptr::copy_nonoverlapping(
            y_plane.get_unchecked(cx..).as_ptr(),
            y_buffer.as_mut_ptr(),
            diff,
        );

        let ux_size = match chroma_subsampling {
            YuvChromaSubsampling::Yuv420 | YuvChromaSubsampling::Yuv422 => diff.div_ceil(2) * 2,
            YuvChromaSubsampling::Yuv444 => diff * 2,
        };

        std::ptr::copy_nonoverlapping(
            uv_plane.get_unchecked(ux..).as_ptr(),
            uv_buffer.as_mut_ptr(),
            ux_size,
        );

        decode_8_part(
            dst_buffer.as_mut_slice(),
            y_buffer.as_slice(),
            uv_buffer.as_slice(),
        );

        let dst_shift = cx * channels;
        std::ptr::copy_nonoverlapping(
            dst_buffer.as_mut_ptr(),
            rgba.get_unchecked_mut(dst_shift..).as_mut_ptr(),
            diff * channels,
        );

        cx += diff;
        ux += ux_size;
    }

    ProcessedOffset { cx, ux }
}
