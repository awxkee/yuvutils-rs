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
use crate::internals::ProcessedOffset;
use crate::neon::utils::*;
use crate::yuv_support::{
    CbCrForwardTransform, YuvChromaRange, YuvChromaSubsampling, YuvSourceChannels,
};
use std::arch::aarch64::*;

#[inline(always)]
unsafe fn vmlal_laneq_dot3<const L0: i32, const L1: i32, const L2: i32>(
    bias: int32x4_t,
    r: int16x4_t,
    g: int16x4_t,
    b: int16x4_t,
    w_a: int16x8_t,
    w_b: int16x8_t,
) -> int32x4_t {
    let mut acc = vmlal_laneq_s16::<L0>(bias, r, w_a);
    acc = vmlal_laneq_s16::<L1>(acc, g, w_a);
    acc = vmlal_laneq_s16::<L1>(acc, g, w_b);
    vmlal_laneq_s16::<L2>(acc, b, w_a)
}

#[inline(always)]
unsafe fn vmlal_high_laneq_dot3<const L0: i32, const L1: i32, const L2: i32>(
    bias: int32x4_t,
    r: int16x8_t,
    g: int16x8_t,
    b: int16x8_t,
    w_a: int16x8_t,
    w_b: int16x8_t,
) -> int32x4_t {
    let mut acc = vmlal_high_laneq_s16::<L0>(bias, r, w_a);
    acc = vmlal_high_laneq_s16::<L1>(acc, g, w_a);
    acc = vmlal_high_laneq_s16::<L1>(acc, g, w_b);
    vmlal_high_laneq_s16::<L2>(acc, b, w_a)
}

#[inline(always)]
unsafe fn encode_16_part_prof<const ORIGIN_CHANNELS: u8, const SAMPLING: u8>(
    src: &[u8],
    y_dst: &mut [u8],
    u_dst: &mut [u8],
    v_dst: &mut [u8],
    y_bias: int32x4_t,
    uv_bias: int32x4_t,
    v_weights_a: int16x8_t,
    v_weights_b: int16x8_t,
    v_cr_b: int16x8_t,
) {
    const PRECISION: i32 = 16;
    let chroma_subsampling: YuvChromaSubsampling = SAMPLING.into();
    let (r_values_u8, g_values_u8, b_values_u8) =
        neon_vld_rgb_for_yuv::<ORIGIN_CHANNELS>(src.as_ptr());

    let r_high = vreinterpretq_s16_u16(vmovl_high_u8(r_values_u8));
    let g_high = vreinterpretq_s16_u16(vmovl_high_u8(g_values_u8));
    let b_high = vreinterpretq_s16_u16(vmovl_high_u8(b_values_u8));

    let r_h_low = vget_low_s16(r_high);
    let g_h_low = vget_low_s16(g_high);
    let b_h_low = vget_low_s16(b_high);

    let y_h_high =
        vmlal_high_laneq_dot3::<0, 1, 2>(y_bias, r_high, g_high, b_high, v_weights_a, v_weights_b);
    let y_h_low =
        vmlal_laneq_dot3::<0, 1, 2>(y_bias, r_h_low, g_h_low, b_h_low, v_weights_a, v_weights_b);
    let y_high = vreinterpretq_u16_s16(vcombine_s16(
        vqshrn_n_s32::<PRECISION>(y_h_low),
        vqshrn_n_s32::<PRECISION>(y_h_high),
    ));

    let r_low = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(r_values_u8)));
    let g_low = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(g_values_u8)));
    let b_low = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(b_values_u8)));

    let r_l_low = vget_low_s16(r_low);
    let g_l_low = vget_low_s16(g_low);
    let b_l_low = vget_low_s16(b_low);

    let y_l_high =
        vmlal_high_laneq_dot3::<0, 1, 2>(y_bias, r_low, g_low, b_low, v_weights_a, v_weights_b);
    let y_l_low =
        vmlal_laneq_dot3::<0, 1, 2>(y_bias, r_l_low, g_l_low, b_l_low, v_weights_a, v_weights_b);
    let y_low = vreinterpretq_u16_s16(vcombine_s16(
        vqshrn_n_s32::<PRECISION>(y_l_low),
        vqshrn_n_s32::<PRECISION>(y_l_high),
    ));

    if chroma_subsampling == YuvChromaSubsampling::Yuv444 {
        let mut cb_h_high = vmlal_high_laneq_s16::<3>(uv_bias, r_high, v_weights_a);
        let mut cb_h_low = vmlal_laneq_s16::<3>(uv_bias, r_h_low, v_weights_a);
        let mut cr_h_high = vmlal_high_laneq_s16::<6>(uv_bias, r_high, v_weights_a);
        cr_h_high = vmlal_high_laneq_s16::<6>(cr_h_high, r_high, v_weights_b);
        let mut cr_h_low = vmlal_laneq_s16::<6>(uv_bias, r_h_low, v_weights_a);
        cr_h_low = vmlal_laneq_s16::<6>(cr_h_low, r_h_low, v_weights_b);
        let mut cb_l_high = vmlal_high_laneq_s16::<3>(uv_bias, r_low, v_weights_a);
        let mut cb_l_low = vmlal_laneq_s16::<3>(uv_bias, r_l_low, v_weights_a);
        let mut cr_l_high = vmlal_high_laneq_s16::<6>(uv_bias, r_low, v_weights_a);
        cr_l_high = vmlal_high_laneq_s16::<6>(cr_l_high, r_low, v_weights_b);
        let mut cr_l_low = vmlal_laneq_s16::<6>(uv_bias, r_l_low, v_weights_a);
        cr_l_low = vmlal_laneq_s16::<6>(cr_l_low, r_l_low, v_weights_b);

        cb_h_high = vmlal_high_laneq_s16::<4>(cb_h_high, g_high, v_weights_a);
        cb_h_low = vmlal_laneq_s16::<4>(cb_h_low, g_h_low, v_weights_a);
        cr_h_high = vmlal_high_laneq_s16::<7>(cr_h_high, g_high, v_weights_a);
        cr_h_low = vmlal_laneq_s16::<7>(cr_h_low, g_h_low, v_weights_a);
        cb_l_high = vmlal_high_laneq_s16::<4>(cb_l_high, g_low, v_weights_a);
        cb_l_low = vmlal_laneq_s16::<4>(cb_l_low, g_l_low, v_weights_a);
        cr_l_high = vmlal_high_laneq_s16::<7>(cr_l_high, g_low, v_weights_a);
        cr_l_low = vmlal_laneq_s16::<7>(cr_l_low, g_l_low, v_weights_a);

        cb_h_high = vmlal_high_laneq_s16::<5>(cb_h_high, b_high, v_weights_a);
        cb_h_high = vmlal_high_laneq_s16::<5>(cb_h_high, b_high, v_weights_b);
        cb_h_low = vmlal_laneq_s16::<5>(cb_h_low, b_h_low, v_weights_a);
        cb_h_low = vmlal_laneq_s16::<5>(cb_h_low, b_h_low, v_weights_b);
        cr_h_high = vmlal_high_laneq_s16::<0>(cr_h_high, b_high, v_cr_b);
        cr_h_low = vmlal_laneq_s16::<0>(cr_h_low, b_h_low, v_cr_b);
        cb_l_high = vmlal_high_laneq_s16::<5>(cb_l_high, b_low, v_weights_a);
        cb_l_high = vmlal_high_laneq_s16::<5>(cb_l_high, b_low, v_weights_b);
        cb_l_low = vmlal_laneq_s16::<5>(cb_l_low, b_l_low, v_weights_a);
        cb_l_low = vmlal_laneq_s16::<5>(cb_l_low, b_l_low, v_weights_b);
        cr_l_high = vmlal_high_laneq_s16::<0>(cr_l_high, b_low, v_cr_b);
        cr_l_low = vmlal_laneq_s16::<0>(cr_l_low, b_l_low, v_cr_b);

        let cb_high = vreinterpretq_u16_s16(vcombine_s16(
            vshrn_n_s32::<PRECISION>(cb_h_low),
            vshrn_n_s32::<PRECISION>(cb_h_high),
        ));
        let cr_high = vreinterpretq_u16_s16(vcombine_s16(
            vshrn_n_s32::<PRECISION>(cr_h_low),
            vshrn_n_s32::<PRECISION>(cr_h_high),
        ));
        let cb_low = vreinterpretq_u16_s16(vcombine_s16(
            vshrn_n_s32::<PRECISION>(cb_l_low),
            vshrn_n_s32::<PRECISION>(cb_l_high),
        ));
        let cr_low = vreinterpretq_u16_s16(vcombine_s16(
            vshrn_n_s32::<PRECISION>(cr_l_low),
            vshrn_n_s32::<PRECISION>(cr_l_high),
        ));
        let cb = vcombine_u8(vmovn_u16(cb_low), vmovn_u16(cb_high));
        let cr = vcombine_u8(vmovn_u16(cr_low), vmovn_u16(cr_high));
        vst1q_u8(u_dst.as_mut_ptr(), cb);
        vst1q_u8(v_dst.as_mut_ptr(), cr);
    } else if chroma_subsampling == YuvChromaSubsampling::Yuv422
        || (chroma_subsampling == YuvChromaSubsampling::Yuv420)
    {
        let rlv = vpaddlq_u8(r_values_u8);
        let glv = vpaddlq_u8(g_values_u8);
        let blv = vpaddlq_u8(b_values_u8);
        let r1 = vreinterpretq_s16_u16(vrshrq_n_u16::<1>(rlv));
        let g1 = vreinterpretq_s16_u16(vrshrq_n_u16::<1>(glv));
        let b1 = vreinterpretq_s16_u16(vrshrq_n_u16::<1>(blv));

        let mut cb_h = vmlal_high_laneq_s16::<3>(uv_bias, r1, v_weights_a);
        let mut cb_l = vmlal_laneq_s16::<3>(uv_bias, vget_low_s16(r1), v_weights_a);
        let mut cr_h = vmlal_high_laneq_s16::<6>(uv_bias, r1, v_weights_a);
        cr_h = vmlal_high_laneq_s16::<6>(cr_h, r1, v_weights_b);
        let mut cr_l = vmlal_laneq_s16::<6>(uv_bias, vget_low_s16(r1), v_weights_a);
        cr_l = vmlal_laneq_s16::<6>(cr_l, vget_low_s16(r1), v_weights_b);

        cb_h = vmlal_high_laneq_s16::<4>(cb_h, g1, v_weights_a);
        cb_l = vmlal_laneq_s16::<4>(cb_l, vget_low_s16(g1), v_weights_a);
        cr_h = vmlal_high_laneq_s16::<7>(cr_h, g1, v_weights_a);
        cr_l = vmlal_laneq_s16::<7>(cr_l, vget_low_s16(g1), v_weights_a);

        cb_h = vmlal_high_laneq_s16::<5>(cb_h, b1, v_weights_a);
        cb_h = vmlal_high_laneq_s16::<5>(cb_h, b1, v_weights_b);
        cb_l = vmlal_laneq_s16::<5>(cb_l, vget_low_s16(b1), v_weights_a);
        cb_l = vmlal_laneq_s16::<5>(cb_l, vget_low_s16(b1), v_weights_b);
        cr_h = vmlal_high_laneq_s16::<0>(cr_h, b1, v_cr_b);
        cr_l = vmlal_laneq_s16::<0>(cr_l, vget_low_s16(b1), v_cr_b);

        let cb_l = vqshrn_n_s32::<PRECISION>(cb_l);
        let cb_h = vqshrn_n_s32::<PRECISION>(cb_h);
        let cr_l = vqshrn_n_s32::<PRECISION>(cr_l);
        let cr_h = vqshrn_n_s32::<PRECISION>(cr_h);

        let cb = vmovn_u16(vreinterpretq_u16_s16(vcombine_s16(cb_l, cb_h)));
        let cr = vmovn_u16(vreinterpretq_u16_s16(vcombine_s16(cr_l, cr_h)));

        vst1_u8(u_dst.as_mut_ptr(), cb);
        vst1_u8(v_dst.as_mut_ptr(), cr);
    }

    let y = vcombine_u8(vmovn_u16(y_low), vmovn_u16(y_high));
    vst1q_u8(y_dst.as_mut_ptr(), y);
}

#[inline(always)]
pub(crate) unsafe fn neon_rgba_to_yuv_prof<const ORIGIN_CHANNELS: u8, const SAMPLING: u8>(
    transform: &CbCrForwardTransform<i32>,
    range: &YuvChromaRange,
    y_plane: &mut [u8],
    u_plane: &mut [u8],
    v_plane: &mut [u8],
    rgba: &[u8],
    start_cx: usize,
    start_ux: usize,
    width: usize,
) -> ProcessedOffset {
    let chroma_subsampling: YuvChromaSubsampling = SAMPLING.into();
    let source_channels: YuvSourceChannels = ORIGIN_CHANNELS.into();
    let channels = source_channels.get_channels_count();

    const PRECISION: i32 = 16;

    let rounding_const_bias: i32 = (1 << (PRECISION - 1)) - 1;
    let bias_y = range.bias_y as i32 * (1 << PRECISION) + rounding_const_bias;
    let bias_uv = range.bias_uv as i32 * (1 << PRECISION) + rounding_const_bias;

    let y_bias = vdupq_n_s32(bias_y);
    let uv_bias = vdupq_n_s32(bias_uv);

    let yg_a = (transform.yg / 2) as i16;
    let yg_b = (transform.yg - transform.yg / 2) as i16;
    let cb_b_a = (transform.cb_b / 2) as i16;
    let cb_b_b = (transform.cb_b - transform.cb_b / 2) as i16;
    let cr_r_a = (transform.cr_r / 2) as i16;
    let cr_r_b = (transform.cr_r - transform.cr_r / 2) as i16;

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

    let weights_b_arr: [i16; 8] = [0, yg_b, 0, 0, 0, cb_b_b, cr_r_b, 0];
    let v_weights_b = vld1q_s16(weights_b_arr.as_ptr());

    let v_cr_b = vdupq_n_s16(transform.cr_b as i16);

    let mut cx = start_cx;
    let mut ux = start_ux;

    while cx + 16 <= width {
        encode_16_part_prof::<ORIGIN_CHANNELS, SAMPLING>(
            rgba.get_unchecked(cx * channels..),
            y_plane.get_unchecked_mut(cx..),
            u_plane.get_unchecked_mut(ux..),
            v_plane.get_unchecked_mut(ux..),
            y_bias,
            uv_bias,
            v_weights_a,
            v_weights_b,
            v_cr_b,
        );

        if chroma_subsampling == YuvChromaSubsampling::Yuv444 {
            ux += 16;
        } else if chroma_subsampling == YuvChromaSubsampling::Yuv422
            || (chroma_subsampling == YuvChromaSubsampling::Yuv420)
        {
            ux += 8;
        }

        cx += 16;
    }

    if cx < width {
        let diff = width - cx;
        assert!(diff <= 16);
        let mut src_buffer: [u8; 16 * 4] = [0; 16 * 4];
        let mut y_buffer: [u8; 16] = [0; 16];
        let mut u_buffer: [u8; 16] = [0; 16];
        let mut v_buffer: [u8; 16] = [0; 16];

        if chroma_subsampling != YuvChromaSubsampling::Yuv444 && diff % 2 != 0 {
            let lst = (width - 1) * channels;
            let last_items = rgba.get_unchecked(lst..(lst + channels));
            let dvb = diff * channels;
            let dst = src_buffer.get_unchecked_mut(dvb..(dvb + channels));
            for (dst, src) in dst.iter_mut().zip(last_items) {
                *dst = *src;
            }
        }

        std::ptr::copy_nonoverlapping(
            rgba.get_unchecked(cx * channels..).as_ptr(),
            src_buffer.as_mut_ptr().cast(),
            diff * channels,
        );

        encode_16_part_prof::<ORIGIN_CHANNELS, SAMPLING>(
            src_buffer.as_slice(),
            y_buffer.as_mut_slice(),
            u_buffer.as_mut_slice(),
            v_buffer.as_mut_slice(),
            y_bias,
            uv_bias,
            v_weights_a,
            v_weights_b,
            v_cr_b,
        );

        std::ptr::copy_nonoverlapping(
            y_buffer.as_ptr().cast(),
            y_plane.get_unchecked_mut(cx..).as_mut_ptr(),
            diff,
        );

        cx += diff;
        if chroma_subsampling == YuvChromaSubsampling::Yuv444 {
            std::ptr::copy_nonoverlapping(
                u_buffer.as_ptr().cast(),
                u_plane.get_unchecked_mut(ux..).as_mut_ptr(),
                diff,
            );
            std::ptr::copy_nonoverlapping(
                v_buffer.as_ptr().cast(),
                v_plane.get_unchecked_mut(ux..).as_mut_ptr(),
                diff,
            );

            ux += diff;
        } else if (chroma_subsampling == YuvChromaSubsampling::Yuv420)
            || (chroma_subsampling == YuvChromaSubsampling::Yuv422)
        {
            let hv = diff.div_ceil(2);
            std::ptr::copy_nonoverlapping(
                u_buffer.as_ptr().cast(),
                u_plane.get_unchecked_mut(ux..).as_mut_ptr(),
                hv,
            );
            std::ptr::copy_nonoverlapping(
                v_buffer.as_ptr().cast(),
                v_plane.get_unchecked_mut(ux..).as_mut_ptr(),
                hv,
            );

            ux += hv;
        }
    }

    ProcessedOffset { cx, ux }
}
