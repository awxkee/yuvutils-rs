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
use crate::yuv_support::{
    CbCrForwardTransform, YuvChromaRange, YuvChromaSubsampling, YuvSourceChannels,
};
use std::arch::aarch64::*;

#[target_feature(enable = "rdm")]
pub(crate) unsafe fn neon_rgba_to_yuv_rdm<
    const ORIGIN_CHANNELS: u8,
    const SAMPLING: u8,
    const PRECISION: i32,
>(
    transform: &CbCrForwardTransform<i32>,
    range: &YuvChromaRange,
    y_plane: &mut [u8],
    u_plane: &mut [u8],
    v_plane: &mut [u8],
    rgba: &[u8],
    start_cx: usize,
    start_ux: usize,
    width: usize,
    compute_uv_row: bool,
) -> ProcessedOffset {
    let chroma_subsampling: YuvChromaSubsampling = SAMPLING.into();
    let source_channels: YuvSourceChannels = ORIGIN_CHANNELS.into();
    let channels = source_channels.get_channels_count();

    const V_SHR: i32 = 4;
    const V_SCALE: i32 = 7;
    let rounding_const_bias: i16 = 1 << (V_SHR - 1);
    let bias_y = range.bias_y as i16 * (1 << V_SHR) + rounding_const_bias;
    let bias_uv = range.bias_uv as i16 * (1 << V_SHR) + rounding_const_bias;

    let y_ptr = y_plane;
    let u_ptr = u_plane;
    let v_ptr = v_plane;
    let rgba_ptr = rgba.as_ptr();

    let i_bias_y = vdupq_n_s16(range.bias_y as i16);
    let i_cap_y = vdupq_n_u16(range.range_y as u16 + range.bias_y as u16);
    let i_cap_uv = vdupq_n_u16(range.bias_y as u16 + range.range_uv as u16);

    let y_bias = vdupq_n_s16(bias_y);
    let uv_bias = vdupq_n_s16(bias_uv);
    let v_yr = vdupq_n_s16(transform.yr as i16);
    let v_yg = vdupq_n_s16(transform.yg as i16);
    let v_yb = vdupq_n_s16(transform.yb as i16);
    let v_cb_r = vdupq_n_s16(transform.cb_r as i16);
    let v_cb_g = vdupq_n_s16(transform.cb_g as i16);
    let v_cb_b = vdupq_n_s16(transform.cb_b as i16);
    let v_cr_r = vdupq_n_s16(transform.cr_r as i16);
    let v_cr_g = vdupq_n_s16(transform.cr_g as i16);
    let v_cr_b = vdupq_n_s16(transform.cr_b as i16);

    let mut cx = start_cx;
    let mut ux = start_ux;

    while cx + 16 < width {
        let r_values_u8: uint8x16_t;
        let g_values_u8: uint8x16_t;
        let b_values_u8: uint8x16_t;

        match source_channels {
            YuvSourceChannels::Rgb | YuvSourceChannels::Bgr => {
                let rgb_values = vld3q_u8(rgba_ptr.add(cx * channels));
                if source_channels == YuvSourceChannels::Rgb {
                    r_values_u8 = rgb_values.0;
                    g_values_u8 = rgb_values.1;
                    b_values_u8 = rgb_values.2;
                } else {
                    r_values_u8 = rgb_values.2;
                    g_values_u8 = rgb_values.1;
                    b_values_u8 = rgb_values.0;
                }
            }
            YuvSourceChannels::Rgba => {
                let rgb_values = vld4q_u8(rgba_ptr.add(cx * channels));
                r_values_u8 = rgb_values.0;
                g_values_u8 = rgb_values.1;
                b_values_u8 = rgb_values.2;
            }
            YuvSourceChannels::Bgra => {
                let rgb_values = vld4q_u8(rgba_ptr.add(cx * channels));
                r_values_u8 = rgb_values.2;
                g_values_u8 = rgb_values.1;
                b_values_u8 = rgb_values.0;
            }
        }

        let r0hi = vreinterpretq_s16_u16(vshll_high_n_u8::<V_SCALE>(r_values_u8));
        let g0hi = vreinterpretq_s16_u16(vshll_high_n_u8::<V_SCALE>(g_values_u8));
        let b0hi = vreinterpretq_s16_u16(vshll_high_n_u8::<V_SCALE>(b_values_u8));

        let mut y_high = vqrdmlahq_s16(y_bias, r0hi, v_yr);
        y_high = vqrdmlahq_s16(y_high, g0hi, v_yg);
        y_high = vqrdmlahq_s16(y_high, b0hi, v_yb);

        let y_high = vminq_u16(
            vreinterpretq_u16_s16(vmaxq_s16(vshrq_n_s16::<V_SHR>(y_high), i_bias_y)),
            i_cap_y,
        );

        let r0lo = vreinterpretq_s16_u16(vshll_n_u8::<V_SCALE>(vget_low_u8(r_values_u8)));
        let g0lo = vreinterpretq_s16_u16(vshll_n_u8::<V_SCALE>(vget_low_u8(g_values_u8)));
        let b0lo = vreinterpretq_s16_u16(vshll_n_u8::<V_SCALE>(vget_low_u8(b_values_u8)));

        let mut y_low = vqrdmlahq_s16(y_bias, r0lo, v_yr);
        y_low = vqrdmlahq_s16(y_low, g0lo, v_yg);
        y_low = vqrdmlahq_s16(y_low, b0lo, v_yb);

        let y_low = vminq_u16(
            vreinterpretq_u16_s16(vmaxq_s16(vshrq_n_s16::<V_SHR>(y_low), i_bias_y)),
            i_cap_y,
        );

        let y = vcombine_u8(vqmovn_u16(y_low), vqmovn_u16(y_high));
        vst1q_u8(y_ptr.get_unchecked_mut(cx..).as_mut_ptr(), y);

        if chroma_subsampling == YuvChromaSubsampling::Yuv444 {
            let mut cb_high = vqrdmlahq_s16(uv_bias, r0hi, v_cb_r);
            cb_high = vqrdmlahq_s16(cb_high, g0hi, v_cb_g);
            cb_high = vqrdmlahq_s16(cb_high, b0hi, v_cb_b);

            let cb_high = vminq_u16(
                vreinterpretq_u16_s16(vmaxq_s16(vshrq_n_s16::<V_SHR>(cb_high), i_bias_y)),
                i_cap_uv,
            );

            let mut cr_high = vqrdmlahq_s16(uv_bias, r0hi, v_cr_r);
            cr_high = vqrdmlahq_s16(cr_high, g0hi, v_cr_g);
            cr_high = vqrdmlahq_s16(cr_high, b0hi, v_cr_b);

            let cr_high = vminq_u16(
                vreinterpretq_u16_s16(vmaxq_s16(vshrq_n_s16::<V_SHR>(cr_high), i_bias_y)),
                i_cap_uv,
            );

            let mut cb_low = vqrdmlahq_s16(uv_bias, r0lo, v_cb_r);
            cb_low = vqrdmlahq_s16(cb_low, g0lo, v_cb_g);
            cb_low = vqrdmlahq_s16(cb_low, b0lo, v_cb_b);

            let cb_low = vminq_u16(
                vreinterpretq_u16_s16(vmaxq_s16(vshrq_n_s16::<V_SHR>(cb_low), i_bias_y)),
                i_cap_uv,
            );

            let mut cr_low = vqrdmlahq_s16(uv_bias, r0lo, v_cr_r);
            cr_low = vqrdmlahq_s16(cr_low, g0lo, v_cr_g);
            cr_low = vqrdmlahq_s16(cr_low, b0lo, v_cr_b);

            let cr_low = vminq_u16(
                vreinterpretq_u16_s16(vmaxq_s16(vshrq_n_s16::<V_SHR>(cr_low), i_bias_y)),
                i_cap_uv,
            );
            let cb = vcombine_u8(vqmovn_u16(cb_low), vqmovn_u16(cb_high));
            let cr = vcombine_u8(vqmovn_u16(cr_low), vqmovn_u16(cr_high));

            vst1q_u8(u_ptr.get_unchecked_mut(ux..).as_mut_ptr(), cb);
            vst1q_u8(v_ptr.get_unchecked_mut(ux..).as_mut_ptr(), cr);

            ux += 16;
        } else if (chroma_subsampling == YuvChromaSubsampling::Yuv420 && compute_uv_row)
            || (chroma_subsampling == YuvChromaSubsampling::Yuv422)
        {
            let r1 = vreinterpretq_s16_u16(vshlq_n_u16::<V_SCALE>(vrshrq_n_u16::<1>(vpaddlq_u8(
                r_values_u8,
            ))));
            let g1 = vreinterpretq_s16_u16(vshlq_n_u16::<V_SCALE>(vrshrq_n_u16::<1>(vpaddlq_u8(
                g_values_u8,
            ))));
            let b1 = vreinterpretq_s16_u16(vshlq_n_u16::<V_SCALE>(vrshrq_n_u16::<1>(vpaddlq_u8(
                b_values_u8,
            ))));

            let mut cbl = vqrdmlahq_s16(uv_bias, r1, v_cb_r);
            cbl = vqrdmlahq_s16(cbl, g1, v_cb_g);
            cbl = vqrdmlahq_s16(cbl, b1, v_cb_b);

            let cb = vqmovn_u16(vminq_u16(
                vreinterpretq_u16_s16(vmaxq_s16(vshrq_n_s16::<V_SHR>(cbl), i_bias_y)),
                i_cap_uv,
            ));

            let mut crl = vqrdmlahq_s16(uv_bias, r1, v_cr_r);
            crl = vqrdmlahq_s16(crl, g1, v_cr_g);
            crl = vqrdmlahq_s16(crl, b1, v_cr_b);

            let cr = vqmovn_u16(vminq_u16(
                vreinterpretq_u16_s16(vmaxq_s16(vshrq_n_s16::<V_SHR>(crl), i_bias_y)),
                i_cap_uv,
            ));

            vst1_u8(u_ptr.get_unchecked_mut(ux..).as_mut_ptr(), cb);
            vst1_u8(v_ptr.get_unchecked_mut(ux..).as_mut_ptr(), cr);

            ux += 8;
        }

        cx += 16;
    }

    ProcessedOffset { cx, ux }
}

#[inline(always)]
pub(crate) unsafe fn neon_rgba_to_yuv<
    const ORIGIN_CHANNELS: u8,
    const SAMPLING: u8,
    const PRECISION: i32,
>(
    transform: &CbCrForwardTransform<i32>,
    range: &YuvChromaRange,
    y_plane: &mut [u8],
    u_plane: &mut [u8],
    v_plane: &mut [u8],
    rgba: &[u8],
    start_cx: usize,
    start_ux: usize,
    width: usize,
    compute_uv_row: bool,
) -> ProcessedOffset {
    let chroma_subsampling: YuvChromaSubsampling = SAMPLING.into();
    let source_channels: YuvSourceChannels = ORIGIN_CHANNELS.into();
    let channels = source_channels.get_channels_count();

    let rounding_const_bias: i32 = 1 << (PRECISION - 1);
    let bias_y = range.bias_y as i32 * (1 << PRECISION) + rounding_const_bias;
    let bias_uv = range.bias_uv as i32 * (1 << PRECISION) + rounding_const_bias;

    let y_ptr = y_plane;
    let u_ptr = u_plane;
    let v_ptr = v_plane;
    let rgba_ptr = rgba.as_ptr();

    let i_bias_y = vdupq_n_s16(range.bias_y as i16);
    let i_cap_y = vdupq_n_u16(range.range_y as u16 + range.bias_y as u16);
    let i_cap_uv = vdupq_n_u16(range.bias_y as u16 + range.range_uv as u16);

    let y_bias = vdupq_n_s32(bias_y);
    let uv_bias = vdupq_n_s32(bias_uv);
    let v_yr = vdupq_n_s16(transform.yr as i16);
    let v_yg = vdupq_n_s16(transform.yg as i16);
    let v_yb = vdupq_n_s16(transform.yb as i16);
    let v_cb_r = vdupq_n_s16(transform.cb_r as i16);
    let v_cb_g = vdupq_n_s16(transform.cb_g as i16);
    let v_cb_b = vdupq_n_s16(transform.cb_b as i16);
    let v_cr_r = vdupq_n_s16(transform.cr_r as i16);
    let v_cr_g = vdupq_n_s16(transform.cr_g as i16);
    let v_cr_b = vdupq_n_s16(transform.cr_b as i16);

    let mut cx = start_cx;
    let mut ux = start_ux;

    let v_zeros = vdupq_n_s32(0i32);

    while cx + 16 < width {
        let r_values_u8: uint8x16_t;
        let g_values_u8: uint8x16_t;
        let b_values_u8: uint8x16_t;

        match source_channels {
            YuvSourceChannels::Rgb | YuvSourceChannels::Bgr => {
                let rgb_values = vld3q_u8(rgba_ptr.add(cx * channels));
                if source_channels == YuvSourceChannels::Rgb {
                    r_values_u8 = rgb_values.0;
                    g_values_u8 = rgb_values.1;
                    b_values_u8 = rgb_values.2;
                } else {
                    r_values_u8 = rgb_values.2;
                    g_values_u8 = rgb_values.1;
                    b_values_u8 = rgb_values.0;
                }
            }
            YuvSourceChannels::Rgba => {
                let rgb_values = vld4q_u8(rgba_ptr.add(cx * channels));
                r_values_u8 = rgb_values.0;
                g_values_u8 = rgb_values.1;
                b_values_u8 = rgb_values.2;
            }
            YuvSourceChannels::Bgra => {
                let rgb_values = vld4q_u8(rgba_ptr.add(cx * channels));
                r_values_u8 = rgb_values.2;
                g_values_u8 = rgb_values.1;
                b_values_u8 = rgb_values.0;
            }
        }

        let r_high = vreinterpretq_s16_u16(vmovl_high_u8(r_values_u8));
        let g_high = vreinterpretq_s16_u16(vmovl_high_u8(g_values_u8));
        let b_high = vreinterpretq_s16_u16(vmovl_high_u8(b_values_u8));

        let r_h_low = vget_low_s16(r_high);
        let g_h_low = vget_low_s16(g_high);
        let b_h_low = vget_low_s16(b_high);

        let mut y_h_high = vmlal_high_s16(y_bias, r_high, v_yr);
        y_h_high = vmlal_high_s16(y_h_high, g_high, v_yg);
        y_h_high = vmlal_high_s16(y_h_high, b_high, v_yb);
        y_h_high = vmaxq_s32(y_h_high, v_zeros);

        let mut y_h_low = vmlal_s16(y_bias, r_h_low, vget_low_s16(v_yr));
        y_h_low = vmlal_s16(y_h_low, g_h_low, vget_low_s16(v_yg));
        y_h_low = vmlal_s16(y_h_low, b_h_low, vget_low_s16(v_yb));
        y_h_low = vmaxq_s32(y_h_low, v_zeros);

        let y_high = vminq_u16(
            vreinterpretq_u16_s16(vmaxq_s16(
                vcombine_s16(
                    vshrn_n_s32::<PRECISION>(y_h_low),
                    vshrn_n_s32::<PRECISION>(y_h_high),
                ),
                i_bias_y,
            )),
            i_cap_y,
        );

        let r_low = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(r_values_u8)));
        let g_low = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(g_values_u8)));
        let b_low = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(b_values_u8)));

        let r_l_low = vget_low_s16(r_low);
        let g_l_low = vget_low_s16(g_low);
        let b_l_low = vget_low_s16(b_low);

        let mut y_l_high = vmlal_high_s16(y_bias, r_low, v_yr);
        y_l_high = vmlal_high_s16(y_l_high, g_low, v_yg);
        y_l_high = vmlal_high_s16(y_l_high, b_low, v_yb);
        y_l_high = vmaxq_s32(y_l_high, v_zeros);

        let mut y_l_low = vmlal_s16(y_bias, r_l_low, vget_low_s16(v_yr));
        y_l_low = vmlal_s16(y_l_low, g_l_low, vget_low_s16(v_yg));
        y_l_low = vmlal_s16(y_l_low, b_l_low, vget_low_s16(v_yb));
        y_l_low = vmaxq_s32(y_l_low, v_zeros);

        let y_low = vminq_u16(
            vreinterpretq_u16_s16(vmaxq_s16(
                vcombine_s16(
                    vshrn_n_s32::<PRECISION>(y_l_low),
                    vshrn_n_s32::<PRECISION>(y_l_high),
                ),
                i_bias_y,
            )),
            i_cap_y,
        );

        let y = vcombine_u8(vqmovn_u16(y_low), vqmovn_u16(y_high));
        vst1q_u8(y_ptr.get_unchecked_mut(cx..).as_mut_ptr(), y);

        if chroma_subsampling == YuvChromaSubsampling::Yuv444 {
            let mut cb_h_high = vmlal_high_s16(uv_bias, r_high, v_cb_r);
            cb_h_high = vmlal_high_s16(cb_h_high, g_high, v_cb_g);
            cb_h_high = vmlal_high_s16(cb_h_high, b_high, v_cb_b);

            let mut cb_h_low = vmlal_s16(uv_bias, r_h_low, vget_low_s16(v_cb_r));
            cb_h_low = vmlal_s16(cb_h_low, g_h_low, vget_low_s16(v_cb_g));
            cb_h_low = vmlal_s16(cb_h_low, b_h_low, vget_low_s16(v_cb_b));

            let cb_high = vminq_u16(
                vreinterpretq_u16_s16(vmaxq_s16(
                    vcombine_s16(
                        vshrn_n_s32::<PRECISION>(cb_h_low),
                        vshrn_n_s32::<PRECISION>(cb_h_high),
                    ),
                    i_bias_y,
                )),
                i_cap_uv,
            );

            let mut cr_h_high = vmlal_high_s16(uv_bias, r_high, v_cr_r);
            cr_h_high = vmlal_high_s16(cr_h_high, g_high, v_cr_g);
            cr_h_high = vmlal_high_s16(cr_h_high, b_high, v_cr_b);

            let mut cr_h_low = vmlal_s16(uv_bias, r_h_low, vget_low_s16(v_cr_r));
            cr_h_low = vmlal_s16(cr_h_low, g_h_low, vget_low_s16(v_cr_g));
            cr_h_low = vmlal_s16(cr_h_low, b_h_low, vget_low_s16(v_cr_b));

            let cr_high = vminq_u16(
                vreinterpretq_u16_s16(vmaxq_s16(
                    vcombine_s16(
                        vshrn_n_s32::<PRECISION>(cr_h_low),
                        vshrn_n_s32::<PRECISION>(cr_h_high),
                    ),
                    i_bias_y,
                )),
                i_cap_uv,
            );

            let mut cb_l_high = vmlal_high_s16(uv_bias, r_low, v_cb_r);
            cb_l_high = vmlal_high_s16(cb_l_high, g_low, v_cb_g);
            cb_l_high = vmlal_high_s16(cb_l_high, b_low, v_cb_b);

            let mut cb_l_low = vmlal_s16(uv_bias, r_l_low, vget_low_s16(v_cb_r));
            cb_l_low = vmlal_s16(cb_l_low, g_l_low, vget_low_s16(v_cb_g));
            cb_l_low = vmlal_s16(cb_l_low, b_l_low, vget_low_s16(v_cb_b));

            let cb_low = vminq_u16(
                vreinterpretq_u16_s16(vmaxq_s16(
                    vcombine_s16(
                        vshrn_n_s32::<PRECISION>(cb_l_low),
                        vshrn_n_s32::<PRECISION>(cb_l_high),
                    ),
                    i_bias_y,
                )),
                i_cap_uv,
            );

            let mut cr_l_high = vmlal_high_s16(uv_bias, r_low, v_cr_r);
            cr_l_high = vmlal_high_s16(cr_l_high, g_low, v_cr_g);
            cr_l_high = vmlal_high_s16(cr_l_high, b_low, v_cr_b);

            let mut cr_l_low = vmlal_s16(uv_bias, r_l_low, vget_low_s16(v_cr_r));
            cr_l_low = vmlal_s16(cr_l_low, g_l_low, vget_low_s16(v_cr_g));
            cr_l_low = vmlal_s16(cr_l_low, b_l_low, vget_low_s16(v_cr_b));

            let cr_low = vminq_u16(
                vreinterpretq_u16_s16(vmaxq_s16(
                    vcombine_s16(
                        vshrn_n_s32::<PRECISION>(cr_l_low),
                        vshrn_n_s32::<PRECISION>(cr_l_high),
                    ),
                    i_bias_y,
                )),
                i_cap_uv,
            );
            let cb = vcombine_u8(vqmovn_u16(cb_low), vqmovn_u16(cb_high));
            let cr = vcombine_u8(vqmovn_u16(cr_low), vqmovn_u16(cr_high));
            vst1q_u8(u_ptr.get_unchecked_mut(ux..).as_mut_ptr(), cb);
            vst1q_u8(v_ptr.get_unchecked_mut(ux..).as_mut_ptr(), cr);

            ux += 16;
        } else if chroma_subsampling == YuvChromaSubsampling::Yuv422
            || (chroma_subsampling == YuvChromaSubsampling::Yuv420 && compute_uv_row)
        {
            let r1 = vreinterpretq_s16_u16(vrshrq_n_u16::<1>(vpaddlq_u8(r_values_u8)));
            let g1 = vreinterpretq_s16_u16(vrshrq_n_u16::<1>(vpaddlq_u8(g_values_u8)));
            let b1 = vreinterpretq_s16_u16(vrshrq_n_u16::<1>(vpaddlq_u8(b_values_u8)));

            let mut cb_h = vmlal_high_s16(uv_bias, r1, v_cb_r);
            cb_h = vmlal_high_s16(cb_h, g1, v_cb_g);
            cb_h = vmlal_high_s16(cb_h, b1, v_cb_b);

            let mut cb_l = vmlal_s16(uv_bias, vget_low_s16(r1), vget_low_s16(v_cb_r));
            cb_l = vmlal_s16(cb_l, vget_low_s16(g1), vget_low_s16(v_cb_g));
            cb_l = vmlal_s16(cb_l, vget_low_s16(b1), vget_low_s16(v_cb_b));

            let cb = vqmovn_u16(vminq_u16(
                vreinterpretq_u16_s16(vmaxq_s16(
                    vcombine_s16(
                        vshrn_n_s32::<PRECISION>(cb_l),
                        vshrn_n_s32::<PRECISION>(cb_h),
                    ),
                    i_bias_y,
                )),
                i_cap_uv,
            ));

            let mut cr_h = vmlal_high_s16(uv_bias, r1, v_cr_r);
            cr_h = vmlal_high_s16(cr_h, g1, v_cr_g);
            cr_h = vmlal_high_s16(cr_h, b1, v_cr_b);

            let mut cr_l = vmlal_s16(uv_bias, vget_low_s16(r1), vget_low_s16(v_cr_r));
            cr_l = vmlal_s16(cr_l, vget_low_s16(g1), vget_low_s16(v_cr_g));
            cr_l = vmlal_s16(cr_l, vget_low_s16(b1), vget_low_s16(v_cr_b));

            let cr = vqmovn_u16(vminq_u16(
                vreinterpretq_u16_s16(vmaxq_s16(
                    vcombine_s16(
                        vshrn_n_s32::<PRECISION>(cr_l),
                        vshrn_n_s32::<PRECISION>(cr_h),
                    ),
                    i_bias_y,
                )),
                i_cap_uv,
            ));

            vst1_u8(u_ptr.get_unchecked_mut(ux..).as_mut_ptr(), cb);
            vst1_u8(v_ptr.get_unchecked_mut(ux..).as_mut_ptr(), cr);

            ux += 8;
        }

        cx += 16;
    }

    ProcessedOffset { cx, ux }
}
