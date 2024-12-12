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
use crate::neon::neon_simd_support::neon_vld_rgb16_for_yuv;
use crate::yuv_support::{CbCrForwardTransform, YuvChromaRange, YuvSourceChannels};
use crate::{YuvBytesPacking, YuvEndianness};
use std::arch::aarch64::*;

pub(crate) unsafe fn neon_rgba_to_yuv_p16_420<
    const ORIGIN_CHANNELS: u8,
    const ENDIANNESS: u8,
    const BYTES_POSITION: u8,
    const PRECISION: i32,
    const BIT_DEPTH: usize,
>(
    transform: &CbCrForwardTransform<i32>,
    range: &YuvChromaRange,
    y_plane0: &mut [u16],
    y_plane1: &mut [u16],
    u_plane: &mut [u16],
    v_plane: &mut [u16],
    rgba0: &[u16],
    rgba1: &[u16],
    start_cx: usize,
    start_ux: usize,
    width: usize,
) -> ProcessedOffset {
    let source_channels: YuvSourceChannels = ORIGIN_CHANNELS.into();
    let endianness: YuvEndianness = ENDIANNESS.into();
    let bytes_position: YuvBytesPacking = BYTES_POSITION.into();
    let channels = source_channels.get_channels_count();

    let rounding_const_bias: i32 = (1 << (PRECISION - 1)) - 1;
    let bias_y = range.bias_y as i32 * (1 << PRECISION) + rounding_const_bias;
    let bias_uv = range.bias_uv as i32 * (1 << PRECISION) + rounding_const_bias;

    let u_ptr = u_plane.as_mut_ptr();
    let v_ptr = v_plane.as_mut_ptr();

    let y_bias = vdupq_n_s32(bias_y);
    let uv_bias = vdupq_n_s32(bias_uv);

    let weights_arr: [i16; 8] = [
        transform.yr as i16,
        transform.yg as i16,
        transform.yb as i16,
        transform.cb_r as i16,
        transform.cb_g as i16,
        transform.cb_b as i16,
        transform.cr_r as i16,
        transform.cr_g as i16,
    ];
    let v_weights = vld1q_s16(weights_arr.as_ptr());
    let v_cr_b = vdupq_n_s16(transform.cr_b as i16);

    let mut cx = start_cx;
    let mut ux = start_ux;

    let v_shift_count = vdupq_n_s16(16 - BIT_DEPTH as i16);

    let i_bias_y = vdup_n_u16(range.bias_y as u16);
    let i_cap_y = vdupq_n_u16(range.range_y as u16 + range.bias_y as u16);
    let i_cap_uv = vdup_n_u16(range.bias_y as u16 + range.range_uv as u16);

    while cx + 8 < width {
        let src_ptr0 = rgba0.get_unchecked(cx * channels..);
        let (r_values0, g_values0, b_values0) =
            neon_vld_rgb16_for_yuv::<ORIGIN_CHANNELS>(src_ptr0.as_ptr());

        let src_ptr1 = rgba1.get_unchecked(cx * channels..);
        let (r_values1, g_values1, b_values1) =
            neon_vld_rgb16_for_yuv::<ORIGIN_CHANNELS>(src_ptr1.as_ptr());

        let mut y0_h =
            vmlal_high_laneq_s16::<0>(y_bias, vreinterpretq_s16_u16(r_values0), v_weights);
        y0_h = vmlal_high_laneq_s16::<1>(y0_h, vreinterpretq_s16_u16(g_values0), v_weights);
        y0_h = vmlal_high_laneq_s16::<2>(y0_h, vreinterpretq_s16_u16(b_values0), v_weights);

        let mut y0_l = vmlal_laneq_s16::<0>(
            y_bias,
            vreinterpret_s16_u16(vget_low_u16(r_values0)),
            v_weights,
        );
        y0_l = vmlal_laneq_s16::<1>(
            y0_l,
            vreinterpret_s16_u16(vget_low_u16(g_values0)),
            v_weights,
        );
        y0_l = vmlal_laneq_s16::<2>(
            y0_l,
            vreinterpret_s16_u16(vget_low_u16(b_values0)),
            v_weights,
        );

        let mut y0_vl = vminq_u16(
            vcombine_u16(
                vqshrun_n_s32::<PRECISION>(y0_l),
                vqshrun_n_s32::<PRECISION>(y0_h),
            ),
            i_cap_y,
        );

        let mut y1_h =
            vmlal_high_laneq_s16::<0>(y_bias, vreinterpretq_s16_u16(r_values1), v_weights);
        y1_h = vmlal_high_laneq_s16::<1>(y1_h, vreinterpretq_s16_u16(g_values1), v_weights);
        y1_h = vmlal_high_laneq_s16::<2>(y1_h, vreinterpretq_s16_u16(b_values1), v_weights);

        let mut y1_l = vmlal_laneq_s16::<0>(
            y_bias,
            vreinterpret_s16_u16(vget_low_u16(r_values1)),
            v_weights,
        );
        y1_l = vmlal_laneq_s16::<1>(
            y1_l,
            vreinterpret_s16_u16(vget_low_u16(g_values1)),
            v_weights,
        );
        y1_l = vmlal_laneq_s16::<2>(
            y1_l,
            vreinterpret_s16_u16(vget_low_u16(b_values1)),
            v_weights,
        );

        let mut y1_vl = vminq_u16(
            vcombine_u16(
                vqshrun_n_s32::<PRECISION>(y1_l),
                vqshrun_n_s32::<PRECISION>(y1_h),
            ),
            i_cap_y,
        );

        if bytes_position == YuvBytesPacking::MostSignificantBytes {
            y0_vl = vshlq_u16(y0_vl, v_shift_count);
            y1_vl = vshlq_u16(y1_vl, v_shift_count);
        }

        if endianness == YuvEndianness::BigEndian {
            y0_vl = vreinterpretq_u16_u8(vrev16q_u8(vreinterpretq_u8_u16(y0_vl)));
            y1_vl = vreinterpretq_u16_u8(vrev16q_u8(vreinterpretq_u8_u16(y1_vl)));
        }

        vst1q_u16(y_plane0.get_unchecked_mut(cx..).as_mut_ptr(), y0_vl);
        vst1q_u16(y_plane1.get_unchecked_mut(cx..).as_mut_ptr(), y1_vl);

        let r_values = vrshrn_n_u32::<1>(vpaddlq_u16(vhaddq_u16(r_values0, r_values1)));
        let g_values = vrshrn_n_u32::<1>(vpaddlq_u16(vhaddq_u16(g_values0, g_values1)));
        let b_values = vrshrn_n_u32::<1>(vpaddlq_u16(vhaddq_u16(b_values0, b_values1)));

        let mut cb_h = vmlal_laneq_s16::<3>(uv_bias, vreinterpret_s16_u16(r_values), v_weights);
        cb_h = vmlal_laneq_s16::<4>(cb_h, vreinterpret_s16_u16(g_values), v_weights);
        cb_h = vmlal_laneq_s16::<5>(cb_h, vreinterpret_s16_u16(b_values), v_weights);

        let mut cb_vl = vmin_u16(
            vmax_u16(vqshrun_n_s32::<PRECISION>(cb_h), i_bias_y),
            i_cap_uv,
        );

        let mut cr_h = vmlal_laneq_s16::<6>(uv_bias, vreinterpret_s16_u16(r_values), v_weights);
        cr_h = vmlal_laneq_s16::<7>(cr_h, vreinterpret_s16_u16(g_values), v_weights);
        cr_h = vmlal_laneq_s16::<0>(cr_h, vreinterpret_s16_u16(b_values), v_cr_b);

        let mut cr_vl = vmin_u16(
            vmax_u16(vqshrun_n_s32::<PRECISION>(cr_h), i_bias_y),
            i_cap_uv,
        );

        if bytes_position == YuvBytesPacking::MostSignificantBytes {
            cb_vl = vshl_u16(cb_vl, vget_low_s16(v_shift_count));
            cr_vl = vshl_u16(cr_vl, vget_low_s16(v_shift_count));
        }

        if endianness == YuvEndianness::BigEndian {
            cb_vl = vreinterpret_u16_u8(vrev16_u8(vreinterpret_u8_u16(cb_vl)));
            cr_vl = vreinterpret_u16_u8(vrev16_u8(vreinterpret_u8_u16(cr_vl)));
        }

        vst1_u16(u_ptr.add(ux), cb_vl);
        vst1_u16(v_ptr.add(ux), cr_vl);

        ux += 4;
        cx += 8;
    }

    ProcessedOffset { ux, cx }
}

#[target_feature(enable = "rdm")]
pub(crate) unsafe fn neon_rgba_to_yuv_p16_rdm_420<
    const ORIGIN_CHANNELS: u8,
    const ENDIANNESS: u8,
    const BYTES_POSITION: u8,
    const PRECISION: i32,
    const BIT_DEPTH: usize,
>(
    transform: &CbCrForwardTransform<i32>,
    range: &YuvChromaRange,
    y_plane0: &mut [u16],
    y_plane1: &mut [u16],
    u_plane: &mut [u16],
    v_plane: &mut [u16],
    rgba0: &[u16],
    rgba1: &[u16],
    start_cx: usize,
    start_ux: usize,
    width: usize,
) -> ProcessedOffset {
    let source_channels: YuvSourceChannels = ORIGIN_CHANNELS.into();
    let endianness: YuvEndianness = ENDIANNESS.into();
    let bytes_position: YuvBytesPacking = BYTES_POSITION.into();
    let channels = source_channels.get_channels_count();

    let bias_y = range.bias_y as i16;
    let bias_uv = range.bias_uv as i16;

    let u_ptr = u_plane.as_mut_ptr();
    let v_ptr = v_plane.as_mut_ptr();

    let y_bias = vdupq_n_s16(bias_y);
    let uv_bias = vdupq_n_s16(bias_uv);

    const SCALE: i32 = 2;

    let weights_arr: [i16; 8] = [
        transform.yr as i16,
        transform.yg as i16,
        transform.yb as i16,
        transform.cb_r as i16,
        transform.cb_g as i16,
        transform.cb_b as i16,
        transform.cr_r as i16,
        transform.cr_g as i16,
    ];
    let v_weights = vld1q_s16(weights_arr.as_ptr());
    let v_cr_b = vdupq_n_s16(transform.cr_b as i16);

    let i_bias_y = vdupq_n_s16(range.bias_y as i16);
    let i_cap_y = vdupq_n_s16(range.range_y as i16 + range.bias_y as i16);
    let i_cap_uv = vdupq_n_s16(range.bias_y as i16 + range.range_uv as i16);

    let mut cx = start_cx;
    let mut ux = start_ux;

    let v_shift_count = vdupq_n_s16(16 - BIT_DEPTH as i16);

    while cx + 8 < width {
        let src_ptr0 = rgba0.get_unchecked(cx * channels..);
        let (mut r_values0, mut g_values0, mut b_values0) =
            neon_vld_rgb16_for_yuv::<ORIGIN_CHANNELS>(src_ptr0.as_ptr());

        let src_ptr1 = rgba1.get_unchecked(cx * channels..);
        let (mut r_values1, mut g_values1, mut b_values1) =
            neon_vld_rgb16_for_yuv::<ORIGIN_CHANNELS>(src_ptr1.as_ptr());

        r_values0 = vshlq_n_u16::<SCALE>(r_values0);
        g_values0 = vshlq_n_u16::<SCALE>(g_values0);
        b_values0 = vshlq_n_u16::<SCALE>(b_values0);

        r_values1 = vshlq_n_u16::<SCALE>(r_values1);
        g_values1 = vshlq_n_u16::<SCALE>(g_values1);
        b_values1 = vshlq_n_u16::<SCALE>(b_values1);

        let mut y0_values =
            vqrdmlahq_laneq_s16::<0>(y_bias, vreinterpretq_s16_u16(r_values0), v_weights);
        y0_values =
            vqrdmlahq_laneq_s16::<1>(y0_values, vreinterpretq_s16_u16(g_values0), v_weights);
        y0_values =
            vqrdmlahq_laneq_s16::<2>(y0_values, vreinterpretq_s16_u16(b_values0), v_weights);

        let mut y1_values =
            vqrdmlahq_laneq_s16::<0>(y_bias, vreinterpretq_s16_u16(r_values1), v_weights);
        y1_values =
            vqrdmlahq_laneq_s16::<1>(y1_values, vreinterpretq_s16_u16(g_values1), v_weights);
        y1_values =
            vqrdmlahq_laneq_s16::<2>(y1_values, vreinterpretq_s16_u16(b_values1), v_weights);

        let mut y0_vl = vreinterpretq_u16_s16(vminq_s16(y0_values, i_cap_y));
        let mut y1_vl = vreinterpretq_u16_s16(vminq_s16(y1_values, i_cap_y));

        if bytes_position == YuvBytesPacking::MostSignificantBytes {
            y0_vl = vshlq_u16(y0_vl, v_shift_count);
            y1_vl = vshlq_u16(y1_vl, v_shift_count);
        }

        if endianness == YuvEndianness::BigEndian {
            y0_vl = vreinterpretq_u16_u8(vrev16q_u8(vreinterpretq_u8_u16(y0_vl)));
            y1_vl = vreinterpretq_u16_u8(vrev16q_u8(vreinterpretq_u8_u16(y1_vl)));
        }

        vst1q_u16(y_plane0.get_unchecked_mut(cx..).as_mut_ptr(), y0_vl);
        vst1q_u16(y_plane1.get_unchecked_mut(cx..).as_mut_ptr(), y1_vl);

        let r1 = vreinterpret_s16_u16(vrshrn_n_u32::<1>(vpaddlq_u16(vhaddq_u16(
            r_values0, r_values1,
        ))));
        let g1 = vreinterpret_s16_u16(vrshrn_n_u32::<1>(vpaddlq_u16(vhaddq_u16(
            g_values0, g_values1,
        ))));
        let b1 = vreinterpret_s16_u16(vrshrn_n_u32::<1>(vpaddlq_u16(vhaddq_u16(
            b_values0, b_values1,
        ))));

        let mut cbk = vqrdmlah_laneq_s16::<3>(vget_low_s16(uv_bias), r1, v_weights);
        cbk = vqrdmlah_laneq_s16::<4>(cbk, g1, v_weights);
        cbk = vqrdmlah_laneq_s16::<5>(cbk, b1, v_weights);

        let mut cb = vreinterpret_u16_s16(vmin_s16(
            vmax_s16(cbk, vget_low_s16(i_bias_y)),
            vget_low_s16(i_cap_uv),
        ));

        let mut crk = vqrdmlah_laneq_s16::<6>(vget_low_s16(uv_bias), r1, v_weights);
        crk = vqrdmlah_laneq_s16::<7>(crk, g1, v_weights);
        crk = vqrdmlah_laneq_s16::<0>(crk, b1, v_cr_b);

        let mut cr = vreinterpret_u16_s16(vmin_s16(
            vmax_s16(crk, vget_low_s16(i_bias_y)),
            vget_low_s16(i_cap_uv),
        ));

        if bytes_position == YuvBytesPacking::MostSignificantBytes {
            cb = vshl_u16(cb, vget_low_s16(v_shift_count));
            cr = vshl_u16(cr, vget_low_s16(v_shift_count));
        }

        if endianness == YuvEndianness::BigEndian {
            cb = vreinterpret_u16_u8(vrev16_u8(vreinterpret_u8_u16(cb)));
            cr = vreinterpret_u16_u8(vrev16_u8(vreinterpret_u8_u16(cr)));
        }

        vst1_u16(u_ptr.add(ux), cb);
        vst1_u16(v_ptr.add(ux), cr);

        ux += 4;
        cx += 8;
    }

    ProcessedOffset { ux, cx }
}