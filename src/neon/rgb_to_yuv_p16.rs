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
    CbCrForwardTransform, YuvChromaRange, YuvChromaSample, YuvSourceChannels,
};
use crate::{YuvBytesPacking, YuvEndianness};
use std::arch::aarch64::*;

pub unsafe fn neon_rgba_to_yuv_p16<
    const ORIGIN_CHANNELS: u8,
    const SAMPLING: u8,
    const ENDIANNESS: u8,
    const BYTES_POSITION: u8,
>(
    transform: &CbCrForwardTransform<i32>,
    range: &YuvChromaRange,
    y_plane: *mut u16,
    u_plane: *mut u16,
    v_plane: *mut u16,
    rgba: *const u16,
    start_cx: usize,
    start_ux: usize,
    width: usize,
    compute_uv_row: bool,
    bit_depth: u32,
) -> ProcessedOffset {
    let chroma_subsampling: YuvChromaSample = SAMPLING.into();
    let source_channels: YuvSourceChannels = ORIGIN_CHANNELS.into();
    let endianness: YuvEndianness = ENDIANNESS.into();
    let bytes_position: YuvBytesPacking = BYTES_POSITION.into();
    let channels = source_channels.get_channels_count();

    const ROUNDING_CONST_BIAS: i32 = 1 << 7;
    let bias_y = range.bias_y as i32 * (1 << 8) + ROUNDING_CONST_BIAS;
    let bias_uv = range.bias_uv as i32 * (1 << 8) + ROUNDING_CONST_BIAS;

    let mut src_ptr = rgba;

    let y_ptr = y_plane;
    let u_ptr = u_plane;
    let v_ptr = v_plane;

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

    let v_shift_count = vdupq_n_s16(16 - bit_depth as i16);

    while cx + 8 < width {
        let r_values;
        let g_values;
        let b_values;

        match source_channels {
            YuvSourceChannels::Rgb | YuvSourceChannels::Bgr => {
                let rgb_values = vld3q_u16(src_ptr);
                if source_channels == YuvSourceChannels::Rgb {
                    r_values = rgb_values.0;
                    g_values = rgb_values.1;
                    b_values = rgb_values.2;
                } else {
                    r_values = rgb_values.2;
                    g_values = rgb_values.1;
                    b_values = rgb_values.0;
                }
            }
            YuvSourceChannels::Rgba => {
                let rgb_values = vld4q_u16(src_ptr);
                r_values = rgb_values.0;
                g_values = rgb_values.1;
                b_values = rgb_values.2;
            }
            YuvSourceChannels::Bgra => {
                let rgb_values = vld4q_u16(src_ptr);
                r_values = rgb_values.2;
                g_values = rgb_values.1;
                b_values = rgb_values.0;
            }
        }

        let mut y_h = vmlal_high_s16(y_bias, vreinterpretq_s16_u16(r_values), v_yr);
        y_h = vmlal_high_s16(y_h, vreinterpretq_s16_u16(g_values), v_yg);
        y_h = vmlal_high_s16(y_h, vreinterpretq_s16_u16(b_values), v_yb);

        let mut y_l = vmlal_s16(
            y_bias,
            vreinterpret_s16_u16(vget_low_u16(r_values)),
            vget_low_s16(v_yr),
        );
        y_l = vmlal_s16(
            y_l,
            vreinterpret_s16_u16(vget_low_u16(g_values)),
            vget_low_s16(v_yg),
        );
        y_l = vmlal_s16(
            y_l,
            vreinterpret_s16_u16(vget_low_u16(b_values)),
            vget_low_s16(v_yb),
        );

        let mut y_vl = vcombine_u16(vqshrun_n_s32::<8>(y_l), vqshrun_n_s32::<8>(y_h));

        if bytes_position == YuvBytesPacking::MostSignificantBytes {
            y_vl = vshlq_u16(y_vl, v_shift_count);
        }

        if endianness == YuvEndianness::BigEndian {
            y_vl = vreinterpretq_u16_u8(vrev16q_u8(vreinterpretq_u8_u16(y_vl)));
        }

        vst1q_u16(y_ptr.add(cx), y_vl);

        if compute_uv_row {
            let mut cb_h = vmlal_high_s16(uv_bias, vreinterpretq_s16_u16(r_values), v_cb_r);
            cb_h = vmlal_high_s16(cb_h, vreinterpretq_s16_u16(g_values), v_cb_g);
            cb_h = vmlal_high_s16(cb_h, vreinterpretq_s16_u16(b_values), v_cb_b);

            let mut cb_l = vmlal_s16(
                uv_bias,
                vreinterpret_s16_u16(vget_low_u16(r_values)),
                vget_low_s16(v_cb_r),
            );
            cb_l = vmlal_s16(
                cb_l,
                vreinterpret_s16_u16(vget_low_u16(g_values)),
                vget_low_s16(v_cb_g),
            );
            cb_l = vmlal_s16(
                cb_l,
                vreinterpret_s16_u16(vget_low_u16(b_values)),
                vget_low_s16(v_cb_b),
            );

            let mut cb_vl = vcombine_u16(vqshrun_n_s32::<8>(cb_l), vqshrun_n_s32::<8>(cb_h));

            let mut cr_h = vmlal_high_s16(uv_bias, vreinterpretq_s16_u16(r_values), v_cr_r);
            cr_h = vmlal_high_s16(cr_h, vreinterpretq_s16_u16(g_values), v_cr_g);
            cr_h = vmlal_high_s16(cr_h, vreinterpretq_s16_u16(b_values), v_cr_b);

            let mut cr_l = vmlal_s16(
                uv_bias,
                vreinterpret_s16_u16(vget_low_u16(r_values)),
                vget_low_s16(v_cr_r),
            );
            cr_l = vmlal_s16(
                cr_l,
                vreinterpret_s16_u16(vget_low_u16(g_values)),
                vget_low_s16(v_cr_g),
            );
            cr_l = vmlal_s16(
                cr_l,
                vreinterpret_s16_u16(vget_low_u16(b_values)),
                vget_low_s16(v_cr_b),
            );

            let mut cr_vl = vcombine_u16(vqshrun_n_s32::<8>(cr_l), vqshrun_n_s32::<8>(cr_h));

            match chroma_subsampling {
                YuvChromaSample::YUV420 | YuvChromaSample::YUV422 => {
                    let mut cb_s = vrshrn_n_u32::<1>(vpaddlq_u16(cb_vl));
                    let mut cr_s = vrshrn_n_u32::<1>(vpaddlq_u16(cr_vl));

                    if bytes_position == YuvBytesPacking::MostSignificantBytes {
                        cb_s = vshl_u16(cb_s, vget_low_s16(v_shift_count));
                        cr_s = vshl_u16(cr_s, vget_low_s16(v_shift_count));
                    }

                    if endianness == YuvEndianness::BigEndian {
                        cb_s = vreinterpret_u16_u8(vrev16_u8(vreinterpret_u8_u16(cb_s)));
                        cr_s = vreinterpret_u16_u8(vrev16_u8(vreinterpret_u8_u16(cr_s)));
                    }

                    vst1_u16(u_ptr.add(ux), cb_s);
                    vst1_u16(v_ptr.add(ux), cr_s);

                    ux += 4;
                }
                YuvChromaSample::YUV444 => {
                    if bytes_position == YuvBytesPacking::MostSignificantBytes {
                        cb_vl = vshlq_u16(cb_vl, v_shift_count);
                        cr_vl = vshlq_u16(cr_vl, v_shift_count);
                    }

                    if endianness == YuvEndianness::BigEndian {
                        cb_vl = vreinterpretq_u16_u8(vrev16q_u8(vreinterpretq_u8_u16(cb_vl)));
                        cr_vl = vreinterpretq_u16_u8(vrev16q_u8(vreinterpretq_u8_u16(cr_vl)));
                    }

                    vst1q_u16(u_ptr.add(ux), cb_vl);
                    vst1q_u16(v_ptr.add(ux), cr_vl);

                    ux += 8;
                }
            }
        }

        cx += 8;

        src_ptr = src_ptr.add(channels * 8);
    }

    ProcessedOffset { ux, cx }
}
