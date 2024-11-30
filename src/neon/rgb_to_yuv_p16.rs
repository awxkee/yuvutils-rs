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
use crate::{YuvBytesPacking, YuvEndianness};
use std::arch::aarch64::*;

pub(crate) unsafe fn neon_rgba_to_yuv_p16<
    const ORIGIN_CHANNELS: u8,
    const SAMPLING: u8,
    const ENDIANNESS: u8,
    const BYTES_POSITION: u8,
    const PRECISION: i32,
    const BIT_DEPTH: usize,
>(
    transform: &CbCrForwardTransform<i32>,
    range: &YuvChromaRange,
    y_plane: &mut [u16],
    u_plane: &mut [u16],
    v_plane: &mut [u16],
    rgba: &[u16],
    start_cx: usize,
    start_ux: usize,
    width: usize,
    compute_uv_row: bool,
) -> ProcessedOffset {
    let chroma_subsampling: YuvChromaSubsampling = SAMPLING.into();
    let source_channels: YuvSourceChannels = ORIGIN_CHANNELS.into();
    let endianness: YuvEndianness = ENDIANNESS.into();
    let bytes_position: YuvBytesPacking = BYTES_POSITION.into();
    let channels = source_channels.get_channels_count();

    let rounding_const_bias: i32 = (1 << (PRECISION - 1)) - 1;
    let bias_y = range.bias_y as i32 * (1 << PRECISION) + rounding_const_bias;
    let bias_uv = range.bias_uv as i32 * (1 << PRECISION) + rounding_const_bias;

    let y_ptr = y_plane.as_mut_ptr();
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

    let i_bias_y = vdupq_n_u16(range.bias_y as u16);
    let i_cap_y = vdupq_n_u16(range.range_y as u16 + range.bias_y as u16);
    let i_cap_uv = vdupq_n_u16(range.bias_y as u16 + range.range_uv as u16);

    while cx + 8 < width {
        let r_values;
        let g_values;
        let b_values;

        let src_ptr = rgba.get_unchecked(cx * channels..);

        match source_channels {
            YuvSourceChannels::Rgb | YuvSourceChannels::Bgr => {
                let rgb_values = vld3q_u16(src_ptr.as_ptr());
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
                let rgb_values = vld4q_u16(src_ptr.as_ptr());
                r_values = rgb_values.0;
                g_values = rgb_values.1;
                b_values = rgb_values.2;
            }
            YuvSourceChannels::Bgra => {
                let rgb_values = vld4q_u16(src_ptr.as_ptr());
                r_values = rgb_values.2;
                g_values = rgb_values.1;
                b_values = rgb_values.0;
            }
        }

        let mut y_h = vmlal_high_laneq_s16::<0>(y_bias, vreinterpretq_s16_u16(r_values), v_weights);
        y_h = vmlal_high_laneq_s16::<1>(y_h, vreinterpretq_s16_u16(g_values), v_weights);
        y_h = vmlal_high_laneq_s16::<2>(y_h, vreinterpretq_s16_u16(b_values), v_weights);

        let mut y_l = vmlal_laneq_s16::<0>(
            y_bias,
            vreinterpret_s16_u16(vget_low_u16(r_values)),
            v_weights,
        );
        y_l = vmlal_laneq_s16::<1>(y_l, vreinterpret_s16_u16(vget_low_u16(g_values)), v_weights);
        y_l = vmlal_laneq_s16::<2>(y_l, vreinterpret_s16_u16(vget_low_u16(b_values)), v_weights);

        let mut y_vl = vminq_u16(
            vmaxq_u16(
                vcombine_u16(
                    vqshrun_n_s32::<PRECISION>(y_l),
                    vqshrun_n_s32::<PRECISION>(y_h),
                ),
                i_bias_y,
            ),
            i_cap_y,
        );

        if bytes_position == YuvBytesPacking::MostSignificantBytes {
            y_vl = vshlq_u16(y_vl, v_shift_count);
        }

        if endianness == YuvEndianness::BigEndian {
            y_vl = vreinterpretq_u16_u8(vrev16q_u8(vreinterpretq_u8_u16(y_vl)));
        }

        vst1q_u16(y_ptr.add(cx), y_vl);

        if compute_uv_row {
            let mut cb_h =
                vmlal_high_laneq_s16::<3>(uv_bias, vreinterpretq_s16_u16(r_values), v_weights);
            cb_h = vmlal_high_laneq_s16::<4>(cb_h, vreinterpretq_s16_u16(g_values), v_weights);
            cb_h = vmlal_high_laneq_s16::<5>(cb_h, vreinterpretq_s16_u16(b_values), v_weights);

            let mut cb_l = vmlal_laneq_s16::<3>(
                uv_bias,
                vreinterpret_s16_u16(vget_low_u16(r_values)),
                v_weights,
            );
            cb_l = vmlal_laneq_s16::<4>(
                cb_l,
                vreinterpret_s16_u16(vget_low_u16(g_values)),
                v_weights,
            );
            cb_l = vmlal_laneq_s16::<5>(
                cb_l,
                vreinterpret_s16_u16(vget_low_u16(b_values)),
                v_weights,
            );

            let mut cb_vl = vminq_u16(
                vmaxq_u16(
                    vcombine_u16(
                        vqshrun_n_s32::<PRECISION>(cb_l),
                        vqshrun_n_s32::<PRECISION>(cb_h),
                    ),
                    i_bias_y,
                ),
                i_cap_uv,
            );

            let mut cr_h =
                vmlal_high_laneq_s16::<6>(uv_bias, vreinterpretq_s16_u16(r_values), v_weights);
            cr_h = vmlal_high_laneq_s16::<7>(cr_h, vreinterpretq_s16_u16(g_values), v_weights);
            cr_h = vmlal_high_laneq_s16::<0>(cr_h, vreinterpretq_s16_u16(b_values), v_cr_b);

            let mut cr_l = vmlal_laneq_s16::<6>(
                uv_bias,
                vreinterpret_s16_u16(vget_low_u16(r_values)),
                v_weights,
            );
            cr_l = vmlal_laneq_s16::<7>(
                cr_l,
                vreinterpret_s16_u16(vget_low_u16(g_values)),
                v_weights,
            );
            cr_l = vmlal_laneq_s16::<0>(cr_l, vreinterpret_s16_u16(vget_low_u16(b_values)), v_cr_b);

            let mut cr_vl = vminq_u16(
                vmaxq_u16(
                    vcombine_u16(
                        vqshrun_n_s32::<PRECISION>(cr_l),
                        vqshrun_n_s32::<PRECISION>(cr_h),
                    ),
                    i_bias_y,
                ),
                i_cap_uv,
            );

            match chroma_subsampling {
                YuvChromaSubsampling::Yuv420 | YuvChromaSubsampling::Yuv422 => {
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
                YuvChromaSubsampling::Yuv444 => {
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
    }

    ProcessedOffset { ux, cx }
}

#[target_feature(enable = "rdm")]
pub(crate) unsafe fn neon_rgba_to_yuv_p16_rdm<
    const ORIGIN_CHANNELS: u8,
    const SAMPLING: u8,
    const ENDIANNESS: u8,
    const BYTES_POSITION: u8,
    const PRECISION: i32,
    const BIT_DEPTH: usize,
>(
    transform: &CbCrForwardTransform<i32>,
    range: &YuvChromaRange,
    y_plane: &mut [u16],
    u_plane: &mut [u16],
    v_plane: &mut [u16],
    rgba: &[u16],
    start_cx: usize,
    start_ux: usize,
    width: usize,
    compute_uv_row: bool,
) -> ProcessedOffset {
    let chroma_subsampling: YuvChromaSubsampling = SAMPLING.into();
    let source_channels: YuvSourceChannels = ORIGIN_CHANNELS.into();
    let endianness: YuvEndianness = ENDIANNESS.into();
    let bytes_position: YuvBytesPacking = BYTES_POSITION.into();
    let channels = source_channels.get_channels_count();

    let bias_y = range.bias_y as i16;
    let bias_uv = range.bias_uv as i16;

    let y_ptr = y_plane.as_mut_ptr();
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
        let mut r_values;
        let mut g_values;
        let mut b_values;

        let src_ptr = rgba.get_unchecked(cx * channels..);

        match source_channels {
            YuvSourceChannels::Rgb | YuvSourceChannels::Bgr => {
                let rgb_values = vld3q_u16(src_ptr.as_ptr());
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
                let rgb_values = vld4q_u16(src_ptr.as_ptr());
                r_values = rgb_values.0;
                g_values = rgb_values.1;
                b_values = rgb_values.2;
            }
            YuvSourceChannels::Bgra => {
                let rgb_values = vld4q_u16(src_ptr.as_ptr());
                r_values = rgb_values.2;
                g_values = rgb_values.1;
                b_values = rgb_values.0;
            }
        }

        r_values = vshlq_n_u16::<SCALE>(r_values);
        g_values = vshlq_n_u16::<SCALE>(g_values);
        b_values = vshlq_n_u16::<SCALE>(b_values);

        let mut y_values =
            vqrdmlahq_laneq_s16::<0>(y_bias, vreinterpretq_s16_u16(r_values), v_weights);
        y_values = vqrdmlahq_laneq_s16::<1>(y_values, vreinterpretq_s16_u16(g_values), v_weights);
        y_values = vqrdmlahq_laneq_s16::<2>(y_values, vreinterpretq_s16_u16(b_values), v_weights);

        let mut y_vl = vreinterpretq_u16_s16(vminq_s16(vmaxq_s16(y_values, i_bias_y), i_cap_y));

        if bytes_position == YuvBytesPacking::MostSignificantBytes {
            y_vl = vshlq_u16(y_vl, v_shift_count);
        }

        if endianness == YuvEndianness::BigEndian {
            y_vl = vreinterpretq_u16_u8(vrev16q_u8(vreinterpretq_u8_u16(y_vl)));
        }

        vst1q_u16(y_ptr.add(cx), y_vl);

        if chroma_subsampling == YuvChromaSubsampling::Yuv444 {
            let mut cb_h =
                vqrdmlahq_laneq_s16::<3>(uv_bias, vreinterpretq_s16_u16(r_values), v_weights);
            cb_h = vqrdmlahq_laneq_s16::<4>(cb_h, vreinterpretq_s16_u16(g_values), v_weights);
            cb_h = vqrdmlahq_laneq_s16::<5>(cb_h, vreinterpretq_s16_u16(b_values), v_weights);

            let mut cb_vl = vreinterpretq_u16_s16(vminq_s16(vmaxq_s16(cb_h, i_bias_y), i_cap_uv));

            let mut cr_h =
                vqrdmlahq_laneq_s16::<6>(uv_bias, vreinterpretq_s16_u16(r_values), v_weights);
            cr_h = vqrdmlahq_laneq_s16::<7>(cr_h, vreinterpretq_s16_u16(g_values), v_weights);
            cr_h = vqrdmlahq_laneq_s16::<0>(cr_h, vreinterpretq_s16_u16(b_values), v_cr_b);

            let mut cr_vl = vreinterpretq_u16_s16(vminq_s16(vmaxq_s16(cr_h, i_bias_y), i_cap_uv));

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
        } else if (chroma_subsampling == YuvChromaSubsampling::Yuv420 && compute_uv_row)
            || (chroma_subsampling == YuvChromaSubsampling::Yuv422)
        {
            let r1 = vreinterpret_s16_u16(vrshrn_n_u32::<1>(vpaddlq_u16(r_values)));
            let g1 = vreinterpret_s16_u16(vrshrn_n_u32::<1>(vpaddlq_u16(g_values)));
            let b1 = vreinterpret_s16_u16(vrshrn_n_u32::<1>(vpaddlq_u16(b_values)));

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
        }

        cx += 8;
    }

    ProcessedOffset { ux, cx }
}
