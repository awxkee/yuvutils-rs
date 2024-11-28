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
use crate::yuv_support::{
    CbCrInverseTransform, YuvBytesPacking, YuvChromaRange, YuvEndianness, YuvSourceChannels,
};

#[target_feature(enable = "rdm")]
pub(crate) unsafe fn neon_y_p16_to_rgba16_rdm<
    const DESTINATION_CHANNELS: u8,
    const ENDIANNESS: u8,
    const BYTES_POSITION: u8,
>(
    y_ld_ptr: *const u16,
    rgba: *mut u16,
    width: u32,
    range: &YuvChromaRange,
    transform: &CbCrInverseTransform<i32>,
    start_cx: usize,
    bit_depth: usize,
) -> ProcessedOffset {
    let destination_channels: YuvSourceChannels = DESTINATION_CHANNELS.into();
    let channels = destination_channels.get_channels_count();
    let endianness: YuvEndianness = ENDIANNESS.into();
    let bytes_position: YuvBytesPacking = BYTES_POSITION.into();
    let dst_ptr = rgba;

    let y_corr = vdupq_n_s16(range.bias_y as i16);
    let v_min_values = vdupq_n_s16(0i16);
    let v_alpha = vdupq_n_u16((1 << bit_depth) - 1);
    let v_msb_shift = vdupq_n_s16(bit_depth as i16 - 16);

    let mut cx = start_cx;

    const V_SCALE: i32 = 2;

    while cx + 8 < width as usize {
        let y_values: int16x8_t;

        match endianness {
            YuvEndianness::BigEndian => {
                let mut y_u_values = vreinterpretq_u16_u8(vrev16q_u8(vreinterpretq_u8_u16(
                    vld1q_u16(y_ld_ptr.add(cx)),
                )));
                if bytes_position == YuvBytesPacking::MostSignificantBytes {
                    y_u_values = vshlq_u16(y_u_values, v_msb_shift);
                }
                y_values = vsubq_s16(vreinterpretq_s16_u16(y_u_values), y_corr);
            }
            YuvEndianness::LittleEndian => {
                let mut y_vl = vld1q_u16(y_ld_ptr.add(cx));
                if bytes_position == YuvBytesPacking::MostSignificantBytes {
                    y_vl = vshlq_u16(y_vl, v_msb_shift);
                }
                y_values = vsubq_s16(vreinterpretq_s16_u16(y_vl), y_corr);
            }
        }

        let y_high = vqrdmulhq_n_s16(vshlq_n_s16::<V_SCALE>(y_values), transform.y_coef as i16);

        let r_values = vreinterpretq_u16_s16(vmaxq_s16(y_high, v_min_values));

        match destination_channels {
            YuvSourceChannels::Rgb => {
                let dst_pack = uint16x8x3_t(r_values, r_values, r_values);
                vst3q_u16(dst_ptr.add(cx * channels), dst_pack);
            }
            YuvSourceChannels::Bgr => {
                let dst_pack = uint16x8x3_t(r_values, r_values, r_values);
                vst3q_u16(dst_ptr.add(cx * channels), dst_pack);
            }
            YuvSourceChannels::Rgba => {
                let dst_pack = uint16x8x4_t(r_values, r_values, r_values, v_alpha);
                vst4q_u16(dst_ptr.add(cx * channels), dst_pack);
            }
            YuvSourceChannels::Bgra => {
                let dst_pack = uint16x8x4_t(r_values, r_values, r_values, v_alpha);
                vst4q_u16(dst_ptr.add(cx * channels), dst_pack);
            }
        }

        cx += 8;
    }

    ProcessedOffset { cx, ux: 0 }
}

pub(crate) unsafe fn neon_y_p16_to_rgba16_row<
    const DESTINATION_CHANNELS: u8,
    const ENDIANNESS: u8,
    const BYTES_POSITION: u8,
    const PRECISION: i32,
>(
    y_ld_ptr: *const u16,
    rgba: *mut u16,
    width: u32,
    range: &YuvChromaRange,
    transform: &CbCrInverseTransform<i32>,
    start_cx: usize,
    bit_depth: usize,
) -> ProcessedOffset {
    let destination_channels: YuvSourceChannels = DESTINATION_CHANNELS.into();
    let channels = destination_channels.get_channels_count();
    let endianness: YuvEndianness = ENDIANNESS.into();
    let bytes_position: YuvBytesPacking = BYTES_POSITION.into();
    let dst_ptr = rgba;

    let y_corr = vdupq_n_s16(range.bias_y as i16);
    let v_luma_coeff = vdupq_n_s16(transform.y_coef as i16);
    let v_min_values = vdupq_n_s16(0i16);
    let v_alpha = vdupq_n_u16((1 << bit_depth) - 1);
    let v_msb_shift = vdupq_n_s16(bit_depth as i16 - 16);

    let mut cx = start_cx;

    while cx + 8 < width as usize {
        let y_values: int16x8_t;

        match endianness {
            YuvEndianness::BigEndian => {
                let mut y_u_values = vreinterpretq_u16_u8(vrev16q_u8(vreinterpretq_u8_u16(
                    vld1q_u16(y_ld_ptr.add(cx)),
                )));
                if bytes_position == YuvBytesPacking::MostSignificantBytes {
                    y_u_values = vshlq_u16(y_u_values, v_msb_shift);
                }
                y_values = vsubq_s16(vreinterpretq_s16_u16(y_u_values), y_corr);
            }
            YuvEndianness::LittleEndian => {
                let mut y_vl = vld1q_u16(y_ld_ptr.add(cx));
                if bytes_position == YuvBytesPacking::MostSignificantBytes {
                    y_vl = vshlq_u16(y_vl, v_msb_shift);
                }
                y_values = vsubq_s16(vreinterpretq_s16_u16(y_vl), y_corr);
            }
        }

        let y_high = vmull_high_s16(y_values, v_luma_coeff);

        let r_high = vrshrn_n_s32::<PRECISION>(y_high);

        let y_low = vmull_s16(vget_low_s16(y_values), vget_low_s16(v_luma_coeff));

        let r_low = vrshrn_n_s32::<PRECISION>(y_low);

        let r_values = vreinterpretq_u16_s16(vmaxq_s16(vcombine_s16(r_low, r_high), v_min_values));

        match destination_channels {
            YuvSourceChannels::Rgb => {
                let dst_pack = uint16x8x3_t(r_values, r_values, r_values);
                vst3q_u16(dst_ptr.add(cx * channels), dst_pack);
            }
            YuvSourceChannels::Bgr => {
                let dst_pack = uint16x8x3_t(r_values, r_values, r_values);
                vst3q_u16(dst_ptr.add(cx * channels), dst_pack);
            }
            YuvSourceChannels::Rgba => {
                let dst_pack = uint16x8x4_t(r_values, r_values, r_values, v_alpha);
                vst4q_u16(dst_ptr.add(cx * channels), dst_pack);
            }
            YuvSourceChannels::Bgra => {
                let dst_pack = uint16x8x4_t(r_values, r_values, r_values, v_alpha);
                vst4q_u16(dst_ptr.add(cx * channels), dst_pack);
            }
        }

        cx += 8;
    }

    ProcessedOffset { cx, ux: 0 }
}
