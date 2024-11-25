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

use crate::{YuvBytesPacking, YuvEndianness};
use std::arch::aarch64::*;

#[inline(always)]
pub(crate) unsafe fn vdotl_s16<const PRECISION: i32>(
    acc: (int32x4_t, int32x4_t),
    v0: int16x8_t,
    c0: int16x8_t,
) -> int16x8_t {
    let hi = vmlal_high_s16(acc.1, v0, c0);
    let lo = vmlal_s16(acc.0, vget_low_s16(v0), vget_low_s16(c0));
    vcombine_s16(vrshrn_n_s32::<PRECISION>(lo), vrshrn_n_s32::<PRECISION>(hi))
}

#[inline(always)]
pub(crate) unsafe fn vdotl_s16_x2<const PRECISION: i32>(
    acc: (int32x4_t, int32x4_t),
    v0: int16x8_t,
    c0: int16x8_t,
    v1: int16x8_t,
    c1: int16x8_t,
) -> int16x8_t {
    let mut hi = vmlal_high_s16(acc.1, v0, c0);
    let mut lo = vmlal_s16(acc.0, vget_low_s16(v0), vget_low_s16(c0));
    hi = vmlal_high_s16(hi, v1, c1);
    lo = vmlal_s16(lo, vget_low_s16(v1), vget_low_s16(c1));
    vcombine_s16(vrshrn_n_s32::<PRECISION>(lo), vrshrn_n_s32::<PRECISION>(hi))
}

#[inline(always)]
pub(crate) unsafe fn vaddn_dot<const PRECISION: i32>(
    acc: (int32x4_t, int32x4_t),
    w: (int32x4_t, int32x4_t),
) -> int16x8_t {
    vcombine_s16(
        vrshrn_n_s32::<PRECISION>(vaddq_s32(acc.0, w.0)),
        vrshrn_n_s32::<PRECISION>(vaddq_s32(acc.1, w.1)),
    )
}

#[inline(always)]
pub(crate) unsafe fn vweight_x2(
    v0: int16x8_t,
    c0: int16x8_t,
    v1: int16x8_t,
    c1: int16x8_t,
) -> (int32x4_t, int32x4_t) {
    let mut lo = vmull_s16(vget_low_s16(v0), vget_low_s16(c0));
    let mut hi = vmull_high_s16(v0, c0);
    lo = vmlal_s16(lo, vget_low_s16(v1), vget_low_s16(c1));
    hi = vmlal_high_s16(hi, v1, c1);
    (lo, hi)
}

#[inline(always)]
pub(crate) unsafe fn vmullq_s16(v: int16x8_t, q: int16x8_t) -> (int32x4_t, int32x4_t) {
    (
        vmull_s16(vget_low_s16(v), vget_low_s16(q)),
        vmull_high_s16(v, q),
    )
}

#[inline(always)]
pub(crate) unsafe fn neon_div_by_255(v: uint16x8_t) -> uint8x8_t {
    let addition = vdupq_n_u16(127);
    vqshrn_n_u16::<8>(vrsraq_n_u16::<8>(vaddq_u16(v, addition), v))
}

#[inline(always)]
pub(crate) unsafe fn neon_premultiply_alpha(v: uint8x16_t, a_values: uint8x16_t) -> uint8x16_t {
    let acc_hi = vmull_high_u8(v, a_values);
    let acc_lo = vmull_u8(vget_low_u8(v), vget_low_u8(a_values));
    let hi = neon_div_by_255(acc_hi);
    let lo = neon_div_by_255(acc_lo);
    vcombine_u8(lo, hi)
}

#[inline(always)]
pub(crate) unsafe fn vld_s16_endian<const ENDIANNESS: u8, const BYTES_POSITION: u8>(
    ptr: *const u16,
    msb: int16x4_t,
) -> int16x4_t {
    let endianness: YuvEndianness = ENDIANNESS.into();
    let bytes_position: YuvBytesPacking = BYTES_POSITION.into();
    let mut v = vld1_u16(ptr);
    if endianness == YuvEndianness::BigEndian {
        v = vreinterpret_u16_u8(vrev16_u8(vreinterpret_u8_u16(v)));
    }
    if bytes_position == YuvBytesPacking::MostSignificantBytes {
        v = vshl_u16(v, msb);
    }
    vreinterpret_s16_u16(v)
}

#[inline(always)]
pub(crate) unsafe fn vldq_s16_endian<const ENDIANNESS: u8, const BYTES_POSITION: u8>(
    ptr: *const u16,
    msb: int16x8_t,
) -> int16x8_t {
    let endianness: YuvEndianness = ENDIANNESS.into();
    let bytes_position: YuvBytesPacking = BYTES_POSITION.into();
    let mut v = vld1q_u16(ptr);
    if endianness == YuvEndianness::BigEndian {
        v = vreinterpretq_u16_u8(vrev16q_u8(vreinterpretq_u8_u16(v)));
    }
    if bytes_position == YuvBytesPacking::MostSignificantBytes {
        v = vshlq_u16(v, msb);
    }
    vreinterpretq_s16_u16(v)
}
