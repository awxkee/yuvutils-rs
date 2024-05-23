/*
 * // Copyright (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use std::arch::aarch64::*;

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[inline(always)]
pub unsafe fn neon_div_by_255(v: uint16x8_t) -> uint16x8_t {
    let rounding = vdupq_n_u16(1 << 7);
    let x = vqaddq_u16(v, rounding);
    let multiplier = vdupq_n_u16(0x8080);
    let hi = vmull_high_u16(x, multiplier);
    let lo = vmull_u16(vget_low_u16(x), vget_low_u16(multiplier));

    let hi_16 = vqshrn_n_u32::<7>(vshrq_n_u32::<16>(hi));
    let lo_16 = vqshrn_n_u32::<7>(vshrq_n_u32::<16>(lo));
    vcombine_u16(lo_16, hi_16)
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[inline(always)]
pub unsafe fn neon_premultiply_alpha(v: uint8x16_t, a_values: uint8x16_t) -> uint8x16_t {
    let acc_hi = vmull_high_u8(v, a_values);
    let acc_lo = vmull_u8(vget_low_u8(v), vget_low_u8(a_values));
    let hi = vqmovn_u16(neon_div_by_255(acc_hi));
    let lo = vqmovn_u16(neon_div_by_255(acc_lo));
    vcombine_u8(lo, hi)
}
