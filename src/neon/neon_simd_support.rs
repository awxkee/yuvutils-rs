/*
 * // Copyright (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

use std::arch::aarch64::*;

#[inline(always)]
pub unsafe fn neon_div_by_255(v: uint16x8_t) -> uint8x8_t {
    let addition = vdupq_n_u16(127);
    vqshrn_n_u16::<8>(vrsraq_n_u16::<8>(vaddq_u16(v, addition), v))
}

#[inline(always)]
pub unsafe fn neon_premultiply_alpha(v: uint8x16_t, a_values: uint8x16_t) -> uint8x16_t {
    let acc_hi = vmull_high_u8(v, a_values);
    let acc_lo = vmull_u8(vget_low_u8(v), vget_low_u8(a_values));
    let hi = neon_div_by_255(acc_hi);
    let lo = neon_div_by_255(acc_lo);
    vcombine_u8(lo, hi)
}
