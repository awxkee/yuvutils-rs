/*
 * // Copyright (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

use std::arch::aarch64::*;
#[inline(always)]
pub unsafe fn neon_rgb_to_ycgco_r(
    r: int16x8_t,
    g: int16x8_t,
    b: int16x8_t,
    y_reduction: int16x4_t,
    y_bias: int32x4_t,
) -> (uint16x8_t, uint16x8_t, uint16x8_t) {
    let r_l = vmull_s16(vget_low_s16(r), y_reduction);
    let g_l = vmull_s16(vget_low_s16(g), y_reduction);
    let b_l = vmull_s16(vget_low_s16(b), y_reduction);

    let co_l = vsubq_s32(r_l, b_l);
    let t_l = vaddq_s32(b_l, vshrq_n_s32::<1>(co_l));
    let cg_l = vsubq_s32(g_l, t_l);
    let y_l = vaddq_s32(vaddq_s32(vshrq_n_s32::<1>(cg_l), t_l), y_bias);
    let co_l = vaddq_s32(co_l, y_bias);
    let cg_l = vaddq_s32(cg_l, y_bias);

    let r_h = vmull_s16(vget_low_s16(r), y_reduction);
    let g_h = vmull_s16(vget_low_s16(g), y_reduction);
    let b_h = vmull_s16(vget_low_s16(b), y_reduction);

    let co_h = vsubq_s32(r_h, b_h);
    let t_h = vaddq_s32(b_h, vshrq_n_s32::<1>(co_h));
    let cg_h = vsubq_s32(g_h, t_h);
    let y_h = vaddq_s32(vaddq_s32(vshrq_n_s32::<1>(cg_h), t_h), y_bias);
    let co_h = vaddq_s32(co_h, y_bias);
    let cg_h = vaddq_s32(cg_h, y_bias);

    (
        vcombine_u16(vqshrun_n_s32::<8>(y_l), vqshrun_n_s32::<8>(y_h)),
        vcombine_u16(vqshrun_n_s32::<8>(cg_l), vqshrun_n_s32::<8>(cg_h)),
        vcombine_u16(vqshrun_n_s32::<8>(co_l), vqshrun_n_s32::<8>(co_h)),
    )
}
