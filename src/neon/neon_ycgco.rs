/*
 * // Copyright (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

use std::arch::aarch64::*;

#[inline(always)]
pub unsafe fn neon_rgb_to_ycgco(
    r: int16x8_t,
    g: int16x8_t,
    b: int16x8_t,
    y_reduction: int16x8_t,
    uv_reduction: int16x8_t,
    y_bias: int32x4_t,
    uv_bias: int32x4_t,
) -> (uint16x8_t, uint16x8_t, uint16x8_t) {
    let r_l = vget_low_s16(r);
    let g_l = vget_low_s16(g);
    let b_l = vget_low_s16(b);

    let low_y_reduction = vget_low_s16(y_reduction);
    let hg_0 = vshrq_n_s32::<1>(vmull_s16(g_l, low_y_reduction));

    let yl_0 = vqshrun_n_s32::<8>(vaddq_s32(
        vaddq_s32(
            vshrq_n_s32::<2>(vaddq_s32(
                vmull_s16(r_l, low_y_reduction),
                vmull_s16(b_l, low_y_reduction),
            )),
            hg_0,
        ),
        y_bias,
    ));

    let low_uv_reduction = vget_low_s16(uv_reduction);

    let r_l = vmull_s16(r_l, low_uv_reduction);
    let g_l = vmull_s16(g_l, low_uv_reduction);
    let b_l = vmull_s16(b_l, low_uv_reduction);

    let cg_l = vqshrun_n_s32::<8>(vaddq_s32(
        vsubq_s32(vshrq_n_s32::<1>(g_l), vshrq_n_s32::<2>(vaddq_s32(r_l, b_l))),
        uv_bias,
    ));

    let co_l = vqshrun_n_s32::<8>(vaddq_s32(vshrq_n_s32::<1>(vsubq_s32(r_l, b_l)), uv_bias));

    let hg_1 = vshrq_n_s32::<1>(vmull_high_s16(g, y_reduction));

    let yh_0 = vqshrun_n_s32::<8>(vaddq_s32(
        vaddq_s32(
            vshrq_n_s32::<2>(vaddq_s32(
                vmull_high_s16(r, y_reduction),
                vmull_high_s16(b, y_reduction),
            )),
            hg_1,
        ),
        y_bias,
    ));

    let r_h = vmull_high_s16(r, uv_reduction);
    let g_h = vmull_high_s16(g, uv_reduction);
    let b_h = vmull_high_s16(b, uv_reduction);

    let cg_h = vqshrun_n_s32::<8>(vaddq_s32(
        vsubq_s32(vshrq_n_s32::<1>(g_h), vshrq_n_s32::<2>(vaddq_s32(r_h, b_h))),
        uv_bias,
    ));

    let co_h = vqshrun_n_s32::<8>(vaddq_s32(vshrq_n_s32::<1>(vsubq_s32(r_h, b_h)), uv_bias));

    (
        vcombine_u16(yl_0, yh_0),
        vcombine_u16(cg_l, cg_h),
        vcombine_u16(co_l, co_h),
    )
}
