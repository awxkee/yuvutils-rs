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

#[inline(always)]
pub(crate) unsafe fn neon_rgb_to_ycgco(
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
