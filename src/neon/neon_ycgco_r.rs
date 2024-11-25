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
pub(crate) unsafe fn neon_rgb_to_ycgco_r(
    r: int16x8_t,
    g: int16x8_t,
    b: int16x8_t,
    y_reduction: int32x4_t,
    uv_reduction: int32x4_t,
    y_bias: int32x4_t,
    uv_bias: int32x4_t,
) -> (uint16x8_t, uint16x8_t, uint16x8_t) {
    let r_l = vmovl_s16(vget_low_s16(r));
    let g_l = vmovl_s16(vget_low_s16(g));
    let b_l = vmovl_s16(vget_low_s16(b));

    let co_l = vsubq_s32(r_l, b_l);
    let t_l = vaddq_s32(b_l, vshrq_n_s32::<1>(co_l));
    let cg_l = vsubq_s32(g_l, t_l);
    let y_l = vaddq_s32(
        vmulq_s32(vaddq_s32(vshrq_n_s32::<1>(cg_l), t_l), y_reduction),
        y_bias,
    );
    let co_l = vaddq_s32(vmulq_s32(co_l, uv_reduction), uv_bias);
    let cg_l = vaddq_s32(vmulq_s32(cg_l, uv_reduction), uv_bias);

    let r_h = vmovl_high_s16(r);
    let g_h = vmovl_high_s16(g);
    let b_h = vmovl_high_s16(b);

    let co_h = vsubq_s32(r_h, b_h);
    let t_h = vaddq_s32(b_h, vshrq_n_s32::<1>(co_h));
    let cg_h = vsubq_s32(g_h, t_h);
    let y_h = vaddq_s32(
        vmulq_s32(vaddq_s32(vshrq_n_s32::<1>(cg_h), t_h), y_reduction),
        y_bias,
    );
    let co_h = vaddq_s32(vmulq_s32(co_h, uv_reduction), uv_bias);
    let cg_h = vaddq_s32(vmulq_s32(cg_h, uv_reduction), uv_bias);

    (
        vcombine_u16(vqshrun_n_s32::<8>(y_l), vqshrun_n_s32::<8>(y_h)),
        vcombine_u16(vqshrun_n_s32::<8>(cg_l), vqshrun_n_s32::<8>(cg_h)),
        vcombine_u16(vqshrun_n_s32::<8>(co_l), vqshrun_n_s32::<8>(co_h)),
    )
}

#[inline(always)]
pub(crate) unsafe fn neon_ycgco_r_to_rgb(
    y: int16x8_t,
    cg: int16x8_t,
    co: int16x8_t,
    y_reduction: int16x8_t,
    uv_reduction: int16x8_t,
    y_bias: int16x8_t,
    uv_bias: int16x8_t,
) -> (uint8x8_t, uint8x8_t, uint8x8_t) {
    let y = vsubq_s16(y, y_bias);
    let cg = vsubq_s16(cg, uv_bias);
    let co = vsubq_s16(co, uv_bias);
    let y_l = vmulq_s16(y, y_reduction);
    let cg_l = vmulq_s16(cg, uv_reduction);
    let co_l = vmulq_s16(co, uv_reduction);

    let t_l = vqsubq_s16(y_l, vshrq_n_s16::<1>(cg_l));
    let g = vqrshrun_n_s16::<6>(vqaddq_s16(t_l, cg_l));
    let b = vqsubq_s16(t_l, vshrq_n_s16::<1>(co_l));
    let r = vqrshrun_n_s16::<6>(vqaddq_s16(b, co_l));
    let b = vqrshrun_n_s16::<6>(b);

    (r, g, b)
}
