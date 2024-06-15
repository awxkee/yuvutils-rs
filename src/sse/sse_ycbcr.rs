/*
 * // Copyright (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

use crate::sse::sse_support::sse_promote_i16_toi32;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline(always)]
#[allow(dead_code)]
pub unsafe fn sse_rgb_to_ycbcr(
    r: __m128i,
    g: __m128i,
    b: __m128i,
    bias: __m128i,
    coeff_r: __m128i,
    coeff_g: __m128i,
    coeff_b: __m128i,
) -> __m128i {
    let r_l = _mm_cvtepi16_epi32(r);
    let g_l = _mm_cvtepi16_epi32(g);
    let b_l = _mm_cvtepi16_epi32(b);

    let vl = _mm_srai_epi32::<8>(_mm_add_epi32(
        bias,
        _mm_add_epi32(
            _mm_add_epi32(_mm_mullo_epi32(coeff_r, r_l), _mm_mullo_epi32(coeff_g, g_l)),
            _mm_mullo_epi32(coeff_b, b_l),
        ),
    ));

    let r_h = sse_promote_i16_toi32(r);
    let g_h = sse_promote_i16_toi32(g);
    let b_h = sse_promote_i16_toi32(b);

    let vh = _mm_srai_epi32::<8>(_mm_add_epi32(
        bias,
        _mm_add_epi32(
            _mm_add_epi32(_mm_mullo_epi32(coeff_r, r_h), _mm_mullo_epi32(coeff_g, g_h)),
            _mm_mullo_epi32(coeff_b, b_h),
        ),
    ));

    _mm_packus_epi32(vl, vh)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline(always)]
#[allow(dead_code)]
pub unsafe fn sse_rgb_to_ycgco(
    r: __m128i,
    g: __m128i,
    b: __m128i,
    y_reduction: __m128i,
    uv_reduction: __m128i,
    y_bias: __m128i,
    uv_bias: __m128i,
) -> (__m128i, __m128i, __m128i) {
    let mut r_l = _mm_cvtepi16_epi32(r);
    let mut g_l = _mm_cvtepi16_epi32(g);
    let mut b_l = _mm_cvtepi16_epi32(b);

    let hg_0 = _mm_srai_epi32::<1>(_mm_mullo_epi32(g_l, y_reduction));

    let yl_0 = _mm_srai_epi32::<8>(_mm_add_epi32(
        _mm_add_epi32(
            _mm_srai_epi32::<2>(_mm_add_epi32(
                _mm_mullo_epi32(r_l, y_reduction),
                _mm_mullo_epi32(b_l, y_reduction),
            )),
            hg_0,
        ),
        y_bias,
    ));

    r_l = _mm_mullo_epi32(r_l, uv_reduction);
    g_l = _mm_mullo_epi32(g_l, uv_reduction);
    b_l = _mm_mullo_epi32(b_l, uv_reduction);

    let cg_l = _mm_srai_epi32::<8>(_mm_add_epi32(
        _mm_sub_epi32(
            _mm_srai_epi32::<1>(g_l),
            _mm_srai_epi32::<2>(_mm_add_epi32(r_l, b_l)),
        ),
        uv_bias,
    ));

    let co_l = _mm_srai_epi32::<8>(_mm_add_epi32(
        _mm_srai_epi32::<1>(_mm_sub_epi32(r_l, b_l)),
        uv_bias,
    ));

    let mut r_h = sse_promote_i16_toi32(r);
    let mut g_h = sse_promote_i16_toi32(g);
    let mut b_h = sse_promote_i16_toi32(b);

    let hg_1 = _mm_srai_epi32::<1>(_mm_mullo_epi32(g_h, y_reduction));

    let yh_0 = _mm_srai_epi32::<8>(_mm_add_epi32(
        _mm_add_epi32(
            _mm_srai_epi32::<2>(_mm_add_epi32(
                _mm_mullo_epi32(r_h, y_reduction),
                _mm_mullo_epi32(b_h, y_reduction),
            )),
            hg_1,
        ),
        y_bias,
    ));

    r_h = _mm_mullo_epi32(r_h, uv_reduction);
    g_h = _mm_mullo_epi32(g_h, uv_reduction);
    b_h = _mm_mullo_epi32(b_h, uv_reduction);

    let cg_h = _mm_srai_epi32::<8>(_mm_add_epi32(
        _mm_sub_epi32(
            _mm_srai_epi32::<1>(g_h),
            _mm_srai_epi32::<2>(_mm_add_epi32(r_h, b_h)),
        ),
        uv_bias,
    ));

    let co_h = _mm_srai_epi32::<8>(_mm_add_epi32(
        _mm_srai_epi32::<1>(_mm_sub_epi32(r_h, b_h)),
        uv_bias,
    ));

    (
        _mm_packus_epi32(yl_0, yh_0),
        _mm_packus_epi32(cg_l, cg_h),
        _mm_packus_epi32(co_l, co_h),
    )
}
