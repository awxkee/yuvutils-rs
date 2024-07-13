/*
 * // Copyright (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn avx512_rgb_to_ycgco(
    r: __m512i,
    g: __m512i,
    b: __m512i,
    y_reduction: __m512i,
    uv_reduction: __m512i,
    y_bias: __m512i,
    uv_bias: __m512i,
) -> (__m512i, __m512i, __m512i) {
    let mut r_l = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(r));
    let mut g_l = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(g));
    let mut b_l = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(b));

    let hg_0 = _mm512_srai_epi32::<1>(_mm512_mullo_epi32(g_l, y_reduction));

    let yl_0 = _mm512_srai_epi32::<8>(_mm512_add_epi32(
        _mm512_add_epi32(
            _mm512_srai_epi32::<2>(_mm512_add_epi32(
                _mm512_mullo_epi32(r_l, y_reduction),
                _mm512_mullo_epi32(b_l, y_reduction),
            )),
            hg_0,
        ),
        y_bias,
    ));

    r_l = _mm512_mullo_epi32(r_l, uv_reduction);
    g_l = _mm512_mullo_epi32(g_l, uv_reduction);
    b_l = _mm512_mullo_epi32(b_l, uv_reduction);

    let cg_l = _mm512_srai_epi32::<8>(_mm512_add_epi32(
        _mm512_sub_epi32(
            _mm512_srai_epi32::<1>(g_l),
            _mm512_srai_epi32::<2>(_mm512_add_epi32(r_l, b_l)),
        ),
        uv_bias,
    ));

    let co_l = _mm512_srai_epi32::<8>(_mm512_add_epi32(
        _mm512_srai_epi32::<1>(_mm512_sub_epi32(r_l, b_l)),
        uv_bias,
    ));

    let mut r_h = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(r));
    let mut g_h = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(g));
    let mut b_h = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(b));

    let hg_1 = _mm512_srai_epi32::<1>(_mm512_mullo_epi32(g_h, y_reduction));

    let yh_0 = _mm512_srai_epi32::<8>(_mm512_add_epi32(
        _mm512_add_epi32(
            _mm512_srai_epi32::<2>(_mm512_add_epi32(
                _mm512_mullo_epi32(r_h, y_reduction),
                _mm512_mullo_epi32(b_h, y_reduction),
            )),
            hg_1,
        ),
        y_bias,
    ));

    r_h = _mm512_mullo_epi32(r_h, uv_reduction);
    g_h = _mm512_mullo_epi32(g_h, uv_reduction);
    b_h = _mm512_mullo_epi32(b_h, uv_reduction);

    let cg_h = _mm512_srai_epi32::<8>(_mm512_add_epi32(
        _mm512_sub_epi32(
            _mm512_srai_epi32::<1>(g_h),
            _mm512_srai_epi32::<2>(_mm512_add_epi32(r_h, b_h)),
        ),
        uv_bias,
    ));

    let co_h = _mm512_srai_epi32::<8>(_mm512_add_epi32(
        _mm512_srai_epi32::<1>(_mm512_sub_epi32(r_h, b_h)),
        uv_bias,
    ));

    let mask = _mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7);
    (
        _mm512_permutexvar_epi64(mask, _mm512_packus_epi32(yl_0, yh_0)),
        _mm512_permutexvar_epi64(mask, _mm512_packus_epi32(cg_l, cg_h)),
        _mm512_permutexvar_epi64(mask, _mm512_packus_epi32(co_l, co_h)),
    )
}
