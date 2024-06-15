/*
 * // Copyright (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

use crate::avx2::avx2_utils::shuffle;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline(always)]
pub unsafe fn avx2_rgb_to_ycbcr(
    r: __m256i,
    g: __m256i,
    b: __m256i,
    bias: __m256i,
    coeff_r: __m256i,
    coeff_g: __m256i,
    coeff_b: __m256i,
) -> __m256i {
    let r_l = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(r));
    let g_l = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(g));
    let b_l = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(b));

    let vl = _mm256_srai_epi32::<8>(_mm256_add_epi32(
        bias,
        _mm256_add_epi32(
            _mm256_add_epi32(
                _mm256_mullo_epi32(coeff_r, r_l),
                _mm256_mullo_epi32(coeff_g, g_l),
            ),
            _mm256_mullo_epi32(coeff_b, b_l),
        ),
    ));

    let r_h = _mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(r));
    let g_h = _mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(g));
    let b_h = _mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(b));

    let vh = _mm256_srai_epi32::<8>(_mm256_add_epi32(
        bias,
        _mm256_add_epi32(
            _mm256_add_epi32(
                _mm256_mullo_epi32(coeff_r, r_h),
                _mm256_mullo_epi32(coeff_g, g_h),
            ),
            _mm256_mullo_epi32(coeff_b, b_h),
        ),
    ));

    let k = _mm256_packus_epi32(vl, vh);
    const MASK: i32 = shuffle(3, 1, 2, 0);
    _mm256_permute4x64_epi64::<MASK>(k)
}
