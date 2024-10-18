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

use crate::avx2::avx2_utils::shuffle;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[inline]
#[target_feature(enable = "avx2")]
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
                _mm256_madd_epi16(coeff_r, r_l),
                _mm256_madd_epi16(coeff_g, g_l),
            ),
            _mm256_madd_epi16(coeff_b, b_l),
        ),
    ));

    let r_h = _mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(r));
    let g_h = _mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(g));
    let b_h = _mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(b));

    let vh = _mm256_srai_epi32::<8>(_mm256_add_epi32(
        bias,
        _mm256_add_epi32(
            _mm256_add_epi32(
                _mm256_madd_epi16(coeff_r, r_h),
                _mm256_madd_epi16(coeff_g, g_h),
            ),
            _mm256_madd_epi16(coeff_b, b_h),
        ),
    ));

    let k = _mm256_packus_epi32(vl, vh);
    const MASK: i32 = shuffle(3, 1, 2, 0);
    _mm256_permute4x64_epi64::<MASK>(k)
}
