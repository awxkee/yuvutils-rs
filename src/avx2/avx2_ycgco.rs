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
pub unsafe fn avx2_rgb_to_ycgco(
    r: __m256i,
    g: __m256i,
    b: __m256i,
    y_reduction: __m256i,
    uv_reduction: __m256i,
    y_bias: __m256i,
    uv_bias: __m256i,
) -> (__m256i, __m256i, __m256i) {
    let mut r_l = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(r));
    let mut g_l = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(g));
    let mut b_l = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(b));

    let hg_0 = _mm256_srai_epi32::<1>(_mm256_madd_epi16(g_l, y_reduction));

    let yl_0 = _mm256_srai_epi32::<8>(_mm256_add_epi32(
        _mm256_add_epi32(
            _mm256_srai_epi32::<2>(_mm256_add_epi32(
                _mm256_madd_epi16(r_l, y_reduction),
                _mm256_madd_epi16(b_l, y_reduction),
            )),
            hg_0,
        ),
        y_bias,
    ));

    r_l = _mm256_madd_epi16(r_l, uv_reduction);
    g_l = _mm256_madd_epi16(g_l, uv_reduction);
    b_l = _mm256_madd_epi16(b_l, uv_reduction);

    let cg_l = _mm256_srai_epi32::<8>(_mm256_add_epi32(
        _mm256_sub_epi32(
            _mm256_srai_epi32::<1>(g_l),
            _mm256_srai_epi32::<2>(_mm256_add_epi32(r_l, b_l)),
        ),
        uv_bias,
    ));

    let co_l = _mm256_srai_epi32::<8>(_mm256_add_epi32(
        _mm256_srai_epi32::<1>(_mm256_sub_epi32(r_l, b_l)),
        uv_bias,
    ));

    let mut r_h = _mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(r));
    let mut g_h = _mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(g));
    let mut b_h = _mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(b));

    let hg_1 = _mm256_srai_epi32::<1>(_mm256_madd_epi16(g_h, y_reduction));

    let yh_0 = _mm256_srai_epi32::<8>(_mm256_add_epi32(
        _mm256_add_epi32(
            _mm256_srai_epi32::<2>(_mm256_add_epi32(
                _mm256_madd_epi16(r_h, y_reduction),
                _mm256_madd_epi16(b_h, y_reduction),
            )),
            hg_1,
        ),
        y_bias,
    ));

    r_h = _mm256_madd_epi16(r_h, uv_reduction);
    g_h = _mm256_madd_epi16(g_h, uv_reduction);
    b_h = _mm256_madd_epi16(b_h, uv_reduction);

    let cg_h = _mm256_srai_epi32::<8>(_mm256_add_epi32(
        _mm256_sub_epi32(
            _mm256_srai_epi32::<1>(g_h),
            _mm256_srai_epi32::<2>(_mm256_add_epi32(r_h, b_h)),
        ),
        uv_bias,
    ));

    let co_h = _mm256_srai_epi32::<8>(_mm256_add_epi32(
        _mm256_srai_epi32::<1>(_mm256_sub_epi32(r_h, b_h)),
        uv_bias,
    ));

    const MASK: i32 = shuffle(3, 1, 2, 0);
    (
        _mm256_permute4x64_epi64::<MASK>(_mm256_packus_epi32(yl_0, yh_0)),
        _mm256_permute4x64_epi64::<MASK>(_mm256_packus_epi32(cg_l, cg_h)),
        _mm256_permute4x64_epi64::<MASK>(_mm256_packus_epi32(co_l, co_h)),
    )
}
