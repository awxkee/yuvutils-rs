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

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[inline]
pub unsafe fn sse_rgb_to_ycbcr(
    r: __m128i,
    g: __m128i,
    b: __m128i,
    bias: __m128i,
    coeff_r: __m128i,
    coeff_g: __m128i,
    coeff_b: __m128i,
) -> __m128i {
    let zeros_si = _mm_setzero_si128();
    let r_l = _mm_unpacklo_epi16(r, zeros_si);
    let g_l = _mm_unpacklo_epi16(g, zeros_si);
    let b_l = _mm_unpacklo_epi16(b, zeros_si);

    let vl = _mm_srai_epi32::<8>(_mm_add_epi32(
        bias,
        _mm_add_epi32(
            _mm_add_epi32(_mm_madd_epi16(coeff_r, r_l), _mm_madd_epi16(coeff_g, g_l)),
            _mm_madd_epi16(coeff_b, b_l),
        ),
    ));

    let r_h = _mm_unpackhi_epi16(r, zeros_si);
    let g_h = _mm_unpackhi_epi16(g, zeros_si);
    let b_h = _mm_unpackhi_epi16(b, zeros_si);

    let vh = _mm_srai_epi32::<8>(_mm_add_epi32(
        bias,
        _mm_add_epi32(
            _mm_add_epi32(_mm_madd_epi16(coeff_r, r_h), _mm_madd_epi16(coeff_g, g_h)),
            _mm_madd_epi16(coeff_b, b_h),
        ),
    ));

    _mm_packus_epi32(vl, vh)
}

#[inline]
pub unsafe fn sse_rgb_to_ycgco(
    r: __m128i,
    g: __m128i,
    b: __m128i,
    y_reduction: __m128i,
    uv_reduction: __m128i,
    y_bias: __m128i,
    uv_bias: __m128i,
) -> (__m128i, __m128i, __m128i) {
    let zeros_si = _mm_setzero_si128();
    let mut r_l = _mm_unpacklo_epi16(r, zeros_si);
    let mut g_l = _mm_unpacklo_epi16(g, zeros_si);
    let mut b_l = _mm_unpacklo_epi16(b, zeros_si);

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

    let mut r_h = _mm_unpackhi_epi16(r, zeros_si);
    let mut g_h = _mm_unpackhi_epi16(g, zeros_si);
    let mut b_h = _mm_unpackhi_epi16(b, zeros_si);

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
