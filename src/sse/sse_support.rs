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

#[inline]
#[target_feature(enable = "sse4.1")]
pub unsafe fn sse_interleave_even(x: __m128i) -> __m128i {
    #[rustfmt::skip]
    let shuffle = _mm_setr_epi8(0, 0, 2, 2, 4, 4, 6, 6,
                                     8, 8, 10, 10, 12, 12, 14, 14);
    _mm_shuffle_epi8(x, shuffle)
}

#[inline]
#[target_feature(enable = "sse4.1")]
pub unsafe fn sse_interleave_odd(x: __m128i) -> __m128i {
    let shuffle = _mm_setr_epi8(1, 1, 3, 3, 5, 5, 7, 7, 9, 9, 11, 11, 13, 13, 15, 15);
    _mm_shuffle_epi8(x, shuffle)
}

#[inline]
#[target_feature(enable = "sse4.1")]
pub unsafe fn sse_interleave_rgba(
    r: __m128i,
    g: __m128i,
    b: __m128i,
    a: __m128i,
) -> (__m128i, __m128i, __m128i, __m128i) {
    let rg_lo = _mm_unpacklo_epi8(r, g);
    let rg_hi = _mm_unpackhi_epi8(r, g);
    let ba_lo = _mm_unpacklo_epi8(b, a);
    let ba_hi = _mm_unpackhi_epi8(b, a);

    let rgba_0_lo = _mm_unpacklo_epi16(rg_lo, ba_lo);
    let rgba_0_hi = _mm_unpackhi_epi16(rg_lo, ba_lo);
    let rgba_1_lo = _mm_unpacklo_epi16(rg_hi, ba_hi);
    let rgba_1_hi = _mm_unpackhi_epi16(rg_hi, ba_hi);
    (rgba_0_lo, rgba_0_hi, rgba_1_lo, rgba_1_hi)
}

#[inline]
#[target_feature(enable = "sse4.1")]
pub unsafe fn sse_store_rgba(ptr: *mut u8, r: __m128i, g: __m128i, b: __m128i, a: __m128i) {
    let (row1, row2, row3, row4) = sse_interleave_rgba(r, g, b, a);
    _mm_storeu_si128(ptr as *mut __m128i, row1);
    _mm_storeu_si128(ptr.add(16) as *mut __m128i, row2);
    _mm_storeu_si128(ptr.add(32) as *mut __m128i, row3);
    _mm_storeu_si128(ptr.add(48) as *mut __m128i, row4);
}

#[inline]
#[target_feature(enable = "sse4.1")]
pub unsafe fn sse_store_rgba_half_epi8(
    ptr: *mut u8,
    r: __m128i,
    g: __m128i,
    b: __m128i,
    a: __m128i,
) {
    let (row1, row2, _, _) = sse_interleave_rgba(r, g, b, a);
    _mm_storeu_si128(ptr as *mut __m128i, row1);
    _mm_storeu_si128(ptr.add(16) as *mut __m128i, row2);
}

#[inline]
#[target_feature(enable = "sse4.1")]
pub unsafe fn sse_deinterleave_rgba(
    rgba0: __m128i,
    rgba1: __m128i,
    rgba2: __m128i,
    rgba3: __m128i,
) -> (__m128i, __m128i, __m128i, __m128i) {
    let t0 = _mm_unpacklo_epi8(rgba0, rgba1); // r1 R1 g1 G1 b1 B1 a1 A1 r2 R2 g2 G2 b2 B2 a2 A2
    let t1 = _mm_unpackhi_epi8(rgba0, rgba1);
    let t2 = _mm_unpacklo_epi8(rgba2, rgba3); // r4 R4 g4 G4 b4 B4 a4 A4 r5 R5 g5 G5 b5 B5 a5 A5
    let t3 = _mm_unpackhi_epi8(rgba2, rgba3);

    let t4 = _mm_unpacklo_epi16(t0, t2); // r1 R1 r4 R4 g1 G1 G4 g4 G4 b1 B1 b4 B4 a1 A1 a4 A4
    let t5 = _mm_unpackhi_epi16(t0, t2);
    let t6 = _mm_unpacklo_epi16(t1, t3);
    let t7 = _mm_unpackhi_epi16(t1, t3);

    let l1 = _mm_unpacklo_epi32(t4, t6); // r1 R1 r4 R4 g1 G1 G4 g4 G4 b1 B1 b4 B4 a1 A1 a4 A4
    let l2 = _mm_unpackhi_epi32(t4, t6);
    let l3 = _mm_unpacklo_epi32(t5, t7);
    let l4 = _mm_unpackhi_epi32(t5, t7);

    #[rustfmt::skip]
    let shuffle = _mm_setr_epi8(0, 4, 8, 12,
                                        1, 5, 9, 13,
                                        2, 6, 10, 14,
                                        3, 7, 11, 15,
    );

    let r1 = _mm_shuffle_epi8(_mm_unpacklo_epi32(l1, l3), shuffle);
    let r2 = _mm_shuffle_epi8(_mm_unpackhi_epi32(l1, l3), shuffle);
    let r3 = _mm_shuffle_epi8(_mm_unpacklo_epi32(l2, l4), shuffle);
    let r4 = _mm_shuffle_epi8(_mm_unpackhi_epi32(l2, l4), shuffle);

    (r1, r2, r3, r4)
}

#[inline]
#[target_feature(enable = "sse4.1")]
pub unsafe fn sse_deinterleave_rgb(
    rgb0: __m128i,
    rgb1: __m128i,
    rgb2: __m128i,
) -> (__m128i, __m128i, __m128i) {
    #[rustfmt::skip]
    let idx = _mm_setr_epi8(0, 3, 6, 9,
                                    12, 15, 2, 5, 8,
                                    11, 14, 1, 4, 7,
                                    10, 13);

    let r6b5g5_0 = _mm_shuffle_epi8(rgb0, idx);
    let g6r5b5_1 = _mm_shuffle_epi8(rgb1, idx);
    let b6g5r5_2 = _mm_shuffle_epi8(rgb2, idx);

    #[rustfmt::skip]
    let mask010 = _mm_setr_epi8(0, 0, 0, 0,
                                        0, 0, -1, -1, -1,
                                        -1, -1, 0, 0, 0,
                                        0, 0);

    #[rustfmt::skip]
    let mask001 = _mm_setr_epi8(0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0,
                                    -1, -1, -1, -1, -1);

    let b2g2b1 = _mm_blendv_epi8(b6g5r5_2, g6r5b5_1, mask001);
    let b2b0b1 = _mm_blendv_epi8(b2g2b1, r6b5g5_0, mask010);

    let r0r1b1 = _mm_blendv_epi8(r6b5g5_0, g6r5b5_1, mask010);
    let r0r1r2 = _mm_blendv_epi8(r0r1b1, b6g5r5_2, mask001);

    let g1r1g0 = _mm_blendv_epi8(g6r5b5_1, r6b5g5_0, mask001);
    let g1g2g0 = _mm_blendv_epi8(g1r1g0, b6g5r5_2, mask010);

    let g0g1g2 = _mm_alignr_epi8::<11>(g1g2g0, g1g2g0);
    let b0b1b2 = _mm_alignr_epi8::<6>(b2b0b1, b2b0b1);

    (r0r1r2, g0g1g2, b0b1b2)
}

#[inline]
#[target_feature(enable = "sse4.1")]
pub unsafe fn sse_interleave_rgb(
    r: __m128i,
    g: __m128i,
    b: __m128i,
) -> (__m128i, __m128i, __m128i) {
    let sh_a = _mm_setr_epi8(0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14, 9, 4, 15, 10, 5);
    let sh_b = _mm_setr_epi8(5, 0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14, 9, 4, 15, 10);
    let sh_c = _mm_setr_epi8(10, 5, 0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14, 9, 4, 15);
    let a0 = _mm_shuffle_epi8(r, sh_a);
    let b0 = _mm_shuffle_epi8(g, sh_b);
    let c0 = _mm_shuffle_epi8(b, sh_c);

    let m0 = _mm_setr_epi8(0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0);
    let m1 = _mm_setr_epi8(0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0);
    let v0 = _mm_blendv_epi8(_mm_blendv_epi8(a0, b0, m1), c0, m0);
    let v1 = _mm_blendv_epi8(_mm_blendv_epi8(b0, c0, m1), a0, m0);
    let v2 = _mm_blendv_epi8(_mm_blendv_epi8(c0, a0, m1), b0, m0);
    (v0, v1, v2)
}

#[inline]
#[target_feature(enable = "sse4.1")]
pub unsafe fn _mm_deinterleave_x2_epi8(a: __m128i, b: __m128i) -> (__m128i, __m128i) {
    let t10 = _mm_unpacklo_epi8(a, b);
    let t11 = _mm_unpackhi_epi8(a, b);

    let t20 = _mm_unpacklo_epi8(t10, t11);
    let t21 = _mm_unpackhi_epi8(t10, t11);

    let t30 = _mm_unpacklo_epi8(t20, t21);
    let t31 = _mm_unpackhi_epi8(t20, t21);

    let av = _mm_unpacklo_epi8(t30, t31);
    let bv = _mm_unpackhi_epi8(t30, t31);
    (av, bv)
}

#[inline]
#[target_feature(enable = "sse4.1")]
pub unsafe fn _mm_deinterleave_x2_epi16(a: __m128i, b: __m128i) -> (__m128i, __m128i) {
    let v2 = _mm_unpacklo_epi16(a, b); // a0 a4 b0 b4 a1 a5 b1 b5
    let v3 = _mm_unpackhi_epi16(a, b); // a2 a6 b2 b6 a3 a7 b3 b7
    let v4 = _mm_unpacklo_epi16(v2, v3); // a0 a2 a4 a6 b0 b2 b4 b6
    let v5 = _mm_unpackhi_epi16(v2, v3); // a1 a3 a5 a7 b1 b3 b5 b7

    let av = _mm_unpacklo_epi16(v4, v5); // a0 a1 a2 a3 a4 a5 a6 a7
    let bv = _mm_unpackhi_epi16(v4, v5); // b0 b1 ab b3 b4 b5 b6 b7
    (av, bv)
}

#[inline]
#[target_feature(enable = "sse4.1")]
pub unsafe fn sse_store_rgb_u8(ptr: *mut u8, r: __m128i, g: __m128i, b: __m128i) {
    let (v0, v1, v2) = sse_interleave_rgb(r, g, b);
    _mm_storeu_si128(ptr as *mut __m128i, v0);
    _mm_storeu_si128(ptr.add(16) as *mut __m128i, v1);
    _mm_storeu_si128(ptr.add(32) as *mut __m128i, v2);
}

#[inline]
#[target_feature(enable = "sse4.1")]
pub unsafe fn sse_store_rgb_half_u8(ptr: *mut u8, r: __m128i, g: __m128i, b: __m128i) {
    let (v0, v1, _) = sse_interleave_rgb(r, g, b);
    _mm_storeu_si128(ptr as *mut __m128i, v0);
    std::ptr::copy_nonoverlapping(&v1 as *const _ as *const u8, ptr.add(16), 8);
}

#[inline]
#[target_feature(enable = "sse4.1")]
pub unsafe fn sse_pairwise_widen_avg(v: __m128i) -> __m128i {
    let sums = _mm_maddubs_epi16(v, _mm_set1_epi8(1));
    let shifted = _mm_srli_epi16::<1>(_mm_add_epi16(sums, _mm_set1_epi16(1)));
    _mm_packus_epi16(shifted, shifted)
}

#[inline]
#[target_feature(enable = "sse4.1")]
pub unsafe fn sse_div_by255(v: __m128i) -> __m128i {
    let addition = _mm_set1_epi16(127);
    _mm_srli_epi16::<8>(_mm_add_epi16(
        _mm_add_epi16(v, addition),
        _mm_srli_epi16::<8>(v),
    ))
}

#[inline(always)]
pub unsafe fn sse_pairwise_avg_epi16(a: __m128i, b: __m128i) -> __m128i {
    let sums = _mm_hadd_epi16(a, b);
    _mm_srli_epi16::<1>(_mm_add_epi16(sums, _mm_set1_epi16(1)))
}

#[inline(always)]
pub unsafe fn sse_avg_epi16(a: __m128i) -> __m128i {
    let sums = _mm_madd_epi16(a, _mm_set1_epi16(1));
    let shifted = _mm_srli_epi32::<1>(_mm_add_epi32(sums, _mm_set1_epi32(1)));
    _mm_packus_epi32(shifted, shifted)
}

#[inline(always)]
pub unsafe fn _mm_loadu_si128_x2(ptr: *const u8) -> (__m128i, __m128i) {
    (
        _mm_loadu_si128(ptr as *const __m128i),
        _mm_loadu_si128(ptr.add(16) as *const __m128i),
    )
}

#[allow(non_camel_case_types)]
#[derive(Copy, Clone)]
pub struct __mm128x4(pub __m128i, pub __m128i, pub __m128i, pub __m128i);

#[inline(always)]
pub unsafe fn _mm_combinel_epi8(a: __m128i, b: __m128i) -> __m128i {
    let a_low = _mm_castps_si128(_mm_movelh_ps(_mm_castsi128_ps(a), _mm_castsi128_ps(a)));
    let b_low = _mm_castps_si128(_mm_movelh_ps(_mm_castsi128_ps(b), _mm_castsi128_ps(b)));

    _mm_unpacklo_epi64(a_low, b_low)
}

#[inline(always)]
pub unsafe fn _mm_combineh_epi8(a: __m128i, b: __m128i) -> __m128i {
    let a_low = _mm_castps_si128(_mm_movehl_ps(_mm_castsi128_ps(a), _mm_castsi128_ps(a)));
    let b_low = _mm_castps_si128(_mm_movehl_ps(_mm_castsi128_ps(b), _mm_castsi128_ps(b)));

    _mm_unpacklo_epi64(a_low, b_low)
}

#[inline]
#[target_feature(enable = "sse4.1")]
pub unsafe fn _mm_storeu_si128_x4(ptr: *mut u8, vals: __mm128x4) {
    _mm_storeu_si128(ptr as *mut __m128i, vals.0);
    _mm_storeu_si128(ptr.add(16) as *mut __m128i, vals.1);
    _mm_storeu_si128(ptr.add(32) as *mut __m128i, vals.2);
    _mm_storeu_si128(ptr.add(48) as *mut __m128i, vals.3);
}

#[inline]
#[target_feature(enable = "sse4.1")]
pub unsafe fn _mm_getlow_epi8(a: __m128i) -> __m128i {
    _mm_castps_si128(_mm_movelh_ps(
        _mm_castsi128_ps(a),
        _mm_castsi128_ps(_mm_setzero_si128()),
    ))
}

#[inline]
#[target_feature(enable = "sse4.1")]
pub unsafe fn _mm_gethigh_epi8(a: __m128i) -> __m128i {
    _mm_srli_si128::<8>(a)
}

#[inline]
#[target_feature(enable = "sse4.1")]
pub unsafe fn _mm_deinterleave_rgba_epi16(
    rgba0: __m128i,
    rgba1: __m128i,
    rgba2: __m128i,
    rgba3: __m128i,
) -> (__m128i, __m128i, __m128i, __m128i) {
    let v0 = _mm_unpacklo_epi16(rgba0, rgba2); // a0 a4 b0 b4 ...
    let v1 = _mm_unpackhi_epi16(rgba0, rgba2); // a1 a5 b1 b5 ...
    let v2 = _mm_unpacklo_epi16(rgba1, rgba3); // a2 a6 b2 b6 ...
    let v3 = _mm_unpackhi_epi16(rgba1, rgba3); // a3 a7 b3 b7 ...

    let u0 = _mm_unpacklo_epi16(v0, v2); // a0 a2 a4 a6 ...
    let u1 = _mm_unpacklo_epi16(v1, v3); // a1 a3 a5 a7 ...
    let u2 = _mm_unpackhi_epi16(v0, v2); // c0 c2 c4 c6 ...
    let u3 = _mm_unpackhi_epi16(v1, v3); // c1 c3 c5 c7 ...

    let a = _mm_unpacklo_epi16(u0, u1);
    let b = _mm_unpackhi_epi16(u0, u1);
    let c = _mm_unpacklo_epi16(u2, u3);
    let d = _mm_unpackhi_epi16(u2, u3);
    (a, b, c, d)
}

#[inline]
#[target_feature(enable = "sse4.1")]
pub unsafe fn _mm_deinterleave_rgb_epi16(
    rgba0: __m128i,
    rgba1: __m128i,
    rgba2: __m128i,
) -> (__m128i, __m128i, __m128i) {
    let a0 = _mm_blend_epi16::<0x24>(_mm_blend_epi16::<0x92>(rgba0, rgba1), rgba2);
    let b0 = _mm_blend_epi16::<0x24>(_mm_blend_epi16::<0x92>(rgba2, rgba0), rgba1);
    let c0 = _mm_blend_epi16::<0x24>(_mm_blend_epi16::<0x92>(rgba1, rgba2), rgba0);

    let sh_a = _mm_setr_epi8(0, 1, 6, 7, 12, 13, 2, 3, 8, 9, 14, 15, 4, 5, 10, 11);
    let sh_b = _mm_setr_epi8(2, 3, 8, 9, 14, 15, 4, 5, 10, 11, 0, 1, 6, 7, 12, 13);
    let sh_c = _mm_setr_epi8(4, 5, 10, 11, 0, 1, 6, 7, 12, 13, 2, 3, 8, 9, 14, 15);
    let a0 = _mm_shuffle_epi8(a0, sh_a);
    let b0 = _mm_shuffle_epi8(b0, sh_b);
    let c0 = _mm_shuffle_epi8(c0, sh_c);
    (a0, b0, c0)
}

#[inline]
#[target_feature(enable = "sse4.1")]
pub unsafe fn _mm_interleave_rgb_epi16(
    a: __m128i,
    b: __m128i,
    c: __m128i,
) -> (__m128i, __m128i, __m128i) {
    let sh_a = _mm_setr_epi8(0, 1, 6, 7, 12, 13, 2, 3, 8, 9, 14, 15, 4, 5, 10, 11);
    let sh_b = _mm_setr_epi8(10, 11, 0, 1, 6, 7, 12, 13, 2, 3, 8, 9, 14, 15, 4, 5);
    let sh_c = _mm_setr_epi8(4, 5, 10, 11, 0, 1, 6, 7, 12, 13, 2, 3, 8, 9, 14, 15);
    let a0 = _mm_shuffle_epi8(a, sh_a);
    let b0 = _mm_shuffle_epi8(b, sh_b);
    let c0 = _mm_shuffle_epi8(c, sh_c);

    let v0 = _mm_blend_epi16::<0x24>(_mm_blend_epi16::<0x92>(a0, b0), c0);
    let v1 = _mm_blend_epi16::<0x24>(_mm_blend_epi16::<0x92>(c0, a0), b0);
    let v2 = _mm_blend_epi16::<0x24>(_mm_blend_epi16::<0x92>(b0, c0), a0);
    (v0, v1, v2)
}

#[inline]
#[target_feature(enable = "sse4.1")]
pub unsafe fn _mm_interleave_rgba_epi16(
    a: __m128i,
    b: __m128i,
    c: __m128i,
    d: __m128i,
) -> (__m128i, __m128i, __m128i, __m128i) {
    // a0 a1 a2 a3 ....
    // b0 b1 b2 b3 ....
    // c0 c1 c2 c3 ....
    // d0 d1 d2 d3 ....
    let u0 = _mm_unpacklo_epi16(a, c); // a0 c0 a1 c1 ...
    let u1 = _mm_unpackhi_epi16(a, c); // a4 c4 a5 c5 ...
    let u2 = _mm_unpacklo_epi16(b, d); // b0 d0 b1 d1 ...
    let u3 = _mm_unpackhi_epi16(b, d); // b4 d4 b5 d5 ...

    let v0 = _mm_unpacklo_epi16(u0, u2); // a0 b0 c0 d0 ...
    let v1 = _mm_unpackhi_epi16(u0, u2); // a2 b2 c2 d2 ...
    let v2 = _mm_unpacklo_epi16(u1, u3); // a4 b4 c4 d4 ...
    let v3 = _mm_unpackhi_epi16(u1, u3); // a6 b6 c6 d6 ...
    (v0, v1, v2, v3)
}
