/*
 * // Copyright (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

#[cfg(target_arch = "x86")]
#[cfg(feature = "nightly_avx512")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
#[cfg(feature = "nightly_avx512")]
use std::arch::x86_64::*;

use crate::avx512bw::avx512_setr::{_v512_set_epu16, _v512_set_epu32};

#[inline]
pub unsafe fn avx512_pack_u16(lo: __m512i, hi: __m512i) -> __m512i {
    let mask = _mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7);
    _mm512_permutexvar_epi64(mask, _mm512_packus_epi16(lo, hi))
}

#[inline]
pub unsafe fn avx512_interleave_rgb(
    a: __m512i,
    b: __m512i,
    c: __m512i,
) -> (__m512i, __m512i, __m512i) {
    let g1g0 = _mm512_shuffle_epi8(
        b,
        _mm512_set4_epi32(0x0e0f0c0d, 0x0a0b0809, 0x06070405, 0x02030001),
    );
    let b0g0 = _mm512_mask_blend_epi8(0xAAAAAAAAAAAAAAAA, a, g1g0);
    let r0b1 = _mm512_mask_blend_epi8(0xAAAAAAAAAAAAAAAA, c, a);
    let g1r1 = _mm512_mask_blend_epi8(0xAAAAAAAAAAAAAAAA, g1g0, c);

    let mask0 = _v512_set_epu16(
        42, 10, 31, 41, 9, 30, 40, 8, 29, 39, 7, 28, 38, 6, 27, 37, 5, 26, 36, 4, 25, 35, 3, 24,
        34, 2, 23, 33, 1, 22, 32, 0,
    );
    let mask1 = _v512_set_epu16(
        21, 52, 41, 20, 51, 40, 19, 50, 39, 18, 49, 38, 17, 48, 37, 16, 47, 36, 15, 46, 35, 14, 45,
        34, 13, 44, 33, 12, 43, 32, 11, 42,
    );
    let mask2 = _v512_set_epu16(
        63, 31, 20, 62, 30, 19, 61, 29, 18, 60, 28, 17, 59, 27, 16, 58, 26, 15, 57, 25, 14, 56, 24,
        13, 55, 23, 12, 54, 22, 11, 53, 21,
    );
    let b0g0b2 = _mm512_permutex2var_epi16(b0g0, mask0, r0b1);
    let r1b1r0 = _mm512_permutex2var_epi16(b0g0, mask1, g1r1);
    let g2r2g1 = _mm512_permutex2var_epi16(r0b1, mask2, g1r1);

    let bgr0 = _mm512_mask_blend_epi16(0x24924924, b0g0b2, r1b1r0);
    let bgr1 = _mm512_mask_blend_epi16(0x24924924, r1b1r0, g2r2g1);
    let bgr2 = _mm512_mask_blend_epi16(0x24924924, g2r2g1, b0g0b2);
    (bgr0, bgr1, bgr2)
}

#[inline]
pub unsafe fn avx512_rgb_u8(dst: *mut u8, a: __m512i, b: __m512i, c: __m512i) {
    let (rgb0, rgb1, rgb2) = avx512_interleave_rgb(a, b, c);
    _mm512_storeu_si512(dst as *mut i32, rgb0);
    _mm512_storeu_si512(dst.add(64) as *mut i32, rgb1);
    _mm512_storeu_si512(dst.add(128) as *mut i32, rgb2);
}

#[inline]
pub unsafe fn avx512_zip(a: __m512i, b: __m512i) -> (__m512i, __m512i) {
    let low = _mm512_unpacklo_epi8(a, b);
    let high = _mm512_unpackhi_epi8(a, b);
    let ab0 = _mm512_permutex2var_epi64(low, _mm512_set_epi64(11, 10, 3, 2, 9, 8, 1, 0), high);
    let ab1 = _mm512_permutex2var_epi64(low, _mm512_set_epi64(15, 14, 7, 6, 13, 12, 5, 4), high);
    (ab0, ab1)
}

#[inline]
pub unsafe fn avx512_interleave_rgba(
    a: __m512i,
    b: __m512i,
    c: __m512i,
    d: __m512i,
) -> (__m512i, __m512i, __m512i, __m512i) {
    let (br01, br23) = avx512_zip(a, c);
    let (ga01, ga23) = avx512_zip(b, d);
    let (bgra0, bgra1) = avx512_zip(br01, ga01);
    let (bgra2, bgra3) = avx512_zip(br23, ga23);
    (bgra0, bgra1, bgra2, bgra3)
}

#[inline]
pub unsafe fn avx512_rgba_u8(dst: *mut u8, a: __m512i, b: __m512i, c: __m512i, d: __m512i) {
    let (rgb0, rgb1, rgb2, rgb3) = avx512_interleave_rgba(a, b, c, d);
    _mm512_storeu_si512(dst as *mut i32, rgb0);
    _mm512_storeu_si512(dst.add(64) as *mut i32, rgb1);
    _mm512_storeu_si512(dst.add(128) as *mut i32, rgb2);
    _mm512_storeu_si512(dst.add(128 + 64) as *mut i32, rgb3);
}

#[inline]
pub unsafe fn avx512_div_by255(v: __m512i) -> __m512i {
    let rounding = _mm512_set1_epi16(1 << 7);
    let x = _mm512_adds_epi16(v, rounding);
    let multiplier = _mm512_set1_epi16(-32640);
    let r = _mm512_mulhi_epu16(x, multiplier);
    _mm512_srli_epi16::<7>(r)
}

#[inline]
pub unsafe fn avx512_deinterleave_rgb(
    bgr0: __m512i,
    bgr1: __m512i,
    bgr2: __m512i,
) -> (__m512i, __m512i, __m512i) {
    let mask0 = _v512_set_epu16(
        61, 58, 55, 52, 49, 46, 43, 40, 37, 34, 63, 60, 57, 54, 51, 48, 45, 42, 39, 36, 33, 30, 27,
        24, 21, 18, 15, 12, 9, 6, 3, 0,
    );
    let b01g1 = _mm512_permutex2var_epi16(bgr0, mask0, bgr1);
    let r12b2 = _mm512_permutex2var_epi16(bgr1, mask0, bgr2);
    let g20r0 = _mm512_permutex2var_epi16(bgr2, mask0, bgr0);

    let b0g0 = _mm512_mask_blend_epi32(0xf800, b01g1, r12b2);
    let r0b1 = _mm512_permutex2var_epi16(
        bgr1,
        _v512_set_epu16(
            42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 29, 26, 23, 20, 17, 14, 11, 8, 5, 2, 53,
            52, 51, 50, 49, 48, 47, 46, 45, 44, 43,
        ),
        g20r0,
    );
    let g1r1 = _mm512_alignr_epi32::<11>(r12b2, g20r0);
    let a = _mm512_mask_blend_epi8(0xAAAAAAAAAAAAAAAA, b0g0, r0b1);
    let c = _mm512_mask_blend_epi8(0xAAAAAAAAAAAAAAAA, r0b1, g1r1);
    let b = _mm512_shuffle_epi8(
        _mm512_mask_blend_epi8(0xAAAAAAAAAAAAAAAA, g1r1, b0g0),
        _mm512_set4_epi32(0x0e0f0c0d, 0x0a0b0809, 0x06070405, 0x02030001),
    );
    (a, b, c)
}

#[inline]
pub unsafe fn avx512_deinterleave_rgba(
    bgra0: __m512i,
    bgra1: __m512i,
    bgra2: __m512i,
    bgra3: __m512i,
) -> (__m512i, __m512i, __m512i, __m512i) {
    let mask = _mm512_set4_epi32(0x0f0b0703, 0x0e0a0602, 0x0d090501, 0x0c080400);
    let b0g0r0a0 = _mm512_shuffle_epi8(bgra0, mask);
    let b1g1r1a1 = _mm512_shuffle_epi8(bgra1, mask);
    let b2g2r2a2 = _mm512_shuffle_epi8(bgra2, mask);
    let b3g3r3a3 = _mm512_shuffle_epi8(bgra3, mask);

    let mask0 = _v512_set_epu32(30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 0);
    let mask1 = _v512_set_epu32(31, 29, 27, 25, 23, 21, 19, 17, 15, 13, 11, 9, 7, 5, 3, 1);

    let br01 = _mm512_permutex2var_epi32(b0g0r0a0, mask0, b1g1r1a1);
    let ga01 = _mm512_permutex2var_epi32(b0g0r0a0, mask1, b1g1r1a1);
    let br23 = _mm512_permutex2var_epi32(b2g2r2a2, mask0, b3g3r3a3);
    let ga23 = _mm512_permutex2var_epi32(b2g2r2a2, mask1, b3g3r3a3);

    let a = _mm512_permutex2var_epi32(br01, mask0, br23);
    let c = _mm512_permutex2var_epi32(br01, mask1, br23);
    let b = _mm512_permutex2var_epi32(ga01, mask0, ga23);
    let d = _mm512_permutex2var_epi32(ga01, mask1, ga23);
    (a, b, c, d)
}

#[inline]
pub unsafe fn avx512_pairwise_widen_avg(v: __m512i) -> __m512i {
    let sums = _mm512_maddubs_epi16(v, _mm512_set1_epi8(1));
    let shifted = _mm512_srli_epi16::<1>(_mm512_add_epi16(sums, _mm512_set1_epi16(1)));
    let packed_lo = _mm512_packus_epi16(shifted, shifted);
    let mask = _mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7);
    _mm512_permutexvar_epi64(mask, packed_lo)
}

#[inline]
pub unsafe fn avx512_rgb_to_ycbcr(
    r: __m512i,
    g: __m512i,
    b: __m512i,
    bias: __m512i,
    coeff_r: __m512i,
    coeff_g: __m512i,
    coeff_b: __m512i,
) -> __m512i {
    let r_l = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(r));
    let g_l = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(g));
    let b_l = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(b));

    let vl = _mm512_srai_epi32::<8>(_mm512_add_epi32(
        bias,
        _mm512_add_epi32(
            _mm512_add_epi32(
                _mm512_madd_epi16(coeff_r, r_l),
                _mm512_madd_epi16(coeff_g, g_l),
            ),
            _mm512_madd_epi16(coeff_b, b_l),
        ),
    ));

    let r_h = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(r));
    let g_h = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(g));
    let b_h = _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(b));

    let vh = _mm512_srai_epi32::<8>(_mm512_add_epi32(
        bias,
        _mm512_add_epi32(
            _mm512_add_epi32(
                _mm512_madd_epi16(coeff_r, r_h),
                _mm512_madd_epi16(coeff_g, g_h),
            ),
            _mm512_madd_epi16(coeff_b, b_h),
        ),
    ));

    let packed = _mm512_packus_epi32(vl, vh);
    let mask = _mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7);
    _mm512_permutexvar_epi64(mask, packed)
}

#[inline]
pub unsafe fn avx512_interleave_odd_epi8(a: __m512i, b: __m512i) -> __m512i {
    let mask_a = _mm512_set1_epi16(0x00FF);
    let masked_a = _mm512_slli_epi16::<8>(_mm512_and_si512(a, mask_a));
    let b_s = _mm512_and_si512(b, mask_a);
    _mm512_or_si512(masked_a, b_s)
}

#[inline]
pub unsafe fn avx512_interleave_even_epi8(a: __m512i, b: __m512i) -> __m512i {
    let mask_a = _mm512_slli_epi16::<8>(_mm512_srli_epi16::<8>(a));
    let masked_a = _mm512_and_si512(a, mask_a);
    let b_s = _mm512_srli_epi16::<8>(b);
    _mm512_or_si512(masked_a, b_s)
}

pub const fn shuffle(z: u32, y: u32, x: u32, w: u32) -> i32 {
    // Checked: we want to reinterpret the bits
    ((z << 6) | (y << 4) | (x << 2) | w) as i32
}

#[inline]
pub unsafe fn avx2_zip(a: __m256i, b: __m256i) -> (__m256i, __m256i) {
    const MASK: i32 = shuffle(3, 1, 2, 0);
    let v0 = _mm256_permute4x64_epi64::<MASK>(a);
    let v1 = _mm256_permute4x64_epi64::<MASK>(b);
    let b0 = _mm256_unpacklo_epi8(v0, v1);
    let b1 = _mm256_unpackhi_epi8(v0, v1);
    (b0, b1)
}
