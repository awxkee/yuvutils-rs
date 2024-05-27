#[cfg(target_arch = "x86_64")]
#[cfg(feature = "nightly_avx512")]
use std::arch::x86_64::*;

#[cfg(all(target_arch = "x86_64"))]
#[cfg(feature = "nightly_avx512")]
#[inline(always)]
#[allow(dead_code)]
pub unsafe fn avx512_combine(lo: __m256i, hi: __m256i) -> __m512i {
    _mm512_inserti64x4::<1>(_mm512_castsi256_si512(lo), hi)
}

#[cfg(all(target_arch = "x86_64"))]
#[cfg(feature = "nightly_avx512")]
#[inline(always)]
#[allow(dead_code)]
pub unsafe fn avx512_pack_u16(lo: __m512i, hi: __m512i) -> __m512i {
    let mask = _mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7);
    _mm512_permutexvar_epi64(mask, _mm512_packus_epi16(lo, hi))
}

#[cfg(all(target_arch = "x86_64"))]
#[cfg(feature = "nightly_avx512")]
#[inline(always)]
#[allow(dead_code)]
unsafe fn _v512_set_epu32(
    a15: i64,
    a14: i64,
    a13: i64,
    a12: i64,
    a11: i64,
    a10: i64,
    a9: i64,
    a8: i64,
    a7: i64,
    a6: i64,
    a5: i64,
    a4: i64,
    a3: i64,
    a2: i64,
    a1: i64,
    a0: i64,
) -> __m512i {
    _mm512_set_epi64(
        ((a15) << 32) | (a14),
        ((a13) << 32) | (a12),
        ((a11) << 32) | (a10),
        ((a9) << 32) | (a8),
        ((a7) << 32) | (a6),
        ((a5) << 32) | (a4),
        ((a3) << 32) | (a2),
        ((a1) << 32) | (a0),
    )
}

#[cfg(all(target_arch = "x86_64"))]
#[cfg(feature = "nightly_avx512")]
#[inline(always)]
#[allow(dead_code)]
unsafe fn _v512_set_epu16(
    a31: i64,
    a30: i64,
    a29: i64,
    a28: i64,
    a27: i64,
    a26: i64,
    a25: i64,
    a24: i64,
    a23: i64,
    a22: i64,
    a21: i64,
    a20: i64,
    a19: i64,
    a18: i64,
    a17: i64,
    a16: i64,
    a15: i64,
    a14: i64,
    a13: i64,
    a12: i64,
    a11: i64,
    a10: i64,
    a9: i64,
    a8: i64,
    a7: i64,
    a6: i64,
    a5: i64,
    a4: i64,
    a3: i64,
    a2: i64,
    a1: i64,
    a0: i64,
) -> __m512i {
    _v512_set_epu32(
        ((a31) << 16) | (a30),
        ((a29) << 16) | (a28),
        ((a27) << 16) | (a26),
        ((a25) << 16) | (a24),
        ((a23) << 16) | (a22),
        ((a21) << 16) | (a20),
        ((a19) << 16) | (a18),
        ((a17) << 16) | (a16),
        ((a15) << 16) | (a14),
        ((a13) << 16) | (a12),
        ((a11) << 16) | (a10),
        ((a9) << 16) | (a8),
        ((a7) << 16) | (a6),
        ((a5) << 16) | (a4),
        ((a3) << 16) | (a2),
        ((a1) << 16) | (a0),
    )
}

#[cfg(all(target_arch = "x86_64"))]
#[cfg(feature = "nightly_avx512")]
#[inline(always)]
#[allow(dead_code)]
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

#[cfg(all(target_arch = "x86_64"))]
#[cfg(feature = "nightly_avx512")]
#[inline(always)]
#[allow(dead_code)]
pub unsafe fn avx512_rgb_u8(dst: *mut u8, a: __m512i, b: __m512i, c: __m512i) {
    let (rgb0, rgb1, rgb2) = avx512_interleave_rgb(a, b, c);
    _mm512_storeu_si512(dst as *mut i32, rgb0);
    _mm512_storeu_si512(dst.add(64) as *mut i32, rgb1);
    _mm512_storeu_si512(dst.add(128) as *mut i32, rgb2);
}

#[cfg(all(target_arch = "x86_64"))]
#[cfg(feature = "nightly_avx512")]
#[inline(always)]
#[allow(dead_code)]
pub unsafe fn avx512_zip(a: __m512i, b: __m512i) -> (__m512i, __m512i) {
    let low = _mm512_unpacklo_epi8(a, b);
    let high = _mm512_unpackhi_epi8(a, b);
    let ab0 = _mm512_permutex2var_epi64(low, _mm512_set_epi64(11, 10, 3, 2, 9, 8, 1, 0), high);
    let ab1 = _mm512_permutex2var_epi64(low, _mm512_set_epi64(15, 14, 7, 6, 13, 12, 5, 4), high);
    (ab0, ab1)
}

#[cfg(all(target_arch = "x86_64"))]
#[cfg(feature = "nightly_avx512")]
#[inline(always)]
#[allow(dead_code)]
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

#[cfg(all(target_arch = "x86_64"))]
#[cfg(feature = "nightly_avx512")]
#[inline(always)]
#[allow(dead_code)]
pub unsafe fn avx512_rgba_u8(dst: *mut u8, a: __m512i, b: __m512i, c: __m512i, d: __m512i) {
    let (rgb0, rgb1, rgb2, rgb3) = avx512_interleave_rgba(a, b, c, d);
    _mm512_storeu_si512(dst as *mut i32, rgb0);
    _mm512_storeu_si512(dst.add(64) as *mut i32, rgb1);
    _mm512_storeu_si512(dst.add(128) as *mut i32, rgb2);
    _mm512_storeu_si512(dst.add(128 + 64) as *mut i32, rgb3);
}

#[cfg(all(target_arch = "x86_64"))]
#[cfg(feature = "nightly_avx512")]
#[inline(always)]
#[allow(dead_code)]
pub unsafe fn avx512_div_by255(v: __m512i) -> __m512i {
    let rounding = _mm512_set1_epi16(1 << 7);
    let x = _mm512_adds_epi16(v, rounding);
    let multiplier = _mm512_set1_epi16(-32640);
    let r = _mm512_mulhi_epu16(x, multiplier);
    return _mm512_srli_epi16::<7>(r);
}
