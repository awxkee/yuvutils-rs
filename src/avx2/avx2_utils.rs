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

#[inline(always)]
pub const fn shuffle(z: u32, y: u32, x: u32, w: u32) -> i32 {
    // Checked: we want to reinterpret the bits
    ((z << 6) | (y << 4) | (x << 2) | w) as i32
}

#[inline]
#[target_feature(enable = "avx2")]
pub unsafe fn avx2_pack_u16(s_1: __m256i, s_2: __m256i) -> __m256i {
    let packed = _mm256_packus_epi16(s_1, s_2);
    const MASK: i32 = shuffle(3, 1, 2, 0);
    _mm256_permute4x64_epi64::<MASK>(packed)
}

#[inline]
#[target_feature(enable = "avx2")]
pub unsafe fn _mm256_interleave_epi8(a: __m256i, b: __m256i) -> (__m256i, __m256i) {
    let xy_l = _mm256_unpacklo_epi8(a, b);
    let xy_h = _mm256_unpackhi_epi8(a, b);

    let xy0 = _mm256_permute2x128_si256::<32>(xy_l, xy_h);
    let xy1 = _mm256_permute2x128_si256::<49>(xy_l, xy_h);
    (xy0, xy1)
}

#[inline]
#[target_feature(enable = "avx2")]
pub unsafe fn _mm256_deinterleave_rgba_epi8(
    rgba0: __m256i,
    rgba1: __m256i,
    rgba2: __m256i,
    rgba3: __m256i,
) -> (__m256i, __m256i, __m256i, __m256i) {
    #[rustfmt::skip]
    let sh = _mm256_setr_epi8(
        0, 4, 8, 12, 1, 5,
        9, 13, 2, 6, 10, 14,
        3, 7, 11, 15, 0, 4,
        8, 12, 1, 5, 9, 13,
        2, 6, 10, 14, 3, 7,
        11, 15,
    );

    let p0 = _mm256_shuffle_epi8(rgba0, sh);
    let p1 = _mm256_shuffle_epi8(rgba1, sh);
    let p2 = _mm256_shuffle_epi8(rgba2, sh);
    let p3 = _mm256_shuffle_epi8(rgba3, sh);

    let p01l = _mm256_unpacklo_epi32(p0, p1);
    let p01h = _mm256_unpackhi_epi32(p0, p1);
    let p23l = _mm256_unpacklo_epi32(p2, p3);
    let p23h = _mm256_unpackhi_epi32(p2, p3);

    let pll = _mm256_permute2x128_si256::<32>(p01l, p23l);
    let plh = _mm256_permute2x128_si256::<49>(p01l, p23l);
    let phl = _mm256_permute2x128_si256::<32>(p01h, p23h);
    let phh = _mm256_permute2x128_si256::<49>(p01h, p23h);

    let b0 = _mm256_unpacklo_epi32(pll, plh);
    let g0 = _mm256_unpackhi_epi32(pll, plh);
    let r0 = _mm256_unpacklo_epi32(phl, phh);
    let a0 = _mm256_unpackhi_epi32(phl, phh);

    (b0, g0, r0, a0)
}

#[inline]
#[target_feature(enable = "avx2")]
pub unsafe fn avx2_store_u8_rgb(ptr: *mut u8, r: __m256i, g: __m256i, b: __m256i) {
    let (rgb1, rgb2, rgb3) = avx2_interleave_rgb(r, g, b);

    _mm256_storeu_si256(ptr as *mut __m256i, rgb1);
    _mm256_storeu_si256(ptr.add(32) as *mut __m256i, rgb2);
    _mm256_storeu_si256(ptr.add(64) as *mut __m256i, rgb3);
}

#[inline]
#[target_feature(enable = "avx2")]
pub unsafe fn _mm256_store_interleaved_epi8(
    ptr: *mut u8,
    r: __m256i,
    g: __m256i,
    b: __m256i,
    a: __m256i,
) {
    let bg0 = _mm256_unpacklo_epi8(r, g);
    let bg1 = _mm256_unpackhi_epi8(r, g);
    let ra0 = _mm256_unpacklo_epi8(b, a);
    let ra1 = _mm256_unpackhi_epi8(b, a);

    let rgba0_ = _mm256_unpacklo_epi16(bg0, ra0);
    let rgba1_ = _mm256_unpackhi_epi16(bg0, ra0);
    let rgba2_ = _mm256_unpacklo_epi16(bg1, ra1);
    let rgba3_ = _mm256_unpackhi_epi16(bg1, ra1);

    let rgba0 = _mm256_permute2x128_si256::<32>(rgba0_, rgba1_);
    let rgba2 = _mm256_permute2x128_si256::<49>(rgba0_, rgba1_);
    let rgba1 = _mm256_permute2x128_si256::<32>(rgba2_, rgba3_);
    let rgba3 = _mm256_permute2x128_si256::<49>(rgba2_, rgba3_);

    _mm256_storeu_si256(ptr as *mut __m256i, rgba0);
    _mm256_storeu_si256(ptr.add(32) as *mut __m256i, rgba1);
    _mm256_storeu_si256(ptr.add(64) as *mut __m256i, rgba2);
    _mm256_storeu_si256(ptr.add(96) as *mut __m256i, rgba3);
}

#[inline]
#[target_feature(enable = "avx2")]
pub unsafe fn avx2_interleave_odd(x: __m256i) -> __m256i {
    #[rustfmt::skip]
    let shuffle = _mm256_setr_epi8(1, 1, 3, 3,
                                   5, 5, 7, 7,
                                   9, 9, 11, 11,
                                   13, 13, 15, 15,
                                   17, 17, 19, 19,
                                   21, 21, 23, 23,
                                   25, 25, 27, 27,
                                   29, 29, 31, 31);
    let new_lane = _mm256_shuffle_epi8(x, shuffle);
    new_lane
}

#[inline]
#[target_feature(enable = "avx2")]
pub unsafe fn avx2_interleave_even(x: __m256i) -> __m256i {
    #[rustfmt::skip]
    let shuffle = _mm256_setr_epi8(0, 0, 2, 2,
                                   4, 4, 6, 6,
                                   8, 8, 10, 10,
                                   12, 12, 14, 14,
                                   16, 16, 18, 18,
                                   20, 20, 22, 22,
                                   24, 24, 26, 26,
                                   28, 28, 30, 30);
    let new_lane = _mm256_shuffle_epi8(x, shuffle);
    new_lane
}

#[inline]
#[target_feature(enable = "avx2")]
pub unsafe fn avx2_interleave_rgb(
    r: __m256i,
    g: __m256i,
    b: __m256i,
) -> (__m256i, __m256i, __m256i) {
    let sh_b = _mm256_setr_epi8(
        0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14, 9, 4, 15, 10, 5, 0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14,
        9, 4, 15, 10, 5,
    );
    let sh_g = _mm256_setr_epi8(
        5, 0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14, 9, 4, 15, 10, 5, 0, 11, 6, 1, 12, 7, 2, 13, 8, 3,
        14, 9, 4, 15, 10,
    );
    let sh_r = _mm256_setr_epi8(
        10, 5, 0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14, 9, 4, 15, 10, 5, 0, 11, 6, 1, 12, 7, 2, 13, 8,
        3, 14, 9, 4, 15,
    );

    let b0 = _mm256_shuffle_epi8(r, sh_b);
    let g0 = _mm256_shuffle_epi8(g, sh_g);
    let r0 = _mm256_shuffle_epi8(b, sh_r);

    let m0 = _mm256_setr_epi8(
        0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1,
        0, 0, -1, 0, 0,
    );
    let m1 = _mm256_setr_epi8(
        0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0,
        -1, 0, 0, -1, 0,
    );

    let p0 = _mm256_blendv_epi8(_mm256_blendv_epi8(b0, g0, m0), r0, m1);
    let p1 = _mm256_blendv_epi8(_mm256_blendv_epi8(g0, r0, m0), b0, m1);
    let p2 = _mm256_blendv_epi8(_mm256_blendv_epi8(r0, b0, m0), g0, m1);

    let bgr0 = _mm256_permute2x128_si256::<32>(p0, p1);
    let bgr1 = _mm256_permute2x128_si256::<48>(p2, p0);
    let bgr2 = _mm256_permute2x128_si256::<49>(p1, p2);

    (bgr0, bgr1, bgr2)
}

#[inline]
#[target_feature(enable = "avx2")]
pub unsafe fn avx2_deinterleave_rgb(
    rgb0: __m256i,
    rgb1: __m256i,
    rgb2: __m256i,
) -> (__m256i, __m256i, __m256i) {
    let s02_low = _mm256_permute2x128_si256::<32>(rgb0, rgb2);
    let s02_high = _mm256_permute2x128_si256::<49>(rgb0, rgb2);

    #[rustfmt::skip]
    let m0 = _mm256_setr_epi8(
        0, 0, -1, 0, 0,
        -1, 0, 0, -1, 0,
        0, -1, 0, 0, -1,
        0, 0, -1, 0, 0,
        -1, 0, 0, -1, 0,
        0, -1, 0, 0, -1,
        0, 0,
    );

    #[rustfmt::skip]
    let m1 = _mm256_setr_epi8(
        0, -1, 0, 0, -1,
        0, 0, -1, 0, 0,
        -1, 0, 0, -1, 0,
        0, -1, 0, 0, -1,
        0, 0, -1, 0, 0,
        -1, 0, 0, -1, 0,
        0, -1,
    );

    let b0 = _mm256_blendv_epi8(_mm256_blendv_epi8(s02_low, s02_high, m0), rgb1, m1);
    let g0 = _mm256_blendv_epi8(_mm256_blendv_epi8(s02_high, s02_low, m1), rgb1, m0);
    let r0 = _mm256_blendv_epi8(_mm256_blendv_epi8(rgb1, s02_low, m0), s02_high, m1);

    #[rustfmt::skip]
    let sh_b = _mm256_setr_epi8(
        0, 3, 6, 9, 12,
        15, 2, 5, 8, 11,
        14, 1, 4, 7, 10,
        13, 0, 3, 6, 9,
        12, 15, 2, 5, 8,
        11, 14, 1, 4, 7,
        10, 13,
    );

    #[rustfmt::skip]
    let sh_g = _mm256_setr_epi8(
        1, 4, 7, 10, 13,
        0, 3, 6, 9, 12,
        15, 2, 5, 8, 11,
        14, 1, 4, 7, 10,
        13, 0, 3, 6, 9,
        12, 15, 2, 5, 8,
        11, 14,
    );

    #[rustfmt::skip]
    let sh_r = _mm256_setr_epi8(
        2, 5, 8, 11, 14,
        1, 4, 7, 10, 13,
        0, 3, 6, 9, 12,
        15, 2, 5, 8, 11,
        14, 1, 4, 7, 10,
        13, 0, 3, 6, 9,
        12, 15,
    );
    let b0 = _mm256_shuffle_epi8(b0, sh_b);
    let g0 = _mm256_shuffle_epi8(g0, sh_g);
    let r0 = _mm256_shuffle_epi8(r0, sh_r);
    (b0, g0, r0)
}

// #[inline]
// #[target_feature(enable = "avx2")]
// pub unsafe fn avx2_reshuffle_odd(v: __m256i) -> __m256i {
//     const MASK: i32 = shuffle(3, 1, 2, 0);
//     _mm256_permute4x64_epi64::<MASK>(v)
// }

#[inline]
#[target_feature(enable = "avx2")]
pub unsafe fn avx2_pairwise_widen_avg(v: __m256i) -> __m256i {
    let sums = _mm256_maddubs_epi16(v, _mm256_set1_epi8(1));
    let shifted = _mm256_srli_epi16::<1>(_mm256_add_epi16(sums, _mm256_set1_epi16(1)));
    let packed_lo = _mm256_packus_epi16(shifted, shifted);
    const MASK: i32 = shuffle(3, 1, 2, 0);
    _mm256_permute4x64_epi64::<MASK>(packed_lo)
}

#[inline]
#[target_feature(enable = "avx2")]
pub unsafe fn avx2_div_by255(v: __m256i) -> __m256i {
    let addition = _mm256_set1_epi16(127);
    _mm256_srli_epi16::<8>(_mm256_add_epi16(
        _mm256_add_epi16(v, addition),
        _mm256_srli_epi16::<8>(v),
    ))
}

#[inline]
#[target_feature(enable = "avx2")]
pub unsafe fn _mm256_deinterleave_x2_epi8(a: __m256i, b: __m256i) -> (__m256i, __m256i) {
    let sh = _mm256_setr_epi8(
        0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15, 0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5,
        7, 9, 11, 13, 15,
    );
    let p0 = _mm256_shuffle_epi8(a, sh);
    let p1 = _mm256_shuffle_epi8(b, sh);
    let pl = _mm256_permute2x128_si256::<32>(p0, p1);
    let ph = _mm256_permute2x128_si256::<49>(p0, p1);
    let a0 = _mm256_unpacklo_epi64(pl, ph);
    let b0 = _mm256_unpackhi_epi64(pl, ph);
    (a0, b0)
}
