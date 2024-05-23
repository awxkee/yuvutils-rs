#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
#[inline(always)]
#[allow(dead_code)]
pub const fn shuffle(z: u32, y: u32, x: u32, w: u32) -> i32 {
    // Checked: we want to reinterpret the bits
    ((z << 6) | (y << 4) | (x << 2) | w) as i32
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
#[allow(dead_code)]
pub unsafe fn demote_i16_to_u8(s_1: __m256i, s_2: __m256i) -> __m256i {
    let packed = _mm256_packus_epi16(s_1, s_2);
    const MASK: i32 = shuffle(3, 1, 2, 0);
    return _mm256_permute4x64_epi64::<MASK>(packed);
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
#[allow(dead_code)]
pub unsafe fn sse_promote_i16_toi32(s: __m128i) -> __m128i {
    _mm_cvtepi16_epi32(_mm_srli_si128::<8>(s))
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub unsafe fn sse_interleave_even(x: __m128i) -> __m128i {
    #[rustfmt::skip]
        let shuffle = _mm_setr_epi8(0, 0, 2, 2, 4, 4, 6, 6,
                                    8, 8, 10, 10, 12, 12, 14, 14);
    let new_lane = _mm_shuffle_epi8(x, shuffle);
    return new_lane;
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
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
    return new_lane;
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub unsafe fn avx2_interleave_even_2_epi8(a: __m256i, b: __m256i) -> __m256i {
    let mask_a = _mm256_slli_epi16::<8>(_mm256_srli_epi16::<8>(a));
    let masked_a = _mm256_and_si256(a, mask_a);
    let b_s = _mm256_srli_epi16::<8>(b);
    return _mm256_or_si256(masked_a, b_s);
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub unsafe fn avx2_interleave_odd_2_epi8(a: __m256i, b: __m256i) -> __m256i {
    let mask_a = _mm256_set1_epi16(0x00FF);
    let masked_a = _mm256_slli_epi16::<8>(_mm256_and_si256(a, mask_a));
    let b_s = _mm256_and_si256(b, mask_a);
    return _mm256_or_si256(masked_a, b_s);
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
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
    return new_lane;
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub unsafe fn sse_interleave_odd(x: __m128i) -> __m128i {
    #[rustfmt::skip]
        let shuffle = _mm_setr_epi8(1, 1, 3, 3, 5, 5, 7, 7,
                                    9, 9, 11, 11, 13, 13, 15, 15);
    let new_lane = _mm_shuffle_epi8(x, shuffle);
    return new_lane;
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub unsafe fn avx2_interleave_rgb(
    r: __m256i,
    g: __m256i,
    b: __m256i,
) -> (__m256i, __m256i, __m256i) {
    let permute_0_r_row = _mm256_setr_epi8(
        0x0, -1, -1, 0x1, -1, -1, 0x2, -1, -1, 0x3, -1, -1, 0x4, -1, -1, 0x5, -1, -1, 0x6, -1, -1,
        0x7, -1, -1, 0x8, -1, -1, 0x9, -1, -1, 0xA, -1,
    );
    let permute_0_g_row = _mm256_setr_epi8(
        -1, 0x3, -1, -1, 0x4, -1, -1, 0x5, -1, -1, 0x6, -1, -1, 0x7, -1, -1, 0x8, -1, -1, 0x9, -1,
        -1, 0xA, -1, -1, 0xB, -1, -1, 0xC, -1, -1, 0xD,
    );
    let permute_0_b_row = _mm256_setr_epi8(
        -1, -1, 0x6, -1, -1, 0x7, -1, -1, 0x8, -1, -1, 0x9, -1, -1, 0xA, -1, -1, 0xB, -1, -1, 0xC,
        -1, -1, 0xD, -1, -1, 0xE, -1, -1, 0xF, -1, -1,
    );

    let permute_1_r_row = _mm256_setr_epi8(
        -1, 0x0, -1, -1, 0x1, -1, -1, 0x2, -1, -1, 0x3, -1, -1, 0x4, -1, -1, 0x5, -1, -1, 0x6, -1,
        -1, 0x7, -1, -1, 0x8, -1, -1, 0x9, -1, -1, 0xA,
    );
    let permute_1_g_row = _mm256_setr_epi8(
        -1, -1, 0x3, -1, -1, 0x4, -1, -1, 0x5, -1, -1, 0x6, -1, -1, 0x7, -1, -1, 0x8, -1, -1, 0x9,
        -1, -1, 0xA, -1, -1, 0xB, -1, -1, 0xC, -1, -1,
    );
    let permute_1_b_row = _mm256_setr_epi8(
        0x5, -1, -1, 0x6, -1, -1, 0x7, -1, -1, 0x8, -1, -1, 0x9, -1, -1, 0xA, -1, -1, 0xB, -1, -1,
        0xC, -1, -1, 0xD, -1, -1, 0xE, -1, -1, 0xF, -1,
    );

    let permute_2_r_row = _mm256_setr_epi8(
        -1, -1, 0x0, -1, -1, 0x1, -1, -1, 0x2, -1, -1, 0x3, -1, -1, 0x4, -1, -1, 0x5, -1, -1, 0x6,
        -1, -1, 0x7, -1, -1, 0x8, -1, -1, 0x9, -1, -1,
    );
    let permute_2_g_row = _mm256_setr_epi8(
        0x2, -1, -1, 0x3, -1, -1, 0x4, -1, -1, 0x5, -1, -1, 0x6, -1, -1, 0x7, -1, -1, 0x8, -1, -1,
        0x9, -1, -1, 0xA, -1, -1, 0xB, -1, -1, 0xC, -1,
    );
    let permute_2_b_row = _mm256_setr_epi8(
        -1, 0x5, -1, -1, 0x6, -1, -1, 0x7, -1, -1, 0x8, -1, -1, 0x9, -1, -1, 0xA, -1, -1, 0xB, -1,
        -1, 0xC, -1, -1, 0xD, -1, -1, 0xE, -1, -1, 0xF,
    );

    let rgb1 = _mm256_or_si256(
        _mm256_shuffle_epi8(_mm256_permute4x64_epi64::<0x44>(r), permute_0_r_row),
        _mm256_or_si256(
            _mm256_shuffle_epi8(_mm256_permute4x64_epi64::<0x44>(g), permute_1_r_row),
            _mm256_shuffle_epi8(_mm256_permute4x64_epi64::<0x44>(b), permute_2_r_row),
        ),
    );
    let rgb2 = _mm256_or_si256(
        _mm256_shuffle_epi8(_mm256_permute4x64_epi64::<0x99>(r), permute_0_g_row),
        _mm256_or_si256(
            _mm256_shuffle_epi8(_mm256_permute4x64_epi64::<0x99>(g), permute_1_g_row),
            _mm256_shuffle_epi8(_mm256_permute4x64_epi64::<0x99>(b), permute_2_g_row),
        ),
    );
    let rgb3 = _mm256_or_si256(
        _mm256_shuffle_epi8(_mm256_permute4x64_epi64::<0xEE>(r), permute_0_b_row),
        _mm256_or_si256(
            _mm256_shuffle_epi8(_mm256_permute4x64_epi64::<0xEE>(g), permute_1_b_row),
            _mm256_shuffle_epi8(_mm256_permute4x64_epi64::<0xEE>(b), permute_2_b_row),
        ),
    );
    (rgb1, rgb2, rgb3)
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
#[allow(dead_code)]
pub unsafe fn avx2_deinterleave_rgb(
    rgb0: __m256i,
    rgb1: __m256i,
    rgb2: __m256i,
) -> (__m256i, __m256i, __m256i) {
    let s02_low = _mm256_permute2x128_si256::<32>(rgb0, rgb2);
    let s02_high = _mm256_permute2x128_si256::<49>(rgb0, rgb2);

    let m0 = _mm256_setr_epi8(
        0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1,
        0, 0, -1, 0, 0,
    );
    let m1 = _mm256_setr_epi8(
        0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0,
        0, -1, 0, 0, -1,
    );

    let b0 = _mm256_blendv_epi8(_mm256_blendv_epi8(s02_low, s02_high, m0), rgb1, m1);
    let g0 = _mm256_blendv_epi8(_mm256_blendv_epi8(s02_high, s02_low, m1), rgb1, m0);
    let r0 = _mm256_blendv_epi8(_mm256_blendv_epi8(rgb1, s02_low, m0), s02_high, m1);

    let sh_b = _mm256_setr_epi8(
        0, 3, 6, 9, 12, 15, 2, 5, 8, 11, 14, 1, 4, 7, 10, 13, 0, 3, 6, 9, 12, 15, 2, 5, 8, 11, 14,
        1, 4, 7, 10, 13,
    );
    let sh_g = _mm256_setr_epi8(
        1, 4, 7, 10, 13, 0, 3, 6, 9, 12, 15, 2, 5, 8, 11, 14, 1, 4, 7, 10, 13, 0, 3, 6, 9, 12, 15,
        2, 5, 8, 11, 14,
    );
    let sh_r = _mm256_setr_epi8(
        2, 5, 8, 11, 14, 1, 4, 7, 10, 13, 0, 3, 6, 9, 12, 15, 2, 5, 8, 11, 14, 1, 4, 7, 10, 13, 0,
        3, 6, 9, 12, 15,
    );
    let b0 = _mm256_shuffle_epi8(b0, sh_b);
    let g0 = _mm256_shuffle_epi8(g0, sh_g);
    let r0 = _mm256_shuffle_epi8(r0, sh_r);
    (b0, g0, r0)
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
#[allow(dead_code)]
pub unsafe fn avx2_deinterleave_rgba(
    rgba0: __m256i,
    rgba1: __m256i,
    rgba2: __m256i,
    rgba3: __m256i,
) -> (__m256i, __m256i, __m256i, __m256i) {
    let sh = _mm256_setr_epi8(
        0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15, 0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10,
        14, 3, 7, 11, 15,
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

#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub unsafe fn avx2_store_u8_rgb(ptr: *mut u8, r: __m256i, g: __m256i, b: __m256i) {
    let (rgb1, rgb2, rgb3) = avx2_interleave_rgb(r, g, b);

    _mm256_storeu_si256(ptr as *mut __m256i, rgb1);
    _mm256_storeu_si256(ptr.add(32) as *mut __m256i, rgb2);
    _mm256_storeu_si256(ptr.add(64) as *mut __m256i, rgb3);
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub unsafe fn avx2_interleave(a: __m256i, b: __m256i) -> (__m256i, __m256i) {
    let v02 = _mm256_unpacklo_epi16(a, b);
    let v13 = _mm256_unpackhi_epi16(a, b);
    return (
        _mm256_permute2x128_si256::<0x20>(v02, v13),
        _mm256_permute2x128_si256::<0x31>(v02, v13),
    );
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub unsafe fn avx2_interleave_u8(a: __m256i, b: __m256i) -> (__m256i, __m256i) {
    let v02 = _mm256_unpacklo_epi8(a, b);
    let v13 = _mm256_unpackhi_epi8(a, b);
    return (
        _mm256_permute2x128_si256::<0x20>(v02, v13),
        _mm256_permute2x128_si256::<0x31>(v02, v13),
    );
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub unsafe fn avx2_store_u8_rgba(ptr: *mut u8, r: __m256i, g: __m256i, b: __m256i, a: __m256i) {
    let (rg_low, rg_high) = avx2_interleave_u8(r, g);
    let (ba_low, ba_high) = avx2_interleave_u8(b, a);

    let (rgba0, rgba1) = avx2_interleave(rg_low, ba_low);
    let (rgba2, rgba3) = avx2_interleave(rg_high, ba_high);

    _mm256_storeu_si256(ptr as *mut __m256i, rgba0);
    _mm256_storeu_si256(ptr.add(32) as *mut __m256i, rgba1);
    _mm256_storeu_si256(ptr.add(64) as *mut __m256i, rgba2);
    _mm256_storeu_si256(ptr.add(96) as *mut __m256i, rgba3);
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
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

#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub unsafe fn sse_store_rgba(ptr: *mut u8, r: __m128i, g: __m128i, b: __m128i, a: __m128i) {
    let (row1, row2, row3, row4) = sse_interleave_rgba(r, g, b, a);
    _mm_storeu_si128(ptr as *mut __m128i, row1);
    _mm_storeu_si128(ptr.add(16) as *mut __m128i, row2);
    _mm_storeu_si128(ptr.add(32) as *mut __m128i, row3);
    _mm_storeu_si128(ptr.add(48) as *mut __m128i, row4);
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
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
        let shuffle = _mm_setr_epi8(
        0, 4, 8, 12,
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

#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub unsafe fn sse_deinterleave_rgb(
    rgb0: __m128i,
    rgb1: __m128i,
    rgb2: __m128i,
) -> (__m128i, __m128i, __m128i) {
    let idx = _mm_setr_epi8(0, 3, 6, 9, 12, 15, 2, 5, 8, 11, 14, 1, 4, 7, 10, 13);

    let r6b5g5_0 = _mm_shuffle_epi8(rgb0, idx);
    let g6r5b5_1 = _mm_shuffle_epi8(rgb1, idx);
    let b6g5r5_2 = _mm_shuffle_epi8(rgb2, idx);

    let mask010 = _mm_setr_epi8(0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0);
    let mask001 = _mm_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1);

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

#[cfg(target_arch = "x86_64")]
#[inline(always)]
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

#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub unsafe fn sse_store_rgb_u8(ptr: *mut u8, r: __m128i, g: __m128i, b: __m128i) {
    let (v0, v1, v2) = sse_interleave_rgb(r, g, b);
    _mm_storeu_si128(ptr as *mut __m128i, v0);
    _mm_storeu_si128(ptr.add(16) as *mut __m128i, v1);
    _mm_storeu_si128(ptr.add(32) as *mut __m128i, v2);
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub unsafe fn avx2_pairwise_add(v: __m256i) -> __m256i {
    let evens = _mm256_setr_epi8(
        0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, -1, -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1, -1,
    );
    let odds = _mm256_setr_epi8(
        1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, -1, -1, -1, -1, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1, -1,
    );

    let evens_16 = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(_mm256_shuffle_epi8(v, evens)));
    let odds_16 = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(_mm256_shuffle_epi8(v, odds)));
    let sums = _mm256_avg_epu16(evens_16, odds_16);
    let packed_lo = _mm256_packus_epi16(sums, sums);
    const MASK: i32 = shuffle(3, 1, 2, 0);
    return _mm256_permute4x64_epi64::<MASK>(packed_lo);
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub unsafe fn sse_pairwise_add(v: __m128i) -> __m128i {
    let evens = _mm_setr_epi8(0, 2, 4, 6, 8, 10, 12, 14, -1, -1, -1, -1, -1, -1, -1, -1);
    let odds = _mm_setr_epi8(1, 3, 5, 7, 9, 11, 13, -1, -1, -1, -1, -1, -1, -1, -1, -1);

    let evens_16 = _mm_cvtepu8_epi16(_mm_shuffle_epi8(v, evens));
    let odds_16 = _mm_cvtepu8_epi16(_mm_shuffle_epi8(v, odds));
    let sums = _mm_avg_epu16(evens_16, odds_16);
    let packed = _mm_packus_epi16(sums, sums);
    packed
}
