#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
#[inline(always)]
#[allow(dead_code)]
const fn shuffle(z: u32, y: u32, x: u32, w: u32) -> i32 {
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
#[allow(dead_code)]
pub unsafe fn sse_rgb_to_ycbcr(
    r: __m128i,
    g: __m128i,
    b: __m128i,
    bias: __m128i,
    coeff_r: __m128i,
    coeff_g: __m128i,
    coeff_b: __m128i,
) -> __m128i {
    let r_l = _mm_cvtepi16_epi32(r);
    let g_l = _mm_cvtepi16_epi32(g);
    let b_l = _mm_cvtepi16_epi32(b);

    let min_acceptable_values = _mm_setzero_si128();

    let vl = _mm_srai_epi32::<8>(_mm_max_epi32(
        _mm_add_epi32(
            bias,
            _mm_add_epi32(
                _mm_add_epi32(_mm_mullo_epi32(coeff_r, r_l), _mm_mullo_epi32(coeff_g, g_l)),
                _mm_mullo_epi32(coeff_b, b_l),
            ),
        ),
        min_acceptable_values,
    ));

    let r_h = sse_promote_i16_toi32(r);
    let g_h = sse_promote_i16_toi32(g);
    let b_h = sse_promote_i16_toi32(b);

    let vh = _mm_srai_epi32::<8>(_mm_max_epi32(
        _mm_add_epi32(
            bias,
            _mm_add_epi32(
                _mm_add_epi32(_mm_mullo_epi32(coeff_r, r_h), _mm_mullo_epi32(coeff_g, g_h)),
                _mm_mullo_epi32(coeff_b, b_h),
            ),
        ),
        min_acceptable_values,
    ));

    _mm_packus_epi32(vl, vh)
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
pub unsafe fn store_u8_rgb_avx2(
    ptr: *mut u8,
    r: __m256i,
    g: __m256i,
    b: __m256i,
    use_transient: bool,
) {
    let zero = _mm256_setzero_si256();
    let (rg_low, rg_high) = avx2_interleave_u8(r, g);
    let (ba_low, ba_high) = avx2_interleave_u8(b, zero);

    let (rgba0, rgba1) = avx2_interleave(rg_low, ba_low);
    let (rgba2, rgba3) = avx2_interleave(rg_high, ba_high);

    #[rustfmt::skip]
        let shuffle_mask = _mm256_setr_epi8(
        0, 1, 2,
        4, 5, 6,
        8, 9, 10,
        12, 13, 14,
        16, 17, 18,
        20, 21, 22,
        24, 25, 26,
        28, 29, 30,
        -1, -1, -1, -1,
        -1, -1, -1, -1,
    );

    let rgb0 = _mm256_shuffle_epi8(rgba0, shuffle_mask);
    let rgb1 = _mm256_shuffle_epi8(rgba1, shuffle_mask);
    let rgb2 = _mm256_shuffle_epi8(rgba2, shuffle_mask);
    let rgb3 = _mm256_shuffle_epi8(rgba3, shuffle_mask);

    _mm256_storeu_si256(ptr as *mut __m256i, rgb0);
    _mm256_storeu_si256(ptr.add(24) as *mut __m256i, rgb1);
    _mm256_storeu_si256(ptr.add(48) as *mut __m256i, rgb2);
    if use_transient {
        _mm256_storeu_si256(ptr.add(72) as *mut __m256i, rgb3);
    } else {
        // let mut transient: [u8; 32] = [0u8; 32];
        // _mm256_storeu_si256(transient.as_mut_ptr() as *mut __m256i, rgb3);
        // std::ptr::copy_nonoverlapping(transient.as_ptr(), ptr.add(72), 24);
        std::ptr::copy_nonoverlapping(&rgb3 as *const _ as *const u8, ptr.add(72), 24);
    }
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
pub unsafe fn store_u8_rgba_avx2(ptr: *mut u8, r: __m256i, g: __m256i, b: __m256i, a: __m256i) {
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
pub unsafe fn store_u8_rgba_sse(ptr: *mut u8, r: __m128i, g: __m128i, b: __m128i, a: __m128i) {
    let rg_lo = _mm_unpacklo_epi8(r, g);
    let rg_hi = _mm_unpackhi_epi8(r, g);
    let ba_lo = _mm_unpacklo_epi8(b, a);
    let ba_hi = _mm_unpackhi_epi8(b, a);

    let rgba_0_lo = _mm_unpacklo_epi16(rg_lo, ba_lo);
    let rgba_0_hi = _mm_unpackhi_epi16(rg_lo, ba_lo);
    let rgba_1_lo = _mm_unpacklo_epi16(rg_hi, ba_hi);
    let rgba_1_hi = _mm_unpackhi_epi16(rg_hi, ba_hi);

    _mm_storeu_si128(ptr as *mut __m128i, rgba_0_lo);
    _mm_storeu_si128(ptr.add(16) as *mut __m128i, rgba_0_hi);
    _mm_storeu_si128(ptr.add(32) as *mut __m128i, rgba_1_lo);
    _mm_storeu_si128(ptr.add(48) as *mut __m128i, rgba_1_hi);
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
pub unsafe fn store_u8_rgb_sse(
    ptr: *mut u8,
    r: __m128i,
    g: __m128i,
    b: __m128i,
    use_transient: bool,
) {
    #[rustfmt::skip]
        let shuffle = _mm_setr_epi8(0, 1, 2,
                                    4, 5, 6,
                                    8, 9, 10,
                                    12, 13, 14,
                                    -1, -1, -1, -1);
    let zeros = _mm_setzero_si128();
    let rg_lo = _mm_unpacklo_epi8(r, g);
    let rg_hi = _mm_unpackhi_epi8(r, g);
    let ba_lo = _mm_unpacklo_epi8(b, zeros);
    let ba_hi = _mm_unpackhi_epi8(b, zeros);

    let rgba_0_lo = _mm_unpacklo_epi16(rg_lo, ba_lo);
    let rgba_0_hi = _mm_unpackhi_epi16(rg_lo, ba_lo);
    let rgba_1_lo = _mm_unpacklo_epi16(rg_hi, ba_hi);
    let rgba_1_hi = _mm_unpackhi_epi16(rg_hi, ba_hi);

    let rgb0 = _mm_shuffle_epi8(rgba_0_lo, shuffle);
    let rgb1 = _mm_shuffle_epi8(rgba_0_hi, shuffle);
    let rgb2 = _mm_shuffle_epi8(rgba_1_lo, shuffle);
    let rgb3 = _mm_shuffle_epi8(rgba_1_hi, shuffle);

    _mm_storeu_si128(ptr as *mut __m128i, rgb0);
    _mm_storeu_si128(ptr.add(12) as *mut __m128i, rgb1);
    _mm_storeu_si128(ptr.add(24) as *mut __m128i, rgb2);
    if use_transient {
        _mm_storeu_si128(ptr.add(36) as *mut __m128i, rgb3);
    } else {
        std::ptr::copy(&rgb3 as *const _ as *const u8, ptr.add(36), 12);
    }
}
