#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
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
pub unsafe fn sse_interleave_even(x: __m128i) -> __m128i {
    #[rustfmt::skip]
        let shuffle = _mm_setr_epi8(0, 0, 2, 2, 4, 4, 6, 6,
                                    8, 8, 10, 10, 12, 12, 14, 14);
    let new_lane = _mm_shuffle_epi8(x, shuffle);
    return new_lane;
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub unsafe fn avx2_concat(vec1: __m128i, vec2: __m128i) -> __m256i {
    return _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(vec1), vec2);
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
    let mask_a = _mm256_set1_epi16(0xF00);
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
        let mut transient: [u8; 32] = [0u8; 32];
        _mm256_storeu_si256(transient.as_mut_ptr() as *mut __m256i, rgb3);
        std::ptr::copy_nonoverlapping(transient.as_ptr(), ptr.add(72), 24);
    }
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub unsafe fn avx2_interleave(a: __m256i, b: __m256i) -> (__m256i, __m256i) {
    let v02 = _mm256_unpacklo_epi16(a, b);
    let v13 = _mm256_unpackhi_epi16(a, b);
    return (
        _mm256_permute2x128_si256::<0x20>(v02, v13),
        _mm256_permute2x128_si256::<0x31>(v02, v13)
    );
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub unsafe fn avx2_interleave_u8(a: __m256i, b: __m256i) -> (__m256i, __m256i) {
    let v02 = _mm256_unpacklo_epi8(a, b);
    let v13 = _mm256_unpackhi_epi8(a, b);
    return (
        _mm256_permute2x128_si256::<0x20>(v02, v13),
        _mm256_permute2x128_si256::<0x31>(v02, v13)
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
        let mut transient: [u8; 16] = [0u8; 16];
        _mm_storeu_si128(transient.as_mut_ptr() as *mut __m128i, rgb3);
        std::ptr::copy(transient.as_ptr(), ptr.add(36), 12);
    }
}
