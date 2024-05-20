#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub unsafe fn store_u8_rgb_avx2(ptr: *mut u8, r: __m256i, g: __m256i, b: __m256i, use_transient: bool) {
    let rg_lo = _mm256_unpacklo_epi8(r, g);
    let rg_hi = _mm256_unpackhi_epi8(r, g);
    let zero = _mm256_setzero_si256();
    let b0_lo = _mm256_unpacklo_epi8(b, zero);
    let b0_hi = _mm256_unpackhi_epi8(b, zero);

    let rgb0_lo = _mm256_unpacklo_epi16(rg_lo, b0_lo);
    let rgb0_hi = _mm256_unpackhi_epi16(rg_lo, b0_lo);
    let rgb1_lo = _mm256_unpacklo_epi16(rg_hi, b0_hi);
    let rgb1_hi = _mm256_unpackhi_epi16(rg_hi, b0_hi);

    let shuffle_mask = _mm256_setr_epi8(
        0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22, 24, 25, 26, 28,
        29, 30, -1, -1, -1, -1, -1, -1, -1, -1,
    );

    let rgb0 = _mm256_shuffle_epi8(rgb0_lo, shuffle_mask);
    let rgb1 = _mm256_shuffle_epi8(rgb0_hi, shuffle_mask);
    let rgb2 = _mm256_shuffle_epi8(rgb1_lo, shuffle_mask);
    let rgb3 = _mm256_shuffle_epi8(rgb1_hi, shuffle_mask);

    _mm256_storeu_si256(ptr as *mut __m256i, rgb0);
    _mm256_storeu_si256(ptr.add(24) as *mut __m256i, rgb1);
    _mm256_storeu_si256(ptr.add(48) as *mut __m256i, rgb2);
    // We need always to write 104 bytes, however 32 initial offset is safe only for 96, then if there are some exceed it is required to use transient buffer
    if use_transient {
        _mm256_storeu_si256(ptr.add(72) as *mut __m256i, rgb3);
    } else {
        let mut transient: [u8; 32] = [0u8; 32];
        _mm256_storeu_si256(transient.as_mut_ptr() as *mut __m256i, rgb3);
        std::ptr::copy_nonoverlapping(
            transient.as_ptr(),
            ptr.add(72),
            24,
        );
    }
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub unsafe fn store_u8_rgba_avx2(ptr: *mut u8, r: __m256i, g: __m256i, b: __m256i, a: __m256i) {
    let rg_low = _mm256_unpacklo_epi8(r, g); // [r0, g0, r1, g1, r2, g2, r3, g3]
    let rg_high = _mm256_unpackhi_epi8(r, g); // [r4, g4, r5, g5, r6, g6, r7, g7]
    let ba_low = _mm256_unpacklo_epi8(b, a); // [b0, a0, b1, a1, b2, a2, b3, a3]
    let ba_high = _mm256_unpackhi_epi8(b, a); // [b4, a4, b5, a5, b6, a6, b7, a7]

    let rgba0 = _mm256_unpacklo_epi16(rg_low, ba_low); // [r0, g0, b0, a0, r1, g1, b1, a1]
    let rgba1 = _mm256_unpackhi_epi16(rg_low, ba_low); // [r2, g2, b2, a2, r3, g3, b3, a3]
    let rgba2 = _mm256_unpacklo_epi16(rg_high, ba_high); // [r4, g4, b4, a4, r5, g5, b5, a5]
    let rgba3 = _mm256_unpackhi_epi16(rg_high, ba_high);

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
    let shuffle = _mm_setr_epi8(0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, -1, -1, -1, -1);
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
