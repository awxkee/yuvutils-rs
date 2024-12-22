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
#[cfg(feature = "nightly_avx512")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
#[cfg(feature = "nightly_avx512")]
use std::arch::x86_64::*;

use crate::avx512bw::avx512_setr::{_v512_set_epu16, _v512_set_epu32, _v512_set_epu8};
use crate::yuv_support::YuvSourceChannels;

#[inline(always)]
pub(crate) unsafe fn avx512_pack_u16(lo: __m512i, hi: __m512i) -> __m512i {
    let mask = _mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7);
    _mm512_permutexvar_epi64(mask, _mm512_packus_epi16(lo, hi))
}

#[inline(always)]
pub(crate) unsafe fn avx512_pack_u32(lo: __m512i, hi: __m512i) -> __m512i {
    let mask = _mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7);
    _mm512_permutexvar_epi64(mask, _mm512_packus_epi32(lo, hi))
}

#[inline(always)]
pub(crate) unsafe fn avx512_interleave_rgb16(
    a: __m512i,
    b: __m512i,
    c: __m512i,
) -> (__m512i, __m512i, __m512i) {
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
    let b0g0b2 = _mm512_permutex2var_epi16(a, mask0, b);
    let r1b1r0 = _mm512_permutex2var_epi16(a, mask1, c);
    let g2r2g1 = _mm512_permutex2var_epi16(b, mask2, c);

    let bgr0 = _mm512_mask_blend_epi16(0x24924924, b0g0b2, r1b1r0);
    let bgr1 = _mm512_mask_blend_epi16(0x24924924, r1b1r0, g2r2g1);
    let bgr2 = _mm512_mask_blend_epi16(0x24924924, g2r2g1, b0g0b2);
    (bgr0, bgr1, bgr2)
}

#[inline(always)]
pub(crate) unsafe fn avx512_deinterleave_rgb16(
    a: __m512i,
    b: __m512i,
    c: __m512i,
) -> (__m512i, __m512i, __m512i) {
    let mask0 = _v512_set_epu16(
        61, 58, 55, 52, 49, 46, 43, 40, 37, 34, 63, 60, 57, 54, 51, 48, 45, 42, 39, 36, 33, 30, 27,
        24, 21, 18, 15, 12, 9, 6, 3, 0,
    );
    let b01g1 = _mm512_permutex2var_epi16(a, mask0, b);
    let r12b2 = _mm512_permutex2var_epi16(b, mask0, c);
    let g20r0 = _mm512_permutex2var_epi16(c, mask0, a);

    let a0 = _mm512_mask_blend_epi32(0xf800, b01g1, r12b2);
    let b0 = _mm512_permutex2var_epi16(
        b,
        _v512_set_epu16(
            42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 29, 26, 23, 20, 17, 14, 11, 8, 5, 2, 53,
            52, 51, 50, 49, 48, 47, 46, 45, 44, 43,
        ),
        g20r0,
    );
    let c0 = _mm512_alignr_epi32::<11>(r12b2, g20r0);
    (a0, b0, c0)
}

#[inline(always)]
pub(crate) unsafe fn avx512_interleave_rgb<const HAS_VBMI: bool>(
    a: __m512i,
    b: __m512i,
    c: __m512i,
) -> (__m512i, __m512i, __m512i) {
    if HAS_VBMI {
        let mask0 = _v512_set_epu8(
            127, 84, 20, 126, 83, 19, 125, 82, 18, 124, 81, 17, 123, 80, 16, 122, 79, 15, 121, 78,
            14, 120, 77, 13, 119, 76, 12, 118, 75, 11, 117, 74, 10, 116, 73, 9, 115, 72, 8, 114,
            71, 7, 113, 70, 6, 112, 69, 5, 111, 68, 4, 110, 67, 3, 109, 66, 2, 108, 65, 1, 107, 64,
            0, 106,
        );
        let mask1 = _v512_set_epu8(
            21, 42, 105, 20, 41, 104, 19, 40, 103, 18, 39, 102, 17, 38, 101, 16, 37, 100, 15, 36,
            99, 14, 35, 98, 13, 34, 97, 12, 33, 96, 11, 32, 95, 10, 31, 94, 9, 30, 93, 8, 29, 92,
            7, 28, 91, 6, 27, 90, 5, 26, 89, 4, 25, 88, 3, 24, 87, 2, 23, 86, 1, 22, 85, 0,
        );
        let mask2 = _v512_set_epu8(
            106, 127, 63, 105, 126, 62, 104, 125, 61, 103, 124, 60, 102, 123, 59, 101, 122, 58,
            100, 121, 57, 99, 120, 56, 98, 119, 55, 97, 118, 54, 96, 117, 53, 95, 116, 52, 94, 115,
            51, 93, 114, 50, 92, 113, 49, 91, 112, 48, 90, 111, 47, 89, 110, 46, 88, 109, 45, 87,
            108, 44, 86, 107, 43, 85,
        );
        let r2g0r0 = _mm512_permutex2var_epi8(b, mask0, c);
        let b0r1b1 = _mm512_permutex2var_epi8(a, mask1, c);
        let g1b2g2 = _mm512_permutex2var_epi8(a, mask2, b);

        let bgr0 = _mm512_mask_blend_epi8(0x9249249249249249, r2g0r0, b0r1b1);
        let bgr1 = _mm512_mask_blend_epi8(0x9249249249249249, b0r1b1, g1b2g2);
        let bgr2 = _mm512_mask_blend_epi8(0x9249249249249249, g1b2g2, r2g0r0);
        (bgr0, bgr1, bgr2)
    } else {
        let g1g0 = _mm512_shuffle_epi8(
            b,
            _mm512_set4_epi32(0x0e0f0c0d, 0x0a0b0809, 0x06070405, 0x02030001),
        );
        let b0g0 = _mm512_mask_blend_epi8(0xAAAAAAAAAAAAAAAA, a, g1g0);
        let r0b1 = _mm512_mask_blend_epi8(0xAAAAAAAAAAAAAAAA, c, a);
        let g1r1 = _mm512_mask_blend_epi8(0xAAAAAAAAAAAAAAAA, g1g0, c);

        let mask0 = _v512_set_epu16(
            42, 10, 31, 41, 9, 30, 40, 8, 29, 39, 7, 28, 38, 6, 27, 37, 5, 26, 36, 4, 25, 35, 3,
            24, 34, 2, 23, 33, 1, 22, 32, 0,
        );
        let mask1 = _v512_set_epu16(
            21, 52, 41, 20, 51, 40, 19, 50, 39, 18, 49, 38, 17, 48, 37, 16, 47, 36, 15, 46, 35, 14,
            45, 34, 13, 44, 33, 12, 43, 32, 11, 42,
        );
        let mask2 = _v512_set_epu16(
            63, 31, 20, 62, 30, 19, 61, 29, 18, 60, 28, 17, 59, 27, 16, 58, 26, 15, 57, 25, 14, 56,
            24, 13, 55, 23, 12, 54, 22, 11, 53, 21,
        );
        let b0g0b2 = _mm512_permutex2var_epi16(b0g0, mask0, r0b1);
        let r1b1r0 = _mm512_permutex2var_epi16(b0g0, mask1, g1r1);
        let g2r2g1 = _mm512_permutex2var_epi16(r0b1, mask2, g1r1);

        let bgr0 = _mm512_mask_blend_epi16(0x24924924, b0g0b2, r1b1r0);
        let bgr1 = _mm512_mask_blend_epi16(0x24924924, r1b1r0, g2r2g1);
        let bgr2 = _mm512_mask_blend_epi16(0x24924924, g2r2g1, b0g0b2);
        (bgr0, bgr1, bgr2)
    }
}

#[inline(always)]
pub(crate) unsafe fn avx512_put_rgb_u8<const HAS_VBMI: bool>(
    dst: *mut u8,
    a: __m512i,
    b: __m512i,
    c: __m512i,
) {
    let (rgb0, rgb1, rgb2) = avx512_interleave_rgb::<HAS_VBMI>(a, b, c);
    _mm512_storeu_si512(dst as *mut i32, rgb0);
    _mm512_storeu_si512(dst.add(64) as *mut i32, rgb1);
    _mm512_storeu_si512(dst.add(128) as *mut i32, rgb2);
}

#[inline(always)]
pub(crate) unsafe fn avx512_put_rgb16(dst: *mut u16, a: __m512i, b: __m512i, c: __m512i) {
    let (rgb0, rgb1, rgb2) = avx512_interleave_rgb16(a, b, c);
    _mm512_storeu_si512(dst as *mut i32, rgb0);
    _mm512_storeu_si512(dst.add(32) as *mut i32, rgb1);
    _mm512_storeu_si512(dst.add(64) as *mut i32, rgb2);
}

#[inline(always)]
pub(crate) unsafe fn avx512_put_half_rgb_u8<const HAS_VBMI: bool>(
    dst: *mut u8,
    a: __m512i,
    b: __m512i,
    c: __m512i,
) {
    let (rgb0, rgb1, _) = avx512_interleave_rgb::<HAS_VBMI>(a, b, c);
    _mm512_storeu_si512(dst as *mut i32, rgb0);
    _mm256_storeu_si256(dst.add(64) as *mut __m256i, _mm512_castsi512_si256(rgb1));
}

#[inline(always)]
pub(crate) unsafe fn avx512_interleave_rgba16(
    a: __m512i,
    b: __m512i,
    c: __m512i,
    d: __m512i,
) -> (__m512i, __m512i, __m512i, __m512i) {
    let (br01, br23) = avx512_zip_epi16(a, c);
    let (ga01, ga23) = avx512_zip_epi16(b, d);
    let (bgra0, bgra1) = avx512_zip_epi16(br01, ga01);
    let (bgra2, bgra3) = avx512_zip_epi16(br23, ga23);
    (bgra0, bgra1, bgra2, bgra3)
}

#[inline(always)]
pub(crate) unsafe fn avx512_deinterleave_rgba16(
    a: __m512i,
    b: __m512i,
    c: __m512i,
    d: __m512i,
) -> (__m512i, __m512i, __m512i, __m512i) {
    let mask0 = _v512_set_epu16(
        62, 60, 58, 56, 54, 52, 50, 48, 46, 44, 42, 40, 38, 36, 34, 32, 30, 28, 26, 24, 22, 20, 18,
        16, 14, 12, 10, 8, 6, 4, 2, 0,
    );
    let mask1 = _v512_set_epu16(
        63, 61, 59, 57, 55, 53, 51, 49, 47, 45, 43, 41, 39, 37, 35, 33, 31, 29, 27, 25, 23, 21, 19,
        17, 15, 13, 11, 9, 7, 5, 3, 1,
    );

    let br01 = _mm512_permutex2var_epi16(a, mask0, b);
    let ga01 = _mm512_permutex2var_epi16(a, mask1, b);
    let br23 = _mm512_permutex2var_epi16(c, mask0, d);
    let ga23 = _mm512_permutex2var_epi16(c, mask1, d);

    let a0 = _mm512_permutex2var_epi16(br01, mask0, br23);
    let c0 = _mm512_permutex2var_epi16(br01, mask1, br23);
    let b0 = _mm512_permutex2var_epi16(ga01, mask0, ga23);
    let d0 = _mm512_permutex2var_epi16(ga01, mask1, ga23);
    (a0, b0, c0, d0)
}

#[inline(always)]
pub(crate) unsafe fn avx512_interleave_rgba<const HAS_VBMI: bool>(
    a: __m512i,
    b: __m512i,
    c: __m512i,
    d: __m512i,
) -> (__m512i, __m512i, __m512i, __m512i) {
    let (br01, br23) = avx512_zip_epi8::<HAS_VBMI>(a, c);
    let (ga01, ga23) = avx512_zip_epi8::<HAS_VBMI>(b, d);
    let (bgra0, bgra1) = avx512_zip_epi8::<HAS_VBMI>(br01, ga01);
    let (bgra2, bgra3) = avx512_zip_epi8::<HAS_VBMI>(br23, ga23);
    (bgra0, bgra1, bgra2, bgra3)
}

#[inline(always)]
pub(crate) unsafe fn avx512_put_rgba_u8<const HAS_VBMI: bool>(
    dst: *mut u8,
    a: __m512i,
    b: __m512i,
    c: __m512i,
    d: __m512i,
) {
    let (rgb0, rgb1, rgb2, rgb3) = avx512_interleave_rgba::<HAS_VBMI>(a, b, c, d);
    _mm512_storeu_si512(dst as *mut i32, rgb0);
    _mm512_storeu_si512(dst.add(64) as *mut i32, rgb1);
    _mm512_storeu_si512(dst.add(128) as *mut i32, rgb2);
    _mm512_storeu_si512(dst.add(128 + 64) as *mut i32, rgb3);
}

#[inline(always)]
pub(crate) unsafe fn avx512_put_half_rgba_u8<const HAS_VBMI: bool>(
    dst: *mut u8,
    a: __m512i,
    b: __m512i,
    c: __m512i,
    d: __m512i,
) {
    let (rgb0, rgb1, _, _) = avx512_interleave_rgba::<HAS_VBMI>(a, b, c, d);
    _mm512_storeu_si512(dst as *mut i32, rgb0);
    _mm512_storeu_si512(dst.add(64) as *mut i32, rgb1);
}

#[inline(always)]
pub(crate) unsafe fn avx512_put_rgba16(
    dst: *mut u16,
    a: __m512i,
    b: __m512i,
    c: __m512i,
    d: __m512i,
) {
    let (rgb0, rgb1, rgb2, rgb3) = avx512_interleave_rgba16(a, b, c, d);
    _mm512_storeu_si512(dst as *mut i32, rgb0);
    _mm512_storeu_si512(dst.add(32) as *mut i32, rgb1);
    _mm512_storeu_si512(dst.add(64) as *mut i32, rgb2);
    _mm512_storeu_si512(dst.add(96) as *mut i32, rgb3);
}

#[inline(always)]
pub(crate) unsafe fn avx512_div_by255(v: __m512i) -> __m512i {
    let addition = _mm512_set1_epi16(127);
    _mm512_srli_epi16::<8>(_mm512_add_epi16(
        _mm512_add_epi16(v, addition),
        _mm512_srli_epi16::<8>(v),
    ))
}

#[inline(always)]
pub(crate) unsafe fn avx512_deinterleave_rgb<const HAS_VBMI: bool>(
    bgr0: __m512i,
    bgr1: __m512i,
    bgr2: __m512i,
) -> (__m512i, __m512i, __m512i) {
    if HAS_VBMI {
        let b0g0b1 = _mm512_mask_blend_epi8(0xb6db6db6db6db6db, bgr1, bgr0);
        let g1r1g2 = _mm512_mask_blend_epi8(0xb6db6db6db6db6db, bgr2, bgr1);
        let r2b2r0 = _mm512_mask_blend_epi8(0xb6db6db6db6db6db, bgr0, bgr2);
        /*
                   _v512_set_epu8(
               125,
               122, 119, 116,
               113, 110, 107,
               104, 101, 98,
               95, 92, 89,
               86, 83, 80,
               77, 74, 71,
               68, 65, 63,
               62, 60, 59,
               57, 56, 54,
               53, 51, 50,
               48, 47, 45,
               44, 42, 41,
               39, 38, 36,
               35, 33, 32,
               30, 29, 27,
               26, 24, 23,
               21, 20, 18,
               17, 15, 14,
               12, 11, 9,
               8, 6, 5,
               3, 2, 0,
           ),
        */
        let a = _mm512_permutex2var_epi8(
            b0g0b1,
            _v512_set_epu8(
                125, 122, 119, 116, 113, 110, 107, 104, 101, 98, 95, 92, 89, 86, 83, 80, 77, 74,
                71, 68, 65, 62, 59, 56, 53, 50, 47, 44, 41, 38, 35, 32, 29, 26, 23, 20, 17, 14, 11,
                8, 5, 2, 63, 60, 57, 54, 51, 48, 45, 42, 39, 36, 33, 30, 27, 24, 21, 18, 15, 12, 9,
                6, 3, 0,
            ),
            bgr2,
        );
        /*
        [63, 61, 60, 58, 57, 55, 54, 52, 51, 49, 48, 46, 45, 43, 42, 40, 39, 37, 36, 34, 33, 31, 30, 28, 27,
         25, 24, 23,
         21, 20, 18,
         17, 15, 14,
         12, 11, 9,
         8, 6, 5,
         3, 2, 0,
         62, 59, 56,
         53, 50, 47,
         44, 41, 38,
         35, 32, 29,
         26, 23, 20,
         17, 14, 11,
         8, 5, 2]
         */
        /*
        63,
               62, 60, 59,
               57, 56, 54,
               57, 51, 50,
               48, 47, 45,
               44, 42, 41,
               39, 38, 36,
               35, 33, 32,
               30, 29, 27,
               26, 24, 23,
               21, 20, 18,
               17, 15, 14,
               12, 11, 9,
               8, 6, 5,
               3, 2, 0,
               61 | 0x40, 58 | 0x40, 55 | 0x40,
               52 | 0x40, 49 | 0x40, 46 | 0x40,
               43 | 0x40, 40 | 0x40, 37 | 0x40,
               34 | 0x40, 31 | 0x40, 28 | 0x40,
               25 | 0x40, 22 | 0x40, 19 | 0x40,
               16 | 0x40, 13 | 0x40, 10 | 0x40,
               7 | 0x40, 4 | 0x40, 1 | 0x40,
        */
        let b = _mm512_permutex2var_epi8(
            g1r1g2,
            _v512_set_epu8(
                62, 59, 56, 58, 50, 47, 44, 41, 38, 35, 32, 29, 26, 23, 20, 17, 14, 11, 8, 5, 2, 63,
                60, 57, 54, 51, 48, 45, 42, 39, 36, 33, 30, 27, 24, 21, 18, 15, 12, 9, 6, 3, 0,
                125, 122, 119, 116, 113, 110, 107, 104, 101, 98, 95, 92, 89, 86, 83, 80, 77, 74,
                71, 68, 65,
            ),
            bgr0,
        );
        let c = _mm512_permutex2var_epi8(
            r2b2r0,
            _v512_set_epu8(
                63, 60, 57, 54, 51, 48, 45, 42, 39, 36, 33, 30, 27, 24, 21, 18, 15, 12, 9, 6, 3, 0,
                125, 122, 119, 116, 113, 110, 107, 104, 101, 98, 95, 92, 89, 86, 83, 80, 77, 74,
                71, 68, 65, 62, 59, 56, 53, 50, 47, 44, 41, 38, 35, 32, 29, 26, 23, 20, 17, 14, 11,
                8, 5, 2,
            ),
            bgr1,
        );
        (a, b, c)
    } else {
        let mask0 = _v512_set_epu16(
            61, 58, 55, 52, 49, 46, 43, 40, 37, 34, 63, 60, 57, 54, 51, 48, 45, 42, 39, 36, 33, 30,
            27, 24, 21, 18, 15, 12, 9, 6, 3, 0,
        );
        let b01g1 = _mm512_permutex2var_epi16(bgr0, mask0, bgr1);
        let r12b2 = _mm512_permutex2var_epi16(bgr1, mask0, bgr2);
        let g20r0 = _mm512_permutex2var_epi16(bgr2, mask0, bgr0);

        let b0g0 = _mm512_mask_blend_epi32(0xf800, b01g1, r12b2);
        let r0b1 = _mm512_permutex2var_epi16(
            bgr1,
            _v512_set_epu16(
                42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 29, 26, 23, 20, 17, 14, 11, 8, 5, 2,
                53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43,
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
}

#[inline(always)]
pub(crate) unsafe fn avx512_deinterleave_rgba<const HAS_VBMI: bool>(
    bgra0: __m512i,
    bgra1: __m512i,
    bgra2: __m512i,
    bgra3: __m512i,
) -> (__m512i, __m512i, __m512i, __m512i) {
    if HAS_VBMI {
        let mask0 = _v512_set_epu8(
            126, 124, 122, 120, 118, 116, 114, 112, 110, 108, 106, 104, 102, 100, 98, 96, 94, 92,
            90, 88, 86, 84, 82, 80, 78, 76, 74, 72, 70, 68, 66, 64, 62, 60, 58, 56, 54, 52, 50, 48,
            46, 44, 42, 40, 38, 36, 34, 32, 30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2,
            0,
        );
        let mask1 = _v512_set_epu8(
            127, 125, 123, 121, 119, 117, 115, 113, 111, 109, 107, 105, 103, 101, 99, 97, 95, 93,
            91, 89, 87, 85, 83, 81, 79, 77, 75, 73, 71, 69, 67, 65, 63, 61, 59, 57, 55, 53, 51, 49,
            47, 45, 43, 41, 39, 37, 35, 33, 31, 29, 27, 25, 23, 21, 19, 17, 15, 13, 11, 9, 7, 5, 3,
            1,
        );

        let br01 = _mm512_permutex2var_epi8(bgra0, mask0, bgra1);
        let ga01 = _mm512_permutex2var_epi8(bgra0, mask1, bgra1);
        let br23 = _mm512_permutex2var_epi8(bgra2, mask0, bgra3);
        let ga23 = _mm512_permutex2var_epi8(bgra2, mask1, bgra3);

        let a = _mm512_permutex2var_epi8(br01, mask0, br23);
        let c = _mm512_permutex2var_epi8(br01, mask1, br23);
        let b = _mm512_permutex2var_epi8(ga01, mask0, ga23);
        let d = _mm512_permutex2var_epi8(ga01, mask1, ga23);
        (a, b, c, d)
    } else {
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
}

#[inline(always)]
pub(crate) unsafe fn avx512_pairwise_widen_avg(v: __m512i) -> __m512i {
    let sums = _mm512_maddubs_epi16(v, _mm512_set1_epi8(1));
    let shifted = _mm512_srli_epi16::<1>(_mm512_add_epi16(sums, _mm512_set1_epi16(1)));
    let packed_lo = _mm512_packus_epi16(shifted, shifted);
    let mask = _mm512_setr_epi64(0, 2, 4, 6, 1, 3, 5, 7);
    _mm512_permutexvar_epi64(mask, packed_lo)
}

pub(crate) const fn shuffle(z: u32, y: u32, x: u32, w: u32) -> i32 {
    // Checked: we want to reinterpret the bits
    ((z << 6) | (y << 4) | (x << 2) | w) as i32
}

#[inline(always)]
pub(crate) unsafe fn avx2_zip(a: __m256i, b: __m256i) -> (__m256i, __m256i) {
    const MASK: i32 = shuffle(3, 1, 2, 0);
    let v0 = _mm256_permute4x64_epi64::<MASK>(a);
    let v1 = _mm256_permute4x64_epi64::<MASK>(b);
    let b0 = _mm256_unpacklo_epi8(v0, v1);
    let b1 = _mm256_unpackhi_epi8(v0, v1);
    (b0, b1)
}

#[inline(always)]
pub(crate) unsafe fn avx512_store_rgba_for_yuv_u8<const DN: u8, const HAS_VBMI: bool>(
    dst: *mut u8,
    r: __m512i,
    g: __m512i,
    b: __m512i,
    a: __m512i,
) {
    let destination_channels: YuvSourceChannels = DN.into();
    match destination_channels {
        YuvSourceChannels::Rgb => {
            avx512_put_rgb_u8::<HAS_VBMI>(dst, r, g, b);
        }
        YuvSourceChannels::Bgr => {
            avx512_put_rgb_u8::<HAS_VBMI>(dst, b, g, r);
        }
        YuvSourceChannels::Rgba => {
            avx512_put_rgba_u8::<HAS_VBMI>(dst, r, g, b, a);
        }
        YuvSourceChannels::Bgra => {
            avx512_put_rgba_u8::<HAS_VBMI>(dst, b, g, r, a);
        }
    }
}

#[inline(always)]
pub(crate) unsafe fn avx512_store_half_rgba_for_yuv_u8<const DN: u8, const HAS_VBMI: bool>(
    dst: *mut u8,
    r: __m512i,
    g: __m512i,
    b: __m512i,
    a: __m512i,
) {
    let destination_channels: YuvSourceChannels = DN.into();
    match destination_channels {
        YuvSourceChannels::Rgb => {
            avx512_put_half_rgb_u8::<HAS_VBMI>(dst, r, g, b);
        }
        YuvSourceChannels::Bgr => {
            avx512_put_half_rgb_u8::<HAS_VBMI>(dst, b, g, r);
        }
        YuvSourceChannels::Rgba => {
            avx512_put_half_rgba_u8::<HAS_VBMI>(dst, r, g, b, a);
        }
        YuvSourceChannels::Bgra => {
            avx512_put_half_rgba_u8::<HAS_VBMI>(dst, b, g, r, a);
        }
    }
}

#[inline(always)]
pub(crate) unsafe fn avx512_store_rgba16_for_yuv<const DN: u8>(
    dst: *mut u16,
    r: __m512i,
    g: __m512i,
    b: __m512i,
    a: __m512i,
) {
    let destination_channels: YuvSourceChannels = DN.into();
    match destination_channels {
        YuvSourceChannels::Rgb => {
            avx512_put_rgb16(dst, r, g, b);
        }
        YuvSourceChannels::Bgr => {
            avx512_put_rgb16(dst, b, g, r);
        }
        YuvSourceChannels::Rgba => {
            avx512_put_rgba16(dst, r, g, b, a);
        }
        YuvSourceChannels::Bgra => {
            avx512_put_rgba16(dst, b, g, r, a);
        }
    }
}

#[inline(always)]
pub(crate) unsafe fn avx512_unzip_epi8<const HAS_VBMI: bool>(
    a: __m512i,
    b: __m512i,
) -> (__m512i, __m512i) {
    if HAS_VBMI {
        let mask0 = _v512_set_epu8(
            126, 124, 122, 120, 118, 116, 114, 112, 110, 108, 106, 104, 102, 100, 98, 96, 94, 92,
            90, 88, 86, 84, 82, 80, 78, 76, 74, 72, 70, 68, 66, 64, 62, 60, 58, 56, 54, 52, 50, 48,
            46, 44, 42, 40, 38, 36, 34, 32, 30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2,
            0,
        );
        let mask1 = _v512_set_epu8(
            127, 125, 123, 121, 119, 117, 115, 113, 111, 109, 107, 105, 103, 101, 99, 97, 95, 93,
            91, 89, 87, 85, 83, 81, 79, 77, 75, 73, 71, 69, 67, 65, 63, 61, 59, 57, 55, 53, 51, 49,
            47, 45, 43, 41, 39, 37, 35, 33, 31, 29, 27, 25, 23, 21, 19, 17, 15, 13, 11, 9, 7, 5, 3,
            1,
        );
        let a0 = _mm512_permutex2var_epi8(a, mask0, b);
        let b0 = _mm512_permutex2var_epi8(a, mask1, b);
        (a0, b0)
    } else {
        let mask0 = _mm512_set4_epi32(0x0f0d0b09, 0x07050301, 0x0e0c0a08, 0x06040200);
        let a0b0 = _mm512_shuffle_epi8(a, mask0);
        let a1b1 = _mm512_shuffle_epi8(b, mask0);
        let mask1 = _mm512_set_epi64(14, 12, 10, 8, 6, 4, 2, 0);
        let mask2 = _mm512_set_epi64(15, 13, 11, 9, 7, 5, 3, 1);
        let a0 = _mm512_permutex2var_epi64(a0b0, mask1, a1b1);
        let b0 = _mm512_permutex2var_epi64(a0b0, mask2, a1b1);
        (a0, b0)
    }
}

#[inline(always)]
pub(crate) unsafe fn avx512_zip_epi16(a: __m512i, b: __m512i) -> (__m512i, __m512i) {
    let mask0 = _v512_set_epu16(
        47, 15, 46, 14, 45, 13, 44, 12, 43, 11, 42, 10, 41, 9, 40, 8, 39, 7, 38, 6, 37, 5, 36, 4,
        35, 3, 34, 2, 33, 1, 32, 0,
    );
    let ab0 = _mm512_permutex2var_epi16(a, mask0, b);
    let mask1 = _v512_set_epu16(
        63, 31, 62, 30, 61, 29, 60, 28, 59, 27, 58, 26, 57, 25, 56, 24, 55, 23, 54, 22, 53, 21, 52,
        20, 51, 19, 50, 18, 49, 17, 48, 16,
    );
    let ab1 = _mm512_permutex2var_epi16(a, mask1, b);
    (ab0, ab1)
}

#[inline(always)]
pub(crate) unsafe fn avx512_zip_epi8<const HAS_VBMI: bool>(
    a: __m512i,
    b: __m512i,
) -> (__m512i, __m512i) {
    if HAS_VBMI {
        let mask0 = _v512_set_epu8(
            95, 31, 94, 30, 93, 29, 92, 28, 91, 27, 90, 26, 89, 25, 88, 24, 87, 23, 86, 22, 85, 21,
            84, 20, 83, 19, 82, 18, 81, 17, 80, 16, 79, 15, 78, 14, 77, 13, 76, 12, 75, 11, 74, 10,
            73, 9, 72, 8, 71, 7, 70, 6, 69, 5, 68, 4, 67, 3, 66, 2, 65, 1, 64, 0,
        );
        let ab0 = _mm512_permutex2var_epi8(a, mask0, b);
        let mask1 = _v512_set_epu8(
            127, 63, 126, 62, 125, 61, 124, 60, 123, 59, 122, 58, 121, 57, 120, 56, 119, 55, 118,
            54, 117, 53, 116, 52, 115, 51, 114, 50, 113, 49, 112, 48, 111, 47, 110, 46, 109, 45,
            108, 44, 107, 43, 106, 42, 105, 41, 104, 40, 103, 39, 102, 38, 101, 37, 100, 36, 99,
            35, 98, 34, 97, 33, 96, 32,
        );
        let ab1 = _mm512_permutex2var_epi8(a, mask1, b);
        (ab0, ab1)
    } else {
        let low = _mm512_unpacklo_epi8(a, b);
        let high = _mm512_unpackhi_epi8(a, b);
        let ab0 = _mm512_permutex2var_epi64(low, _mm512_set_epi64(11, 10, 3, 2, 9, 8, 1, 0), high);
        let ab1 =
            _mm512_permutex2var_epi64(low, _mm512_set_epi64(15, 14, 7, 6, 13, 12, 5, 4), high);
        (ab0, ab1)
    }
}

#[inline(always)]
pub(crate) unsafe fn avx512_load_rgb_u8<const CN: u8, const HAS_VBMI: bool>(
    src: *const u8,
) -> (__m512i, __m512i, __m512i) {
    let source_channels: YuvSourceChannels = CN.into();
    let (r_values, g_values, b_values);
    match source_channels {
        YuvSourceChannels::Rgb | YuvSourceChannels::Bgr => {
            let row_1 = _mm512_loadu_si512(src as *const i32);
            let row_2 = _mm512_loadu_si512(src.add(64) as *const i32);
            let row_3 = _mm512_loadu_si512(src.add(128) as *const i32);

            let (it1, it2, it3) = avx512_deinterleave_rgb::<HAS_VBMI>(row_1, row_2, row_3);
            if source_channels == YuvSourceChannels::Rgb {
                r_values = it1;
                g_values = it2;
                b_values = it3;
            } else {
                r_values = it3;
                g_values = it2;
                b_values = it1;
            }
        }
        YuvSourceChannels::Rgba | YuvSourceChannels::Bgra => {
            let row_1 = _mm512_loadu_si512(src as *const i32);
            let row_2 = _mm512_loadu_si512(src.add(64) as *const i32);
            let row_3 = _mm512_loadu_si512(src.add(128) as *const i32);
            let row_4 = _mm512_loadu_si512(src.add(128 + 64) as *const i32);

            let (it1, it2, it3, _) =
                avx512_deinterleave_rgba::<HAS_VBMI>(row_1, row_2, row_3, row_4);
            if source_channels == YuvSourceChannels::Rgba {
                r_values = it1;
                g_values = it2;
                b_values = it3;
            } else {
                r_values = it3;
                g_values = it2;
                b_values = it1;
            }
        }
    }
    (r_values, g_values, b_values)
}

#[inline(always)]
pub(crate) unsafe fn avx512_load_half_rgb_u8<const CN: u8, const HAS_VBMI: bool>(
    src: *const u8,
) -> (__m512i, __m512i, __m512i) {
    let source_channels: YuvSourceChannels = CN.into();
    let (r_values, g_values, b_values);
    match source_channels {
        YuvSourceChannels::Rgb | YuvSourceChannels::Bgr => {
            let row_1 = _mm512_loadu_si512(src as *const i32);
            let row_2 = _mm256_loadu_si256(src.add(64) as *const __m256i);

            let (it1, it2, it3) = avx512_deinterleave_rgb::<HAS_VBMI>(
                row_1,
                _mm512_castsi256_si512(row_2),
                _mm512_setzero_si512(),
            );
            if source_channels == YuvSourceChannels::Rgb {
                r_values = it1;
                g_values = it2;
                b_values = it3;
            } else {
                r_values = it3;
                g_values = it2;
                b_values = it1;
            }
        }
        YuvSourceChannels::Rgba | YuvSourceChannels::Bgra => {
            let row_1 = _mm512_loadu_si512(src as *const i32);
            let row_2 = _mm512_loadu_si512(src.add(64) as *const i32);
            let (it1, it2, it3, _) = avx512_deinterleave_rgba::<HAS_VBMI>(
                row_1,
                row_2,
                _mm512_setzero_si512(),
                _mm512_setzero_si512(),
            );
            if source_channels == YuvSourceChannels::Rgba {
                r_values = it1;
                g_values = it2;
                b_values = it3;
            } else {
                r_values = it3;
                g_values = it2;
                b_values = it1;
            }
        }
    }
    (r_values, g_values, b_values)
}

#[inline(always)]
pub(crate) unsafe fn avx512_pairwise_avg_epi16_epi8(a: __m512i, b: __m512i) -> __m512i {
    let v = _mm512_avg_epu8(a, b);
    _mm512_srli_epi16::<1>(_mm512_add_epi16(
        _mm512_maddubs_epi16(v, _mm512_set1_epi8(1)),
        _mm512_set1_epi16(1),
    ))
}

#[inline(always)]
pub(crate) unsafe fn avx512_pairwise_avg_epi8(a: __m512i) -> __m512i {
    _mm512_srli_epi16::<1>(_mm512_add_epi16(
        _mm512_maddubs_epi16(a, _mm512_set1_epi8(1)),
        _mm512_set1_epi16(1),
    ))
}

#[inline(always)]
pub(crate) unsafe fn _mm512_from_msb_epi16<const BIT_DEPTH: usize>(a: __m512i) -> __m512i {
    if BIT_DEPTH == 10 {
        _mm512_srli_epi16::<6>(a)
    } else if BIT_DEPTH == 12 {
        _mm512_srli_epi16::<4>(a)
    } else if BIT_DEPTH == 14 {
        _mm512_srli_epi16::<2>(a)
    } else {
        a
    }
}

#[inline(always)]
pub(crate) unsafe fn _mm512_store_shr_epi16_epi8<const BIT_DEPTH: usize>(a: __m512i) -> __m512i {
    if BIT_DEPTH == 10 {
        _mm512_srai_epi16::<2>(a)
    } else if BIT_DEPTH == 12 {
        _mm512_srai_epi16::<4>(a)
    } else if BIT_DEPTH == 14 {
        _mm512_srai_epi16::<6>(a)
    } else if BIT_DEPTH == 16 {
        _mm512_srai_epi16::<8>(a)
    } else {
        a
    }
}

#[inline(always)]
pub(crate) unsafe fn avx512_create(lo: __m256i, hi: __m256i) -> __m512i {
    _mm512_inserti64x4::<1>(_mm512_castsi256_si512(lo), hi)
}

#[inline(always)]
pub(crate) unsafe fn _mm512_load_deinterleave_rgb16_for_yuv<const CHANS: u8>(
    ptr: *const u16,
) -> (__m512i, __m512i, __m512i) {
    let r_values;
    let g_values;
    let b_values;

    let source_channels: YuvSourceChannels = CHANS.into();

    let row0 = _mm512_loadu_si512(ptr as *const i32);
    let row1 = _mm512_loadu_si512(ptr.add(32) as *const i32);
    let row2 = _mm512_loadu_si512(ptr.add(64) as *const i32);

    match source_channels {
        YuvSourceChannels::Rgb | YuvSourceChannels::Bgr => {
            let rgb_values = avx512_deinterleave_rgb16(row0, row1, row2);
            if source_channels == YuvSourceChannels::Rgb {
                r_values = rgb_values.0;
                g_values = rgb_values.1;
                b_values = rgb_values.2;
            } else {
                r_values = rgb_values.2;
                g_values = rgb_values.1;
                b_values = rgb_values.0;
            }
        }
        YuvSourceChannels::Rgba => {
            let row3 = _mm512_loadu_si512(ptr.add(96) as *const i32);
            let rgb_values = avx512_deinterleave_rgba16(row0, row1, row2, row3);
            r_values = rgb_values.0;
            g_values = rgb_values.1;
            b_values = rgb_values.2;
        }
        YuvSourceChannels::Bgra => {
            let row3 = _mm512_loadu_si512(ptr.add(96) as *const i32);
            let rgb_values = avx512_deinterleave_rgba16(row0, row1, row2, row3);
            r_values = rgb_values.2;
            g_values = rgb_values.1;
            b_values = rgb_values.0;
        }
    }
    (r_values, g_values, b_values)
}

#[inline(always)]
pub(crate) unsafe fn _mm512_to_msb_epi16<const BIT_DEPTH: usize>(a: __m512i) -> __m512i {
    if BIT_DEPTH == 10 {
        _mm512_slli_epi16::<6>(a)
    } else if BIT_DEPTH == 12 {
        _mm512_slli_epi16::<4>(a)
    } else if BIT_DEPTH == 14 {
        _mm512_slli_epi16::<2>(a)
    } else {
        a
    }
}

#[inline(always)]
pub(crate) unsafe fn avx512_avg_epi16(a: __m512i) -> __m512i {
    let sums = _mm512_madd_epi16(a, _mm512_set1_epi16(1));
    let l0 = _mm512_srli_epi32::<1>(_mm512_add_epi32(sums, _mm512_set1_epi32(1)));
    avx512_pack_u32(l0, l0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn check_rgb8_deinterleaving() {
        unsafe {
            let mut rgb_store = vec![0u8; 64 * 3];
            let has_avx512vbmi = std::is_x86_feature_detected!("avx512vbmi");
            let has_avx512bw = std::is_x86_feature_detected!("avx512bw");
            if !has_avx512bw {
                println!("Launched test on a platform that does not support it");
                return;
            }
            for (i, chunk) in rgb_store.chunks_exact_mut(3).enumerate() {
                chunk[0] = i as u8;
                chunk[1] = i as u8 + 70;
                chunk[2] = i as u8 + 150;
            }
            let deinterleaving = if has_avx512vbmi {
                avx512_load_rgb_u8::<{ YuvSourceChannels::Rgb as u8 }, true>(rgb_store.as_ptr())
            } else {
                avx512_load_rgb_u8::<{ YuvSourceChannels::Rgb as u8 }, false>(rgb_store.as_ptr())
            };
            let mut r_lane = vec![0u8; 64];
            let mut g_lane = vec![0u8; 64];
            let mut b_lane = vec![0u8; 64];
            _mm512_storeu_si512(r_lane.as_mut_ptr() as *mut i32, deinterleaving.0);
            _mm512_storeu_si512(g_lane.as_mut_ptr() as *mut i32, deinterleaving.1);
            _mm512_storeu_si512(b_lane.as_mut_ptr() as *mut i32, deinterleaving.2);
            println!("R lane:");
            for (i, lane) in r_lane.chunks_exact(8).enumerate() {
                for (k, &item) in lane.iter().enumerate() {
                    print!("{}: {}, ", i * 8 + k, item);
                }
                println!();
            }
            println!("\n");
            for (i, lane) in r_lane.chunks_exact(8).enumerate() {
                println!("R lane {i} was {:?}", lane);
            }
            println!("\n");
            println!("G lane:");
            for (i, lane) in g_lane.chunks_exact(8).enumerate() {
                for (k, &item) in lane.iter().enumerate() {
                    print!("{}: {}, ", i * 8 + k, item);
                }
                println!();
            }
            println!();
            for (i, lane) in g_lane.chunks_exact(8).enumerate() {
                println!("G lane {i} was {:?}", lane);
            }
            println!("\n");
            for (i, lane) in b_lane.chunks_exact(8).enumerate() {
                println!("B lane {i} was {:?}", lane);
            }
            assert!(r_lane.iter().all(|&x| x == 1), "R lane was {:?}", r_lane);
            assert!(g_lane.iter().all(|&x| x == 1), "G lane was {:?}", g_lane);
            assert!(b_lane.iter().all(|&x| x == 1), "B lane was {:?}", b_lane);
        }
    }
}
