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
pub(crate) unsafe fn avx512_interleave_rgb(
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

#[inline(always)]
pub(crate) unsafe fn avx512_rgb_u8(dst: *mut u8, a: __m512i, b: __m512i, c: __m512i) {
    let (rgb0, rgb1, rgb2) = avx512_interleave_rgb(a, b, c);
    _mm512_storeu_si512(dst as *mut i32, rgb0);
    _mm512_storeu_si512(dst.add(64) as *mut i32, rgb1);
    _mm512_storeu_si512(dst.add(128) as *mut i32, rgb2);
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
pub(crate) unsafe fn avx512_rgba_u8<const HAS_VBMI: bool>(
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
pub(crate) unsafe fn avx512_div_by255(v: __m512i) -> __m512i {
    let addition = _mm512_set1_epi16(127);
    _mm512_srli_epi16::<8>(_mm512_add_epi16(
        _mm512_add_epi16(v, addition),
        _mm512_srli_epi16::<8>(v),
    ))
}

#[inline(always)]
pub(crate) unsafe fn avx512_deinterleave_rgb(
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

#[inline(always)]
pub(crate) unsafe fn avx512_deinterleave_rgba(
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
pub(crate) unsafe fn avx512_store_u8<const DN: u8, const HAS_VBMI: bool>(
    dst: *mut u8,
    r: __m512i,
    g: __m512i,
    b: __m512i,
    a: __m512i,
) {
    let destination_channels: YuvSourceChannels = DN.into();
    match destination_channels {
        YuvSourceChannels::Rgb => {
            avx512_rgb_u8(dst, r, g, b);
        }
        YuvSourceChannels::Bgr => {
            avx512_rgb_u8(dst, b, g, r);
        }
        YuvSourceChannels::Rgba => {
            avx512_rgba_u8::<HAS_VBMI>(dst, r, g, b, a);
        }
        YuvSourceChannels::Bgra => {
            avx512_rgba_u8::<HAS_VBMI>(dst, b, g, r, a);
        }
    }
}

#[inline(always)]
pub(crate) unsafe fn avx2_unzip_epi8<const HAS_VBMI: bool>(
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
pub(crate) unsafe fn avx512_load_rgb_u8<const CN: u8>(
    src: *const u8,
) -> (__m512i, __m512i, __m512i) {
    let source_channels: YuvSourceChannels = CN.into();
    let (r_values, g_values, b_values);
    match source_channels {
        YuvSourceChannels::Rgb | YuvSourceChannels::Bgr => {
            let row_1 = _mm512_loadu_si512(src as *const i32);
            let row_2 = _mm512_loadu_si512(src.add(64) as *const i32);
            let row_3 = _mm512_loadu_si512(src.add(128) as *const i32);

            let (it1, it2, it3) = avx512_deinterleave_rgb(row_1, row_2, row_3);
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

            let (it1, it2, it3, _) = avx512_deinterleave_rgba(row_1, row_2, row_3, row_4);
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
    let v1 = _mm512_srli_epi16::<1>(_mm512_add_epi16(
        _mm512_maddubs_epi16(v, _mm512_set1_epi8(1)),
        _mm512_set1_epi16(1),
    ));
    v1
}

#[inline(always)]
pub(crate) unsafe fn avx512_pairwise_avg_epi8(a: __m512i) -> __m512i {
    _mm512_srli_epi16::<1>(_mm512_add_epi16(
        _mm512_maddubs_epi16(a, _mm512_set1_epi8(1)),
        _mm512_set1_epi16(1),
    ))
}
