/*
 * // Copyright (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */
use std::arch::wasm32::*;

#[inline]
#[target_feature(enable = "simd128")]
pub unsafe fn v128_load_half(ptr: *const u8) -> v128 {
    u64x2_replace_lane(i64x2_splat(0), (ptr as *const u64).read_unaligned())
}

/// Packs two u16x8 into one u8x16 using unsigned saturation
#[inline]
#[target_feature(enable = "simd128")]
pub unsafe fn u16x8_pack_sat_u8x16(a: v128, b: v128) -> v128 {
    let maxval = u16x8_splat(255);
    let a1 = v128_bitselect(maxval, a, u16x8_gt(a, maxval));
    let b1 = v128_bitselect(maxval, b, u16x8_gt(b, maxval));
    u8x16_shuffle::<0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30>(a1, b1)
}

#[inline]
#[target_feature(enable = "simd128")]
#[allow(dead_code)]
pub unsafe fn wasm_unpacklo_i8x16(a: v128, b: v128) -> v128 {
    u8x16_shuffle::<0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23>(a, b)
}

#[inline]
#[target_feature(enable = "simd128")]
pub unsafe fn wasm_unpacklo_i16x8(a: v128, b: v128) -> v128 {
    u8x16_shuffle::<0, 1, 16, 17, 2, 3, 18, 19, 4, 5, 20, 21, 6, 7, 22, 23>(a, b)
}

#[inline]
#[target_feature(enable = "simd128")]
#[allow(dead_code)]
pub unsafe fn wasm_unpacklo_i32x4(a: v128, b: v128) -> v128 {
    u8x16_shuffle::<0, 1, 2, 3, 16, 17, 18, 19, 4, 5, 6, 7, 20, 21, 22, 23>(a, b)
}

#[inline]
#[target_feature(enable = "simd128")]
#[allow(dead_code)]
pub unsafe fn wasm_unpacklo_i64x2(a: v128, b: v128) -> v128 {
    u8x16_shuffle::<0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23>(a, b)
}

#[inline]
#[target_feature(enable = "simd128")]
#[allow(dead_code)]
pub unsafe fn wasm_unpackhi_i8x16(a: v128, b: v128) -> v128 {
    u8x16_shuffle::<8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31>(a, b)
}

#[inline]
#[target_feature(enable = "simd128")]
#[allow(dead_code)]
pub unsafe fn wasm_unpackhi_i16x8(a: v128, b: v128) -> v128 {
    u8x16_shuffle::<8, 9, 24, 25, 10, 11, 26, 27, 12, 13, 28, 29, 14, 15, 30, 31>(a, b)
}

#[inline]
#[target_feature(enable = "simd128")]
#[allow(dead_code)]
pub unsafe fn wasm_unpackhi_i32x4(a: v128, b: v128) -> v128 {
    u8x16_shuffle::<8, 9, 10, 11, 24, 25, 26, 27, 12, 13, 14, 15, 28, 29, 30, 31>(a, b)
}

#[inline]
#[target_feature(enable = "simd128")]
#[allow(dead_code)]
pub unsafe fn wasm_unpackhi_i64x2(a: v128, b: v128) -> v128 {
    u8x16_shuffle::<8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31>(a, b)
}

#[inline]
#[target_feature(enable = "simd128")]
pub unsafe fn wasm_interleave_even_u8(a: v128) -> v128 {
    u8x16_shuffle::<0, 0, 2, 2, 4, 4, 6, 6, 8, 8, 10, 10, 12, 12, 14, 14>(a, a)
}

#[inline]
#[target_feature(enable = "simd128")]
pub unsafe fn wasm_interleave_odd_u8(a: v128) -> v128 {
    u8x16_shuffle::<1, 1, 3, 3, 5, 5, 7, 7, 9, 9, 11, 11, 13, 13, 15, 15>(a, a)
}
