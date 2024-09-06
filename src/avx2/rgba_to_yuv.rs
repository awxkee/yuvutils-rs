/*
 * // Copyright (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

use crate::avx2::avx2_utils::{
    _mm256_deinterleave_rgba_epi8, avx2_deinterleave_rgb, avx2_pack_u16, avx2_pairwise_widen_avg,
};
use crate::avx2::avx2_ycbcr::avx2_rgb_to_ycbcr;
use crate::internals::ProcessedOffset;
use crate::yuv_support::{
    CbCrForwardTransform, YuvChromaRange, YuvChromaSample, YuvSourceChannels,
};
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[target_feature(enable = "avx2")]
pub unsafe fn avx2_rgba_to_yuv<const ORIGIN_CHANNELS: u8, const SAMPLING: u8>(
    transform: &CbCrForwardTransform<i32>,
    range: &YuvChromaRange,
    y_plane: *mut u8,
    u_plane: *mut u8,
    v_plane: *mut u8,
    rgba: &[u8],
    rgba_offset: usize,
    start_cx: usize,
    start_ux: usize,
    width: usize,
    compute_uv_row: bool,
) -> ProcessedOffset {
    let chroma_subsampling: YuvChromaSample = SAMPLING.into();
    let source_channels: YuvSourceChannels = ORIGIN_CHANNELS.into();
    let channels = source_channels.get_channels_count();

    let y_ptr = y_plane;
    let u_ptr = u_plane;
    let v_ptr = v_plane;
    let rgba_ptr = rgba.as_ptr().add(rgba_offset);

    let mut cx = start_cx;
    let mut uv_x = start_ux;

    const ROUNDING_CONST_BIAS: i32 = 1 << 7;
    let bias_y = range.bias_y as i32 * (1 << 8) + ROUNDING_CONST_BIAS;
    let bias_uv = range.bias_uv as i32 * (1 << 8) + ROUNDING_CONST_BIAS;

    let y_bias = _mm256_set1_epi32(bias_y);
    let uv_bias = _mm256_set1_epi32(bias_uv);
    let v_yr = _mm256_set1_epi16(transform.yr as i16);
    let v_yg = _mm256_set1_epi16(transform.yg as i16);
    let v_yb = _mm256_set1_epi16(transform.yb as i16);
    let v_cb_r = _mm256_set1_epi16(transform.cb_r as i16);
    let v_cb_g = _mm256_set1_epi16(transform.cb_g as i16);
    let v_cb_b = _mm256_set1_epi16(transform.cb_b as i16);
    let v_cr_r = _mm256_set1_epi16(transform.cr_r as i16);
    let v_cr_g = _mm256_set1_epi16(transform.cr_g as i16);
    let v_cr_b = _mm256_set1_epi16(transform.cr_b as i16);

    while cx + 32 < width {
        let (r_values, g_values, b_values);

        let px = cx * channels;

        match source_channels {
            YuvSourceChannels::Rgb | YuvSourceChannels::Bgr => {
                let source_ptr = rgba_ptr.add(px);
                let row_1 = _mm256_loadu_si256(source_ptr as *const __m256i);
                let row_2 = _mm256_loadu_si256(source_ptr.add(32) as *const __m256i);
                let row_3 = _mm256_loadu_si256(source_ptr.add(64) as *const __m256i);

                let (it1, it2, it3) = avx2_deinterleave_rgb(row_1, row_2, row_3);
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
                let source_ptr = rgba_ptr.add(px);
                let row_1 = _mm256_loadu_si256(source_ptr as *const __m256i);
                let row_2 = _mm256_loadu_si256(source_ptr.add(32) as *const __m256i);
                let row_3 = _mm256_loadu_si256(source_ptr.add(64) as *const __m256i);
                let row_4 = _mm256_loadu_si256(source_ptr.add(96) as *const __m256i);

                let (it1, it2, it3, _) = _mm256_deinterleave_rgba_epi8(row_1, row_2, row_3, row_4);
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

        let r_low = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(r_values));
        let r_high = _mm256_cvtepu8_epi16(_mm256_extracti128_si256::<1>(r_values));
        let g_low = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(g_values));
        let g_high = _mm256_cvtepu8_epi16(_mm256_extracti128_si256::<1>(g_values));
        let b_low = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(b_values));
        let b_high = _mm256_cvtepu8_epi16(_mm256_extracti128_si256::<1>(b_values));

        let y_l = avx2_rgb_to_ycbcr(r_low, g_low, b_low, y_bias, v_yr, v_yg, v_yb);

        let y_h = avx2_rgb_to_ycbcr(r_high, g_high, b_high, y_bias, v_yr, v_yg, v_yb);

        let y_yuv = avx2_pack_u16(y_l, y_h);
        _mm256_storeu_si256(y_ptr.add(cx) as *mut __m256i, y_yuv);

        if compute_uv_row {
            let cb_l = avx2_rgb_to_ycbcr(r_low, g_low, b_low, uv_bias, v_cb_r, v_cb_g, v_cb_b);
            let cr_l = avx2_rgb_to_ycbcr(r_low, g_low, b_low, uv_bias, v_cr_r, v_cr_g, v_cr_b);
            let cb_h = avx2_rgb_to_ycbcr(r_high, g_high, b_high, uv_bias, v_cb_r, v_cb_g, v_cb_b);
            let cr_h = avx2_rgb_to_ycbcr(r_high, g_high, b_high, uv_bias, v_cr_r, v_cr_g, v_cr_b);

            let cb = avx2_pack_u16(cb_l, cb_h);

            let cr = avx2_pack_u16(cr_l, cr_h);

            match chroma_subsampling {
                YuvChromaSample::YUV420 | YuvChromaSample::YUV422 => {
                    let cb_h = _mm256_castsi256_si128(avx2_pairwise_widen_avg(cb));
                    let cr_h = _mm256_castsi256_si128(avx2_pairwise_widen_avg(cr));
                    _mm_storeu_si128(u_ptr.add(uv_x) as *mut _ as *mut __m128i, cb_h);
                    _mm_storeu_si128(v_ptr.add(uv_x) as *mut _ as *mut __m128i, cr_h);
                    uv_x += 16;
                }
                YuvChromaSample::YUV444 => {
                    _mm256_storeu_si256(u_ptr.add(uv_x) as *mut __m256i, cb);
                    _mm256_storeu_si256(v_ptr.add(uv_x) as *mut __m256i, cr);
                    uv_x += 32;
                }
            }
        }

        cx += 32;
    }

    ProcessedOffset { cx, ux: uv_x }
}
