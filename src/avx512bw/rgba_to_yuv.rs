/*
 * // Copyright (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

use crate::avx512bw::avx512_utils::{
    avx512_deinterleave_rgb, avx512_deinterleave_rgba, avx512_pack_u16, avx512_pairwise_widen_avg,
    avx512_rgb_to_ycbcr,
};
use crate::internals::ProcessedOffset;
use crate::yuv_support::{
    CbCrForwardTransform, YuvChromaRange, YuvChromaSample, YuvSourceChannels,
};
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[target_feature(enable = "avx512bw")]
pub unsafe fn avx512_rgba_to_yuv<const ORIGIN_CHANNELS: u8, const SAMPLING: u8>(
    transform: &CbCrForwardTransform<i32>,
    range: &YuvChromaRange,
    y_plane: *mut u8,
    u_plane: *mut u8,
    v_plane: *mut u8,
    rgba: &[u8],
    y_offset: usize,
    u_offset: usize,
    v_offset: usize,
    rgba_offset: usize,
    start_cx: usize,
    start_ux: usize,
    width: usize,
) -> ProcessedOffset {
    let chroma_subsampling: YuvChromaSample = SAMPLING.into();
    let source_channels: YuvSourceChannels = ORIGIN_CHANNELS.into();
    let channels = source_channels.get_channels_count();

    let y_ptr = y_plane.add(y_offset);
    let u_ptr = u_plane.add(u_offset);
    let v_ptr = v_plane.add(v_offset);
    let rgba_ptr = rgba.as_ptr().add(rgba_offset);

    let mut cx = start_cx;
    let mut uv_x = start_ux;

    let bias_y = ((range.bias_y as f32 + 0.5f32) * (1i32 << 8i32) as f32) as i32;
    let bias_uv = ((range.bias_uv as f32 + 0.5f32) * (1i32 << 8i32) as f32) as i32;

    while cx + 64 < width {
        let y_bias = _mm512_set1_epi32(bias_y);
        let uv_bias = _mm512_set1_epi32(bias_uv);
        let v_yr = _mm512_set1_epi32(transform.yr);
        let v_yg = _mm512_set1_epi32(transform.yg);
        let v_yb = _mm512_set1_epi32(transform.yb);
        let v_cb_r = _mm512_set1_epi32(transform.cb_r);
        let v_cb_g = _mm512_set1_epi32(transform.cb_g);
        let v_cb_b = _mm512_set1_epi32(transform.cb_b);
        let v_cr_r = _mm512_set1_epi32(transform.cr_r);
        let v_cr_g = _mm512_set1_epi32(transform.cr_g);
        let v_cr_b = _mm512_set1_epi32(transform.cr_b);

        let (r_values, g_values, b_values);

        let px = cx * channels;

        match source_channels {
            YuvSourceChannels::Rgb => {
                let row_1 = _mm512_loadu_si512(rgba_ptr.add(px) as *const i32);
                let row_2 = _mm512_loadu_si512(rgba_ptr.add(px + 64) as *const i32);
                let row_3 = _mm512_loadu_si512(rgba_ptr.add(px + 128) as *const i32);

                let (it1, it2, it3) = avx512_deinterleave_rgb(row_1, row_2, row_3);
                r_values = it1;
                g_values = it2;
                b_values = it3;
            }
            YuvSourceChannels::Rgba | YuvSourceChannels::Bgra => {
                let row_1 = _mm512_loadu_si512(rgba_ptr.add(px) as *const i32);
                let row_2 = _mm512_loadu_si512(rgba_ptr.add(px + 64) as *const i32);
                let row_3 = _mm512_loadu_si512(rgba_ptr.add(px + 128) as *const i32);
                let row_4 = _mm512_loadu_si512(rgba_ptr.add(px + 128 + 64) as *const i32);

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

        let r_low = _mm512_cvtepu8_epi16(_mm512_castsi512_si256(r_values));
        let r_high = _mm512_cvtepu8_epi16(_mm512_extracti64x4_epi64::<1>(r_values));
        let g_low = _mm512_cvtepu8_epi16(_mm512_castsi512_si256(g_values));
        let g_high = _mm512_cvtepu8_epi16(_mm512_extracti64x4_epi64::<1>(g_values));
        let b_low = _mm512_cvtepu8_epi16(_mm512_castsi512_si256(b_values));
        let b_high = _mm512_cvtepu8_epi16(_mm512_extracti64x4_epi64::<1>(b_values));

        let y_l = avx512_rgb_to_ycbcr(r_low, g_low, b_low, y_bias, v_yr, v_yg, v_yb);
        let cb_l = avx512_rgb_to_ycbcr(r_low, g_low, b_low, uv_bias, v_cb_r, v_cb_g, v_cb_b);
        let cr_l = avx512_rgb_to_ycbcr(r_low, g_low, b_low, uv_bias, v_cr_r, v_cr_g, v_cr_b);

        let y_h = avx512_rgb_to_ycbcr(r_high, g_high, b_high, y_bias, v_yr, v_yg, v_yb);

        let y_yuv = avx512_pack_u16(y_l, y_h);

        let cb_h = avx512_rgb_to_ycbcr(r_high, g_high, b_high, uv_bias, v_cb_r, v_cb_g, v_cb_b);
        let cr_h = avx512_rgb_to_ycbcr(r_high, g_high, b_high, uv_bias, v_cr_r, v_cr_g, v_cr_b);

        let cb = avx512_pack_u16(cb_l, cb_h);

        let cr = avx512_pack_u16(cr_l, cr_h);

        _mm512_storeu_si512(y_ptr.add(cx) as *mut i32, y_yuv);

        match chroma_subsampling {
            YuvChromaSample::YUV420 | YuvChromaSample::YUV422 => {
                let cb_h = _mm512_castsi512_si256(avx512_pairwise_widen_avg(cb));
                let cr_h = _mm512_castsi512_si256(avx512_pairwise_widen_avg(cr));
                _mm256_storeu_si256(u_ptr.add(uv_x) as *mut _ as *mut __m256i, cb_h);
                _mm256_storeu_si256(v_ptr.add(uv_x) as *mut _ as *mut __m256i, cr_h);
                uv_x += 32;
            }
            YuvChromaSample::YUV444 => {
                _mm512_storeu_si512(u_ptr.add(uv_x) as *mut i32, cb);
                _mm512_storeu_si512(v_ptr.add(uv_x) as *mut i32, cr);
                uv_x += 64;
            }
        }

        cx += 64;
    }

    return ProcessedOffset { cx, ux: uv_x };
}
