/*
 * // Copyright (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

#[cfg(feature = "nightly_avx512")]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use crate::avx512_utils::{avx512_div_by255, avx512_pack_u16, avx512_rgb_u8, avx512_rgba_u8};
#[allow(unused_imports)]
use crate::internals::ProcessedOffset;
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use crate::neon_simd_support::neon_premultiply_alpha;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use crate::x86_simd_support::*;
use crate::yuv_support::{get_yuv_range, YuvChromaRange, YuvChromaSample, YuvSourceChannels};
use crate::YuvRange;
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use std::arch::aarch64::*;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(feature = "nightly_avx512")]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline(always)]
#[allow(dead_code)]
unsafe fn avx512_process_row<const DESTINATION_CHANNELS: u8, const SAMPLING: u8>(
    range: &YuvChromaRange,
    y_plane: &[u8],
    cg_plane: &[u8],
    co_plane: &[u8],
    a_plane: &[u8],
    rgba: &mut [u8],
    start_cx: usize,
    start_ux: usize,
    y_offset: usize,
    u_offset: usize,
    v_offset: usize,
    a_offset: usize,
    rgba_offset: usize,
    width: usize,
    premultiply_alpha: bool,
) -> ProcessedOffset {
    let chroma_subsampling: YuvChromaSample = SAMPLING.into();
    let destination_channels: YuvSourceChannels = DESTINATION_CHANNELS.into();
    let channels = destination_channels.get_channels_count();
    let bias_y = range.bias_y as i32;
    let bias_uv = range.bias_uv as i32;

    let mut cx = start_cx;
    let mut uv_x = start_ux;

    let y_ptr = y_plane.as_ptr().add(y_offset);
    let u_ptr = cg_plane.as_ptr().add(u_offset);
    let v_ptr = co_plane.as_ptr().add(v_offset);
    let a_ptr = a_plane.as_ptr().add(a_offset);
    let rgba_ptr = rgba.as_mut_ptr().add(rgba_offset);

    let max_colors = 2i32.pow(8) - 1i32;
    let precision_scale = (1 << 6) as f32;

    let range_reduction_y =
        (max_colors as f32 / range.range_y as f32 * precision_scale).round() as i32;
    let range_reduction_uv =
        (max_colors as f32 / range.range_uv as f32 * precision_scale).round() as i32;

    let y_corr = _mm512_set1_epi16(bias_y as i16);
    let uv_corr = _mm512_set1_epi16(bias_uv as i16);
    let y_reduction = _mm512_set1_epi16(range_reduction_y as i16);
    let uv_reduction = _mm512_set1_epi16(range_reduction_uv as i16);
    let v_alpha = _mm512_set1_epi16(-128);
    let v_min_zeros = _mm512_setzero_si512();

    while cx + 64 < width {
        let y_values = _mm512_loadu_si512(y_ptr.add(cx) as *const i32);
        let a_values = _mm512_loadu_si512(a_ptr.add(cx) as *const i32);

        let u_high_u8;
        let v_high_u8;
        let u_low_u8;
        let v_low_u8;

        match chroma_subsampling {
            YuvChromaSample::YUV420 | YuvChromaSample::YUV422 => {
                let u_values = _mm256_loadu_si256(u_ptr.add(uv_x) as *const __m256i);
                let v_values = _mm256_loadu_si256(v_ptr.add(uv_x) as *const __m256i);

                const MASK: i32 = shuffle(3, 1, 2, 0);
                u_high_u8 =
                    _mm256_permute4x64_epi64::<MASK>(_mm256_unpackhi_epi8(u_values, u_values));
                v_high_u8 =
                    _mm256_permute4x64_epi64::<MASK>(_mm256_unpackhi_epi8(v_values, v_values));
                u_low_u8 =
                    _mm256_permute4x64_epi64::<MASK>(_mm256_unpacklo_epi8(u_values, u_values));
                v_low_u8 =
                    _mm256_permute4x64_epi64::<MASK>(_mm256_unpacklo_epi8(v_values, v_values));
            }
            YuvChromaSample::YUV444 => {
                let u_values = _mm512_loadu_si512(u_ptr.add(uv_x) as *const i32);
                let v_values = _mm512_loadu_si512(v_ptr.add(uv_x) as *const i32);

                u_high_u8 = _mm512_extracti64x4_epi64::<1>(u_values);
                v_high_u8 = _mm512_extracti64x4_epi64::<1>(v_values);
                u_low_u8 = _mm512_castsi512_si256(u_values);
                v_low_u8 = _mm512_castsi512_si256(v_values);
            }
        }

        let cg_high = _mm512_mullo_epi16(
            _mm512_subs_epi16(_mm512_cvtepu8_epi16(u_high_u8), uv_corr),
            uv_reduction,
        );
        let co_high = _mm512_mullo_epi16(
            _mm512_subs_epi16(_mm512_cvtepu8_epi16(v_high_u8), uv_corr),
            uv_reduction,
        );
        let y_high = _mm512_mullo_epi16(
            _mm512_sub_epi16(
                _mm512_cvtepu8_epi16(_mm512_extracti64x4_epi64::<1>(y_values)),
                y_corr,
            ),
            y_reduction,
        );

        let t_high = _mm512_subs_epi16(y_high, cg_high);

        let r_high = _mm512_srai_epi16::<6>(_mm512_max_epi16(
            _mm512_adds_epi16(t_high, co_high),
            v_min_zeros,
        ));
        let b_high = _mm512_srai_epi16::<6>(_mm512_max_epi16(
            _mm512_subs_epi16(t_high, co_high),
            v_min_zeros,
        ));
        let g_high = _mm512_srai_epi16::<6>(_mm512_max_epi16(
            _mm512_adds_epi16(y_high, cg_high),
            v_min_zeros,
        ));

        let cg_low = _mm512_mullo_epi16(
            _mm512_subs_epi16(_mm512_cvtepu8_epi16(u_low_u8), uv_corr),
            uv_reduction,
        );
        let co_low = _mm512_mullo_epi16(
            _mm512_subs_epi16(_mm512_cvtepu8_epi16(v_low_u8), uv_corr),
            uv_reduction,
        );
        let y_low = _mm512_mullo_epi16(
            _mm512_sub_epi16(
                _mm512_cvtepu8_epi16(_mm512_castsi512_si256(y_values)),
                y_corr,
            ),
            y_reduction,
        );

        let t_low = _mm512_subs_epi16(y_low, cg_low);

        let r_low = _mm512_srai_epi16::<6>(_mm512_max_epi16(
            _mm512_adds_epi16(t_low, co_low),
            v_min_zeros,
        ));
        let b_low = _mm512_srai_epi16::<6>(_mm512_max_epi16(
            _mm512_subs_epi16(t_low, co_low),
            v_min_zeros,
        ));
        let g_low = _mm512_srai_epi16::<6>(_mm512_max_epi16(
            _mm512_adds_epi16(y_low, cg_low),
            v_min_zeros,
        ));

        let (r_values, g_values, b_values);

        if premultiply_alpha {
            let a_high = _mm512_cvtepu8_epi16(_mm512_extracti64x4_epi64::<1>(a_values));
            let a_low = _mm512_cvtepu8_epi16(_mm512_castsi512_si256(a_values));

            let r_l = avx512_div_by255(_mm512_mullo_epi16(r_low, a_low));
            let r_h = avx512_div_by255(_mm512_mullo_epi16(r_high, a_high));
            let g_l = avx512_div_by255(_mm512_mullo_epi16(g_low, a_low));
            let g_h = avx512_div_by255(_mm512_mullo_epi16(g_high, a_high));
            let b_l = avx512_div_by255(_mm512_mullo_epi16(b_low, a_low));
            let b_h = avx512_div_by255(_mm512_mullo_epi16(b_high, a_high));

            r_values = avx512_pack_u16(r_l, r_h);
            g_values = avx512_pack_u16(g_l, g_h);
            b_values = avx512_pack_u16(b_l, b_h);
        } else {
            r_values = avx512_pack_u16(r_low, r_high);
            g_values = avx512_pack_u16(g_low, g_high);
            b_values = avx512_pack_u16(b_low, b_high);
        }

        let dst_shift = cx * channels;

        match destination_channels {
            YuvSourceChannels::Rgb => {
                let ptr = rgba_ptr.add(dst_shift);
                avx512_rgb_u8(ptr, r_values, g_values, b_values);
            }
            YuvSourceChannels::Rgba => {
                avx512_rgba_u8(
                    rgba_ptr.add(dst_shift),
                    r_values,
                    g_values,
                    b_values,
                    v_alpha,
                );
            }
            YuvSourceChannels::Bgra => {
                avx512_rgba_u8(
                    rgba_ptr.add(dst_shift),
                    b_values,
                    g_values,
                    r_values,
                    v_alpha,
                );
            }
        }

        cx += 64;

        match chroma_subsampling {
            YuvChromaSample::YUV420 | YuvChromaSample::YUV422 => {
                uv_x += 32;
            }
            YuvChromaSample::YUV444 => {
                uv_x += 64;
            }
        }
    }

    return ProcessedOffset { cx, ux: uv_x };
}

#[cfg(target_feature = "avx2")]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline(always)]
#[allow(dead_code)]
unsafe fn avx2_process_row<const DESTINATION_CHANNELS: u8, const SAMPLING: u8>(
    range: &YuvChromaRange,
    y_plane: &[u8],
    cg_plane: &[u8],
    v_plane: &[u8],
    a_plane: &[u8],
    rgba: &mut [u8],
    start_cx: usize,
    start_ux: usize,
    y_offset: usize,
    u_offset: usize,
    v_offset: usize,
    a_offset: usize,
    rgba_offset: usize,
    width: usize,
    premultiply_alpha: bool,
) -> ProcessedOffset {
    let chroma_subsampling: YuvChromaSample = SAMPLING.into();
    let destination_channels: YuvSourceChannels = DESTINATION_CHANNELS.into();
    let channels = destination_channels.get_channels_count();
    let bias_y = range.bias_y as i32;
    let bias_uv = range.bias_uv as i32;

    let mut cx = start_cx;
    let mut uv_x = start_ux;

    let y_ptr = y_plane.as_ptr().add(y_offset);
    let u_ptr = cg_plane.as_ptr().add(u_offset);
    let v_ptr = v_plane.as_ptr().add(v_offset);
    let a_ptr = a_plane.as_ptr().add(a_offset);
    let rgba_ptr = rgba.as_mut_ptr().add(rgba_offset);

    let max_colors = 2i32.pow(8) - 1i32;
    let precision_scale = (1 << 6) as f32;

    let range_reduction_y =
        (max_colors as f32 / range.range_y as f32 * precision_scale).round() as i32;
    let range_reduction_uv =
        (max_colors as f32 / range.range_uv as f32 * precision_scale).round() as i32;

    let y_corr = _mm256_set1_epi16(bias_y as i16);
    let uv_corr = _mm256_set1_epi16(bias_uv as i16);
    let y_reduction = _mm256_set1_epi16(range_reduction_y as i16);
    let uv_reduction = _mm256_set1_epi16(range_reduction_uv as i16);
    let v_alpha = _mm256_set1_epi16(-128);
    let v_min_zeros = _mm256_setzero_si256();

    while cx + 32 < width {
        let y_values = _mm256_loadu_si256(y_ptr.add(cx) as *const __m256i);
        let a_values = _mm256_loadu_si256(a_ptr.add(cx) as *const __m256i);

        let u_high_u8;
        let v_high_u8;
        let u_low_u8;
        let v_low_u8;

        match chroma_subsampling {
            YuvChromaSample::YUV420 | YuvChromaSample::YUV422 => {
                let u_values = _mm_loadu_si128(u_ptr.add(uv_x) as *const __m128i);
                let v_values = _mm_loadu_si128(v_ptr.add(uv_x) as *const __m128i);

                u_high_u8 = sse_interleave_even(_mm_unpackhi_epi8(u_values, u_values));
                v_high_u8 = sse_interleave_odd(_mm_unpackhi_epi8(v_values, v_values));
                u_low_u8 = sse_interleave_even(_mm_unpacklo_epi8(u_values, u_values));
                v_low_u8 = sse_interleave_odd(_mm_unpacklo_epi8(v_values, v_values));
            }
            YuvChromaSample::YUV444 => {
                let u_values = _mm256_loadu_si256(u_ptr.add(uv_x) as *const __m256i);
                let v_values = _mm256_loadu_si256(v_ptr.add(uv_x) as *const __m256i);

                u_high_u8 = _mm256_extracti128_si256::<1>(u_values);
                v_high_u8 = _mm256_extracti128_si256::<1>(v_values);
                u_low_u8 = _mm256_castsi256_si128(u_values);
                v_low_u8 = _mm256_castsi256_si128(v_values);
            }
        }

        let cg_high = _mm256_mullo_epi16(
            _mm256_subs_epi16(_mm256_cvtepu8_epi16(u_high_u8), uv_corr),
            uv_reduction,
        );
        let co_high = _mm256_mullo_epi16(
            _mm256_subs_epi16(_mm256_cvtepu8_epi16(v_high_u8), uv_corr),
            uv_reduction,
        );
        let y_high = _mm256_mullo_epi16(
            _mm256_sub_epi16(
                _mm256_cvtepu8_epi16(_mm256_extracti128_si256::<1>(y_values)),
                y_corr,
            ),
            y_reduction,
        );

        let t_high = _mm256_subs_epi16(y_high, cg_high);

        let r_high = _mm256_srai_epi16::<6>(_mm256_max_epi16(
            _mm256_adds_epi16(t_high, co_high),
            v_min_zeros,
        ));
        let b_high = _mm256_srai_epi16::<6>(_mm256_max_epi16(
            _mm256_subs_epi16(t_high, co_high),
            v_min_zeros,
        ));
        let g_high = _mm256_srai_epi16::<6>(_mm256_max_epi16(
            _mm256_adds_epi16(y_high, cg_high),
            v_min_zeros,
        ));

        let cg_low = _mm256_mullo_epi16(
            _mm256_subs_epi16(_mm256_cvtepu8_epi16(u_low_u8), uv_corr),
            uv_reduction,
        );
        let co_low = _mm256_mullo_epi16(
            _mm256_subs_epi16(_mm256_cvtepu8_epi16(v_low_u8), uv_corr),
            uv_reduction,
        );
        let y_low = _mm256_mullo_epi16(
            _mm256_sub_epi16(
                _mm256_cvtepu8_epi16(_mm256_castsi256_si128(y_values)),
                y_corr,
            ),
            y_reduction,
        );

        let t_low = _mm256_subs_epi16(y_low, cg_low);

        let r_low = _mm256_srai_epi16::<6>(_mm256_max_epi16(
            _mm256_adds_epi16(t_low, co_low),
            v_min_zeros,
        ));
        let b_low = _mm256_srai_epi16::<6>(_mm256_max_epi16(
            _mm256_subs_epi16(t_low, co_low),
            v_min_zeros,
        ));
        let g_low = _mm256_srai_epi16::<6>(_mm256_max_epi16(
            _mm256_adds_epi16(y_low, cg_low),
            v_min_zeros,
        ));

        let (r_values, g_values, b_values);

        if premultiply_alpha {
            let a_high = _mm256_cvtepu8_epi16(_mm256_extracti128_si256::<1>(a_values));
            let a_low = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(a_values));

            let r_l = avx2_div_by255(_mm256_mullo_epi16(r_low, a_low));
            let r_h = avx2_div_by255(_mm256_mullo_epi16(r_high, a_high));
            let g_l = avx2_div_by255(_mm256_mullo_epi16(g_low, a_low));
            let g_h = avx2_div_by255(_mm256_mullo_epi16(g_high, a_high));
            let b_l = avx2_div_by255(_mm256_mullo_epi16(b_low, a_low));
            let b_h = avx2_div_by255(_mm256_mullo_epi16(b_high, a_high));

            r_values = avx2_pack_u16(r_l, r_h);
            g_values = avx2_pack_u16(g_l, g_h);
            b_values = avx2_pack_u16(b_l, b_h);
        } else {
            r_values = avx2_pack_u16(r_low, r_high);
            g_values = avx2_pack_u16(g_low, g_high);
            b_values = avx2_pack_u16(b_low, b_high);
        }

        let dst_shift = cx * channels;

        match destination_channels {
            YuvSourceChannels::Rgb => {
                let ptr = rgba_ptr.add(dst_shift);
                avx2_store_u8_rgb(ptr, r_values, g_values, b_values);
            }
            YuvSourceChannels::Rgba => {
                avx2_store_u8_rgba(
                    rgba_ptr.add(dst_shift),
                    r_values,
                    g_values,
                    b_values,
                    v_alpha,
                );
            }
            YuvSourceChannels::Bgra => {
                avx2_store_u8_rgba(
                    rgba_ptr.add(dst_shift),
                    b_values,
                    g_values,
                    r_values,
                    v_alpha,
                );
            }
        }

        cx += 32;

        match chroma_subsampling {
            YuvChromaSample::YUV420 | YuvChromaSample::YUV422 => {
                uv_x += 16;
            }
            YuvChromaSample::YUV444 => {
                uv_x += 32;
            }
        }
    }

    return ProcessedOffset { cx, ux: uv_x };
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline(always)]
#[allow(dead_code)]
unsafe fn sse_process_row<const DESTINATION_CHANNELS: u8, const SAMPLING: u8>(
    range: &YuvChromaRange,
    y_plane: &[u8],
    cg_plane: &[u8],
    v_plane: &[u8],
    a_plane: &[u8],
    rgba: &mut [u8],
    start_cx: usize,
    start_ux: usize,
    y_offset: usize,
    u_offset: usize,
    v_offset: usize,
    a_offset: usize,
    rgba_offset: usize,
    width: usize,
    premultiply_alpha: bool,
) -> ProcessedOffset {
    let chroma_subsampling: YuvChromaSample = SAMPLING.into();
    let destination_channels: YuvSourceChannels = DESTINATION_CHANNELS.into();
    let channels = destination_channels.get_channels_count();
    let bias_y = range.bias_y as i32;
    let bias_uv = range.bias_uv as i32;

    let mut cx = start_cx;
    let mut uv_x = start_ux;

    let y_ptr = y_plane.as_ptr().add(y_offset);
    let u_ptr = cg_plane.as_ptr().add(u_offset);
    let v_ptr = v_plane.as_ptr().add(v_offset);
    let a_ptr = a_plane.as_ptr().add(a_offset);
    let rgba_ptr = rgba.as_mut_ptr().add(rgba_offset);

    let max_colors = 2i32.pow(8) - 1i32;
    let precision_scale = (1 << 6) as f32;

    let range_reduction_y =
        (max_colors as f32 / range.range_y as f32 * precision_scale).round() as i32;
    let range_reduction_uv =
        (max_colors as f32 / range.range_uv as f32 * precision_scale).round() as i32;

    let y_corr = _mm_set1_epi16(bias_y as i16);
    let uv_corr = _mm_set1_epi16(bias_uv as i16);
    let y_reduction = _mm_set1_epi16(range_reduction_y as i16);
    let uv_reduction = _mm_set1_epi16(range_reduction_uv as i16);
    let v_alpha = _mm_set1_epi16(-128);
    let v_min_zeros = _mm_setzero_si128();

    while cx + 16 < width {
        let y_values = _mm_loadu_si128(y_ptr.add(cx) as *const __m128i);
        let a_values = _mm_loadu_si128(a_ptr.add(cx) as *const __m128i);

        let u_high_u8;
        let v_high_u8;
        let u_low_u8;
        let v_low_u8;

        match chroma_subsampling {
            YuvChromaSample::YUV420 | YuvChromaSample::YUV422 => {
                let reshuffle = _mm_setr_epi8(0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7);

                let u_values = _mm_shuffle_epi8(_mm_loadu_si64(u_ptr.add(uv_x)), reshuffle);
                let v_values = _mm_shuffle_epi8(_mm_loadu_si64(v_ptr.add(uv_x)), reshuffle);

                u_high_u8 = _mm_unpackhi_epi8(u_values, u_values);
                v_high_u8 = _mm_unpackhi_epi8(v_values, v_values);
                u_low_u8 = _mm_unpacklo_epi8(u_values, u_values);
                v_low_u8 = _mm_unpacklo_epi8(v_values, v_values);
            }
            YuvChromaSample::YUV444 => {
                let u_values = _mm_loadu_si128(u_ptr.add(uv_x) as *const __m128i);
                let v_values = _mm_loadu_si128(v_ptr.add(uv_x) as *const __m128i);

                u_high_u8 = _mm_unpackhi_epi8(u_values, u_values);
                v_high_u8 = _mm_unpackhi_epi8(v_values, v_values);
                u_low_u8 = _mm_unpacklo_epi8(u_values, u_values);
                v_low_u8 = _mm_unpacklo_epi8(v_values, v_values);
            }
        }

        let cg_high = _mm_mullo_epi16(
            _mm_subs_epi16(_mm_unpackhi_epi8(u_high_u8, v_min_zeros), uv_corr),
            uv_reduction,
        );
        let co_high = _mm_mullo_epi16(
            _mm_subs_epi16(_mm_unpackhi_epi8(v_high_u8, v_min_zeros), uv_corr),
            uv_reduction,
        );
        let y_high = _mm_mullo_epi16(
            _mm_sub_epi16(_mm_unpackhi_epi8(y_values, v_min_zeros), y_corr),
            y_reduction,
        );

        let t_high = _mm_subs_epi16(y_high, cg_high);

        let r_high =
            _mm_srai_epi16::<6>(_mm_max_epi16(_mm_adds_epi16(t_high, co_high), v_min_zeros));
        let b_high =
            _mm_srai_epi16::<6>(_mm_max_epi16(_mm_subs_epi16(t_high, co_high), v_min_zeros));
        let g_high =
            _mm_srai_epi16::<6>(_mm_max_epi16(_mm_adds_epi16(y_high, cg_high), v_min_zeros));

        let cg_low = _mm_mullo_epi16(
            _mm_subs_epi16(_mm_cvtepu8_epi16(u_low_u8), uv_corr),
            uv_reduction,
        );
        let co_low = _mm_mullo_epi16(
            _mm_subs_epi16(_mm_cvtepu8_epi16(v_low_u8), uv_corr),
            uv_reduction,
        );
        let y_low = _mm_mullo_epi16(
            _mm_sub_epi16(_mm_cvtepu8_epi16(y_values), y_corr),
            y_reduction,
        );

        let t_low = _mm_subs_epi16(y_low, cg_low);

        let r_low = _mm_srai_epi16::<6>(_mm_max_epi16(_mm_adds_epi16(t_low, co_low), v_min_zeros));
        let b_low = _mm_srai_epi16::<6>(_mm_max_epi16(_mm_subs_epi16(t_low, co_low), v_min_zeros));
        let g_low = _mm_srai_epi16::<6>(_mm_max_epi16(_mm_adds_epi16(y_low, cg_low), v_min_zeros));

        let (r_values, g_values, b_values);

        if premultiply_alpha {
            let a_h = _mm_unpackhi_epi8(a_values, v_min_zeros);
            let a_l = _mm_cvtepu8_epi16(a_values);
            let r_h_16 = sse_div_by255(_mm_mullo_epi16(r_high, a_h));
            let r_l_16 = sse_div_by255(_mm_mullo_epi16(r_low, a_l));
            let g_h_16 = sse_div_by255(_mm_mullo_epi16(g_high, a_h));
            let g_l_16 = sse_div_by255(_mm_mullo_epi16(g_low, a_l));
            let b_h_16 = sse_div_by255(_mm_mullo_epi16(b_high, a_h));
            let b_l_16 = sse_div_by255(_mm_mullo_epi16(b_low, a_l));

            r_values = _mm_packus_epi16(r_l_16, r_h_16);
            g_values = _mm_packus_epi16(g_l_16, g_h_16);
            b_values = _mm_packus_epi16(b_l_16, b_h_16);
        } else {
            r_values = _mm_packus_epi16(r_low, r_high);
            g_values = _mm_packus_epi16(g_low, g_high);
            b_values = _mm_packus_epi16(b_low, b_high);
        }

        let dst_shift = cx * channels;

        match destination_channels {
            YuvSourceChannels::Rgb => {
                sse_store_rgb_u8(rgba_ptr.add(dst_shift), r_values, g_values, b_values);
            }
            YuvSourceChannels::Rgba => {
                sse_store_rgba(
                    rgba_ptr.add(dst_shift),
                    r_values,
                    g_values,
                    b_values,
                    v_alpha,
                );
            }
            YuvSourceChannels::Bgra => {
                sse_store_rgba(
                    rgba_ptr.add(dst_shift),
                    b_values,
                    g_values,
                    r_values,
                    v_alpha,
                );
            }
        }

        cx += 16;

        match chroma_subsampling {
            YuvChromaSample::YUV420 | YuvChromaSample::YUV422 => {
                uv_x += 8;
            }
            YuvChromaSample::YUV444 => {
                uv_x += 16;
            }
        }
    }

    return ProcessedOffset { cx, ux: uv_x };
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[inline(always)]
#[allow(dead_code)]
unsafe fn neon_process_row<const DESTINATION_CHANNELS: u8, const SAMPLING: u8>(
    range: &YuvChromaRange,
    y_plane: &[u8],
    cg_plane: &[u8],
    v_plane: &[u8],
    a_plane: &[u8],
    rgba: &mut [u8],
    start_cx: usize,
    start_ux: usize,
    y_offset: usize,
    u_offset: usize,
    v_offset: usize,
    a_offset: usize,
    rgba_offset: usize,
    width: usize,
    premultiply_alpha: bool,
) -> ProcessedOffset {
    let chroma_subsampling: YuvChromaSample = SAMPLING.into();
    let destination_channels: YuvSourceChannels = DESTINATION_CHANNELS.into();
    let channels = destination_channels.get_channels_count();
    let bias_y = range.bias_y as i32;
    let bias_uv = range.bias_uv as i32;

    let mut cx = start_cx;
    let mut uv_x = start_ux;

    let y_ptr = y_plane.as_ptr().add(y_offset);
    let u_ptr = cg_plane.as_ptr().add(u_offset);
    let v_ptr = v_plane.as_ptr().add(v_offset);
    let a_ptr = a_plane.as_ptr().add(a_offset);
    let rgba_ptr = rgba.as_mut_ptr().add(rgba_offset);

    let max_colors = 2i32.pow(8) - 1i32;
    let precision_scale = (1 << 6) as f32;

    let range_reduction_y =
        (max_colors as f32 / range.range_y as f32 * precision_scale).round() as i32;
    let range_reduction_uv =
        (max_colors as f32 / range.range_uv as f32 * precision_scale).round() as i32;

    let y_corr = vdupq_n_u8(bias_y as u8);
    let uv_corr = vdupq_n_s16(bias_uv as i16);
    let y_reduction = vdupq_n_u8(range_reduction_y as u8);
    let uv_reduction = vdupq_n_s16(range_reduction_uv as i16);
    let v_alpha = vdupq_n_u8(255u8);
    let v_min_zeros = vdupq_n_s16(0i16);

    while cx + 16 < width {
        let y_values = vsubq_u8(vld1q_u8(y_ptr.add(cx)), y_corr);
        let a_values = vld1q_u8(a_ptr.add(cx));

        let u_high_u8: uint8x8_t;
        let v_high_u8: uint8x8_t;
        let u_low_u8: uint8x8_t;
        let v_low_u8: uint8x8_t;

        match chroma_subsampling {
            YuvChromaSample::YUV420 | YuvChromaSample::YUV422 => {
                let u_values = vld1_u8(u_ptr.add(uv_x));
                let v_values = vld1_u8(v_ptr.add(uv_x));

                u_high_u8 = vzip2_u8(u_values, u_values);
                v_high_u8 = vzip2_u8(v_values, v_values);
                u_low_u8 = vzip1_u8(u_values, u_values);
                v_low_u8 = vzip1_u8(v_values, v_values);
            }
            YuvChromaSample::YUV444 => {
                let u_values = vld1q_u8(u_ptr.add(uv_x));
                let v_values = vld1q_u8(v_ptr.add(uv_x));

                u_high_u8 = vget_high_u8(u_values);
                v_high_u8 = vget_high_u8(v_values);
                u_low_u8 = vget_low_u8(u_values);
                v_low_u8 = vget_low_u8(v_values);
            }
        }

        let cg_high = vmulq_s16(
            vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(u_high_u8)), uv_corr),
            uv_reduction,
        );
        let co_high = vmulq_s16(
            vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(v_high_u8)), uv_corr),
            uv_reduction,
        );
        let y_high = vreinterpretq_s16_u16(vmull_high_u8(y_values, y_reduction));

        let t_high = vqsubq_s16(y_high, cg_high);

        let r_high = vqshrun_n_s16::<6>(vmaxq_s16(vqaddq_s16(t_high, co_high), v_min_zeros));
        let b_high = vqshrun_n_s16::<6>(vmaxq_s16(vqsubq_s16(t_high, co_high), v_min_zeros));
        let g_high = vqshrun_n_s16::<6>(vmaxq_s16(vqaddq_s16(y_high, cg_high), v_min_zeros));

        let cg_low = vmulq_s16(
            vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(u_low_u8)), uv_corr),
            uv_reduction,
        );
        let co_low = vmulq_s16(
            vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(v_low_u8)), uv_corr),
            uv_reduction,
        );
        let y_low =
            vreinterpretq_s16_u16(vmull_u8(vget_low_u8(y_values), vget_low_u8(y_reduction)));

        let t_low = vqsubq_s16(y_low, cg_low);

        let r_low = vqshrun_n_s16::<6>(vmaxq_s16(vqaddq_s16(t_low, co_low), v_min_zeros));
        let b_low = vqshrun_n_s16::<6>(vmaxq_s16(vqsubq_s16(t_low, co_low), v_min_zeros));
        let g_low = vqshrun_n_s16::<6>(vmaxq_s16(vqaddq_s16(y_low, cg_low), v_min_zeros));

        let mut r_values = vcombine_u8(r_low, r_high);
        let mut g_values = vcombine_u8(g_low, g_high);
        let mut b_values = vcombine_u8(b_low, b_high);

        if premultiply_alpha {
            r_values = neon_premultiply_alpha(r_values, a_values);
            g_values = neon_premultiply_alpha(g_values, a_values);
            b_values = neon_premultiply_alpha(b_values, a_values);
        }

        let dst_shift = cx * channels;

        match destination_channels {
            YuvSourceChannels::Rgb => {
                let dst_pack: uint8x16x3_t = uint8x16x3_t(r_values, g_values, b_values);
                vst3q_u8(rgba_ptr.add(dst_shift), dst_pack);
            }
            YuvSourceChannels::Rgba => {
                let dst_pack: uint8x16x4_t = uint8x16x4_t(r_values, g_values, b_values, v_alpha);
                vst4q_u8(rgba_ptr.add(dst_shift), dst_pack);
            }
            YuvSourceChannels::Bgra => {
                let dst_pack: uint8x16x4_t = uint8x16x4_t(b_values, g_values, r_values, v_alpha);
                vst4q_u8(rgba_ptr.add(dst_shift), dst_pack);
            }
        }

        cx += 16;

        match chroma_subsampling {
            YuvChromaSample::YUV420 | YuvChromaSample::YUV422 => {
                uv_x += 8;
            }
            YuvChromaSample::YUV444 => {
                uv_x += 16;
            }
        }
    }

    return ProcessedOffset { cx, ux: uv_x };
}

fn ycgco_ro_rgbx<const DESTINATION_CHANNELS: u8, const SAMPLING: u8>(
    y_plane: &[u8],
    y_stride: u32,
    cg_plane: &[u8],
    cg_stride: u32,
    co_plane: &[u8],
    co_stride: u32,
    a_plane: &[u8],
    a_stride: u32,
    rgba: &mut [u8],
    rgba_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    premultiply_alpha: bool,
) {
    let chroma_subsampling: YuvChromaSample = SAMPLING.into();
    let destination_channels: YuvSourceChannels = DESTINATION_CHANNELS.into();
    let channels = destination_channels.get_channels_count();
    let range = get_yuv_range(8, range);
    let bias_y = range.bias_y as i32;
    let bias_uv = range.bias_uv as i32;

    let mut y_offset = 0usize;
    let mut u_offset = 0usize;
    let mut v_offset = 0usize;
    let mut a_offset = 0usize;
    let mut rgba_offset = 0usize;

    let iterator_step = match chroma_subsampling {
        YuvChromaSample::YUV420 => 2usize,
        YuvChromaSample::YUV422 => 2usize,
        YuvChromaSample::YUV444 => 1usize,
    };

    let max_colors = 2i32.pow(8) - 1i32;
    let precision_scale = (1 << 6) as f32;

    let range_reduction_y =
        (max_colors as f32 / range.range_y as f32 * precision_scale).round() as i32;
    let range_reduction_uv =
        (max_colors as f32 / range.range_uv as f32 * precision_scale).round() as i32;

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    let mut _use_sse = false;
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    let mut _use_avx2 = false;
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    let mut _use_avx512 = false;

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::arch::is_x86_feature_detected!("sse4.1") {
            _use_sse = true;
        }
        #[cfg(target_feature = "avx2")]
        if std::arch::is_x86_feature_detected!("avx2") {
            _use_avx2 = true;
        }
        #[cfg(feature = "nightly_avx512")]
        if std::arch::is_x86_feature_detected!("avx512bw") {
            _use_avx512 = true;
        }
    }

    for y in 0..height as usize {
        #[allow(unused_variables)]
        #[allow(unused_mut)]
        let mut cx = 0usize;

        #[allow(unused_variables)]
        #[allow(unused_mut)]
        let mut uv_x = 0usize;

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        unsafe {
            #[cfg(feature = "nightly_avx512")]
            if _use_avx512 {
                let processed = avx512_process_row::<DESTINATION_CHANNELS, SAMPLING>(
                    &range,
                    y_plane,
                    cg_plane,
                    co_plane,
                    a_plane,
                    rgba,
                    cx,
                    uv_x,
                    y_offset,
                    u_offset,
                    v_offset,
                    a_offset,
                    rgba_offset,
                    width as usize,
                    premultiply_alpha,
                );
                cx = processed.cx;
                uv_x = processed.ux;
            }
            #[cfg(target_feature = "avx2")]
            if _use_avx2 {
                let processed = avx2_process_row::<DESTINATION_CHANNELS, SAMPLING>(
                    &range,
                    y_plane,
                    cg_plane,
                    co_plane,
                    a_plane,
                    rgba,
                    cx,
                    uv_x,
                    y_offset,
                    u_offset,
                    v_offset,
                    a_offset,
                    rgba_offset,
                    width as usize,
                    premultiply_alpha,
                );
                cx = processed.cx;
                uv_x = processed.ux;
            }
            if _use_sse {
                let processed = sse_process_row::<DESTINATION_CHANNELS, SAMPLING>(
                    &range,
                    y_plane,
                    cg_plane,
                    co_plane,
                    a_plane,
                    rgba,
                    cx,
                    uv_x,
                    y_offset,
                    u_offset,
                    v_offset,
                    a_offset,
                    rgba_offset,
                    width as usize,
                    premultiply_alpha,
                );
                cx = processed.cx;
                uv_x = processed.ux;
            }
        }

        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        unsafe {
            let processed = neon_process_row::<DESTINATION_CHANNELS, SAMPLING>(
                &range,
                y_plane,
                cg_plane,
                co_plane,
                a_plane,
                rgba,
                cx,
                uv_x,
                y_offset,
                u_offset,
                v_offset,
                a_offset,
                rgba_offset,
                width as usize,
                premultiply_alpha,
            );
            cx = processed.cx;
            uv_x = processed.ux;
        }

        for x in (cx..width as usize).step_by(iterator_step) {
            let y_value = (unsafe { *y_plane.get_unchecked(y_offset + x) } as i32 - bias_y)
                * range_reduction_y;

            let cg_pos = match chroma_subsampling {
                YuvChromaSample::YUV420 | YuvChromaSample::YUV422 => u_offset + uv_x,
                YuvChromaSample::YUV444 => u_offset + uv_x,
            };

            let cg_value =
                (unsafe { *cg_plane.get_unchecked(cg_pos) } as i32 - bias_uv) * range_reduction_uv;

            let v_pos = match chroma_subsampling {
                YuvChromaSample::YUV420 | YuvChromaSample::YUV422 => v_offset + uv_x,
                YuvChromaSample::YUV444 => v_offset + uv_x,
            };

            let co_value =
                (unsafe { *co_plane.get_unchecked(v_pos) } as i32 - bias_uv) * range_reduction_uv;

            let t = y_value - cg_value;

            let mut r = ((t + co_value) >> 6).min(255).max(0);
            let mut b = ((t - co_value) >> 6).min(255).max(0);
            let mut g = ((y_value + cg_value) >> 6).min(255).max(0);

            let a_value = unsafe { *a_plane.get_unchecked(a_offset + x) };
            if premultiply_alpha {
                r = (r * a_value as i32) / 255;
                g = (g * a_value as i32) / 255;
                b = (b * a_value as i32) / 255;
            }

            let px = x * channels;

            let rgba_shift = rgba_offset + px;

            unsafe {
                *rgba.get_unchecked_mut(rgba_shift + destination_channels.get_r_channel_offset()) =
                    r as u8
            };
            unsafe {
                *rgba.get_unchecked_mut(rgba_shift + destination_channels.get_g_channel_offset()) =
                    g as u8
            };
            unsafe {
                *rgba.get_unchecked_mut(rgba_shift + destination_channels.get_b_channel_offset()) =
                    b as u8
            };
            if destination_channels.has_alpha() {
                unsafe {
                    *rgba.get_unchecked_mut(
                        rgba_shift + destination_channels.get_a_channel_offset(),
                    ) = 255
                };
            }

            if chroma_subsampling == YuvChromaSample::YUV420
                || chroma_subsampling == YuvChromaSample::YUV422
            {
                let next_x = x + 1;
                if next_x < width as usize {
                    let y_value = (unsafe { *y_plane.get_unchecked(y_offset + next_x) } as i32
                        - bias_y)
                        * range_reduction_y;

                    let mut r = ((t + co_value) >> 6).min(255).max(0);
                    let mut b = ((t - co_value) >> 6).min(255).max(0);
                    let mut g = ((y_value + cg_value) >> 6).min(255).max(0);

                    let next_px = next_x * channels;

                    let rgba_shift = rgba_offset + next_px;

                    let a_value = unsafe { *a_plane.get_unchecked(a_offset + next_x) };
                    if premultiply_alpha {
                        r = (r * a_value as i32) / 255;
                        g = (g * a_value as i32) / 255;
                        b = (b * a_value as i32) / 255;
                    }

                    unsafe {
                        *rgba.get_unchecked_mut(
                            rgba_shift + destination_channels.get_r_channel_offset(),
                        ) = r as u8
                    };
                    unsafe {
                        *rgba.get_unchecked_mut(
                            rgba_shift + destination_channels.get_g_channel_offset(),
                        ) = g as u8
                    };
                    unsafe {
                        *rgba.get_unchecked_mut(
                            rgba_shift + destination_channels.get_b_channel_offset(),
                        ) = b as u8
                    };
                    if destination_channels.has_alpha() {
                        unsafe {
                            *rgba.get_unchecked_mut(
                                rgba_shift + destination_channels.get_a_channel_offset(),
                            ) = a_value;
                        };
                    }
                }
            }

            uv_x += 1;
        }

        y_offset += y_stride as usize;
        rgba_offset += rgba_stride as usize;
        a_offset += a_stride as usize;
        match chroma_subsampling {
            YuvChromaSample::YUV420 => {
                if y & 1 == 1 {
                    u_offset += cg_stride as usize;
                    v_offset += co_stride as usize;
                }
            }
            YuvChromaSample::YUV444 | YuvChromaSample::YUV422 => {
                u_offset += cg_stride as usize;
                v_offset += co_stride as usize;
            }
        }
    }
}

/// Convert YCgCo 420 planar format to RGBA format.
///
/// This function takes YCgCo 420 planar format data with 8-bit precision,
/// and converts it to RGBA format with 8-bit per channel precision.
///
/// # Arguments
///
/// * `y_plane` - A slice to load the Y (luminance) plane data.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `cg_plane` - A slice to load the U (chrominance) plane data.
/// * `cg_stride` - The stride (bytes per row) for the U plane.
/// * `co_plane` - A slice to load the V (chrominance) plane data.
/// * `co_stride` - The stride (bytes per row) for the V plane.
/// * `a_plane` - A slice to load alpha plane to append to result.
/// * `a_stride` - The stride (bytes per row) for the alpha plane.
/// * `width` - The width of the YUV image.
/// * `height` - The height of the YUV image.
/// * `rgba_data` - A mutable slice to store the converted RGBA data.
/// * `range` - The YUV range (limited or full).
/// * `premultiply_alpha` - Flag to premultiply alpha or not
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGRA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn ycgco420_with_alpha_to_rgba(
    y_plane: &[u8],
    y_stride: u32,
    cg_plane: &[u8],
    cg_stride: u32,
    co_plane: &[u8],
    co_stride: u32,
    a_plane: &[u8],
    a_stride: u32,
    rgba: &mut [u8],
    rgba_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    premultiply_alpha: bool,
) {
    ycgco_ro_rgbx::<{ YuvSourceChannels::Rgba as u8 }, { YuvChromaSample::YUV420 as u8 }>(
        y_plane,
        y_stride,
        cg_plane,
        cg_stride,
        co_plane,
        co_stride,
        a_plane,
        a_stride,
        rgba,
        rgba_stride,
        width,
        height,
        range,
        premultiply_alpha,
    )
}

/// Convert YCgCo 420 planar format to BGRA format.
///
/// This function takes YCgCo 420 planar format data with 8-bit precision,
/// and converts it to BGRA format with 8-bit per channel precision.
///
/// # Arguments
///
/// * `y_plane` - A slice to load the Y (luminance) plane data.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `cg_plane` - A slice to load the U (chrominance) plane data.
/// * `cg_stride` - The stride (bytes per row) for the U plane.
/// * `co_plane` - A slice to load the V (chrominance) plane data.
/// * `co_stride` - The stride (bytes per row) for the V plane.
/// * `a_plane` - A slice to load alpha plane to append to result.
/// * `a_stride` - The stride (bytes per row) for the alpha plane.
/// * `width` - The width of the YUV image.
/// * `height` - The height of the YUV image.
/// * `bgra_data` - A mutable slice to store the converted BGRA data.
/// * `range` - The YUV range (limited or full).
/// * `premultiply_alpha` - Flag to premultiply alpha or not
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGRA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn ycgco420_with_alpha_to_bgra(
    y_plane: &[u8],
    y_stride: u32,
    cg_plane: &[u8],
    cg_stride: u32,
    co_plane: &[u8],
    co_stride: u32,
    a_plane: &[u8],
    a_stride: u32,
    bgra: &mut [u8],
    bgra_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    premultiply_alpha: bool,
) {
    ycgco_ro_rgbx::<{ YuvSourceChannels::Bgra as u8 }, { YuvChromaSample::YUV420 as u8 }>(
        y_plane,
        y_stride,
        cg_plane,
        cg_stride,
        co_plane,
        co_stride,
        a_plane,
        a_stride,
        bgra,
        bgra_stride,
        width,
        height,
        range,
        premultiply_alpha,
    )
}

/// Convert YCgCo 422 planar format to RGBA format.
///
/// This function takes YCgCo 422 data with 8-bit precision,
/// and converts it to RGBA format with 8-bit per channel precision.
///
/// # Arguments
///
/// * `y_plane` - A slice to load the Y (luminance) plane data.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `cg_plane` - A slice to load the U (chrominance) plane data.
/// * `cg_stride` - The stride (bytes per row) for the U plane.
/// * `co_plane` - A slice to load the V (chrominance) plane data.
/// * `co_stride` - The stride (bytes per row) for the V plane.
/// * `a_plane` - A slice to load alpha plane to append to result.
/// * `a_stride` - The stride (bytes per row) for the alpha plane.
/// * `width` - The width of the YUV image.
/// * `height` - The height of the YUV image.
/// * `bgra_data` - A mutable slice to store the converted RGBA data.
/// * `range` - The YUV range (limited or full).
/// * `premultiply_alpha` - Flag to premultiply alpha or not
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGRA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn ycgco422_with_alpha_to_rgba(
    y_plane: &[u8],
    y_stride: u32,
    cg_plane: &[u8],
    cg_stride: u32,
    co_plane: &[u8],
    co_stride: u32,
    a_plane: &[u8],
    a_stride: u32,
    rgba: &mut [u8],
    rgba_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    premultiply_alpha: bool,
) {
    ycgco_ro_rgbx::<{ YuvSourceChannels::Rgba as u8 }, { YuvChromaSample::YUV422 as u8 }>(
        y_plane,
        y_stride,
        cg_plane,
        cg_stride,
        co_plane,
        co_stride,
        a_plane,
        a_stride,
        rgba,
        rgba_stride,
        width,
        height,
        range,
        premultiply_alpha,
    )
}

/// Convert YCgCo 422 planar format to BGRA format.
///
/// This function takes YCgCo 422 data with 8-bit precision,
/// and converts it to BGRA format with 8-bit per channel precision.
///
/// # Arguments
///
/// * `y_plane` - A slice to load the Y (luminance) plane data.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `cg_plane` - A slice to load the U (chrominance) plane data.
/// * `cg_stride` - The stride (bytes per row) for the U plane.
/// * `co_plane` - A slice to load the V (chrominance) plane data.
/// * `co_stride` - The stride (bytes per row) for the V plane.
/// * `a_plane` - A slice to load alpha plane to append to result.
/// * `a_stride` - The stride (bytes per row) for the alpha plane.
/// * `width` - The width of the YUV image.
/// * `height` - The height of the YUV image.
/// * `bgra_data` - A mutable slice to store the converted BGRA data.
/// * `range` - The YUV range (limited or full).
/// * `premultiply_alpha` - Flag to premultiply alpha or not
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGRA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn ycgco422_with_alpha_to_bgra(
    y_plane: &[u8],
    y_stride: u32,
    cg_plane: &[u8],
    cg_stride: u32,
    co_plane: &[u8],
    co_stride: u32,
    a_plane: &[u8],
    a_stride: u32,
    bgra: &mut [u8],
    bgra_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    premultiply_alpha: bool,
) {
    ycgco_ro_rgbx::<{ YuvSourceChannels::Bgra as u8 }, { YuvChromaSample::YUV422 as u8 }>(
        y_plane,
        y_stride,
        cg_plane,
        cg_stride,
        co_plane,
        co_stride,
        a_plane,
        a_stride,
        bgra,
        bgra_stride,
        width,
        height,
        range,
        premultiply_alpha,
    )
}

/// Convert YCgCo 444 planar format to RGBA format.
///
/// This function takes YCgCo 444 data with 8-bit precision,
/// and converts it to RGBA format with 8-bit per channel precision.
///
/// # Arguments
///
/// * `y_plane` - A slice to load the Y (luminance) plane data.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `cg_plane` - A slice to load the U (chrominance) plane data.
/// * `cg_stride` - The stride (bytes per row) for the U plane.
/// * `co_plane` - A slice to load the V (chrominance) plane data.
/// * `co_stride` - The stride (bytes per row) for the V plane.
/// * `a_plane` - A slice to load alpha plane to append to result.
/// * `a_stride` - The stride (bytes per row) for the alpha plane.
/// * `width` - The width of the YUV image.
/// * `height` - The height of the YUV image.
/// * `rgba_data` - A mutable slice to store the converted RGBA data.
/// * `range` - The YUV range (limited or full).
/// * `premultiply_alpha` - Flag to premultiply alpha or not
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGRA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn ycgco444_with_alpha_to_rgba(
    y_plane: &[u8],
    y_stride: u32,
    cg_plane: &[u8],
    cg_stride: u32,
    co_plane: &[u8],
    co_stride: u32,
    a_plane: &[u8],
    a_stride: u32,
    rgba: &mut [u8],
    rgba_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    premultiply_alpha: bool,
) {
    ycgco_ro_rgbx::<{ YuvSourceChannels::Rgba as u8 }, { YuvChromaSample::YUV444 as u8 }>(
        y_plane,
        y_stride,
        cg_plane,
        cg_stride,
        co_plane,
        co_stride,
        a_plane,
        a_stride,
        rgba,
        rgba_stride,
        width,
        height,
        range,
        premultiply_alpha,
    )
}

/// Convert YCgCo 444 planar format to BGRA format.
///
/// This function takes YCgCo 444 data with 8-bit precision,
/// and converts it to BGRA format with 8-bit per channel precision.
///
/// # Arguments
///
/// * `y_plane` - A slice to load the Y (luminance) plane data.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `cg_plane` - A slice to load the U (chrominance) plane data.
/// * `cg_stride` - The stride (bytes per row) for the U plane.
/// * `co_plane` - A slice to load the V (chrominance) plane data.
/// * `co_stride` - The stride (bytes per row) for the V plane.
/// * `a_plane` - A slice to load alpha plane to append to result.
/// * `a_stride` - The stride (bytes per row) for the alpha plane.
/// * `width` - The width of the YUV image.
/// * `height` - The height of the YUV image.
/// * `bgra_data` - A mutable slice to store the converted BGRA data.
/// * `range` - The YUV range (limited or full).
/// * `premultiply_alpha` - Flag to premultiply alpha or not
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGRA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn ycgco444_with_alpha_to_bgra(
    y_plane: &[u8],
    y_stride: u32,
    cg_plane: &[u8],
    cg_stride: u32,
    co_plane: &[u8],
    co_stride: u32,
    a_plane: &[u8],
    a_stride: u32,
    bgra: &mut [u8],
    bgra_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    premultiply_alpha: bool,
) {
    ycgco_ro_rgbx::<{ YuvSourceChannels::Bgra as u8 }, { YuvChromaSample::YUV444 as u8 }>(
        y_plane,
        y_stride,
        cg_plane,
        cg_stride,
        co_plane,
        co_stride,
        a_plane,
        a_stride,
        bgra,
        bgra_stride,
        width,
        height,
        range,
        premultiply_alpha,
    )
}
