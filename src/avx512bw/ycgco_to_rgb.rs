/*
 * // Copyright (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

use crate::avx512bw::avx512_utils::{avx512_pack_u16, avx512_rgb_u8, avx512_rgba_u8, shuffle};
use crate::internals::ProcessedOffset;
use crate::yuv_support::{YuvChromaRange, YuvChromaSample, YuvSourceChannels};
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[target_feature(enable = "avx512bw")]
pub unsafe fn avx512_ycgco_to_rgb_row<const DESTINATION_CHANNELS: u8, const SAMPLING: u8>(
    range: &YuvChromaRange,
    y_plane: &[u8],
    cg_plane: &[u8],
    v_plane: &[u8],
    rgba: &mut [u8],
    start_cx: usize,
    start_ux: usize,
    y_offset: usize,
    u_offset: usize,
    v_offset: usize,
    rgba_offset: usize,
    width: usize,
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

        let r_values = avx512_pack_u16(r_low, r_high);
        let g_values = avx512_pack_u16(g_low, g_high);
        let b_values = avx512_pack_u16(b_low, b_high);

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
