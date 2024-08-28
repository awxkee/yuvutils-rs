/*
 * // Copyright (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

use crate::avx512bw::avx512_utils::*;
use crate::internals::ProcessedOffset;
use crate::yuv_support::{
    CbCrInverseTransform, YuvChromaRange, YuvChromaSample, YuvSourceChannels,
};
#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

#[target_feature(enable = "avx512bw")]
pub unsafe fn avx512_yuv_to_rgba_alpha<const DESTINATION_CHANNELS: u8, const SAMPLING: u8>(
    range: &YuvChromaRange,
    transform: &CbCrInverseTransform<i32>,
    y_plane: &[u8],
    u_plane: &[u8],
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
    use_premultiply: bool,
) -> ProcessedOffset {
    let chroma_subsampling: YuvChromaSample = SAMPLING.into();
    let destination_channels: YuvSourceChannels = DESTINATION_CHANNELS.into();
    let channels = destination_channels.get_channels_count();

    let mut cx = start_cx;
    let mut uv_x = start_ux;
    let y_ptr = y_plane.as_ptr();
    let u_ptr = u_plane.as_ptr();
    let v_ptr = v_plane.as_ptr();
    let rgba_ptr = rgba.as_mut_ptr();

    let y_corr = _mm512_set1_epi8(range.bias_y as i8);
    let uv_corr = _mm512_set1_epi16(range.bias_uv as i16);
    let v_luma_coeff = _mm512_set1_epi16(transform.y_coef as i16);
    let v_cr_coeff = _mm512_set1_epi16(transform.cr_coef as i16);
    let v_cb_coeff = _mm512_set1_epi16(transform.cb_coef as i16);
    let v_min_values = _mm512_setzero_si512();
    let v_g_coeff_1 = _mm512_set1_epi16(-1 * transform.g_coeff_1 as i16);
    let v_g_coeff_2 = _mm512_set1_epi16(-1 * transform.g_coeff_2 as i16);

    while cx + 64 < width {
        let y_values = _mm512_subs_epi8(
            _mm512_loadu_si512(y_ptr.add(y_offset + cx) as *const i32),
            y_corr,
        );

        let (u_high_u8, v_high_u8, u_low_u8, v_low_u8);

        match chroma_subsampling {
            YuvChromaSample::YUV420 | YuvChromaSample::YUV422 => {
                let u_values = _mm256_loadu_si256(u_ptr.add(u_offset + uv_x) as *const __m256i);
                let v_values = _mm256_loadu_si256(v_ptr.add(v_offset + uv_x) as *const __m256i);

                let (u_low, u_high) = avx2_zip(u_values, u_values);
                let (v_low, v_high) = avx2_zip(v_values, v_values);

                u_high_u8 = u_high;
                v_high_u8 = v_high;
                u_low_u8 = u_low;
                v_low_u8 = v_low;
            }
            YuvChromaSample::YUV444 => {
                let u_values = _mm512_loadu_si512(u_ptr.add(u_offset + uv_x) as *const i32);
                let v_values = _mm512_loadu_si512(v_ptr.add(v_offset + uv_x) as *const i32);

                u_high_u8 = _mm512_extracti64x4_epi64::<1>(u_values);
                v_high_u8 = _mm512_extracti64x4_epi64::<1>(v_values);
                u_low_u8 = _mm512_castsi512_si256(u_values);
                v_low_u8 = _mm512_castsi512_si256(v_values);
            }
        }

        let u_high = _mm512_subs_epi16(_mm512_cvtepu8_epi16(u_high_u8), uv_corr);
        let v_high = _mm512_subs_epi16(_mm512_cvtepu8_epi16(v_high_u8), uv_corr);
        let y_high = _mm512_mullo_epi16(
            _mm512_cvtepu8_epi16(_mm512_extracti64x4_epi64::<1>(y_values)),
            v_luma_coeff,
        );

        let r_high = _mm512_srli_epi16::<6>(_mm512_max_epi16(
            _mm512_adds_epi16(y_high, _mm512_mullo_epi16(v_high, v_cr_coeff)),
            v_min_values,
        ));
        let b_high = _mm512_srli_epi16::<6>(_mm512_max_epi16(
            _mm512_adds_epi16(y_high, _mm512_mullo_epi16(u_high, v_cb_coeff)),
            v_min_values,
        ));
        let g_high = _mm512_srli_epi16::<6>(_mm512_max_epi16(
            _mm512_adds_epi16(
                y_high,
                _mm512_adds_epi16(
                    _mm512_mullo_epi16(v_high, v_g_coeff_1),
                    _mm512_mullo_epi16(u_high, v_g_coeff_2),
                ),
            ),
            v_min_values,
        ));

        let u_low = _mm512_subs_epi16(_mm512_cvtepu8_epi16(u_low_u8), uv_corr);
        let v_low = _mm512_subs_epi16(_mm512_cvtepu8_epi16(v_low_u8), uv_corr);
        let y_low = _mm512_mullo_epi16(
            _mm512_cvtepu8_epi16(_mm512_castsi512_si256(y_values)),
            v_luma_coeff,
        );

        let r_low = _mm512_srli_epi16::<6>(_mm512_max_epi16(
            _mm512_adds_epi16(y_low, _mm512_mullo_epi16(v_low, v_cr_coeff)),
            v_min_values,
        ));
        let b_low = _mm512_srli_epi16::<6>(_mm512_max_epi16(
            _mm512_adds_epi16(y_low, _mm512_mullo_epi16(u_low, v_cb_coeff)),
            v_min_values,
        ));
        let g_low = _mm512_srli_epi16::<6>(_mm512_max_epi16(
            _mm512_adds_epi16(
                y_low,
                _mm512_adds_epi16(
                    _mm512_mullo_epi16(v_low, v_g_coeff_1),
                    _mm512_mullo_epi16(u_low, v_g_coeff_2),
                ),
            ),
            v_min_values,
        ));

        let a_values = _mm512_loadu_si512(a_plane.as_ptr().add(a_offset + cx) as *const i32);

        let (r_values, g_values, b_values);

        if use_premultiply {
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

        let dst_shift = rgba_offset + cx * channels;

        match destination_channels {
            YuvSourceChannels::Rgb => {
                // We need always to write 104 bytes, however 32 initial offset is safe only for 96, then if there are some exceed it is required to use transient buffer
                let ptr = rgba_ptr.add(dst_shift);
                avx512_rgb_u8(ptr, r_values, g_values, b_values);
            }
            YuvSourceChannels::Rgba => {
                avx512_rgba_u8(
                    rgba_ptr.add(dst_shift),
                    r_values,
                    g_values,
                    b_values,
                    a_values,
                );
            }
            YuvSourceChannels::Bgra => {
                avx512_rgba_u8(
                    rgba_ptr.add(dst_shift),
                    b_values,
                    g_values,
                    r_values,
                    a_values,
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

    ProcessedOffset { cx, ux: uv_x }
}
