/*
 * // Copyright (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

use crate::internals::ProcessedOffset;
use crate::sse::sse_support::{sse_store_rgb_u8, sse_store_rgba};
use crate::sse::{sse_store_rgb_half_u8, sse_store_rgba_half_epi8};
use crate::yuv_support::{
    CbCrInverseTransform, YuvChromaRange, YuvChromaSample, YuvSourceChannels,
};
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[target_feature(enable = "sse4.1")]
pub unsafe fn sse_yuv_to_rgba_row<const DESTINATION_CHANNELS: u8, const SAMPLING: u8>(
    range: &YuvChromaRange,
    transform: &CbCrInverseTransform<i32>,
    y_plane: &[u8],
    u_plane: &[u8],
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

    let mut cx = start_cx;
    let mut uv_x = start_ux;

    let y_ptr = y_plane.as_ptr();
    let u_ptr = u_plane.as_ptr();
    let v_ptr = v_plane.as_ptr();
    let rgba_ptr = rgba.as_mut_ptr();

    let y_corr = _mm_set1_epi8(range.bias_y as i8);
    let uv_corr = _mm_set1_epi16(range.bias_uv as i16);
    let v_luma_coeff = _mm_set1_epi16(transform.y_coef as i16);
    let v_cr_coeff = _mm_set1_epi16(transform.cr_coef as i16);
    let v_cb_coeff = _mm_set1_epi16(transform.cb_coef as i16);
    let v_g_coeff_1 = _mm_set1_epi16(-1 * transform.g_coeff_1 as i16);
    let v_g_coeff_2 = _mm_set1_epi16(-1 * transform.g_coeff_2 as i16);
    let v_alpha = _mm_set1_epi8(255u8 as i8);
    let rounding_const = _mm_set1_epi16(1 << 5);

    let zeros = _mm_setzero_si128();

    while cx + 16 < width {
        let y_values = _mm_subs_epu8(
            _mm_loadu_si128(y_ptr.add(y_offset + cx) as *const __m128i),
            y_corr,
        );

        let (u_high_u16, v_high_u16, u_low_u16, v_low_u16);

        match chroma_subsampling {
            YuvChromaSample::YUV420 | YuvChromaSample::YUV422 => {
                let reshuffle = _mm_setr_epi8(0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7);
                let u_values =
                    _mm_shuffle_epi8(_mm_loadu_si64(u_ptr.add(u_offset + uv_x)), reshuffle);
                let v_values =
                    _mm_shuffle_epi8(_mm_loadu_si64(v_ptr.add(v_offset + uv_x)), reshuffle);

                u_high_u16 = _mm_unpackhi_epi8(u_values, zeros);
                v_high_u16 = _mm_unpackhi_epi8(v_values, zeros);
                u_low_u16 = _mm_unpacklo_epi8(u_values, zeros);
                v_low_u16 = _mm_unpacklo_epi8(v_values, zeros);
            }
            YuvChromaSample::YUV444 => {
                let u_values = _mm_loadu_si128(u_ptr.add(u_offset + uv_x) as *const __m128i);
                let v_values = _mm_loadu_si128(v_ptr.add(v_offset + uv_x) as *const __m128i);

                u_high_u16 = _mm_unpackhi_epi8(u_values, zeros);
                v_high_u16 = _mm_unpackhi_epi8(v_values, zeros);
                u_low_u16 = _mm_unpacklo_epi8(u_values, zeros);
                v_low_u16 = _mm_unpacklo_epi8(v_values, zeros);
            }
        }

        let u_high = _mm_subs_epi16(u_high_u16, uv_corr);
        let v_high = _mm_subs_epi16(v_high_u16, uv_corr);
        let y_high = _mm_mullo_epi16(_mm_unpackhi_epi8(y_values, zeros), v_luma_coeff);

        let r_high = _mm_srai_epi16::<6>(_mm_adds_epi16(
            _mm_max_epi16(
                _mm_adds_epi16(y_high, _mm_mullo_epi16(v_high, v_cr_coeff)),
                zeros,
            ),
            rounding_const,
        ));
        let b_high = _mm_srai_epi16::<6>(_mm_adds_epi16(
            _mm_max_epi16(
                _mm_adds_epi16(y_high, _mm_mullo_epi16(u_high, v_cb_coeff)),
                zeros,
            ),
            rounding_const,
        ));
        let g_high = _mm_srai_epi16::<6>(_mm_adds_epi16(
            _mm_max_epi16(
                _mm_adds_epi16(
                    y_high,
                    _mm_adds_epi16(
                        _mm_mullo_epi16(v_high, v_g_coeff_1),
                        _mm_mullo_epi16(u_high, v_g_coeff_2),
                    ),
                ),
                zeros,
            ),
            rounding_const,
        ));

        let u_low = _mm_subs_epi16(u_low_u16, uv_corr);
        let v_low = _mm_subs_epi16(v_low_u16, uv_corr);
        let y_low = _mm_mullo_epi16(_mm_cvtepu8_epi16(y_values), v_luma_coeff);

        let r_low = _mm_srai_epi16::<6>(_mm_adds_epi16(
            _mm_max_epi16(
                _mm_adds_epi16(y_low, _mm_mullo_epi16(v_low, v_cr_coeff)),
                zeros,
            ),
            rounding_const,
        ));
        let b_low = _mm_srai_epi16::<6>(_mm_adds_epi16(
            _mm_max_epi16(
                _mm_adds_epi16(y_low, _mm_mullo_epi16(u_low, v_cb_coeff)),
                zeros,
            ),
            rounding_const,
        ));
        let g_low = _mm_srai_epi16::<6>(_mm_adds_epi16(
            _mm_max_epi16(
                _mm_adds_epi16(
                    y_low,
                    _mm_adds_epi16(
                        _mm_mullo_epi16(v_low, v_g_coeff_1),
                        _mm_mullo_epi16(u_low, v_g_coeff_2),
                    ),
                ),
                zeros,
            ),
            rounding_const,
        ));

        let r_values = _mm_packus_epi16(r_low, r_high);
        let g_values = _mm_packus_epi16(g_low, g_high);
        let b_values = _mm_packus_epi16(b_low, b_high);

        let dst_shift = rgba_offset + cx * channels;

        match destination_channels {
            YuvSourceChannels::Rgb => {
                sse_store_rgb_u8(rgba_ptr.add(dst_shift), r_values, g_values, b_values);
            }
            YuvSourceChannels::Bgr => {
                sse_store_rgb_u8(rgba_ptr.add(dst_shift), b_values, g_values, r_values);
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

    while cx + 8 < width {
        let y_values = _mm_subs_epi8(_mm_loadu_si64(y_ptr.add(y_offset + cx)), y_corr);

        let (u_low_u16, v_low_u16);

        match chroma_subsampling {
            YuvChromaSample::YUV420 | YuvChromaSample::YUV422 => {
                let reshuffle = _mm_setr_epi8(0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7);
                let u_values =
                    _mm_shuffle_epi8(_mm_loadu_si32(u_ptr.add(u_offset + uv_x)), reshuffle);
                let v_values =
                    _mm_shuffle_epi8(_mm_loadu_si32(v_ptr.add(v_offset + uv_x)), reshuffle);

                u_low_u16 = _mm_unpacklo_epi8(u_values, zeros);
                v_low_u16 = _mm_unpacklo_epi8(v_values, zeros);
            }
            YuvChromaSample::YUV444 => {
                let u_values = _mm_loadu_si64(u_ptr.add(u_offset + uv_x));
                let v_values = _mm_loadu_si64(v_ptr.add(v_offset + uv_x));

                u_low_u16 = _mm_unpacklo_epi8(u_values, zeros);
                v_low_u16 = _mm_unpacklo_epi8(v_values, zeros);
            }
        }

        let u_low = _mm_subs_epi16(u_low_u16, uv_corr);
        let v_low = _mm_subs_epi16(v_low_u16, uv_corr);
        let y_low = _mm_mullo_epi16(_mm_cvtepu8_epi16(y_values), v_luma_coeff);

        let r_low = _mm_srai_epi16::<6>(_mm_adds_epi16(
            _mm_max_epi16(
                _mm_adds_epi16(y_low, _mm_mullo_epi16(v_low, v_cr_coeff)),
                zeros,
            ),
            rounding_const,
        ));
        let b_low = _mm_srai_epi16::<6>(_mm_adds_epi16(
            _mm_max_epi16(
                _mm_adds_epi16(y_low, _mm_mullo_epi16(u_low, v_cb_coeff)),
                zeros,
            ),
            rounding_const,
        ));
        let g_low = _mm_srai_epi16::<6>(_mm_adds_epi16(
            _mm_max_epi16(
                _mm_adds_epi16(
                    y_low,
                    _mm_adds_epi16(
                        _mm_mullo_epi16(v_low, v_g_coeff_1),
                        _mm_mullo_epi16(u_low, v_g_coeff_2),
                    ),
                ),
                zeros,
            ),
            rounding_const,
        ));

        let r_values = _mm_packus_epi16(r_low, zeros);
        let g_values = _mm_packus_epi16(g_low, zeros);
        let b_values = _mm_packus_epi16(b_low, zeros);

        let dst_shift = rgba_offset + cx * channels;

        match destination_channels {
            YuvSourceChannels::Rgb => {
                sse_store_rgb_half_u8(rgba_ptr.add(dst_shift), r_values, g_values, b_values);
            }
            YuvSourceChannels::Bgr => {
                sse_store_rgb_half_u8(rgba_ptr.add(dst_shift), b_values, g_values, r_values);
            }
            YuvSourceChannels::Rgba => {
                sse_store_rgba_half_epi8(
                    rgba_ptr.add(dst_shift),
                    r_values,
                    g_values,
                    b_values,
                    v_alpha,
                );
            }
            YuvSourceChannels::Bgra => {
                sse_store_rgba_half_epi8(
                    rgba_ptr.add(dst_shift),
                    b_values,
                    g_values,
                    r_values,
                    v_alpha,
                );
            }
        }

        cx += 8;

        match chroma_subsampling {
            YuvChromaSample::YUV420 | YuvChromaSample::YUV422 => {
                uv_x += 4;
            }
            YuvChromaSample::YUV444 => {
                uv_x += 8;
            }
        }
    }

    ProcessedOffset { cx, ux: uv_x }
}
