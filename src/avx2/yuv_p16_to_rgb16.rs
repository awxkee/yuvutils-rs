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

use crate::avx2::avx2_utils::{
    _mm256_expand_bp_by2, _mm256_from_msb_epi16, _mm256_interleave_epi16,
    _mm256_store_interleave_rgb16_for_yuv,
};
use crate::internals::ProcessedOffset;
use crate::sse::_mm_from_msb_epi16;
use crate::yuv_support::{
    to_channels_layout, to_subsampling, CbCrInverseTransform, YuvBytesPacking, YuvChromaRange,
    YuvChromaSubsampling, YuvEndianness, YuvSourceChannels,
};
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

pub(crate) unsafe fn avx_yuv_p16_to_rgba_row<
    const DESTINATION_CHANNELS: u8,
    const SAMPLING: u8,
    const ENDIANNESS: u8,
    const BYTES_POSITION: u8,
    const BIT_DEPTH: usize,
    const PRECISION: i32,
>(
    y_plane: &[u16],
    u_plane: &[u16],
    v_plane: &[u16],
    bgra: &mut [u16],
    width: u32,
    range: &YuvChromaRange,
    transform: &CbCrInverseTransform<i32>,
    start_cx: usize,
    start_ux: usize,
) -> ProcessedOffset {
    unsafe {
        avx_yuv_p16_to_rgba_row_impl::<
            DESTINATION_CHANNELS,
            SAMPLING,
            ENDIANNESS,
            BYTES_POSITION,
            BIT_DEPTH,
            PRECISION,
        >(
            y_plane, u_plane, v_plane, bgra, width, range, transform, start_cx, start_ux,
        )
    }
}

#[target_feature(enable = "avx2")]
unsafe fn avx_yuv_p16_to_rgba_row_impl<
    const DESTINATION_CHANNELS: u8,
    const SAMPLING: u8,
    const ENDIANNESS: u8,
    const BYTES_POSITION: u8,
    const BIT_DEPTH: usize,
    const PRECISION: i32,
>(
    y_plane: &[u16],
    u_plane: &[u16],
    v_plane: &[u16],
    bgra: &mut [u16],
    width: u32,
    range: &YuvChromaRange,
    transform: &CbCrInverseTransform<i32>,
    start_cx: usize,
    start_ux: usize,
) -> ProcessedOffset {
    let destination_channels: YuvSourceChannels = to_channels_layout(DESTINATION_CHANNELS);
    let channels = destination_channels.get_channels_count();
    let chroma_subsampling: YuvChromaSubsampling = to_subsampling(SAMPLING);
    let endianness: YuvEndianness = ENDIANNESS.into();
    let bytes_position: YuvBytesPacking = BYTES_POSITION.into();
    let cr_coef = transform.cr_coef;
    let cb_coef = transform.cb_coef;
    let y_coef = transform.y_coef;
    let g_coef_1 = transform.g_coeff_1;
    let g_coef_2 = transform.g_coeff_2;

    let bias_y = range.bias_y as i32;
    let bias_uv = range.bias_uv as i32;

    let dst_ptr = bgra;

    let v_max_colors = _mm256_set1_epi16((1i16 << BIT_DEPTH as i16) - 1);

    let y_corr = _mm256_set1_epi16(bias_y as i16);
    let uv_corr = _mm256_set1_epi16(bias_uv as i16);
    let uv_corr_q = _mm256_set1_epi16(bias_uv as i16);
    let v_luma_coeff = _mm256_set1_epi16(y_coef as i16);
    let v_cr_coeff = _mm256_set1_epi16(cr_coef as i16);
    let v_cb_coeff = _mm256_set1_epi16(cb_coef as i16);
    let zeros = _mm256_setzero_si256();
    let v_g_coeff_1 = _mm256_set1_epi16(-(g_coef_1 as i16));
    let v_g_coeff_2 = _mm256_set1_epi16(-(g_coef_2 as i16));

    let mut cx = start_cx;
    let mut ux = start_ux;

    let big_endian_shuffle_flag = _mm256_setr_epi8(
        1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14, 1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10,
        13, 12, 15, 14,
    );
    let big_endian_shuffle_flag_sse =
        _mm_setr_epi8(1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14);

    const SCALE: i32 = 2;

    while cx + 32 < width as usize {
        let dst_ptr = dst_ptr.get_unchecked_mut(cx * channels..);

        let mut y_vl0 = _mm256_loadu_si256(y_plane.get_unchecked(cx..).as_ptr() as *const __m256i);
        let mut y_vl1 =
            _mm256_loadu_si256(y_plane.get_unchecked((cx + 16)..).as_ptr() as *const __m256i);
        if endianness == YuvEndianness::BigEndian {
            y_vl0 = _mm256_shuffle_epi8(y_vl0, big_endian_shuffle_flag);
            y_vl1 = _mm256_shuffle_epi8(y_vl1, big_endian_shuffle_flag);
        }
        if bytes_position == YuvBytesPacking::MostSignificantBytes {
            y_vl0 = _mm256_from_msb_epi16::<BIT_DEPTH>(y_vl0);
            y_vl1 = _mm256_from_msb_epi16::<BIT_DEPTH>(y_vl1);
        }
        let mut y_values0 = _mm256_subs_epu16(y_vl0, y_corr);
        let mut y_values1 = _mm256_subs_epu16(y_vl1, y_corr);

        let mut u_values0;
        let mut v_values0;

        let mut u_values1;
        let mut v_values1;

        match chroma_subsampling {
            YuvChromaSubsampling::Yuv420 | YuvChromaSubsampling::Yuv422 => {
                let mut u_vals =
                    _mm256_loadu_si256(u_plane.get_unchecked(ux..).as_ptr() as *const __m256i);
                let mut v_vals =
                    _mm256_loadu_si256(v_plane.get_unchecked(ux..).as_ptr() as *const __m256i);

                if endianness == YuvEndianness::BigEndian {
                    u_vals = _mm256_shuffle_epi8(u_vals, big_endian_shuffle_flag);
                    v_vals = _mm256_shuffle_epi8(v_vals, big_endian_shuffle_flag);
                }
                if bytes_position == YuvBytesPacking::MostSignificantBytes {
                    u_vals = _mm256_from_msb_epi16::<BIT_DEPTH>(u_vals);
                    v_vals = _mm256_from_msb_epi16::<BIT_DEPTH>(v_vals);
                }

                (u_values0, u_values1) = _mm256_interleave_epi16(u_vals, u_vals);
                (v_values0, v_values1) = _mm256_interleave_epi16(v_vals, v_vals);

                u_values0 = _mm256_sub_epi16(u_values0, uv_corr);
                v_values0 = _mm256_sub_epi16(v_values0, uv_corr);

                u_values1 = _mm256_sub_epi16(u_values1, uv_corr);
                v_values1 = _mm256_sub_epi16(v_values1, uv_corr);
            }
            YuvChromaSubsampling::Yuv444 => {
                let mut u_vals0 =
                    _mm256_loadu_si256(u_plane.get_unchecked(ux..).as_ptr() as *const __m256i);
                let mut v_vals0 =
                    _mm256_loadu_si256(v_plane.get_unchecked(ux..).as_ptr() as *const __m256i);
                let mut u_vals1 = _mm256_loadu_si256(
                    u_plane.get_unchecked((ux + 16)..).as_ptr() as *const __m256i
                );
                let mut v_vals1 = _mm256_loadu_si256(
                    v_plane.get_unchecked((ux + 16)..).as_ptr() as *const __m256i
                );

                if endianness == YuvEndianness::BigEndian {
                    u_vals0 = _mm256_shuffle_epi8(u_vals0, big_endian_shuffle_flag);
                    v_vals0 = _mm256_shuffle_epi8(v_vals0, big_endian_shuffle_flag);

                    u_vals1 = _mm256_shuffle_epi8(u_vals1, big_endian_shuffle_flag);
                    v_vals1 = _mm256_shuffle_epi8(v_vals1, big_endian_shuffle_flag);
                }
                if bytes_position == YuvBytesPacking::MostSignificantBytes {
                    u_vals0 = _mm256_from_msb_epi16::<BIT_DEPTH>(u_vals0);
                    v_vals0 = _mm256_from_msb_epi16::<BIT_DEPTH>(v_vals0);
                    u_vals1 = _mm256_from_msb_epi16::<BIT_DEPTH>(u_vals1);
                    v_vals1 = _mm256_from_msb_epi16::<BIT_DEPTH>(v_vals1);
                }
                u_values0 = _mm256_sub_epi16(u_vals0, uv_corr);
                v_values0 = _mm256_sub_epi16(v_vals0, uv_corr);

                u_values1 = _mm256_sub_epi16(u_vals1, uv_corr);
                v_values1 = _mm256_sub_epi16(v_vals1, uv_corr);
            }
        }

        u_values0 = _mm256_slli_epi16::<SCALE>(u_values0);
        v_values0 = _mm256_slli_epi16::<SCALE>(v_values0);
        y_values0 = _mm256_expand_bp_by2::<BIT_DEPTH>(y_values0);

        u_values1 = _mm256_slli_epi16::<SCALE>(u_values1);
        v_values1 = _mm256_slli_epi16::<SCALE>(v_values1);
        y_values1 = _mm256_expand_bp_by2::<BIT_DEPTH>(y_values1);

        let y_vals0 = _mm256_mulhrs_epi16(y_values0, v_luma_coeff);
        let y_vals1 = _mm256_mulhrs_epi16(y_values1, v_luma_coeff);

        let r_vals0 = _mm256_add_epi16(y_vals0, _mm256_mulhrs_epi16(v_values0, v_cr_coeff));
        let b_vals0 = _mm256_add_epi16(y_vals0, _mm256_mulhrs_epi16(u_values0, v_cb_coeff));
        let g_vals0 = _mm256_add_epi16(
            _mm256_add_epi16(y_vals0, _mm256_mulhrs_epi16(v_values0, v_g_coeff_1)),
            _mm256_mulhrs_epi16(u_values0, v_g_coeff_2),
        );

        let r_vals1 = _mm256_add_epi16(y_vals1, _mm256_mulhrs_epi16(v_values1, v_cr_coeff));
        let b_vals1 = _mm256_add_epi16(y_vals1, _mm256_mulhrs_epi16(u_values1, v_cb_coeff));
        let g_vals1 = _mm256_add_epi16(
            _mm256_add_epi16(y_vals1, _mm256_mulhrs_epi16(v_values1, v_g_coeff_1)),
            _mm256_mulhrs_epi16(u_values1, v_g_coeff_2),
        );

        let r_values0 = _mm256_min_epu16(_mm256_max_epi16(r_vals0, zeros), v_max_colors);
        let g_values0 = _mm256_min_epu16(_mm256_max_epi16(g_vals0, zeros), v_max_colors);
        let b_values0 = _mm256_min_epu16(_mm256_max_epi16(b_vals0, zeros), v_max_colors);

        let r_values1 = _mm256_min_epu16(_mm256_max_epi16(r_vals1, zeros), v_max_colors);
        let g_values1 = _mm256_min_epu16(_mm256_max_epi16(g_vals1, zeros), v_max_colors);
        let b_values1 = _mm256_min_epu16(_mm256_max_epi16(b_vals1, zeros), v_max_colors);

        _mm256_store_interleave_rgb16_for_yuv::<DESTINATION_CHANNELS>(
            dst_ptr.as_mut_ptr(),
            r_values0,
            g_values0,
            b_values0,
            v_max_colors,
        );

        _mm256_store_interleave_rgb16_for_yuv::<DESTINATION_CHANNELS>(
            dst_ptr.get_unchecked_mut(16 * channels..).as_mut_ptr(),
            r_values1,
            g_values1,
            b_values1,
            v_max_colors,
        );

        cx += 32;

        match chroma_subsampling {
            YuvChromaSubsampling::Yuv420 | YuvChromaSubsampling::Yuv422 => {
                ux += 16;
            }
            YuvChromaSubsampling::Yuv444 => {
                ux += 32;
            }
        }
    }

    while cx + 16 < width as usize {
        let dst_ptr = dst_ptr.get_unchecked_mut(cx * channels..);

        let mut y_vl = _mm256_loadu_si256(y_plane.get_unchecked(cx..).as_ptr() as *const __m256i);

        if endianness == YuvEndianness::BigEndian {
            y_vl = _mm256_shuffle_epi8(y_vl, big_endian_shuffle_flag);
        }
        if bytes_position == YuvBytesPacking::MostSignificantBytes {
            y_vl = _mm256_from_msb_epi16::<BIT_DEPTH>(y_vl);
        }
        let mut y_values = _mm256_subs_epu16(y_vl, y_corr);

        let mut u_values;
        let mut v_values;

        match chroma_subsampling {
            YuvChromaSubsampling::Yuv420 | YuvChromaSubsampling::Yuv422 => {
                let mut u_vals =
                    _mm_loadu_si128(u_plane.get_unchecked(ux..).as_ptr() as *const __m128i);
                let mut v_vals =
                    _mm_loadu_si128(v_plane.get_unchecked(ux..).as_ptr() as *const __m128i);

                if endianness == YuvEndianness::BigEndian {
                    u_vals = _mm_shuffle_epi8(u_vals, big_endian_shuffle_flag_sse);
                    v_vals = _mm_shuffle_epi8(v_vals, big_endian_shuffle_flag_sse);
                }
                if bytes_position == YuvBytesPacking::MostSignificantBytes {
                    u_vals = _mm_from_msb_epi16::<BIT_DEPTH>(u_vals);
                    v_vals = _mm_from_msb_epi16::<BIT_DEPTH>(v_vals);
                }

                let u_expanded = _mm256_set_m128i(
                    _mm_unpackhi_epi16(u_vals, u_vals),
                    _mm_unpacklo_epi16(u_vals, u_vals),
                ); // [A7, A6, ..., A0 | A7, A6, ..., A0]
                let v_expanded = _mm256_set_m128i(
                    _mm_unpackhi_epi16(v_vals, v_vals),
                    _mm_unpacklo_epi16(v_vals, v_vals),
                ); // [A7, A6, ..., A0 | A7, A6, ..., A0]
                u_values = _mm256_sub_epi16(u_expanded, uv_corr);
                v_values = _mm256_sub_epi16(v_expanded, uv_corr);
            }
            YuvChromaSubsampling::Yuv444 => {
                let mut u_vals =
                    _mm256_loadu_si256(u_plane.get_unchecked(ux..).as_ptr() as *const __m256i);
                let mut v_vals =
                    _mm256_loadu_si256(v_plane.get_unchecked(ux..).as_ptr() as *const __m256i);

                if endianness == YuvEndianness::BigEndian {
                    u_vals = _mm256_shuffle_epi8(u_vals, big_endian_shuffle_flag);
                    v_vals = _mm256_shuffle_epi8(v_vals, big_endian_shuffle_flag);
                }
                if bytes_position == YuvBytesPacking::MostSignificantBytes {
                    u_vals = _mm256_from_msb_epi16::<BIT_DEPTH>(u_vals);
                    v_vals = _mm256_from_msb_epi16::<BIT_DEPTH>(v_vals);
                }
                u_values = _mm256_sub_epi16(u_vals, uv_corr_q);
                v_values = _mm256_sub_epi16(v_vals, uv_corr_q);
            }
        }

        u_values = _mm256_slli_epi16::<SCALE>(u_values);
        v_values = _mm256_slli_epi16::<SCALE>(v_values);
        y_values = _mm256_expand_bp_by2::<BIT_DEPTH>(y_values);

        let y_vals = _mm256_mulhrs_epi16(y_values, v_luma_coeff);

        let r_vals = _mm256_add_epi16(y_vals, _mm256_mulhrs_epi16(v_values, v_cr_coeff));
        let b_vals = _mm256_add_epi16(y_vals, _mm256_mulhrs_epi16(u_values, v_cb_coeff));
        let g_vals = _mm256_add_epi16(
            _mm256_add_epi16(y_vals, _mm256_mulhrs_epi16(v_values, v_g_coeff_1)),
            _mm256_mulhrs_epi16(u_values, v_g_coeff_2),
        );

        let r_values = _mm256_min_epu16(_mm256_max_epi16(r_vals, zeros), v_max_colors);
        let g_values = _mm256_min_epu16(_mm256_max_epi16(g_vals, zeros), v_max_colors);
        let b_values = _mm256_min_epu16(_mm256_max_epi16(b_vals, zeros), v_max_colors);

        _mm256_store_interleave_rgb16_for_yuv::<DESTINATION_CHANNELS>(
            dst_ptr.as_mut_ptr(),
            r_values,
            g_values,
            b_values,
            v_max_colors,
        );

        cx += 16;
        match chroma_subsampling {
            YuvChromaSubsampling::Yuv420 | YuvChromaSubsampling::Yuv422 => {
                ux += 8;
            }
            YuvChromaSubsampling::Yuv444 => {
                ux += 16;
            }
        }
    }

    if cx < width as usize {
        let diff = width as usize - cx;
        assert!(diff <= 16);

        let mut y_buffer: [u16; 16] = [0; 16];
        let mut u_buffer: [u16; 16] = [0; 16];
        let mut v_buffer: [u16; 16] = [0; 16];

        std::ptr::copy_nonoverlapping(
            y_plane.get_unchecked(cx..).as_ptr(),
            y_buffer.as_mut_ptr(),
            diff,
        );

        let dst_ptr = dst_ptr.get_unchecked_mut(cx * channels..);

        let mut y_vl = _mm256_loadu_si256(y_buffer.as_ptr() as *const __m256i);

        if endianness == YuvEndianness::BigEndian {
            y_vl = _mm256_shuffle_epi8(y_vl, big_endian_shuffle_flag);
        }
        if bytes_position == YuvBytesPacking::MostSignificantBytes {
            y_vl = _mm256_from_msb_epi16::<BIT_DEPTH>(y_vl);
        }
        let mut y_values = _mm256_subs_epu16(y_vl, y_corr);

        let mut u_values;
        let mut v_values;

        match chroma_subsampling {
            YuvChromaSubsampling::Yuv420 | YuvChromaSubsampling::Yuv422 => {
                std::ptr::copy_nonoverlapping(
                    u_plane.get_unchecked(ux..).as_ptr(),
                    u_buffer.as_mut_ptr(),
                    diff.div_ceil(2),
                );
                std::ptr::copy_nonoverlapping(
                    v_plane.get_unchecked(ux..).as_ptr(),
                    v_buffer.as_mut_ptr(),
                    diff.div_ceil(2),
                );
                let mut u_vals = _mm_loadu_si128(u_buffer.as_ptr() as *const __m128i);
                let mut v_vals = _mm_loadu_si128(v_buffer.as_ptr() as *const __m128i);

                if endianness == YuvEndianness::BigEndian {
                    u_vals = _mm_shuffle_epi8(u_vals, big_endian_shuffle_flag_sse);
                    v_vals = _mm_shuffle_epi8(v_vals, big_endian_shuffle_flag_sse);
                }
                if bytes_position == YuvBytesPacking::MostSignificantBytes {
                    u_vals = _mm_from_msb_epi16::<BIT_DEPTH>(u_vals);
                    v_vals = _mm_from_msb_epi16::<BIT_DEPTH>(v_vals);
                }

                let u_expanded = _mm256_set_m128i(
                    _mm_unpackhi_epi16(u_vals, u_vals),
                    _mm_unpacklo_epi16(u_vals, u_vals),
                ); // [A7, A6, ..., A0 | A7, A6, ..., A0]
                let v_expanded = _mm256_set_m128i(
                    _mm_unpackhi_epi16(v_vals, v_vals),
                    _mm_unpacklo_epi16(v_vals, v_vals),
                ); // [A7, A6, ..., A0 | A7, A6, ..., A0]
                u_values = _mm256_sub_epi16(u_expanded, uv_corr);
                v_values = _mm256_sub_epi16(v_expanded, uv_corr);
            }
            YuvChromaSubsampling::Yuv444 => {
                std::ptr::copy_nonoverlapping(
                    u_plane.get_unchecked(cx..).as_ptr(),
                    u_buffer.as_mut_ptr(),
                    diff,
                );
                std::ptr::copy_nonoverlapping(
                    v_plane.get_unchecked(cx..).as_ptr(),
                    v_buffer.as_mut_ptr(),
                    diff,
                );
                let mut u_vals = _mm256_loadu_si256(u_buffer.as_ptr() as *const __m256i);
                let mut v_vals = _mm256_loadu_si256(v_buffer.as_ptr() as *const __m256i);

                if endianness == YuvEndianness::BigEndian {
                    u_vals = _mm256_shuffle_epi8(u_vals, big_endian_shuffle_flag);
                    v_vals = _mm256_shuffle_epi8(v_vals, big_endian_shuffle_flag);
                }
                if bytes_position == YuvBytesPacking::MostSignificantBytes {
                    u_vals = _mm256_from_msb_epi16::<BIT_DEPTH>(u_vals);
                    v_vals = _mm256_from_msb_epi16::<BIT_DEPTH>(v_vals);
                }
                u_values = _mm256_sub_epi16(u_vals, uv_corr_q);
                v_values = _mm256_sub_epi16(v_vals, uv_corr_q);
            }
        }

        u_values = _mm256_slli_epi16::<SCALE>(u_values);
        v_values = _mm256_slli_epi16::<SCALE>(v_values);
        y_values = _mm256_expand_bp_by2::<BIT_DEPTH>(y_values);

        let y_vals = _mm256_mulhrs_epi16(y_values, v_luma_coeff);

        let r_vals = _mm256_add_epi16(y_vals, _mm256_mulhrs_epi16(v_values, v_cr_coeff));
        let b_vals = _mm256_add_epi16(y_vals, _mm256_mulhrs_epi16(u_values, v_cb_coeff));
        let g_vals = _mm256_add_epi16(
            _mm256_add_epi16(y_vals, _mm256_mulhrs_epi16(v_values, v_g_coeff_1)),
            _mm256_mulhrs_epi16(u_values, v_g_coeff_2),
        );

        let r_values = _mm256_min_epu16(_mm256_max_epi16(r_vals, zeros), v_max_colors);
        let g_values = _mm256_min_epu16(_mm256_max_epi16(g_vals, zeros), v_max_colors);
        let b_values = _mm256_min_epu16(_mm256_max_epi16(b_vals, zeros), v_max_colors);

        let mut buffer: [u16; 16 * 4] = [0u16; 16 * 4];

        _mm256_store_interleave_rgb16_for_yuv::<DESTINATION_CHANNELS>(
            buffer.as_mut_ptr(),
            r_values,
            g_values,
            b_values,
            v_max_colors,
        );

        std::ptr::copy_nonoverlapping(buffer.as_ptr(), dst_ptr.as_mut_ptr(), diff * channels);

        cx += diff;
        match chroma_subsampling {
            YuvChromaSubsampling::Yuv420 | YuvChromaSubsampling::Yuv422 => {
                ux += diff.div_ceil(2);
            }
            YuvChromaSubsampling::Yuv444 => {
                ux += diff;
            }
        }
    }

    ProcessedOffset { cx, ux }
}
