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

use crate::avx2::avx2_utils::{_mm256_interleave_rgba_epi16, shuffle};
use crate::internals::ProcessedOffset;
use crate::yuv_support::{
    CbCrInverseTransform, YuvBytesPacking, YuvChromaRange, YuvChromaSubsampling, YuvEndianness,
    YuvSourceChannels,
};
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

pub(crate) unsafe fn avx_yuv_p16_to_rgba_alpha_row<
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
    a_plane: &[u16],
    bgra: &mut [u16],
    width: u32,
    range: &YuvChromaRange,
    transform: &CbCrInverseTransform<i32>,
    start_cx: usize,
    start_ux: usize,
) -> ProcessedOffset {
    unsafe {
        avx_yuv_p16_to_rgba_row_alpha_impl::<
            DESTINATION_CHANNELS,
            SAMPLING,
            ENDIANNESS,
            BYTES_POSITION,
            BIT_DEPTH,
            PRECISION,
        >(
            y_plane, u_plane, v_plane, a_plane, bgra, width, range, transform, start_cx, start_ux,
        )
    }
}

#[target_feature(enable = "avx2")]
unsafe fn avx_yuv_p16_to_rgba_row_alpha_impl<
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
    a_plane: &[u16],
    bgra: &mut [u16],
    width: u32,
    range: &YuvChromaRange,
    transform: &CbCrInverseTransform<i32>,
    start_cx: usize,
    start_ux: usize,
) -> ProcessedOffset {
    let destination_channels: YuvSourceChannels = DESTINATION_CHANNELS.into();
    assert!(
        destination_channels == YuvSourceChannels::Rgba
            || destination_channels == YuvSourceChannels::Bgra
    );
    let channels = destination_channels.get_channels_count();
    let chroma_subsampling: YuvChromaSubsampling = SAMPLING.into();
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
    let uv_corr_q = _mm256_set1_epi16((bias_uv as i16) << 1);
    let v_luma_coeff = _mm256_set1_epi16((y_coef as i16) << 1);
    let v_cr_coeff = _mm256_set1_epi16((cr_coef as i16) << 1);
    let v_cb_coeff = _mm256_set1_epi16((cb_coef as i16) << 1);
    let zeros = _mm256_setzero_si256();
    let v_g_coeff_1 = _mm256_set1_epi16(-((g_coef_1 as i16) << 1));
    let v_g_coeff_2 = _mm256_set1_epi16(-((g_coef_2 as i16) << 1));

    let mut cx = start_cx;
    let mut ux = start_ux;

    let v_big_shift_count = _mm_set1_epi64x(16i64 - BIT_DEPTH as i64);

    let big_endian_shuffle_flag = _mm256_setr_epi8(
        1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14, 1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10,
        13, 12, 15, 14,
    );
    let big_endian_shuffle_flag_sse =
        _mm_setr_epi8(1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14);

    while cx + 16 < width as usize {
        let dst_ptr = dst_ptr.get_unchecked_mut(cx * channels..);

        let mut y_vl = _mm256_loadu_si256(y_plane.get_unchecked(cx..).as_ptr() as *const __m256i);
        if endianness == YuvEndianness::BigEndian {
            y_vl = _mm256_shuffle_epi8(y_vl, big_endian_shuffle_flag);
            y_vl = _mm256_permute2x128_si256::<0x01>(y_vl, y_vl);
        }
        if bytes_position == YuvBytesPacking::MostSignificantBytes {
            y_vl = _mm256_srl_epi16(y_vl, v_big_shift_count);
        }
        let mut y_values = _mm256_sub_epi16(y_vl, y_corr);

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
                    u_vals = _mm_srl_epi16(u_vals, v_big_shift_count);
                    v_vals = _mm_srl_epi16(v_vals, v_big_shift_count);
                }

                let u_expanded = _mm256_set_m128i(u_vals, u_vals); // [A7, A6, ..., A0 | A7, A6, ..., A0]
                let v_expanded = _mm256_set_m128i(v_vals, v_vals); // [A7, A6, ..., A0 | A7, A6, ..., A0]
                const MASK: i32 = shuffle(3, 1, 2, 0);
                let u_vl =
                    _mm256_permute4x64_epi64::<MASK>(_mm256_unpacklo_epi16(u_expanded, u_expanded));
                let v_vl =
                    _mm256_permute4x64_epi64::<MASK>(_mm256_unpacklo_epi16(v_expanded, v_expanded));

                u_values = _mm256_sub_epi16(u_vl, uv_corr);
                v_values = _mm256_sub_epi16(v_vl, uv_corr);
            }
            YuvChromaSubsampling::Yuv444 => {
                let mut u_vals =
                    _mm256_loadu_si256(u_plane.get_unchecked(ux..).as_ptr() as *const __m256i);
                let mut v_vals =
                    _mm256_loadu_si256(v_plane.get_unchecked(ux..).as_ptr() as *const __m256i);

                if endianness == YuvEndianness::BigEndian {
                    u_vals = _mm256_shuffle_epi8(u_vals, big_endian_shuffle_flag);
                    v_vals = _mm256_shuffle_epi8(v_vals, big_endian_shuffle_flag);
                    u_vals = _mm256_permute2x128_si256::<0x01>(u_vals, u_vals);
                    v_vals = _mm256_permute2x128_si256::<0x01>(v_vals, v_vals);
                }
                if bytes_position == YuvBytesPacking::MostSignificantBytes {
                    u_vals = _mm256_srl_epi16(u_vals, v_big_shift_count);
                    v_vals = _mm256_srl_epi16(v_vals, v_big_shift_count);
                }
                u_values = _mm256_sub_epi16(u_vals, uv_corr_q);
                v_values = _mm256_sub_epi16(v_vals, uv_corr_q);
            }
        }

        u_values = _mm256_slli_epi16::<3>(u_values);
        v_values = _mm256_slli_epi16::<3>(v_values);
        y_values = _mm256_slli_epi16::<3>(y_values);

        let y_vals = _mm256_mulhi_epi16(y_values, v_luma_coeff);

        let r_vals = _mm256_add_epi16(y_vals, _mm256_mulhi_epi16(v_values, v_cr_coeff));
        let b_vals = _mm256_add_epi16(y_vals, _mm256_mulhi_epi16(u_values, v_cb_coeff));
        let g_vals = _mm256_add_epi16(
            _mm256_add_epi16(y_vals, _mm256_mulhi_epi16(v_values, v_g_coeff_1)),
            _mm256_mulhi_epi16(u_values, v_g_coeff_2),
        );

        let r_values = _mm256_min_epu16(_mm256_max_epi16(r_vals, zeros), v_max_colors);
        let g_values = _mm256_min_epu16(_mm256_max_epi16(g_vals, zeros), v_max_colors);
        let b_values = _mm256_min_epu16(_mm256_max_epi16(b_vals, zeros), v_max_colors);

        let a_values = _mm256_loadu_si256(a_plane.get_unchecked(cx..).as_ptr() as *const __m256i);

        match destination_channels {
            YuvSourceChannels::Rgb | YuvSourceChannels::Bgr => {
                unreachable!("This method can't be called with Rgb/Bgr");
            }
            YuvSourceChannels::Rgba => {
                let dst_pack = _mm256_interleave_rgba_epi16(r_values, g_values, b_values, a_values);
                _mm256_storeu_si256(dst_ptr.as_mut_ptr() as *mut __m256i, dst_pack.0);
                _mm256_storeu_si256(
                    dst_ptr.get_unchecked_mut(16..).as_mut_ptr() as *mut __m256i,
                    dst_pack.1,
                );
                _mm256_storeu_si256(
                    dst_ptr.get_unchecked_mut(32..).as_mut_ptr() as *mut __m256i,
                    dst_pack.2,
                );
                _mm256_storeu_si256(
                    dst_ptr.get_unchecked_mut(48..).as_mut_ptr() as *mut __m256i,
                    dst_pack.3,
                );
            }
            YuvSourceChannels::Bgra => {
                let dst_pack = _mm256_interleave_rgba_epi16(b_values, g_values, r_values, a_values);
                _mm256_storeu_si256(dst_ptr.as_mut_ptr() as *mut __m256i, dst_pack.0);
                _mm256_storeu_si256(
                    dst_ptr.get_unchecked_mut(16..).as_mut_ptr() as *mut __m256i,
                    dst_pack.1,
                );
                _mm256_storeu_si256(
                    dst_ptr.get_unchecked_mut(32..).as_mut_ptr() as *mut __m256i,
                    dst_pack.2,
                );
                _mm256_storeu_si256(
                    dst_ptr.get_unchecked_mut(48..).as_mut_ptr() as *mut __m256i,
                    dst_pack.3,
                );
            }
        }

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

    ProcessedOffset { cx, ux }
}
