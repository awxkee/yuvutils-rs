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
    _avx_from_msb_epi16, _mm256_from_msb_epi16, _mm256_interleave_epi16, _mm256_mul_add_epi16,
    _mm256_store_interleave_rgb16_for_yuv, shuffle,
};
use crate::internals::ProcessedOffset;
use crate::yuv_support::{
    CbCrInverseTransform, YuvBytesPacking, YuvChromaRange, YuvChromaSubsampling, YuvEndianness,
    YuvSourceChannels,
};
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
use std::mem::MaybeUninit;

pub(crate) unsafe fn avx_yuv_p16_to_rgba_d16_row<
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
        avx_yuv_p16_to_rgba_row_d16_impl::<
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
unsafe fn avx_yuv_p16_to_rgba_row_d16_impl<
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
    let destination_channels: YuvSourceChannels = DESTINATION_CHANNELS.into();
    let channels = destination_channels.get_channels_count();
    let chroma_subsampling: YuvChromaSubsampling = SAMPLING.into();
    let _endianness: YuvEndianness = ENDIANNESS.into();
    let bytes_position: YuvBytesPacking = BYTES_POSITION.into();
    let bias_y = range.bias_y as i32;
    let bias_uv = range.bias_uv as i32;

    let dst_ptr = bgra;

    const D: bool = false;

    let v_max_colors = _mm256_set1_epi16(((1i32 << BIT_DEPTH as i32) - 1i32) as i16);
    let v_luma_coeff = if BIT_DEPTH == 16 {
        _mm256_set1_epi32(transform.y_coef)
    } else {
        _mm256_set1_epi16(transform.y_coef as i16)
    };
    let cr_c = transform.cr_coef.to_ne_bytes();
    let v_cr_coeff = _mm256_set1_epi32(i32::from_ne_bytes([0, 0, cr_c[0], cr_c[1]]));
    let v_cb_part = (transform.cb_coef as u32).to_ne_bytes();
    let v_cb_coeff = _mm256_set1_epi32(i32::from_ne_bytes([v_cb_part[0], v_cb_part[1], 0, 0]));
    let g_trn1 = (-transform.g_coeff_1).to_ne_bytes();
    let g_trn2 = (-transform.g_coeff_2).to_ne_bytes();
    let v_g_coeff_1 = _mm256_set1_epi32(i32::from_ne_bytes([
        g_trn2[0], g_trn2[1], g_trn1[0], g_trn1[1],
    ]));

    let base_value = _mm256_set1_epi32((1 << (PRECISION - 1)) - 1);

    let mut cx = start_cx;
    let mut ux = start_ux;

    #[cfg(feature = "big_endian")]
    let big_endian_shuffle_flag = _mm256_setr_epi8(
        1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14, 1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10,
        13, 12, 15, 14,
    );

    #[cfg(feature = "big_endian")]
    let big_endian_shuffle_flag_sse =
        _mm_setr_epi8(1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14);

    while cx + 16 < width as usize {
        let dst_ptr = dst_ptr.get_unchecked_mut(cx * channels..);

        let mut y_vl = _mm256_loadu_si256(y_plane.get_unchecked(cx..).as_ptr() as *const __m256i);

        #[cfg(feature = "big_endian")]
        if _endianness == YuvEndianness::BigEndian {
            y_vl = _mm256_shuffle_epi8(y_vl, big_endian_shuffle_flag);
        }

        if bytes_position == YuvBytesPacking::MostSignificantBytes {
            y_vl = _mm256_from_msb_epi16::<BIT_DEPTH>(y_vl);
        }

        let y_corr = _mm256_set1_epi16(bias_y as i16);
        let uv_corr = _mm256_set1_epi16(bias_uv as i16);

        let y_values = _mm256_subs_epu16(y_vl, y_corr);

        let mut u_values;
        let mut v_values;

        match chroma_subsampling {
            YuvChromaSubsampling::Yuv420 | YuvChromaSubsampling::Yuv422 => {
                let mut u_vals =
                    _mm_loadu_si128(u_plane.get_unchecked(ux..).as_ptr() as *const __m128i);
                let mut v_vals =
                    _mm_loadu_si128(v_plane.get_unchecked(ux..).as_ptr() as *const __m128i);

                #[cfg(feature = "big_endian")]
                if _endianness == YuvEndianness::BigEndian {
                    u_vals = _mm_shuffle_epi8(u_vals, big_endian_shuffle_flag_sse);
                    v_vals = _mm_shuffle_epi8(v_vals, big_endian_shuffle_flag_sse);
                }
                if bytes_position == YuvBytesPacking::MostSignificantBytes {
                    u_vals = _avx_from_msb_epi16::<BIT_DEPTH>(u_vals);
                    v_vals = _avx_from_msb_epi16::<BIT_DEPTH>(v_vals);
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

                #[cfg(feature = "big_endian")]
                if _endianness == YuvEndianness::BigEndian {
                    u_vals = _mm256_shuffle_epi8(u_vals, big_endian_shuffle_flag);
                    v_vals = _mm256_shuffle_epi8(v_vals, big_endian_shuffle_flag);
                }

                if bytes_position == YuvBytesPacking::MostSignificantBytes {
                    u_vals = _mm256_from_msb_epi16::<BIT_DEPTH>(u_vals);
                    v_vals = _mm256_from_msb_epi16::<BIT_DEPTH>(v_vals);
                }
                u_values = _mm256_sub_epi16(u_vals, uv_corr);
                v_values = _mm256_sub_epi16(v_vals, uv_corr);
            }
        }

        const MASK: i32 = shuffle(3, 1, 2, 0);
        u_values = _mm256_permute4x64_epi64::<MASK>(u_values);
        v_values = _mm256_permute4x64_epi64::<MASK>(v_values);

        let (uv_values0, uv_values1) = _mm256_interleave_epi16(u_values, v_values);

        let mut y_0 = _mm256_unpacklo_epi16(y_values, _mm256_setzero_si256());
        let mut y_1 = _mm256_unpackhi_epi16(y_values, _mm256_setzero_si256());

        y_0 = if BIT_DEPTH == 16 {
            _mm256_add_epi32(base_value, _mm256_mullo_epi32(y_0, v_luma_coeff))
        } else {
            _mm256_mul_add_epi16::<D>(base_value, y_0, v_luma_coeff)
        };
        y_1 = if BIT_DEPTH == 16 {
            _mm256_add_epi32(base_value, _mm256_mullo_epi32(y_1, v_luma_coeff))
        } else {
            _mm256_mul_add_epi16::<D>(base_value, y_1, v_luma_coeff)
        };

        let mut g_0 = _mm256_mul_add_epi16::<D>(y_0, uv_values0, v_g_coeff_1);
        let mut g_1 = _mm256_mul_add_epi16::<D>(y_1, uv_values1, v_g_coeff_1);

        let mut r_0 = _mm256_mul_add_epi16::<D>(y_0, uv_values0, v_cr_coeff);
        let mut r_1 = _mm256_mul_add_epi16::<D>(y_1, uv_values1, v_cr_coeff);

        let mut b_0 = _mm256_mul_add_epi16::<D>(y_0, uv_values0, v_cb_coeff);
        let mut b_1 = _mm256_mul_add_epi16::<D>(y_1, uv_values1, v_cb_coeff);

        g_0 = _mm256_srai_epi32::<PRECISION>(g_0);
        g_1 = _mm256_srai_epi32::<PRECISION>(g_1);

        r_0 = _mm256_srai_epi32::<PRECISION>(r_0);
        r_1 = _mm256_srai_epi32::<PRECISION>(r_1);

        b_0 = _mm256_srai_epi32::<PRECISION>(b_0);
        b_1 = _mm256_srai_epi32::<PRECISION>(b_1);

        let rm = _mm256_packus_epi32(r_0, r_1);
        let gm = _mm256_packus_epi32(g_0, g_1);
        let bm = _mm256_packus_epi32(b_0, b_1);

        let r_values = if BIT_DEPTH == 16 {
            rm
        } else {
            _mm256_min_epu16(rm, v_max_colors)
        };
        let g_values = if BIT_DEPTH == 16 {
            gm
        } else {
            _mm256_min_epu16(gm, v_max_colors)
        };
        let b_values = if BIT_DEPTH == 16 {
            bm
        } else {
            _mm256_min_epu16(bm, v_max_colors)
        };

        _mm256_store_interleave_rgb16_for_yuv::<DESTINATION_CHANNELS>(
            dst_ptr.as_mut_ptr() as *mut _,
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

        let mut y_buffer: [MaybeUninit<u16>; 16] = [MaybeUninit::uninit(); 16];
        let mut u_buffer: [MaybeUninit<u16>; 16] = [MaybeUninit::uninit(); 16];
        let mut v_buffer: [MaybeUninit<u16>; 16] = [MaybeUninit::uninit(); 16];
        let mut buffer: [MaybeUninit<u16>; 16 * 4] = [MaybeUninit::uninit(); 16 * 4];

        std::ptr::copy_nonoverlapping(
            y_plane.get_unchecked(cx..).as_ptr(),
            y_buffer.as_mut_ptr().cast(),
            diff,
        );

        let dst_ptr = dst_ptr.get_unchecked_mut(cx * channels..);

        let mut y_vl = _mm256_loadu_si256(y_buffer.as_ptr() as *const __m256i);

        #[cfg(feature = "big_endian")]
        if _endianness == YuvEndianness::BigEndian {
            y_vl = _mm256_shuffle_epi8(y_vl, big_endian_shuffle_flag);
        }

        if bytes_position == YuvBytesPacking::MostSignificantBytes {
            y_vl = _mm256_from_msb_epi16::<BIT_DEPTH>(y_vl);
        }

        let y_corr = _mm256_set1_epi16(bias_y as i16);
        let uv_corr = _mm256_set1_epi16(bias_uv as i16);

        let y_values = _mm256_subs_epu16(y_vl, y_corr);

        let mut u_values;
        let mut v_values;

        match chroma_subsampling {
            YuvChromaSubsampling::Yuv420 | YuvChromaSubsampling::Yuv422 => {
                std::ptr::copy_nonoverlapping(
                    u_plane.get_unchecked(ux..).as_ptr(),
                    u_buffer.as_mut_ptr().cast(),
                    diff.div_ceil(2),
                );
                std::ptr::copy_nonoverlapping(
                    v_plane.get_unchecked(ux..).as_ptr(),
                    v_buffer.as_mut_ptr().cast(),
                    diff.div_ceil(2),
                );
                let mut u_vals = _mm_loadu_si128(u_buffer.as_ptr() as *const __m128i);
                let mut v_vals = _mm_loadu_si128(v_buffer.as_ptr() as *const __m128i);

                #[cfg(feature = "big_endian")]
                if _endianness == YuvEndianness::BigEndian {
                    u_vals = _mm_shuffle_epi8(u_vals, big_endian_shuffle_flag_sse);
                    v_vals = _mm_shuffle_epi8(v_vals, big_endian_shuffle_flag_sse);
                }

                if bytes_position == YuvBytesPacking::MostSignificantBytes {
                    u_vals = _avx_from_msb_epi16::<BIT_DEPTH>(u_vals);
                    v_vals = _avx_from_msb_epi16::<BIT_DEPTH>(v_vals);
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
                    u_buffer.as_mut_ptr().cast(),
                    diff,
                );
                std::ptr::copy_nonoverlapping(
                    v_plane.get_unchecked(cx..).as_ptr(),
                    v_buffer.as_mut_ptr().cast(),
                    diff,
                );
                let mut u_vals = _mm256_loadu_si256(u_buffer.as_ptr() as *const __m256i);
                let mut v_vals = _mm256_loadu_si256(v_buffer.as_ptr() as *const __m256i);

                #[cfg(feature = "big_endian")]
                if _endianness == YuvEndianness::BigEndian {
                    u_vals = _mm256_shuffle_epi8(u_vals, big_endian_shuffle_flag);
                    v_vals = _mm256_shuffle_epi8(v_vals, big_endian_shuffle_flag);
                }

                if bytes_position == YuvBytesPacking::MostSignificantBytes {
                    u_vals = _mm256_from_msb_epi16::<BIT_DEPTH>(u_vals);
                    v_vals = _mm256_from_msb_epi16::<BIT_DEPTH>(v_vals);
                }
                u_values = _mm256_sub_epi16(u_vals, uv_corr);
                v_values = _mm256_sub_epi16(v_vals, uv_corr);
            }
        }

        const MASK: i32 = shuffle(3, 1, 2, 0);
        u_values = _mm256_permute4x64_epi64::<MASK>(u_values);
        v_values = _mm256_permute4x64_epi64::<MASK>(v_values);

        let (uv_values0, uv_values1) = _mm256_interleave_epi16(u_values, v_values);

        let mut y_0 = _mm256_unpacklo_epi16(y_values, _mm256_setzero_si256());
        let mut y_1 = _mm256_unpackhi_epi16(y_values, _mm256_setzero_si256());

        y_0 = if BIT_DEPTH == 16 {
            _mm256_add_epi32(base_value, _mm256_mullo_epi32(y_0, v_luma_coeff))
        } else {
            _mm256_mul_add_epi16::<D>(base_value, y_0, v_luma_coeff)
        };
        y_1 = if BIT_DEPTH == 16 {
            _mm256_add_epi32(base_value, _mm256_mullo_epi32(y_1, v_luma_coeff))
        } else {
            _mm256_mul_add_epi16::<D>(base_value, y_1, v_luma_coeff)
        };

        let mut g_0 = _mm256_mul_add_epi16::<D>(y_0, uv_values0, v_g_coeff_1);
        let mut g_1 = _mm256_mul_add_epi16::<D>(y_1, uv_values1, v_g_coeff_1);

        let mut r_0 = _mm256_mul_add_epi16::<D>(y_0, uv_values0, v_cr_coeff);
        let mut r_1 = _mm256_mul_add_epi16::<D>(y_1, uv_values1, v_cr_coeff);

        let mut b_0 = _mm256_mul_add_epi16::<D>(y_0, uv_values0, v_cb_coeff);
        let mut b_1 = _mm256_mul_add_epi16::<D>(y_1, uv_values1, v_cb_coeff);

        g_0 = _mm256_srai_epi32::<PRECISION>(g_0);
        g_1 = _mm256_srai_epi32::<PRECISION>(g_1);

        r_0 = _mm256_srai_epi32::<PRECISION>(r_0);
        r_1 = _mm256_srai_epi32::<PRECISION>(r_1);

        b_0 = _mm256_srai_epi32::<PRECISION>(b_0);
        b_1 = _mm256_srai_epi32::<PRECISION>(b_1);

        let rm = _mm256_packus_epi32(r_0, r_1);
        let gm = _mm256_packus_epi32(g_0, g_1);
        let bm = _mm256_packus_epi32(b_0, b_1);

        let r_values = if BIT_DEPTH == 16 {
            rm
        } else {
            _mm256_min_epu16(rm, v_max_colors)
        };
        let g_values = if BIT_DEPTH == 16 {
            gm
        } else {
            _mm256_min_epu16(gm, v_max_colors)
        };
        let b_values = if BIT_DEPTH == 16 {
            bm
        } else {
            _mm256_min_epu16(bm, v_max_colors)
        };

        _mm256_store_interleave_rgb16_for_yuv::<DESTINATION_CHANNELS>(
            buffer.as_mut_ptr() as *mut _,
            r_values,
            g_values,
            b_values,
            v_max_colors,
        );

        std::ptr::copy_nonoverlapping(
            buffer.as_ptr().cast(),
            dst_ptr.as_mut_ptr(),
            diff * channels,
        );

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
