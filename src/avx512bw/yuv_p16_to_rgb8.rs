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
#[cfg(feature = "big_endian")]
use crate::avx512bw::avx512_setr::_v512_setr_epu8;
use crate::avx512bw::avx512_utils::{
    _mm256_from_msb_epi16, _mm256_interleave_epi16, _mm512_from_msb_epi16,
    _mm512_store_shr_epi16_epi8, avx512_create, avx512_pack_u16, avx512_store_half_rgba_for_yuv_u8,
    avx512_store_rgba_for_yuv_u8, avx512_zip_epi16,
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

pub(crate) unsafe fn avx512_yuv_p16_to_rgba8_row<
    const DESTINATION_CHANNELS: u8,
    const SAMPLING: u8,
    const ENDIANNESS: u8,
    const BYTES_POSITION: u8,
    const BIT_DEPTH: usize,
    const PRECISION: i32,
    const HAS_VBMI: bool,
>(
    y_plane: &[u16],
    u_plane: &[u16],
    v_plane: &[u16],
    bgra: &mut [u8],
    width: u32,
    range: &YuvChromaRange,
    transform: &CbCrInverseTransform<i32>,
    start_cx: usize,
    start_ux: usize,
) -> ProcessedOffset {
    unsafe {
        if HAS_VBMI {
            avx_yuv_p16_to_rgba_row8_bmi_impl::<
                DESTINATION_CHANNELS,
                SAMPLING,
                ENDIANNESS,
                BYTES_POSITION,
                BIT_DEPTH,
                PRECISION,
            >(
                y_plane, u_plane, v_plane, bgra, width, range, transform, start_cx, start_ux,
            )
        } else {
            avx_yuv_p16_to_rgba_row8_def_impl::<
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
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
unsafe fn avx_yuv_p16_to_rgba_row8_def_impl<
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
    bgra: &mut [u8],
    width: u32,
    range: &YuvChromaRange,
    transform: &CbCrInverseTransform<i32>,
    start_cx: usize,
    start_ux: usize,
) -> ProcessedOffset {
    avx_yuv_p16_to_rgba_row8_impl::<
        DESTINATION_CHANNELS,
        SAMPLING,
        ENDIANNESS,
        BYTES_POSITION,
        BIT_DEPTH,
        PRECISION,
        false,
    >(
        y_plane, u_plane, v_plane, bgra, width, range, transform, start_cx, start_ux,
    )
}

#[target_feature(enable = "avx512f", enable = "avx512bw", enable = "avx512vbmi")]
unsafe fn avx_yuv_p16_to_rgba_row8_bmi_impl<
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
    bgra: &mut [u8],
    width: u32,
    range: &YuvChromaRange,
    transform: &CbCrInverseTransform<i32>,
    start_cx: usize,
    start_ux: usize,
) -> ProcessedOffset {
    avx_yuv_p16_to_rgba_row8_impl::<
        DESTINATION_CHANNELS,
        SAMPLING,
        ENDIANNESS,
        BYTES_POSITION,
        BIT_DEPTH,
        PRECISION,
        true,
    >(
        y_plane, u_plane, v_plane, bgra, width, range, transform, start_cx, start_ux,
    )
}

#[inline(always)]
unsafe fn avx_yuv_p16_to_rgba_row8_impl<
    const DESTINATION_CHANNELS: u8,
    const SAMPLING: u8,
    const ENDIANNESS: u8,
    const BYTES_POSITION: u8,
    const BIT_DEPTH: usize,
    const PRECISION: i32,
    const HAS_VBMI: bool,
>(
    y_plane: &[u16],
    u_plane: &[u16],
    v_plane: &[u16],
    bgra: &mut [u8],
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
    let cr_coef = transform.cr_coef;
    let cb_coef = transform.cb_coef;
    let y_coef = transform.y_coef;
    let g_coef_1 = transform.g_coeff_1;
    let g_coef_2 = transform.g_coeff_2;

    let bias_y = range.bias_y as i32;
    let bias_uv = range.bias_uv as i32;

    let dst_ptr = bgra;

    let v_max_colors = _mm512_set1_epi16((1i16 << BIT_DEPTH as i16) - 1);

    let y_corr = _mm512_set1_epi16(bias_y as i16);
    let uv_corr = _mm512_set1_epi16(bias_uv as i16);
    let uv_corr_q = _mm512_set1_epi16(bias_uv as i16);
    let v_luma_coeff = _mm512_set1_epi16(y_coef as i16);
    let v_cr_coeff = _mm512_set1_epi16(cr_coef as i16);
    let v_cb_coeff = _mm512_set1_epi16(cb_coef as i16);
    let zeros = _mm512_setzero_si512();
    let v_g_coeff_1 = _mm512_set1_epi16(-(g_coef_1 as i16));
    let v_g_coeff_2 = _mm512_set1_epi16(-(g_coef_2 as i16));

    let mut cx = start_cx;
    let mut ux = start_ux;
    #[cfg(feature = "big_endian")]
    let big_endian_shuffle_flag = _v512_setr_epu8(
        1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14, 17, 16, 19, 18, 21, 20, 23, 22, 25,
        24, 27, 26, 29, 28, 31, 30, 33, 32, 35, 34, 37, 36, 39, 38, 41, 40, 43, 42, 45, 44, 47, 46,
        49, 48, 51, 50, 53, 52, 55, 54, 57, 56, 59, 58, 61, 60, 63, 62,
    );
    #[cfg(feature = "big_endian")]
    let big_endian_shuffle_flag_avx = _mm256_setr_epi8(
        1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14, 1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10,
        13, 12, 15, 14,
    );
    const SCALE: u32 = 2;

    while cx + 64 < width as usize {
        let dst_ptr = dst_ptr.get_unchecked_mut(cx * channels..);

        let mut y_vl0 = _mm512_loadu_si512(y_plane.get_unchecked(cx..).as_ptr() as *const i32);
        let mut y_vl1 =
            _mm512_loadu_si512(y_plane.get_unchecked((cx + 32)..).as_ptr() as *const i32);
        #[cfg(feature = "big_endian")]
        if _endianness == YuvEndianness::BigEndian {
            y_vl0 = _mm512_shuffle_epi8(y_vl0, big_endian_shuffle_flag);
            y_vl1 = _mm512_shuffle_epi8(y_vl1, big_endian_shuffle_flag);
        }

        if bytes_position == YuvBytesPacking::MostSignificantBytes {
            y_vl0 = _mm512_from_msb_epi16::<BIT_DEPTH>(y_vl0);
            y_vl1 = _mm512_from_msb_epi16::<BIT_DEPTH>(y_vl1);
        }
        let mut y_values0 = _mm512_subs_epu16(y_vl0, y_corr);
        let mut y_values1 = _mm512_subs_epu16(y_vl1, y_corr);

        let mut u_values0;
        let mut v_values0;

        let mut u_values1;
        let mut v_values1;

        match chroma_subsampling {
            YuvChromaSubsampling::Yuv420 | YuvChromaSubsampling::Yuv422 => {
                let mut u_vals =
                    _mm512_loadu_si512(u_plane.get_unchecked(ux..).as_ptr() as *const i32);
                let mut v_vals =
                    _mm512_loadu_si512(v_plane.get_unchecked(ux..).as_ptr() as *const i32);
                #[cfg(feature = "big_endian")]
                if _endianness == YuvEndianness::BigEndian {
                    u_vals = _mm512_shuffle_epi8(u_vals, big_endian_shuffle_flag);
                    v_vals = _mm512_shuffle_epi8(v_vals, big_endian_shuffle_flag);
                }
                if bytes_position == YuvBytesPacking::MostSignificantBytes {
                    u_vals = _mm512_from_msb_epi16::<BIT_DEPTH>(u_vals);
                    v_vals = _mm512_from_msb_epi16::<BIT_DEPTH>(v_vals);
                }

                (u_values0, u_values1) = avx512_zip_epi16(u_vals, u_vals);
                (v_values0, v_values1) = avx512_zip_epi16(v_vals, v_vals);

                u_values0 = _mm512_sub_epi16(u_values0, uv_corr);
                v_values0 = _mm512_sub_epi16(v_values0, uv_corr);

                u_values1 = _mm512_sub_epi16(u_values1, uv_corr);
                v_values1 = _mm512_sub_epi16(v_values1, uv_corr);
            }
            YuvChromaSubsampling::Yuv444 => {
                let mut u_vals0 =
                    _mm512_loadu_si512(u_plane.get_unchecked(ux..).as_ptr() as *const i32);
                let mut v_vals0 =
                    _mm512_loadu_si512(v_plane.get_unchecked(ux..).as_ptr() as *const i32);
                let mut u_vals1 =
                    _mm512_loadu_si512(u_plane.get_unchecked((ux + 32)..).as_ptr() as *const i32);
                let mut v_vals1 =
                    _mm512_loadu_si512(v_plane.get_unchecked((ux + 32)..).as_ptr() as *const i32);
                #[cfg(feature = "big_endian")]
                if _endianness == YuvEndianness::BigEndian {
                    u_vals0 = _mm512_shuffle_epi8(u_vals0, big_endian_shuffle_flag);
                    v_vals0 = _mm512_shuffle_epi8(v_vals0, big_endian_shuffle_flag);

                    u_vals1 = _mm512_shuffle_epi8(u_vals1, big_endian_shuffle_flag);
                    v_vals1 = _mm512_shuffle_epi8(v_vals1, big_endian_shuffle_flag);
                }
                if bytes_position == YuvBytesPacking::MostSignificantBytes {
                    u_vals0 = _mm512_from_msb_epi16::<BIT_DEPTH>(u_vals0);
                    v_vals0 = _mm512_from_msb_epi16::<BIT_DEPTH>(v_vals0);
                    u_vals1 = _mm512_from_msb_epi16::<BIT_DEPTH>(u_vals1);
                    v_vals1 = _mm512_from_msb_epi16::<BIT_DEPTH>(v_vals1);
                }
                u_values0 = _mm512_sub_epi16(u_vals0, uv_corr);
                v_values0 = _mm512_sub_epi16(v_vals0, uv_corr);

                u_values1 = _mm512_sub_epi16(u_vals1, uv_corr);
                v_values1 = _mm512_sub_epi16(v_vals1, uv_corr);
            }
        }

        u_values0 = _mm512_slli_epi16::<SCALE>(u_values0);
        v_values0 = _mm512_slli_epi16::<SCALE>(v_values0);
        y_values0 = _mm512_slli_epi16::<SCALE>(y_values0);

        u_values1 = _mm512_slli_epi16::<SCALE>(u_values1);
        v_values1 = _mm512_slli_epi16::<SCALE>(v_values1);
        y_values1 = _mm512_slli_epi16::<SCALE>(y_values1);

        let y_vals0 = _mm512_mulhrs_epi16(y_values0, v_luma_coeff);
        let y_vals1 = _mm512_mulhrs_epi16(y_values1, v_luma_coeff);

        let r_vals0 = _mm512_add_epi16(y_vals0, _mm512_mulhrs_epi16(v_values0, v_cr_coeff));
        let b_vals0 = _mm512_add_epi16(y_vals0, _mm512_mulhrs_epi16(u_values0, v_cb_coeff));
        let g_vals0 = _mm512_add_epi16(
            _mm512_add_epi16(y_vals0, _mm512_mulhrs_epi16(v_values0, v_g_coeff_1)),
            _mm512_mulhrs_epi16(u_values0, v_g_coeff_2),
        );

        let r_vals1 = _mm512_add_epi16(y_vals1, _mm512_mulhrs_epi16(v_values1, v_cr_coeff));
        let b_vals1 = _mm512_add_epi16(y_vals1, _mm512_mulhrs_epi16(u_values1, v_cb_coeff));
        let g_vals1 = _mm512_add_epi16(
            _mm512_add_epi16(y_vals1, _mm512_mulhrs_epi16(v_values1, v_g_coeff_1)),
            _mm512_mulhrs_epi16(u_values1, v_g_coeff_2),
        );

        let r_values0 = avx512_pack_u16(
            _mm512_store_shr_epi16_epi8::<BIT_DEPTH>(r_vals0),
            _mm512_store_shr_epi16_epi8::<BIT_DEPTH>(r_vals1),
        );
        let g_values0 = avx512_pack_u16(
            _mm512_store_shr_epi16_epi8::<BIT_DEPTH>(g_vals0),
            _mm512_store_shr_epi16_epi8::<BIT_DEPTH>(g_vals1),
        );
        let b_values0 = avx512_pack_u16(
            _mm512_store_shr_epi16_epi8::<BIT_DEPTH>(b_vals0),
            _mm512_store_shr_epi16_epi8::<BIT_DEPTH>(b_vals1),
        );

        avx512_store_rgba_for_yuv_u8::<DESTINATION_CHANNELS, HAS_VBMI>(
            dst_ptr.as_mut_ptr(),
            r_values0,
            g_values0,
            b_values0,
            v_max_colors,
        );

        cx += 64;

        match chroma_subsampling {
            YuvChromaSubsampling::Yuv420 | YuvChromaSubsampling::Yuv422 => {
                ux += 32;
            }
            YuvChromaSubsampling::Yuv444 => {
                ux += 64;
            }
        }
    }

    while cx + 32 < width as usize {
        let dst_ptr = dst_ptr.get_unchecked_mut(cx * channels..);

        let mut y_vl = _mm512_loadu_si512(y_plane.get_unchecked(cx..).as_ptr() as *const i32);
        #[cfg(feature = "big_endian")]
        if _endianness == YuvEndianness::BigEndian {
            y_vl = _mm512_shuffle_epi8(y_vl, big_endian_shuffle_flag);
        }
        if bytes_position == YuvBytesPacking::MostSignificantBytes {
            y_vl = _mm512_from_msb_epi16::<BIT_DEPTH>(y_vl);
        }
        let mut y_values = _mm512_subs_epu16(y_vl, y_corr);

        let mut u_values;
        let mut v_values;

        match chroma_subsampling {
            YuvChromaSubsampling::Yuv420 | YuvChromaSubsampling::Yuv422 => {
                let mut u_vals =
                    _mm256_loadu_si256(u_plane.get_unchecked(ux..).as_ptr() as *const __m256i);
                let mut v_vals =
                    _mm256_loadu_si256(v_plane.get_unchecked(ux..).as_ptr() as *const __m256i);
                #[cfg(feature = "big_endian")]
                if _endianness == YuvEndianness::BigEndian {
                    u_vals = _mm256_shuffle_epi8(u_vals, big_endian_shuffle_flag_avx);
                    v_vals = _mm256_shuffle_epi8(v_vals, big_endian_shuffle_flag_avx);
                }
                if bytes_position == YuvBytesPacking::MostSignificantBytes {
                    u_vals = _mm256_from_msb_epi16::<BIT_DEPTH>(u_vals);
                    v_vals = _mm256_from_msb_epi16::<BIT_DEPTH>(v_vals);
                }

                let (u_values0, u_values1) = _mm256_interleave_epi16(u_vals, u_vals);
                let (v_values0, v_values1) = _mm256_interleave_epi16(v_vals, v_vals);

                u_values = _mm512_sub_epi16(avx512_create(u_values0, u_values1), uv_corr);
                v_values = _mm512_sub_epi16(avx512_create(v_values0, v_values1), uv_corr);
            }
            YuvChromaSubsampling::Yuv444 => {
                let mut u_vals =
                    _mm512_loadu_si512(u_plane.get_unchecked(ux..).as_ptr() as *const i32);
                let mut v_vals =
                    _mm512_loadu_si512(v_plane.get_unchecked(ux..).as_ptr() as *const i32);
                #[cfg(feature = "big_endian")]
                if _endianness == YuvEndianness::BigEndian {
                    u_vals = _mm512_shuffle_epi8(u_vals, big_endian_shuffle_flag);
                    v_vals = _mm512_shuffle_epi8(v_vals, big_endian_shuffle_flag);
                }
                if bytes_position == YuvBytesPacking::MostSignificantBytes {
                    u_vals = _mm512_from_msb_epi16::<BIT_DEPTH>(u_vals);
                    v_vals = _mm512_from_msb_epi16::<BIT_DEPTH>(v_vals);
                }
                u_values = _mm512_sub_epi16(u_vals, uv_corr_q);
                v_values = _mm512_sub_epi16(v_vals, uv_corr_q);
            }
        }

        u_values = _mm512_slli_epi16::<SCALE>(u_values);
        v_values = _mm512_slli_epi16::<SCALE>(v_values);
        y_values = _mm512_slli_epi16::<SCALE>(y_values);

        let y_vals = _mm512_mulhrs_epi16(y_values, v_luma_coeff);

        let r_vals = _mm512_add_epi16(y_vals, _mm512_mulhrs_epi16(v_values, v_cr_coeff));
        let b_vals = _mm512_add_epi16(y_vals, _mm512_mulhrs_epi16(u_values, v_cb_coeff));
        let g_vals = _mm512_add_epi16(
            _mm512_add_epi16(y_vals, _mm512_mulhrs_epi16(v_values, v_g_coeff_1)),
            _mm512_mulhrs_epi16(u_values, v_g_coeff_2),
        );

        let r_values = avx512_pack_u16(_mm512_store_shr_epi16_epi8::<BIT_DEPTH>(r_vals), zeros);
        let g_values = avx512_pack_u16(_mm512_store_shr_epi16_epi8::<BIT_DEPTH>(g_vals), zeros);
        let b_values = avx512_pack_u16(_mm512_store_shr_epi16_epi8::<BIT_DEPTH>(b_vals), zeros);

        avx512_store_half_rgba_for_yuv_u8::<DESTINATION_CHANNELS, HAS_VBMI>(
            dst_ptr.as_mut_ptr(),
            r_values,
            g_values,
            b_values,
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

    ProcessedOffset { cx, ux }
}
