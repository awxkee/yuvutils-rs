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
    _mm256_affine_transform, _mm256_affine_uv_dot, _mm256_havg_epi16_epi32,
    _mm256_load_deinterleave_rgb16_for_yuv, _mm256_to_msb_epi16,
};
use crate::internals::ProcessedOffset;
use crate::yuv_support::{
    to_channels_layout, to_subsampling, CbCrForwardTransform, YuvChromaRange, YuvChromaSubsampling,
    YuvSourceChannels,
};
use crate::{YuvBytesPacking, YuvEndianness};
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

pub(crate) fn avx_rgba_to_yuv_p16<
    const ORIGIN_CHANNELS: u8,
    const SAMPLING: u8,
    const ENDIANNESS: u8,
    const BYTES_POSITION: u8,
    const PRECISION: i32,
    const BIT_DEPTH: usize,
>(
    transform: &CbCrForwardTransform<i32>,
    range: &YuvChromaRange,
    y_plane: &mut [u16],
    u_plane: &mut [u16],
    v_plane: &mut [u16],
    rgba: &[u16],
    start_cx: usize,
    start_ux: usize,
    width: usize,
) -> ProcessedOffset {
    unsafe {
        #[cfg(feature = "nightly_avx512")]
        if std::arch::is_x86_feature_detected!("avxvnni") {
            return avx_rgba_to_yuv_vnni::<
                ORIGIN_CHANNELS,
                SAMPLING,
                ENDIANNESS,
                BYTES_POSITION,
                PRECISION,
                BIT_DEPTH,
            >(
                transform, range, y_plane, u_plane, v_plane, rgba, start_cx, start_ux, width,
            );
        }
        avx_rgba_to_yuv_reg::<
            ORIGIN_CHANNELS,
            SAMPLING,
            ENDIANNESS,
            BYTES_POSITION,
            PRECISION,
            BIT_DEPTH,
        >(
            transform, range, y_plane, u_plane, v_plane, rgba, start_cx, start_ux, width,
        )
    }
}

#[target_feature(enable = "avx2")]
unsafe fn avx_rgba_to_yuv_reg<
    const ORIGIN_CHANNELS: u8,
    const SAMPLING: u8,
    const ENDIANNESS: u8,
    const BYTES_POSITION: u8,
    const PRECISION: i32,
    const BIT_DEPTH: usize,
>(
    transform: &CbCrForwardTransform<i32>,
    range: &YuvChromaRange,
    y_plane: &mut [u16],
    u_plane: &mut [u16],
    v_plane: &mut [u16],
    rgba: &[u16],
    start_cx: usize,
    start_ux: usize,
    width: usize,
) -> ProcessedOffset {
    avx_rgba_to_yuv_impl::<
        ORIGIN_CHANNELS,
        SAMPLING,
        ENDIANNESS,
        BYTES_POSITION,
        PRECISION,
        BIT_DEPTH,
        false,
    >(
        transform, range, y_plane, u_plane, v_plane, rgba, start_cx, start_ux, width,
    )
}

#[cfg(feature = "nightly_avx512")]
#[target_feature(enable = "avx2", enable = "avxvnni")]
unsafe fn avx_rgba_to_yuv_vnni<
    const ORIGIN_CHANNELS: u8,
    const SAMPLING: u8,
    const ENDIANNESS: u8,
    const BYTES_POSITION: u8,
    const PRECISION: i32,
    const BIT_DEPTH: usize,
>(
    transform: &CbCrForwardTransform<i32>,
    range: &YuvChromaRange,
    y_plane: &mut [u16],
    u_plane: &mut [u16],
    v_plane: &mut [u16],
    rgba: &[u16],
    start_cx: usize,
    start_ux: usize,
    width: usize,
) -> ProcessedOffset {
    avx_rgba_to_yuv_impl::<
        ORIGIN_CHANNELS,
        SAMPLING,
        ENDIANNESS,
        BYTES_POSITION,
        PRECISION,
        BIT_DEPTH,
        true,
    >(
        transform, range, y_plane, u_plane, v_plane, rgba, start_cx, start_ux, width,
    )
}

#[inline(always)]
unsafe fn avx_rgba_to_yuv_impl<
    const ORIGIN_CHANNELS: u8,
    const SAMPLING: u8,
    const ENDIANNESS: u8,
    const BYTES_POSITION: u8,
    const PRECISION: i32,
    const BIT_DEPTH: usize,
    const HAS_DOT: bool,
>(
    transform: &CbCrForwardTransform<i32>,
    range: &YuvChromaRange,
    y_plane: &mut [u16],
    u_plane: &mut [u16],
    v_plane: &mut [u16],
    rgba: &[u16],
    start_cx: usize,
    start_ux: usize,
    width: usize,
) -> ProcessedOffset {
    let chroma_subsampling: YuvChromaSubsampling = to_subsampling(SAMPLING);
    let source_channels: YuvSourceChannels = to_channels_layout(ORIGIN_CHANNELS);
    let endianness: YuvEndianness = ENDIANNESS.into();
    let bytes_position: YuvBytesPacking = BYTES_POSITION.into();
    let channels = source_channels.get_channels_count();

    let rounding_const_bias: i32 = (1 << (PRECISION - 1)) - 1;
    let bias_y = range.bias_y as i32 * (1 << PRECISION) + rounding_const_bias;
    let bias_uv = range.bias_uv as i32 * (1 << PRECISION) + rounding_const_bias;

    let src_ptr = rgba;

    let y_ptr = y_plane;
    let u_ptr = u_plane;
    let v_ptr = v_plane;

    let y_bias = _mm256_set1_epi32(bias_y);
    let uv_bias = _mm256_set1_epi32(bias_uv);
    let v_yr_yg = _mm256_set1_epi32(transform._interleaved_yr_yg());
    let v_yb = _mm256_set1_epi16(transform.yb as i16);
    let v_cbr_cbg = _mm256_set1_epi32(transform._interleaved_cbr_cbg());
    let v_cb_b = _mm256_set1_epi16(transform.cb_b as i16);
    let v_crr_vcrg = _mm256_set1_epi32(transform._interleaved_crr_crg());
    let v_cr_b = _mm256_set1_epi16(transform.cr_b as i16);

    let big_endian_shuffle_flag = _mm256_setr_epi8(
        1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14, 1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10,
        13, 12, 15, 14,
    );

    let mut cx = start_cx;
    let mut ux = start_ux;

    while cx + 16 < width {
        let src_ptr = src_ptr.get_unchecked(cx * channels..);
        let (r_values, g_values, b_values) =
            _mm256_load_deinterleave_rgb16_for_yuv::<ORIGIN_CHANNELS>(src_ptr.as_ptr());

        let zeros = _mm256_setzero_si256();
        let (r_g_lo, r_g_hi) = (
            _mm256_unpacklo_epi16(r_values, g_values),
            _mm256_unpackhi_epi16(r_values, g_values),
        );
        let b_hi = _mm256_unpackhi_epi16(b_values, zeros);
        let b_lo = _mm256_unpacklo_epi16(b_values, zeros);

        let mut y_vl = _mm256_affine_uv_dot::<PRECISION, HAS_DOT>(
            y_bias, r_g_lo, r_g_hi, b_lo, b_hi, v_yr_yg, v_yb,
        );

        if bytes_position == YuvBytesPacking::MostSignificantBytes {
            y_vl = _mm256_to_msb_epi16::<BIT_DEPTH>(y_vl);
        }

        if endianness == YuvEndianness::BigEndian {
            y_vl = _mm256_shuffle_epi8(y_vl, big_endian_shuffle_flag);
        }

        _mm256_storeu_si256(
            y_ptr.get_unchecked_mut(cx..).as_mut_ptr() as *mut __m256i,
            y_vl,
        );

        if chroma_subsampling == YuvChromaSubsampling::Yuv444 {
            let mut cb_vl = _mm256_affine_uv_dot::<PRECISION, HAS_DOT>(
                uv_bias, r_g_lo, r_g_hi, b_lo, b_hi, v_cbr_cbg, v_cb_b,
            );

            let mut cr_vl = _mm256_affine_uv_dot::<PRECISION, HAS_DOT>(
                uv_bias, r_g_lo, r_g_hi, b_lo, b_hi, v_crr_vcrg, v_cr_b,
            );

            if bytes_position == YuvBytesPacking::MostSignificantBytes {
                cb_vl = _mm256_to_msb_epi16::<BIT_DEPTH>(cb_vl);
                cr_vl = _mm256_to_msb_epi16::<BIT_DEPTH>(cr_vl);
            }

            if endianness == YuvEndianness::BigEndian {
                cb_vl = _mm256_shuffle_epi8(cb_vl, big_endian_shuffle_flag);
                cr_vl = _mm256_shuffle_epi8(cr_vl, big_endian_shuffle_flag);
            }

            _mm256_storeu_si256(
                u_ptr.get_unchecked_mut(ux..).as_mut_ptr() as *mut __m256i,
                cb_vl,
            );
            _mm256_storeu_si256(
                v_ptr.get_unchecked_mut(ux..).as_mut_ptr() as *mut __m256i,
                cr_vl,
            );

            ux += 16;
        } else {
            let r_values = _mm256_havg_epi16_epi32(r_values);
            let g_values = _mm256_havg_epi16_epi32(g_values);
            let b_values = _mm256_havg_epi16_epi32(b_values);

            let r_g_values = _mm256_or_si256(r_values, _mm256_slli_epi32::<16>(g_values));

            let mut cb_s = _mm256_affine_transform::<PRECISION, HAS_DOT>(
                uv_bias, r_g_values, b_values, v_cbr_cbg, v_cb_b,
            );

            let mut cr_s = _mm256_affine_transform::<PRECISION, HAS_DOT>(
                uv_bias, r_g_values, b_values, v_crr_vcrg, v_cr_b,
            );

            if bytes_position == YuvBytesPacking::MostSignificantBytes {
                cb_s = _mm256_to_msb_epi16::<BIT_DEPTH>(cb_s);
                cr_s = _mm256_to_msb_epi16::<BIT_DEPTH>(cr_s);
            }

            if endianness == YuvEndianness::BigEndian {
                cb_s = _mm256_shuffle_epi8(cb_s, big_endian_shuffle_flag);
                cr_s = _mm256_shuffle_epi8(cr_s, big_endian_shuffle_flag);
            }

            _mm_storeu_si128(
                u_ptr.get_unchecked_mut(ux..).as_mut_ptr() as *mut __m128i,
                _mm256_castsi256_si128(cb_s),
            );
            _mm_storeu_si128(
                v_ptr.get_unchecked_mut(ux..).as_mut_ptr() as *mut __m128i,
                _mm256_castsi256_si128(cr_s),
            );

            ux += 8;
        }

        cx += 16;
    }

    ProcessedOffset { ux, cx }
}
