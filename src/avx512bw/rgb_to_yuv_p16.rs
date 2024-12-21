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
use crate::avx512bw::avx512_setr::_v512_setr_epu8;
use crate::avx512bw::avx512_utils::{
    _mm512_load_deinterleave_rgb16_for_yuv, _mm512_to_msb_epi16, avx512_avg_epi16,
};
use crate::internals::ProcessedOffset;
use crate::yuv_support::{
    CbCrForwardTransform, YuvChromaRange, YuvChromaSubsampling, YuvSourceChannels,
};
use crate::{YuvBytesPacking, YuvEndianness};
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

// Slightly lower precision option, suitable only for bit-depth <= 12
pub(crate) fn avx512_rgba_to_yuv_p16_lp<
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
        avx_rgba_to_yuv_impl_lp::<
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

#[target_feature(enable = "avx512f", enable = "avx512bw")]
unsafe fn avx_rgba_to_yuv_impl_lp<
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
    let chroma_subsampling: YuvChromaSubsampling = SAMPLING.into();
    let source_channels: YuvSourceChannels = ORIGIN_CHANNELS.into();
    let endianness: YuvEndianness = ENDIANNESS.into();
    let bytes_position: YuvBytesPacking = BYTES_POSITION.into();
    let channels = source_channels.get_channels_count();

    let bias_y = range.bias_y as i16;
    let bias_uv = range.bias_uv as i16;

    let src_ptr = rgba;

    let y_ptr = y_plane;
    let u_ptr = u_plane;
    let v_ptr = v_plane;

    let y_bias = _mm512_set1_epi16(bias_y);
    let uv_bias = _mm512_set1_epi16(bias_uv);
    let v_yr = _mm512_set1_epi16(transform.yr as i16);
    let v_yg = _mm512_set1_epi16(transform.yg as i16);
    let v_yb = _mm512_set1_epi16(transform.yb as i16);
    let v_cb_r = _mm512_set1_epi16(transform.cb_r as i16);
    let v_cb_g = _mm512_set1_epi16(transform.cb_g as i16);
    let v_cb_b = _mm512_set1_epi16(transform.cb_b as i16);
    let v_cr_r = _mm512_set1_epi16(transform.cr_r as i16);
    let v_cr_g = _mm512_set1_epi16(transform.cr_g as i16);
    let v_cr_b = _mm512_set1_epi16(transform.cr_b as i16);

    let big_endian_shuffle_flag = _v512_setr_epu8(
        1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14, 17, 16, 19, 18, 21, 20, 23, 22, 25,
        24, 27, 26, 29, 28, 31, 30, 33, 32, 35, 34, 37, 36, 39, 38, 41, 40, 43, 42, 45, 44, 47, 46,
        49, 48, 51, 50, 53, 52, 55, 54, 57, 56, 59, 58, 61, 60, 63, 62,
    );

    let mut cx = start_cx;
    let mut ux = start_ux;

    let i_cap_y = _mm512_set1_epi16((range.range_y as u16 + range.bias_y as u16) as i16);
    let i_cap_uv = _mm512_set1_epi16((range.bias_y as u16 + range.range_uv as u16) as i16);

    const SCALE: u32 = 2;

    while cx + 32 < width {
        let src_ptr = src_ptr.get_unchecked(cx * channels..);
        let (mut r_values, mut g_values, mut b_values) =
            _mm512_load_deinterleave_rgb16_for_yuv::<ORIGIN_CHANNELS>(src_ptr.as_ptr());

        r_values = _mm512_slli_epi16::<SCALE>(r_values);
        g_values = _mm512_slli_epi16::<SCALE>(g_values);
        b_values = _mm512_slli_epi16::<SCALE>(b_values);

        let mut y_h = _mm512_add_epi16(y_bias, _mm512_mulhrs_epi16(r_values, v_yr));
        y_h = _mm512_add_epi16(y_h, _mm512_mulhrs_epi16(g_values, v_yg));
        y_h = _mm512_add_epi16(y_h, _mm512_mulhrs_epi16(b_values, v_yb));

        let mut y_vl = _mm512_min_epu16(y_h, i_cap_y);

        if bytes_position == YuvBytesPacking::MostSignificantBytes {
            y_vl = _mm512_to_msb_epi16::<BIT_DEPTH>(y_vl);
        }

        if endianness == YuvEndianness::BigEndian {
            y_vl = _mm512_shuffle_epi8(y_vl, big_endian_shuffle_flag);
        }

        _mm512_storeu_si512(y_ptr.get_unchecked_mut(cx..).as_mut_ptr() as *mut i32, y_vl);

        if chroma_subsampling == YuvChromaSubsampling::Yuv444 {
            let mut cb_h = _mm512_add_epi16(uv_bias, _mm512_mulhrs_epi16(r_values, v_cb_r));
            cb_h = _mm512_add_epi16(cb_h, _mm512_mulhrs_epi16(g_values, v_cb_g));
            cb_h = _mm512_add_epi16(cb_h, _mm512_mulhrs_epi16(b_values, v_cb_b));

            let mut cb_vl = _mm512_max_epu16(_mm512_min_epu16(cb_h, i_cap_uv), y_bias);

            let mut cr_h = _mm512_add_epi16(uv_bias, _mm512_mulhrs_epi16(r_values, v_cr_r));
            cr_h = _mm512_add_epi16(cr_h, _mm512_mulhrs_epi16(g_values, v_cr_g));
            cr_h = _mm512_add_epi16(cr_h, _mm512_mulhrs_epi16(b_values, v_cr_b));

            let mut cr_vl = _mm512_max_epu16(_mm512_min_epu16(cr_h, i_cap_uv), y_bias);

            if bytes_position == YuvBytesPacking::MostSignificantBytes {
                cb_vl = _mm512_to_msb_epi16::<BIT_DEPTH>(cb_vl);
                cr_vl = _mm512_to_msb_epi16::<BIT_DEPTH>(cr_vl);
            }

            if endianness == YuvEndianness::BigEndian {
                cb_vl = _mm512_shuffle_epi8(cb_vl, big_endian_shuffle_flag);
                cr_vl = _mm512_shuffle_epi8(cr_vl, big_endian_shuffle_flag);
            }

            _mm512_storeu_si512(
                u_ptr.get_unchecked_mut(ux..).as_mut_ptr() as *mut i32,
                cb_vl,
            );
            _mm512_storeu_si512(
                v_ptr.get_unchecked_mut(ux..).as_mut_ptr() as *mut i32,
                cr_vl,
            );

            ux += 32;
        } else {
            let r_values = avx512_avg_epi16(r_values);
            let g_values = avx512_avg_epi16(g_values);
            let b_values = avx512_avg_epi16(b_values);

            let mut cb_h = _mm512_add_epi16(uv_bias, _mm512_mulhrs_epi16(r_values, v_cb_r));
            cb_h = _mm512_add_epi16(cb_h, _mm512_mulhrs_epi16(g_values, v_cb_g));
            cb_h = _mm512_add_epi16(cb_h, _mm512_mulhrs_epi16(b_values, v_cb_b));

            let mut cb_s = _mm512_max_epu16(_mm512_min_epu16(cb_h, i_cap_uv), y_bias);

            let mut cr_h = _mm512_add_epi16(uv_bias, _mm512_mulhrs_epi16(r_values, v_cr_r));
            cr_h = _mm512_add_epi16(cr_h, _mm512_mulhrs_epi16(g_values, v_cr_g));
            cr_h = _mm512_add_epi16(cr_h, _mm512_mulhrs_epi16(b_values, v_cr_b));

            let mut cr_s = _mm512_max_epu16(_mm512_min_epu16(cr_h, i_cap_uv), y_bias);

            if bytes_position == YuvBytesPacking::MostSignificantBytes {
                cb_s = _mm512_to_msb_epi16::<BIT_DEPTH>(cb_s);
                cr_s = _mm512_to_msb_epi16::<BIT_DEPTH>(cr_s);
            }

            if endianness == YuvEndianness::BigEndian {
                cb_s = _mm512_shuffle_epi8(cb_s, big_endian_shuffle_flag);
                cr_s = _mm512_shuffle_epi8(cr_s, big_endian_shuffle_flag);
            }

            _mm256_storeu_si256(
                u_ptr.get_unchecked_mut(ux..).as_mut_ptr() as *mut __m256i,
                _mm512_castsi512_si256(cb_s),
            );
            _mm256_storeu_si256(
                v_ptr.get_unchecked_mut(ux..).as_mut_ptr() as *mut __m256i,
                _mm512_castsi512_si256(cr_s),
            );
            ux += 16;
        }

        cx += 32;
    }

    ProcessedOffset { ux, cx }
}
