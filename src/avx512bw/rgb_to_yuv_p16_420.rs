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
use crate::yuv_support::{CbCrForwardTransform, YuvChromaRange, YuvSourceChannels};
use crate::{YuvBytesPacking, YuvEndianness};
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

// Slightly lower precision option, suitable only for bit-depth <= 12
pub(crate) fn avx512_rgba_to_yuv_p16_lp420<
    const ORIGIN_CHANNELS: u8,
    const ENDIANNESS: u8,
    const BYTES_POSITION: u8,
    const PRECISION: i32,
    const BIT_DEPTH: usize,
>(
    transform: &CbCrForwardTransform<i32>,
    range: &YuvChromaRange,
    y_plane0: &mut [u16],
    y_plane1: &mut [u16],
    u_plane: &mut [u16],
    v_plane: &mut [u16],
    rgba0: &[u16],
    rgba1: &[u16],
    start_cx: usize,
    start_ux: usize,
    width: usize,
) -> ProcessedOffset {
    unsafe {
        avx512_rgba_to_yuv_impl_lp::<
            ORIGIN_CHANNELS,
            ENDIANNESS,
            BYTES_POSITION,
            PRECISION,
            BIT_DEPTH,
        >(
            transform, range, y_plane0, y_plane1, u_plane, v_plane, rgba0, rgba1, start_cx,
            start_ux, width,
        )
    }
}

#[target_feature(enable = "avx512bw")]
unsafe fn avx512_rgba_to_yuv_impl_lp<
    const ORIGIN_CHANNELS: u8,
    const ENDIANNESS: u8,
    const BYTES_POSITION: u8,
    const PRECISION: i32,
    const BIT_DEPTH: usize,
>(
    transform: &CbCrForwardTransform<i32>,
    range: &YuvChromaRange,
    y_plane0: &mut [u16],
    y_plane1: &mut [u16],
    u_plane: &mut [u16],
    v_plane: &mut [u16],
    rgba0: &[u16],
    rgba1: &[u16],
    start_cx: usize,
    start_ux: usize,
    width: usize,
) -> ProcessedOffset {
    let source_channels: YuvSourceChannels = ORIGIN_CHANNELS.into();
    let endianness: YuvEndianness = ENDIANNESS.into();
    let bytes_position: YuvBytesPacking = BYTES_POSITION.into();
    let channels = source_channels.get_channels_count();

    let bias_y = range.bias_y as i16;
    let bias_uv = range.bias_uv as i16;

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
        let src_ptr0 = rgba0.get_unchecked(cx * channels..);
        let (mut r_values0, mut g_values0, mut b_values0) =
            _mm512_load_deinterleave_rgb16_for_yuv::<ORIGIN_CHANNELS>(src_ptr0.as_ptr());
        let src_ptr1 = rgba1.get_unchecked(cx * channels..);
        let (mut r_values1, mut g_values1, mut b_values1) =
            _mm512_load_deinterleave_rgb16_for_yuv::<ORIGIN_CHANNELS>(src_ptr1.as_ptr());

        r_values0 = _mm512_slli_epi16::<SCALE>(r_values0);
        g_values0 = _mm512_slli_epi16::<SCALE>(g_values0);
        b_values0 = _mm512_slli_epi16::<SCALE>(b_values0);

        r_values1 = _mm512_slli_epi16::<SCALE>(r_values1);
        g_values1 = _mm512_slli_epi16::<SCALE>(g_values1);
        b_values1 = _mm512_slli_epi16::<SCALE>(b_values1);

        let mut y0_h = _mm512_add_epi16(y_bias, _mm512_mulhrs_epi16(r_values0, v_yr));
        y0_h = _mm512_add_epi16(y0_h, _mm512_mulhrs_epi16(g_values0, v_yg));
        y0_h = _mm512_add_epi16(y0_h, _mm512_mulhrs_epi16(b_values0, v_yb));

        let mut y0_vl = _mm512_min_epu16(y0_h, i_cap_y);

        let mut y1_h = _mm512_add_epi16(y_bias, _mm512_mulhrs_epi16(r_values1, v_yr));
        y1_h = _mm512_add_epi16(y1_h, _mm512_mulhrs_epi16(g_values1, v_yg));
        y1_h = _mm512_add_epi16(y1_h, _mm512_mulhrs_epi16(b_values1, v_yb));

        let mut y1_vl = _mm512_min_epu16(y1_h, i_cap_y);

        if bytes_position == YuvBytesPacking::MostSignificantBytes {
            y0_vl = _mm512_to_msb_epi16::<BIT_DEPTH>(y0_vl);
            y1_vl = _mm512_to_msb_epi16::<BIT_DEPTH>(y1_vl);
        }

        if endianness == YuvEndianness::BigEndian {
            y0_vl = _mm512_shuffle_epi8(y0_vl, big_endian_shuffle_flag);
            y1_vl = _mm512_shuffle_epi8(y1_vl, big_endian_shuffle_flag);
        }

        _mm512_storeu_si512(
            y_plane0.get_unchecked_mut(cx..).as_mut_ptr() as *mut i32,
            y0_vl,
        );

        _mm512_storeu_si512(
            y_plane1.get_unchecked_mut(cx..).as_mut_ptr() as *mut i32,
            y1_vl,
        );

        let r_values = avx512_avg_epi16(_mm512_avg_epu16(r_values0, r_values1));
        let g_values = avx512_avg_epi16(_mm512_avg_epu16(g_values0, g_values1));
        let b_values = avx512_avg_epi16(_mm512_avg_epu16(b_values0, b_values1));

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
        cx += 32;
    }

    ProcessedOffset { ux, cx }
}
