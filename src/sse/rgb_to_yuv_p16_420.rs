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
use crate::internals::ProcessedOffset;
use crate::sse::{_mm_havg_epi16_epi32, _mm_load_deinterleave_rgb16_for_yuv, sse_avg_epi16};
use crate::yuv_support::{CbCrForwardTransform, YuvChromaRange, YuvSourceChannels};
use crate::{YuvBytesPacking, YuvEndianness};
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

pub(crate) fn sse_rgba_to_yuv_p16_420<
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
        sse_rgba_to_yuv_impl::<ORIGIN_CHANNELS, ENDIANNESS, BYTES_POSITION, PRECISION, BIT_DEPTH>(
            transform, range, y_plane0, y_plane1, u_plane, v_plane, rgba0, rgba1, start_cx,
            start_ux, width,
        )
    }
}

#[target_feature(enable = "sse4.1")]
unsafe fn sse_rgba_to_yuv_impl<
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

    let rounding_const_bias: i32 = (1 << (PRECISION - 1)) - 1;
    let bias_y = range.bias_y as i32 * (1 << PRECISION) + rounding_const_bias;
    let bias_uv = range.bias_uv as i32 * (1 << PRECISION) + rounding_const_bias;

    let u_ptr = u_plane;
    let v_ptr = v_plane;

    let y_bias = _mm_set1_epi32(bias_y);
    let uv_bias = _mm_set1_epi32(bias_uv);
    let v_yr = _mm_set1_epi16(transform.yr as i16);
    let v_yg = _mm_set1_epi16(transform.yg as i16);
    let v_yb = _mm_set1_epi16(transform.yb as i16);
    let v_cb_r = _mm_set1_epi16(transform.cb_r as i16);
    let v_cb_g = _mm_set1_epi16(transform.cb_g as i16);
    let v_cb_b = _mm_set1_epi16(transform.cb_b as i16);
    let v_cr_r = _mm_set1_epi16(transform.cr_r as i16);
    let v_cr_g = _mm_set1_epi16(transform.cr_g as i16);
    let v_cr_b = _mm_set1_epi16(transform.cr_b as i16);

    let big_endian_shuffle_flag =
        _mm_setr_epi8(1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14);

    let mut cx = start_cx;
    let mut ux = start_ux;

    let v_shift_count = _mm_set1_epi64x(16 - BIT_DEPTH as i64);

    let zeros = _mm_setzero_si128();

    let i_bias_y = _mm_set1_epi16(range.bias_y as i16);
    let i_cap_y = _mm_set1_epi16((range.range_y as u16 + range.bias_y as u16) as i16);
    let i_cap_uv = _mm_set1_epi16((range.bias_y as u16 + range.range_uv as u16) as i16);

    while cx + 8 < width {
        let src_ptr0 = rgba0.get_unchecked(cx * channels..);
        let (r_values0, g_values0, b_values0) =
            _mm_load_deinterleave_rgb16_for_yuv::<ORIGIN_CHANNELS>(src_ptr0.as_ptr());
        let src_ptr1 = rgba1.get_unchecked(cx * channels..);
        let (r_values1, g_values1, b_values1) =
            _mm_load_deinterleave_rgb16_for_yuv::<ORIGIN_CHANNELS>(src_ptr1.as_ptr());

        let r0_hi = _mm_unpackhi_epi16(r_values0, zeros);
        let g0_hi = _mm_unpackhi_epi16(g_values0, zeros);
        let b0_hi = _mm_unpackhi_epi16(b_values0, zeros);
        let r0_lo = _mm_unpacklo_epi16(r_values0, zeros);
        let g0_lo = _mm_unpacklo_epi16(g_values0, zeros);
        let b0_lo = _mm_unpacklo_epi16(b_values0, zeros);

        let r1_hi = _mm_unpackhi_epi16(r_values1, zeros);
        let g1_hi = _mm_unpackhi_epi16(g_values1, zeros);
        let b1_hi = _mm_unpackhi_epi16(b_values1, zeros);
        let r1_lo = _mm_unpacklo_epi16(r_values1, zeros);
        let g1_lo = _mm_unpacklo_epi16(g_values1, zeros);
        let b1_lo = _mm_unpacklo_epi16(b_values1, zeros);

        let mut y0_h = _mm_add_epi32(y_bias, _mm_madd_epi16(r0_hi, v_yr));
        y0_h = _mm_add_epi32(y0_h, _mm_madd_epi16(g0_hi, v_yg));
        y0_h = _mm_add_epi32(y0_h, _mm_madd_epi16(b0_hi, v_yb));

        let mut y0_l = _mm_add_epi32(y_bias, _mm_madd_epi16(r0_lo, v_yr));
        y0_l = _mm_add_epi32(y0_l, _mm_madd_epi16(g0_lo, v_yg));
        y0_l = _mm_add_epi32(y0_l, _mm_madd_epi16(b0_lo, v_yb));

        let mut y0_vl = _mm_min_epu16(
            _mm_packus_epi32(
                _mm_srai_epi32::<PRECISION>(y0_l),
                _mm_srai_epi32::<PRECISION>(y0_h),
            ),
            i_cap_y,
        );

        let mut y1_h = _mm_add_epi32(y_bias, _mm_madd_epi16(r1_hi, v_yr));
        y1_h = _mm_add_epi32(y1_h, _mm_madd_epi16(g1_hi, v_yg));
        y1_h = _mm_add_epi32(y1_h, _mm_madd_epi16(b1_hi, v_yb));

        let mut y1_l = _mm_add_epi32(y_bias, _mm_madd_epi16(r1_lo, v_yr));
        y1_l = _mm_add_epi32(y1_l, _mm_madd_epi16(g1_lo, v_yg));
        y1_l = _mm_add_epi32(y1_l, _mm_madd_epi16(b1_lo, v_yb));

        let mut y1_vl = _mm_min_epu16(
            _mm_packus_epi32(
                _mm_srai_epi32::<PRECISION>(y1_l),
                _mm_srai_epi32::<PRECISION>(y1_h),
            ),
            i_cap_y,
        );

        if bytes_position == YuvBytesPacking::MostSignificantBytes {
            y0_vl = _mm_sll_epi16(y0_vl, v_shift_count);
            y1_vl = _mm_sll_epi16(y1_vl, v_shift_count);
        }

        if endianness == YuvEndianness::BigEndian {
            y0_vl = _mm_shuffle_epi8(y0_vl, big_endian_shuffle_flag);
            y1_vl = _mm_shuffle_epi8(y1_vl, big_endian_shuffle_flag);
        }

        _mm_storeu_si128(
            y_plane0.get_unchecked_mut(cx..).as_mut_ptr() as *mut __m128i,
            y0_vl,
        );
        _mm_storeu_si128(
            y_plane1.get_unchecked_mut(cx..).as_mut_ptr() as *mut __m128i,
            y1_vl,
        );

        let r_values = _mm_havg_epi16_epi32(_mm_avg_epu16(r_values0, r_values1));
        let g_values = _mm_havg_epi16_epi32(_mm_avg_epu16(g_values0, g_values1));
        let b_values = _mm_havg_epi16_epi32(_mm_avg_epu16(b_values0, b_values1));

        let mut cb_h = _mm_add_epi32(uv_bias, _mm_madd_epi16(r_values, v_cb_r));
        cb_h = _mm_add_epi32(cb_h, _mm_madd_epi16(g_values, v_cb_g));
        cb_h = _mm_add_epi32(cb_h, _mm_madd_epi16(b_values, v_cb_b));

        let mut cb_s = _mm_max_epu16(
            _mm_min_epu16(
                _mm_packus_epi32(_mm_srai_epi32::<PRECISION>(cb_h), _mm_setzero_si128()),
                i_cap_uv,
            ),
            i_bias_y,
        );

        let mut cr_h = _mm_add_epi32(uv_bias, _mm_madd_epi16(r_values, v_cr_r));
        cr_h = _mm_add_epi32(cr_h, _mm_madd_epi16(g_values, v_cr_g));
        cr_h = _mm_add_epi32(cr_h, _mm_madd_epi16(b_values, v_cr_b));

        let mut cr_s = _mm_max_epu16(
            _mm_min_epu16(
                _mm_packus_epi32(_mm_srai_epi32::<PRECISION>(cr_h), _mm_setzero_si128()),
                i_cap_uv,
            ),
            i_bias_y,
        );

        if bytes_position == YuvBytesPacking::MostSignificantBytes {
            cb_s = _mm_sll_epi16(cb_s, v_shift_count);
            cr_s = _mm_sll_epi16(cr_s, v_shift_count);
        }

        if endianness == YuvEndianness::BigEndian {
            cb_s = _mm_shuffle_epi8(cb_s, big_endian_shuffle_flag);
            cr_s = _mm_shuffle_epi8(cr_s, big_endian_shuffle_flag);
        }

        std::ptr::copy_nonoverlapping(
            &cb_s as *const _ as *const u8,
            u_ptr.get_unchecked_mut(ux..).as_mut_ptr() as *mut u8,
            8,
        );
        std::ptr::copy_nonoverlapping(
            &cr_s as *const _ as *const u8,
            v_ptr.get_unchecked_mut(ux..).as_mut_ptr() as *mut u8,
            8,
        );

        ux += 4;

        cx += 8;
    }

    ProcessedOffset { ux, cx }
}

// Slightly lower precision option, suitable only for bit-depth <= 12
pub(crate) fn sse_rgba_to_yuv_p16_lp420<
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
        sse_rgba_to_yuv_impl_lp::<ORIGIN_CHANNELS, ENDIANNESS, BYTES_POSITION, PRECISION, BIT_DEPTH>(
            transform, range, y_plane0, y_plane1, u_plane, v_plane, rgba0, rgba1, start_cx,
            start_ux, width,
        )
    }
}

#[target_feature(enable = "sse4.1")]
unsafe fn sse_rgba_to_yuv_impl_lp<
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

    let y_bias = _mm_set1_epi16(bias_y);
    let uv_bias = _mm_set1_epi16(bias_uv);
    let v_yr = _mm_set1_epi16(transform.yr as i16);
    let v_yg = _mm_set1_epi16(transform.yg as i16);
    let v_yb = _mm_set1_epi16(transform.yb as i16);
    let v_cb_r = _mm_set1_epi16(transform.cb_r as i16);
    let v_cb_g = _mm_set1_epi16(transform.cb_g as i16);
    let v_cb_b = _mm_set1_epi16(transform.cb_b as i16);
    let v_cr_r = _mm_set1_epi16(transform.cr_r as i16);
    let v_cr_g = _mm_set1_epi16(transform.cr_g as i16);
    let v_cr_b = _mm_set1_epi16(transform.cr_b as i16);

    let big_endian_shuffle_flag =
        _mm_setr_epi8(1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14);

    let mut cx = start_cx;
    let mut ux = start_ux;

    let v_shift_count = _mm_set1_epi64x(16 - BIT_DEPTH as i64);

    let i_bias_y = _mm_set1_epi16(range.bias_y as i16);
    let i_cap_y = _mm_set1_epi16((range.range_y as u16 + range.bias_y as u16) as i16);
    let i_cap_uv = _mm_set1_epi16((range.bias_y as u16 + range.range_uv as u16) as i16);

    const SCALE: i32 = 2;

    while cx + 8 < width {
        let src_ptr0 = rgba0.get_unchecked(cx * channels..);
        let (mut r_values0, mut g_values0, mut b_values0) =
            _mm_load_deinterleave_rgb16_for_yuv::<ORIGIN_CHANNELS>(src_ptr0.as_ptr());
        let src_ptr1 = rgba1.get_unchecked(cx * channels..);
        let (mut r_values1, mut g_values1, mut b_values1) =
            _mm_load_deinterleave_rgb16_for_yuv::<ORIGIN_CHANNELS>(src_ptr1.as_ptr());

        r_values0 = _mm_slli_epi16::<SCALE>(r_values0);
        g_values0 = _mm_slli_epi16::<SCALE>(g_values0);
        b_values0 = _mm_slli_epi16::<SCALE>(b_values0);

        r_values1 = _mm_slli_epi16::<SCALE>(r_values1);
        g_values1 = _mm_slli_epi16::<SCALE>(g_values1);
        b_values1 = _mm_slli_epi16::<SCALE>(b_values1);

        let mut y0_h = _mm_add_epi16(y_bias, _mm_mulhrs_epi16(r_values0, v_yr));
        y0_h = _mm_add_epi16(y0_h, _mm_mulhrs_epi16(g_values0, v_yg));
        y0_h = _mm_add_epi16(y0_h, _mm_mulhrs_epi16(b_values0, v_yb));

        let mut y0_vl = _mm_min_epu16(y0_h, i_cap_y);

        let mut y1_h = _mm_add_epi16(y_bias, _mm_mulhrs_epi16(r_values1, v_yr));
        y1_h = _mm_add_epi16(y1_h, _mm_mulhrs_epi16(g_values1, v_yg));
        y1_h = _mm_add_epi16(y1_h, _mm_mulhrs_epi16(b_values1, v_yb));

        let mut y1_vl = _mm_min_epu16(y1_h, i_cap_y);

        if bytes_position == YuvBytesPacking::MostSignificantBytes {
            y0_vl = _mm_sll_epi16(y0_vl, v_shift_count);
            y1_vl = _mm_sll_epi16(y1_vl, v_shift_count);
        }

        if endianness == YuvEndianness::BigEndian {
            y0_vl = _mm_shuffle_epi8(y0_vl, big_endian_shuffle_flag);
            y1_vl = _mm_shuffle_epi8(y1_vl, big_endian_shuffle_flag);
        }

        _mm_storeu_si128(
            y_plane0.get_unchecked_mut(cx..).as_mut_ptr() as *mut __m128i,
            y0_vl,
        );

        _mm_storeu_si128(
            y_plane1.get_unchecked_mut(cx..).as_mut_ptr() as *mut __m128i,
            y1_vl,
        );

        let r_values = sse_avg_epi16(_mm_avg_epu16(r_values0, r_values1));
        let g_values = sse_avg_epi16(_mm_avg_epu16(g_values0, g_values1));
        let b_values = sse_avg_epi16(_mm_avg_epu16(b_values0, b_values1));

        let mut cb_h = _mm_add_epi16(uv_bias, _mm_mulhrs_epi16(r_values, v_cb_r));
        cb_h = _mm_add_epi16(cb_h, _mm_mulhrs_epi16(g_values, v_cb_g));
        cb_h = _mm_add_epi16(cb_h, _mm_mulhrs_epi16(b_values, v_cb_b));

        let mut cb_s = _mm_max_epu16(_mm_min_epu16(cb_h, i_cap_uv), i_bias_y);

        let mut cr_h = _mm_add_epi16(uv_bias, _mm_mulhrs_epi16(r_values, v_cr_r));
        cr_h = _mm_add_epi16(cr_h, _mm_mulhrs_epi16(g_values, v_cr_g));
        cr_h = _mm_add_epi16(cr_h, _mm_mulhrs_epi16(b_values, v_cr_b));

        let mut cr_s = _mm_max_epu16(_mm_min_epu16(cr_h, i_cap_uv), i_bias_y);

        if bytes_position == YuvBytesPacking::MostSignificantBytes {
            cb_s = _mm_sll_epi16(cb_s, v_shift_count);
            cr_s = _mm_sll_epi16(cr_s, v_shift_count);
        }

        if endianness == YuvEndianness::BigEndian {
            cb_s = _mm_shuffle_epi8(cb_s, big_endian_shuffle_flag);
            cr_s = _mm_shuffle_epi8(cr_s, big_endian_shuffle_flag);
        }

        std::ptr::copy_nonoverlapping(
            &cb_s as *const _ as *const u8,
            u_ptr.get_unchecked_mut(ux..).as_mut_ptr() as *mut u8,
            8,
        );
        std::ptr::copy_nonoverlapping(
            &cr_s as *const _ as *const u8,
            v_ptr.get_unchecked_mut(ux..).as_mut_ptr() as *mut u8,
            8,
        );

        ux += 4;
        cx += 8;
    }

    ProcessedOffset { ux, cx }
}
