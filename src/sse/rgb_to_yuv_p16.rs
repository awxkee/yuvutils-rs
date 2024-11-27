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
use crate::sse::{_mm_deinterleave_rgb_epi16, _mm_deinterleave_rgba_epi16, sse_avg_epi16};
use crate::yuv_support::{
    CbCrForwardTransform, YuvChromaRange, YuvChromaSubsampling, YuvSourceChannels,
};
use crate::{YuvBytesPacking, YuvEndianness};
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

pub(crate) fn sse_rgba_to_yuv_p16<
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
    compute_uv_row: bool,
) -> ProcessedOffset {
    unsafe {
        sse_rgba_to_yuv_impl::<
            ORIGIN_CHANNELS,
            SAMPLING,
            ENDIANNESS,
            BYTES_POSITION,
            PRECISION,
            BIT_DEPTH,
        >(
            transform,
            range,
            y_plane,
            u_plane,
            v_plane,
            rgba,
            start_cx,
            start_ux,
            width,
            compute_uv_row,
        )
    }
}

#[target_feature(enable = "sse4.1")]
unsafe fn sse_rgba_to_yuv_impl<
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
    compute_uv_row: bool,
) -> ProcessedOffset {
    let chroma_subsampling: YuvChromaSubsampling = SAMPLING.into();
    let source_channels: YuvSourceChannels = ORIGIN_CHANNELS.into();
    let endianness: YuvEndianness = ENDIANNESS.into();
    let bytes_position: YuvBytesPacking = BYTES_POSITION.into();
    let channels = source_channels.get_channels_count();

    let rounding_const_bias: i32 = 1 << (PRECISION - 1);
    let bias_y = range.bias_y as i32 * (1 << PRECISION) + rounding_const_bias;
    let bias_uv = range.bias_uv as i32 * (1 << PRECISION) + rounding_const_bias;

    let src_ptr = rgba;

    let y_ptr = y_plane;
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
        let r_values;
        let g_values;
        let b_values;

        let src_ptr = src_ptr.get_unchecked(cx * channels..);

        let row0 = _mm_loadu_si128(src_ptr.as_ptr() as *const __m128i);
        let row1 = _mm_loadu_si128(src_ptr.get_unchecked(8..).as_ptr() as *const __m128i);
        let row2 = _mm_loadu_si128(src_ptr.get_unchecked(16..).as_ptr() as *const __m128i);

        match source_channels {
            YuvSourceChannels::Rgb | YuvSourceChannels::Bgr => {
                let rgb_values = _mm_deinterleave_rgb_epi16(row0, row1, row2);
                if source_channels == YuvSourceChannels::Rgb {
                    r_values = rgb_values.0;
                    g_values = rgb_values.1;
                    b_values = rgb_values.2;
                } else {
                    r_values = rgb_values.2;
                    g_values = rgb_values.1;
                    b_values = rgb_values.0;
                }
            }
            YuvSourceChannels::Rgba => {
                let row3 = _mm_loadu_si128(src_ptr.get_unchecked(24..).as_ptr() as *const __m128i);
                let rgb_values = _mm_deinterleave_rgba_epi16(row0, row1, row2, row3);
                r_values = rgb_values.0;
                g_values = rgb_values.1;
                b_values = rgb_values.2;
            }
            YuvSourceChannels::Bgra => {
                let row3 = _mm_loadu_si128(src_ptr.get_unchecked(24..).as_ptr() as *const __m128i);
                let rgb_values = _mm_deinterleave_rgba_epi16(row0, row1, row2, row3);
                r_values = rgb_values.2;
                g_values = rgb_values.1;
                b_values = rgb_values.0;
            }
        }

        let r_hi = _mm_unpackhi_epi16(r_values, zeros);
        let g_hi = _mm_unpackhi_epi16(g_values, zeros);
        let b_hi = _mm_unpackhi_epi16(b_values, zeros);
        let r_lo = _mm_unpacklo_epi16(r_values, zeros);
        let g_lo = _mm_unpacklo_epi16(g_values, zeros);
        let b_lo = _mm_unpacklo_epi16(b_values, zeros);

        let mut y_h = _mm_add_epi32(y_bias, _mm_madd_epi16(r_hi, v_yr));
        y_h = _mm_add_epi32(y_h, _mm_madd_epi16(g_hi, v_yg));
        y_h = _mm_add_epi32(y_h, _mm_madd_epi16(b_hi, v_yb));

        let mut y_l = _mm_add_epi32(y_bias, _mm_madd_epi16(r_lo, v_yr));
        y_l = _mm_add_epi32(y_l, _mm_madd_epi16(g_lo, v_yg));
        y_l = _mm_add_epi32(y_l, _mm_madd_epi16(b_lo, v_yb));

        let mut y_vl = _mm_min_epu16(
            _mm_max_epu16(
                _mm_packus_epi32(
                    _mm_srai_epi32::<PRECISION>(y_l),
                    _mm_srai_epi32::<PRECISION>(y_h),
                ),
                i_bias_y,
            ),
            i_cap_y,
        );

        if bytes_position == YuvBytesPacking::MostSignificantBytes {
            y_vl = _mm_sll_epi32(y_vl, v_shift_count);
        }

        if endianness == YuvEndianness::BigEndian {
            y_vl = _mm_shuffle_epi8(y_vl, big_endian_shuffle_flag);
        }

        _mm_storeu_si128(
            y_ptr.get_unchecked_mut(cx..).as_mut_ptr() as *mut __m128i,
            y_vl,
        );

        if compute_uv_row {
            let mut cb_h = _mm_add_epi32(uv_bias, _mm_madd_epi16(r_hi, v_cb_r));
            cb_h = _mm_add_epi32(cb_h, _mm_madd_epi16(g_hi, v_cb_g));
            cb_h = _mm_add_epi32(cb_h, _mm_madd_epi16(b_hi, v_cb_b));

            let mut cb_l = _mm_add_epi32(uv_bias, _mm_madd_epi16(r_lo, v_cb_r));
            cb_l = _mm_add_epi32(cb_l, _mm_madd_epi16(g_lo, v_cb_g));
            cb_l = _mm_add_epi32(cb_l, _mm_madd_epi16(b_lo, v_cb_b));

            let mut cb_vl = _mm_max_epu16(
                _mm_min_epu16(
                    _mm_packus_epi32(
                        _mm_srai_epi32::<PRECISION>(cb_l),
                        _mm_srai_epi32::<PRECISION>(cb_h),
                    ),
                    i_cap_uv,
                ),
                i_bias_y,
            );

            let mut cr_h = _mm_add_epi32(uv_bias, _mm_madd_epi16(r_hi, v_cr_r));
            cr_h = _mm_add_epi32(cr_h, _mm_madd_epi16(g_hi, v_cr_g));
            cr_h = _mm_add_epi32(cr_h, _mm_madd_epi16(b_hi, v_cr_b));

            let mut cr_l = _mm_add_epi32(uv_bias, _mm_madd_epi16(r_lo, v_cr_r));
            cr_l = _mm_add_epi32(cr_l, _mm_madd_epi16(g_lo, v_cr_g));
            cr_l = _mm_add_epi32(cr_l, _mm_madd_epi16(b_lo, v_cr_b));

            let mut cr_vl = _mm_max_epu16(
                _mm_min_epu16(
                    _mm_packus_epi32(
                        _mm_srai_epi32::<PRECISION>(cr_l),
                        _mm_srai_epi32::<PRECISION>(cr_h),
                    ),
                    i_cap_uv,
                ),
                i_bias_y,
            );

            match chroma_subsampling {
                YuvChromaSubsampling::Yuv420 | YuvChromaSubsampling::Yuv422 => {
                    let mut cb_s = sse_avg_epi16(cb_vl);
                    let mut cr_s = sse_avg_epi16(cr_vl);

                    if bytes_position == YuvBytesPacking::MostSignificantBytes {
                        cb_s = _mm_sll_epi32(cb_s, v_shift_count);
                        cr_s = _mm_sll_epi32(cr_s, v_shift_count);
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
                }
                YuvChromaSubsampling::Yuv444 => {
                    if bytes_position == YuvBytesPacking::MostSignificantBytes {
                        cb_vl = _mm_sll_epi32(cb_vl, v_shift_count);
                        cr_vl = _mm_sll_epi32(cr_vl, v_shift_count);
                    }

                    if endianness == YuvEndianness::BigEndian {
                        cb_vl = _mm_shuffle_epi8(cb_vl, big_endian_shuffle_flag);
                        cr_vl = _mm_shuffle_epi8(cr_vl, big_endian_shuffle_flag);
                    }

                    _mm_storeu_si128(
                        u_ptr.get_unchecked_mut(ux..).as_mut_ptr() as *mut __m128i,
                        cb_vl,
                    );
                    _mm_storeu_si128(
                        v_ptr.get_unchecked_mut(ux..).as_mut_ptr() as *mut __m128i,
                        cr_vl,
                    );

                    ux += 8;
                }
            }
        }

        cx += 8;
    }

    ProcessedOffset { ux, cx }
}

// Slightly lower precision option, suitable only for bit-depth <= 12
pub(crate) fn sse_rgba_to_yuv_p16_lp<
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
    compute_uv_row: bool,
) -> ProcessedOffset {
    unsafe {
        sse_rgba_to_yuv_impl_lp::<
            ORIGIN_CHANNELS,
            SAMPLING,
            ENDIANNESS,
            BYTES_POSITION,
            PRECISION,
            BIT_DEPTH,
        >(
            transform,
            range,
            y_plane,
            u_plane,
            v_plane,
            rgba,
            start_cx,
            start_ux,
            width,
            compute_uv_row,
        )
    }
}

#[target_feature(enable = "sse4.1")]
unsafe fn sse_rgba_to_yuv_impl_lp<
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
    compute_uv_row: bool,
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
        let mut r_values;
        let mut g_values;
        let mut b_values;

        let src_ptr = src_ptr.get_unchecked(cx * channels..);

        let row0 = _mm_loadu_si128(src_ptr.as_ptr() as *const __m128i);
        let row1 = _mm_loadu_si128(src_ptr.get_unchecked(8..).as_ptr() as *const __m128i);
        let row2 = _mm_loadu_si128(src_ptr.get_unchecked(16..).as_ptr() as *const __m128i);

        match source_channels {
            YuvSourceChannels::Rgb | YuvSourceChannels::Bgr => {
                let rgb_values = _mm_deinterleave_rgb_epi16(row0, row1, row2);
                if source_channels == YuvSourceChannels::Rgb {
                    r_values = rgb_values.0;
                    g_values = rgb_values.1;
                    b_values = rgb_values.2;
                } else {
                    r_values = rgb_values.2;
                    g_values = rgb_values.1;
                    b_values = rgb_values.0;
                }
            }
            YuvSourceChannels::Rgba => {
                let row3 = _mm_loadu_si128(src_ptr.get_unchecked(24..).as_ptr() as *const __m128i);
                let rgb_values = _mm_deinterleave_rgba_epi16(row0, row1, row2, row3);
                r_values = rgb_values.0;
                g_values = rgb_values.1;
                b_values = rgb_values.2;
            }
            YuvSourceChannels::Bgra => {
                let row3 = _mm_loadu_si128(src_ptr.get_unchecked(24..).as_ptr() as *const __m128i);
                let rgb_values = _mm_deinterleave_rgba_epi16(row0, row1, row2, row3);
                r_values = rgb_values.2;
                g_values = rgb_values.1;
                b_values = rgb_values.0;
            }
        }

        r_values = _mm_slli_epi16::<SCALE>(r_values);
        g_values = _mm_slli_epi16::<SCALE>(g_values);
        b_values = _mm_slli_epi16::<SCALE>(b_values);

        let mut y_h = _mm_add_epi16(y_bias, _mm_mulhrs_epi16(r_values, v_yr));
        y_h = _mm_add_epi16(y_h, _mm_mulhrs_epi16(g_values, v_yg));
        y_h = _mm_add_epi16(y_h, _mm_mulhrs_epi16(b_values, v_yb));

        let mut y_vl = _mm_min_epu16(_mm_max_epu16(y_h, i_bias_y), i_cap_y);

        if bytes_position == YuvBytesPacking::MostSignificantBytes {
            y_vl = _mm_sll_epi32(y_vl, v_shift_count);
        }

        if endianness == YuvEndianness::BigEndian {
            y_vl = _mm_shuffle_epi8(y_vl, big_endian_shuffle_flag);
        }

        _mm_storeu_si128(
            y_ptr.get_unchecked_mut(cx..).as_mut_ptr() as *mut __m128i,
            y_vl,
        );

        if compute_uv_row {
            let mut cb_h = _mm_add_epi16(uv_bias, _mm_mulhrs_epi16(r_values, v_cb_r));
            cb_h = _mm_add_epi16(cb_h, _mm_mulhrs_epi16(g_values, v_cb_g));
            cb_h = _mm_add_epi16(cb_h, _mm_mulhrs_epi16(b_values, v_cb_b));

            let mut cb_vl = _mm_max_epu16(_mm_min_epu16(cb_h, i_cap_uv), i_bias_y);

            let mut cr_h = _mm_add_epi16(uv_bias, _mm_mulhrs_epi16(r_values, v_cr_r));
            cr_h = _mm_add_epi16(cr_h, _mm_mulhrs_epi16(g_values, v_cr_g));
            cr_h = _mm_add_epi16(cr_h, _mm_mulhrs_epi16(b_values, v_cr_b));

            let mut cr_vl = _mm_max_epu16(_mm_min_epu16(cr_h, i_cap_uv), i_bias_y);

            match chroma_subsampling {
                YuvChromaSubsampling::Yuv420 | YuvChromaSubsampling::Yuv422 => {
                    let mut cb_s = sse_avg_epi16(cb_vl);
                    let mut cr_s = sse_avg_epi16(cr_vl);

                    if bytes_position == YuvBytesPacking::MostSignificantBytes {
                        cb_s = _mm_sll_epi32(cb_s, v_shift_count);
                        cr_s = _mm_sll_epi32(cr_s, v_shift_count);
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
                }
                YuvChromaSubsampling::Yuv444 => {
                    if bytes_position == YuvBytesPacking::MostSignificantBytes {
                        cb_vl = _mm_sll_epi32(cb_vl, v_shift_count);
                        cr_vl = _mm_sll_epi32(cr_vl, v_shift_count);
                    }

                    if endianness == YuvEndianness::BigEndian {
                        cb_vl = _mm_shuffle_epi8(cb_vl, big_endian_shuffle_flag);
                        cr_vl = _mm_shuffle_epi8(cr_vl, big_endian_shuffle_flag);
                    }

                    _mm_storeu_si128(
                        u_ptr.get_unchecked_mut(ux..).as_mut_ptr() as *mut __m128i,
                        cb_vl,
                    );
                    _mm_storeu_si128(
                        v_ptr.get_unchecked_mut(ux..).as_mut_ptr() as *mut __m128i,
                        cr_vl,
                    );

                    ux += 8;
                }
            }
        }

        cx += 8;
    }

    ProcessedOffset { ux, cx }
}
