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
use crate::sse::{
    _mm_deinterleave_x2_epi8, _mm_expand8_hi_to_10, _mm_expand8_lo_to_10,
    _mm_store_interleave_half_rgb_for_yuv, _mm_store_interleave_rgb_for_yuv,
};
use crate::yuv_support::{CbCrInverseTransform, YuvChromaRange, YuvNVOrder, YuvSourceChannels};
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// This is common NV row conversion to RGBx, supports only 4:2:2 subsampling
pub(crate) fn sse_yuv_nv_to_rgba422<const UV_ORDER: u8, const DESTINATION_CHANNELS: u8>(
    range: &YuvChromaRange,
    transform: &CbCrInverseTransform<i32>,
    y_plane: &[u8],
    uv_plane: &[u8],
    rgba: &mut [u8],
    start_cx: usize,
    start_ux: usize,
    width: usize,
) -> ProcessedOffset {
    unsafe {
        sse_yuv_nv_to_rgba_impl422::<UV_ORDER, DESTINATION_CHANNELS>(
            range, transform, y_plane, uv_plane, rgba, start_cx, start_ux, width,
        )
    }
}

#[target_feature(enable = "sse4.1")]
unsafe fn sse_yuv_nv_to_rgba_impl422<const UV_ORDER: u8, const DESTINATION_CHANNELS: u8>(
    range: &YuvChromaRange,
    transform: &CbCrInverseTransform<i32>,
    y_plane: &[u8],
    uv_plane: &[u8],
    rgba: &mut [u8],
    start_cx: usize,
    start_ux: usize,
    width: usize,
) -> ProcessedOffset {
    let order: YuvNVOrder = UV_ORDER.into();
    let destination_channels: YuvSourceChannels = DESTINATION_CHANNELS.into();
    let channels = destination_channels.get_channels_count();

    let mut cx = start_cx;
    let mut uv_x = start_ux;

    let y_ptr = y_plane.as_ptr();
    let uv_ptr = uv_plane.as_ptr();
    let rgba_ptr = rgba.as_mut_ptr();

    let y_corr = _mm_set1_epi8(range.bias_y as i8);
    let uv_corr = _mm_set1_epi16(((range.bias_uv as i16) << 2) | ((range.bias_uv as i16) >> 6));
    let v_luma_coeff = _mm_set1_epi16(transform.y_coef as i16);
    let v_cr_coeff = _mm_set1_epi16(transform.cr_coef as i16);
    let v_cb_coeff = _mm_set1_epi16(transform.cb_coef as i16);
    let v_g_coeff_1 = _mm_set1_epi16(transform.g_coeff_1 as i16);
    let v_g_coeff_2 = _mm_set1_epi16(transform.g_coeff_2 as i16);

    let zeros = _mm_setzero_si128();

    while cx + 16 < width {
        let y_values = _mm_subs_epu8(_mm_loadu_si128(y_ptr.add(cx) as *const __m128i), y_corr);

        let uv_values_ = _mm_loadu_si128(uv_ptr.add(uv_x) as *const __m128i);
        let (mut u, mut v) = _mm_deinterleave_x2_epi8(uv_values_, zeros);

        u = _mm_sub_epi16(_mm_srli_epi16::<6>(_mm_unpacklo_epi8(u, u)), uv_corr);
        v = _mm_sub_epi16(_mm_srli_epi16::<6>(_mm_unpacklo_epi8(v, v)), uv_corr);

        if order == YuvNVOrder::VU {
            let j = u;
            u = v;
            v = j;
        }

        let v_u = _mm_mulhrs_epi16(u, v_cb_coeff);
        let v_v = _mm_mulhrs_epi16(v, v_cr_coeff);

        let v_g = _mm_add_epi16(
            _mm_mulhrs_epi16(v, v_g_coeff_1),
            _mm_mulhrs_epi16(u, v_g_coeff_2),
        );

        let (v_u_l, v_u_h) = (_mm_unpacklo_epi16(v_u, v_u), _mm_unpackhi_epi16(v_u, v_u));

        let (v_v_l, v_v_h) = (_mm_unpacklo_epi16(v_v, v_v), _mm_unpackhi_epi16(v_v, v_v));

        let (v_g_l, v_g_h) = (_mm_unpacklo_epi16(v_g, v_g), _mm_unpackhi_epi16(v_g, v_g));

        let y_high = _mm_mulhrs_epi16(_mm_expand8_hi_to_10(y_values), v_luma_coeff);

        let r_high = _mm_add_epi16(y_high, v_v_h);
        let b_high = _mm_add_epi16(y_high, v_u_h);
        let g_high = _mm_sub_epi16(y_high, v_g_h);

        let y_low = _mm_mulhrs_epi16(_mm_expand8_lo_to_10(y_values), v_luma_coeff);

        let r_low = _mm_add_epi16(y_low, v_v_l);
        let b_low = _mm_add_epi16(y_low, v_u_l);
        let g_low = _mm_sub_epi16(y_low, v_g_l);

        let r_values = _mm_packus_epi16(r_low, r_high);
        let g_values = _mm_packus_epi16(g_low, g_high);
        let b_values = _mm_packus_epi16(b_low, b_high);

        let dst_shift = cx * channels;

        let v_alpha = _mm_set1_epi8(255u8 as i8);

        _mm_store_interleave_rgb_for_yuv::<DESTINATION_CHANNELS>(
            rgba_ptr.add(dst_shift),
            r_values,
            g_values,
            b_values,
            v_alpha,
        );

        cx += 16;
        uv_x += 16;
    }

    while cx + 8 < width {
        let y_values = _mm_subs_epi8(_mm_loadu_si64(y_ptr.add(cx)), y_corr);

        let uv_values_ = _mm_loadu_si64(uv_ptr.add(uv_x));
        let (mut u, mut v) = _mm_deinterleave_x2_epi8(uv_values_, zeros);

        let distribute_shuffle = _mm_setr_epi8(0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3);

        u = _mm_sub_epi16(
            _mm_srli_epi16::<6>(_mm_shuffle_epi8(u, distribute_shuffle)),
            uv_corr,
        );
        v = _mm_sub_epi16(
            _mm_srli_epi16::<6>(_mm_shuffle_epi8(v, distribute_shuffle)),
            uv_corr,
        );

        if order == YuvNVOrder::VU {
            let j = u;
            u = v;
            v = j;
        }

        let v_u = _mm_mulhrs_epi16(u, v_cb_coeff);
        let v_v = _mm_mulhrs_epi16(v, v_cr_coeff);

        let v_g = _mm_add_epi16(
            _mm_mulhrs_epi16(v, v_g_coeff_1),
            _mm_mulhrs_epi16(u, v_g_coeff_2),
        );

        let y_low = _mm_mulhrs_epi16(_mm_expand8_lo_to_10(y_values), v_luma_coeff);

        let r_low = _mm_add_epi16(y_low, v_v);
        let b_low = _mm_add_epi16(y_low, v_u);
        let g_low = _mm_sub_epi16(y_low, v_g);

        let r_values = _mm_packus_epi16(r_low, zeros);
        let g_values = _mm_packus_epi16(g_low, zeros);
        let b_values = _mm_packus_epi16(b_low, zeros);

        let dst_shift = cx * channels;
        let dst_ptr = rgba_ptr.add(dst_shift);

        let v_alpha = _mm_set1_epi8(255u8 as i8);

        _mm_store_interleave_half_rgb_for_yuv::<DESTINATION_CHANNELS>(
            dst_ptr, r_values, g_values, b_values, v_alpha,
        );

        cx += 8;
        uv_x += 8;
    }

    ProcessedOffset { cx, ux: uv_x }
}
