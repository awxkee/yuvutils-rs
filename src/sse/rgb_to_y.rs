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

use crate::sse::{
    _mm_affine_dot, _mm_load_deinterleave_half_rgb_for_yuv, _mm_load_deinterleave_rgb_for_yuv,
};
use crate::yuv_support::{CbCrForwardTransform, YuvChromaRange, YuvSourceChannels};
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
use std::ops::Shl;

pub(crate) fn sse_rgb_to_y<const ORIGIN_CHANNELS: u8, const PRECISION: i32>(
    transform: &CbCrForwardTransform<i32>,
    range: &YuvChromaRange,
    y_plane: &mut [u8],
    rgba: &[u8],
    start_cx: usize,
    width: usize,
) -> usize {
    unsafe {
        sse_rgb_to_y_impl::<ORIGIN_CHANNELS, PRECISION>(
            transform, range, y_plane, rgba, start_cx, width,
        )
    }
}

#[target_feature(enable = "sse4.1")]
unsafe fn sse_rgb_to_y_impl<const ORIGIN_CHANNELS: u8, const PRECISION: i32>(
    transform: &CbCrForwardTransform<i32>,
    range: &YuvChromaRange,
    y_plane: &mut [u8],
    rgba: &[u8],
    start_cx: usize,
    width: usize,
) -> usize {
    let source_channels: YuvSourceChannels = ORIGIN_CHANNELS.into();
    let channels = source_channels.get_channels_count();

    let y_ptr = y_plane;
    let rgba_ptr = rgba.as_ptr();

    let mut cx = start_cx;

    let bias_y = range.bias_y as i16;

    let zeros = _mm_setzero_si128();

    let y_base = _mm_set1_epi32(bias_y as i32 * (1 << PRECISION) + (1 << (PRECISION - 1)) - 1);
    let v_yr_yg = _mm_set1_epi32(transform.yg.shl(16) | transform.yr);
    let v_yb = _mm_set1_epi16(transform.yb as i16);

    while cx + 16 < width {
        let px = cx * channels;
        let (r_values, g_values, b_values) =
            _mm_load_deinterleave_rgb_for_yuv::<ORIGIN_CHANNELS>(rgba_ptr.add(px));

        let r_lo16 = _mm_unpacklo_epi8(r_values, zeros);
        let r_hi16 = _mm_unpackhi_epi8(r_values, zeros);
        let g_lo16 = _mm_unpacklo_epi8(g_values, zeros);
        let g_hi16 = _mm_unpackhi_epi8(g_values, zeros);
        let b_lo16 = _mm_unpacklo_epi8(b_values, zeros);
        let b_hi16 = _mm_unpackhi_epi8(b_values, zeros);

        let y_l = _mm_affine_dot::<PRECISION>(y_base, r_lo16, g_lo16, b_lo16, v_yr_yg, v_yb);
        let y_h = _mm_affine_dot::<PRECISION>(y_base, r_hi16, g_hi16, b_hi16, v_yr_yg, v_yb);

        let y_yuv = _mm_packus_epi16(y_l, y_h);

        _mm_storeu_si128(
            y_ptr.get_unchecked_mut(cx..).as_mut_ptr() as *mut __m128i,
            y_yuv,
        );

        cx += 16;
    }

    while cx + 8 < width {
        let px = cx * channels;
        let (r_values, g_values, b_values) =
            _mm_load_deinterleave_half_rgb_for_yuv::<ORIGIN_CHANNELS>(rgba_ptr.add(px));

        let r_lo16 = _mm_unpacklo_epi8(r_values, zeros);
        let g_lo16 = _mm_unpacklo_epi8(g_values, zeros);
        let b_lo16 = _mm_unpacklo_epi8(b_values, zeros);

        let y_l = _mm_affine_dot::<PRECISION>(y_base, r_lo16, g_lo16, b_lo16, v_yr_yg, v_yb);

        let y_yuv = _mm_packus_epi16(y_l, _mm_setzero_si128());

        std::ptr::copy_nonoverlapping(
            &y_yuv as *const _ as *const u8,
            y_ptr.get_unchecked_mut(cx..).as_mut_ptr(),
            8,
        );

        cx += 8;
    }

    cx
}
