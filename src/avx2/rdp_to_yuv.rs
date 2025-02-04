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
    _mm256_deinterleave_rgba_epi8, _mm256_double_affine_uv_dot, _mm256_double_affine_uv_s_dot,
    avx2_deinterleave_rgb,
};
use crate::internals::ProcessedOffset;
use crate::rdp::RdpChannels;
use crate::yuv_support::CbCrForwardTransform;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

pub(crate) fn rdp_avx2_rgba_to_yuv<const ORIGIN_CHANNELS: u8, const Q: i32>(
    transform: &CbCrForwardTransform<i32>,
    y_plane: &mut [i16],
    u_plane: &mut [i16],
    v_plane: &mut [i16],
    rgba: &[u8],
    width: usize,
) -> ProcessedOffset {
    unsafe {
        rdp_avx2_rgba_to_yuv_impl::<ORIGIN_CHANNELS, Q>(
            transform, y_plane, u_plane, v_plane, rgba, width,
        )
    }
}

#[inline(always)]
unsafe fn _mm256_load_rdp_deinterleave_rgb_for_yuv<const ORIGINS: u8>(
    ptr: *const u8,
) -> (__m256i, __m256i, __m256i) {
    let source_channels: RdpChannels = ORIGINS.into();

    let (r_values, g_values, b_values);

    let row_1 = _mm256_loadu_si256(ptr as *const __m256i);
    let row_2 = _mm256_loadu_si256(ptr.add(32) as *const __m256i);
    let row_3 = _mm256_loadu_si256(ptr.add(64) as *const __m256i);

    match source_channels {
        RdpChannels::Rgb | RdpChannels::Bgr => {
            let (it1, it2, it3) = avx2_deinterleave_rgb(row_1, row_2, row_3);
            if source_channels == RdpChannels::Rgb {
                r_values = it1;
                g_values = it2;
                b_values = it3;
            } else {
                r_values = it3;
                g_values = it2;
                b_values = it1;
            }
        }
        RdpChannels::Rgba | RdpChannels::Bgra => {
            let row_4 = _mm256_loadu_si256(ptr.add(96) as *const __m256i);

            let (it1, it2, it3, _) = _mm256_deinterleave_rgba_epi8(row_1, row_2, row_3, row_4);
            if source_channels == RdpChannels::Rgba {
                r_values = it1;
                g_values = it2;
                b_values = it3;
            } else {
                r_values = it3;
                g_values = it2;
                b_values = it1;
            }
        }
        RdpChannels::Abgr => {
            let row_4 = _mm256_loadu_si256(ptr.add(96) as *const __m256i);

            let (_, b, g, r) = _mm256_deinterleave_rgba_epi8(row_1, row_2, row_3, row_4);
            r_values = r;
            g_values = g;
            b_values = b;
        }
        RdpChannels::Argb => {
            let row_4 = _mm256_loadu_si256(ptr.add(96) as *const __m256i);

            let (_, r, g, b) = _mm256_deinterleave_rgba_epi8(row_1, row_2, row_3, row_4);
            r_values = r;
            g_values = g;
            b_values = b;
        }
    }

    (r_values, g_values, b_values)
}

#[target_feature(enable = "avx2")]
unsafe fn rdp_avx2_rgba_to_yuv_impl<const ORIGIN_CHANNELS: u8, const Q: i32>(
    transform: &CbCrForwardTransform<i32>,
    y_plane: &mut [i16],
    u_plane: &mut [i16],
    v_plane: &mut [i16],
    rgba: &[u8],
    width: usize,
) -> ProcessedOffset {
    let source_channels: RdpChannels = ORIGIN_CHANNELS.into();
    let channels = source_channels.get_channels_count();

    let src_ptr = rgba;
    let uv_bias = _mm256_setzero_si256();
    let v_cbr_cbg = _mm256_set1_epi32(transform._interleaved_cbr_cbg());
    let v_cb_b = _mm256_set1_epi16(transform.cb_b as i16);
    let v_crr_vcrg = _mm256_set1_epi32(transform._interleaved_crr_crg());
    let v_cr_b = _mm256_set1_epi16(transform.cr_b as i16);
    let j_y_bias = _mm256_set1_epi16(4096);

    let mut cx = 0;
    let mut ux = 0;

    while cx + 32 < width {
        let src_ptr = src_ptr.get_unchecked(cx * channels..);

        let (r_values, g_values, b_values) =
            _mm256_load_rdp_deinterleave_rgb_for_yuv::<ORIGIN_CHANNELS>(src_ptr.as_ptr());

        let r_lo = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(r_values));
        let g_lo = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(g_values));
        let b_lo = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(b_values));
        let r_hi = _mm256_cvtepu8_epi16(_mm256_extracti128_si256::<1>(r_values));
        let g_hi = _mm256_cvtepu8_epi16(_mm256_extracti128_si256::<1>(g_values));
        let b_hi = _mm256_cvtepu8_epi16(_mm256_extracti128_si256::<1>(b_values));

        let (r_g_lo0, r_g_hi0) = (
            _mm256_unpacklo_epi16(r_lo, g_lo),
            _mm256_unpackhi_epi16(r_lo, g_lo),
        );
        let b_hi0 = _mm256_unpackhi_epi16(b_lo, _mm256_setzero_si256());
        let b_lo0 = _mm256_unpacklo_epi16(b_lo, _mm256_setzero_si256());

        let (r_g_lo1, r_g_hi1) = (
            _mm256_unpacklo_epi16(r_hi, g_hi),
            _mm256_unpackhi_epi16(r_hi, g_hi),
        );
        let b_hi1 = _mm256_unpackhi_epi16(b_hi, _mm256_setzero_si256());
        let b_lo1 = _mm256_unpacklo_epi16(b_hi, _mm256_setzero_si256());

        let v_yr_yg = _mm256_set1_epi32(transform._interleaved_yr_yg());
        let v_yb = _mm256_set1_epi16(transform.yb as i16);

        let (mut y_vl0, mut y_vl1) = _mm256_double_affine_uv_s_dot::<Q>(
            r_g_lo0, r_g_hi0, r_g_lo1, r_g_hi1, b_lo0, b_hi0, b_lo1, b_hi1, v_yr_yg, v_yb,
        );

        y_vl0 = _mm256_sub_epi16(y_vl0, j_y_bias);
        y_vl1 = _mm256_sub_epi16(y_vl1, j_y_bias);

        _mm256_storeu_si256(
            y_plane.get_unchecked_mut(cx..).as_mut_ptr() as *mut __m256i,
            y_vl0,
        );
        _mm256_storeu_si256(
            y_plane.get_unchecked_mut((cx + 16)..).as_mut_ptr() as *mut __m256i,
            y_vl1,
        );

        let cb_vl = _mm256_double_affine_uv_dot::<Q>(
            uv_bias, r_g_lo0, r_g_hi0, r_g_lo1, r_g_hi1, b_lo0, b_hi0, b_lo1, b_hi1, v_cbr_cbg,
            v_cb_b,
        );

        _mm256_storeu_si256(
            u_plane.get_unchecked_mut(ux..).as_mut_ptr() as *mut __m256i,
            cb_vl.0,
        );
        _mm256_storeu_si256(
            u_plane.get_unchecked_mut((ux + 16)..).as_mut_ptr() as *mut __m256i,
            cb_vl.1,
        );

        let cr_vl = _mm256_double_affine_uv_dot::<Q>(
            uv_bias, r_g_lo0, r_g_hi0, r_g_lo1, r_g_hi1, b_lo0, b_hi0, b_lo1, b_hi1, v_crr_vcrg,
            v_cr_b,
        );

        _mm256_storeu_si256(
            v_plane.get_unchecked_mut(ux..).as_mut_ptr() as *mut __m256i,
            cr_vl.0,
        );

        _mm256_storeu_si256(
            v_plane.get_unchecked_mut((ux + 16)..).as_mut_ptr() as *mut __m256i,
            cr_vl.1,
        );

        ux += 32;
        cx += 32;
    }

    if cx < width {
        let diff = width - cx;
        assert!(diff <= 32);
        let mut src_buffer: [u8; 32 * 4] = [0; 32 * 4];
        let mut y_buffer: [i16; 32] = [0; 32];
        let mut u_buffer: [i16; 32] = [0; 32];
        let mut v_buffer: [i16; 32] = [0; 32];

        std::ptr::copy_nonoverlapping(
            rgba.get_unchecked(cx * channels..).as_ptr(),
            src_buffer.as_mut_ptr(),
            diff * channels,
        );

        let (r_values, g_values, b_values) =
            _mm256_load_rdp_deinterleave_rgb_for_yuv::<ORIGIN_CHANNELS>(src_buffer.as_ptr());

        let r_lo = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(r_values));
        let g_lo = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(g_values));
        let b_lo = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(b_values));
        let r_hi = _mm256_cvtepu8_epi16(_mm256_extracti128_si256::<1>(r_values));
        let g_hi = _mm256_cvtepu8_epi16(_mm256_extracti128_si256::<1>(g_values));
        let b_hi = _mm256_cvtepu8_epi16(_mm256_extracti128_si256::<1>(b_values));

        let (r_g_lo0, r_g_hi0) = (
            _mm256_unpacklo_epi16(r_lo, g_lo),
            _mm256_unpackhi_epi16(r_lo, g_lo),
        );
        let b_hi0 = _mm256_unpackhi_epi16(b_lo, _mm256_setzero_si256());
        let b_lo0 = _mm256_unpacklo_epi16(b_lo, _mm256_setzero_si256());

        let (r_g_lo1, r_g_hi1) = (
            _mm256_unpacklo_epi16(r_hi, g_hi),
            _mm256_unpackhi_epi16(r_hi, g_hi),
        );
        let b_hi1 = _mm256_unpackhi_epi16(b_hi, _mm256_setzero_si256());
        let b_lo1 = _mm256_unpacklo_epi16(b_hi, _mm256_setzero_si256());

        let v_yr_yg = _mm256_set1_epi32(transform._interleaved_yr_yg());
        let v_yb = _mm256_set1_epi16(transform.yb as i16);

        let (mut y_vl0, mut y_vl1) = _mm256_double_affine_uv_s_dot::<Q>(
            r_g_lo0, r_g_hi0, r_g_lo1, r_g_hi1, b_lo0, b_hi0, b_lo1, b_hi1, v_yr_yg, v_yb,
        );

        y_vl0 = _mm256_sub_epi16(y_vl0, j_y_bias);
        y_vl1 = _mm256_sub_epi16(y_vl1, j_y_bias);

        _mm256_storeu_si256(y_buffer.as_mut_ptr() as *mut __m256i, y_vl0);
        _mm256_storeu_si256(
            y_buffer.get_unchecked_mut(16..).as_mut_ptr() as *mut __m256i,
            y_vl1,
        );

        let cb_vl = _mm256_double_affine_uv_dot::<Q>(
            uv_bias, r_g_lo0, r_g_hi0, r_g_lo1, r_g_hi1, b_lo0, b_hi0, b_lo1, b_hi1, v_cbr_cbg,
            v_cb_b,
        );

        _mm256_storeu_si256(u_buffer.as_mut_ptr() as *mut __m256i, cb_vl.0);
        _mm256_storeu_si256(
            u_buffer.get_unchecked_mut(16..).as_mut_ptr() as *mut __m256i,
            cb_vl.1,
        );

        let cr_vl = _mm256_double_affine_uv_dot::<Q>(
            uv_bias, r_g_lo0, r_g_hi0, r_g_lo1, r_g_hi1, b_lo0, b_hi0, b_lo1, b_hi1, v_crr_vcrg,
            v_cr_b,
        );

        _mm256_storeu_si256(v_buffer.as_mut_ptr() as *mut __m256i, cr_vl.0);

        _mm256_storeu_si256(
            v_buffer.get_unchecked_mut(16..).as_mut_ptr() as *mut __m256i,
            cr_vl.1,
        );

        std::ptr::copy_nonoverlapping(
            u_buffer.as_ptr(),
            u_plane.get_unchecked_mut(ux..).as_mut_ptr(),
            diff,
        );
        std::ptr::copy_nonoverlapping(
            v_buffer.as_ptr(),
            v_plane.get_unchecked_mut(ux..).as_mut_ptr(),
            diff,
        );

        std::ptr::copy_nonoverlapping(
            y_buffer.as_ptr(),
            y_plane.get_unchecked_mut(cx..).as_mut_ptr(),
            diff,
        );

        cx += diff;
        ux += diff;
    }

    ProcessedOffset { ux, cx }
}
