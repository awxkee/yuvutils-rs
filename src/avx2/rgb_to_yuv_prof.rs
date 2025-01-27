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
    _mm256_affine_dot, _mm256_interleave_epi16, _mm256_load_deinterleave_rgb_for_yuv,
    avx2_pack_u16, avx_pairwise_avg_epi16_epi8_j,
};
use crate::internals::ProcessedOffset;
use crate::yuv_support::{
    CbCrForwardTransform, YuvChromaRange, YuvChromaSubsampling, YuvSourceChannels,
};
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

pub(crate) fn avx2_rgba_to_yuv_prof<
    const ORIGIN_CHANNELS: u8,
    const SAMPLING: u8,
    const PRECISION: i32,
>(
    transform: &CbCrForwardTransform<i32>,
    range: &YuvChromaRange,
    y_plane: &mut [u8],
    u_plane: &mut [u8],
    v_plane: &mut [u8],
    rgba: &[u8],
    start_cx: usize,
    start_ux: usize,
    width: usize,
) -> ProcessedOffset {
    unsafe {
        #[cfg(feature = "nightly_avx512")]
        if std::arch::is_x86_feature_detected!("avxvnni") {
            return avx2_rgba_to_yuv_dot::<ORIGIN_CHANNELS, SAMPLING, PRECISION>(
                transform, range, y_plane, u_plane, v_plane, rgba, start_cx, start_ux, width,
            );
        }
        avx2_rgba_to_yuv_def::<ORIGIN_CHANNELS, SAMPLING, PRECISION>(
            transform, range, y_plane, u_plane, v_plane, rgba, start_cx, start_ux, width,
        )
    }
}

#[inline(always)]
unsafe fn encode_32_part<
    const ORIGIN_CHANNELS: u8,
    const SAMPLING: u8,
    const PRECISION: i32,
    const HAS_DOT: bool,
>(
    src: &[u8],
    y_dst: &mut [u8],
    u_dst: &mut [u8],
    v_dst: &mut [u8],
    range: &YuvChromaRange,
    transform: &CbCrForwardTransform<i32>,
) {
    let chroma_subsampling: YuvChromaSubsampling = SAMPLING.into();

    let rounding_const_y = (1 << (PRECISION - 1)) - 1;
    let y_bias = _mm256_set1_epi32(range.bias_y as i32 * (1 << PRECISION) + rounding_const_y);
    let v_yr_yg = _mm256_set1_epi32(transform._interleaved_yr_yg());
    let v_yb = _mm256_set1_epi32(transform.yb);

    let precision_uv = match chroma_subsampling {
        YuvChromaSubsampling::Yuv420 | YuvChromaSubsampling::Yuv422 => PRECISION + 1,
        YuvChromaSubsampling::Yuv444 => PRECISION,
    };
    let rounding_const_uv = (1 << (precision_uv - 1)) - 1;

    let uv_bias = _mm256_set1_epi32(range.bias_uv as i32 * (1 << precision_uv) + rounding_const_uv);
    let v_cb_r_g = _mm256_set1_epi32(transform._interleaved_cbr_cbg());
    let v_cb_b = _mm256_set1_epi32(transform.cb_b);
    let v_cr_r_g = _mm256_set1_epi32(transform._interleaved_crr_crg());
    let v_cr_b = _mm256_set1_epi32(transform.cr_b);

    let (r_values, g_values, b_values) =
        _mm256_load_deinterleave_rgb_for_yuv::<ORIGIN_CHANNELS>(src.as_ptr());

    let rl0 = _mm256_unpacklo_epi8(r_values, _mm256_setzero_si256());
    let gl0 = _mm256_unpacklo_epi8(g_values, _mm256_setzero_si256());
    let bl0 = _mm256_unpacklo_epi8(b_values, _mm256_setzero_si256());

    let (rl_gl0, rl_gl1) = _mm256_interleave_epi16(rl0, gl0);
    let (b_lo0, b_lo1) = _mm256_interleave_epi16(bl0, _mm256_setzero_si256());

    let y00_vl = _mm256_affine_dot::<PRECISION, HAS_DOT>(
        y_bias, rl_gl0, rl_gl1, b_lo0, b_lo1, v_yr_yg, v_yb,
    );

    let rh0 = _mm256_unpackhi_epi8(r_values, _mm256_setzero_si256());
    let gh0 = _mm256_unpackhi_epi8(g_values, _mm256_setzero_si256());
    let bh0 = _mm256_unpackhi_epi8(b_values, _mm256_setzero_si256());

    let (rl_gh0, rl_gh1) = _mm256_interleave_epi16(rh0, gh0);
    let (b_h0, b_h1) = _mm256_interleave_epi16(bh0, _mm256_setzero_si256());

    let y01_vl =
        _mm256_affine_dot::<PRECISION, HAS_DOT>(y_bias, rl_gh0, rl_gh1, b_h0, b_h1, v_yr_yg, v_yb);

    let y0_values = _mm256_packus_epi16(y00_vl, y01_vl);
    _mm256_storeu_si256(y_dst.as_mut_ptr() as *mut __m256i, y0_values);

    if chroma_subsampling == YuvChromaSubsampling::Yuv444 {
        let cb_l = _mm256_affine_dot::<PRECISION, HAS_DOT>(
            uv_bias, rl_gl0, rl_gl1, b_lo0, b_lo1, v_cb_r_g, v_cb_b,
        );
        let cb_h = _mm256_affine_dot::<PRECISION, HAS_DOT>(
            uv_bias, rl_gh0, rl_gh1, b_h0, b_h1, v_cb_r_g, v_cb_b,
        );
        let cr_l = _mm256_affine_dot::<PRECISION, HAS_DOT>(
            uv_bias, rl_gl0, rl_gl1, b_lo0, b_lo1, v_cr_r_g, v_cr_b,
        );
        let cr_h = _mm256_affine_dot::<PRECISION, HAS_DOT>(
            uv_bias, rl_gh0, rl_gh1, b_h0, b_h1, v_cr_r_g, v_cr_b,
        );

        let cb = _mm256_packus_epi16(cb_l, cb_h);
        let cr = _mm256_packus_epi16(cr_l, cr_h);

        _mm256_storeu_si256(u_dst.as_mut_ptr() as *mut __m256i, cb);
        _mm256_storeu_si256(v_dst.as_mut_ptr() as *mut __m256i, cr);
    } else if chroma_subsampling == YuvChromaSubsampling::Yuv422
        || (chroma_subsampling == YuvChromaSubsampling::Yuv420)
    {
        let r1 = avx_pairwise_avg_epi16_epi8_j(r_values, 1);
        let g1 = avx_pairwise_avg_epi16_epi8_j(g_values, 1);
        let b1 = avx_pairwise_avg_epi16_epi8_j(b_values, 1);

        let (rhv0, rhv1) = _mm256_interleave_epi16(r1, g1);
        let (bhv0, bhv1) = _mm256_interleave_epi16(b1, _mm256_setzero_si256());

        let cb_s =
            _mm256_affine_dot::<16, HAS_DOT>(uv_bias, rhv0, rhv1, bhv0, bhv1, v_cb_r_g, v_cb_b);

        let cr_s =
            _mm256_affine_dot::<16, HAS_DOT>(uv_bias, rhv0, rhv1, bhv0, bhv1, v_cr_r_g, v_cr_b);

        let cb = avx2_pack_u16(cb_s, cb_s);
        let cr = avx2_pack_u16(cr_s, cr_s);

        _mm_storeu_si128(u_dst.as_mut_ptr() as *mut _, _mm256_castsi256_si128(cb));
        _mm_storeu_si128(v_dst.as_mut_ptr() as *mut _, _mm256_castsi256_si128(cr));
    }
}

#[target_feature(enable = "avx2")]
unsafe fn avx2_rgba_to_yuv_def<
    const ORIGIN_CHANNELS: u8,
    const SAMPLING: u8,
    const PRECISION: i32,
>(
    transform: &CbCrForwardTransform<i32>,
    range: &YuvChromaRange,
    y_plane: &mut [u8],
    u_plane: &mut [u8],
    v_plane: &mut [u8],
    rgba: &[u8],
    start_cx: usize,
    start_ux: usize,
    width: usize,
) -> ProcessedOffset {
    avx2_rgba_to_yuv_impl_prof::<ORIGIN_CHANNELS, SAMPLING, PRECISION, false>(
        transform, range, y_plane, u_plane, v_plane, rgba, start_cx, start_ux, width,
    )
}

#[cfg(feature = "nightly_avx512")]
#[target_feature(enable = "avx2", enable = "avxvnni")]
unsafe fn avx2_rgba_to_yuv_dot<
    const ORIGIN_CHANNELS: u8,
    const SAMPLING: u8,
    const PRECISION: i32,
>(
    transform: &CbCrForwardTransform<i32>,
    range: &YuvChromaRange,
    y_plane: &mut [u8],
    u_plane: &mut [u8],
    v_plane: &mut [u8],
    rgba: &[u8],
    start_cx: usize,
    start_ux: usize,
    width: usize,
) -> ProcessedOffset {
    avx2_rgba_to_yuv_impl_prof::<ORIGIN_CHANNELS, SAMPLING, PRECISION, true>(
        transform, range, y_plane, u_plane, v_plane, rgba, start_cx, start_ux, width,
    )
}

#[inline(always)]
unsafe fn avx2_rgba_to_yuv_impl_prof<
    const ORIGIN_CHANNELS: u8,
    const SAMPLING: u8,
    const PRECISION: i32,
    const HAS_DOT: bool,
>(
    transform: &CbCrForwardTransform<i32>,
    range: &YuvChromaRange,
    y_plane: &mut [u8],
    u_plane: &mut [u8],
    v_plane: &mut [u8],
    rgba: &[u8],
    start_cx: usize,
    start_ux: usize,
    width: usize,
) -> ProcessedOffset {
    let chroma_subsampling: YuvChromaSubsampling = SAMPLING.into();
    let source_channels: YuvSourceChannels = ORIGIN_CHANNELS.into();
    let channels = source_channels.get_channels_count();

    let mut cx = start_cx;
    let mut uv_x = start_ux;

    while cx + 32 < width {
        let px = cx * channels;

        encode_32_part::<ORIGIN_CHANNELS, SAMPLING, PRECISION, HAS_DOT>(
            rgba.get_unchecked(px..),
            y_plane.get_unchecked_mut(cx..),
            u_plane.get_unchecked_mut(uv_x..),
            v_plane.get_unchecked_mut(uv_x..),
            range,
            transform,
        );

        if chroma_subsampling == YuvChromaSubsampling::Yuv444 {
            uv_x += 32;
        } else if chroma_subsampling == YuvChromaSubsampling::Yuv422
            || (chroma_subsampling == YuvChromaSubsampling::Yuv420)
        {
            uv_x += 16;
        }

        cx += 32;
    }

    if cx < width {
        let diff = width - cx;
        assert!(diff <= 32);

        let mut src_buffer: [u8; 32 * 4] = [0; 32 * 4];
        let mut y_buffer0: [u8; 32] = [0; 32];
        let mut u_buffer: [u8; 32] = [0; 32];
        let mut v_buffer: [u8; 32] = [0; 32];

        std::ptr::copy_nonoverlapping(
            rgba.get_unchecked(cx * channels..).as_ptr(),
            src_buffer.as_mut_ptr(),
            diff * channels,
        );

        // Replicate last item to one more position for subsampling
        if chroma_subsampling != YuvChromaSubsampling::Yuv444 && diff % 2 != 0 {
            let lst = (width - 1) * channels;
            let last_items = rgba.get_unchecked(lst..(lst + channels));
            let dvb = diff * channels;
            let dst = src_buffer.get_unchecked_mut(dvb..(dvb + channels));
            for (dst, src) in dst.iter_mut().zip(last_items) {
                *dst = *src;
            }
        }

        encode_32_part::<ORIGIN_CHANNELS, SAMPLING, PRECISION, HAS_DOT>(
            src_buffer.as_slice(),
            y_buffer0.as_mut_slice(),
            u_buffer.as_mut_slice(),
            v_buffer.as_mut_slice(),
            range,
            transform,
        );

        std::ptr::copy_nonoverlapping(
            y_buffer0.as_mut_ptr(),
            y_plane.get_unchecked_mut(cx..).as_mut_ptr(),
            diff,
        );

        let ux_size = match chroma_subsampling {
            YuvChromaSubsampling::Yuv420 | YuvChromaSubsampling::Yuv422 => diff.div_ceil(2),
            YuvChromaSubsampling::Yuv444 => diff,
        };

        std::ptr::copy_nonoverlapping(
            u_buffer.as_mut_ptr(),
            u_plane.get_unchecked_mut(uv_x..).as_mut_ptr(),
            ux_size,
        );
        std::ptr::copy_nonoverlapping(
            v_buffer.as_mut_ptr(),
            v_plane.get_unchecked_mut(uv_x..).as_mut_ptr(),
            ux_size,
        );

        cx += diff;
        uv_x += ux_size;
    }

    ProcessedOffset { cx, ux: uv_x }
}
