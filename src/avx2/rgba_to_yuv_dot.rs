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

use crate::avx2::avx2_utils::{avx2_pack_u16, shuffle};
use crate::internals::ProcessedOffset;
use crate::yuv_support::{
    CbCrForwardTransform, YuvChromaRange, YuvChromaSubsampling, YuvSourceChannels,
};
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

pub(crate) fn avx2_rgba_to_yuv_dot_rgba<const ORIGIN_CHANNELS: u8, const SAMPLING: u8>(
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
        avx2_rgba_to_yuv_dot_rgba_impl_ubs::<ORIGIN_CHANNELS, SAMPLING>(
            transform, range, y_plane, u_plane, v_plane, rgba, start_cx, start_ux, width,
        )
    }
}

#[target_feature(enable = "avx2")]
unsafe fn avx2_rgba_to_yuv_dot_rgba_impl_ubs<const ORIGIN_CHANNELS: u8, const SAMPLING: u8>(
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
    assert!(
        source_channels == YuvSourceChannels::Rgba || source_channels == YuvSourceChannels::Bgra
    );
    let channels = source_channels.get_channels_count();

    let y_ptr = y_plane;
    let u_ptr = u_plane;
    let v_ptr = v_plane;

    const A_E: i32 = 7;
    let y_bias = _mm256_set1_epi16(range.bias_y as i16 * (1 << A_E) + (1 << (A_E - 1)) - 1);
    let uv_bias = _mm256_set1_epi16(range.bias_uv as i16 * (1 << A_E) + (1 << (A_E - 1)) - 1);

    let y_weights = if source_channels == YuvSourceChannels::Rgba {
        _mm256_setr_epi8(
            transform.yr as i8,
            transform.yg as i8,
            transform.yb as i8,
            0,
            transform.yr as i8,
            transform.yg as i8,
            transform.yb as i8,
            0,
            transform.yr as i8,
            transform.yg as i8,
            transform.yb as i8,
            0,
            transform.yr as i8,
            transform.yg as i8,
            transform.yb as i8,
            0,
            transform.yr as i8,
            transform.yg as i8,
            transform.yb as i8,
            0,
            transform.yr as i8,
            transform.yg as i8,
            transform.yb as i8,
            0,
            transform.yr as i8,
            transform.yg as i8,
            transform.yb as i8,
            0,
            transform.yr as i8,
            transform.yg as i8,
            transform.yb as i8,
            0,
        )
    } else {
        _mm256_setr_epi8(
            transform.yb as i8,
            transform.yg as i8,
            transform.yr as i8,
            0,
            transform.yb as i8,
            transform.yg as i8,
            transform.yr as i8,
            0,
            transform.yb as i8,
            transform.yg as i8,
            transform.yr as i8,
            0,
            transform.yb as i8,
            transform.yg as i8,
            transform.yr as i8,
            0,
            transform.yb as i8,
            transform.yg as i8,
            transform.yr as i8,
            0,
            transform.yb as i8,
            transform.yg as i8,
            transform.yr as i8,
            0,
            transform.yb as i8,
            transform.yg as i8,
            transform.yr as i8,
            0,
            transform.yb as i8,
            transform.yg as i8,
            transform.yr as i8,
            0,
        )
    };
    let cb_weights = if source_channels == YuvSourceChannels::Rgba {
        _mm256_setr_epi8(
            transform.cb_r as i8,
            transform.cb_g as i8,
            transform.cb_b as i8,
            0,
            transform.cb_r as i8,
            transform.cb_g as i8,
            transform.cb_b as i8,
            0,
            transform.cb_r as i8,
            transform.cb_g as i8,
            transform.cb_b as i8,
            0,
            transform.cb_r as i8,
            transform.cb_g as i8,
            transform.cb_b as i8,
            0,
            transform.cb_r as i8,
            transform.cb_g as i8,
            transform.cb_b as i8,
            0,
            transform.cb_r as i8,
            transform.cb_g as i8,
            transform.cb_b as i8,
            0,
            transform.cb_r as i8,
            transform.cb_g as i8,
            transform.cb_b as i8,
            0,
            transform.cb_r as i8,
            transform.cb_g as i8,
            transform.cb_b as i8,
            0,
        )
    } else {
        _mm256_setr_epi8(
            transform.cb_b as i8,
            transform.cb_g as i8,
            transform.cb_r as i8,
            0,
            transform.cb_b as i8,
            transform.cb_g as i8,
            transform.cb_r as i8,
            0,
            transform.cb_b as i8,
            transform.cb_g as i8,
            transform.cb_r as i8,
            0,
            transform.cb_b as i8,
            transform.cb_g as i8,
            transform.cb_r as i8,
            0,
            transform.cb_b as i8,
            transform.cb_g as i8,
            transform.cb_r as i8,
            0,
            transform.cb_b as i8,
            transform.cb_g as i8,
            transform.cb_r as i8,
            0,
            transform.cb_b as i8,
            transform.cb_g as i8,
            transform.cb_r as i8,
            0,
            transform.cb_b as i8,
            transform.cb_g as i8,
            transform.cb_r as i8,
            0,
        )
    };
    let cr_weights = if source_channels == YuvSourceChannels::Rgba {
        _mm256_setr_epi8(
            transform.cr_r as i8,
            transform.cr_g as i8,
            transform.cr_b as i8,
            0,
            transform.cr_r as i8,
            transform.cr_g as i8,
            transform.cr_b as i8,
            0,
            transform.cr_r as i8,
            transform.cr_g as i8,
            transform.cr_b as i8,
            0,
            transform.cr_r as i8,
            transform.cr_g as i8,
            transform.cr_b as i8,
            0,
            transform.cr_r as i8,
            transform.cr_g as i8,
            transform.cr_b as i8,
            0,
            transform.cr_r as i8,
            transform.cr_g as i8,
            transform.cr_b as i8,
            0,
            transform.cr_r as i8,
            transform.cr_g as i8,
            transform.cr_b as i8,
            0,
            transform.cr_r as i8,
            transform.cr_g as i8,
            transform.cr_b as i8,
            0,
        )
    } else {
        _mm256_setr_epi8(
            transform.cr_b as i8,
            transform.cr_g as i8,
            transform.cr_r as i8,
            0,
            transform.cr_b as i8,
            transform.cr_g as i8,
            transform.cr_r as i8,
            0,
            transform.cr_b as i8,
            transform.cr_g as i8,
            transform.cr_r as i8,
            0,
            transform.cr_b as i8,
            transform.cr_g as i8,
            transform.cr_r as i8,
            0,
            transform.cr_b as i8,
            transform.cr_g as i8,
            transform.cr_r as i8,
            0,
            transform.cr_b as i8,
            transform.cr_g as i8,
            transform.cr_r as i8,
            0,
            transform.cr_b as i8,
            transform.cr_g as i8,
            transform.cr_r as i8,
            0,
            transform.cr_b as i8,
            transform.cr_g as i8,
            transform.cr_r as i8,
            0,
        )
    };

    let v422_shuffle = _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7);

    let mut cx = start_cx;
    let mut ux = start_ux;

    while cx + 32 < width {
        let src = rgba.get_unchecked(cx * channels..).as_ptr();

        let v0 = _mm256_loadu_si256(src as *const __m256i);
        let v1 = _mm256_loadu_si256(src.add(32) as *const __m256i);
        let v2 = _mm256_loadu_si256(src.add(64) as *const __m256i);
        let v3 = _mm256_loadu_si256(src.add(96) as *const __m256i);

        let y0s = _mm256_maddubs_epi16(v0, y_weights);
        let y1s = _mm256_maddubs_epi16(v1, y_weights);
        let y2s = _mm256_maddubs_epi16(v2, y_weights);
        let y3s = _mm256_maddubs_epi16(v3, y_weights);

        let v0_s = if chroma_subsampling != YuvChromaSubsampling::Yuv444 {
            _mm256_permutevar8x32_epi32(v0, v422_shuffle)
        } else {
            _mm256_setzero_si256()
        };
        let v1_s = if chroma_subsampling != YuvChromaSubsampling::Yuv444 {
            _mm256_permutevar8x32_epi32(v1, v422_shuffle)
        } else {
            _mm256_setzero_si256()
        };
        let v2_s = if chroma_subsampling != YuvChromaSubsampling::Yuv444 {
            _mm256_permutevar8x32_epi32(v2, v422_shuffle)
        } else {
            _mm256_setzero_si256()
        };
        let v3_s = if chroma_subsampling != YuvChromaSubsampling::Yuv444 {
            _mm256_permutevar8x32_epi32(v3, v422_shuffle)
        } else {
            _mm256_setzero_si256()
        };

        const MASK: i32 = shuffle(3, 1, 2, 0);

        let mut y0m = _mm256_hadd_epi16(y0s, y1s);
        let mut y1m = _mm256_hadd_epi16(y2s, y3s);

        y0m = _mm256_add_epi16(y0m, y_bias);
        y1m = _mm256_add_epi16(y1m, y_bias);

        y0m = _mm256_srai_epi16::<A_E>(y0m);
        y1m = _mm256_srai_epi16::<A_E>(y1m);

        y0m = _mm256_permute4x64_epi64::<MASK>(y0m);
        y1m = _mm256_permute4x64_epi64::<MASK>(y1m);

        let y_vl = avx2_pack_u16(y0m, y1m);

        _mm256_storeu_si256(y_ptr.get_unchecked_mut(cx..).as_mut_ptr() as *mut _, y_vl);

        if chroma_subsampling == YuvChromaSubsampling::Yuv444 {
            let cb0 = _mm256_maddubs_epi16(v0, cb_weights);
            let cb1 = _mm256_maddubs_epi16(v1, cb_weights);
            let cb2 = _mm256_maddubs_epi16(v2, cb_weights);
            let cb3 = _mm256_maddubs_epi16(v3, cb_weights);

            let cr0 = _mm256_maddubs_epi16(v0, cr_weights);
            let cr1 = _mm256_maddubs_epi16(v1, cr_weights);
            let cr2 = _mm256_maddubs_epi16(v2, cr_weights);
            let cr3 = _mm256_maddubs_epi16(v3, cr_weights);

            let mut cb00 = _mm256_hadd_epi16(cb0, cb1);
            let mut cb01 = _mm256_hadd_epi16(cb2, cb3);

            let mut cr00 = _mm256_hadd_epi16(cr0, cr1);
            let mut cr01 = _mm256_hadd_epi16(cr2, cr3);

            cb00 = _mm256_add_epi16(cb00, uv_bias);
            cb01 = _mm256_add_epi16(cb01, uv_bias);
            cr00 = _mm256_add_epi16(cr00, uv_bias);
            cr01 = _mm256_add_epi16(cr01, uv_bias);

            cb00 = _mm256_srai_epi16::<A_E>(cb00);
            cb01 = _mm256_srai_epi16::<A_E>(cb01);
            cr00 = _mm256_srai_epi16::<A_E>(cr00);
            cr01 = _mm256_srai_epi16::<A_E>(cr01);

            cb00 = _mm256_permute4x64_epi64::<MASK>(cb00);
            cb01 = _mm256_permute4x64_epi64::<MASK>(cb01);
            cr00 = _mm256_permute4x64_epi64::<MASK>(cr00);
            cr01 = _mm256_permute4x64_epi64::<MASK>(cr01);

            let cb_vl = avx2_pack_u16(cb00, cb01);
            let cr_vl = avx2_pack_u16(cr00, cr01);

            _mm256_storeu_si256(u_ptr.get_unchecked_mut(ux..).as_mut_ptr() as *mut _, cb_vl);
            _mm256_storeu_si256(v_ptr.get_unchecked_mut(ux..).as_mut_ptr() as *mut _, cr_vl);

            ux += 32;
        } else if (chroma_subsampling == YuvChromaSubsampling::Yuv420)
            || (chroma_subsampling == YuvChromaSubsampling::Yuv422)
        {
            let h0 = _mm256_extracti128_si256::<1>(v0_s);
            let h1 = _mm256_extracti128_si256::<1>(v1_s);
            let h2 = _mm256_extracti128_si256::<1>(v2_s);
            let h3 = _mm256_extracti128_si256::<1>(v3_s);

            let vh0 = _mm_avg_epu8(_mm256_castsi256_si128(v0_s), h0);
            let vh1 = _mm_avg_epu8(_mm256_castsi256_si128(v1_s), h1);
            let vh2 = _mm_avg_epu8(_mm256_castsi256_si128(v2_s), h2);
            let vh3 = _mm_avg_epu8(_mm256_castsi256_si128(v3_s), h3);

            let v0_f = _mm256_set_m128i(vh1, vh0);
            let v1_f = _mm256_set_m128i(vh3, vh2);

            let cb0 = _mm256_maddubs_epi16(v0_f, cb_weights);
            let cb1 = _mm256_maddubs_epi16(v1_f, cb_weights);

            let cr0 = _mm256_maddubs_epi16(v0_f, cr_weights);
            let cr1 = _mm256_maddubs_epi16(v1_f, cr_weights);

            let mut cb00 = _mm256_hadd_epi16(cb0, cb1);
            let mut cr00 = _mm256_hadd_epi16(cr0, cr1);

            cb00 = _mm256_add_epi16(cb00, uv_bias);
            cr00 = _mm256_add_epi16(cr00, uv_bias);

            cb00 = _mm256_srai_epi16::<A_E>(cb00);
            cr00 = _mm256_srai_epi16::<A_E>(cr00);

            cb00 = _mm256_permute4x64_epi64::<MASK>(cb00);
            cr00 = _mm256_permute4x64_epi64::<MASK>(cr00);

            let cb_vl = avx2_pack_u16(cb00, cb00);
            let cr_vl = avx2_pack_u16(cr00, cr00);

            _mm_storeu_si128(
                u_ptr.get_unchecked_mut(ux..).as_mut_ptr() as *mut _,
                _mm256_castsi256_si128(cb_vl),
            );
            _mm_storeu_si128(
                v_ptr.get_unchecked_mut(ux..).as_mut_ptr() as *mut _,
                _mm256_castsi256_si128(cr_vl),
            );

            ux += 16;
        }

        cx += 32;
    }

    if cx < width {
        let diff = width - cx;
        assert!(diff <= 32);

        let mut src_buffer: [u8; 32 * 4] = [0; 32 * 4];
        let mut y_buffer: [u8; 32] = [0; 32];
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

        let v0 = _mm256_loadu_si256(src_buffer.as_ptr() as *const __m256i);
        let v1 = _mm256_loadu_si256(src_buffer.as_ptr().add(32) as *const __m256i);
        let v2 = _mm256_loadu_si256(src_buffer.as_ptr().add(64) as *const __m256i);
        let v3 = _mm256_loadu_si256(src_buffer.as_ptr().add(96) as *const __m256i);

        let y0s = _mm256_maddubs_epi16(v0, y_weights);
        let y1s = _mm256_maddubs_epi16(v1, y_weights);
        let y2s = _mm256_maddubs_epi16(v2, y_weights);
        let y3s = _mm256_maddubs_epi16(v3, y_weights);

        let v0_s = if chroma_subsampling != YuvChromaSubsampling::Yuv444 {
            _mm256_permutevar8x32_epi32(v0, v422_shuffle)
        } else {
            _mm256_setzero_si256()
        };
        let v1_s = if chroma_subsampling != YuvChromaSubsampling::Yuv444 {
            _mm256_permutevar8x32_epi32(v1, v422_shuffle)
        } else {
            _mm256_setzero_si256()
        };
        let v2_s = if chroma_subsampling != YuvChromaSubsampling::Yuv444 {
            _mm256_permutevar8x32_epi32(v2, v422_shuffle)
        } else {
            _mm256_setzero_si256()
        };
        let v3_s = if chroma_subsampling != YuvChromaSubsampling::Yuv444 {
            _mm256_permutevar8x32_epi32(v3, v422_shuffle)
        } else {
            _mm256_setzero_si256()
        };

        const MASK: i32 = shuffle(3, 1, 2, 0);

        let mut y0m = _mm256_hadd_epi16(y0s, y1s);
        let mut y1m = _mm256_hadd_epi16(y2s, y3s);

        y0m = _mm256_add_epi16(y0m, y_bias);
        y1m = _mm256_add_epi16(y1m, y_bias);

        y0m = _mm256_srai_epi16::<A_E>(y0m);
        y1m = _mm256_srai_epi16::<A_E>(y1m);

        y0m = _mm256_permute4x64_epi64::<MASK>(y0m);
        y1m = _mm256_permute4x64_epi64::<MASK>(y1m);

        let y_vl = avx2_pack_u16(y0m, y1m);

        _mm256_storeu_si256(y_buffer.as_mut_ptr() as *mut _, y_vl);

        if chroma_subsampling == YuvChromaSubsampling::Yuv444 {
            let cb0 = _mm256_maddubs_epi16(v0, cb_weights);
            let cb1 = _mm256_maddubs_epi16(v1, cb_weights);
            let cb2 = _mm256_maddubs_epi16(v2, cb_weights);
            let cb3 = _mm256_maddubs_epi16(v3, cb_weights);

            let cr0 = _mm256_maddubs_epi16(v0, cr_weights);
            let cr1 = _mm256_maddubs_epi16(v1, cr_weights);
            let cr2 = _mm256_maddubs_epi16(v2, cr_weights);
            let cr3 = _mm256_maddubs_epi16(v3, cr_weights);

            let mut cb00 = _mm256_hadd_epi16(cb0, cb1);
            let mut cb01 = _mm256_hadd_epi16(cb2, cb3);

            let mut cr00 = _mm256_hadd_epi16(cr0, cr1);
            let mut cr01 = _mm256_hadd_epi16(cr2, cr3);

            cb00 = _mm256_add_epi16(cb00, uv_bias);
            cb01 = _mm256_add_epi16(cb01, uv_bias);
            cr00 = _mm256_add_epi16(cr00, uv_bias);
            cr01 = _mm256_add_epi16(cr01, uv_bias);

            cb00 = _mm256_srai_epi16::<A_E>(cb00);
            cb01 = _mm256_srai_epi16::<A_E>(cb01);
            cr00 = _mm256_srai_epi16::<A_E>(cr00);
            cr01 = _mm256_srai_epi16::<A_E>(cr01);

            cb00 = _mm256_permute4x64_epi64::<MASK>(cb00);
            cb01 = _mm256_permute4x64_epi64::<MASK>(cb01);
            cr00 = _mm256_permute4x64_epi64::<MASK>(cr00);
            cr01 = _mm256_permute4x64_epi64::<MASK>(cr01);

            let cb_vl = avx2_pack_u16(cb00, cb01);
            let cr_vl = avx2_pack_u16(cr00, cr01);

            _mm256_storeu_si256(u_buffer.as_mut_ptr() as *mut _, cb_vl);
            _mm256_storeu_si256(v_buffer.as_mut_ptr() as *mut _, cr_vl);
        } else if (chroma_subsampling == YuvChromaSubsampling::Yuv420)
            || (chroma_subsampling == YuvChromaSubsampling::Yuv422)
        {
            let h0 = _mm256_extracti128_si256::<1>(v0_s);
            let h1 = _mm256_extracti128_si256::<1>(v1_s);
            let h2 = _mm256_extracti128_si256::<1>(v2_s);
            let h3 = _mm256_extracti128_si256::<1>(v3_s);

            let vh0 = _mm_avg_epu8(_mm256_castsi256_si128(v0_s), h0);
            let vh1 = _mm_avg_epu8(_mm256_castsi256_si128(v1_s), h1);
            let vh2 = _mm_avg_epu8(_mm256_castsi256_si128(v2_s), h2);
            let vh3 = _mm_avg_epu8(_mm256_castsi256_si128(v3_s), h3);

            let v0_f = _mm256_set_m128i(vh1, vh0);
            let v1_f = _mm256_set_m128i(vh3, vh2);

            let cb0 = _mm256_maddubs_epi16(v0_f, cb_weights);
            let cb1 = _mm256_maddubs_epi16(v1_f, cb_weights);

            let cr0 = _mm256_maddubs_epi16(v0_f, cr_weights);
            let cr1 = _mm256_maddubs_epi16(v1_f, cr_weights);

            let mut cb00 = _mm256_hadd_epi16(cb0, cb1);
            let mut cr00 = _mm256_hadd_epi16(cr0, cr1);

            cb00 = _mm256_add_epi16(cb00, uv_bias);
            cr00 = _mm256_add_epi16(cr00, uv_bias);

            cb00 = _mm256_srai_epi16::<A_E>(cb00);
            cr00 = _mm256_srai_epi16::<A_E>(cr00);

            cb00 = _mm256_permute4x64_epi64::<MASK>(cb00);
            cr00 = _mm256_permute4x64_epi64::<MASK>(cr00);

            let cb_vl = avx2_pack_u16(cb00, cb00);
            let cr_vl = avx2_pack_u16(cr00, cr00);

            _mm_storeu_si128(
                u_buffer.as_mut_ptr() as *mut _,
                _mm256_castsi256_si128(cb_vl),
            );
            _mm_storeu_si128(
                v_buffer.as_mut_ptr() as *mut _,
                _mm256_castsi256_si128(cr_vl),
            );
        }

        cx += diff;

        if chroma_subsampling == YuvChromaSubsampling::Yuv444 {
            std::ptr::copy_nonoverlapping(
                u_buffer.as_ptr(),
                u_ptr.get_unchecked_mut(ux..).as_mut_ptr(),
                diff,
            );
            std::ptr::copy_nonoverlapping(
                v_buffer.as_ptr(),
                v_ptr.get_unchecked_mut(ux..).as_mut_ptr(),
                diff,
            );

            ux += diff;
        } else if (chroma_subsampling == YuvChromaSubsampling::Yuv420)
            || (chroma_subsampling == YuvChromaSubsampling::Yuv422)
        {
            let hv = diff.div_ceil(2);
            std::ptr::copy_nonoverlapping(
                u_buffer.as_ptr(),
                u_ptr.get_unchecked_mut(ux..).as_mut_ptr(),
                hv,
            );
            std::ptr::copy_nonoverlapping(
                v_buffer.as_ptr(),
                v_ptr.get_unchecked_mut(ux..).as_mut_ptr(),
                hv,
            );

            ux += hv;
        }
    }

    ProcessedOffset { cx, ux }
}
