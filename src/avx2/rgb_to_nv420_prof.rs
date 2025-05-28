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
    _mm256_affine_dot, _mm256_interleave_epi16, _mm256_interleave_x2_epi8,
    _mm256_load_deinterleave_rgb_for_yuv, avx2_pack_u16, avx_pairwise_avg_epi16_epi8_j, shuffle,
};
use crate::internals::ProcessedOffset;
use crate::yuv_support::{CbCrForwardTransform, YuvChromaRange, YuvNVOrder, YuvSourceChannels};
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// This is special path for 2 rows of BiPlanar 4:2:0 to reuse variables instead of computing them
pub(crate) fn avx2_rgba_to_nv420_prof<
    const ORIGIN_CHANNELS: u8,
    const UV_ORDER: u8,
    const PRECISION: i32,
>(
    y_plane0: &mut [u8],
    y_plane1: &mut [u8],
    uv_plane: &mut [u8],
    rgba0: &[u8],
    rgba1: &[u8],
    width: u32,
    range: &YuvChromaRange,
    transform: &CbCrForwardTransform<i32>,
    start_cx: usize,
    start_ux: usize,
) -> ProcessedOffset {
    unsafe {
        let source_channels: YuvSourceChannels = ORIGIN_CHANNELS.into();
        if source_channels == YuvSourceChannels::Bgra || source_channels == YuvSourceChannels::Rgba
        {
            return avx2_rgba_to_nv_prof_4chan::<ORIGIN_CHANNELS, UV_ORDER, PRECISION>(
                y_plane0, y_plane1, uv_plane, rgba0, rgba1, width, range, transform, start_cx,
                start_ux,
            );
        }
        #[cfg(feature = "nightly_avx512")]
        #[allow(clippy::incompatible_msrv)]
        if std::arch::is_x86_feature_detected!("avxvnni") {
            return avx2_rgba_to_nv_prof_dot::<ORIGIN_CHANNELS, UV_ORDER, PRECISION>(
                y_plane0, y_plane1, uv_plane, rgba0, rgba1, width, range, transform, start_cx,
                start_ux,
            );
        }
        avx2_rgba_to_nv_prof_def::<ORIGIN_CHANNELS, UV_ORDER, PRECISION>(
            y_plane0, y_plane1, uv_plane, rgba0, rgba1, width, range, transform, start_cx, start_ux,
        )
    }
}

#[inline(always)]
unsafe fn encode_32_part<
    const ORIGIN_CHANNELS: u8,
    const UV_ORDER: u8,
    const PRECISION: i32,
    const HAS_DOT: bool,
>(
    src0: &[u8],
    src1: &[u8],
    y_dst0: &mut [u8],
    y_dst1: &mut [u8],
    uv_dst: &mut [u8],
    range: &YuvChromaRange,
    transform: &CbCrForwardTransform<i32>,
) {
    let order: YuvNVOrder = UV_ORDER.into();

    let rounding_const_y = (1 << (PRECISION - 1)) - 1;
    let y_bias = _mm256_set1_epi32(range.bias_y as i32 * (1 << PRECISION) + rounding_const_y);
    let v_yr_yg = _mm256_set1_epi32(transform._interleaved_yr_yg());
    let v_yb = _mm256_set1_epi32(transform.yb);

    let precision_uv = PRECISION + 1;
    let rounding_const_uv = (1 << (precision_uv - 1)) - 1;

    let uv_bias = _mm256_set1_epi32(range.bias_uv as i32 * (1 << precision_uv) + rounding_const_uv);
    let v_cb_r_g = _mm256_set1_epi32(transform._interleaved_cbr_cbg());
    let v_cb_b = _mm256_set1_epi32(transform.cb_b);
    let v_cr_r_g = _mm256_set1_epi32(transform._interleaved_crr_crg());
    let v_cr_b = _mm256_set1_epi32(transform.cr_b);

    let (r_values0, g_values0, b_values0) =
        _mm256_load_deinterleave_rgb_for_yuv::<ORIGIN_CHANNELS>(src0.as_ptr());
    let (r_values1, g_values1, b_values1) =
        _mm256_load_deinterleave_rgb_for_yuv::<ORIGIN_CHANNELS>(src1.as_ptr());

    let rl0 = _mm256_unpacklo_epi8(r_values0, _mm256_setzero_si256());
    let gl0 = _mm256_unpacklo_epi8(g_values0, _mm256_setzero_si256());
    let bl0 = _mm256_unpacklo_epi8(b_values0, _mm256_setzero_si256());

    let (rl_gl0, rl_gl1) = _mm256_interleave_epi16(rl0, gl0);
    let (b_lo0, b_lo1) = _mm256_interleave_epi16(bl0, _mm256_setzero_si256());

    let y00_vl = _mm256_affine_dot::<PRECISION, HAS_DOT>(
        y_bias, rl_gl0, rl_gl1, b_lo0, b_lo1, v_yr_yg, v_yb,
    );

    let rh0 = _mm256_unpackhi_epi8(r_values0, _mm256_setzero_si256());
    let gh0 = _mm256_unpackhi_epi8(g_values0, _mm256_setzero_si256());
    let bh0 = _mm256_unpackhi_epi8(b_values0, _mm256_setzero_si256());

    let (rl_gh0, rl_gh1) = _mm256_interleave_epi16(rh0, gh0);
    let (b_h0, b_h1) = _mm256_interleave_epi16(bh0, _mm256_setzero_si256());

    let y01_vl =
        _mm256_affine_dot::<PRECISION, HAS_DOT>(y_bias, rl_gh0, rl_gh1, b_h0, b_h1, v_yr_yg, v_yb);

    let y0_values = _mm256_packus_epi16(y00_vl, y01_vl);
    _mm256_storeu_si256(y_dst0.as_mut_ptr() as *mut __m256i, y0_values);

    let rl1 = _mm256_unpacklo_epi8(r_values1, _mm256_setzero_si256());
    let gl1 = _mm256_unpacklo_epi8(g_values1, _mm256_setzero_si256());
    let bl1 = _mm256_unpacklo_epi8(b_values1, _mm256_setzero_si256());

    let (rl_gl01, rl_gl11) = _mm256_interleave_epi16(rl1, gl1);
    let (b_lo10, b_lo11) = _mm256_interleave_epi16(bl1, _mm256_setzero_si256());

    let y10_vl = _mm256_affine_dot::<PRECISION, HAS_DOT>(
        y_bias, rl_gl01, rl_gl11, b_lo10, b_lo11, v_yr_yg, v_yb,
    );

    let rh1 = _mm256_unpackhi_epi8(r_values1, _mm256_setzero_si256());
    let gh1 = _mm256_unpackhi_epi8(g_values1, _mm256_setzero_si256());
    let bh1 = _mm256_unpackhi_epi8(b_values1, _mm256_setzero_si256());

    let (rl_gh11, rl_gh110) = _mm256_interleave_epi16(rh1, gh1);
    let (b_h11, b_h111) = _mm256_interleave_epi16(bh1, _mm256_setzero_si256());

    let y11_vl = _mm256_affine_dot::<PRECISION, HAS_DOT>(
        y_bias, rl_gh11, rl_gh110, b_h11, b_h111, v_yr_yg, v_yb,
    );

    let y0_values = _mm256_packus_epi16(y10_vl, y11_vl);
    _mm256_storeu_si256(y_dst1.as_mut_ptr() as *mut __m256i, y0_values);

    let r_avg = _mm256_avg_epu8(r_values0, r_values1);
    let g_avg = _mm256_avg_epu8(g_values0, g_values1);
    let b_avg = _mm256_avg_epu8(b_values0, b_values1);

    let r1 = avx_pairwise_avg_epi16_epi8_j(r_avg, 1);
    let g1 = avx_pairwise_avg_epi16_epi8_j(g_avg, 1);
    let b1 = avx_pairwise_avg_epi16_epi8_j(b_avg, 1);

    let (rhv0, rhv1) = _mm256_interleave_epi16(r1, g1);
    let (bhv0, bhv1) = _mm256_interleave_epi16(b1, _mm256_setzero_si256());

    let cb_s = _mm256_affine_dot::<16, HAS_DOT>(uv_bias, rhv0, rhv1, bhv0, bhv1, v_cb_r_g, v_cb_b);

    let cr_s = _mm256_affine_dot::<16, HAS_DOT>(uv_bias, rhv0, rhv1, bhv0, bhv1, v_cr_r_g, v_cr_b);

    let cb = avx2_pack_u16(cb_s, cb_s);
    let cr = avx2_pack_u16(cr_s, cr_s);

    let (row0, _) = match order {
        YuvNVOrder::UV => _mm256_interleave_x2_epi8(cb, cr),
        YuvNVOrder::VU => _mm256_interleave_x2_epi8(cr, cb),
    };

    _mm256_storeu_si256(uv_dst.as_mut_ptr() as *mut __m256i, row0);
}

#[target_feature(enable = "avx2")]
unsafe fn avx2_rgba_to_nv_prof_def<
    const ORIGIN_CHANNELS: u8,
    const UV_ORDER: u8,
    const PRECISION: i32,
>(
    y_plane0: &mut [u8],
    y_plane1: &mut [u8],
    uv_plane: &mut [u8],
    rgba0: &[u8],
    rgba1: &[u8],
    width: u32,
    range: &YuvChromaRange,
    transform: &CbCrForwardTransform<i32>,
    start_cx: usize,
    start_ux: usize,
) -> ProcessedOffset {
    avx2_rgba_to_nv_prof_impl::<ORIGIN_CHANNELS, UV_ORDER, PRECISION, false>(
        y_plane0, y_plane1, uv_plane, rgba0, rgba1, width, range, transform, start_cx, start_ux,
    )
}

#[cfg(feature = "nightly_avx512")]
#[target_feature(enable = "avx2", enable = "avxvnni")]
unsafe fn avx2_rgba_to_nv_prof_dot<
    const ORIGIN_CHANNELS: u8,
    const UV_ORDER: u8,
    const PRECISION: i32,
>(
    y_plane0: &mut [u8],
    y_plane1: &mut [u8],
    uv_plane: &mut [u8],
    rgba0: &[u8],
    rgba1: &[u8],
    width: u32,
    range: &YuvChromaRange,
    transform: &CbCrForwardTransform<i32>,
    start_cx: usize,
    start_ux: usize,
) -> ProcessedOffset {
    avx2_rgba_to_nv_prof_impl::<ORIGIN_CHANNELS, UV_ORDER, PRECISION, true>(
        y_plane0, y_plane1, uv_plane, rgba0, rgba1, width, range, transform, start_cx, start_ux,
    )
}

#[inline(always)]
unsafe fn avx2_rgba_to_nv_prof_impl<
    const ORIGIN_CHANNELS: u8,
    const UV_ORDER: u8,
    const PRECISION: i32,
    const HAS_DOT: bool,
>(
    y_plane0: &mut [u8],
    y_plane1: &mut [u8],
    uv_plane: &mut [u8],
    rgba0: &[u8],
    rgba1: &[u8],
    width: u32,
    range: &YuvChromaRange,
    transform: &CbCrForwardTransform<i32>,
    start_cx: usize,
    start_ux: usize,
) -> ProcessedOffset {
    let source_channels: YuvSourceChannels = ORIGIN_CHANNELS.into();
    let channels = source_channels.get_channels_count();

    let mut cx = start_cx;
    let mut uv_x = start_ux;

    while cx + 32 < width as usize {
        let px = cx * channels;

        encode_32_part::<ORIGIN_CHANNELS, UV_ORDER, PRECISION, HAS_DOT>(
            rgba0.get_unchecked(px..),
            rgba1.get_unchecked(px..),
            y_plane0.get_unchecked_mut(cx..),
            y_plane1.get_unchecked_mut(cx..),
            uv_plane.get_unchecked_mut(uv_x..),
            range,
            transform,
        );

        uv_x += 32;
        cx += 32;
    }

    if cx < width as usize {
        let diff = width as usize - cx;
        assert!(diff <= 32);

        let mut src_buffer0: [u8; 32 * 4] = [0; 32 * 4];
        let mut src_buffer1: [u8; 32 * 4] = [0; 32 * 4];
        let mut y_buffer0: [u8; 32] = [0; 32];
        let mut y_buffer1: [u8; 32] = [0; 32];
        let mut uv_buffer: [u8; 32 * 2] = [0; 32 * 2];

        std::ptr::copy_nonoverlapping(
            rgba0.get_unchecked(cx * channels..).as_ptr(),
            src_buffer0.as_mut_ptr(),
            diff * channels,
        );

        std::ptr::copy_nonoverlapping(
            rgba1.get_unchecked(cx * channels..).as_ptr(),
            src_buffer1.as_mut_ptr(),
            diff * channels,
        );

        // Replicate last item to one more position for subsampling
        if diff % 2 != 0 {
            let lst = (width as usize - 1) * channels;
            let last_items0 = rgba0.get_unchecked(lst..(lst + channels));
            let last_items1 = rgba1.get_unchecked(lst..(lst + channels));
            let dvb = diff * channels;
            let dst0 = src_buffer0.get_unchecked_mut(dvb..(dvb + channels));
            let dst1 = src_buffer1.get_unchecked_mut(dvb..(dvb + channels));
            for (dst, src) in dst0.iter_mut().zip(last_items0) {
                *dst = *src;
            }
            for (dst, src) in dst1.iter_mut().zip(last_items1) {
                *dst = *src;
            }
        }

        encode_32_part::<ORIGIN_CHANNELS, UV_ORDER, PRECISION, HAS_DOT>(
            src_buffer0.as_slice(),
            src_buffer1.as_slice(),
            y_buffer0.as_mut_slice(),
            y_buffer1.as_mut_slice(),
            uv_buffer.as_mut_slice(),
            range,
            transform,
        );

        std::ptr::copy_nonoverlapping(
            y_buffer0.as_mut_ptr(),
            y_plane0.get_unchecked_mut(cx..).as_mut_ptr(),
            diff,
        );

        std::ptr::copy_nonoverlapping(
            y_buffer1.as_mut_ptr(),
            y_plane1.get_unchecked_mut(cx..).as_mut_ptr(),
            diff,
        );

        let ux_size = diff.div_ceil(2) * 2;

        std::ptr::copy_nonoverlapping(
            uv_buffer.as_mut_ptr(),
            uv_plane.get_unchecked_mut(uv_x..).as_mut_ptr(),
            ux_size,
        );

        cx += diff;
        uv_x += ux_size;
    }

    ProcessedOffset { cx, ux: uv_x }
}

#[target_feature(enable = "avx2")]
unsafe fn avx2_rgba_to_nv_prof_4chan<
    const ORIGIN_CHANNELS: u8,
    const UV_ORDER: u8,
    const PRECISION: i32,
>(
    y_plane0: &mut [u8],
    y_plane1: &mut [u8],
    uv_plane: &mut [u8],
    rgba0: &[u8],
    rgba1: &[u8],
    width: u32,
    range: &YuvChromaRange,
    transform: &CbCrForwardTransform<i32>,
    start_cx: usize,
    start_ux: usize,
) -> ProcessedOffset {
    let source_channels: YuvSourceChannels = ORIGIN_CHANNELS.into();
    assert!(
        source_channels == YuvSourceChannels::Rgba || source_channels == YuvSourceChannels::Bgra
    );
    let channels = source_channels.get_channels_count();

    let y_transform = _mm256_set1_epi64x(transform.avx_make_transform_y(source_channels));
    let cb_transform = _mm256_set1_epi64x(transform.avx_make_transform_cb(source_channels));
    let cr_transform = _mm256_set1_epi64x(transform.avx_make_transform_cr(source_channels));
    let rounding_const_y = (1 << (PRECISION - 1)) - 1;
    let y_bias = _mm256_set1_epi32(range.bias_y as i32 * (1 << PRECISION) + rounding_const_y);
    let precision_uv = PRECISION + 1;
    let rounding_const_uv = (1 << (precision_uv - 1)) - 1;
    let shuf_uv = _mm256_setr_epi8(
        0, 4, 1, 5, 2, 6, -1, -1, 8, 12, 9, 13, 10, 14, -1, -1, 0, 4, 1, 5, 2, 6, -1, -1, 8, 12, 9,
        13, 10, 14, -1, -1,
    );

    let uv_bias = _mm256_set1_epi32(range.bias_uv as i32 * (1 << precision_uv) + rounding_const_uv);

    let shuf_uv_back = _mm256_setr_epi32(0, 4, -1, -1, -1, -1, -1, -1);

    let mut cx = start_cx;
    let mut uv_x = start_ux;

    let order: YuvNVOrder = UV_ORDER.into();

    while cx + 16 < width as usize {
        let src_ptr0 = rgba0.get_unchecked(cx * channels..);
        let src_ptr1 = rgba1.get_unchecked(cx * channels..);

        let row_z0_0 = _mm256_loadu_si256(src_ptr0.as_ptr() as *const _);
        let row_z0_1 = _mm256_loadu_si256(src_ptr1.as_ptr() as *const _);

        let row_z1_0 = _mm256_loadu_si256(src_ptr0.get_unchecked(32..).as_ptr() as *const _);
        let row_z1_1 = _mm256_loadu_si256(src_ptr1.get_unchecked(32..).as_ptr() as *const _);

        let avg0 = _mm256_avg_epu8(row_z0_0, row_z0_1);
        let avg1 = _mm256_avg_epu8(row_z1_0, row_z1_1);

        let avgu0_v = avx_pairwise_avg_epi16_epi8_j(_mm256_shuffle_epi8(avg0, shuf_uv), 1);
        let avgu1_v = avx_pairwise_avg_epi16_epi8_j(_mm256_shuffle_epi8(avg1, shuf_uv), 1);

        let y_row0 = _mm256_madd_epi16(
            _mm256_unpacklo_epi8(row_z0_0, _mm256_setzero_si256()),
            y_transform,
        );
        let y_row1 = _mm256_madd_epi16(
            _mm256_unpackhi_epi8(row_z0_0, _mm256_setzero_si256()),
            y_transform,
        );

        let mut f_y0 = _mm256_hadd_epi32(y_row0, y_row1);
        f_y0 = _mm256_add_epi32(f_y0, y_bias);
        f_y0 = _mm256_srai_epi32::<PRECISION>(f_y0);

        let y1_row0 = _mm256_madd_epi16(
            _mm256_unpacklo_epi8(row_z1_0, _mm256_setzero_si256()),
            y_transform,
        );
        let y1_row1 = _mm256_madd_epi16(
            _mm256_unpackhi_epi8(row_z1_0, _mm256_setzero_si256()),
            y_transform,
        );

        let mut f_y1 = _mm256_hadd_epi32(y1_row0, y1_row1);
        f_y1 = _mm256_add_epi32(f_y1, y_bias);
        f_y1 = _mm256_srai_epi32::<PRECISION>(f_y1);

        let y_row2 = _mm256_madd_epi16(
            _mm256_unpacklo_epi8(row_z0_1, _mm256_setzero_si256()),
            y_transform,
        );
        let y_row3 = _mm256_madd_epi16(
            _mm256_unpackhi_epi8(row_z0_1, _mm256_setzero_si256()),
            y_transform,
        );

        let mut f_y2 = _mm256_hadd_epi32(y_row2, y_row3);
        f_y2 = _mm256_add_epi32(f_y2, y_bias);
        f_y2 = _mm256_srai_epi32::<PRECISION>(f_y2);

        let y_row4 = _mm256_madd_epi16(
            _mm256_unpacklo_epi8(row_z1_1, _mm256_setzero_si256()),
            y_transform,
        );
        let y_row5 = _mm256_madd_epi16(
            _mm256_unpackhi_epi8(row_z1_1, _mm256_setzero_si256()),
            y_transform,
        );

        let mut f_y3 = _mm256_hadd_epi32(y_row4, y_row5);
        f_y3 = _mm256_add_epi32(f_y3, y_bias);
        f_y3 = _mm256_srai_epi32::<PRECISION>(f_y3);

        let z_y = _mm256_permute4x64_epi64::<M>(_mm256_packus_epi16(
            _mm256_permute4x64_epi64::<M>(_mm256_packus_epi32(f_y0, f_y1)),
            _mm256_permute4x64_epi64::<M>(_mm256_packus_epi32(f_y2, f_y3)),
        ));

        _mm_storeu_si128(
            y_plane0.get_unchecked_mut(cx..).as_mut_ptr() as *mut _,
            _mm256_castsi256_si128(z_y),
        );
        _mm_storeu_si128(
            y_plane1.get_unchecked_mut(cx..).as_mut_ptr() as *mut _,
            _mm256_extracti128_si256::<1>(z_y),
        );

        let cb_row0 = _mm256_madd_epi16(avgu0_v, cb_transform);
        let cb_row1 = _mm256_madd_epi16(avgu1_v, cb_transform);
        let cr_row0 = _mm256_madd_epi16(avgu0_v, cr_transform);
        let cr_row1 = _mm256_madd_epi16(avgu1_v, cr_transform);

        const M: i32 = shuffle(3, 1, 2, 0);

        let mut f_cb0 = _mm256_permute4x64_epi64::<M>(_mm256_hadd_epi32(cb_row0, cb_row1));
        let mut f_cr0 = _mm256_permute4x64_epi64::<M>(_mm256_hadd_epi32(cr_row0, cr_row1));

        f_cb0 = _mm256_add_epi32(f_cb0, uv_bias);
        f_cr0 = _mm256_add_epi32(f_cr0, uv_bias);

        f_cb0 = _mm256_srai_epi32::<16>(f_cb0);
        f_cr0 = _mm256_srai_epi32::<16>(f_cr0);

        let z_cb = _mm256_permutevar8x32_epi32(
            _mm256_packus_epi16(
                _mm256_packus_epi32(f_cb0, _mm256_setzero_si256()),
                _mm256_setzero_si256(),
            ),
            shuf_uv_back,
        );

        let z_cr = _mm256_permutevar8x32_epi32(
            _mm256_packus_epi16(
                _mm256_packus_epi32(f_cr0, _mm256_setzero_si256()),
                _mm256_setzero_si256(),
            ),
            shuf_uv_back,
        );

        let z_cb_cr = match order {
            YuvNVOrder::UV => {
                _mm_unpacklo_epi8(_mm256_castsi256_si128(z_cb), _mm256_castsi256_si128(z_cr))
            }
            YuvNVOrder::VU => {
                _mm_unpacklo_epi8(_mm256_castsi256_si128(z_cr), _mm256_castsi256_si128(z_cb))
            }
        };

        _mm_storeu_si128(
            uv_plane.get_unchecked_mut(uv_x..).as_mut_ptr() as *mut _,
            z_cb_cr,
        );

        uv_x += 16;
        cx += 16;
    }

    if cx < width as usize {
        let diff = width as usize - cx;
        assert!(diff <= 16);

        let mut src_buffer0: [u8; 16 * 4] = [0; 16 * 4];
        let mut src_buffer1: [u8; 16 * 4] = [0; 16 * 4];
        let mut y_buffer0: [u8; 16] = [0; 16];
        let mut y_buffer1: [u8; 16] = [0; 16];
        let mut uv_buffer: [u8; 16 * 2] = [0; 16 * 2];

        std::ptr::copy_nonoverlapping(
            rgba0.get_unchecked(cx * channels..).as_ptr(),
            src_buffer0.as_mut_ptr(),
            diff * channels,
        );

        std::ptr::copy_nonoverlapping(
            rgba1.get_unchecked(cx * channels..).as_ptr(),
            src_buffer1.as_mut_ptr(),
            diff * channels,
        );

        // Replicate last item to one more position for subsampling
        if diff % 2 != 0 {
            let lst = (width as usize - 1) * channels;
            let last_items0 = rgba0.get_unchecked(lst..(lst + channels));
            let last_items1 = rgba1.get_unchecked(lst..(lst + channels));
            let dvb = diff * channels;
            let dst0 = src_buffer0.get_unchecked_mut(dvb..(dvb + channels));
            let dst1 = src_buffer1.get_unchecked_mut(dvb..(dvb + channels));
            for (dst, src) in dst0.iter_mut().zip(last_items0) {
                *dst = *src;
            }
            for (dst, src) in dst1.iter_mut().zip(last_items1) {
                *dst = *src;
            }
        }

        let row_z0_0 = _mm256_loadu_si256(src_buffer0.as_ptr() as *const _);
        let row_z0_1 = _mm256_loadu_si256(src_buffer1.as_ptr() as *const _);

        let row_z1_0 = _mm256_loadu_si256(src_buffer0.get_unchecked(32..).as_ptr() as *const _);
        let row_z1_1 = _mm256_loadu_si256(src_buffer1.get_unchecked(32..).as_ptr() as *const _);

        let avg0 = _mm256_avg_epu8(row_z0_0, row_z0_1);
        let avg1 = _mm256_avg_epu8(row_z1_0, row_z1_1);

        let avgu0_v = avx_pairwise_avg_epi16_epi8_j(_mm256_shuffle_epi8(avg0, shuf_uv), 1);
        let avgu1_v = avx_pairwise_avg_epi16_epi8_j(_mm256_shuffle_epi8(avg1, shuf_uv), 1);

        let y_row0 = _mm256_madd_epi16(
            _mm256_unpacklo_epi8(row_z0_0, _mm256_setzero_si256()),
            y_transform,
        );
        let y_row1 = _mm256_madd_epi16(
            _mm256_unpackhi_epi8(row_z0_0, _mm256_setzero_si256()),
            y_transform,
        );

        let mut f_y0 = _mm256_hadd_epi32(y_row0, y_row1);
        f_y0 = _mm256_add_epi32(f_y0, y_bias);
        f_y0 = _mm256_srai_epi32::<PRECISION>(f_y0);

        let y1_row0 = _mm256_madd_epi16(
            _mm256_unpacklo_epi8(row_z1_0, _mm256_setzero_si256()),
            y_transform,
        );
        let y1_row1 = _mm256_madd_epi16(
            _mm256_unpackhi_epi8(row_z1_0, _mm256_setzero_si256()),
            y_transform,
        );

        let mut f_y1 = _mm256_hadd_epi32(y1_row0, y1_row1);
        f_y1 = _mm256_add_epi32(f_y1, y_bias);
        f_y1 = _mm256_srai_epi32::<PRECISION>(f_y1);

        let y_row2 = _mm256_madd_epi16(
            _mm256_unpacklo_epi8(row_z0_1, _mm256_setzero_si256()),
            y_transform,
        );
        let y_row3 = _mm256_madd_epi16(
            _mm256_unpackhi_epi8(row_z0_1, _mm256_setzero_si256()),
            y_transform,
        );

        let mut f_y2 = _mm256_hadd_epi32(y_row2, y_row3);
        f_y2 = _mm256_add_epi32(f_y2, y_bias);
        f_y2 = _mm256_srai_epi32::<PRECISION>(f_y2);

        let y_row4 = _mm256_madd_epi16(
            _mm256_unpacklo_epi8(row_z1_1, _mm256_setzero_si256()),
            y_transform,
        );
        let y_row5 = _mm256_madd_epi16(
            _mm256_unpackhi_epi8(row_z1_1, _mm256_setzero_si256()),
            y_transform,
        );

        let mut f_y3 = _mm256_hadd_epi32(y_row4, y_row5);
        f_y3 = _mm256_add_epi32(f_y3, y_bias);
        f_y3 = _mm256_srai_epi32::<PRECISION>(f_y3);

        let z_y = _mm256_permute4x64_epi64::<M>(_mm256_packus_epi16(
            _mm256_permute4x64_epi64::<M>(_mm256_packus_epi32(f_y0, f_y1)),
            _mm256_permute4x64_epi64::<M>(_mm256_packus_epi32(f_y2, f_y3)),
        ));

        _mm_storeu_si128(
            y_buffer0.as_mut_ptr() as *mut _,
            _mm256_castsi256_si128(z_y),
        );
        _mm_storeu_si128(
            y_buffer1.as_mut_ptr() as *mut _,
            _mm256_extracti128_si256::<1>(z_y),
        );

        let cb_row0 = _mm256_madd_epi16(avgu0_v, cb_transform);
        let cb_row1 = _mm256_madd_epi16(avgu1_v, cb_transform);
        let cr_row0 = _mm256_madd_epi16(avgu0_v, cr_transform);
        let cr_row1 = _mm256_madd_epi16(avgu1_v, cr_transform);

        const M: i32 = shuffle(3, 1, 2, 0);

        let mut f_cb0 = _mm256_permute4x64_epi64::<M>(_mm256_hadd_epi32(cb_row0, cb_row1));
        let mut f_cr0 = _mm256_permute4x64_epi64::<M>(_mm256_hadd_epi32(cr_row0, cr_row1));

        f_cb0 = _mm256_add_epi32(f_cb0, uv_bias);
        f_cr0 = _mm256_add_epi32(f_cr0, uv_bias);

        f_cb0 = _mm256_srai_epi32::<16>(f_cb0);
        f_cr0 = _mm256_srai_epi32::<16>(f_cr0);

        let z_cb = _mm256_permutevar8x32_epi32(
            _mm256_packus_epi16(
                _mm256_packus_epi32(f_cb0, _mm256_setzero_si256()),
                _mm256_setzero_si256(),
            ),
            shuf_uv_back,
        );
        let z_cr = _mm256_permutevar8x32_epi32(
            _mm256_packus_epi16(
                _mm256_packus_epi32(f_cr0, _mm256_setzero_si256()),
                _mm256_setzero_si256(),
            ),
            shuf_uv_back,
        );

        let z_cb_cr = match order {
            YuvNVOrder::UV => {
                _mm_unpacklo_epi8(_mm256_castsi256_si128(z_cb), _mm256_castsi256_si128(z_cr))
            }
            YuvNVOrder::VU => {
                _mm_unpacklo_epi8(_mm256_castsi256_si128(z_cr), _mm256_castsi256_si128(z_cb))
            }
        };

        _mm_storeu_si128(uv_buffer.as_mut_ptr() as *mut _, z_cb_cr);

        std::ptr::copy_nonoverlapping(
            y_buffer0.as_mut_ptr(),
            y_plane0.get_unchecked_mut(cx..).as_mut_ptr(),
            diff,
        );

        std::ptr::copy_nonoverlapping(
            y_buffer1.as_mut_ptr(),
            y_plane1.get_unchecked_mut(cx..).as_mut_ptr(),
            diff,
        );

        let ux_size = diff.div_ceil(2) * 2;

        std::ptr::copy_nonoverlapping(
            uv_buffer.as_mut_ptr(),
            uv_plane.get_unchecked_mut(uv_x..).as_mut_ptr(),
            ux_size,
        );

        cx += diff;
        uv_x += ux_size;
    }

    ProcessedOffset { cx, ux: uv_x }
}
