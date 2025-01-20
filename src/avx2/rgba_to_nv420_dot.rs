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

use crate::avx2::avx2_utils::{avx2_pack_u16, avx2_pack_u32, shuffle};
use crate::internals::ProcessedOffset;
use crate::yuv_support::{CbCrForwardTransform, YuvChromaRange, YuvNVOrder, YuvSourceChannels};
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// This is special path for 2 rows of BiPlanar 4:2:0 to reuse variables instead of computing them
pub(crate) fn avx2_rgba_to_nv420_vnni<
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
        avx2_rgba_to_nv_vnni_impl::<ORIGIN_CHANNELS, UV_ORDER, PRECISION>(
            y_plane0, y_plane1, uv_plane, rgba0, rgba1, width, range, transform, start_cx, start_ux,
        )
    }
}

#[target_feature(enable = "avx2", enable = "avxvnni")]
unsafe fn avx2_rgba_to_nv_vnni_impl<
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
    let order: YuvNVOrder = UV_ORDER.into();
    let source_channels: YuvSourceChannels = ORIGIN_CHANNELS.into();
    assert!(
        source_channels == YuvSourceChannels::Rgba || source_channels == YuvSourceChannels::Bgra
    );
    let channels = source_channels.get_channels_count();

    let uv_ptr = uv_plane.as_mut_ptr();

    let mut cx = start_cx;
    let mut uv_x = start_ux;

    let precision_scale: i32 = 1 << PRECISION;
    let precision_rounding: i32 = 1 << (PRECISION - 1) - 1;

    let by = range.bias_y as i32 * precision_scale + precision_rounding;
    let y_bias = _mm256_setr_epi32(by, 0, by, 0, by, 0, by, 0);
    let bu = range.bias_uv as i32 * precision_scale + precision_rounding;
    let uv_bias = _mm256_setr_epi32(bu, 0, bu, 0, bu, 0, bu, 0);

    let v_y_weights = if source_channels == YuvSourceChannels::Rgba {
        _mm256_setr_epi16(
            transform.yr as i16,
            transform.yg as i16,
            transform.yb as i16,
            0,
            transform.yr as i16,
            transform.yg as i16,
            transform.yb as i16,
            0,
            transform.yr as i16,
            transform.yg as i16,
            transform.yb as i16,
            0,
            transform.yr as i16,
            transform.yg as i16,
            transform.yb as i16,
            0,
        )
    } else if source_channels == YuvSourceChannels::Bgra {
        _mm256_setr_epi16(
            transform.yb as i16,
            transform.yg as i16,
            transform.yr as i16,
            0,
            transform.yb as i16,
            transform.yg as i16,
            transform.yr as i16,
            0,
            transform.yb as i16,
            transform.yg as i16,
            transform.yr as i16,
            0,
            transform.yb as i16,
            transform.yg as i16,
            transform.yr as i16,
            0,
        )
    } else {
        unreachable!()
    };

    let v_cb_weights = if source_channels == YuvSourceChannels::Rgba {
        _mm256_setr_epi16(
            transform.cb_r as i16,
            transform.cb_g as i16,
            transform.cb_b as i16,
            0,
            transform.cb_r as i16,
            transform.cb_g as i16,
            transform.cb_b as i16,
            0,
            transform.cb_r as i16,
            transform.cb_g as i16,
            transform.cb_b as i16,
            0,
            transform.cb_r as i16,
            transform.cb_g as i16,
            transform.cb_b as i16,
            0,
        )
    } else if source_channels == YuvSourceChannels::Bgra {
        _mm256_setr_epi16(
            transform.cb_b as i16,
            transform.cb_g as i16,
            transform.cb_r as i16,
            0,
            transform.cb_b as i16,
            transform.cb_g as i16,
            transform.cb_r as i16,
            0,
            transform.cb_b as i16,
            transform.cb_g as i16,
            transform.cb_r as i16,
            0,
            transform.cb_b as i16,
            transform.cb_g as i16,
            transform.cb_r as i16,
            0,
        )
    } else {
        unreachable!()
    };

    let v_cr_weights = if source_channels == YuvSourceChannels::Rgba {
        _mm256_setr_epi16(
            transform.cr_r as i16,
            transform.cr_g as i16,
            transform.cr_b as i16,
            0,
            transform.cr_r as i16,
            transform.cr_g as i16,
            transform.cr_b as i16,
            0,
            transform.cr_r as i16,
            transform.cr_g as i16,
            transform.cr_b as i16,
            0,
            transform.cr_r as i16,
            transform.cr_g as i16,
            transform.cr_b as i16,
            0,
        )
    } else if source_channels == YuvSourceChannels::Bgra {
        _mm256_setr_epi16(
            transform.cr_b as i16,
            transform.cr_g as i16,
            transform.cr_r as i16,
            0,
            transform.cr_b as i16,
            transform.cr_g as i16,
            transform.cr_r as i16,
            0,
            transform.cr_b as i16,
            transform.cr_g as i16,
            transform.cr_r as i16,
            0,
            transform.cr_b as i16,
            transform.cr_g as i16,
            transform.cr_r as i16,
            0,
        )
    } else {
        unreachable!()
    };

    let zeros = _mm256_setzero_si256();

    while cx + 16 < width as usize {
        let px = cx * channels;

        let row00 = _mm256_loadu_si256(rgba0.get_unchecked(px..).as_ptr() as *const __m256i);
        let row01 = _mm256_loadu_si256(rgba0.get_unchecked((px + 32)..).as_ptr() as *const __m256i);

        let row10 = _mm256_loadu_si256(rgba1.get_unchecked(px..).as_ptr() as *const __m256i);
        let row11 = _mm256_loadu_si256(rgba1.get_unchecked((px + 32)..).as_ptr() as *const __m256i);

        let row00_lo = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(row00));
        let row00_hi = _mm256_cvtepu8_epi16(_mm256_extracti128_si256::<1>(row00));

        let row01_lo = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(row01));
        let row01_hi = _mm256_cvtepu8_epi16(_mm256_extracti128_si256::<1>(row01));

        let y00_lo = _mm256_dpwssd_avx_epi32(y_bias, row00_lo, v_y_weights);
        let y00_hi = _mm256_dpwssd_avx_epi32(y_bias, row00_hi, v_y_weights);
        let y01_lo = _mm256_dpwssd_avx_epi32(y_bias, row01_lo, v_y_weights);
        let y01_hi = _mm256_dpwssd_avx_epi32(y_bias, row01_hi, v_y_weights);

        const MASK: i32 = shuffle(3, 1, 2, 0);

        let y0_16 = avx2_pack_u32(
            _mm256_srai_epi32::<PRECISION>(_mm256_permute4x64_epi64::<MASK>(_mm256_hadd_epi32(
                y00_lo, y00_hi,
            ))),
            _mm256_srai_epi32::<PRECISION>(_mm256_permute4x64_epi64::<MASK>(_mm256_hadd_epi32(
                y01_lo, y01_hi,
            ))),
        );
        let y0 = avx2_pack_u16(y0_16, zeros);
        _mm_storeu_si128(
            y_plane0.get_unchecked_mut(cx..).as_mut_ptr() as *mut _,
            _mm256_castsi256_si128(y0),
        );

        let row10_lo = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(row10));
        let row10_hi = _mm256_cvtepu8_epi16(_mm256_extracti128_si256::<1>(row10));
        let row11_lo = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(row11));
        let row11_hi = _mm256_cvtepu8_epi16(_mm256_extracti128_si256::<1>(row11));

        let y10_lo = _mm256_dpwssd_avx_epi32(y_bias, row10_lo, v_y_weights);
        let y10_hi = _mm256_dpwssd_avx_epi32(y_bias, row10_hi, v_y_weights);
        let y11_lo = _mm256_dpwssd_avx_epi32(y_bias, row11_lo, v_y_weights);
        let y11_hi = _mm256_dpwssd_avx_epi32(y_bias, row11_hi, v_y_weights);

        let y1_16 = avx2_pack_u32(
            _mm256_srai_epi32::<PRECISION>(_mm256_permute4x64_epi64::<MASK>(_mm256_hadd_epi32(
                y10_lo, y10_hi,
            ))),
            _mm256_srai_epi32::<PRECISION>(_mm256_permute4x64_epi64::<MASK>(_mm256_hadd_epi32(
                y11_lo, y11_hi,
            ))),
        );
        let y1 = avx2_pack_u16(y1_16, y1_16);
        _mm_storeu_si128(
            y_plane1.get_unchecked_mut(cx..).as_mut_ptr() as *mut _,
            _mm256_castsi256_si128(y1),
        );

        let avg_rows_l0 = _mm256_avg_epu8(row00, row10);
        let avg_rows_l1 = _mm256_avg_epu8(row01, row11);

        let permute_order = _mm256_setr_epi32(1, 0, 3, 2, 5, 4, 7, 6);

        let avg_rw0 = _mm256_avg_epu8(
            avg_rows_l0,
            _mm256_permutevar8x32_epi32(avg_rows_l0, permute_order),
        );
        let avg_rw1 = _mm256_avg_epu8(
            avg_rows_l1,
            _mm256_permutevar8x32_epi32(avg_rows_l1, permute_order),
        );

        let permute_lowers = _mm256_setr_epi32(0, 2, 0, 0, 4, 6, 0, 0);

        let c_row0 =
            _mm256_unpacklo_epi8(_mm256_permutevar8x32_epi32(avg_rw0, permute_lowers), zeros);
        let c_row1 =
            _mm256_unpacklo_epi8(_mm256_permutevar8x32_epi32(avg_rw1, permute_lowers), zeros);

        let cb_32_lo = _mm256_dpwssd_avx_epi32(uv_bias, c_row0, v_cb_weights);
        let cr_32_lo = _mm256_dpwssd_avx_epi32(uv_bias, c_row0, v_cr_weights);
        let cb_32_hi = _mm256_dpwssd_avx_epi32(uv_bias, c_row1, v_cb_weights);
        let cr_32_hi = _mm256_dpwssd_avx_epi32(uv_bias, c_row1, v_cr_weights);

        let cb_16 = avx2_pack_u32(
            _mm256_srai_epi32::<PRECISION>(_mm256_permute4x64_epi64::<MASK>(_mm256_hadd_epi32(
                cb_32_lo, cb_32_hi,
            ))),
            cb_32_lo,
        );

        let cr_16 = avx2_pack_u32(
            _mm256_srai_epi32::<PRECISION>(_mm256_permute4x64_epi64::<MASK>(_mm256_hadd_epi32(
                cr_32_lo, cr_32_hi,
            ))),
            cr_32_lo,
        );

        let cb = avx2_pack_u16(cb_16, cb_16);
        let cr = avx2_pack_u16(cr_16, cr_16);

        let row0 = match order {
            YuvNVOrder::UV => {
                _mm_unpacklo_epi8(_mm256_castsi256_si128(cb), _mm256_castsi256_si128(cr))
            }
            YuvNVOrder::VU => {
                _mm_unpacklo_epi8(_mm256_castsi256_si128(cr), _mm256_castsi256_si128(cb))
            }
        };

        _mm_storeu_si128(uv_ptr.add(uv_x) as *mut _, row0);

        uv_x += 16;
        cx += 16;
    }

    while cx + 8 < width as usize {
        let px = cx * channels;

        let row0 = _mm256_loadu_si256(rgba0.get_unchecked(px..).as_ptr() as *const __m256i);
        let row1 = _mm256_loadu_si256(rgba1.get_unchecked(px..).as_ptr() as *const __m256i);

        let row0_lo = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(row0));
        let row0_hi = _mm256_cvtepu8_epi16(_mm256_extracti128_si256::<1>(row0));

        let y0_lo = _mm256_dpwssd_avx_epi32(y_bias, row0_lo, v_y_weights);
        let y0_hi = _mm256_dpwssd_avx_epi32(y_bias, row0_hi, v_y_weights);

        const MASK: i32 = shuffle(3, 1, 2, 0);

        let y0_16 = avx2_pack_u32(
            _mm256_srai_epi32::<PRECISION>(_mm256_permute4x64_epi64::<MASK>(_mm256_hadd_epi32(
                y0_lo, y0_hi,
            ))),
            zeros,
        );
        let y0 = avx2_pack_u16(y0_16, zeros);
        _mm_storeu_si64(
            y_plane0.get_unchecked_mut(cx..).as_mut_ptr() as *mut _,
            _mm256_castsi256_si128(y0),
        );

        let row1_lo = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(row1));
        let row1_hi = _mm256_cvtepu8_epi16(_mm256_extracti128_si256::<1>(row1));

        let y1_lo = _mm256_dpwssd_avx_epi32(y_bias, row1_lo, v_y_weights);
        let y1_hi = _mm256_dpwssd_avx_epi32(y_bias, row1_hi, v_y_weights);

        let y1_16 = avx2_pack_u32(
            _mm256_srai_epi32::<PRECISION>(_mm256_permute4x64_epi64::<MASK>(_mm256_hadd_epi32(
                y1_lo, y1_hi,
            ))),
            y1_lo,
        );
        let y1 = avx2_pack_u16(y1_16, y1_16);
        _mm_storeu_si64(
            y_plane1.get_unchecked_mut(cx..).as_mut_ptr() as *mut _,
            _mm256_castsi256_si128(y1),
        );

        let avg_rows_l = _mm256_avg_epu8(row0, row1);

        let permute_order = _mm256_setr_epi32(1, 0, 3, 2, 5, 4, 7, 6);

        let avg_rw = _mm256_avg_epu8(
            avg_rows_l,
            _mm256_permutevar8x32_epi32(avg_rows_l, permute_order),
        );

        let permute_lowers = _mm256_setr_epi32(0, 2, 0, 0, 4, 6, 0, 0);

        let c_row =
            _mm256_unpacklo_epi8(_mm256_permutevar8x32_epi32(avg_rw, permute_lowers), zeros);

        let cb_32_lo = _mm256_dpwssd_avx_epi32(uv_bias, c_row, v_cb_weights);

        let cr_32_lo = _mm256_dpwssd_avx_epi32(uv_bias, c_row, v_cr_weights);

        let cb_16 = avx2_pack_u32(
            _mm256_srai_epi32::<PRECISION>(_mm256_permute4x64_epi64::<MASK>(_mm256_hadd_epi32(
                cb_32_lo, cb_32_lo,
            ))),
            cb_32_lo,
        );

        let cr_16 = avx2_pack_u32(
            _mm256_srai_epi32::<PRECISION>(_mm256_permute4x64_epi64::<MASK>(_mm256_hadd_epi32(
                cr_32_lo, cr_32_lo,
            ))),
            cr_32_lo,
        );

        let cb = avx2_pack_u16(cb_16, cb_16);
        let cr = avx2_pack_u16(cr_16, cr_16);

        let row0 = match order {
            YuvNVOrder::UV => {
                _mm_unpacklo_epi8(_mm256_castsi256_si128(cb), _mm256_castsi256_si128(cr))
            }
            YuvNVOrder::VU => {
                _mm_unpacklo_epi8(_mm256_castsi256_si128(cr), _mm256_castsi256_si128(cb))
            }
        };

        _mm_storeu_si64(uv_ptr.add(uv_x) as *mut _, row0);

        uv_x += 8;
        cx += 8;
    }

    ProcessedOffset { cx, ux: uv_x }
}
