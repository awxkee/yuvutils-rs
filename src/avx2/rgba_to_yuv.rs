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
    _mm256_load_deinterleave_half_rgb_for_yuv, _mm256_load_deinterleave_rgb_for_yuv, avx2_pack_u16,
    avx_pairwise_avg_epi16_epi8_j,
};
use crate::internals::ProcessedOffset;
use crate::yuv_support::{
    CbCrForwardTransform, YuvChromaRange, YuvChromaSubsampling, YuvSourceChannels,
};
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

pub(crate) fn avx2_rgba_to_yuv<
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
        avx2_rgba_to_yuv_impl::<ORIGIN_CHANNELS, SAMPLING, PRECISION>(
            transform, range, y_plane, u_plane, v_plane, rgba, start_cx, start_ux, width,
        )
    }
}

#[target_feature(enable = "avx2")]
unsafe fn avx2_rgba_to_yuv_impl<
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
    let chroma_subsampling: YuvChromaSubsampling = SAMPLING.into();
    let source_channels: YuvSourceChannels = ORIGIN_CHANNELS.into();
    let channels = source_channels.get_channels_count();

    let y_ptr = y_plane.as_mut_ptr();
    let u_ptr = u_plane.as_mut_ptr();
    let v_ptr = v_plane.as_mut_ptr();
    let rgba_ptr = rgba.as_ptr();

    let mut cx = start_cx;
    let mut uv_x = start_ux;

    const V_S: i32 = 4;
    const A_E: i32 = 2;
    let y_bias = _mm256_set1_epi16(range.bias_y as i16 * (1 << A_E));
    let uv_bias = _mm256_set1_epi16(range.bias_uv as i16 * (1 << A_E) + (1 << (A_E - 1)) - 1);
    let v_yr = _mm256_set1_epi16(transform.yr as i16);
    let v_yg = _mm256_set1_epi16(transform.yg as i16);
    let v_yb = _mm256_set1_epi16(transform.yb as i16);
    let v_cb_r = _mm256_set1_epi16(transform.cb_r as i16);
    let v_cb_g = _mm256_set1_epi16(transform.cb_g as i16);
    let v_cb_b = _mm256_set1_epi16(transform.cb_b as i16);
    let v_cr_r = _mm256_set1_epi16(transform.cr_r as i16);
    let v_cr_g = _mm256_set1_epi16(transform.cr_g as i16);
    let v_cr_b = _mm256_set1_epi16(transform.cr_b as i16);

    while cx + 32 < width {
        let px = cx * channels;
        let (r_values, g_values, b_values) =
            _mm256_load_deinterleave_rgb_for_yuv::<ORIGIN_CHANNELS>(rgba_ptr.add(px));

        let r_low = _mm256_srli_epi16::<V_S>(_mm256_unpacklo_epi8(r_values, r_values));
        let r_high = _mm256_srli_epi16::<V_S>(_mm256_unpackhi_epi8(r_values, r_values));
        let g_low = _mm256_srli_epi16::<V_S>(_mm256_unpacklo_epi8(g_values, g_values));
        let g_high = _mm256_srli_epi16::<V_S>(_mm256_unpackhi_epi8(g_values, g_values));
        let b_low = _mm256_srli_epi16::<V_S>(_mm256_unpacklo_epi8(b_values, b_values));
        let b_high = _mm256_srli_epi16::<V_S>(_mm256_unpackhi_epi8(b_values, b_values));

        let y_l = _mm256_srli_epi16::<A_E>(_mm256_add_epi16(
            y_bias,
            _mm256_add_epi16(
                _mm256_add_epi16(
                    _mm256_mulhrs_epi16(r_low, v_yr),
                    _mm256_mulhrs_epi16(g_low, v_yg),
                ),
                _mm256_mulhrs_epi16(b_low, v_yb),
            ),
        ));

        let y_h = _mm256_srli_epi16::<A_E>(_mm256_add_epi16(
            y_bias,
            _mm256_add_epi16(
                _mm256_add_epi16(
                    _mm256_mulhrs_epi16(r_high, v_yr),
                    _mm256_mulhrs_epi16(g_high, v_yg),
                ),
                _mm256_mulhrs_epi16(b_high, v_yb),
            ),
        ));

        let y_yuv = _mm256_packus_epi16(y_l, y_h);
        _mm256_storeu_si256(y_ptr.add(cx) as *mut __m256i, y_yuv);

        if chroma_subsampling == YuvChromaSubsampling::Yuv444 {
            let cb_l = _mm256_srli_epi16::<A_E>(_mm256_add_epi16(
                uv_bias,
                _mm256_add_epi16(
                    _mm256_add_epi16(
                        _mm256_mulhrs_epi16(r_low, v_cb_r),
                        _mm256_mulhrs_epi16(g_low, v_cb_g),
                    ),
                    _mm256_mulhrs_epi16(b_low, v_cb_b),
                ),
            ));
            let cr_l = _mm256_srli_epi16::<A_E>(_mm256_add_epi16(
                uv_bias,
                _mm256_add_epi16(
                    _mm256_add_epi16(
                        _mm256_mulhrs_epi16(r_low, v_cr_r),
                        _mm256_mulhrs_epi16(g_low, v_cr_g),
                    ),
                    _mm256_mulhrs_epi16(b_low, v_cr_b),
                ),
            ));
            let cb_h = _mm256_srli_epi16::<A_E>(_mm256_add_epi16(
                uv_bias,
                _mm256_add_epi16(
                    _mm256_add_epi16(
                        _mm256_mulhrs_epi16(r_high, v_cb_r),
                        _mm256_mulhrs_epi16(g_high, v_cb_g),
                    ),
                    _mm256_mulhrs_epi16(b_high, v_cb_b),
                ),
            ));
            let cr_h = _mm256_srli_epi16::<A_E>(_mm256_add_epi16(
                uv_bias,
                _mm256_add_epi16(
                    _mm256_add_epi16(
                        _mm256_mulhrs_epi16(r_high, v_cr_r),
                        _mm256_mulhrs_epi16(g_high, v_cr_g),
                    ),
                    _mm256_mulhrs_epi16(b_high, v_cr_b),
                ),
            ));

            let cb = _mm256_packus_epi16(cb_l, cb_h);
            let cr = _mm256_packus_epi16(cr_l, cr_h);

            _mm256_storeu_si256(u_ptr.add(uv_x) as *mut __m256i, cb);
            _mm256_storeu_si256(v_ptr.add(uv_x) as *mut __m256i, cr);
            uv_x += 32;
        } else if chroma_subsampling == YuvChromaSubsampling::Yuv422
            || (chroma_subsampling == YuvChromaSubsampling::Yuv420)
        {
            let r1 = avx_pairwise_avg_epi16_epi8_j(r_values, 1 << (16 - V_S - 8 - 1));
            let g1 = avx_pairwise_avg_epi16_epi8_j(g_values, 1 << (16 - V_S - 8 - 1));
            let b1 = avx_pairwise_avg_epi16_epi8_j(b_values, 1 << (16 - V_S - 8 - 1));

            let cb = _mm256_srli_epi16::<A_E>(_mm256_add_epi16(
                uv_bias,
                _mm256_add_epi16(
                    _mm256_add_epi16(
                        _mm256_mulhrs_epi16(r1, v_cb_r),
                        _mm256_mulhrs_epi16(g1, v_cb_g),
                    ),
                    _mm256_mulhrs_epi16(b1, v_cb_b),
                ),
            ));
            let cr = _mm256_srli_epi16::<A_E>(_mm256_add_epi16(
                uv_bias,
                _mm256_add_epi16(
                    _mm256_add_epi16(
                        _mm256_mulhrs_epi16(r1, v_cr_r),
                        _mm256_mulhrs_epi16(g1, v_cr_g),
                    ),
                    _mm256_mulhrs_epi16(b1, v_cr_b),
                ),
            ));

            let cb = avx2_pack_u16(cb, cb);
            let cr = avx2_pack_u16(cr, cr);

            _mm_storeu_si128(
                u_ptr.add(uv_x) as *mut _ as *mut __m128i,
                _mm256_castsi256_si128(cb),
            );
            _mm_storeu_si128(
                v_ptr.add(uv_x) as *mut _ as *mut __m128i,
                _mm256_castsi256_si128(cr),
            );
            uv_x += 16;
        }

        cx += 32;
    }

    while cx + 16 < width {
        let px = cx * channels;
        let (mut r_values, mut g_values, mut b_values) =
            _mm256_load_deinterleave_half_rgb_for_yuv::<ORIGIN_CHANNELS>(rgba_ptr.add(px));

        let (r_v, g_v, b_v) = (r_values, g_values, b_values);

        r_values = _mm256_permute4x64_epi64::<0x50>(r_values);
        g_values = _mm256_permute4x64_epi64::<0x50>(g_values);
        b_values = _mm256_permute4x64_epi64::<0x50>(b_values);

        let r_low = _mm256_srli_epi16::<V_S>(_mm256_unpacklo_epi8(r_values, r_values));
        let g_low = _mm256_srli_epi16::<V_S>(_mm256_unpacklo_epi8(g_values, g_values));
        let b_low = _mm256_srli_epi16::<V_S>(_mm256_unpacklo_epi8(b_values, b_values));

        let y_l = _mm256_srli_epi16::<A_E>(_mm256_add_epi16(
            y_bias,
            _mm256_add_epi16(
                _mm256_add_epi16(
                    _mm256_mulhrs_epi16(r_low, v_yr),
                    _mm256_mulhrs_epi16(g_low, v_yg),
                ),
                _mm256_mulhrs_epi16(b_low, v_yb),
            ),
        ));

        let y_yuv = avx2_pack_u16(y_l, y_l);
        _mm_storeu_si128(y_ptr.add(cx) as *mut __m128i, _mm256_castsi256_si128(y_yuv));

        if chroma_subsampling == YuvChromaSubsampling::Yuv444 {
            let cb_l = _mm256_srli_epi16::<A_E>(_mm256_add_epi16(
                uv_bias,
                _mm256_add_epi16(
                    _mm256_add_epi16(
                        _mm256_mulhrs_epi16(r_low, v_cb_r),
                        _mm256_mulhrs_epi16(g_low, v_cb_g),
                    ),
                    _mm256_mulhrs_epi16(b_low, v_cb_b),
                ),
            ));
            let cr_l = _mm256_srli_epi16::<A_E>(_mm256_add_epi16(
                uv_bias,
                _mm256_add_epi16(
                    _mm256_add_epi16(
                        _mm256_mulhrs_epi16(r_low, v_cr_r),
                        _mm256_mulhrs_epi16(g_low, v_cr_g),
                    ),
                    _mm256_mulhrs_epi16(b_low, v_cr_b),
                ),
            ));

            let cb = avx2_pack_u16(cb_l, cb_l);
            let cr = avx2_pack_u16(cr_l, cr_l);

            _mm_storeu_si128(u_ptr.add(uv_x) as *mut __m128i, _mm256_castsi256_si128(cb));
            _mm_storeu_si128(v_ptr.add(uv_x) as *mut __m128i, _mm256_castsi256_si128(cr));
            uv_x += 16;
        } else if chroma_subsampling == YuvChromaSubsampling::Yuv422
            || (chroma_subsampling == YuvChromaSubsampling::Yuv420)
        {
            let r1 = avx_pairwise_avg_epi16_epi8_j(r_v, 1 << (16 - V_S - 8 - 1));
            let g1 = avx_pairwise_avg_epi16_epi8_j(g_v, 1 << (16 - V_S - 8 - 1));
            let b1 = avx_pairwise_avg_epi16_epi8_j(b_v, 1 << (16 - V_S - 8 - 1));

            let cb = _mm256_srli_epi16::<A_E>(_mm256_add_epi16(
                uv_bias,
                _mm256_add_epi16(
                    _mm256_add_epi16(
                        _mm256_mulhrs_epi16(r1, v_cb_r),
                        _mm256_mulhrs_epi16(g1, v_cb_g),
                    ),
                    _mm256_mulhrs_epi16(b1, v_cb_b),
                ),
            ));

            let cr = _mm256_srli_epi16::<A_E>(_mm256_add_epi16(
                uv_bias,
                _mm256_add_epi16(
                    _mm256_add_epi16(
                        _mm256_mulhrs_epi16(r1, v_cr_r),
                        _mm256_mulhrs_epi16(g1, v_cr_g),
                    ),
                    _mm256_mulhrs_epi16(b1, v_cr_b),
                ),
            ));

            let cb = _mm256_castsi256_si128(avx2_pack_u16(cb, cb));
            let cr = _mm256_castsi256_si128(avx2_pack_u16(cr, cr));

            _mm_storeu_si64(u_ptr.add(uv_x), cb);
            _mm_storeu_si64(v_ptr.add(uv_x), cr);

            uv_x += 8;
        }

        cx += 16;
    }

    ProcessedOffset { cx, ux: uv_x }
}
