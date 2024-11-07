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
    _mm256_deinterleave_rgba_epi8, _mm256_interleave_x2_epi8, avx2_deinterleave_rgb, avx2_pack_u16,
    avx2_pairwise_widen_avg,
};
use crate::internals::ProcessedOffset;
use crate::yuv_support::{
    CbCrForwardTransform, YuvChromaRange, YuvChromaSubsample, YuvNVOrder, YuvSourceChannels,
};
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

pub fn avx2_rgba_to_nv<const ORIGIN_CHANNELS: u8, const UV_ORDER: u8, const SAMPLING: u8>(
    y_plane: &mut [u8],
    uv_plane: &mut [u8],
    rgba: &[u8],
    width: u32,
    range: &YuvChromaRange,
    transform: &CbCrForwardTransform<i32>,
    start_cx: usize,
    start_ux: usize,
    compute_uv_row: bool,
) -> ProcessedOffset {
    unsafe {
        avx2_rgba_to_nv_impl::<ORIGIN_CHANNELS, UV_ORDER, SAMPLING>(
            y_plane,
            uv_plane,
            rgba,
            width,
            range,
            transform,
            start_cx,
            start_ux,
            compute_uv_row,
        )
    }
}

#[target_feature(enable = "avx2")]
unsafe fn avx2_rgba_to_nv_impl<
    const ORIGIN_CHANNELS: u8,
    const UV_ORDER: u8,
    const SAMPLING: u8,
>(
    y_plane: &mut [u8],
    uv_plane: &mut [u8],
    rgba: &[u8],
    width: u32,
    range: &YuvChromaRange,
    transform: &CbCrForwardTransform<i32>,
    start_cx: usize,
    start_ux: usize,
    compute_uv_row: bool,
) -> ProcessedOffset {
    let order: YuvNVOrder = UV_ORDER.into();
    let chroma_subsampling: YuvChromaSubsample = SAMPLING.into();
    let source_channels: YuvSourceChannels = ORIGIN_CHANNELS.into();
    let channels = source_channels.get_channels_count();

    let y_ptr = y_plane.as_mut_ptr();
    let uv_ptr = uv_plane.as_mut_ptr();
    let rgba_ptr = rgba.as_ptr();

    let mut cx = start_cx;
    let mut uv_x = start_ux;

    const V_SHR: i32 = 3;
    const V_SCALE: i32 = 7;
    let rounding_const_bias: i16 = 1 << (V_SHR - 1);
    let bias_y = range.bias_y as i16 * (1 << V_SHR) + rounding_const_bias;
    let bias_uv = range.bias_uv as i16 * (1 << V_SHR) + rounding_const_bias;

    let i_bias_y = _mm256_set1_epi16(range.bias_y as i16);
    let i_cap_y = _mm256_set1_epi16(range.range_y as i16 + range.bias_y as i16);
    let i_cap_uv = _mm256_set1_epi16(range.bias_y as i16 + range.range_uv as i16);

    let y_bias = _mm256_set1_epi16(bias_y);
    let uv_bias = _mm256_set1_epi16(bias_uv);
    let v_yr = _mm256_set1_epi16(transform.yr as i16);
    let v_yg = _mm256_set1_epi16(transform.yg as i16);
    let v_yb = _mm256_set1_epi16(transform.yb as i16);
    let v_cb_r = _mm256_set1_epi16(transform.cb_r as i16);
    let v_cb_g = _mm256_set1_epi16(transform.cb_g as i16);
    let v_cb_b = _mm256_set1_epi16(transform.cb_b as i16);
    let v_cr_r = _mm256_set1_epi16(transform.cr_r as i16);
    let v_cr_g = _mm256_set1_epi16(transform.cr_g as i16);
    let v_cr_b = _mm256_set1_epi16(transform.cr_b as i16);

    while cx + 32 < width as usize {
        let (r_values, g_values, b_values);

        let px = cx * channels;

        match source_channels {
            YuvSourceChannels::Rgb | YuvSourceChannels::Bgr => {
                let source_ptr = rgba_ptr.add(px);
                let row_1 = _mm256_loadu_si256(source_ptr as *const __m256i);
                let row_2 = _mm256_loadu_si256(source_ptr.add(32) as *const __m256i);
                let row_3 = _mm256_loadu_si256(source_ptr.add(64) as *const __m256i);

                let (it1, it2, it3) = avx2_deinterleave_rgb(row_1, row_2, row_3);
                if source_channels == YuvSourceChannels::Rgb {
                    r_values = it1;
                    g_values = it2;
                    b_values = it3;
                } else {
                    r_values = it3;
                    g_values = it2;
                    b_values = it1;
                }
            }
            YuvSourceChannels::Rgba | YuvSourceChannels::Bgra => {
                let source_ptr = rgba_ptr.add(px);
                let row_1 = _mm256_loadu_si256(source_ptr as *const __m256i);
                let row_2 = _mm256_loadu_si256(source_ptr.add(32) as *const __m256i);
                let row_3 = _mm256_loadu_si256(source_ptr.add(64) as *const __m256i);
                let row_4 = _mm256_loadu_si256(source_ptr.add(96) as *const __m256i);

                let (it1, it2, it3, _) = _mm256_deinterleave_rgba_epi8(row_1, row_2, row_3, row_4);
                if source_channels == YuvSourceChannels::Rgba {
                    r_values = it1;
                    g_values = it2;
                    b_values = it3;
                } else {
                    r_values = it3;
                    g_values = it2;
                    b_values = it1;
                }
            }
        }

        let r_low =
            _mm256_slli_epi16::<V_SCALE>(_mm256_cvtepu8_epi16(_mm256_castsi256_si128(r_values)));
        let r_high = _mm256_slli_epi16::<V_SCALE>(_mm256_cvtepu8_epi16(
            _mm256_extracti128_si256::<1>(r_values),
        ));
        let g_low =
            _mm256_slli_epi16::<V_SCALE>(_mm256_cvtepu8_epi16(_mm256_castsi256_si128(g_values)));
        let g_high = _mm256_slli_epi16::<V_SCALE>(_mm256_cvtepu8_epi16(
            _mm256_extracti128_si256::<1>(g_values),
        ));
        let b_low =
            _mm256_slli_epi16::<V_SCALE>(_mm256_cvtepu8_epi16(_mm256_castsi256_si128(b_values)));
        let b_high = _mm256_slli_epi16::<V_SCALE>(_mm256_cvtepu8_epi16(
            _mm256_extracti128_si256::<1>(b_values),
        ));

        let y_l = _mm256_max_epi16(
            _mm256_min_epi16(
                _mm256_srai_epi16::<V_SHR>(_mm256_add_epi16(
                    y_bias,
                    _mm256_add_epi16(
                        _mm256_add_epi16(
                            _mm256_mulhi_epi16(r_low, v_yr),
                            _mm256_mulhi_epi16(g_low, v_yg),
                        ),
                        _mm256_mulhi_epi16(b_low, v_yb),
                    ),
                )),
                i_cap_y,
            ),
            i_bias_y,
        );

        let y_h = _mm256_max_epi16(
            _mm256_min_epi16(
                _mm256_srai_epi16::<V_SHR>(_mm256_add_epi16(
                    y_bias,
                    _mm256_add_epi16(
                        _mm256_add_epi16(
                            _mm256_mulhi_epi16(r_high, v_yr),
                            _mm256_mulhi_epi16(g_high, v_yg),
                        ),
                        _mm256_mulhi_epi16(b_high, v_yb),
                    ),
                )),
                i_cap_y,
            ),
            i_bias_y,
        );

        let y_yuv = avx2_pack_u16(y_l, y_h);
        _mm256_storeu_si256(y_ptr.add(cx) as *mut __m256i, y_yuv);

        if compute_uv_row {
            let cb_l = _mm256_max_epi16(
                _mm256_min_epi16(
                    _mm256_srai_epi16::<V_SHR>(_mm256_add_epi16(
                        uv_bias,
                        _mm256_add_epi16(
                            _mm256_add_epi16(
                                _mm256_mulhi_epi16(r_low, v_cb_r),
                                _mm256_mulhi_epi16(g_low, v_cb_g),
                            ),
                            _mm256_mulhi_epi16(b_low, v_cb_b),
                        ),
                    )),
                    i_cap_uv,
                ),
                i_bias_y,
            );
            let cr_l = _mm256_max_epi16(
                _mm256_min_epi16(
                    _mm256_srai_epi16::<V_SHR>(_mm256_add_epi16(
                        uv_bias,
                        _mm256_add_epi16(
                            _mm256_add_epi16(
                                _mm256_mulhi_epi16(r_low, v_cr_r),
                                _mm256_mulhi_epi16(g_low, v_cr_g),
                            ),
                            _mm256_mulhi_epi16(b_low, v_cr_b),
                        ),
                    )),
                    i_cap_uv,
                ),
                i_bias_y,
            );
            let cb_h = _mm256_max_epi16(
                _mm256_min_epi16(
                    _mm256_srai_epi16::<V_SHR>(_mm256_add_epi16(
                        uv_bias,
                        _mm256_add_epi16(
                            _mm256_add_epi16(
                                _mm256_mulhi_epi16(r_high, v_cb_r),
                                _mm256_mulhi_epi16(g_high, v_cb_g),
                            ),
                            _mm256_mulhi_epi16(b_high, v_cb_b),
                        ),
                    )),
                    i_cap_uv,
                ),
                i_bias_y,
            );
            let cr_h = _mm256_max_epi16(
                _mm256_min_epi16(
                    _mm256_srai_epi16::<V_SHR>(_mm256_add_epi16(
                        uv_bias,
                        _mm256_add_epi16(
                            _mm256_add_epi16(
                                _mm256_mulhi_epi16(r_high, v_cr_r),
                                _mm256_mulhi_epi16(g_high, v_cr_g),
                            ),
                            _mm256_mulhi_epi16(b_high, v_cr_b),
                        ),
                    )),
                    i_cap_uv,
                ),
                i_bias_y,
            );

            let cb = avx2_pack_u16(cb_l, cb_h);

            let cr = avx2_pack_u16(cr_l, cr_h);

            match chroma_subsampling {
                YuvChromaSubsample::Yuv420 | YuvChromaSubsample::Yuv422 => {
                    let cb_h = avx2_pairwise_widen_avg(cb);
                    let cr_h = avx2_pairwise_widen_avg(cr);
                    let (row0, _) = match order {
                        YuvNVOrder::UV => _mm256_interleave_x2_epi8(cb_h, cr_h),
                        YuvNVOrder::VU => _mm256_interleave_x2_epi8(cr_h, cb_h),
                    };
                    _mm256_storeu_si256(uv_ptr.add(uv_x) as *mut __m256i, row0);
                    uv_x += 32;
                }
                YuvChromaSubsample::Yuv444 => {
                    let (row0, row1) = match order {
                        YuvNVOrder::UV => _mm256_interleave_x2_epi8(cb, cr),
                        YuvNVOrder::VU => _mm256_interleave_x2_epi8(cr, cb),
                    };
                    let dst_ptr = uv_ptr.add(uv_x);
                    _mm256_storeu_si256(dst_ptr as *mut __m256i, row0);
                    _mm256_storeu_si256(dst_ptr.add(32) as *mut __m256i, row1);
                    uv_x += 64;
                }
            }
        }

        cx += 32;
    }

    ProcessedOffset { cx, ux: uv_x }
}
