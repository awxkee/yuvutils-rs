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

use crate::avx512bw::avx512_utils::{
    avx512_deinterleave_rgb, avx512_deinterleave_rgba, avx512_pack_u16, avx512_pairwise_widen_avg
    ,
};
use crate::internals::ProcessedOffset;
use crate::yuv_support::{
    CbCrForwardTransform, YuvChromaRange, YuvChromaSample, YuvSourceChannels,
};
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[target_feature(enable = "avx512bw")]
pub unsafe fn avx512_rgba_to_yuv<const ORIGIN_CHANNELS: u8, const SAMPLING: u8>(
    transform: &CbCrForwardTransform<i32>,
    range: &YuvChromaRange,
    y_plane: *mut u8,
    u_plane: *mut u8,
    v_plane: *mut u8,
    rgba: &[u8],
    rgba_offset: usize,
    start_cx: usize,
    start_ux: usize,
    width: usize,
    compute_uv_row: bool,
) -> ProcessedOffset {
    let chroma_subsampling: YuvChromaSample = SAMPLING.into();
    let source_channels: YuvSourceChannels = ORIGIN_CHANNELS.into();
    let channels = source_channels.get_channels_count();

    let y_ptr = y_plane;
    let u_ptr = u_plane;
    let v_ptr = v_plane;
    let rgba_ptr = rgba.as_ptr().add(rgba_offset);

    let mut cx = start_cx;
    let mut uv_x = start_ux;

    const V_SHR: u32 = 3;
    const V_SCALE: u32 = 7;
    let rounding_const_bias: i16 = 1 << (V_SHR - 1);
    let bias_y = range.bias_y as i16 * (1 << V_SHR) + rounding_const_bias;
    let bias_uv = range.bias_uv as i16 * (1 << V_SHR) + rounding_const_bias;

    let i_bias_y = _mm512_set1_epi16(range.bias_y as i16);
    let i_cap_y = _mm512_set1_epi16(range.range_y as i16 + range.bias_y as i16);
    let i_cap_uv = _mm512_set1_epi16(range.bias_y as i16 + range.range_uv as i16);

    while cx + 64 < width {
        let y_bias = _mm512_set1_epi16(bias_y);
        let uv_bias = _mm512_set1_epi16(bias_uv);
        let v_yr = _mm512_set1_epi16(transform.yr as i16);
        let v_yg = _mm512_set1_epi16(transform.yg as i16);
        let v_yb = _mm512_set1_epi16(transform.yb as i16);
        let v_cb_r = _mm512_set1_epi16(transform.cb_r as i16);
        let v_cb_g = _mm512_set1_epi16(transform.cb_g as i16);
        let v_cb_b = _mm512_set1_epi16(transform.cb_b as i16);
        let v_cr_r = _mm512_set1_epi16(transform.cr_r as i16);
        let v_cr_g = _mm512_set1_epi16(transform.cr_g as i16);
        let v_cr_b = _mm512_set1_epi16(transform.cr_b as i16);

        let (r_values, g_values, b_values);

        let px = cx * channels;

        match source_channels {
            YuvSourceChannels::Rgb | YuvSourceChannels::Bgr => {
                let source_ptr = rgba_ptr.add(px);
                let row_1 = _mm512_loadu_si512(source_ptr as *const i32);
                let row_2 = _mm512_loadu_si512(source_ptr.add(64) as *const i32);
                let row_3 = _mm512_loadu_si512(source_ptr.add(128) as *const i32);

                let (it1, it2, it3) = avx512_deinterleave_rgb(row_1, row_2, row_3);
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
                let row_1 = _mm512_loadu_si512(source_ptr as *const i32);
                let row_2 = _mm512_loadu_si512(source_ptr.add(64) as *const i32);
                let row_3 = _mm512_loadu_si512(source_ptr.add(128) as *const i32);
                let row_4 = _mm512_loadu_si512(source_ptr.add(128 + 64) as *const i32);

                let (it1, it2, it3, _) = avx512_deinterleave_rgba(row_1, row_2, row_3, row_4);
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
            _mm512_slli_epi16::<V_SCALE>(_mm512_cvtepu8_epi16(_mm512_castsi512_si256(r_values)));
        let r_high = _mm512_slli_epi16::<V_SCALE>(_mm512_cvtepu8_epi16(
            _mm512_extracti64x4_epi64::<1>(r_values),
        ));
        let g_low =
            _mm512_slli_epi16::<V_SCALE>(_mm512_cvtepu8_epi16(_mm512_castsi512_si256(g_values)));
        let g_high = _mm512_slli_epi16::<V_SCALE>(_mm512_cvtepu8_epi16(
            _mm512_extracti64x4_epi64::<1>(g_values),
        ));
        let b_low =
            _mm512_slli_epi16::<V_SCALE>(_mm512_cvtepu8_epi16(_mm512_castsi512_si256(b_values)));
        let b_high = _mm512_slli_epi16::<V_SCALE>(_mm512_cvtepu8_epi16(
            _mm512_extracti64x4_epi64::<1>(b_values),
        ));

        let y_l = _mm512_max_epi16(
            _mm512_min_epi16(
                _mm512_srai_epi16::<V_SHR>(_mm512_add_epi16(
                    y_bias,
                    _mm512_add_epi16(
                        _mm512_add_epi16(
                            _mm512_mulhi_epi16(r_low, v_yr),
                            _mm512_mulhi_epi16(g_low, v_yg),
                        ),
                        _mm512_mulhi_epi16(b_low, v_yb),
                    ),
                )),
                i_cap_y,
            ),
            i_bias_y,
        );

        let y_h = _mm512_max_epi16(
            _mm512_min_epi16(
                _mm512_srai_epi16::<V_SHR>(_mm512_add_epi16(
                    y_bias,
                    _mm512_add_epi16(
                        _mm512_add_epi16(
                            _mm512_mulhi_epi16(r_high, v_yr),
                            _mm512_mulhi_epi16(g_high, v_yg),
                        ),
                        _mm512_mulhi_epi16(b_high, v_yb),
                    ),
                )),
                i_cap_y,
            ),
            i_bias_y,
        );

        let y_yuv = avx512_pack_u16(y_l, y_h);
        _mm512_storeu_si512(y_ptr.add(cx) as *mut i32, y_yuv);

        if compute_uv_row {
            let cb_l = _mm512_max_epi16(
                _mm512_min_epi16(
                    _mm512_srai_epi16::<V_SHR>(_mm512_add_epi16(
                        uv_bias,
                        _mm512_add_epi16(
                            _mm512_add_epi16(
                                _mm512_mulhi_epi16(r_low, v_cb_r),
                                _mm512_mulhi_epi16(g_low, v_cb_g),
                            ),
                            _mm512_mulhi_epi16(b_low, v_cb_b),
                        ),
                    )),
                    i_cap_uv,
                ),
                i_bias_y,
            );
            let cr_l = _mm512_max_epi16(
                _mm512_min_epi16(
                    _mm512_srai_epi16::<V_SHR>(_mm512_add_epi16(
                        uv_bias,
                        _mm512_add_epi16(
                            _mm512_add_epi16(
                                _mm512_mulhi_epi16(r_low, v_cr_r),
                                _mm512_mulhi_epi16(g_low, v_cr_g),
                            ),
                            _mm512_mulhi_epi16(b_low, v_cr_b),
                        ),
                    )),
                    i_cap_uv,
                ),
                i_bias_y,
            );
            let cb_h = _mm512_max_epi16(
                _mm512_min_epi16(
                    _mm512_srai_epi16::<V_SHR>(_mm512_add_epi16(
                        uv_bias,
                        _mm512_add_epi16(
                            _mm512_add_epi16(
                                _mm512_mulhi_epi16(r_high, v_cb_r),
                                _mm512_mulhi_epi16(g_high, v_cb_g),
                            ),
                            _mm512_mulhi_epi16(b_high, v_cb_b),
                        ),
                    )),
                    i_cap_uv,
                ),
                i_bias_y,
            );
            let cr_h = _mm512_max_epi16(
                _mm512_min_epi16(
                    _mm512_srai_epi16::<V_SHR>(_mm512_add_epi16(
                        uv_bias,
                        _mm512_add_epi16(
                            _mm512_add_epi16(
                                _mm512_mulhi_epi16(r_high, v_cr_r),
                                _mm512_mulhi_epi16(g_high, v_cr_g),
                            ),
                            _mm512_mulhi_epi16(b_high, v_cr_b),
                        ),
                    )),
                    i_cap_uv,
                ),
                i_bias_y,
            );

            let cb = avx512_pack_u16(cb_l, cb_h);

            let cr = avx512_pack_u16(cr_l, cr_h);

            match chroma_subsampling {
                YuvChromaSample::Yuv420 | YuvChromaSample::Yuv422 => {
                    let cb_h = _mm512_castsi512_si256(avx512_pairwise_widen_avg(cb));
                    let cr_h = _mm512_castsi512_si256(avx512_pairwise_widen_avg(cr));
                    _mm256_storeu_si256(u_ptr.add(uv_x) as *mut _ as *mut __m256i, cb_h);
                    _mm256_storeu_si256(v_ptr.add(uv_x) as *mut _ as *mut __m256i, cr_h);
                    uv_x += 32;
                }
                YuvChromaSample::Yuv444 => {
                    _mm512_storeu_si512(u_ptr.add(uv_x) as *mut i32, cb);
                    _mm512_storeu_si512(v_ptr.add(uv_x) as *mut i32, cr);
                    uv_x += 64;
                }
            }
        }

        cx += 64;
    }

    ProcessedOffset { cx, ux: uv_x }
}
