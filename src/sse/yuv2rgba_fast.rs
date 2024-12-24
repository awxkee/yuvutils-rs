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

use crate::internals::{interleaved_epi8, ProcessedOffset};
use crate::sse::{_mm_store_interleave_half_rgb_for_yuv, _mm_store_interleave_rgb_for_yuv};
use crate::yuv_support::{
    CbCrInverseTransform, YuvChromaRange, YuvChromaSubsampling, YuvSourceChannels,
};
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

pub(crate) fn sse_yuv_to_rgba_fast_row<
    const DESTINATION_CHANNELS: u8,
    const SAMPLING: u8,
    const PRECISION: i32,
>(
    range: &YuvChromaRange,
    transform: &CbCrInverseTransform<i32>,
    y_plane: &[u8],
    u_plane: &[u8],
    v_plane: &[u8],
    rgba: &mut [u8],
    start_cx: usize,
    start_ux: usize,
    width: usize,
) -> ProcessedOffset {
    unsafe {
        sse_yuv_to_rgba_fast_row_impl::<DESTINATION_CHANNELS, SAMPLING, PRECISION>(
            range, transform, y_plane, u_plane, v_plane, rgba, start_cx, start_ux, width,
        )
    }
}

#[target_feature(enable = "sse4.1")]
unsafe fn sse_yuv_to_rgba_fast_row_impl<
    const DESTINATION_CHANNELS: u8,
    const SAMPLING: u8,
    const PRECISION: i32,
>(
    range: &YuvChromaRange,
    transform: &CbCrInverseTransform<i32>,
    y_plane: &[u8],
    u_plane: &[u8],
    v_plane: &[u8],
    rgba: &mut [u8],
    start_cx: usize,
    start_ux: usize,
    width: usize,
) -> ProcessedOffset {
    let chroma_subsampling: YuvChromaSubsampling = SAMPLING.into();
    let destination_channels: YuvSourceChannels = DESTINATION_CHANNELS.into();
    let channels = destination_channels.get_channels_count();

    let mut cx = start_cx;
    let mut uv_x = start_ux;

    let y_ptr = y_plane.as_ptr();
    let u_ptr = u_plane.as_ptr();
    let v_ptr = v_plane.as_ptr();
    let rgba_ptr = rgba.as_mut_ptr();

    let y_corr = _mm_set1_epi8(range.bias_y as i8);
    let v_luma_coeff = _mm_set1_epi16((transform.y_coef as u16 * 256) as i16);
    let v_cr_coeff = _mm_set1_epi16(interleaved_epi8(
        transform.cr_coef as i8,
        -transform.cr_coef as i8,
    ));
    let v_cb_coeff = _mm_set1_epi16(interleaved_epi8(
        transform.cb_coef as i8,
        -transform.cb_coef as i8,
    ));
    let v_g_coeff_1 = _mm_set1_epi16(interleaved_epi8(
        transform.g_coeff_1 as i8,
        -transform.g_coeff_1 as i8,
    ));
    let v_g_coeff_2 = _mm_set1_epi16(interleaved_epi8(
        transform.g_coeff_2 as i8,
        -transform.g_coeff_2 as i8,
    ));

    let u_bias_uv = _mm_set1_epi8(range.bias_uv as i8);

    while cx + 16 < width {
        let y_values = _mm_subs_epu8(_mm_loadu_si128(y_ptr.add(cx) as *const __m128i), y_corr);

        let (u_high_u16, v_high_u16, u_low_u16, v_low_u16);

        match chroma_subsampling {
            YuvChromaSubsampling::Yuv420 | YuvChromaSubsampling::Yuv422 => {
                let reshuffle = _mm_setr_epi8(0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7);
                let u_values = _mm_shuffle_epi8(_mm_loadu_si64(u_ptr.add(uv_x)), reshuffle);
                let v_values = _mm_shuffle_epi8(_mm_loadu_si64(v_ptr.add(uv_x)), reshuffle);

                u_high_u16 = _mm_unpackhi_epi8(u_values, u_bias_uv);
                v_high_u16 = _mm_unpackhi_epi8(v_values, u_bias_uv);
                u_low_u16 = _mm_unpacklo_epi8(u_values, u_bias_uv);
                v_low_u16 = _mm_unpacklo_epi8(v_values, u_bias_uv);
            }
            YuvChromaSubsampling::Yuv444 => {
                let u_values = _mm_loadu_si128(u_ptr.add(uv_x) as *const __m128i);
                let v_values = _mm_loadu_si128(v_ptr.add(uv_x) as *const __m128i);

                u_high_u16 = _mm_unpackhi_epi8(u_values, u_bias_uv);
                v_high_u16 = _mm_unpackhi_epi8(v_values, u_bias_uv);
                u_low_u16 = _mm_unpacklo_epi8(u_values, u_bias_uv);
                v_low_u16 = _mm_unpacklo_epi8(v_values, u_bias_uv);
            }
        }

        let y_high = _mm_mulhi_epu16(_mm_unpackhi_epi8(y_values, y_values), v_luma_coeff);

        let r_high = _mm_adds_epi16(y_high, _mm_maddubs_epi16(v_high_u16, v_cr_coeff));
        let b_high = _mm_adds_epi16(y_high, _mm_maddubs_epi16(u_high_u16, v_cb_coeff));
        let g_high = _mm_subs_epi16(
            y_high,
            _mm_adds_epi16(
                _mm_maddubs_epi16(v_high_u16, v_g_coeff_1),
                _mm_maddubs_epi16(u_high_u16, v_g_coeff_2),
            ),
        );

        let y_low = _mm_mulhi_epu16(_mm_unpacklo_epi8(y_values, y_values), v_luma_coeff);

        let r_low = _mm_adds_epi16(y_low, _mm_maddubs_epi16(v_low_u16, v_cr_coeff));

        let b_low = _mm_adds_epi16(y_low, _mm_maddubs_epi16(u_low_u16, v_cb_coeff));
        let g_low = _mm_subs_epi16(
            y_low,
            _mm_adds_epi16(
                _mm_maddubs_epi16(v_low_u16, v_g_coeff_1),
                _mm_maddubs_epi16(u_low_u16, v_g_coeff_2),
            ),
        );

        let r_values = _mm_packus_epi16(
            _mm_srai_epi16::<PRECISION>(r_low),
            _mm_srai_epi16::<PRECISION>(r_high),
        );
        let g_values = _mm_packus_epi16(
            _mm_srai_epi16::<PRECISION>(g_low),
            _mm_srai_epi16::<PRECISION>(g_high),
        );
        let b_values = _mm_packus_epi16(
            _mm_srai_epi16::<PRECISION>(b_low),
            _mm_srai_epi16::<PRECISION>(b_high),
        );

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

        match chroma_subsampling {
            YuvChromaSubsampling::Yuv420 | YuvChromaSubsampling::Yuv422 => {
                uv_x += 8;
            }
            YuvChromaSubsampling::Yuv444 => {
                uv_x += 16;
            }
        }
    }

    while cx + 8 < width {
        let y_values = _mm_subs_epi8(_mm_loadu_si64(y_ptr.add(cx)), y_corr);

        let (u_low_u16, v_low_u16);

        match chroma_subsampling {
            YuvChromaSubsampling::Yuv420 | YuvChromaSubsampling::Yuv422 => {
                let reshuffle = _mm_setr_epi8(0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7);
                let u_value = (u_ptr.add(uv_x) as *const i32).read_unaligned();
                let v_value = (v_ptr.add(uv_x) as *const i32).read_unaligned();
                let u_values = _mm_shuffle_epi8(
                    _mm_insert_epi32::<0>(_mm_setzero_si128(), u_value),
                    reshuffle,
                );
                let v_values = _mm_shuffle_epi8(
                    _mm_insert_epi32::<0>(_mm_setzero_si128(), v_value),
                    reshuffle,
                );

                u_low_u16 = _mm_unpacklo_epi8(u_values, u_bias_uv);
                v_low_u16 = _mm_unpacklo_epi8(v_values, u_bias_uv);
            }
            YuvChromaSubsampling::Yuv444 => {
                let u_values = _mm_loadu_si64(u_ptr.add(uv_x));
                let v_values = _mm_loadu_si64(v_ptr.add(uv_x));

                u_low_u16 = _mm_unpacklo_epi8(u_values, u_bias_uv);
                v_low_u16 = _mm_unpacklo_epi8(v_values, u_bias_uv);
            }
        }

        let y_low = _mm_mulhi_epu16(_mm_unpacklo_epi8(y_values, y_values), v_luma_coeff);

        let r_low = _mm_adds_epi16(y_low, _mm_maddubs_epi16(v_low_u16, v_cr_coeff));
        let b_low = _mm_adds_epi16(y_low, _mm_maddubs_epi16(u_low_u16, v_cb_coeff));
        let g_low = _mm_subs_epi16(
            y_low,
            _mm_adds_epi16(
                _mm_maddubs_epi16(v_low_u16, v_g_coeff_1),
                _mm_maddubs_epi16(u_low_u16, v_g_coeff_2),
            ),
        );

        let r_values = _mm_packus_epi16(_mm_srai_epi16::<PRECISION>(r_low), r_low);
        let g_values = _mm_packus_epi16(_mm_srai_epi16::<PRECISION>(g_low), r_low);
        let b_values = _mm_packus_epi16(_mm_srai_epi16::<PRECISION>(b_low), r_low);

        let dst_shift = cx * channels;

        let v_alpha = _mm_set1_epi8(255u8 as i8);

        _mm_store_interleave_half_rgb_for_yuv::<DESTINATION_CHANNELS>(
            rgba_ptr.add(dst_shift),
            r_values,
            g_values,
            b_values,
            v_alpha,
        );

        cx += 8;

        match chroma_subsampling {
            YuvChromaSubsampling::Yuv420 | YuvChromaSubsampling::Yuv422 => {
                uv_x += 4;
            }
            YuvChromaSubsampling::Yuv444 => {
                uv_x += 8;
            }
        }
    }

    ProcessedOffset { cx, ux: uv_x }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mulhi_addubs() {
        unsafe {
            let v_cr_coeff = _mm_set1_epi16(interleaved_epi8(5, -1));
            let base_val = _mm_set1_epi8(1);
            let ones = _mm_set1_epi8(1);
            let mul_val = _mm_unpacklo_epi8(base_val, ones);
            let product = _mm_maddubs_epi16(mul_val, v_cr_coeff);
            let mut rs: [i16; 8] = [0; 8];
            _mm_storeu_si128(rs.as_mut_ptr() as *mut __m128i, product);
            assert_eq!(rs[0], 4);
        }
    }
}
