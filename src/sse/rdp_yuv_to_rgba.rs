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

use crate::internals::ProcessedOffset;
use crate::sse::sse_support::{sse_store_rgb_u8, sse_store_rgba};
use crate::yuv_support::{CbCrInverseTransform, YuvSourceChannels};
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

pub fn rdp_sse_yuv_to_rgba_row<const DESTINATION_CHANNELS: u8>(
    transform: &CbCrInverseTransform<i32>,
    y_plane: &[u16],
    u_plane: &[u16],
    v_plane: &[u16],
    rgba: &mut [u8],
    start_cx: usize,
    width: usize,
) -> ProcessedOffset {
    unsafe {
        rdp_sse_yuv_to_rgba_row_impl::<DESTINATION_CHANNELS>(
            transform, y_plane, u_plane, v_plane, rgba, start_cx, width,
        )
    }
}

#[target_feature(enable = "sse4.1")]
unsafe fn rdp_sse_yuv_to_rgba_row_impl<const DESTINATION_CHANNELS: u8>(
    transform: &CbCrInverseTransform<i32>,
    y_plane: &[u16],
    u_plane: &[u16],
    v_plane: &[u16],
    rgba: &mut [u8],
    start_cx: usize,
    width: usize,
) -> ProcessedOffset {
    let destination_channels: YuvSourceChannels = DESTINATION_CHANNELS.into();
    let channels = destination_channels.get_channels_count();

    let mut cx = start_cx;

    let rgba_ptr = rgba.as_mut_ptr();

    let y_corr = _mm_set1_epi16(4096);
    let v_cr_coeff = _mm_set1_epi16(transform.cr_coef as i16);
    let v_cb_coeff = _mm_set1_epi16(transform.cb_coef as i16);
    let v_g_coeff_1 = _mm_set1_epi16(transform.g_coeff_1 as i16);
    let v_g_coeff_2 = _mm_set1_epi16(transform.g_coeff_2 as i16);
    let v_alpha = _mm_set1_epi8(255u8 as i8);

    let zeros = _mm_setzero_si128();

    const V_SHR: i32 = 6;
    const UV_SCALE: i32 = 5;

    let rounding_const = _mm_set1_epi16(1 << (V_SHR - 1));

    while cx + 16 < width {
        let y_values0 = _mm_slli_epi16::<1>(_mm_add_epi16(
            _mm_loadu_si128(y_plane.get_unchecked(cx..).as_ptr() as *const __m128i),
            y_corr,
        ));
        let y_values1 = _mm_slli_epi16::<1>(_mm_add_epi16(
            _mm_loadu_si128(y_plane.get_unchecked((cx + 8)..).as_ptr() as *const __m128i),
            y_corr,
        ));

        let u_values_lo = _mm_loadu_si128(u_plane.get_unchecked(cx..).as_ptr() as *const __m128i);
        let u_values_hi =
            _mm_loadu_si128(u_plane.get_unchecked((cx + 8)..).as_ptr() as *const __m128i);
        let v_values_lo = _mm_loadu_si128(v_plane.get_unchecked(cx..).as_ptr() as *const __m128i);
        let v_values_hi =
            _mm_loadu_si128(v_plane.get_unchecked((cx + 8)..).as_ptr() as *const __m128i);

        let r_high = _mm_srai_epi16::<V_SHR>(_mm_add_epi16(
            _mm_max_epi16(
                _mm_add_epi16(
                    y_values1,
                    _mm_slli_epi16::<UV_SCALE>(_mm_mulhi_epi16(v_values_hi, v_cr_coeff)),
                ),
                zeros,
            ),
            rounding_const,
        ));
        let b_high = _mm_srai_epi16::<V_SHR>(_mm_adds_epi16(
            _mm_max_epi16(
                _mm_add_epi16(
                    y_values1,
                    _mm_slli_epi16::<UV_SCALE>(_mm_mulhi_epi16(u_values_hi, v_cb_coeff)),
                ),
                zeros,
            ),
            rounding_const,
        ));
        let g_high = _mm_srai_epi16::<V_SHR>(_mm_add_epi16(
            _mm_max_epi16(
                _mm_sub_epi16(
                    y_values1,
                    _mm_add_epi16(
                        _mm_slli_epi16::<UV_SCALE>(_mm_mulhi_epi16(u_values_hi, v_g_coeff_1)),
                        _mm_slli_epi16::<UV_SCALE>(_mm_mulhi_epi16(v_values_hi, v_g_coeff_2)),
                    ),
                ),
                zeros,
            ),
            rounding_const,
        ));

        let r_low = _mm_srai_epi16::<V_SHR>(_mm_add_epi16(
            _mm_max_epi16(
                _mm_add_epi16(
                    y_values0,
                    _mm_slli_epi16::<UV_SCALE>(_mm_mulhi_epi16(v_values_lo, v_cr_coeff)),
                ),
                zeros,
            ),
            rounding_const,
        ));
        let b_low = _mm_srai_epi16::<V_SHR>(_mm_add_epi16(
            _mm_max_epi16(
                _mm_add_epi16(
                    y_values0,
                    _mm_slli_epi16::<UV_SCALE>(_mm_mulhi_epi16(u_values_lo, v_cb_coeff)),
                ),
                zeros,
            ),
            rounding_const,
        ));
        let g_low = _mm_srai_epi16::<V_SHR>(_mm_add_epi16(
            _mm_max_epi16(
                _mm_sub_epi16(
                    y_values0,
                    _mm_add_epi16(
                        _mm_slli_epi16::<UV_SCALE>(_mm_mulhi_epi16(u_values_lo, v_g_coeff_1)),
                        _mm_slli_epi16::<UV_SCALE>(_mm_mulhi_epi16(v_values_lo, v_g_coeff_2)),
                    ),
                ),
                zeros,
            ),
            rounding_const,
        ));

        let r_values = _mm_packus_epi16(r_low, r_high);
        let g_values = _mm_packus_epi16(g_low, g_high);
        let b_values = _mm_packus_epi16(b_low, b_high);

        let dst_shift = cx * channels;

        match destination_channels {
            YuvSourceChannels::Rgb => {
                sse_store_rgb_u8(rgba_ptr.add(dst_shift), r_values, g_values, b_values);
            }
            YuvSourceChannels::Bgr => {
                sse_store_rgb_u8(rgba_ptr.add(dst_shift), b_values, g_values, r_values);
            }
            YuvSourceChannels::Rgba => {
                sse_store_rgba(
                    rgba_ptr.add(dst_shift),
                    r_values,
                    g_values,
                    b_values,
                    v_alpha,
                );
            }
            YuvSourceChannels::Bgra => {
                sse_store_rgba(
                    rgba_ptr.add(dst_shift),
                    b_values,
                    g_values,
                    r_values,
                    v_alpha,
                );
            }
        }

        cx += 16;
    }

    ProcessedOffset { cx, ux: start_cx }
}
