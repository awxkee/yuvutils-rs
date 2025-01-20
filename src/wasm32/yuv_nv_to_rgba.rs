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
use crate::wasm32::transpose::{v128_load_deinterleave_half_u8_x2, v128_load_deinterleave_u8_x2};
use crate::wasm32::utils::{i16x8_pack_sat_u8x16, wasm_store_rgb, wasm_zip_lo_i8x16};
use crate::yuv_support::{
    to_subsampling, CbCrInverseTransform, YuvChromaRange, YuvChromaSubsampling, YuvNVOrder,
    YuvSourceChannels,
};
use std::arch::wasm32::*;

#[target_feature(enable = "simd128")]
pub(crate) unsafe fn wasm_yuv_nv_to_rgba_row<
    const UV_ORDER: u8,
    const DESTINATION_CHANNELS: u8,
    const YUV_CHROMA_SAMPLING: u8,
>(
    range: &YuvChromaRange,
    transform: &CbCrInverseTransform<i32>,
    y_plane: &[u8],
    uv_plane: &[u8],
    rgba: &mut [u8],
    start_cx: usize,
    start_ux: usize,
    width: usize,
) -> ProcessedOffset {
    let order: YuvNVOrder = UV_ORDER.into();
    let destination_channels: YuvSourceChannels = DESTINATION_CHANNELS.into();
    let chroma_subsampling: YuvChromaSubsampling = to_subsampling(YUV_CHROMA_SAMPLING);
    let channels = destination_channels.get_channels_count();

    let y_ptr = y_plane.as_ptr();
    let uv_ptr = uv_plane.as_ptr();
    let bgra_ptr = rgba.as_mut_ptr();

    let y_corr = u8x16_splat(range.bias_y as u8);
    let uv_corr = i16x8_splat(range.bias_uv as i16);
    let v_luma_coeff = i16x8_splat(transform.y_coef as i16);
    let v_cr_coeff = i16x8_splat(transform.cr_coef as i16);
    let v_cb_coeff = i16x8_splat(transform.cb_coef as i16);

    let v_g_coeff_1 = i16x8_splat(-1i16 * (transform.g_coeff_1 as i16));
    let v_g_coeff_2 = i16x8_splat(-1i16 * (transform.g_coeff_2 as i16));
    let v_alpha = u8x16_splat(255u8);

    const V_SCALE: u32 = 2;

    let mut cx = start_cx;
    let mut ux = start_ux;

    while cx + 16 < width {
        let y_values = u8x16_sub_sat(v128_load(y_ptr.add(cx) as *const v128), y_corr);

        let u_high_u16;
        let v_high_u16;
        let u_low_u16;
        let v_low_u16;

        match chroma_subsampling {
            YuvChromaSubsampling::Yuv420 | YuvChromaSubsampling::Yuv422 => {
                let mut uv_values = v128_load_deinterleave_half_u8_x2(uv_ptr.add(ux));
                if order == YuvNVOrder::VU {
                    uv_values = (uv_values.1, uv_values.0);
                }

                let u_values = wasm_zip_lo_i8x16(uv_values.0, uv_values.0);
                let v_values = wasm_zip_lo_i8x16(uv_values.1, uv_values.1);

                u_high_u16 = u16x8_extend_high_u8x16(u_values);
                v_high_u16 = u16x8_extend_high_u8x16(v_values);
                u_low_u16 = u16x8_extend_low_u8x16(u_values);
                v_low_u16 = u16x8_extend_low_u8x16(v_values);
            }
            YuvChromaSubsampling::Yuv444 => {
                let mut uv_values = v128_load_deinterleave_u8_x2(uv_ptr.add(ux));
                if order == YuvNVOrder::VU {
                    uv_values = (uv_values.1, uv_values.0);
                }
                u_high_u16 = u16x8_extend_high_u8x16(uv_values.0);
                v_high_u16 = u16x8_extend_high_u8x16(uv_values.0);
                u_low_u16 = u16x8_extend_low_u8x16(uv_values.1);
                v_low_u16 = u16x8_extend_low_u8x16(uv_values.1);
            }
        }

        let u_high = i16x8_shl(i16x8_sub(u_high_u16, uv_corr), V_SCALE);
        let v_high = i16x8_shl(i16x8_sub(v_high_u16, uv_corr), V_SCALE);
        let y_high = i16x8_q15mulr_sat(
            i16x8_shl(u16x8_extend_high_u8x16(y_values), V_SCALE),
            v_luma_coeff,
        );

        let r_high = i16x8_add(y_high, i16x8_q15mulr_sat(v_high, v_cr_coeff));
        let b_high = i16x8_add(y_high, i16x8_q15mulr_sat(u_high, v_cb_coeff));
        let g_high = i16x8_add(
            y_high,
            i16x8_add(
                i16x8_q15mulr_sat(v_high, v_g_coeff_1),
                i16x8_q15mulr_sat(u_high, v_g_coeff_2),
            ),
        );

        let u_low = i16x8_shl(i16x8_sub(u_low_u16, uv_corr), V_SCALE);
        let v_low = i16x8_shl(i16x8_sub(v_low_u16, uv_corr), V_SCALE);
        let y_low = i16x8_q15mulr_sat(
            i16x8_shl(u16x8_extend_low_u8x16(y_values), V_SCALE),
            v_luma_coeff,
        );

        let r_low = i16x8_add(y_low, i16x8_q15mulr_sat(v_low, v_cr_coeff));
        let b_low = i16x8_add(y_low, i16x8_q15mulr_sat(u_low, v_cb_coeff));
        let g_low = i16x8_add(
            y_low,
            i16x8_add(
                i16x8_q15mulr_sat(v_low, v_g_coeff_1),
                i16x8_q15mulr_sat(u_low, v_g_coeff_2),
            ),
        );

        let r_values = i16x8_pack_sat_u8x16(r_low, r_high);
        let g_values = i16x8_pack_sat_u8x16(g_low, g_high);
        let b_values = i16x8_pack_sat_u8x16(b_low, b_high);

        let dst_shift = cx * channels;

        wasm_store_rgb::<DESTINATION_CHANNELS>(
            bgra_ptr.add(dst_shift),
            r_values,
            g_values,
            b_values,
            v_alpha,
        );

        cx += 16;

        match chroma_subsampling {
            YuvChromaSubsampling::Yuv420 | YuvChromaSubsampling::Yuv422 => {
                ux += 16;
            }
            YuvChromaSubsampling::Yuv444 => {
                ux += 32;
            }
        }
    }

    ProcessedOffset { cx, ux }
}
