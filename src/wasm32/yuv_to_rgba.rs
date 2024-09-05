/*
 * // Copyright (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

use crate::internals::ProcessedOffset;
use crate::wasm32::transpose::{wasm_store_interleave_u8x3, wasm_store_interleave_u8x4};
use crate::wasm32::utils::{
    u16x8_pack_sat_u8x16, v128_load_half, wasm_unpackhi_i8x16, wasm_unpacklo_i8x16,
};
use crate::yuv_support::{
    CbCrInverseTransform, YuvChromaRange, YuvChromaSample, YuvSourceChannels,
};
use std::arch::wasm32::*;

#[target_feature(enable = "simd128")]
pub unsafe fn wasm_yuv_to_rgba_row<const DESTINATION_CHANNELS: u8, const SAMPLING: u8>(
    range: &YuvChromaRange,
    transform: &CbCrInverseTransform<i32>,
    y_plane: &[u8],
    u_plane: &[u8],
    v_plane: &[u8],
    rgba: &mut [u8],
    start_cx: usize,
    start_ux: usize,
    y_offset: usize,
    u_offset: usize,
    v_offset: usize,
    rgba_offset: usize,
    width: usize,
) -> ProcessedOffset {
    let chroma_subsampling: YuvChromaSample = SAMPLING.into();
    let destination_channels: YuvSourceChannels = DESTINATION_CHANNELS.into();
    let channels = destination_channels.get_channels_count();

    let mut cx = start_cx;
    let mut uv_x = start_ux;

    let y_ptr = y_plane.as_ptr();
    let u_ptr = u_plane.as_ptr();
    let v_ptr = v_plane.as_ptr();
    let rgba_ptr = rgba.as_mut_ptr();

    let y_corr = u8x16_splat(range.bias_y as u8);
    let uv_corr = i16x8_splat(range.bias_uv as i16);
    let v_luma_coeff = u8x16_splat(transform.y_coef as u8);
    let v_cr_coeff = i16x8_splat(transform.cr_coef as i16);
    let v_cb_coeff = i16x8_splat(transform.cb_coef as i16);
    let v_min_values = i16x8_splat(0i16);
    let v_g_coeff_1 = i16x8_splat(-1i16 * transform.g_coeff_1 as i16);
    let v_g_coeff_2 = i16x8_splat(-1i16 * transform.g_coeff_2 as i16);
    let v_alpha = u8x16_splat(255u8);
    let rounding_const = i16x8_splat(1 << 5);

    while cx + 16 < width {
        let y_values = u8x16_sub_sat(v128_load(y_ptr.add(y_offset + cx) as *const v128), y_corr);

        let u_high_u8;
        let v_high_u8;
        let u_low_u8;
        let v_low_u8;

        match chroma_subsampling {
            YuvChromaSample::YUV420 | YuvChromaSample::YUV422 => {
                let u_values = v128_load_half(u_ptr.add(u_offset + uv_x));
                let v_values = v128_load_half(v_ptr.add(v_offset + uv_x));

                u_high_u8 = wasm_unpackhi_i8x16(u_values, u_values);
                v_high_u8 = wasm_unpackhi_i8x16(v_values, v_values);
                u_low_u8 = wasm_unpacklo_i8x16(u_values, u_values);
                v_low_u8 = wasm_unpacklo_i8x16(v_values, v_values);
            }
            YuvChromaSample::YUV444 => {
                let u_values = v128_load(u_ptr.add(u_offset + uv_x) as *const v128);
                let v_values = v128_load(v_ptr.add(v_offset + uv_x) as *const v128);

                u_high_u8 =
                    u8x16_shuffle::<8, 9, 10, 11, 12, 13, 14, 15, 16, 16, 16, 16, 16, 16, 16, 16>(
                        u_values,
                        u8x16_splat(0),
                    );
                v_high_u8 =
                    u8x16_shuffle::<8, 9, 10, 11, 12, 13, 14, 15, 16, 16, 16, 16, 16, 16, 16, 16>(
                        v_values,
                        u8x16_splat(0),
                    );
                u_low_u8 = u_values;
                v_low_u8 = v_values;
            }
        }

        let u_high = i16x8_sub(u16x8_extend_high_u8x16(u_high_u8), uv_corr);
        let v_high = i16x8_sub(u16x8_extend_high_u8x16(v_high_u8), uv_corr);
        let y_high = u16x8_extmul_high_u8x16(y_values, v_luma_coeff);

        let r_high = i16x8_shr(
            i16x8_add_sat(
                i16x8_max(
                    i16x8_add_sat(y_high, i16x8_mul(v_high, v_cr_coeff)),
                    v_min_values,
                ),
                rounding_const,
            ),
            6,
        );
        let b_high = i16x8_shr(
            i16x8_add_sat(
                i16x8_max(
                    i16x8_add_sat(y_high, i16x8_mul(u_high, v_cb_coeff)),
                    v_min_values,
                ),
                rounding_const,
            ),
            6,
        );
        let g_high = i16x8_shr(
            i16x8_add_sat(
                i16x8_max(
                    i16x8_add_sat(
                        y_high,
                        i16x8_add_sat(
                            i16x8_mul(v_high, v_g_coeff_1),
                            i16x8_mul(u_high, v_g_coeff_2),
                        ),
                    ),
                    v_min_values,
                ),
                rounding_const,
            ),
            6,
        );

        let u_low = i16x8_sub(u16x8_extend_low_u8x16(u_low_u8), uv_corr);
        let v_low = i16x8_sub(u16x8_extend_low_u8x16(v_low_u8), uv_corr);
        let y_low = u16x8_extmul_low_u8x16(y_values, v_luma_coeff);

        let r_low = i16x8_shr(
            i16x8_add_sat(
                i16x8_max(
                    i16x8_add_sat(y_low, i16x8_mul(v_low, v_cr_coeff)),
                    v_min_values,
                ),
                rounding_const,
            ),
            6,
        );
        let b_low = i16x8_shr(
            i16x8_add_sat(
                i16x8_max(
                    i16x8_add_sat(y_low, i16x8_mul(u_low, v_cb_coeff)),
                    v_min_values,
                ),
                rounding_const,
            ),
            6,
        );
        let g_low = i16x8_shr(
            i16x8_add_sat(
                i16x8_max(
                    i16x8_add_sat(
                        y_low,
                        i16x8_add_sat(i16x8_mul(v_low, v_g_coeff_1), i16x8_mul(u_low, v_g_coeff_2)),
                    ),
                    v_min_values,
                ),
                rounding_const,
            ),
            6,
        );

        let r_values = u16x8_pack_sat_u8x16(r_low, r_high);
        let g_values = u16x8_pack_sat_u8x16(g_low, g_high);
        let b_values = u16x8_pack_sat_u8x16(b_low, b_high);

        let dst_shift = rgba_offset + cx * channels;

        match destination_channels {
            YuvSourceChannels::Rgb => {
                let dst_pack = (r_values, g_values, b_values);
                wasm_store_interleave_u8x3(rgba_ptr.add(dst_shift), dst_pack);
            }
            YuvSourceChannels::Bgr => {
                let dst_pack = (b_values, g_values, r_values);
                wasm_store_interleave_u8x3(rgba_ptr.add(dst_shift), dst_pack);
            }
            YuvSourceChannels::Rgba => {
                let dst_pack = (r_values, g_values, b_values, v_alpha);
                wasm_store_interleave_u8x4(rgba_ptr.add(dst_shift), dst_pack);
            }
            YuvSourceChannels::Bgra => {
                let dst_pack = (b_values, g_values, r_values, v_alpha);
                wasm_store_interleave_u8x4(rgba_ptr.add(dst_shift), dst_pack);
            }
        }

        cx += 16;

        match chroma_subsampling {
            YuvChromaSample::YUV420 | YuvChromaSample::YUV422 => {
                uv_x += 8;
            }
            YuvChromaSample::YUV444 => {
                uv_x += 16;
            }
        }
    }

    ProcessedOffset { cx, ux: uv_x }
}
