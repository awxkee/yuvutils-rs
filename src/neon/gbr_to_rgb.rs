/*
 * Copyright (c) Radzivon Bartoshyk, 11/2024. All rights reserved.
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
use crate::neon::utils::vmullnq_s16;
use crate::yuv_support::YuvSourceChannels;
use std::arch::aarch64::*;

pub(crate) fn yuv_to_rgba_row_limited_rdm<const DESTINATION_CHANNELS: u8>(
    g_plane: &[u8],
    b_plane: &[u8],
    r_plane: &[u8],
    rgba: &mut [u8],
    start_cx: usize,
    width: usize,
    y_bias: i32,
    y_coeff: i32,
) -> usize {
    unsafe {
        yuv_to_rgba_row_limited_impl_rdm::<DESTINATION_CHANNELS>(
            g_plane, b_plane, r_plane, rgba, start_cx, width, y_bias, y_coeff,
        )
    }
}

#[target_feature(enable = "rdm")]
unsafe fn yuv_to_rgba_row_limited_impl_rdm<const DESTINATION_CHANNELS: u8>(
    g_plane: &[u8],
    b_plane: &[u8],
    r_plane: &[u8],
    rgba: &mut [u8],
    start_cx: usize,
    width: usize,
    y_bias: i32,
    y_coeff: i32,
) -> usize {
    let mut cx = start_cx;

    let destination_channels: YuvSourceChannels = DESTINATION_CHANNELS.into();

    let v_alpha = vdupq_n_u8(255u8);

    const V_SCALE: i32 = 2;

    let vy_coeff = vdupq_n_s16(y_coeff as i16);
    let vy_bias = vdupq_n_u8(y_bias as u8);

    while cx + 16 < width {
        let g_values0 = vqsubq_u8(
            vld1q_u8(g_plane.get_unchecked(cx..).as_ptr() as *const _),
            vy_bias,
        );
        let b_values0 = vqsubq_u8(
            vld1q_u8(b_plane.get_unchecked(cx..).as_ptr() as *const _),
            vy_bias,
        );
        let r_values0 = vqsubq_u8(
            vld1q_u8(r_plane.get_unchecked(cx..).as_ptr() as *const _),
            vy_bias,
        );

        let rl_lo = vqrdmulhq_s16(
            vreinterpretq_s16_u16(vshll_n_u8::<V_SCALE>(vget_low_u8(r_values0))),
            vy_coeff,
        );
        let gl_lo = vqrdmulhq_s16(
            vreinterpretq_s16_u16(vshll_n_u8::<V_SCALE>(vget_low_u8(g_values0))),
            vy_coeff,
        );
        let bl_lo = vqrdmulhq_s16(
            vreinterpretq_s16_u16(vshll_n_u8::<V_SCALE>(vget_low_u8(b_values0))),
            vy_coeff,
        );

        let rl_hi = vqrdmulhq_s16(
            vreinterpretq_s16_u16(vshll_high_n_u8::<V_SCALE>(r_values0)),
            vy_coeff,
        );
        let gl_hi = vqrdmulhq_s16(
            vreinterpretq_s16_u16(vshll_high_n_u8::<V_SCALE>(g_values0)),
            vy_coeff,
        );
        let bl_hi = vqrdmulhq_s16(
            vreinterpretq_s16_u16(vshll_high_n_u8::<V_SCALE>(b_values0)),
            vy_coeff,
        );

        let r_values = vcombine_u8(vqmovun_s16(rl_lo), vqmovun_s16(rl_hi));
        let g_values = vcombine_u8(vqmovun_s16(gl_lo), vqmovun_s16(gl_hi));
        let b_values = vcombine_u8(vqmovun_s16(bl_lo), vqmovun_s16(bl_hi));

        let dst_shift = cx * destination_channels.get_channels_count();
        let rgba_ptr = rgba.get_unchecked_mut(dst_shift..);

        match destination_channels {
            YuvSourceChannels::Rgb => {
                let dst_pack: uint8x16x3_t = uint8x16x3_t(r_values, g_values, b_values);
                vst3q_u8(rgba_ptr.as_mut_ptr(), dst_pack);
            }
            YuvSourceChannels::Bgr => {
                let dst_pack: uint8x16x3_t = uint8x16x3_t(b_values, g_values, r_values);
                vst3q_u8(rgba_ptr.as_mut_ptr(), dst_pack);
            }
            YuvSourceChannels::Rgba => {
                let dst_pack: uint8x16x4_t = uint8x16x4_t(r_values, g_values, b_values, v_alpha);
                vst4q_u8(rgba_ptr.as_mut_ptr(), dst_pack);
            }
            YuvSourceChannels::Bgra => {
                let dst_pack: uint8x16x4_t = uint8x16x4_t(b_values, g_values, r_values, v_alpha);
                vst4q_u8(rgba_ptr.as_mut_ptr(), dst_pack);
            }
        }

        cx += 16;
    }

    cx
}

pub(crate) fn yuv_to_rgba_row_limited<const DESTINATION_CHANNELS: u8, const PRECISION: i32>(
    g_plane: &[u8],
    b_plane: &[u8],
    r_plane: &[u8],
    rgba: &mut [u8],
    start_cx: usize,
    width: usize,
    y_bias: i32,
    y_coeff: i32,
) -> usize {
    unsafe {
        let mut cx = start_cx;

        let destination_channels: YuvSourceChannels = DESTINATION_CHANNELS.into();

        let v_alpha = vdupq_n_u8(255u8);

        let vy_coeff = vdupq_n_u16(y_coeff as u16);
        let vy_bias = vdupq_n_u8(y_bias as u8);

        while cx + 16 < width {
            let g_values0 = vqsubq_u8(
                vld1q_u8(g_plane.get_unchecked(cx..).as_ptr() as *const _),
                vy_bias,
            );
            let b_values0 = vqsubq_u8(
                vld1q_u8(b_plane.get_unchecked(cx..).as_ptr() as *const _),
                vy_bias,
            );
            let r_values0 = vqsubq_u8(
                vld1q_u8(r_plane.get_unchecked(cx..).as_ptr() as *const _),
                vy_bias,
            );

            let rl_lo = vmullnq_s16::<PRECISION>(vmovl_u8(vget_low_u8(r_values0)), vy_coeff);
            let gl_lo = vmullnq_s16::<PRECISION>(vmovl_u8(vget_low_u8(g_values0)), vy_coeff);
            let bl_lo = vmullnq_s16::<PRECISION>(vmovl_u8(vget_low_u8(b_values0)), vy_coeff);

            let rl_hi = vmullnq_s16::<PRECISION>(vmovl_high_u8(r_values0), vy_coeff);
            let gl_hi = vmullnq_s16::<PRECISION>(vmovl_high_u8(g_values0), vy_coeff);
            let bl_hi = vmullnq_s16::<PRECISION>(vmovl_high_u8(b_values0), vy_coeff);

            let r_values = vcombine_u8(vqmovn_u16(rl_lo), vqmovn_u16(rl_hi));
            let g_values = vcombine_u8(vqmovn_u16(gl_lo), vqmovn_u16(gl_hi));
            let b_values = vcombine_u8(vqmovn_u16(bl_lo), vqmovn_u16(bl_hi));

            let dst_shift = cx * destination_channels.get_channels_count();
            let rgba_ptr = rgba.get_unchecked_mut(dst_shift..);

            match destination_channels {
                YuvSourceChannels::Rgb => {
                    let dst_pack: uint8x16x3_t = uint8x16x3_t(r_values, g_values, b_values);
                    vst3q_u8(rgba_ptr.as_mut_ptr(), dst_pack);
                }
                YuvSourceChannels::Bgr => {
                    let dst_pack: uint8x16x3_t = uint8x16x3_t(b_values, g_values, r_values);
                    vst3q_u8(rgba_ptr.as_mut_ptr(), dst_pack);
                }
                YuvSourceChannels::Rgba => {
                    let dst_pack: uint8x16x4_t =
                        uint8x16x4_t(r_values, g_values, b_values, v_alpha);
                    vst4q_u8(rgba_ptr.as_mut_ptr(), dst_pack);
                }
                YuvSourceChannels::Bgra => {
                    let dst_pack: uint8x16x4_t =
                        uint8x16x4_t(b_values, g_values, r_values, v_alpha);
                    vst4q_u8(rgba_ptr.as_mut_ptr(), dst_pack);
                }
            }

            cx += 16;
        }

        cx
    }
}

pub(crate) fn yuv_to_rgba_row_full<const DESTINATION_CHANNELS: u8>(
    g_plane: &[u8],
    b_plane: &[u8],
    r_plane: &[u8],
    rgba: &mut [u8],
    start_cx: usize,
    width: usize,
) -> usize {
    unsafe {
        let mut cx = start_cx;

        let destination_channels: YuvSourceChannels = DESTINATION_CHANNELS.into();

        let v_alpha = vdupq_n_u8(255u8);

        while cx + 16 < width {
            let g_values = vld1q_u8(g_plane.get_unchecked(cx..).as_ptr() as *const _);
            let b_values = vld1q_u8(b_plane.get_unchecked(cx..).as_ptr() as *const _);
            let r_values = vld1q_u8(r_plane.get_unchecked(cx..).as_ptr() as *const _);

            let dst_shift = cx * destination_channels.get_channels_count();
            let rgba_ptr = rgba.get_unchecked_mut(dst_shift..);

            match destination_channels {
                YuvSourceChannels::Rgb => {
                    let dst_pack: uint8x16x3_t = uint8x16x3_t(r_values, g_values, b_values);
                    vst3q_u8(rgba_ptr.as_mut_ptr(), dst_pack);
                }
                YuvSourceChannels::Bgr => {
                    let dst_pack: uint8x16x3_t = uint8x16x3_t(b_values, g_values, r_values);
                    vst3q_u8(rgba_ptr.as_mut_ptr(), dst_pack);
                }
                YuvSourceChannels::Rgba => {
                    let dst_pack: uint8x16x4_t =
                        uint8x16x4_t(r_values, g_values, b_values, v_alpha);
                    vst4q_u8(rgba_ptr.as_mut_ptr(), dst_pack);
                }
                YuvSourceChannels::Bgra => {
                    let dst_pack: uint8x16x4_t =
                        uint8x16x4_t(b_values, g_values, r_values, v_alpha);
                    vst4q_u8(rgba_ptr.as_mut_ptr(), dst_pack);
                }
            }

            cx += 16;
        }

        cx
    }
}
