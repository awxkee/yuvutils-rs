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
#![deny(unreachable_code, unreachable_pub)]
mod gbr_to_rgb;
mod neon_ycgco;
mod rgb_to_y;
mod rgb_to_ycgco;
mod rgb_to_yuv_p16;
mod rgb_to_yuv_p16_420;
mod rgba_to_nv;
mod rgba_to_nv420;
mod rgba_to_yuv;
mod rgba_to_yuv420;
mod shuffle;
mod utils;
mod y_p16_to_rgba16;
mod y_to_rgb;
mod y_to_rgb_alpha;
mod ycgco_to_rgb;
mod ycgco_to_rgb_alpha;
mod yuv_nv_p10_to_rgba;
mod yuv_nv_p16_to_rgb;
mod yuv_nv_to_rgba;
mod yuv_nv_to_rgba420;
mod yuv_p16_to_rgba16;
mod yuv_p16_to_rgba16_alpha;
mod yuv_p16_to_rgba8;
mod yuv_p16_to_rgba_alpha;
mod yuv_to_rgba;
mod yuv_to_rgba420;
mod yuv_to_rgba_alpha;
mod yuv_to_yuy2;
mod yuy2_to_rgb;
mod yuy2_to_yuv;

pub(crate) use gbr_to_rgb::{
    yuv_to_rgba_row_full, yuv_to_rgba_row_limited, yuv_to_rgba_row_limited_rdm,
};
pub(crate) use rgb_to_y::{neon_rgb_to_y_rdm, neon_rgb_to_y_row};
pub(crate) use rgb_to_ycgco::neon_rgb_to_ycgco_row;
pub(crate) use rgb_to_yuv_p16::{neon_rgba_to_yuv_p16, neon_rgba_to_yuv_p16_rdm};
pub(crate) use rgb_to_yuv_p16_420::{neon_rgba_to_yuv_p16_420, neon_rgba_to_yuv_p16_rdm_420};
pub(crate) use rgba_to_nv::{neon_rgbx_to_nv_row, neon_rgbx_to_nv_row_rdm};
pub(crate) use rgba_to_nv420::{neon_rgbx_to_nv_row420, neon_rgbx_to_nv_row_rdm420};
pub(crate) use rgba_to_yuv::{neon_rgba_to_yuv, neon_rgba_to_yuv_rdm};
pub(crate) use rgba_to_yuv420::{neon_rgba_to_yuv420, neon_rgba_to_yuv_rdm420};
pub(crate) use shuffle::ShuffleConverterNeon;
pub(crate) use y_p16_to_rgba16::neon_y_p16_to_rgba16_row;
pub(crate) use y_to_rgb::{neon_y_to_rgb_row, neon_y_to_rgb_row_rdm};
pub(crate) use y_to_rgb_alpha::{neon_y_to_rgb_alpha_row, neon_y_to_rgb_row_alpha_rdm};
pub(crate) use ycgco_to_rgb::neon_ycgco_to_rgb_row;
pub(crate) use ycgco_to_rgb_alpha::neon_ycgco_to_rgb_alpha_row;
pub(crate) use yuv_nv_p10_to_rgba::neon_yuv_nv12_p10_to_rgba_row;
pub(crate) use yuv_nv_p16_to_rgb::{neon_yuv_nv_p16_to_rgba_row, neon_yuv_nv_p16_to_rgba_row_rdm};
pub(crate) use yuv_nv_to_rgba::{neon_yuv_nv_to_rgba_row, neon_yuv_nv_to_rgba_row_rdm};
pub(crate) use yuv_nv_to_rgba420::{neon_yuv_nv_to_rgba_row420, neon_yuv_nv_to_rgba_row_rdm420};
pub(crate) use yuv_p16_to_rgba16::{neon_yuv_p16_to_rgba16_row, neon_yuv_p16_to_rgba16_row_rdm};
pub(crate) use yuv_p16_to_rgba16_alpha::{
    neon_yuv_p16_to_rgba16_alpha_row, neon_yuv_p16_to_rgba16_alpha_row_rdm,
};
pub(crate) use yuv_p16_to_rgba8::neon_yuv_p16_to_rgba_row;
pub(crate) use yuv_p16_to_rgba_alpha::neon_yuv_p16_to_rgba_alpha_row;
pub(crate) use yuv_to_rgba::{neon_yuv_to_rgba_row, neon_yuv_to_rgba_row_rdm};
pub(crate) use yuv_to_rgba420::{neon_yuv_to_rgba_row420, neon_yuv_to_rgba_row_rdm420};
pub(crate) use yuv_to_rgba_alpha::{neon_yuv_to_rgba_alpha, neon_yuv_to_rgba_alpha_rdm};
pub(crate) use yuv_to_yuy2::yuv_to_yuy2_neon_impl;
pub(crate) use yuy2_to_rgb::yuy2_to_rgb_neon;
pub(crate) use yuy2_to_yuv::yuy2_to_yuv_neon_impl;
