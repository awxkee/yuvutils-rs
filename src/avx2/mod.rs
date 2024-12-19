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
mod avx2_utils;
mod avx2_ycgco;
mod gbr_to_rgb;
mod rgb_to_nv;
mod rgb_to_nv420;
mod rgb_to_y;
mod rgb_to_ycgco;
mod rgb_to_yuv_p16;
mod rgb_to_yuv_p16_420;
mod rgba_to_yuv;
mod rgba_to_yuv420;
mod ycgco_to_rgb;
mod ycgco_to_rgba_alpha;
mod yuv_nv_to_rgba;
mod yuv_nv_to_rgba420;
mod yuv_p16_to_rgb16;
mod yuv_p16_to_rgb16_alpha;
mod yuv_to_rgba;
mod yuv_to_rgba420;
mod yuv_to_rgba_alpha;
mod yuv_to_yuv2;
mod yuy2_to_rgb;
mod yuy2_to_yuv;

pub(crate) use gbr_to_rgb::{avx_yuv_to_rgba_row_full, avx_yuv_to_rgba_row_limited};
pub(crate) use rgb_to_nv::avx2_rgba_to_nv;
pub(crate) use rgb_to_nv420::avx2_rgba_to_nv420;
pub(crate) use rgb_to_y::avx2_rgb_to_y_row;
pub(crate) use rgb_to_ycgco::avx2_rgb_to_ycgco_row;
pub(crate) use rgb_to_yuv_p16::{avx_rgba_to_yuv_p16, avx_rgba_to_yuv_p16_lp};
pub(crate) use rgb_to_yuv_p16_420::{avx_rgba_to_yuv_p16_420, avx_rgba_to_yuv_p16_lp420};
pub(crate) use rgba_to_yuv::avx2_rgba_to_yuv;
pub(crate) use rgba_to_yuv420::avx2_rgba_to_yuv420;
pub(crate) use ycgco_to_rgb::avx2_ycgco_to_rgb_row;
pub(crate) use ycgco_to_rgba_alpha::avx2_ycgco_to_rgba_alpha;
pub(crate) use yuv_nv_to_rgba::avx2_yuv_nv_to_rgba_row;
pub(crate) use yuv_nv_to_rgba420::avx2_yuv_nv_to_rgba_row420;
pub(crate) use yuv_p16_to_rgb16::avx_yuv_p16_to_rgba_row;
pub(crate) use yuv_p16_to_rgb16_alpha::avx_yuv_p16_to_rgba_alpha_row;
pub(crate) use yuv_to_rgba::avx2_yuv_to_rgba_row;
pub(crate) use yuv_to_rgba420::avx2_yuv_to_rgba_row420;
pub(crate) use yuv_to_rgba_alpha::avx2_yuv_to_rgba_alpha;
pub(crate) use yuv_to_yuv2::yuv_to_yuy2_avx2_row;
pub(crate) use yuy2_to_rgb::yuy2_to_rgb_avx;
pub(crate) use yuy2_to_yuv::yuy2_to_yuv_avx;
pub(crate) use avx2_utils::_mm256_interleave_epi8;