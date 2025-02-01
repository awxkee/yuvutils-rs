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
#[cfg(feature = "nightly_f16")]
mod f16_converter;
mod gbr_to_rgb;
mod rgb_to_nv;
mod rgb_to_nv420;
#[cfg(feature = "professional_mode")]
mod rgb_to_nv420_prof;
#[cfg(feature = "professional_mode")]
mod rgb_to_nv_prof;
mod rgb_to_y;
mod rgb_to_ycgco;
#[cfg(feature = "professional_mode")]
mod rgb_to_yuv420_prof;
mod rgb_to_yuv_p16;
mod rgb_to_yuv_p16_420;
#[cfg(feature = "professional_mode")]
mod rgb_to_yuv_prof;
#[cfg(feature = "fast_mode")]
mod rgba_to_nv_fast;
#[cfg(feature = "fast_mode")]
mod rgba_to_nv_fast420;
mod rgba_to_yuv;
mod rgba_to_yuv420;
#[cfg(feature = "fast_mode")]
mod rgba_to_yuv_fast;
#[cfg(feature = "fast_mode")]
mod rgba_to_yuv_fast420;
mod shuffle;
mod y_to_rgba;
mod y_to_rgba_alpha;
mod ycgco_to_rgb;
mod ycgco_to_rgba_alpha;
mod yuv_nv_to_rgba;
mod yuv_nv_to_rgba420;
#[cfg(feature = "professional_mode")]
mod yuv_nv_to_rgba420_prof;
mod yuv_nv_to_rgba422;
#[cfg(feature = "fast_mode")]
mod yuv_nv_to_rgba_fast;
#[cfg(feature = "fast_mode")]
mod yuv_nv_to_rgba_fast420;
#[cfg(feature = "professional_mode")]
mod yuv_nv_to_rgba_prof;
mod yuv_p16_to_rgb16;
mod yuv_p16_to_rgb16_alpha;
mod yuv_p16_to_rgb8;
mod yuv_p16_to_rgb8_alpha;
#[cfg(feature = "nightly_f16")]
mod yuv_p16_to_rgb_f16;
mod yuv_to_rgba;
mod yuv_to_rgba420;
mod yuv_to_rgba422;
mod yuv_to_rgba_alpha;
mod yuv_to_yuv2;
#[cfg(feature = "nightly_f16")]
mod yuva_p16_to_rgb_f16;
mod yuy2_to_rgb;
mod yuy2_to_yuv;

#[cfg(feature = "nightly_f16")]
pub(crate) use f16_converter::{SurfaceU16ToFloat16Avx2, SurfaceU8ToFloat16Avx2};
pub(crate) use gbr_to_rgb::{avx_yuv_to_rgba_row_full, avx_yuv_to_rgba_row_limited};
pub(crate) use rgb_to_nv::avx2_rgba_to_nv;
pub(crate) use rgb_to_nv420::avx2_rgba_to_nv420;
#[cfg(feature = "professional_mode")]
pub(crate) use rgb_to_nv420_prof::avx2_rgba_to_nv420_prof;
#[cfg(feature = "professional_mode")]
pub(crate) use rgb_to_nv_prof::avx2_rgba_to_nv_prof;
pub(crate) use rgb_to_y::avx2_rgb_to_y_row;
pub(crate) use rgb_to_ycgco::avx2_rgb_to_ycgco_row;
#[cfg(feature = "professional_mode")]
pub(crate) use rgb_to_yuv420_prof::avx2_rgba_to_yuv420_prof;
pub(crate) use rgb_to_yuv_p16::avx_rgba_to_yuv_p16;
pub(crate) use rgb_to_yuv_p16_420::avx_rgba_to_yuv_p16_420;
#[cfg(feature = "professional_mode")]
pub(crate) use rgb_to_yuv_prof::avx2_rgba_to_yuv_prof;
#[cfg(feature = "fast_mode")]
pub(crate) use rgba_to_nv_fast::avx2_rgba_to_nv_fast_rgba;
#[cfg(feature = "fast_mode")]
pub(crate) use rgba_to_nv_fast420::avx2_rgba_to_nv_fast_rgba420;
pub(crate) use rgba_to_yuv::avx2_rgba_to_yuv;
pub(crate) use rgba_to_yuv420::avx2_rgba_to_yuv420;
#[cfg(feature = "fast_mode")]
pub(crate) use rgba_to_yuv_fast::avx2_rgba_to_yuv_dot_rgba;
#[cfg(feature = "fast_mode")]
pub(crate) use rgba_to_yuv_fast420::avx2_rgba_to_yuv_dot_rgba420;
pub(crate) use shuffle::{ShuffleConverterAvx2, ShuffleQTableConverterAvx2};
pub(crate) use y_to_rgba::avx2_y_to_rgba_row;
pub(crate) use y_to_rgba_alpha::avx2_y_to_rgba_alpha_row;
pub(crate) use ycgco_to_rgb::avx2_ycgco_to_rgb_row;
pub(crate) use ycgco_to_rgba_alpha::avx2_ycgco_to_rgba_alpha;
pub(crate) use yuv_nv_to_rgba::avx2_yuv_nv_to_rgba_row;
pub(crate) use yuv_nv_to_rgba420::avx2_yuv_nv_to_rgba_row420;
#[cfg(feature = "professional_mode")]
pub(crate) use yuv_nv_to_rgba420_prof::avx2_yuv_nv_to_rgba_row420_prof;
pub(crate) use yuv_nv_to_rgba422::avx2_yuv_nv_to_rgba_row422;
#[cfg(feature = "fast_mode")]
pub(crate) use yuv_nv_to_rgba_fast::avx_yuv_nv_to_rgba_fast;
#[cfg(feature = "fast_mode")]
pub(crate) use yuv_nv_to_rgba_fast420::avx_yuv_nv_to_rgba_fast420;
#[cfg(feature = "professional_mode")]
pub(crate) use yuv_nv_to_rgba_prof::avx2_yuv_nv_to_rgba_row_prof;
pub(crate) use yuv_p16_to_rgb16::avx_yuv_p16_to_rgba_row;
pub(crate) use yuv_p16_to_rgb16_alpha::avx_yuv_p16_to_rgba_alpha_row;
pub(crate) use yuv_p16_to_rgb8::avx_yuv_p16_to_rgba8_row;
pub(crate) use yuv_p16_to_rgb8_alpha::avx_yuv_p16_to_rgba8_alpha_row;
#[cfg(feature = "nightly_f16")]
pub(crate) use yuv_p16_to_rgb_f16::avx_yuv_p16_to_rgba_f16_row;
pub(crate) use yuv_to_rgba::avx2_yuv_to_rgba_row;
pub(crate) use yuv_to_rgba420::avx2_yuv_to_rgba_row420;
pub(crate) use yuv_to_rgba422::avx2_yuv_to_rgba_row422;
pub(crate) use yuv_to_rgba_alpha::avx2_yuv_to_rgba_alpha;
pub(crate) use yuv_to_yuv2::yuv_to_yuy2_avx2_row;
#[cfg(feature = "nightly_f16")]
pub(crate) use yuva_p16_to_rgb_f16::avx_yuva_p16_to_rgba_f16_row;
pub(crate) use yuy2_to_rgb::yuy2_to_rgb_avx;
pub(crate) use yuy2_to_yuv::yuy2_to_yuv_avx;
