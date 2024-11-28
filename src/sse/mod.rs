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
mod rgb_to_nv;
mod rgb_to_y;
mod rgb_to_ycgco;
mod rgb_to_ycgco_r;
mod rgb_to_yuv_p16;
mod rgba_to_yuv;
mod rgba_to_yuv420;
pub(crate) mod sse_support;
mod sse_ycbcr;
mod sse_ycgco_r;
mod ycgco_to_rgb;
mod ycgco_to_rgb_alpha;
mod ycgcor_to_rgb;
mod yuv_nv_p16_to_rgb;
mod yuv_nv_to_rgba;
mod yuv_nv_to_rgba420;
mod yuv_p16_to_rgb16;
mod yuv_p16_to_rgb16_alpha;
mod yuv_to_rgba;
mod yuv_to_rgba420;
mod yuv_to_rgba_alpha;
mod yuv_to_yuy2;
mod yuy2_to_rgb;
mod yuy2_to_yuv;

pub(crate) use gbr_to_rgb::{sse_yuv_to_rgba_row_full, sse_yuv_to_rgba_row_limited};
pub(crate) use rgb_to_nv::sse_rgba_to_nv_row;
pub(crate) use rgb_to_y::sse_rgb_to_y;
pub(crate) use rgb_to_ycgco::sse_rgb_to_ycgco_row;
pub(crate) use rgb_to_ycgco_r::sse_rgb_to_ycgcor_row;
pub(crate) use rgb_to_yuv_p16::{sse_rgba_to_yuv_p16, sse_rgba_to_yuv_p16_lp};
pub(crate) use rgba_to_yuv::sse_rgba_to_yuv_row;
pub(crate) use rgba_to_yuv420::sse_rgba_to_yuv_row420;
pub(crate) use sse_support::*;
pub(crate) use ycgco_to_rgb::sse_ycgco_to_rgb_row;
pub(crate) use ycgco_to_rgb_alpha::sse_ycgco_to_rgb_alpha_row;
pub(crate) use ycgcor_to_rgb::sse_ycgcor_type_to_rgb_row;
pub(crate) use yuv_nv_p16_to_rgb::sse_yuv_nv_p16_to_rgba_row;
pub(crate) use yuv_nv_to_rgba::sse_yuv_nv_to_rgba;
pub(crate) use yuv_nv_to_rgba420::sse_yuv_nv_to_rgba420;
pub(crate) use yuv_p16_to_rgb16::sse_yuv_p16_to_rgba_row;
pub(crate) use yuv_p16_to_rgb16_alpha::sse_yuv_p16_to_rgba_alpha_row;
pub(crate) use yuv_to_rgba::sse_yuv_to_rgba_row;
pub(crate) use yuv_to_rgba420::sse_yuv_to_rgba_row420;
pub(crate) use yuv_to_rgba_alpha::sse_yuv_to_rgba_alpha_row;
pub(crate) use yuv_to_yuy2::yuv_to_yuy2_sse;
pub(crate) use yuy2_to_rgb::yuy2_to_rgb_sse;
pub(crate) use yuy2_to_yuv::yuy2_to_yuv_sse;
