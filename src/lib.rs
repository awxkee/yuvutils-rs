/*
 * // Copyright (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */
#![cfg_attr(feature = "nightly_avx512", feature(cfg_version))]
#![cfg_attr(feature = "nightly_avx512", feature(avx512_target_feature))]
#![cfg_attr(feature = "nightly_avx512", feature(stdarch_x86_avx512))]

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod avx2;
#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    feature = "nightly_avx512"
))]
mod avx512bw;
mod from_identity;
mod internals;
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
mod neon;
mod rgb_to_nv_p16;
mod rgb_to_y;
mod rgb_to_ycgco;
mod rgb_to_ycgco_r;
mod rgb_to_yuv_p16;
mod rgba_to_nv;
mod rgba_to_yuv;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod sse;
mod to_identity;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod wasm32;
mod y_to_rgb;
mod ycgco_r_to_rgb;
mod ycgco_to_rgb;
mod ycgco_to_rgb_alpha;
mod ycgcor_support;
mod yuv_nv_p10_to_rgba;
mod yuv_nv_p16_to_rgb;
mod yuv_nv_to_rgba;
mod yuv_p10_rgba;
mod yuv_support;
mod yuv_to_rgba;
mod yuv_to_rgba_alpha;
mod yuv_to_yuy2;
mod yuy2_to_rgb;
mod yuy2_to_yuv;

pub use yuv_support::YuvBytesPacking;
pub use yuv_support::YuvEndianness;
pub use yuv_support::YuvRange;
pub use yuv_support::YuvStandardMatrix;

pub use yuv_nv_p10_to_rgba::yuv_nv12_p10_to_bgr;
pub use yuv_nv_p10_to_rgba::yuv_nv12_p10_to_bgra;
pub use yuv_nv_p10_to_rgba::yuv_nv12_p10_to_rgb;
pub use yuv_nv_p10_to_rgba::yuv_nv12_p10_to_rgba;
pub use yuv_nv_p10_to_rgba::yuv_nv16_p10_to_bgr;
pub use yuv_nv_p10_to_rgba::yuv_nv16_p10_to_bgra;
pub use yuv_nv_p10_to_rgba::yuv_nv16_p10_to_rgb;
pub use yuv_nv_p10_to_rgba::yuv_nv16_p10_to_rgba;
pub use yuv_nv_p10_to_rgba::yuv_nv21_p10_to_bgr;
pub use yuv_nv_p10_to_rgba::yuv_nv21_p10_to_bgra;
pub use yuv_nv_p10_to_rgba::yuv_nv21_p10_to_rgb;
pub use yuv_nv_p10_to_rgba::yuv_nv21_p10_to_rgba;
pub use yuv_nv_p10_to_rgba::yuv_nv61_p10_to_bgr;
pub use yuv_nv_p10_to_rgba::yuv_nv61_p10_to_bgra;
pub use yuv_nv_p10_to_rgba::yuv_nv61_p10_to_rgb;
pub use yuv_nv_p10_to_rgba::yuv_nv61_p10_to_rgba;

pub use yuv_nv_p16_to_rgb::yuv_nv12_to_bgr_p16;
pub use yuv_nv_p16_to_rgb::yuv_nv12_to_bgra_p16;
pub use yuv_nv_p16_to_rgb::yuv_nv12_to_rgb_p16;
pub use yuv_nv_p16_to_rgb::yuv_nv12_to_rgba_p16;
pub use yuv_nv_p16_to_rgb::yuv_nv16_to_bgr_p16;
pub use yuv_nv_p16_to_rgb::yuv_nv16_to_bgra_p16;
pub use yuv_nv_p16_to_rgb::yuv_nv16_to_rgb_p16;
pub use yuv_nv_p16_to_rgb::yuv_nv16_to_rgba_p16;
pub use yuv_nv_p16_to_rgb::yuv_nv21_to_bgr_p16;
pub use yuv_nv_p16_to_rgb::yuv_nv21_to_bgra_p16;
pub use yuv_nv_p16_to_rgb::yuv_nv21_to_rgb_p16;
pub use yuv_nv_p16_to_rgb::yuv_nv21_to_rgba_p16;
pub use yuv_nv_p16_to_rgb::yuv_nv61_to_bgr_p16;
pub use yuv_nv_p16_to_rgb::yuv_nv61_to_bgra_p16;
pub use yuv_nv_p16_to_rgb::yuv_nv61_to_rgb_p16;
pub use yuv_nv_p16_to_rgb::yuv_nv61_to_rgba_p16;

pub use yuv_nv_to_rgba::yuv_nv12_to_bgr;
pub use yuv_nv_to_rgba::yuv_nv12_to_bgra;
pub use yuv_nv_to_rgba::yuv_nv12_to_rgb;
pub use yuv_nv_to_rgba::yuv_nv12_to_rgba;
pub use yuv_nv_to_rgba::yuv_nv21_to_bgr;
pub use yuv_nv_to_rgba::yuv_nv21_to_bgra;
pub use yuv_nv_to_rgba::yuv_nv21_to_rgb;
pub use yuv_nv_to_rgba::yuv_nv21_to_rgba;
pub use yuv_nv_to_rgba::yuv_nv24_to_bgr;
pub use yuv_nv_to_rgba::yuv_nv24_to_bgra;
pub use yuv_nv_to_rgba::yuv_nv24_to_rgb;
pub use yuv_nv_to_rgba::yuv_nv24_to_rgba;
pub use yuv_nv_to_rgba::yuv_nv42_to_bgr;
pub use yuv_nv_to_rgba::yuv_nv42_to_bgra;
pub use yuv_nv_to_rgba::yuv_nv42_to_rgb;
pub use yuv_nv_to_rgba::yuv_nv42_to_rgba;

pub use rgba_to_nv::bgr_to_yuv_nv12;
pub use rgba_to_nv::bgr_to_yuv_nv16;
pub use rgba_to_nv::bgr_to_yuv_nv21;
pub use rgba_to_nv::bgr_to_yuv_nv24;
pub use rgba_to_nv::bgr_to_yuv_nv42;
pub use rgba_to_nv::bgr_to_yuv_nv61;
pub use rgba_to_nv::bgra_to_yuv_nv12;
pub use rgba_to_nv::bgra_to_yuv_nv16;
pub use rgba_to_nv::bgra_to_yuv_nv21;
pub use rgba_to_nv::bgra_to_yuv_nv24;
pub use rgba_to_nv::bgra_to_yuv_nv42;
pub use rgba_to_nv::bgra_to_yuv_nv61;
pub use rgba_to_nv::rgb_to_yuv_nv12;
pub use rgba_to_nv::rgb_to_yuv_nv16;
pub use rgba_to_nv::rgb_to_yuv_nv21;
pub use rgba_to_nv::rgb_to_yuv_nv24;
pub use rgba_to_nv::rgb_to_yuv_nv42;
pub use rgba_to_nv::rgb_to_yuv_nv61;
pub use rgba_to_nv::rgba_to_yuv_nv12;
pub use rgba_to_nv::rgba_to_yuv_nv16;
pub use rgba_to_nv::rgba_to_yuv_nv21;
pub use rgba_to_nv::rgba_to_yuv_nv24;
pub use rgba_to_nv::rgba_to_yuv_nv42;
pub use rgba_to_nv::rgba_to_yuv_nv61;

pub use yuv_to_rgba::yuv420_to_bgr;
pub use yuv_to_rgba::yuv420_to_bgra;
pub use yuv_to_rgba::yuv420_to_rgb;
pub use yuv_to_rgba::yuv420_to_rgba;
pub use yuv_to_rgba::yuv422_to_bgr;
pub use yuv_to_rgba::yuv422_to_bgra;
pub use yuv_to_rgba::yuv422_to_rgb;
pub use yuv_to_rgba::yuv422_to_rgba;
pub use yuv_to_rgba::yuv444_to_bgr;
pub use yuv_to_rgba::yuv444_to_bgra;
pub use yuv_to_rgba::yuv444_to_rgb;
pub use yuv_to_rgba::yuv444_to_rgba;

pub use rgba_to_yuv::bgr_to_yuv420;
pub use rgba_to_yuv::bgr_to_yuv422;
pub use rgba_to_yuv::bgr_to_yuv444;
pub use rgba_to_yuv::bgra_to_yuv420;
pub use rgba_to_yuv::bgra_to_yuv422;
pub use rgba_to_yuv::bgra_to_yuv444;
pub use rgba_to_yuv::rgb_to_yuv420;
pub use rgba_to_yuv::rgb_to_yuv422;
pub use rgba_to_yuv::rgb_to_yuv444;
pub use rgba_to_yuv::rgba_to_yuv420;
pub use rgba_to_yuv::rgba_to_yuv422;
pub use rgba_to_yuv::rgba_to_yuv444;

pub use rgb_to_yuv_p16::bgr_to_yuv420_u16;
pub use rgb_to_yuv_p16::bgr_to_yuv422_u16;
pub use rgb_to_yuv_p16::bgr_to_yuv444_u16;
pub use rgb_to_yuv_p16::bgra_to_yuv420_u16;
pub use rgb_to_yuv_p16::bgra_to_yuv422_u16;
pub use rgb_to_yuv_p16::bgra_to_yuv444_u16;
pub use rgb_to_yuv_p16::rgb_to_yuv420_u16;
pub use rgb_to_yuv_p16::rgb_to_yuv422_u16;
pub use rgb_to_yuv_p16::rgb_to_yuv444_u16;
pub use rgb_to_yuv_p16::rgba_to_yuv420_u16;
pub use rgb_to_yuv_p16::rgba_to_yuv422_u16;
pub use rgb_to_yuv_p16::rgba_to_yuv444_u16;

pub use yuv_to_rgba_alpha::yuv420_with_alpha_to_bgra;
pub use yuv_to_rgba_alpha::yuv420_with_alpha_to_rgba;
pub use yuv_to_rgba_alpha::yuv422_with_alpha_to_bgra;
pub use yuv_to_rgba_alpha::yuv422_with_alpha_to_rgba;
pub use yuv_to_rgba_alpha::yuv444_with_alpha_to_bgra;
pub use yuv_to_rgba_alpha::yuv444_with_alpha_to_rgba;

pub use rgb_to_y::bgr_to_yuv400;
pub use rgb_to_y::bgra_to_yuv400;
pub use rgb_to_y::rgb_to_yuv400;
pub use rgb_to_y::rgba_to_yuv400;
pub use y_to_rgb::yuv400_to_bgr;
pub use y_to_rgb::yuv400_to_bgra;
pub use y_to_rgb::yuv400_to_rgb;
pub use y_to_rgb::yuv400_to_rgba;

pub use yuv_p10_rgba::yuv420_p10_to_bgr;
pub use yuv_p10_rgba::yuv420_p10_to_bgra;
pub use yuv_p10_rgba::yuv420_p10_to_rgb;
pub use yuv_p10_rgba::yuv420_p10_to_rgba;
pub use yuv_p10_rgba::yuv422_p10_to_bgr;
pub use yuv_p10_rgba::yuv422_p10_to_bgra;
pub use yuv_p10_rgba::yuv422_p10_to_rgb;
pub use yuv_p10_rgba::yuv422_p10_to_rgba;
pub use yuv_p10_rgba::yuv444_p10_to_bgr;
pub use yuv_p10_rgba::yuv444_p10_to_bgra;
pub use yuv_p10_rgba::yuv444_p10_to_rgb;
pub use yuv_p10_rgba::yuv444_p10_to_rgba;

pub use rgb_to_ycgco::bgr_to_ycgco420;
pub use rgb_to_ycgco::bgr_to_ycgco422;
pub use rgb_to_ycgco::bgr_to_ycgco444;
pub use rgb_to_ycgco::bgra_to_ycgco420;
pub use rgb_to_ycgco::bgra_to_ycgco422;
pub use rgb_to_ycgco::bgra_to_ycgco444;
pub use rgb_to_ycgco::rgb_to_ycgco420;
pub use rgb_to_ycgco::rgb_to_ycgco422;
pub use rgb_to_ycgco::rgb_to_ycgco444;
pub use rgb_to_ycgco::rgba_to_ycgco420;
pub use rgb_to_ycgco::rgba_to_ycgco422;
pub use rgb_to_ycgco::rgba_to_ycgco444;

pub use ycgco_to_rgb::ycgco420_to_bgr;
pub use ycgco_to_rgb::ycgco420_to_bgra;
pub use ycgco_to_rgb::ycgco420_to_rgb;
pub use ycgco_to_rgb::ycgco420_to_rgba;
pub use ycgco_to_rgb::ycgco422_to_bgr;
pub use ycgco_to_rgb::ycgco422_to_bgra;
pub use ycgco_to_rgb::ycgco422_to_rgb;
pub use ycgco_to_rgb::ycgco422_to_rgba;
pub use ycgco_to_rgb::ycgco444_to_bgr;
pub use ycgco_to_rgb::ycgco444_to_bgra;
pub use ycgco_to_rgb::ycgco444_to_rgb;
pub use ycgco_to_rgb::ycgco444_to_rgba;

pub use yuv_nv_to_rgba::yuv_nv16_to_bgr;
pub use yuv_nv_to_rgba::yuv_nv16_to_bgra;
pub use yuv_nv_to_rgba::yuv_nv16_to_rgb;
pub use yuv_nv_to_rgba::yuv_nv16_to_rgba;
pub use yuv_nv_to_rgba::yuv_nv61_to_bgr;
pub use yuv_nv_to_rgba::yuv_nv61_to_bgra;
pub use yuv_nv_to_rgba::yuv_nv61_to_rgb;
pub use yuv_nv_to_rgba::yuv_nv61_to_rgba;

pub use ycgco_to_rgb_alpha::ycgco420_with_alpha_to_bgra;
pub use ycgco_to_rgb_alpha::ycgco420_with_alpha_to_rgba;
pub use ycgco_to_rgb_alpha::ycgco422_with_alpha_to_bgra;
pub use ycgco_to_rgb_alpha::ycgco422_with_alpha_to_rgba;
pub use ycgco_to_rgb_alpha::ycgco444_with_alpha_to_bgra;
pub use ycgco_to_rgb_alpha::ycgco444_with_alpha_to_rgba;

pub use rgb_to_ycgco_r::*;
pub use ycgco_r_to_rgb::*;

pub use yuv_to_yuy2::yuv420_to_uyvy422;
pub use yuv_to_yuy2::yuv420_to_vyuy422;
pub use yuv_to_yuy2::yuv420_to_yuyv422;
pub use yuv_to_yuy2::yuv420_to_yvyu422;
pub use yuv_to_yuy2::yuv422_to_uyvy422;
pub use yuv_to_yuy2::yuv422_to_vyuy422;
pub use yuv_to_yuy2::yuv422_to_yuyv422;
pub use yuv_to_yuy2::yuv422_to_yvyu422;
pub use yuv_to_yuy2::yuv444_to_uyvy422;
pub use yuv_to_yuy2::yuv444_to_vyuy422;
pub use yuv_to_yuy2::yuv444_to_yuyv422;
pub use yuv_to_yuy2::yuv444_to_yvyu422;

pub use yuy2_to_yuv::uyvy422_to_yuv420;
pub use yuy2_to_yuv::uyvy422_to_yuv422;
pub use yuy2_to_yuv::uyvy422_to_yuv444;
pub use yuy2_to_yuv::vyuy422_to_yuv420;
pub use yuy2_to_yuv::vyuy422_to_yuv422;
pub use yuy2_to_yuv::vyuy422_to_yuv444;
pub use yuy2_to_yuv::yuyv422_to_yuv420;
pub use yuy2_to_yuv::yuyv422_to_yuv422;
pub use yuy2_to_yuv::yuyv422_to_yuv444;
pub use yuy2_to_yuv::yvyu422_to_yuv420;
pub use yuy2_to_yuv::yvyu422_to_yuv422;
pub use yuy2_to_yuv::yvyu422_to_yuv444;

pub use from_identity::gbr_to_bgr;
pub use from_identity::gbr_to_bgra;
pub use from_identity::gbr_to_rgb;
pub use from_identity::gbr_to_rgba;

pub use to_identity::bgr_to_gbr;
pub use to_identity::bgra_to_gbr;
pub use to_identity::rgb_to_gbr;
pub use to_identity::rgba_to_gbr;

pub use rgb_to_nv_p16::bgr_to_yuv_nv12_p16;
pub use rgb_to_nv_p16::bgr_to_yuv_nv16_p16;
pub use rgb_to_nv_p16::bgr_to_yuv_nv21_p16;
pub use rgb_to_nv_p16::bgr_to_yuv_nv24_p16;
pub use rgb_to_nv_p16::bgr_to_yuv_nv42_p16;
pub use rgb_to_nv_p16::bgr_to_yuv_nv61_p16;
pub use rgb_to_nv_p16::bgra_to_yuv_nv12_p16;
pub use rgb_to_nv_p16::bgra_to_yuv_nv16_p16;
pub use rgb_to_nv_p16::bgra_to_yuv_nv21_p16;
pub use rgb_to_nv_p16::bgra_to_yuv_nv24_p16;
pub use rgb_to_nv_p16::bgra_to_yuv_nv42_p16;
pub use rgb_to_nv_p16::bgra_to_yuv_nv61_p16;
pub use rgb_to_nv_p16::rgb_to_yuv_nv12_p16;
pub use rgb_to_nv_p16::rgb_to_yuv_nv16_p16;
pub use rgb_to_nv_p16::rgb_to_yuv_nv21_p16;
pub use rgb_to_nv_p16::rgb_to_yuv_nv24_p16;
pub use rgb_to_nv_p16::rgb_to_yuv_nv42_p16;
pub use rgb_to_nv_p16::rgb_to_yuv_nv61_p16;
pub use rgb_to_nv_p16::rgba_to_yuv_nv12_p16;
pub use rgb_to_nv_p16::rgba_to_yuv_nv16_p16;
pub use rgb_to_nv_p16::rgba_to_yuv_nv21_p16;
pub use rgb_to_nv_p16::rgba_to_yuv_nv24_p16;
pub use rgb_to_nv_p16::rgba_to_yuv_nv42_p16;
pub use rgb_to_nv_p16::rgba_to_yuv_nv61_p16;

pub use yuy2_to_rgb::uyvy422_to_bgr;
pub use yuy2_to_rgb::uyvy422_to_bgra;
pub use yuy2_to_rgb::uyvy422_to_rgb;
pub use yuy2_to_rgb::uyvy422_to_rgba;
pub use yuy2_to_rgb::vyuy422_to_bgr;
pub use yuy2_to_rgb::vyuy422_to_bgra;
pub use yuy2_to_rgb::vyuy422_to_rgb;
pub use yuy2_to_rgb::vyuy422_to_rgba;
pub use yuy2_to_rgb::yuyv422_to_bgr;
pub use yuy2_to_rgb::yuyv422_to_bgra;
pub use yuy2_to_rgb::yuyv422_to_rgb;
pub use yuy2_to_rgb::yuyv422_to_rgba;
pub use yuy2_to_rgb::yvyu422_to_bgr;
pub use yuy2_to_rgb::yvyu422_to_bgra;
pub use yuy2_to_rgb::yvyu422_to_rgb;
pub use yuy2_to_rgb::yvyu422_to_rgba;
