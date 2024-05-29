/*
 * // Copyright (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */
#![cfg_attr(feature = "nightly_avx512", feature(cfg_version))]
#![cfg_attr(feature = "nightly_avx512", feature(avx512_target_feature))]
#![cfg_attr(feature = "nightly_avx512", feature(stdarch_x86_avx512))]

mod x86_simd_support;
mod x86_ycbcr_compute;
mod internals;
mod neon_simd_support;
mod rgb_to_y;
mod rgba_to_nv;
mod rgba_to_yuv;
mod y_to_rgb;
mod yuv_nv_p10_to_rgba;
mod yuv_nv_to_rgba;
mod yuv_p10_rgba;
mod yuv_support;
mod yuv_to_rgba;
mod yuv_to_rgba_alpha;
mod avx512_utils;
mod avx512_setr;
mod rgb_to_ycgco;
mod ycgco_to_rgb;
mod avx512_rgb_to_yuv;

pub use yuv_support::YuvRange;
pub use yuv_support::YuvStandardMatrix;

pub use yuv_nv_p10_to_rgba::yuv_nv12_p10_be_to_bgra;
pub use yuv_nv_p10_to_rgba::yuv_nv12_p10_msb_to_bgra;
pub use yuv_nv_p10_to_rgba::yuv_nv12_p10_msb_to_rgba;
pub use yuv_nv_p10_to_rgba::yuv_nv12_p10_to_bgra;
pub use yuv_nv_p10_to_rgba::yuv_nv16_p10_be_to_bgra;
pub use yuv_nv_p10_to_rgba::yuv_nv16_p10_msb_to_bgra;
pub use yuv_nv_p10_to_rgba::yuv_nv16_p10_msb_to_rgba;
pub use yuv_nv_p10_to_rgba::yuv_nv16_p10_to_bgra;

pub use yuv_nv_to_rgba::yuv_nv12_to_bgra;
pub use yuv_nv_to_rgba::yuv_nv12_to_rgb;
pub use yuv_nv_to_rgba::yuv_nv12_to_rgba;
pub use yuv_nv_to_rgba::yuv_nv21_to_bgra;
pub use yuv_nv_to_rgba::yuv_nv21_to_rgb;
pub use yuv_nv_to_rgba::yuv_nv21_to_rgba;
pub use yuv_nv_to_rgba::yuv_nv24_to_bgra;
pub use yuv_nv_to_rgba::yuv_nv24_to_rgb;
pub use yuv_nv_to_rgba::yuv_nv24_to_rgba;
pub use yuv_nv_to_rgba::yuv_nv42_to_bgra;
pub use yuv_nv_to_rgba::yuv_nv42_to_rgb;
pub use yuv_nv_to_rgba::yuv_nv42_to_rgba;

pub use rgba_to_nv::bgra_to_yuv_nv12;
pub use rgba_to_nv::bgra_to_yuv_nv16;
pub use rgba_to_nv::bgra_to_yuv_nv24;
pub use rgba_to_nv::rgb_to_yuv_nv12;
pub use rgba_to_nv::rgb_to_yuv_nv16;
pub use rgba_to_nv::rgb_to_yuv_nv24;
pub use rgba_to_nv::rgba_to_yuv_nv12;
pub use rgba_to_nv::rgba_to_yuv_nv16;
pub use rgba_to_nv::rgba_to_yuv_nv24;

pub use yuv_to_rgba::yuv420_to_bgra;
pub use yuv_to_rgba::yuv420_to_rgb;
pub use yuv_to_rgba::yuv420_to_rgba;
pub use yuv_to_rgba::yuv422_to_bgra;
pub use yuv_to_rgba::yuv422_to_rgb;
pub use yuv_to_rgba::yuv422_to_rgba;
pub use yuv_to_rgba::yuv444_to_bgra;
pub use yuv_to_rgba::yuv444_to_rgb;
pub use yuv_to_rgba::yuv444_to_rgba;

pub use rgba_to_yuv::bgra_to_yuv420;
pub use rgba_to_yuv::bgra_to_yuv422;
pub use rgba_to_yuv::bgra_to_yuv444;
pub use rgba_to_yuv::rgb_to_yuv420;
pub use rgba_to_yuv::rgb_to_yuv422;
pub use rgba_to_yuv::rgb_to_yuv444;
pub use rgba_to_yuv::rgba_to_yuv420;
pub use rgba_to_yuv::rgba_to_yuv422;
pub use rgba_to_yuv::rgba_to_yuv444;

pub use yuv_to_rgba_alpha::yuv420_with_alpha_to_bgra;
pub use yuv_to_rgba_alpha::yuv420_with_alpha_to_rgba;
pub use yuv_to_rgba_alpha::yuv422_with_alpha_to_bgra;
pub use yuv_to_rgba_alpha::yuv422_with_alpha_to_rgba;
pub use yuv_to_rgba_alpha::yuv444_with_alpha_to_bgra;
pub use yuv_to_rgba_alpha::yuv444_with_alpha_to_rgba;

pub use rgb_to_y::bgra_to_yuv400;
pub use rgb_to_y::rgb_to_yuv400;
pub use rgb_to_y::rgba_to_yuv400;
pub use y_to_rgb::yuv400_to_bgra;
pub use y_to_rgb::yuv400_to_rgb;
pub use y_to_rgb::yuv400_to_rgba;

pub use yuv_p10_rgba::yuv420_p10_be_to_bgra;
pub use yuv_p10_rgba::yuv420_p10_be_to_rgba;
pub use yuv_p10_rgba::yuv420_p10_to_bgra;
pub use yuv_p10_rgba::yuv420_p10_to_rgba;
pub use yuv_p10_rgba::yuv422_p10_be_to_bgra;
pub use yuv_p10_rgba::yuv422_p10_be_to_rgba;
pub use yuv_p10_rgba::yuv422_p10_to_bgra;
pub use yuv_p10_rgba::yuv422_p10_to_rgba;
pub use yuv_p10_rgba::yuv444_p10_be_to_bgra;
pub use yuv_p10_rgba::yuv444_p10_be_to_rgba;
pub use yuv_p10_rgba::yuv444_p10_to_bgra;
pub use yuv_p10_rgba::yuv444_p10_to_rgba;

pub use rgb_to_ycgco::bgra_to_ycgco420;
pub use rgb_to_ycgco::bgra_to_ycgco422;
pub use rgb_to_ycgco::bgra_to_ycgco444;
pub use rgb_to_ycgco::rgb_to_ycgco420;
pub use rgb_to_ycgco::rgb_to_ycgco422;
pub use rgb_to_ycgco::rgb_to_ycgco444;
pub use rgb_to_ycgco::rgba_to_ycgco420;
pub use rgb_to_ycgco::rgba_to_ycgco422;
pub use rgb_to_ycgco::rgba_to_ycgco444;

pub use ycgco_to_rgb::ycgco420_to_bgra;
pub use ycgco_to_rgb::ycgco420_to_rgb;
pub use ycgco_to_rgb::ycgco420_to_rgba;
pub use ycgco_to_rgb::ycgco422_to_bgra;
pub use ycgco_to_rgb::ycgco422_to_rgb;
pub use ycgco_to_rgb::ycgco422_to_rgba;
pub use ycgco_to_rgb::ycgco444_to_bgra;
pub use ycgco_to_rgb::ycgco444_to_rgb;
pub use ycgco_to_rgb::ycgco444_to_rgba;