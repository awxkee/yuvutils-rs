/*
 * // Copyright (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    feature = "nightly_avx512"
))]
use crate::avx512bw::avx512_y_to_rgb_row;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[allow(unused_imports)]
use crate::internals::ProcessedOffset;
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use crate::neon::neon_y_to_rgb_row;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
use crate::wasm32::wasm_y_to_rgb_row;
use crate::yuv_support::*;

// Chroma subsampling always assumed as 400
fn y_to_rgbx<const DESTINATION_CHANNELS: u8>(
    y_plane: &[u8],
    y_stride: u32,
    rgba: &mut [u8],
    rgba_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) {
    let destination_channels: YuvSourceChannels = DESTINATION_CHANNELS.into();
    let channels = destination_channels.get_channels_count();
    let range = get_yuv_range(8, range);
    let kr_kb = get_kr_kb(matrix);
    let transform = get_inverse_transform(255, range.range_y, range.range_uv, kr_kb.kr, kr_kb.kb);

    const PRECISION: i32 = 6;
    const ROUNDING_CONST: i32 = 1 << (PRECISION - 1);
    let inverse_transform = transform.to_integers(PRECISION as u32);
    let y_coef = inverse_transform.y_coef;

    let bias_y = range.bias_y as i32;

    let mut y_offset = 0usize;
    let mut rgba_offset = 0usize;

    #[cfg(all(
        any(target_arch = "x86", target_arch = "x86_64"),
        feature = "nightly_avx512"
    ))]
    let mut _use_avx512 = std::arch::is_x86_feature_detected!("avx512bw");

    for _ in 0..height as usize {
        let mut _cx = 0usize;

        #[cfg(all(
            any(target_arch = "x86", target_arch = "x86_64"),
            feature = "nightly_avx512"
        ))]
        unsafe {
            if _use_avx512 {
                let processed = avx512_y_to_rgb_row::<DESTINATION_CHANNELS>(
                    &range,
                    &inverse_transform,
                    y_plane,
                    rgba,
                    _cx,
                    y_offset,
                    rgba_offset,
                    width as usize,
                );
                _cx = processed;
            }
        }

        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        unsafe {
            let offset = neon_y_to_rgb_row::<DESTINATION_CHANNELS>(
                &range,
                &inverse_transform,
                y_plane,
                rgba,
                _cx,
                y_offset,
                rgba_offset,
                width as usize,
            );
            _cx = offset;
        }

        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        unsafe {
            let offset = wasm_y_to_rgb_row::<DESTINATION_CHANNELS>(
                &range,
                &inverse_transform,
                y_plane,
                rgba,
                _cx,
                y_offset,
                rgba_offset,
                width as usize,
            );
            _cx = offset;
        }

        for x in _cx..width as usize {
            let y_value =
                (unsafe { *y_plane.get_unchecked(y_offset + x) } as i32 - bias_y) * y_coef;

            let r = ((y_value + ROUNDING_CONST) >> PRECISION).min(255i32).max(0);

            let px = x * channels;

            let rgba_shift = rgba_offset + px;

            unsafe {
                let dst = rgba.get_unchecked_mut(rgba_shift..);
                *dst.get_unchecked_mut(destination_channels.get_r_channel_offset()) = r as u8;
                *dst.get_unchecked_mut(destination_channels.get_g_channel_offset()) = r as u8;
                *dst.get_unchecked_mut(destination_channels.get_b_channel_offset()) = r as u8;
                if destination_channels.has_alpha() {
                    *dst.get_unchecked_mut(destination_channels.get_a_channel_offset()) = 255;
                }
            }
        }

        y_offset += y_stride as usize;
        rgba_offset += rgba_stride as usize;
    }
}

/// Convert YUV 400 planar format to RGB format.
///
/// This function takes YUV 400 planar format data with 8-bit precision,
/// and converts it to RGB format with 8-bit per channel precision.
///
/// # Arguments
///
/// * `y_plane` - A slice to load the Y (luminance) plane data.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `width` - The width of the YUV image.
/// * `height` - The height of the YUV image.
/// * `rgb_data` - A mutable slice to store the converted RGB data.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input RGB data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv400_to_rgb(
    y_plane: &[u8],
    y_stride: u32,
    rgb: &mut [u8],
    rgb_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) {
    y_to_rgbx::<{ YuvSourceChannels::Rgb as u8 }>(
        y_plane, y_stride, rgb, rgb_stride, width, height, range, matrix,
    )
}

/// Convert YUV 400 planar format to BGR format.
///
/// This function takes YUV 400 planar format data with 8-bit precision,
/// and converts it to BGR format with 8-bit per channel precision.
///
/// # Arguments
///
/// * `y_plane` - A slice to load the Y (luminance) plane data.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `width` - The width of the YUV image.
/// * `height` - The height of the YUV image.
/// * `rgb_data` - A mutable slice to store the converted BGR data.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGR data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv400_to_bgr(
    y_plane: &[u8],
    y_stride: u32,
    bgr: &mut [u8],
    bgr_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) {
    y_to_rgbx::<{ YuvSourceChannels::Bgr as u8 }>(
        y_plane, y_stride, bgr, bgr_stride, width, height, range, matrix,
    )
}

/// Convert YUV 400 planar format to RGBA format.
///
/// This function takes YUV 400 planar format data with 8-bit precision,
/// and converts it to RGBA format with 8-bit per channel precision.
///
/// # Arguments
///
/// * `y_plane` - A slice to load the Y (luminance) plane data.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `width` - The width of the YUV image.
/// * `height` - The height of the YUV image.
/// * `rgba_data` - A mutable slice to store the converted RGBA data.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGRA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv400_to_rgba(
    y_plane: &[u8],
    y_stride: u32,
    rgba: &mut [u8],
    rgba_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) {
    y_to_rgbx::<{ YuvSourceChannels::Rgba as u8 }>(
        y_plane,
        y_stride,
        rgba,
        rgba_stride,
        width,
        height,
        range,
        matrix,
    )
}

/// Convert YUV 400 planar format to BGRA format.
///
/// This function takes YUV 400 planar format data with 8-bit precision,
/// and converts it to BGRA format with 8-bit per channel precision.
///
/// # Arguments
///
/// * `y_plane` - A slice to load the Y (luminance) plane data.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `width` - The width of the YUV image.
/// * `height` - The height of the YUV image.
/// * `bgra_data` - A mutable slice to store the converted BGRA data.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGRA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv400_to_bgra(
    y_plane: &[u8],
    y_stride: u32,
    bgra: &mut [u8],
    bgra_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) {
    y_to_rgbx::<{ YuvSourceChannels::Bgra as u8 }>(
        y_plane,
        y_stride,
        bgra,
        bgra_stride,
        width,
        height,
        range,
        matrix,
    )
}
