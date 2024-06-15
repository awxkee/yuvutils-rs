/*
 * // Copyright (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    target_feature = "avx2"
))]
use crate::avx2::avx2_rgb_to_y_row;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[cfg(feature = "nightly_avx512")]
use crate::avx512bw::avx512_row_rgb_to_y;
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use crate::neon::neon_rgb_to_y_row;
#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    target_feature = "sse4.1"
))]
use crate::sse::sse_rgb_to_y;
use crate::yuv_support::*;

// Chroma subsampling always assumed as YUV 400
fn rgbx_to_y<const ORIGIN_CHANNELS: u8>(
    y_plane: &mut [u8],
    y_stride: u32,
    rgba: &[u8],
    rgba_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) {
    let source_channels: YuvSourceChannels = ORIGIN_CHANNELS.into();
    let channels = source_channels.get_channels_count();
    let range = get_yuv_range(8, range);
    let kr_kb = get_kr_kb(matrix);
    let max_range_p8 = (2f32.powi(8) - 1f32) as u32;
    let transform_precise = get_forward_transform(
        max_range_p8,
        range.range_y,
        range.range_uv,
        kr_kb.kr,
        kr_kb.kb,
    );
    let transform = transform_precise.to_integers(8);
    let precision_scale = (1 << 8) as f32;
    let bias_y = ((range.bias_y as f32 + 0.5f32) * precision_scale) as i32;

    #[cfg(all(
        any(target_arch = "x86", target_arch = "x86_64"),
        target_feature = "sse4.1"
    ))]
    let mut _use_sse = false;
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    let mut _use_avx = false;
    #[cfg(all(
        any(target_arch = "x86", target_arch = "x86_64"),
        target_feature = "avx512bw"
    ))]
    let mut _use_avx512 = false;

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        #[cfg(all(feature = "nightly_avx512", target_feature = "avx512bw"))]
        if std::arch::is_x86_feature_detected!("avx512bw") {
            _use_avx512 = true;
        }
        #[cfg(all(feature = "nightly_avx512", target_feature = "avx512bw"))]
        if is_x86_feature_detected!("avx2") {
            _use_avx = true;
        }
        #[cfg(target_feature = "sse4.1")]
        if is_x86_feature_detected!("sse4.1") {
            _use_sse = true;
        }
    }

    let mut y_offset = 0usize;
    let mut rgba_offset = 0usize;

    for _ in 0..height as usize {
        #[allow(unused_variables)]
        #[allow(unused_mut)]
        let mut cx = 0usize;

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        unsafe {
            #[cfg(all(feature = "nightly_avx512", target_feature = "avx512bw"))]
            if _use_avx {
                let processed_offset = avx512_row_rgb_to_y::<ORIGIN_CHANNELS>(
                    &transform,
                    &range,
                    y_plane.as_mut_ptr(),
                    &rgba,
                    y_offset,
                    rgba_offset,
                    cx,
                    width as usize,
                );
                cx = processed_offset;
            }
            #[cfg(target_feature = "avx2")]
            if _use_avx {
                let processed_offset = avx2_rgb_to_y_row::<ORIGIN_CHANNELS>(
                    &transform,
                    &range,
                    y_plane.as_mut_ptr(),
                    &rgba,
                    y_offset,
                    rgba_offset,
                    cx,
                    width as usize,
                );
                cx = processed_offset;
            }
            #[cfg(target_feature = "sse4.1")]
            if _use_sse {
                let processed_offset = sse_rgb_to_y::<ORIGIN_CHANNELS>(
                    &transform,
                    &range,
                    y_plane.as_mut_ptr(),
                    &rgba,
                    y_offset,
                    rgba_offset,
                    cx,
                    width as usize,
                );
                cx = processed_offset;
            }
        }

        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        unsafe {
            cx = neon_rgb_to_y_row::<ORIGIN_CHANNELS>(
                &transform,
                &range,
                y_plane.as_mut_ptr(),
                &rgba,
                y_offset,
                rgba_offset,
                cx,
                width as usize,
            );
        }

        for x in cx..width as usize {
            let px = x * channels;
            let dst_offset = rgba_offset + px;
            unsafe {
                let r =
                    *rgba.get_unchecked(dst_offset + source_channels.get_r_channel_offset()) as i32;
                let g =
                    *rgba.get_unchecked(dst_offset + source_channels.get_g_channel_offset()) as i32;
                let b =
                    *rgba.get_unchecked(dst_offset + source_channels.get_b_channel_offset()) as i32;
                let y = (r * transform.yr + g * transform.yg + b * transform.yb + bias_y) >> 8;
                *y_plane.get_unchecked_mut(y_offset + x) = y as u8;
            }
        }

        y_offset += y_stride as usize;
        rgba_offset += rgba_stride as usize;
    }
}

/// Convert RGB image data to YUV 400 planar format.
///
/// This function performs RGB to YUV conversion and stores the result in YUV400 planar format,
/// with Y (luminance) plane
///
/// # Arguments
///
/// * `y_plane` - A mutable slice to store the Y (luminance) plane data.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `rgb` - The input RGB image data slice.
/// * `rgb_stride` - The stride (bytes per row) for the RGB image data.
/// * `width` - The width of the image in pixels.
/// * `height` - The height of the image in pixels.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input RGB data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn rgb_to_yuv400(
    y_plane: &mut [u8],
    y_stride: u32,
    rgb: &[u8],
    rgb_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) {
    rgbx_to_y::<{ YuvSourceChannels::Rgb as u8 }>(
        y_plane, y_stride, rgb, rgb_stride, width, height, range, matrix,
    );
}

/// Convert RGBA image data to YUV 400 planar format.
///
/// This function performs RGBA to YUV conversion and stores the result in YUV400 planar format,
/// with Y (luminance) plane
///
/// # Arguments
///
/// * `y_plane` - A mutable slice to store the Y (luminance) plane data.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `rgba` - The input RGBA image data slice.
/// * `rgba_stride` - The stride (bytes per row) for the RGBA image data.
/// * `width` - The width of the image in pixels.
/// * `height` - The height of the image in pixels.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input RGBA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn rgba_to_yuv400(
    y_plane: &mut [u8],
    y_stride: u32,
    rgba: &[u8],
    rgba_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) {
    rgbx_to_y::<{ YuvSourceChannels::Rgba as u8 }>(
        y_plane,
        y_stride,
        rgba,
        rgba_stride,
        width,
        height,
        range,
        matrix,
    );
}

/// Convert BGRA image data to YUV 400 planar format.
///
/// This function performs BGRA to YUV conversion and stores the result in YUV420 planar format,
/// with Y (luminance) plane
///
/// # Arguments
///
/// * `y_plane` - A mutable slice to store the Y (luminance) plane data.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `bgra` - The input BGRA image data slice.
/// * `bgra_stride` - The stride (bytes per row) for the BGRA image data.
/// * `width` - The width of the image in pixels.
/// * `height` - The height of the image in pixels.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGRA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn bgra_to_yuv400(
    y_plane: &mut [u8],
    y_stride: u32,
    bgra: &[u8],
    bgra_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) {
    rgbx_to_y::<{ YuvSourceChannels::Bgra as u8 }>(
        y_plane,
        y_stride,
        bgra,
        bgra_stride,
        width,
        height,
        range,
        matrix,
    );
}
