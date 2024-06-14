/*
 * // Copyright (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

#[allow(unused_imports)]
use crate::internals::ProcessedOffset;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use crate::x86_simd_support::*;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use crate::x86_ycbcr_compute::*;
use crate::yuv_support::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
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

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn sse_row<const ORIGIN_CHANNELS: u8>(
    transform: &CbCrForwardTransform<i32>,
    range: &YuvChromaRange,
    y_plane: *mut u8,
    rgba: &[u8],
    y_offset: usize,
    rgba_offset: usize,
    start_cx: usize,
    width: usize,
) -> usize {
    let source_channels: YuvSourceChannels = ORIGIN_CHANNELS.into();
    let channels = source_channels.get_channels_count();

    let y_ptr = y_plane.add(y_offset);
    let rgba_ptr = rgba.as_ptr().add(rgba_offset);

    let mut cx = start_cx;

    let bias_y = ((range.bias_y as f32 + 0.5f32) * (1i32 << 8i32) as f32) as i32;

    while cx + 16 < width {
        let y_bias = _mm_set1_epi32(bias_y);
        let v_yr = _mm_set1_epi32(transform.yr);
        let v_yg = _mm_set1_epi32(transform.yg);
        let v_yb = _mm_set1_epi32(transform.yb);

        let (r_values, g_values, b_values);

        let px = cx * channels;

        match source_channels {
            YuvSourceChannels::Rgb => {
                let row_1 = _mm_loadu_si128(rgba_ptr.add(px) as *const __m128i);
                let row_2 = _mm_loadu_si128(rgba_ptr.add(px + 16) as *const __m128i);
                let row_3 = _mm_loadu_si128(rgba_ptr.add(px + 32) as *const __m128i);

                let (it1, it2, it3) = sse_deinterleave_rgb(row_1, row_2, row_3);
                r_values = it1;
                g_values = it2;
                b_values = it3;
            }
            YuvSourceChannels::Rgba | YuvSourceChannels::Bgra => {
                let row_1 = _mm_loadu_si128(rgba_ptr.add(px) as *const __m128i);
                let row_2 = _mm_loadu_si128(rgba_ptr.add(px + 16) as *const __m128i);
                let row_3 = _mm_loadu_si128(rgba_ptr.add(px + 32) as *const __m128i);
                let row_4 = _mm_loadu_si128(rgba_ptr.add(px + 48) as *const __m128i);

                let (it1, it2, it3, _) = sse_deinterleave_rgba(row_1, row_2, row_3, row_4);
                if source_channels == YuvSourceChannels::Rgba {
                    r_values = it1;
                    g_values = it2;
                    b_values = it3;
                } else {
                    r_values = it3;
                    g_values = it2;
                    b_values = it1;
                }
            }
        }

        let r_low = _mm_cvtepu8_epi16(r_values);
        let r_high = _mm_cvtepu8_epi16(_mm_srli_si128::<8>(r_values));
        let g_low = _mm_cvtepu8_epi16(g_values);
        let g_high = _mm_cvtepu8_epi16(_mm_srli_si128::<8>(g_values));
        let b_low = _mm_cvtepu8_epi16(b_values);
        let b_high = _mm_cvtepu8_epi16(_mm_srli_si128::<8>(b_values));

        let y_l = sse_rgb_to_ycbcr(r_low, g_low, b_low, y_bias, v_yr, v_yg, v_yb);

        let y_h = sse_rgb_to_ycbcr(r_high, g_high, b_high, y_bias, v_yr, v_yg, v_yb);

        let y_yuv = _mm_packus_epi16(y_l, y_h);

        _mm_storeu_si128(y_ptr.add(cx) as *mut __m128i, y_yuv);

        cx += 16;
    }

    return cx;
}

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

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    let mut _use_sse = false;
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    let mut _use_avx = false;
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    let mut _use_avx512 = false;

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::arch::is_x86_feature_detected!("avx512bw") {
            _use_avx512 = true;
        }
        #[cfg(target_feature = "avx2")]
        if is_x86_feature_detected!("avx2") {
            _use_avx = true;
        }
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
            #[cfg(feature = "nightly_avx512")]
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
            if _use_sse {
                let processed_offset = sse_row::<ORIGIN_CHANNELS>(
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
