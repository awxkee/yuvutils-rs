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
use crate::avx2::avx2_rgba_to_yuv;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[cfg(feature = "nightly_avx512")]
use crate::avx512bw::avx512_rgba_to_yuv;
#[allow(unused_imports)]
use crate::internals::*;
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use crate::neon::neon_rgba_to_yuv;
#[allow(unused_imports)]
use crate::x86_simd_support::*;
#[allow(unused_imports)]
use crate::x86_ycbcr_compute::*;
#[allow(unused_imports)]
use crate::yuv_support::*;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn sse_row<const ORIGIN_CHANNELS: u8, const SAMPLING: u8>(
    transform: &CbCrForwardTransform<i32>,
    range: &YuvChromaRange,
    y_plane: *mut u8,
    u_plane: *mut u8,
    v_plane: *mut u8,
    rgba: &[u8],
    y_offset: usize,
    u_offset: usize,
    v_offset: usize,
    rgba_offset: usize,
    start_cx: usize,
    start_ux: usize,
    width: usize,
) -> ProcessedOffset {
    let chroma_subsampling: YuvChromaSample = SAMPLING.into();
    let source_channels: YuvSourceChannels = ORIGIN_CHANNELS.into();
    let channels = source_channels.get_channels_count();

    let y_ptr = y_plane.add(y_offset);
    let u_ptr = u_plane.add(u_offset);
    let v_ptr = v_plane.add(v_offset);
    let rgba_ptr = rgba.as_ptr().add(rgba_offset);

    let mut cx = start_cx;
    let mut uv_x = start_ux;

    let bias_y = ((range.bias_y as f32 + 0.5f32) * (1i32 << 8i32) as f32) as i32;
    let bias_uv = ((range.bias_uv as f32 + 0.5f32) * (1i32 << 8i32) as f32) as i32;

    while cx + 16 < width {
        let y_bias = _mm_set1_epi32(bias_y);
        let uv_bias = _mm_set1_epi32(bias_uv);
        let v_yr = _mm_set1_epi32(transform.yr);
        let v_yg = _mm_set1_epi32(transform.yg);
        let v_yb = _mm_set1_epi32(transform.yb);
        let v_cb_r = _mm_set1_epi32(transform.cb_r);
        let v_cb_g = _mm_set1_epi32(transform.cb_g);
        let v_cb_b = _mm_set1_epi32(transform.cb_b);
        let v_cr_r = _mm_set1_epi32(transform.cr_r);
        let v_cr_g = _mm_set1_epi32(transform.cr_g);
        let v_cr_b = _mm_set1_epi32(transform.cr_b);

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
        let cb_l = sse_rgb_to_ycbcr(r_low, g_low, b_low, uv_bias, v_cb_r, v_cb_g, v_cb_b);
        let cr_l = sse_rgb_to_ycbcr(r_low, g_low, b_low, uv_bias, v_cr_r, v_cr_g, v_cr_b);

        let y_h = sse_rgb_to_ycbcr(r_high, g_high, b_high, y_bias, v_yr, v_yg, v_yb);

        let y_yuv = _mm_packus_epi16(y_l, y_h);

        let cb_h = sse_rgb_to_ycbcr(r_high, g_high, b_high, uv_bias, v_cb_r, v_cb_g, v_cb_b);
        let cr_h = sse_rgb_to_ycbcr(r_high, g_high, b_high, uv_bias, v_cr_r, v_cr_g, v_cr_b);

        let cb = _mm_packus_epi16(cb_l, cb_h);

        let cr = _mm_packus_epi16(cr_l, cr_h);

        _mm_storeu_si128(y_ptr.add(cx) as *mut __m128i, y_yuv);

        match chroma_subsampling {
            YuvChromaSample::YUV420 | YuvChromaSample::YUV422 => {
                let cb_h = sse_pairwise_widen_avg(cb);
                let cr_h = sse_pairwise_widen_avg(cr);
                std::ptr::copy_nonoverlapping(&cb_h as *const _ as *const u8, u_ptr.add(uv_x), 8);
                std::ptr::copy_nonoverlapping(&cr_h as *const _ as *const u8, v_ptr.add(uv_x), 8);
                uv_x += 8;
            }
            YuvChromaSample::YUV444 => {
                _mm_storeu_si128(u_ptr.add(uv_x) as *mut __m128i, cb);
                _mm_storeu_si128(v_ptr.add(uv_x) as *mut __m128i, cr);
                uv_x += 16;
            }
        }

        cx += 16;
    }

    return ProcessedOffset { cx, ux: uv_x };
}

fn rgbx_to_yuv8<const ORIGIN_CHANNELS: u8, const SAMPLING: u8>(
    y_plane: &mut [u8],
    y_stride: u32,
    u_plane: &mut [u8],
    u_stride: u32,
    v_plane: &mut [u8],
    v_stride: u32,
    rgba: &[u8],
    rgba_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) {
    let chroma_subsampling: YuvChromaSample = SAMPLING.into();
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
    let bias_uv = ((range.bias_uv as f32 + 0.5f32) * precision_scale) as i32;

    let iterator_step = match chroma_subsampling {
        YuvChromaSample::YUV420 => 2usize,
        YuvChromaSample::YUV422 => 2usize,
        YuvChromaSample::YUV444 => 1usize,
    };

    let mut y_offset = 0usize;
    let mut u_offset = 0usize;
    let mut v_offset = 0usize;
    let mut rgba_offset = 0usize;

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    let mut _use_sse = false;
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    let mut _use_avx = false;
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    let mut _use_avx512 = false;

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        #[cfg(feature = "nightly_avx512")]
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

    for y in 0..height as usize {
        #[allow(unused_variables)]
        #[allow(unused_mut)]
        let mut cx = 0usize;
        #[allow(unused_variables)]
        #[allow(unused_mut)]
        let mut ux = 0usize;

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        unsafe {
            #[cfg(feature = "nightly_avx512")]
            {
                if _use_avx512 {
                    let processed_offset = avx512_rgba_to_yuv::<ORIGIN_CHANNELS, SAMPLING>(
                        &transform,
                        &range,
                        y_plane.as_mut_ptr(),
                        u_plane.as_mut_ptr(),
                        v_plane.as_mut_ptr(),
                        &rgba,
                        y_offset,
                        u_offset,
                        v_offset,
                        rgba_offset,
                        cx,
                        ux,
                        width as usize,
                    );
                    cx = processed_offset.cx;
                    ux = processed_offset.ux;
                }
            }

            #[cfg(target_feature = "avx2")]
            if _use_avx {
                let processed_offset = avx2_rgba_to_yuv::<ORIGIN_CHANNELS, SAMPLING>(
                    &transform,
                    &range,
                    y_plane.as_mut_ptr(),
                    u_plane.as_mut_ptr(),
                    v_plane.as_mut_ptr(),
                    &rgba,
                    y_offset,
                    u_offset,
                    v_offset,
                    rgba_offset,
                    cx,
                    ux,
                    width as usize,
                );
                cx = processed_offset.cx;
                ux = processed_offset.ux;
            }

            if _use_sse {
                let processed_offset = sse_row::<ORIGIN_CHANNELS, SAMPLING>(
                    &transform,
                    &range,
                    y_plane.as_mut_ptr(),
                    u_plane.as_mut_ptr(),
                    v_plane.as_mut_ptr(),
                    &rgba,
                    y_offset,
                    u_offset,
                    v_offset,
                    rgba_offset,
                    cx,
                    ux,
                    width as usize,
                );
                cx = processed_offset.cx;
                ux = processed_offset.ux;
            }
        }

        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        unsafe {
            let offset = neon_rgba_to_yuv::<ORIGIN_CHANNELS, SAMPLING>(
                &transform,
                &range,
                y_plane.as_mut_ptr(),
                u_plane.as_mut_ptr(),
                v_plane.as_mut_ptr(),
                &rgba,
                y_offset,
                u_offset,
                v_offset,
                rgba_offset,
                cx,
                ux,
                width as usize,
            );
            cx = offset.cx;
            ux = offset.ux;
        }

        for x in (cx..width as usize).step_by(iterator_step) {
            let px = x * channels;
            let rgba_shift = rgba_offset + px;
            let r =
                unsafe { *rgba.get_unchecked(rgba_shift + source_channels.get_r_channel_offset()) }
                    as i32;
            let g =
                unsafe { *rgba.get_unchecked(rgba_shift + source_channels.get_g_channel_offset()) }
                    as i32;
            let b =
                unsafe { *rgba.get_unchecked(rgba_shift + source_channels.get_b_channel_offset()) }
                    as i32;
            let y_0 = (r * transform.yr + g * transform.yg + b * transform.yb + bias_y) >> 8;
            let cb = (r * transform.cb_r + g * transform.cb_g + b * transform.cb_b + bias_uv) >> 8;
            let cr = (r * transform.cr_r + g * transform.cr_g + b * transform.cr_b + bias_uv) >> 8;
            unsafe {
                *y_plane.get_unchecked_mut(y_offset + x) = y_0 as u8;
            }
            let u_pos = match chroma_subsampling {
                YuvChromaSample::YUV420 | YuvChromaSample::YUV422 => u_offset + ux,
                YuvChromaSample::YUV444 => u_offset + ux,
            };
            unsafe {
                *u_plane.get_unchecked_mut(u_pos) = cb as u8;
            }
            let v_pos = match chroma_subsampling {
                YuvChromaSample::YUV420 | YuvChromaSample::YUV422 => v_offset + ux,
                YuvChromaSample::YUV444 => v_offset + ux,
            };
            unsafe {
                *v_plane.get_unchecked_mut(v_pos) = cr as u8;
            }
            match chroma_subsampling {
                YuvChromaSample::YUV420 | YuvChromaSample::YUV422 => {
                    if x + 1 < width as usize {
                        let next_px = (x + 1) * channels;
                        let rgba_shift = rgba_offset + next_px;
                        let r = unsafe {
                            *rgba.get_unchecked(rgba_shift + source_channels.get_r_channel_offset())
                        } as i32;
                        let g = unsafe {
                            *rgba.get_unchecked(rgba_shift + source_channels.get_g_channel_offset())
                        } as i32;
                        let b = unsafe {
                            *rgba.get_unchecked(rgba_shift + source_channels.get_b_channel_offset())
                        } as i32;
                        let y_1 =
                            (r * transform.yr + g * transform.yg + b * transform.yb + bias_y) >> 8;
                        unsafe {
                            *y_plane.get_unchecked_mut(y_offset + x + 1) = y_1 as u8;
                        }
                    }
                }
                _ => {}
            }

            ux += 1;
        }

        y_offset += y_stride as usize;
        rgba_offset += rgba_stride as usize;
        match chroma_subsampling {
            YuvChromaSample::YUV420 => {
                if y & 1 == 1 {
                    u_offset += u_stride as usize;
                    v_offset += v_stride as usize;
                }
            }
            YuvChromaSample::YUV444 | YuvChromaSample::YUV422 => {
                u_offset += u_stride as usize;
                v_offset += v_stride as usize;
            }
        }
    }
}

/// Convert RGB image data to YUV 422 planar format.
///
/// This function performs RGB to YUV conversion and stores the result in YUV422 planar format,
/// with separate planes for Y (luminance), U (chrominance), and V (chrominance) components.
///
/// # Arguments
///
/// * `y_plane` - A mutable slice to store the Y (luminance) plane data.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `u_plane` - A mutable slice to store the U (chrominance) plane data.
/// * `u_stride` - The stride (bytes per row) for the U plane.
/// * `v_plane` - A mutable slice to store the V (chrominance) plane data.
/// * `v_stride` - The stride (bytes per row) for the V plane.
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
pub fn rgb_to_yuv422(
    y_plane: &mut [u8],
    y_stride: u32,
    u_plane: &mut [u8],
    u_stride: u32,
    v_plane: &mut [u8],
    v_stride: u32,
    rgb: &[u8],
    rgb_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) {
    rgbx_to_yuv8::<{ YuvSourceChannels::Rgb as u8 }, { YuvChromaSample::YUV422 as u8 }>(
        y_plane, y_stride, u_plane, u_stride, v_plane, v_stride, rgb, rgb_stride, width, height,
        range, matrix,
    );
}

/// Convert RGBA image data to YUV 422 planar format.
///
/// This function performs RGBA to YUV conversion and stores the result in YUV422 planar format,
/// with separate planes for Y (luminance), U (chrominance), and V (chrominance) components.
///
/// # Arguments
///
/// * `y_plane` - A mutable slice to store the Y (luminance) plane data.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `u_plane` - A mutable slice to store the U (chrominance) plane data.
/// * `u_stride` - The stride (bytes per row) for the U plane.
/// * `v_plane` - A mutable slice to store the V (chrominance) plane data.
/// * `v_stride` - The stride (bytes per row) for the V plane.
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
pub fn rgba_to_yuv422(
    y_plane: &mut [u8],
    y_stride: u32,
    u_plane: &mut [u8],
    u_stride: u32,
    v_plane: &mut [u8],
    v_stride: u32,
    rgba: &[u8],
    rgba_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) {
    rgbx_to_yuv8::<{ YuvSourceChannels::Rgba as u8 }, { YuvChromaSample::YUV422 as u8 }>(
        y_plane,
        y_stride,
        u_plane,
        u_stride,
        v_plane,
        v_stride,
        rgba,
        rgba_stride,
        width,
        height,
        range,
        matrix,
    );
}

/// Convert BGRA image data to YUV 422 planar format.
///
/// This function performs BGRA to YUV conversion and stores the result in YUV422 planar format,
/// with separate planes for Y (luminance), U (chrominance), and V (chrominance) components.
///
/// # Arguments
///
/// * `y_plane` - A mutable slice to store the Y (luminance) plane data.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `u_plane` - A mutable slice to store the U (chrominance) plane data.
/// * `u_stride` - The stride (bytes per row) for the U plane.
/// * `v_plane` - A mutable slice to store the V (chrominance) plane data.
/// * `v_stride` - The stride (bytes per row) for the V plane.
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
pub fn bgra_to_yuv422(
    y_plane: &mut [u8],
    y_stride: u32,
    u_plane: &mut [u8],
    u_stride: u32,
    v_plane: &mut [u8],
    v_stride: u32,
    bgra: &[u8],
    bgra_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) {
    rgbx_to_yuv8::<{ YuvSourceChannels::Bgra as u8 }, { YuvChromaSample::YUV422 as u8 }>(
        y_plane,
        y_stride,
        u_plane,
        u_stride,
        v_plane,
        v_stride,
        bgra,
        bgra_stride,
        width,
        height,
        range,
        matrix,
    );
}

/// Convert RGB image data to YUV 420 planar format.
///
/// This function performs RGB to YUV conversion and stores the result in YUV420 planar format,
/// with separate planes for Y (luminance), U (chrominance), and V (chrominance) components.
///
/// # Arguments
///
/// * `y_plane` - A mutable slice to store the Y (luminance) plane data.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `u_plane` - A mutable slice to store the U (chrominance) plane data.
/// * `u_stride` - The stride (bytes per row) for the U plane.
/// * `v_plane` - A mutable slice to store the V (chrominance) plane data.
/// * `v_stride` - The stride (bytes per row) for the V plane.
/// * `rgb` - The input RGB image data slice.
/// * `rgb_stride` - The stride (bytes per row) for the RGB image data.
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
pub fn rgb_to_yuv420(
    y_plane: &mut [u8],
    y_stride: u32,
    u_plane: &mut [u8],
    u_stride: u32,
    v_plane: &mut [u8],
    v_stride: u32,
    rgb: &[u8],
    rgb_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) {
    rgbx_to_yuv8::<{ YuvSourceChannels::Rgb as u8 }, { YuvChromaSample::YUV420 as u8 }>(
        y_plane, y_stride, u_plane, u_stride, v_plane, v_stride, rgb, rgb_stride, width, height,
        range, matrix,
    );
}

/// Convert RGBA image data to YUV 420 planar format.
///
/// This function performs RGBA to YUV conversion and stores the result in YUV420 planar format,
/// with separate planes for Y (luminance), U (chrominance), and V (chrominance) components.
///
/// # Arguments
///
/// * `y_plane` - A mutable slice to store the Y (luminance) plane data.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `u_plane` - A mutable slice to store the U (chrominance) plane data.
/// * `u_stride` - The stride (bytes per row) for the U plane.
/// * `v_plane` - A mutable slice to store the V (chrominance) plane data.
/// * `v_stride` - The stride (bytes per row) for the V plane.
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
pub fn rgba_to_yuv420(
    y_plane: &mut [u8],
    y_stride: u32,
    u_plane: &mut [u8],
    u_stride: u32,
    v_plane: &mut [u8],
    v_stride: u32,
    rgba: &[u8],
    rgba_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) {
    rgbx_to_yuv8::<{ YuvSourceChannels::Rgba as u8 }, { YuvChromaSample::YUV420 as u8 }>(
        y_plane,
        y_stride,
        u_plane,
        u_stride,
        v_plane,
        v_stride,
        rgba,
        rgba_stride,
        width,
        height,
        range,
        matrix,
    );
}

/// Convert BGRA image data to YUV 420 planar format.
///
/// This function performs BGRA to YUV conversion and stores the result in YUV420 planar format,
/// with separate planes for Y (luminance), U (chrominance), and V (chrominance) components.
///
/// # Arguments
///
/// * `y_plane` - A mutable slice to store the Y (luminance) plane data.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `u_plane` - A mutable slice to store the U (chrominance) plane data.
/// * `u_stride` - The stride (bytes per row) for the U plane.
/// * `v_plane` - A mutable slice to store the V (chrominance) plane data.
/// * `v_stride` - The stride (bytes per row) for the V plane.
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
pub fn bgra_to_yuv420(
    y_plane: &mut [u8],
    y_stride: u32,
    u_plane: &mut [u8],
    u_stride: u32,
    v_plane: &mut [u8],
    v_stride: u32,
    bgra: &[u8],
    bgra_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) {
    rgbx_to_yuv8::<{ YuvSourceChannels::Bgra as u8 }, { YuvChromaSample::YUV420 as u8 }>(
        y_plane,
        y_stride,
        u_plane,
        u_stride,
        v_plane,
        v_stride,
        bgra,
        bgra_stride,
        width,
        height,
        range,
        matrix,
    );
}

/// Convert RGB image data to YUV 444 planar format.
///
/// This function performs RGB to YUV conversion and stores the result in YUV444 planar format,
/// with separate planes for Y (luminance), U (chrominance), and V (chrominance) components.
///
/// # Arguments
///
/// * `y_plane` - A mutable slice to store the Y (luminance) plane data.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `u_plane` - A mutable slice to store the U (chrominance) plane data.
/// * `u_stride` - The stride (bytes per row) for the U plane.
/// * `v_plane` - A mutable slice to store the V (chrominance) plane data.
/// * `v_stride` - The stride (bytes per row) for the V plane.
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
pub fn rgb_to_yuv444(
    y_plane: &mut [u8],
    y_stride: u32,
    u_plane: &mut [u8],
    u_stride: u32,
    v_plane: &mut [u8],
    v_stride: u32,
    rgb: &[u8],
    rgb_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) {
    rgbx_to_yuv8::<{ YuvSourceChannels::Rgb as u8 }, { YuvChromaSample::YUV444 as u8 }>(
        y_plane, y_stride, u_plane, u_stride, v_plane, v_stride, rgb, rgb_stride, width, height,
        range, matrix,
    );
}

/// Convert RGBA image data to YUV 444 planar format.
///
/// This function performs RGBA to YUV conversion and stores the result in YUV444 planar format,
/// with separate planes for Y (luminance), U (chrominance), and V (chrominance) components.
///
/// # Arguments
///
/// * `y_plane` - A mutable slice to store the Y (luminance) plane data.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `u_plane` - A mutable slice to store the U (chrominance) plane data.
/// * `u_stride` - The stride (bytes per row) for the U plane.
/// * `v_plane` - A mutable slice to store the V (chrominance) plane data.
/// * `v_stride` - The stride (bytes per row) for the V plane.
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
pub fn rgba_to_yuv444(
    y_plane: &mut [u8],
    y_stride: u32,
    u_plane: &mut [u8],
    u_stride: u32,
    v_plane: &mut [u8],
    v_stride: u32,
    rgba: &[u8],
    rgba_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) {
    rgbx_to_yuv8::<{ YuvSourceChannels::Rgba as u8 }, { YuvChromaSample::YUV444 as u8 }>(
        y_plane,
        y_stride,
        u_plane,
        u_stride,
        v_plane,
        v_stride,
        rgba,
        rgba_stride,
        width,
        height,
        range,
        matrix,
    );
}

/// Convert BGRA image data to YUV 444 planar format.
///
/// This function performs BGRA to YUV conversion and stores the result in YUV444 planar format,
/// with separate planes for Y (luminance), U (chrominance), and V (chrominance) components.
///
/// # Arguments
///
/// * `y_plane` - A mutable slice to store the Y (luminance) plane data.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `u_plane` - A mutable slice to store the U (chrominance) plane data.
/// * `u_stride` - The stride (bytes per row) for the U plane.
/// * `v_plane` - A mutable slice to store the V (chrominance) plane data.
/// * `v_stride` - The stride (bytes per row) for the V plane.
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
pub fn bgra_to_yuv444(
    y_plane: &mut [u8],
    y_stride: u32,
    u_plane: &mut [u8],
    u_stride: u32,
    v_plane: &mut [u8],
    v_stride: u32,
    bgra: &[u8],
    bgra_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) {
    rgbx_to_yuv8::<{ YuvSourceChannels::Bgra as u8 }, { YuvChromaSample::YUV444 as u8 }>(
        y_plane,
        y_stride,
        u_plane,
        u_stride,
        v_plane,
        v_stride,
        bgra,
        bgra_stride,
        width,
        height,
        range,
        matrix,
    );
}
