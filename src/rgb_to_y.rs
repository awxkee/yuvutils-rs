/*
 * // Copyright (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

#[cfg(all(target_arch = "x86_64"))]
#[cfg(feature = "nightly_avx512")]
use crate::avx512_utils::*;
#[allow(unused_imports)]
use crate::internals::ProcessedOffset;
#[cfg(target_arch = "x86_64")]
use crate::x86_simd_support::*;
#[cfg(target_arch = "x86_64")]
use crate::x86_ycbcr_compute::*;
use crate::yuv_support::*;
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use std::arch::aarch64::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(all(target_arch = "x86_64"))]
#[cfg(feature = "nightly_avx512")]
#[inline(always)]
#[allow(dead_code)]
unsafe fn avx512_row<const ORIGIN_CHANNELS: u8>(
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

    while cx + 64 < width {
        let y_bias = _mm512_set1_epi32(bias_y);
        let v_yr = _mm512_set1_epi32(transform.yr);
        let v_yg = _mm512_set1_epi32(transform.yg);
        let v_yb = _mm512_set1_epi32(transform.yb);

        let (r_values, g_values, b_values);

        let px = cx * channels;

        match source_channels {
            YuvSourceChannels::Rgb => {
                let row_1 = _mm512_loadu_si512(rgba_ptr.add(px) as *const i32);
                let row_2 = _mm512_loadu_si512(rgba_ptr.add(px + 64) as *const i32);
                let row_3 = _mm512_loadu_si512(rgba_ptr.add(px + 128) as *const i32);

                let (it1, it2, it3) = avx512_deinterleave_rgb(row_1, row_2, row_3);
                r_values = it1;
                g_values = it2;
                b_values = it3;
            }
            YuvSourceChannels::Rgba | YuvSourceChannels::Bgra => {
                let row_1 = _mm512_loadu_si512(rgba_ptr.add(px) as *const i32);
                let row_2 = _mm512_loadu_si512(rgba_ptr.add(px + 64) as *const i32);
                let row_3 = _mm512_loadu_si512(rgba_ptr.add(px + 128) as *const i32);
                let row_4 = _mm512_loadu_si512(rgba_ptr.add(px + 128 + 64) as *const i32);

                let (it1, it2, it3, _) = avx512_deinterleave_rgba(row_1, row_2, row_3, row_4);
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

        let r_low = _mm512_cvtepu8_epi16(_mm512_castsi512_si256(r_values));
        let r_high = _mm512_cvtepu8_epi16(_mm512_extracti64x4_epi64::<1>(r_values));
        let g_low = _mm512_cvtepu8_epi16(_mm512_castsi512_si256(g_values));
        let g_high = _mm512_cvtepu8_epi16(_mm512_extracti64x4_epi64::<1>(g_values));
        let b_low = _mm512_cvtepu8_epi16(_mm512_castsi512_si256(b_values));
        let b_high = _mm512_cvtepu8_epi16(_mm512_extracti64x4_epi64::<1>(b_values));

        let y_l = avx512_rgb_to_ycbcr(r_low, g_low, b_low, y_bias, v_yr, v_yg, v_yb);

        let y_h = avx512_rgb_to_ycbcr(r_high, g_high, b_high, y_bias, v_yr, v_yg, v_yb);

        let y_yuv = avx512_pack_u16(y_l, y_h);

        _mm512_storeu_si512(y_ptr.add(cx) as *mut i32, y_yuv);

        cx += 64;
    }

    return cx;
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
#[allow(dead_code)]
unsafe fn avx2_row<const ORIGIN_CHANNELS: u8>(
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

    while cx + 32 < width {
        let y_bias = _mm256_set1_epi32(bias_y);
        let v_yr = _mm256_set1_epi32(transform.yr);
        let v_yg = _mm256_set1_epi32(transform.yg);
        let v_yb = _mm256_set1_epi32(transform.yb);

        let (r_values, g_values, b_values);

        let px = cx * channels;

        match source_channels {
            YuvSourceChannels::Rgb => {
                let row_1 = _mm256_loadu_si256(rgba_ptr.add(px) as *const __m256i);
                let row_2 = _mm256_loadu_si256(rgba_ptr.add(px + 32) as *const __m256i);
                let row_3 = _mm256_loadu_si256(rgba_ptr.add(px + 64) as *const __m256i);

                let (it1, it2, it3) = avx2_deinterleave_rgb(row_1, row_2, row_3);
                r_values = it1;
                g_values = it2;
                b_values = it3;
            }
            YuvSourceChannels::Rgba | YuvSourceChannels::Bgra => {
                let row_1 = _mm256_loadu_si256(rgba_ptr.add(px) as *const __m256i);
                let row_2 = _mm256_loadu_si256(rgba_ptr.add(px + 32) as *const __m256i);
                let row_3 = _mm256_loadu_si256(rgba_ptr.add(px + 64) as *const __m256i);
                let row_4 = _mm256_loadu_si256(rgba_ptr.add(px + 96) as *const __m256i);

                let (it1, it2, it3, _) = avx2_deinterleave_rgba(row_1, row_2, row_3, row_4);
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

        let r_low = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(r_values));
        let r_high = _mm256_cvtepu8_epi16(_mm256_extracti128_si256::<1>(r_values));
        let g_low = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(g_values));
        let g_high = _mm256_cvtepu8_epi16(_mm256_extracti128_si256::<1>(g_values));
        let b_low = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(b_values));
        let b_high = _mm256_cvtepu8_epi16(_mm256_extracti128_si256::<1>(b_values));

        let y_l = avx2_rgb_to_ycbcr(r_low, g_low, b_low, y_bias, v_yr, v_yg, v_yb);

        let y_h = avx2_rgb_to_ycbcr(r_high, g_high, b_high, y_bias, v_yr, v_yg, v_yb);

        let y_yuv = avx2_pack_u16(y_l, y_h);

        _mm256_storeu_si256(y_ptr.add(cx) as *mut __m256i, y_yuv);

        cx += 32;
    }

    return cx;
}

#[cfg(target_arch = "x86_64")]
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

    #[cfg(target_arch = "x86_64")]
    let mut _use_sse = false;
    #[cfg(target_arch = "x86_64")]
    let mut _use_avx = false;
    #[cfg(target_arch = "x86_64")]
    let mut _use_avx512 = false;

    #[cfg(target_arch = "x86_64")]
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

        #[cfg(target_arch = "x86_64")]
        unsafe {
            #[cfg(feature = "nightly_avx512")]
            if _use_avx {
                let processed_offset = avx512_row::<ORIGIN_CHANNELS>(
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
                let processed_offset = avx2_row::<ORIGIN_CHANNELS>(
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

        #[cfg(target_arch = "aarch64")]
        #[cfg(target_feature = "neon")]
        unsafe {
            let y_ptr = y_plane.as_mut_ptr();
            let rgba_ptr = rgba.as_ptr();

            let y_bias = vdupq_n_s32(bias_y);
            let v_yr = vdupq_n_s16(transform.yr as i16);
            let v_yg = vdupq_n_s16(transform.yg as i16);
            let v_yb = vdupq_n_s16(transform.yb as i16);
            let v_zeros = vdupq_n_s32(0i32);

            while cx + 16 < width as usize {
                let r_values_u8: uint8x16_t;
                let g_values_u8: uint8x16_t;
                let b_values_u8: uint8x16_t;

                match source_channels {
                    YuvSourceChannels::Rgb => {
                        let rgb_values = vld3q_u8(rgba_ptr.add(rgba_offset + cx * channels));
                        r_values_u8 = rgb_values.0;
                        g_values_u8 = rgb_values.1;
                        b_values_u8 = rgb_values.2;
                    }
                    YuvSourceChannels::Rgba => {
                        let rgb_values = vld4q_u8(rgba_ptr.add(rgba_offset + cx * channels));
                        r_values_u8 = rgb_values.0;
                        g_values_u8 = rgb_values.1;
                        b_values_u8 = rgb_values.2;
                    }
                    YuvSourceChannels::Bgra => {
                        let rgb_values = vld4q_u8(rgba_ptr.add(rgba_offset + cx * channels));
                        r_values_u8 = rgb_values.2;
                        g_values_u8 = rgb_values.1;
                        b_values_u8 = rgb_values.0;
                    }
                }

                let r_high = vreinterpretq_s16_u16(vmovl_high_u8(r_values_u8));
                let g_high = vreinterpretq_s16_u16(vmovl_high_u8(g_values_u8));
                let b_high = vreinterpretq_s16_u16(vmovl_high_u8(b_values_u8));

                let r_h_low = vget_low_s16(r_high);
                let g_h_low = vget_low_s16(g_high);
                let b_h_low = vget_low_s16(b_high);

                let mut y_h_high = vmlal_high_s16(y_bias, r_high, v_yr);
                y_h_high = vmlal_high_s16(y_h_high, g_high, v_yg);
                y_h_high = vmlal_high_s16(y_h_high, b_high, v_yb);
                y_h_high = vmaxq_s32(y_h_high, v_zeros);

                let mut y_h_low = vmlal_s16(y_bias, r_h_low, vget_low_s16(v_yr));
                y_h_low = vmlal_s16(y_h_low, g_h_low, vget_low_s16(v_yg));
                y_h_low = vmlal_s16(y_h_low, b_h_low, vget_low_s16(v_yb));
                y_h_low = vmaxq_s32(y_h_low, v_zeros);

                let y_high =
                    vcombine_u16(vqshrun_n_s32::<8>(y_h_low), vqshrun_n_s32::<8>(y_h_high));

                let r_low = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(r_values_u8)));
                let g_low = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(g_values_u8)));
                let b_low = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(b_values_u8)));

                let r_l_low = vget_low_s16(r_low);
                let g_l_low = vget_low_s16(g_low);
                let b_l_low = vget_low_s16(b_low);

                let mut y_l_high = vmlal_high_s16(y_bias, r_low, v_yr);
                y_l_high = vmlal_high_s16(y_l_high, g_low, v_yg);
                y_l_high = vmlal_high_s16(y_l_high, b_low, v_yb);
                y_l_high = vmaxq_s32(y_l_high, v_zeros);

                let mut y_l_low = vmlal_s16(y_bias, r_l_low, vget_low_s16(v_yr));
                y_l_low = vmlal_s16(y_l_low, g_l_low, vget_low_s16(v_yg));
                y_l_low = vmlal_s16(y_l_low, b_l_low, vget_low_s16(v_yb));
                y_l_low = vmaxq_s32(y_l_low, v_zeros);

                let y_low = vcombine_u16(vqshrun_n_s32::<8>(y_l_low), vqshrun_n_s32::<8>(y_l_high));

                let y = vcombine_u8(vqmovn_u16(y_low), vqmovn_u16(y_high));
                vst1q_u8(y_ptr.add(y_offset + cx), y);

                cx += 16;
            }
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
