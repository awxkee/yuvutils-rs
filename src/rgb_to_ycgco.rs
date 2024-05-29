/*
 * // Copyright (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

#[allow(unused_imports)]
use crate::internals::ProcessedOffset;
#[cfg(target_arch = "x86_64")]
#[allow(unused_imports)]
use crate::x86_simd_support::*;
#[cfg(target_arch = "x86_64")]
#[allow(unused_imports)]
use crate::x86_ycbcr_compute::*;
#[allow(unused_imports)]
use crate::yuv_support::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
#[inline(always)]
#[allow(dead_code)]
unsafe fn avx2_row<const ORIGIN_CHANNELS: u8, const SAMPLING: u8>(
    range: &YuvChromaRange,
    y_plane: *mut u8,
    cg_plane: *mut u8,
    co_plane: *mut u8,
    rgba: &[u8],
    y_offset: usize,
    cg_offset: usize,
    co_offset: usize,
    rgba_offset: usize,
    start_cx: usize,
    start_ux: usize,
    width: usize,
) -> ProcessedOffset {
    let chroma_subsampling: YuvChromaSample = SAMPLING.into();
    let source_channels: YuvSourceChannels = ORIGIN_CHANNELS.into();
    let channels = source_channels.get_channels_count();

    let y_ptr = y_plane.add(y_offset);
    let cg_ptr = cg_plane.add(cg_offset);
    let co_ptr = co_plane.add(co_offset);
    let rgba_ptr = rgba.as_ptr().add(rgba_offset);

    let mut cx = start_cx;
    let mut uv_x = start_ux;

    let bias_y = ((range.bias_y as f32 + 0.5f32) * (1i32 << 8i32) as f32) as i32;
    let bias_uv = ((range.bias_uv as f32 + 0.5f32) * (1i32 << 8i32) as f32) as i32;

    let precision_scale = (1 << 8) as f32;
    let max_colors = 2i32.pow(8) - 1i32;

    let range_reduction_y =
        (range.range_y as f32 / max_colors as f32 * precision_scale).round() as i32;
    let range_reduction_uv =
        (range.range_uv as f32 / max_colors as f32 * precision_scale).round() as i32;

    while cx + 32 < width {
        let y_bias = _mm256_set1_epi32(bias_y);
        let uv_bias = _mm256_set1_epi32(bias_uv);
        let v_range_reduction_y = _mm256_set1_epi32(range_reduction_y);
        let v_range_reduction_uv = _mm256_set1_epi32(range_reduction_uv);

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

        let (y_l, cg_l, co_l) = avx2_rgb_to_ycgco(
            r_low,
            g_low,
            b_low,
            v_range_reduction_y,
            v_range_reduction_uv,
            y_bias,
            uv_bias,
        );
        let (y_h, cg_h, co_h) = avx2_rgb_to_ycgco(
            r_high,
            g_high,
            b_high,
            v_range_reduction_y,
            v_range_reduction_uv,
            y_bias,
            uv_bias,
        );

        let y_intensity = avx2_pack_u16(y_l, y_h);
        let cg = avx2_pack_u16(cg_l, cg_h);
        let co = avx2_pack_u16(co_l, co_h);

        _mm256_storeu_si256(y_ptr.add(cx) as *mut __m256i, y_intensity);

        match chroma_subsampling {
            YuvChromaSample::YUV420 | YuvChromaSample::YUV422 => {
                let cb_h = _mm256_castsi256_si128(avx2_pairwise_widen_avg(cg));
                let cr_h = _mm256_castsi256_si128(avx2_pairwise_widen_avg(co));
                _mm_storeu_si128(cg_ptr.add(uv_x) as *mut _ as *mut __m128i, cb_h);
                _mm_storeu_si128(co_ptr.add(uv_x) as *mut _ as *mut __m128i, cr_h);
                uv_x += 16;
            }
            YuvChromaSample::YUV444 => {
                _mm256_storeu_si256(cg_ptr.add(uv_x) as *mut __m256i, cg);
                _mm256_storeu_si256(co_ptr.add(uv_x) as *mut __m256i, co);
                uv_x += 32;
            }
        }

        cx += 32;
    }

    return ProcessedOffset { cx, ux: uv_x };
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn sse_row<const ORIGIN_CHANNELS: u8, const SAMPLING: u8>(
    range: &YuvChromaRange,
    y_plane: *mut u8,
    cg_plane: *mut u8,
    co_plane: *mut u8,
    rgba: &[u8],
    y_offset: usize,
    cg_offset: usize,
    co_offset: usize,
    rgba_offset: usize,
    start_cx: usize,
    start_ux: usize,
    width: usize,
) -> ProcessedOffset {
    let chroma_subsampling: YuvChromaSample = SAMPLING.into();
    let source_channels: YuvSourceChannels = ORIGIN_CHANNELS.into();
    let channels = source_channels.get_channels_count();

    let y_ptr = y_plane.add(y_offset);
    let cg_ptr = cg_plane.add(cg_offset);
    let co_ptr = co_plane.add(co_offset);
    let rgba_ptr = rgba.as_ptr().add(rgba_offset);

    let mut cx = start_cx;
    let mut uv_x = start_ux;

    let bias_y = ((range.bias_y as f32 + 0.5f32) * (1i32 << 8i32) as f32) as i32;
    let bias_uv = ((range.bias_uv as f32 + 0.5f32) * (1i32 << 8i32) as f32) as i32;

    let precision_scale = (1 << 8) as f32;
    let max_colors = 2i32.pow(8) - 1i32;

    let range_reduction_y =
        (range.range_y as f32 / max_colors as f32 * precision_scale).round() as i32;
    let range_reduction_uv =
        (range.range_uv as f32 / max_colors as f32 * precision_scale).round() as i32;

    while cx + 16 < width {
        let y_bias = _mm_set1_epi32(bias_y);
        let uv_bias = _mm_set1_epi32(bias_uv);
        let v_range_reduction_y = _mm_set1_epi32(range_reduction_y);
        let v_range_reduction_uv = _mm_set1_epi32(range_reduction_uv);

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

        let (y_l, cg_l, co_l) = sse_rgb_to_ycgco(
            r_low,
            g_low,
            b_low,
            v_range_reduction_y,
            v_range_reduction_uv,
            y_bias,
            uv_bias,
        );
        let (y_h, cg_h, co_h) = sse_rgb_to_ycgco(
            r_high,
            g_high,
            b_high,
            v_range_reduction_y,
            v_range_reduction_uv,
            y_bias,
            uv_bias,
        );

        let y_intensity = _mm_packus_epi16(y_l, y_h);
        let cg = _mm_packus_epi16(cg_l, cg_h);
        let co = _mm_packus_epi16(co_l, co_h);

        _mm_storeu_si128(y_ptr.add(cx) as *mut __m128i, y_intensity);

        match chroma_subsampling {
            YuvChromaSample::YUV420 | YuvChromaSample::YUV422 => {
                let cb_h = sse_pairwise_widen_avg(cg);
                let cr_h = sse_pairwise_widen_avg(co);
                std::ptr::copy_nonoverlapping(&cb_h as *const _ as *const u8, cg_ptr.add(uv_x), 8);
                std::ptr::copy_nonoverlapping(&cr_h as *const _ as *const u8, co_ptr.add(uv_x), 8);
                uv_x += 8;
            }
            YuvChromaSample::YUV444 => {
                _mm_storeu_si128(cg_ptr.add(uv_x) as *mut __m128i, cg);
                _mm_storeu_si128(co_ptr.add(uv_x) as *mut __m128i, co);
                uv_x += 16;
            }
        }

        cx += 16;
    }

    return ProcessedOffset { cx, ux: uv_x };
}

fn rgbx_to_ycgco<const ORIGIN_CHANNELS: u8, const SAMPLING: u8>(
    y_plane: &mut [u8],
    y_stride: u32,
    cg_plane: &mut [u8],
    cg_stride: u32,
    co_plane: &mut [u8],
    co_stride: u32,
    rgba: &[u8],
    rgba_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
) {
    let chroma_subsampling: YuvChromaSample = SAMPLING.into();
    let source_channels: YuvSourceChannels = ORIGIN_CHANNELS.into();
    let channels = source_channels.get_channels_count();
    let range = get_yuv_range(8, range);
    let precision_scale = (1 << 8) as f32;
    let bias_y = ((range.bias_y as f32 + 0.5f32) * precision_scale) as i32;
    let bias_uv = ((range.bias_uv as f32 + 0.5f32) * precision_scale) as i32;
    let max_colors = 2i32.pow(8) - 1i32;

    let iterator_step = match chroma_subsampling {
        YuvChromaSample::YUV420 => 2usize,
        YuvChromaSample::YUV422 => 2usize,
        YuvChromaSample::YUV444 => 1usize,
    };

    let range_reduction_y =
        (range.range_y as f32 / max_colors as f32 * precision_scale).round() as i32;
    let range_reduction_uv =
        (range.range_uv as f32 / max_colors as f32 * precision_scale).round() as i32;

    let mut y_offset = 0usize;
    let mut cg_offset = 0usize;
    let mut co_offset = 0usize;
    let mut rgba_offset = 0usize;

    #[cfg(target_arch = "x86_64")]
    let mut _use_sse = false;
    #[cfg(target_arch = "x86_64")]
    let mut _use_avx = false;

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("sse4.1") {
            _use_sse = true;
        }
        #[cfg(target_feature = "avx2")]
        if is_x86_feature_detected!("avx2") {
            _use_avx = true;
        }
    }

    for y in 0..height as usize {
        #[allow(unused_variables)]
        #[allow(unused_mut)]
        let mut cx = 0usize;
        #[allow(unused_variables)]
        #[allow(unused_mut)]
        let mut ux = 0usize;

        #[cfg(target_arch = "x86_64")]
        unsafe {
            #[cfg(target_feature = "avx2")]
            if _use_avx {
                let processed_offset = avx2_row::<ORIGIN_CHANNELS, SAMPLING>(
                    &range,
                    y_plane.as_mut_ptr(),
                    cg_plane.as_mut_ptr(),
                    co_plane.as_mut_ptr(),
                    &rgba,
                    y_offset,
                    cg_offset,
                    co_offset,
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
                    &range,
                    y_plane.as_mut_ptr(),
                    cg_plane.as_mut_ptr(),
                    co_plane.as_mut_ptr(),
                    &rgba,
                    y_offset,
                    cg_offset,
                    co_offset,
                    rgba_offset,
                    cx,
                    ux,
                    width as usize,
                );
                cx = processed_offset.cx;
                ux = processed_offset.ux;
            }
        }

        for x in (cx..width as usize).step_by(iterator_step) {
            let px = x * channels;
            let mut r = rgba[rgba_offset + px + source_channels.get_r_channel_offset()] as i32;
            let mut g = rgba[rgba_offset + px + source_channels.get_g_channel_offset()] as i32;
            let mut b = rgba[rgba_offset + px + source_channels.get_b_channel_offset()] as i32;

            let hg = g * range_reduction_y >> 1;
            let y_0 = (hg + ((r * range_reduction_y + b * range_reduction_y) >> 2) + bias_y) >> 8;
            r *= range_reduction_uv;
            g *= range_reduction_uv;
            b *= range_reduction_uv;
            let cg = (((g >> 1) - ((r + b) >> 2)) + bias_uv) >> 8;
            let co = (((r - b) >> 1) + bias_uv) >> 8;
            y_plane[y_offset + x] = y_0 as u8;
            let u_pos = match chroma_subsampling {
                YuvChromaSample::YUV420 | YuvChromaSample::YUV422 => cg_offset + ux,
                YuvChromaSample::YUV444 => cg_offset + ux,
            };
            cg_plane[u_pos] = cg as u8;
            let v_pos = match chroma_subsampling {
                YuvChromaSample::YUV420 | YuvChromaSample::YUV422 => co_offset + ux,
                YuvChromaSample::YUV444 => co_offset + ux,
            };
            co_plane[v_pos] = co as u8;
            match chroma_subsampling {
                YuvChromaSample::YUV420 | YuvChromaSample::YUV422 => {
                    if x + 1 < width as usize {
                        let next_px = (x + 1) * channels;
                        let r = rgba[rgba_offset + next_px + source_channels.get_r_channel_offset()]
                            as i32;
                        let g = rgba[rgba_offset + next_px + source_channels.get_g_channel_offset()]
                            as i32;
                        let b = rgba[rgba_offset + next_px + source_channels.get_b_channel_offset()]
                            as i32;
                        let hg_1 = g * range_reduction_y >> 1;
                        let y_1 = (hg_1
                            + ((r * range_reduction_y + b * range_reduction_y) >> 2)
                            + bias_y)
                            >> 8;
                        y_plane[y_offset + x + 1] = y_1 as u8;
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
                    cg_offset += cg_stride as usize;
                    co_offset += co_stride as usize;
                }
            }
            YuvChromaSample::YUV444 | YuvChromaSample::YUV422 => {
                cg_offset += cg_stride as usize;
                co_offset += co_stride as usize;
            }
        }
    }
}

/// Convert RGB image data to YCgCo 422 planar format.
///
/// This function performs RGB to YCgCo conversion and stores the result in YUV422 planar format,
/// with separate planes for Y (luminance), Cg (chrominance), and Co (chrominance) components.
/// YCgCo is very fast transformation by its nature. If you just work if intensity (Y channel) and do not require YCbCr prefer this one over YCbCr
///
/// # Arguments
///
/// * `y_plane` - A mutable slice to store the Y (luminance) plane data.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `cg_plane` - A mutable slice to store the Cg (chrominance) plane data.
/// * `cg_stride` - The stride (bytes per row) for the Cg plane.
/// * `co_plane` - A mutable slice to store the Co (chrominance) plane data.
/// * `co_stride` - The stride (bytes per row) for the Co plane.
/// * `rgb` - The input RGB image data slice.
/// * `rgb_stride` - The stride (bytes per row) for the RGB image data.
/// * `width` - The width of the image in pixels.
/// * `height` - The height of the image in pixels.
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input RGB data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn rgb_to_ycgco422(
    y_plane: &mut [u8],
    y_stride: u32,
    cg_plane: &mut [u8],
    cg_stride: u32,
    co_plane: &mut [u8],
    co_stride: u32,
    rgb: &[u8],
    rgb_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
) {
    rgbx_to_ycgco::<{ YuvSourceChannels::Rgb as u8 }, { YuvChromaSample::YUV422 as u8 }>(
        y_plane, y_stride, cg_plane, cg_stride, co_plane, co_stride, rgb, rgb_stride, width,
        height, range,
    );
}

/// Convert RGBA image data to YCgCo 422 planar format.
///
/// This function performs RGBA to YCgCo conversion and stores the result in YUV422 planar format,
/// with separate planes for Y (luminance), Cg (chrominance), and Co (chrominance) components.
/// YCgCo is very fast transformation by its nature. If you just work if intensity (Y channel) and do not require YCbCr prefer this one over YCbCr
///
/// # Arguments
///
/// * `y_plane` - A mutable slice to store the Y (luminance) plane data.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `cg_plane` - A mutable slice to store the Cg (chrominance) plane data.
/// * `cg_stride` - The stride (bytes per row) for the Cg plane.
/// * `co_plane` - A mutable slice to store the Co (chrominance) plane data.
/// * `co_stride` - The stride (bytes per row) for the Co plane.
/// * `rgba` - The input RGBA image data slice.
/// * `rgba_stride` - The stride (bytes per row) for the RGBA image data.
/// * `width` - The width of the image in pixels.
/// * `height` - The height of the image in pixels.
/// * `range` - The YUV range (limited or full).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input RGBA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn rgba_to_ycgco422(
    y_plane: &mut [u8],
    y_stride: u32,
    cg_plane: &mut [u8],
    cg_stride: u32,
    co_plane: &mut [u8],
    co_stride: u32,
    rgba: &[u8],
    rgba_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
) {
    rgbx_to_ycgco::<{ YuvSourceChannels::Rgba as u8 }, { YuvChromaSample::YUV422 as u8 }>(
        y_plane,
        y_stride,
        cg_plane,
        cg_stride,
        co_plane,
        co_stride,
        rgba,
        rgba_stride,
        width,
        height,
        range,
    );
}

/// Convert BGRA image data to YCgCo 422 planar format.
///
/// This function performs BGRA to YCgCo conversion and stores the result in YUV422 planar format,
/// with separate planes for Y (luminance), Cg (chrominance), and Co (chrominance) components.
/// YCgCo is very fast transformation by its nature. If you just work if intensity (Y channel) and do not require YCbCr prefer this one over YCbCr
///
/// # Arguments
///
/// * `y_plane` - A mutable slice to store the Y (luminance) plane data.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `cg_plane` - A mutable slice to store the Cg (chrominance) plane data.
/// * `cg_stride` - The stride (bytes per row) for the Cg plane.
/// * `co_plane` - A mutable slice to store the Co (chrominance) plane data.
/// * `co_stride` - The stride (bytes per row) for the Co plane.
/// * `bgra` - The input BGRA image data slice.
/// * `bgra_stride` - The stride (bytes per row) for the BGRA image data.
/// * `width` - The width of the image in pixels.
/// * `height` - The height of the image in pixels.
/// * `range` - The YUV range (limited or full).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGRA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn bgra_to_ycgco422(
    y_plane: &mut [u8],
    y_stride: u32,
    cg_plane: &mut [u8],
    cg_stride: u32,
    co_plane: &mut [u8],
    co_stride: u32,
    bgra: &[u8],
    bgra_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
) {
    rgbx_to_ycgco::<{ YuvSourceChannels::Bgra as u8 }, { YuvChromaSample::YUV422 as u8 }>(
        y_plane,
        y_stride,
        cg_plane,
        cg_stride,
        co_plane,
        co_stride,
        bgra,
        bgra_stride,
        width,
        height,
        range,
    );
}

/// Convert RGB image data to YCgCo 420 planar format.
///
/// This function performs RGB to YCgCo conversion and stores the result in YUV420 planar format,
/// with separate planes for Y (luminance), Cg (chrominance), and Co (chrominance) components.
/// YCgCo is very fast transformation by its nature. If you just work if intensity (Y channel) and do not require YCbCr prefer this one over YCbCr
///
/// # Arguments
///
/// * `y_plane` - A mutable slice to store the Y (luminance) plane data.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `cg_plane` - A mutable slice to store the Cg (chrominance) plane data.
/// * `cg_stride` - The stride (bytes per row) for the Cg plane.
/// * `co_plane` - A mutable slice to store the Co (chrominance) plane data.
/// * `co_stride` - The stride (bytes per row) for the Co plane.
/// * `rgb` - The input RGB image data slice.
/// * `rgb_stride` - The stride (bytes per row) for the RGB image data.
/// * `width` - The width of the image in pixels.
/// * `height` - The height of the image in pixels.
/// * `range` - The YUV range (limited or full).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input RGBA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn rgb_to_ycgco420(
    y_plane: &mut [u8],
    y_stride: u32,
    cg_plane: &mut [u8],
    cg_stride: u32,
    co_plane: &mut [u8],
    co_stride: u32,
    rgb: &[u8],
    rgb_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
) {
    rgbx_to_ycgco::<{ YuvSourceChannels::Rgb as u8 }, { YuvChromaSample::YUV420 as u8 }>(
        y_plane, y_stride, cg_plane, cg_stride, co_plane, co_stride, rgb, rgb_stride, width,
        height, range,
    );
}

/// Convert RGBA image data to YCgCo 420 planar format.
///
/// This function performs RGBA to YCgCo conversion and stores the result in YUV420 planar format,
/// with separate planes for Y (luminance), Cg (chrominance), and Co (chrominance) components.
/// YCgCo is very fast transformation by its nature. If you just work if intensity (Y channel) and do not require YCbCr prefer this one over YCbCr
///
/// # Arguments
///
/// * `y_plane` - A mutable slice to store the Y (luminance) plane data.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `cg_plane` - A mutable slice to store the Cg (chrominance) plane data.
/// * `cg_stride` - The stride (bytes per row) for the Cg plane.
/// * `co_plane` - A mutable slice to store the Co (chrominance) plane data.
/// * `co_stride` - The stride (bytes per row) for the Co plane.
/// * `rgba` - The input RGBA image data slice.
/// * `rgba_stride` - The stride (bytes per row) for the RGBA image data.
/// * `width` - The width of the image in pixels.
/// * `height` - The height of the image in pixels.
/// * `range` - The YUV range (limited or full).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input RGBA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn rgba_to_ycgco420(
    y_plane: &mut [u8],
    y_stride: u32,
    cg_plane: &mut [u8],
    cg_stride: u32,
    co_plane: &mut [u8],
    co_stride: u32,
    rgba: &[u8],
    rgba_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
) {
    rgbx_to_ycgco::<{ YuvSourceChannels::Rgba as u8 }, { YuvChromaSample::YUV420 as u8 }>(
        y_plane,
        y_stride,
        cg_plane,
        cg_stride,
        co_plane,
        co_stride,
        rgba,
        rgba_stride,
        width,
        height,
        range,
    );
}

/// Convert BGRA image data to YCgCo 420 planar format.
///
/// This function performs BGRA to YCgCo conversion and stores the result in YUV420 planar format,
/// with separate planes for Y (luminance), Cg (chrominance), and Co (chrominance) components.
/// YCgCo is very fast transformation by its nature. If you just work if intensity (Y channel) and do not require YCbCr prefer this one over YCbCr
///
/// # Arguments
///
/// * `y_plane` - A mutable slice to store the Y (luminance) plane data.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `cg_plane` - A mutable slice to store the Cg (chrominance) plane data.
/// * `cg_stride` - The stride (bytes per row) for the Cg plane.
/// * `co_plane` - A mutable slice to store the Co (chrominance) plane data.
/// * `co_stride` - The stride (bytes per row) for the Co plane.
/// * `bgra` - The input BGRA image data slice.
/// * `bgra_stride` - The stride (bytes per row) for the BGRA image data.
/// * `width` - The width of the image in pixels.
/// * `height` - The height of the image in pixels.
/// * `range` - The YUV range (limited or full).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGRA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn bgra_to_ycgco420(
    y_plane: &mut [u8],
    y_stride: u32,
    cg_plane: &mut [u8],
    cg_stride: u32,
    co_plane: &mut [u8],
    co_stride: u32,
    bgra: &[u8],
    bgra_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
) {
    rgbx_to_ycgco::<{ YuvSourceChannels::Bgra as u8 }, { YuvChromaSample::YUV420 as u8 }>(
        y_plane,
        y_stride,
        cg_plane,
        cg_stride,
        co_plane,
        co_stride,
        bgra,
        bgra_stride,
        width,
        height,
        range,
    );
}

/// Convert RGB image data to YCgCo 444 planar format.
///
/// This function performs RGB to YCgCo conversion and stores the result in YUV444 planar format,
/// with separate planes for Y (luminance), Cg (chrominance), and Co (chrominance) components.
/// YCgCo is very fast transformation by its nature. If you just work if intensity (Y channel) and do not require YCbCr prefer this one over YCbCr
///
/// # Arguments
///
/// * `y_plane` - A mutable slice to store the Y (luminance) plane data.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `cg_plane` - A mutable slice to store the Cg (chrominance) plane data.
/// * `cg_stride` - The stride (bytes per row) for the Cg plane.
/// * `co_plane` - A mutable slice to store the Co (chrominance) plane data.
/// * `co_stride` - The stride (bytes per row) for the Co plane.
/// * `rgb` - The input RGB image data slice.
/// * `rgb_stride` - The stride (bytes per row) for the RGB image data.
/// * `width` - The width of the image in pixels.
/// * `height` - The height of the image in pixels.
/// * `range` - The YUV range (limited or full).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input RGB data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn rgb_to_ycgco444(
    y_plane: &mut [u8],
    y_stride: u32,
    cg_plane: &mut [u8],
    cg_stride: u32,
    co_plane: &mut [u8],
    co_stride: u32,
    rgb: &[u8],
    rgb_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
) {
    rgbx_to_ycgco::<{ YuvSourceChannels::Rgb as u8 }, { YuvChromaSample::YUV444 as u8 }>(
        y_plane, y_stride, cg_plane, cg_stride, co_plane, co_stride, rgb, rgb_stride, width,
        height, range,
    );
}

/// Convert RGBA image data to YCgCo 444 planar format.
///
/// This function performs RGBA to YCgCo conversion and stores the result in YUV444 planar format,
/// with separate planes for Y (luminance), Cg (chrominance), and Co (chrominance) components.
/// YCgCo is very fast transformation by its nature. If you just work if intensity (Y channel) and do not require YCbCr prefer this one over YCbCr
///
/// # Arguments
///
/// * `y_plane` - A mutable slice to store the Y (luminance) plane data.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `cg_plane` - A mutable slice to store the Cg (chrominance) plane data.
/// * `cg_stride` - The stride (bytes per row) for the Cg plane.
/// * `co_plane` - A mutable slice to store the Co (chrominance) plane data.
/// * `co_stride` - The stride (bytes per row) for the Co plane.
/// * `rgba` - The input RGBA image data slice.
/// * `rgba_stride` - The stride (bytes per row) for the RGBA image data.
/// * `width` - The width of the image in pixels.
/// * `height` - The height of the image in pixels.
/// * `range` - The YUV range (limited or full).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input RGBA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn rgba_to_ycgco444(
    y_plane: &mut [u8],
    y_stride: u32,
    cg_plane: &mut [u8],
    cg_stride: u32,
    co_plane: &mut [u8],
    co_stride: u32,
    rgba: &[u8],
    rgba_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
) {
    rgbx_to_ycgco::<{ YuvSourceChannels::Rgba as u8 }, { YuvChromaSample::YUV444 as u8 }>(
        y_plane,
        y_stride,
        cg_plane,
        cg_stride,
        co_plane,
        co_stride,
        rgba,
        rgba_stride,
        width,
        height,
        range,
    );
}

/// Convert BGRA image data to YCgCo 444 planar format.
///
/// This function performs BGRA to YCgCo conversion and stores the result in YUV444 planar format,
/// with separate planes for Y (luminance), Cg (chrominance), and Co (chrominance) components.
/// YCgCo is very fast transformation by its nature. If you just work if intensity (Y channel) and do not require YCbCr prefer this one over YCbCr
///
/// # Arguments
///
/// * `y_plane` - A mutable slice to store the Y (luminance) plane data.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `cg_plane` - A mutable slice to store the Cg (chrominance) plane data.
/// * `cg_stride` - The stride (bytes per row) for the Cg plane.
/// * `co_plane` - A mutable slice to store the Co (chrominance) plane data.
/// * `co_stride` - The stride (bytes per row) for the Co plane.
/// * `bgra` - The input BGRA image data slice.
/// * `bgra_stride` - The stride (bytes per row) for the BGRA image data.
/// * `width` - The width of the image in pixels.
/// * `height` - The height of the image in pixels.
/// * `range` - The YUV range (limited or full).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGRA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn bgra_to_ycgco444(
    y_plane: &mut [u8],
    y_stride: u32,
    cg_plane: &mut [u8],
    cg_stride: u32,
    co_plane: &mut [u8],
    co_stride: u32,
    bgra: &[u8],
    bgra_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
) {
    rgbx_to_ycgco::<{ YuvSourceChannels::Bgra as u8 }, { YuvChromaSample::YUV444 as u8 }>(
        y_plane,
        y_stride,
        cg_plane,
        cg_stride,
        co_plane,
        co_stride,
        bgra,
        bgra_stride,
        width,
        height,
        range,
    );
}
