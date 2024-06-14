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
use crate::avx2::avx2_yuv_to_rgba_alpha;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[cfg(feature = "nightly_avx512")]
use crate::avx512bw::avx512_yuv_to_rgba_alpha;
#[allow(unused_imports)]
use crate::internals::ProcessedOffset;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use crate::x86_simd_support::*;
#[allow(unused_imports)]
use crate::yuv_support::*;
use crate::{YuvRange, YuvStandardMatrix};
#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use crate::neon::neon_yuv_to_rgba_alpha;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline(always)]
#[allow(dead_code)]
unsafe fn sse42_process_row<const DESTINATION_CHANNELS: u8, const SAMPLING: u8>(
    range: &YuvChromaRange,
    transform: &CbCrInverseTransform<i32>,
    y_plane: &[u8],
    u_plane: &[u8],
    v_plane: &[u8],
    a_plane: &[u8],
    rgba: &mut [u8],
    start_cx: usize,
    start_ux: usize,
    y_offset: usize,
    u_offset: usize,
    v_offset: usize,
    a_offset: usize,
    rgba_offset: usize,
    width: usize,
    use_premultiply: bool,
) -> ProcessedOffset {
    let chroma_subsampling: YuvChromaSample = SAMPLING.into();
    let destination_channels: YuvSourceChannels = DESTINATION_CHANNELS.into();
    let channels = destination_channels.get_channels_count();

    let mut cx = start_cx;
    let mut uv_x = start_ux;

    let y_ptr = y_plane.as_ptr();
    let u_ptr = u_plane.as_ptr();
    let v_ptr = v_plane.as_ptr();
    let a_ptr = a_plane.as_ptr();
    let rgba_ptr = rgba.as_mut_ptr();

    let y_corr = _mm_set1_epi8(range.bias_y as i8);
    let uv_corr = _mm_set1_epi16(range.bias_uv as i16);
    let v_luma_coeff = _mm_set1_epi16(transform.y_coef as i16);
    let v_cr_coeff = _mm_set1_epi16(transform.cr_coef as i16);
    let v_cb_coeff = _mm_set1_epi16(transform.cb_coef as i16);
    let v_min_values = _mm_setzero_si128();
    let v_g_coeff_1 = _mm_set1_epi16(-1 * transform.g_coeff_1 as i16);
    let v_g_coeff_2 = _mm_set1_epi16(-1 * transform.g_coeff_2 as i16);

    while cx + 16 < width {
        let y_values = _mm_subs_epi8(
            _mm_loadu_si128(y_ptr.add(y_offset + cx) as *const __m128i),
            y_corr,
        );

        let a_values = _mm_loadu_si128(a_ptr.add(a_offset + cx) as *const __m128i);

        let (u_high_u8, v_high_u8, u_low_u8, v_low_u8);

        match chroma_subsampling {
            YuvChromaSample::YUV420 | YuvChromaSample::YUV422 => {
                let reshuffle = _mm_setr_epi8(0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7);
                let (u_values, v_values);
                u_values = _mm_shuffle_epi8(_mm_loadu_si64(u_ptr.add(u_offset + uv_x)), reshuffle);
                v_values = _mm_shuffle_epi8(_mm_loadu_si64(v_ptr.add(v_offset + uv_x)), reshuffle);

                u_high_u8 = _mm_unpackhi_epi8(u_values, u_values);
                v_high_u8 = _mm_unpackhi_epi8(v_values, v_values);
                u_low_u8 = _mm_unpacklo_epi8(u_values, u_values);
                v_low_u8 = _mm_unpacklo_epi8(v_values, v_values);
            }
            YuvChromaSample::YUV444 => {
                let u_values = _mm_loadu_si128(u_ptr.add(u_offset + uv_x) as *const __m128i);
                let v_values = _mm_loadu_si128(v_ptr.add(v_offset + uv_x) as *const __m128i);

                u_high_u8 = _mm_unpackhi_epi8(u_values, u_values);
                v_high_u8 = _mm_unpackhi_epi8(v_values, v_values);
                u_low_u8 = _mm_unpacklo_epi8(u_values, u_values);
                v_low_u8 = _mm_unpacklo_epi8(v_values, v_values);
            }
        }

        let u_high = _mm_subs_epi16(_mm_cvtepu8_epi16(u_high_u8), uv_corr);
        let v_high = _mm_subs_epi16(_mm_cvtepu8_epi16(v_high_u8), uv_corr);
        let y_high = _mm_mullo_epi16(
            _mm_cvtepu8_epi16(_mm_srli_si128::<8>(y_values)),
            v_luma_coeff,
        );

        let r_high = _mm_srai_epi16::<6>(_mm_max_epi16(
            _mm_adds_epi16(y_high, _mm_mullo_epi16(v_high, v_cr_coeff)),
            v_min_values,
        ));
        let b_high = _mm_srai_epi16::<6>(_mm_max_epi16(
            _mm_adds_epi16(y_high, _mm_mullo_epi16(u_high, v_cb_coeff)),
            v_min_values,
        ));
        let g_high = _mm_srai_epi16::<6>(_mm_max_epi16(
            _mm_adds_epi16(
                y_high,
                _mm_adds_epi16(
                    _mm_mullo_epi16(v_high, v_g_coeff_1),
                    _mm_mullo_epi16(u_high, v_g_coeff_2),
                ),
            ),
            v_min_values,
        ));

        let u_low = _mm_sub_epi16(_mm_cvtepu8_epi16(u_low_u8), uv_corr);
        let v_low = _mm_sub_epi16(_mm_cvtepu8_epi16(v_low_u8), uv_corr);
        let y_low = _mm_mullo_epi16(_mm_cvtepu8_epi16(y_values), v_luma_coeff);

        let r_low = _mm_srai_epi16::<6>(_mm_max_epi16(
            _mm_adds_epi16(y_low, _mm_mullo_epi16(v_low, v_cr_coeff)),
            v_min_values,
        ));
        let b_low = _mm_srai_epi16::<6>(_mm_max_epi16(
            _mm_adds_epi16(y_low, _mm_mullo_epi16(u_low, v_cb_coeff)),
            v_min_values,
        ));
        let g_low = _mm_srai_epi16::<6>(_mm_max_epi16(
            _mm_adds_epi16(
                y_low,
                _mm_adds_epi16(
                    _mm_mullo_epi16(v_low, v_g_coeff_1),
                    _mm_mullo_epi16(u_low, v_g_coeff_2),
                ),
            ),
            v_min_values,
        ));

        let (r_values, g_values, b_values);

        if use_premultiply {
            let a_h = _mm_cvtepu8_epi16(_mm_srli_si128::<8>(a_values));
            let a_l = _mm_cvtepu8_epi16(a_values);
            let r_h_16 = sse_div_by255(_mm_mullo_epi16(r_high, a_h));
            let r_l_16 = sse_div_by255(_mm_mullo_epi16(r_low, a_l));
            let g_h_16 = sse_div_by255(_mm_mullo_epi16(g_high, a_h));
            let g_l_16 = sse_div_by255(_mm_mullo_epi16(g_low, a_l));
            let b_h_16 = sse_div_by255(_mm_mullo_epi16(b_high, a_h));
            let b_l_16 = sse_div_by255(_mm_mullo_epi16(b_low, a_l));

            r_values = _mm_packus_epi16(r_l_16, r_h_16);
            g_values = _mm_packus_epi16(g_l_16, g_h_16);
            b_values = _mm_packus_epi16(b_l_16, b_h_16);
        } else {
            r_values = _mm_packus_epi16(r_low, r_high);
            g_values = _mm_packus_epi16(g_low, g_high);
            b_values = _mm_packus_epi16(b_low, b_high);
        }

        let dst_shift = rgba_offset + cx * channels;

        match destination_channels {
            YuvSourceChannels::Rgb => {
                sse_store_rgb_u8(rgba_ptr.add(dst_shift), r_values, g_values, b_values);
            }
            YuvSourceChannels::Rgba => {
                sse_store_rgba(
                    rgba_ptr.add(dst_shift),
                    r_values,
                    g_values,
                    b_values,
                    a_values,
                );
            }
            YuvSourceChannels::Bgra => {
                sse_store_rgba(
                    rgba_ptr.add(dst_shift),
                    b_values,
                    g_values,
                    r_values,
                    a_values,
                );
            }
        }

        cx += 16;

        match chroma_subsampling {
            YuvChromaSample::YUV420 | YuvChromaSample::YUV422 => {
                uv_x += 8;
            }
            YuvChromaSample::YUV444 => {
                uv_x += 16;
            }
        }
    }

    return ProcessedOffset { cx, ux: uv_x };
}

fn yuv_with_alpha_to_rgbx<const DESTINATION_CHANNELS: u8, const SAMPLING: u8>(
    y_plane: &[u8],
    y_stride: u32,
    u_plane: &[u8],
    u_stride: u32,
    v_plane: &[u8],
    v_stride: u32,
    a_plane: &[u8],
    a_stride: u32,
    rgba: &mut [u8],
    rgba_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    premultiply_alpha: bool,
) {
    let chroma_subsampling: YuvChromaSample = SAMPLING.into();
    let destination_channels: YuvSourceChannels = DESTINATION_CHANNELS.into();
    if !destination_channels.has_alpha() {
        panic!("yuv_with_alpha_to_rgbx cannot be called on configuration without alpha");
    }
    let channels = destination_channels.get_channels_count();
    let range = get_yuv_range(8, range);
    let kr_kb = get_kr_kb(matrix);
    let transform = get_inverse_transform(255, range.range_y, range.range_uv, kr_kb.kr, kr_kb.kb);
    let inverse_transform = transform.to_integers(6u32);

    let cr_coef = inverse_transform.cr_coef;
    let cb_coef = inverse_transform.cb_coef;
    let y_coef = inverse_transform.y_coef;
    let g_coef_1 = inverse_transform.g_coeff_1;
    let g_coef_2 = inverse_transform.g_coeff_2;

    let bias_y = range.bias_y as i32;
    let bias_uv = range.bias_uv as i32;

    let mut y_offset = 0usize;
    let mut u_offset = 0usize;
    let mut v_offset = 0usize;
    let mut a_offset = 0usize;
    let mut rgba_offset = 0usize;

    let iterator_step = match chroma_subsampling {
        YuvChromaSample::YUV420 => 2usize,
        YuvChromaSample::YUV422 => 2usize,
        YuvChromaSample::YUV444 => 1usize,
    };

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    let mut _use_avx2 = false;
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    let mut _use_sse = false;
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    let mut _use_avx512 = false;

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        #[cfg(feature = "nightly_avx512")]
        if std::arch::is_x86_feature_detected!("avx512bw") {
            _use_avx512 = true;
        }
        #[cfg(target_feature = "avx2")]
        if std::arch::is_x86_feature_detected!("avx2") {
            _use_avx2 = true;
        }
        if std::arch::is_x86_feature_detected!("sse4.1") {
            _use_sse = true;
        }
    }

    for y in 0..height as usize {
        #[allow(unused_variables)]
        #[allow(unused_mut)]
        let mut cx = 0usize;

        #[allow(unused_variables)]
        #[allow(unused_mut)]
        let mut uv_x = 0usize;

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        unsafe {
            #[cfg(feature = "nightly_avx512")]
            {
                if _use_avx512 {
                    let processed = avx512_yuv_to_rgba_alpha::<DESTINATION_CHANNELS, SAMPLING>(
                        &range,
                        &inverse_transform,
                        y_plane,
                        u_plane,
                        v_plane,
                        a_plane,
                        rgba,
                        cx,
                        uv_x,
                        y_offset,
                        u_offset,
                        v_offset,
                        a_offset,
                        rgba_offset,
                        width as usize,
                        premultiply_alpha,
                    );
                    cx = processed.cx;
                    uv_x = processed.ux;
                }
            }
            #[cfg(target_feature = "avx2")]
            if _use_avx2 {
                let processed = avx2_yuv_to_rgba_alpha::<DESTINATION_CHANNELS, SAMPLING>(
                    &range,
                    &inverse_transform,
                    y_plane,
                    u_plane,
                    v_plane,
                    a_plane,
                    rgba,
                    cx,
                    uv_x,
                    y_offset,
                    u_offset,
                    v_offset,
                    a_offset,
                    rgba_offset,
                    width as usize,
                    premultiply_alpha,
                );
                cx = processed.cx;
                uv_x = processed.ux;
            }
            if _use_sse {
                let processed = sse42_process_row::<DESTINATION_CHANNELS, SAMPLING>(
                    &range,
                    &inverse_transform,
                    y_plane,
                    u_plane,
                    v_plane,
                    a_plane,
                    rgba,
                    cx,
                    uv_x,
                    y_offset,
                    u_offset,
                    v_offset,
                    a_offset,
                    rgba_offset,
                    width as usize,
                    premultiply_alpha,
                );
                cx = processed.cx;
                uv_x = processed.ux;
            }
        }

        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        unsafe {
            let processed = neon_yuv_to_rgba_alpha::<DESTINATION_CHANNELS, SAMPLING>(
                &range,
                &inverse_transform,
                y_plane,
                u_plane,
                v_plane,
                a_plane,
                rgba,
                cx,
                uv_x,
                y_offset,
                u_offset,
                v_offset,
                a_offset,
                rgba_offset,
                width as usize,
                premultiply_alpha,
            );
            cx = processed.cx;
            uv_x = processed.ux;
        }

        for x in (cx..width as usize).step_by(iterator_step) {
            let y_value =
                (unsafe { *y_plane.get_unchecked(y_offset + x) } as i32 - bias_y) * y_coef;

            let u_pos = match chroma_subsampling {
                YuvChromaSample::YUV420 | YuvChromaSample::YUV422 => u_offset + uv_x,
                YuvChromaSample::YUV444 => u_offset + uv_x,
            };

            let cb_value = unsafe { *u_plane.get_unchecked(u_pos) } as i32 - bias_uv;

            let v_pos = match chroma_subsampling {
                YuvChromaSample::YUV420 | YuvChromaSample::YUV422 => v_offset + uv_x,
                YuvChromaSample::YUV444 => v_offset + uv_x,
            };

            let cr_value = unsafe { *v_plane.get_unchecked(v_pos) } as i32 - bias_uv;

            let mut r = ((y_value + cr_coef * cr_value) >> 6).min(255).max(0);
            let mut b = ((y_value + cb_coef * cb_value) >> 6).min(255).max(0);
            let mut g = ((y_value - g_coef_1 * cr_value - g_coef_2 * cb_value) >> 6)
                .min(255)
                .max(0);

            let px = x * channels;

            let rgba_shift = rgba_offset + px;

            let a_value = unsafe { *a_plane.get_unchecked(a_offset + x) };
            if premultiply_alpha {
                r = (r * a_value as i32) / 255;
                g = (g * a_value as i32) / 255;
                b = (b * a_value as i32) / 255;
            }

            unsafe {
                *rgba.get_unchecked_mut(rgba_shift + destination_channels.get_r_channel_offset()) =
                    r as u8;
                *rgba.get_unchecked_mut(rgba_shift + destination_channels.get_g_channel_offset()) =
                    g as u8;
                *rgba.get_unchecked_mut(rgba_shift + destination_channels.get_b_channel_offset()) =
                    b as u8;
                *rgba.get_unchecked_mut(rgba_shift + destination_channels.get_a_channel_offset()) =
                    a_value;
            }

            if chroma_subsampling == YuvChromaSample::YUV420
                || chroma_subsampling == YuvChromaSample::YUV422
            {
                let next_x = x + 1;
                if x + 1 < width as usize {
                    let y_value = (unsafe { *y_plane.get_unchecked(y_offset + x + 1) } as i32
                        - bias_y)
                        * y_coef;

                    let mut r = ((y_value + cr_coef * cr_value) >> 6).min(255).max(0);
                    let mut b = ((y_value + cb_coef * cb_value) >> 6).min(255).max(0);
                    let mut g = ((y_value - g_coef_1 * cr_value - g_coef_2 * cb_value) >> 6)
                        .min(255)
                        .max(0);

                    let next_px = next_x * channels;

                    let rgba_shift = rgba_offset + next_px;

                    let a_value = unsafe { *a_plane.get_unchecked(a_offset + next_x) };
                    if premultiply_alpha {
                        r = (r * a_value as i32) / 255;
                        g = (g * a_value as i32) / 255;
                        b = (b * a_value as i32) / 255;
                    }

                    unsafe {
                        *rgba.get_unchecked_mut(
                            rgba_shift + destination_channels.get_r_channel_offset(),
                        ) = r as u8;
                        *rgba.get_unchecked_mut(
                            rgba_shift + destination_channels.get_g_channel_offset(),
                        ) = g as u8;
                        *rgba.get_unchecked_mut(
                            rgba_shift + destination_channels.get_b_channel_offset(),
                        ) = b as u8;
                        *rgba.get_unchecked_mut(
                            rgba_shift + destination_channels.get_a_channel_offset(),
                        ) = a_value;
                    }
                }
            }

            uv_x += 1;
        }

        y_offset += y_stride as usize;
        rgba_offset += rgba_stride as usize;
        a_offset += a_stride as usize;
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

/// Convert YUV 420 planar format to RGBA format and appends provided alpha channel.
///
/// This function takes YUV 420 planar format data with 8-bit precision,
/// and converts it to RGBA format with 8-bit per channel precision.
///
/// # Arguments
///
/// * `y_plane` - A slice to load the Y (luminance) plane data.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `u_plane` - A slice to load the U (chrominance) plane data.
/// * `u_stride` - The stride (bytes per row) for the U plane.
/// * `v_plane` - A slice to load the V (chrominance) plane data.
/// * `v_stride` - The stride (bytes per row) for the V plane.
/// * `a_plane` - A slice to load alpha plane to append to result.
/// * `a_stride` - The stride (bytes per row) for the alpha plane.
/// * `width` - The width of the YUV image.
/// * `height` - The height of the YUV image.
/// * `rgba_data` - A mutable slice to store the converted RGBA data.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
/// * `premultiply_alpha` - Flag to premultiply alpha or not
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGRA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv420_with_alpha_to_rgba(
    y_plane: &[u8],
    y_stride: u32,
    u_plane: &[u8],
    u_stride: u32,
    v_plane: &[u8],
    v_stride: u32,
    a_plane: &[u8],
    a_stride: u32,
    rgba: &mut [u8],
    rgba_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    premultiply_alpha: bool,
) {
    yuv_with_alpha_to_rgbx::<{ YuvSourceChannels::Rgba as u8 }, { YuvChromaSample::YUV420 as u8 }>(
        y_plane,
        y_stride,
        u_plane,
        u_stride,
        v_plane,
        v_stride,
        a_plane,
        a_stride,
        rgba,
        rgba_stride,
        width,
        height,
        range,
        matrix,
        premultiply_alpha,
    )
}

/// Convert YUV 420 planar format to BGRA format and appends provided alpha channel.
///
/// This function takes YUV 420 planar format data with 8-bit precision,
/// and converts it to BGRA format with 8-bit per channel precision.
///
/// # Arguments
///
/// * `y_plane` - A slice to load the Y (luminance) plane data.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `u_plane` - A slice to load the U (chrominance) plane data.
/// * `u_stride` - The stride (bytes per row) for the U plane.
/// * `v_plane` - A slice to load the V (chrominance) plane data.
/// * `v_stride` - The stride (bytes per row) for the V plane.
/// * `a_plane` - A slice to load alpha plane to append to result.
/// * `a_stride` - The stride (bytes per row) for the alpha plane.
/// * `width` - The width of the YUV image.
/// * `height` - The height of the YUV image.
/// * `bgra_data` - A mutable slice to store the converted BGRA data.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
/// * `premultiply_alpha` - Flag to premultiply alpha or not
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGRA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv420_with_alpha_to_bgra(
    y_plane: &[u8],
    y_stride: u32,
    u_plane: &[u8],
    u_stride: u32,
    v_plane: &[u8],
    v_stride: u32,
    a_plane: &[u8],
    a_stride: u32,
    bgra: &mut [u8],
    bgra_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    premultiply_alpha: bool,
) {
    yuv_with_alpha_to_rgbx::<{ YuvSourceChannels::Bgra as u8 }, { YuvChromaSample::YUV420 as u8 }>(
        y_plane,
        y_stride,
        u_plane,
        u_stride,
        v_plane,
        v_stride,
        a_plane,
        a_stride,
        bgra,
        bgra_stride,
        width,
        height,
        range,
        matrix,
        premultiply_alpha,
    )
}

/// Convert YUV 422 planar format to RGBA format and appends provided alpha channel.
///
/// This function takes YUV 422 planar format data with 8-bit precision,
/// and converts it to RGBA format with 8-bit per channel precision.
///
/// # Arguments
///
/// * `y_plane` - A slice to load the Y (luminance) plane data.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `u_plane` - A slice to load the U (chrominance) plane data.
/// * `u_stride` - The stride (bytes per row) for the U plane.
/// * `v_plane` - A slice to load the V (chrominance) plane data.
/// * `v_stride` - The stride (bytes per row) for the V plane.
/// * `a_plane` - A slice to load alpha plane to append to result.
/// * `a_stride` - The stride (bytes per row) for the alpha plane.
/// * `width` - The width of the YUV image.
/// * `height` - The height of the YUV image.
/// * `rgba_data` - A mutable slice to store the converted RGBA data.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
/// * `premultiply_alpha` - Flag to premultiply alpha or not
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGRA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv422_with_alpha_to_rgba(
    y_plane: &[u8],
    y_stride: u32,
    u_plane: &[u8],
    u_stride: u32,
    v_plane: &[u8],
    v_stride: u32,
    a_plane: &[u8],
    a_stride: u32,
    rgba: &mut [u8],
    rgba_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    premultiply_alpha: bool,
) {
    yuv_with_alpha_to_rgbx::<{ YuvSourceChannels::Rgba as u8 }, { YuvChromaSample::YUV422 as u8 }>(
        y_plane,
        y_stride,
        u_plane,
        u_stride,
        v_plane,
        v_stride,
        a_plane,
        a_stride,
        rgba,
        rgba_stride,
        width,
        height,
        range,
        matrix,
        premultiply_alpha,
    )
}

/// Convert YUV 422 planar format to BGRA format and appends provided alpha channel.
///
/// This function takes YUV 422 planar format data with 8-bit precision,
/// and converts it to BGRA format with 8-bit per channel precision.
///
/// # Arguments
///
/// * `y_plane` - A slice to load the Y (luminance) plane data.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `u_plane` - A slice to load the U (chrominance) plane data.
/// * `u_stride` - The stride (bytes per row) for the U plane.
/// * `v_plane` - A slice to load the V (chrominance) plane data.
/// * `v_stride` - The stride (bytes per row) for the V plane.
/// * `a_plane` - A slice to load alpha plane to append to result.
/// * `a_stride` - The stride (bytes per row) for the alpha plane.
/// * `width` - The width of the YUV image.
/// * `height` - The height of the YUV image.
/// * `bgra_data` - A mutable slice to store the converted BGRA data.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
/// * `premultiply_alpha` - Flag to premultiply alpha or not
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGRA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv422_with_alpha_to_bgra(
    y_plane: &[u8],
    y_stride: u32,
    u_plane: &[u8],
    u_stride: u32,
    v_plane: &[u8],
    v_stride: u32,
    a_plane: &[u8],
    a_stride: u32,
    bgra: &mut [u8],
    bgra_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    premultiply_alpha: bool,
) {
    yuv_with_alpha_to_rgbx::<{ YuvSourceChannels::Bgra as u8 }, { YuvChromaSample::YUV422 as u8 }>(
        y_plane,
        y_stride,
        u_plane,
        u_stride,
        v_plane,
        v_stride,
        a_plane,
        a_stride,
        bgra,
        bgra_stride,
        width,
        height,
        range,
        matrix,
        premultiply_alpha,
    )
}

/// Convert YUV 444 planar format to RGBA format and appends provided alpha channel.
///
/// This function takes YUV 444 planar format data with 8-bit precision,
/// and converts it to RGBA format with 8-bit per channel precision.
///
/// # Arguments
///
/// * `y_plane` - A slice to load the Y (luminance) plane data.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `u_plane` - A slice to load the U (chrominance) plane data.
/// * `u_stride` - The stride (bytes per row) for the U plane.
/// * `v_plane` - A slice to load the V (chrominance) plane data.
/// * `v_stride` - The stride (bytes per row) for the V plane.
/// * `a_plane` - A slice to load alpha plane to append to result.
/// * `a_stride` - The stride (bytes per row) for the alpha plane.
/// * `width` - The width of the YUV image.
/// * `height` - The height of the YUV image.
/// * `rgba_data` - A mutable slice to store the converted RGBA data.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
/// * `premultiply_alpha` - Flag to premultiply alpha or not
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGRA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv444_with_alpha_to_rgba(
    y_plane: &[u8],
    y_stride: u32,
    u_plane: &[u8],
    u_stride: u32,
    v_plane: &[u8],
    v_stride: u32,
    a_plane: &[u8],
    a_stride: u32,
    rgba: &mut [u8],
    rgba_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    premultiply_alpha: bool,
) {
    yuv_with_alpha_to_rgbx::<{ YuvSourceChannels::Rgba as u8 }, { YuvChromaSample::YUV444 as u8 }>(
        y_plane,
        y_stride,
        u_plane,
        u_stride,
        v_plane,
        v_stride,
        a_plane,
        a_stride,
        rgba,
        rgba_stride,
        width,
        height,
        range,
        matrix,
        premultiply_alpha,
    )
}

/// Convert YUV 444 planar format to BGRA format and appends provided alpha channel.
///
/// This function takes YUV 444 planar format data with 8-bit precision,
/// and converts it to BGRA format with 8-bit per channel precision.
///
/// # Arguments
///
/// * `y_plane` - A slice to load the Y (luminance) plane data.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `u_plane` - A slice to load the U (chrominance) plane data.
/// * `u_stride` - The stride (bytes per row) for the U plane.
/// * `v_plane` - A slice to load the V (chrominance) plane data.
/// * `v_stride` - The stride (bytes per row) for the V plane.
/// * `a_plane` - A slice to load alpha plane to append to result.
/// * `a_stride` - The stride (bytes per row) for the alpha plane.
/// * `width` - The width of the YUV image.
/// * `height` - The height of the YUV image.
/// * `bgra_data` - A mutable slice to store the converted BGRA data.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
/// * `premultiply_alpha` - Flag to premultiply alpha or not
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGRA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv444_with_alpha_to_bgra(
    y_plane: &[u8],
    y_stride: u32,
    u_plane: &[u8],
    u_stride: u32,
    v_plane: &[u8],
    v_stride: u32,
    a_plane: &[u8],
    a_stride: u32,
    bgra: &mut [u8],
    bgra_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    premultiply_alpha: bool,
) {
    yuv_with_alpha_to_rgbx::<{ YuvSourceChannels::Bgra as u8 }, { YuvChromaSample::YUV444 as u8 }>(
        y_plane,
        y_stride,
        u_plane,
        u_stride,
        v_plane,
        v_stride,
        a_plane,
        a_stride,
        bgra,
        bgra_stride,
        width,
        height,
        range,
        matrix,
        premultiply_alpha,
    )
}
