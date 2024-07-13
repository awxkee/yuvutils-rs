/*
 * // Copyright (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use crate::neon::neon_ycgcor_to_rgb_row;
#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    target_feature = "sse4.1"
))]
use crate::sse::sse_ycgcor_type_to_rgb_row;
use crate::ycgcor_support::YCgCoR;
use crate::yuv_support::{get_yuv_range, YuvChromaSample, YuvSourceChannels};
use crate::YuvRange;

fn ycgco_r_type_ro_rgbx<const DESTINATION_CHANNELS: u8, const SAMPLING: u8, const R_TYPE: usize>(
    y_plane: &[u16],
    y_stride: u32,
    cg_plane: &[u16],
    cg_stride: u32,
    co_plane: &[u16],
    co_stride: u32,
    rgba: &mut [u8],
    rgba_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
) {
    let chroma_subsampling: YuvChromaSample = SAMPLING.into();
    let destination_channels: YuvSourceChannels = DESTINATION_CHANNELS.into();
    let channels = destination_channels.get_channels_count();
    let range = get_yuv_range(8, range);
    let bias_y = range.bias_y as i32;
    let bias_uv = range.bias_uv as i32;

    let mut y_offset = 0usize;
    let mut u_offset = 0usize;
    let mut v_offset = 0usize;
    let mut rgba_offset = 0usize;

    let iterator_step = match chroma_subsampling {
        YuvChromaSample::YUV420 => 2usize,
        YuvChromaSample::YUV422 => 2usize,
        YuvChromaSample::YUV444 => 1usize,
    };

    let max_colors = 2i32.pow(8) - 1i32;
    let precision_scale = (1 << 6) as f32;
    let range_reduction_y =
        (max_colors as f32 / range.range_y as f32 * precision_scale).round() as i32;
    let range_reduction_uv =
        (max_colors as f32 / range.range_uv as f32 * precision_scale).round() as i32;

    #[cfg(all(
        any(target_arch = "x86", target_arch = "x86_64"),
        target_feature = "sse4.1"
    ))]
    let mut _use_sse = false;

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        #[cfg(target_feature = "sse4.1")]
        if std::arch::is_x86_feature_detected!("sse4.1") {
            _use_sse = true;
        }
    }

    for y in 0..height as usize {
        let mut _cx = 0usize;
        let mut _uv_x = 0usize;

        let y_ptr = unsafe { (y_plane.as_ptr() as *const u8).add(y_offset) as *mut u16 };
        let cg_ptr = unsafe { (cg_plane.as_ptr() as *const u8).add(u_offset) as *mut u16 };
        let co_ptr = unsafe { (co_plane.as_ptr() as *const u8).add(v_offset) as *mut u16 };

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        #[allow(unused_unsafe)]
        unsafe {
            #[cfg(target_feature = "sse4.1")]
            if _use_sse {
                let offset = sse_ycgcor_type_to_rgb_row::<DESTINATION_CHANNELS, SAMPLING>(
                    &range,
                    y_ptr,
                    cg_ptr,
                    co_ptr,
                    rgba,
                    _cx,
                    _uv_x,
                    rgba_offset,
                    width as usize,
                );
                _cx = offset.cx;
                _uv_x = offset.ux;
            }
        }

        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        unsafe {
            let offset = neon_ycgcor_to_rgb_row::<DESTINATION_CHANNELS, SAMPLING>(
                &range,
                y_ptr,
                cg_ptr,
                co_ptr,
                rgba,
                _cx,
                _uv_x,
                rgba_offset,
                width as usize,
            );
            _cx = offset.cx;
            _uv_x = offset.ux;
        }

        for x in (_cx..width as usize).step_by(iterator_step) {
            let y_value =
                (unsafe { y_ptr.add(x).read_unaligned() as i32 } - bias_y) * range_reduction_y;

            let cg_pos = _uv_x;

            let cg_value = (unsafe { cg_ptr.add(cg_pos).read_unaligned() } as i32 - bias_uv)
                * range_reduction_uv;

            let v_pos = _uv_x;

            let co_value = (unsafe { co_ptr.add(v_pos).read_unaligned() } as i32 - bias_uv)
                * range_reduction_uv;

            let t = y_value - (cg_value >> 1);
            let g = (((t + cg_value).max(0)) >> 6).min(255);
            let b = t - (co_value >> 1);
            let r = ((b + co_value).max(0) >> 6).min(255);
            let b = (b.max(0) >> 6).min(255);

            // 11 118 169
            if r == 8 && g == 126 && b == 171 {
                println!("{} {}", x, y);
            }

            let px = x * channels;

            let rgba_shift = rgba_offset + px;

            unsafe {
                *rgba.get_unchecked_mut(rgba_shift + destination_channels.get_r_channel_offset()) =
                    r as u8
            };
            unsafe {
                *rgba.get_unchecked_mut(rgba_shift + destination_channels.get_g_channel_offset()) =
                    g as u8
            };
            unsafe {
                *rgba.get_unchecked_mut(rgba_shift + destination_channels.get_b_channel_offset()) =
                    b as u8
            };
            if destination_channels.has_alpha() {
                unsafe {
                    *rgba.get_unchecked_mut(
                        rgba_shift + destination_channels.get_a_channel_offset(),
                    ) = 255
                };
            }

            if chroma_subsampling == YuvChromaSample::YUV420
                || chroma_subsampling == YuvChromaSample::YUV422
            {
                let next_x = x + 1;
                if next_x < width as usize {
                    let y_value = unsafe { y_ptr.add(next_x).read_unaligned() } as i32 - bias_y;

                    let t = y_value - (cg_value >> 1);
                    let g = (((t + cg_value).max(0)) >> 6).min(255);
                    let b = t - (co_value >> 1);
                    let r = ((b + co_value).max(0) >> 6).min(255);
                    let b = (b.max(0) >> 6).min(255);

                    let next_px = next_x * channels;

                    let rgba_shift = rgba_offset + next_px;

                    unsafe {
                        *rgba.get_unchecked_mut(
                            rgba_shift + destination_channels.get_r_channel_offset(),
                        ) = r as u8
                    };
                    unsafe {
                        *rgba.get_unchecked_mut(
                            rgba_shift + destination_channels.get_g_channel_offset(),
                        ) = g as u8
                    };
                    unsafe {
                        *rgba.get_unchecked_mut(
                            rgba_shift + destination_channels.get_b_channel_offset(),
                        ) = b as u8
                    };
                    if destination_channels.has_alpha() {
                        unsafe {
                            *rgba.get_unchecked_mut(
                                rgba_shift + destination_channels.get_a_channel_offset(),
                            ) = 255
                        };
                    }
                }
            }

            _uv_x += 1;
        }

        y_offset += y_stride as usize;
        rgba_offset += rgba_stride as usize;
        match chroma_subsampling {
            YuvChromaSample::YUV420 => {
                if y & 1 == 1 {
                    u_offset += cg_stride as usize;
                    v_offset += co_stride as usize;
                }
            }
            YuvChromaSample::YUV444 | YuvChromaSample::YUV422 => {
                u_offset += cg_stride as usize;
                v_offset += co_stride as usize;
            }
        }
    }
}

/// Convert YCgCo-Ro 420 planar format to RGB format.
///
/// This function takes YCgCo-Ro 420 planar format data with 8-bit precision,
/// and converts it to RGB format with 8-bit per channel precision.
///
/// # Arguments
///
/// * `y_plane` - A slice to load the Y (luminance) plane data.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `cg_plane` - A slice to load the U (chrominance) plane data.
/// * `cg_stride` - The stride (bytes per row) for the U plane.
/// * `co_plane` - A slice to load the V (chrominance) plane data.
/// * `co_stride` - The stride (bytes per row) for the V plane.
/// * `width` - The width of the YUV image.
/// * `height` - The height of the YUV image.
/// * `rgb_data` - A mutable slice to store the converted RGB data.
/// * `range` - The YUV range (limited or full).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGRA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn ycgcoro420_to_rgb(
    y_plane: &[u16],
    y_stride: u32,
    cg_plane: &[u16],
    cg_stride: u32,
    co_plane: &[u16],
    co_stride: u32,
    rgb: &mut [u8],
    rgb_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
) {
    ycgco_r_type_ro_rgbx::<
        { YuvSourceChannels::Rgb as u8 },
        { YuvChromaSample::YUV420 as u8 },
        { YCgCoR::YCgCoRo as usize },
    >(
        y_plane, y_stride, cg_plane, cg_stride, co_plane, co_stride, rgb, rgb_stride, width,
        height, range,
    )
}

/// Convert YCgCo-Ro 420 planar format to RGBA format.
///
/// This function takes YCgCo-Ro 420 planar format data with 8-bit precision,
/// and converts it to RGBA format with 8-bit per channel precision.
///
/// # Arguments
///
/// * `y_plane` - A slice to load the Y (luminance) plane data.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `cg_plane` - A slice to load the U (chrominance) plane data.
/// * `cg_stride` - The stride (bytes per row) for the U plane.
/// * `co_plane` - A slice to load the V (chrominance) plane data.
/// * `co_stride` - The stride (bytes per row) for the V plane.
/// * `width` - The width of the YUV image.
/// * `height` - The height of the YUV image.
/// * `rgba_data` - A mutable slice to store the converted RGBA data.
/// * `range` - The YUV range (limited or full).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGRA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn ycgcoro420_to_rgba(
    y_plane: &[u16],
    y_stride: u32,
    cg_plane: &[u16],
    cg_stride: u32,
    co_plane: &[u16],
    co_stride: u32,
    rgba: &mut [u8],
    rgba_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
) {
    ycgco_r_type_ro_rgbx::<
        { YuvSourceChannels::Rgba as u8 },
        { YuvChromaSample::YUV420 as u8 },
        { YCgCoR::YCgCoRo as usize },
    >(
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
    )
}

/// Convert YCgCo-Ro 420 planar format to BGRA format.
///
/// This function takes YCgCo-Ro 420 planar format data with 8-bit precision,
/// and converts it to BGRA format with 8-bit per channel precision.
///
/// # Arguments
///
/// * `y_plane` - A slice to load the Y (luminance) plane data.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `cg_plane` - A slice to load the U (chrominance) plane data.
/// * `cg_stride` - The stride (bytes per row) for the U plane.
/// * `co_plane` - A slice to load the V (chrominance) plane data.
/// * `co_stride` - The stride (bytes per row) for the V plane.
/// * `width` - The width of the YUV image.
/// * `height` - The height of the YUV image.
/// * `bgra_data` - A mutable slice to store the converted BGRA data.
/// * `range` - The YUV range (limited or full).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGRA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn ycgcoro420_to_bgra(
    y_plane: &[u16],
    y_stride: u32,
    cg_plane: &[u16],
    cg_stride: u32,
    co_plane: &[u16],
    co_stride: u32,
    bgra: &mut [u8],
    bgra_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
) {
    ycgco_r_type_ro_rgbx::<
        { YuvSourceChannels::Bgra as u8 },
        { YuvChromaSample::YUV420 as u8 },
        { YCgCoR::YCgCoRo as usize },
    >(
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
    )
}

/// Convert YCgCo-Ro 422 planar format to RGB format.
///
/// This function takes YCgCo-Ro 422 data with 8-bit precision,
/// and converts it to RGB format with 8-bit per channel precision.
///
/// # Arguments
///
/// * `y_plane` - A slice to load the Y (luminance) plane data.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `cg_plane` - A slice to load the U (chrominance) plane data.
/// * `cg_stride` - The stride (bytes per row) for the U plane.
/// * `co_plane` - A slice to load the V (chrominance) plane data.
/// * `co_stride` - The stride (bytes per row) for the V plane.
/// * `width` - The width of the YUV image.
/// * `height` - The height of the YUV image.
/// * `rgb_data` - A mutable slice to store the converted RGB data.
/// * `range` - The YUV range (limited or full).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGRA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn ycgcoro422_to_rgb(
    y_plane: &[u16],
    y_stride: u32,
    cg_plane: &[u16],
    cg_stride: u32,
    co_plane: &[u16],
    co_stride: u32,
    rgb: &mut [u8],
    rgb_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
) {
    ycgco_r_type_ro_rgbx::<
        { YuvSourceChannels::Rgb as u8 },
        { YuvChromaSample::YUV422 as u8 },
        { YCgCoR::YCgCoRo as usize },
    >(
        y_plane, y_stride, cg_plane, cg_stride, co_plane, co_stride, rgb, rgb_stride, width,
        height, range,
    )
}

/// Convert YCgCo-Ro 422 planar format to RGBA format.
///
/// This function takes YCgCo-Ro 422 data with 8-bit precision,
/// and converts it to RGBA format with 8-bit per channel precision.
///
/// # Arguments
///
/// * `y_plane` - A slice to load the Y (luminance) plane data.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `cg_plane` - A slice to load the U (chrominance) plane data.
/// * `cg_stride` - The stride (bytes per row) for the U plane.
/// * `co_plane` - A slice to load the V (chrominance) plane data.
/// * `co_stride` - The stride (bytes per row) for the V plane.
/// * `width` - The width of the YUV image.
/// * `height` - The height of the YUV image.
/// * `bgra_data` - A mutable slice to store the converted RGBA data.
/// * `range` - The YUV range (limited or full).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGRA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn ycgcoro422_to_rgba(
    y_plane: &[u16],
    y_stride: u32,
    cg_plane: &[u16],
    cg_stride: u32,
    co_plane: &[u16],
    co_stride: u32,
    rgba: &mut [u8],
    rgba_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
) {
    ycgco_r_type_ro_rgbx::<
        { YuvSourceChannels::Rgba as u8 },
        { YuvChromaSample::YUV422 as u8 },
        { YCgCoR::YCgCoRo as usize },
    >(
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
    )
}

/// Convert YCgCo-Ro 422 planar format to BGRA format.
///
/// This function takes YCgCo-Ro 422 data with 8-bit precision,
/// and converts it to BGRA format with 8-bit per channel precision.
///
/// # Arguments
///
/// * `y_plane` - A slice to load the Y (luminance) plane data.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `cg_plane` - A slice to load the U (chrominance) plane data.
/// * `cg_stride` - The stride (bytes per row) for the U plane.
/// * `co_plane` - A slice to load the V (chrominance) plane data.
/// * `co_stride` - The stride (bytes per row) for the V plane.
/// * `width` - The width of the YUV image.
/// * `height` - The height of the YUV image.
/// * `bgra_data` - A mutable slice to store the converted BGRA data.
/// * `range` - The YUV range (limited or full).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGRA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn ycgcoro422_to_bgra(
    y_plane: &[u16],
    y_stride: u32,
    cg_plane: &[u16],
    cg_stride: u32,
    co_plane: &[u16],
    co_stride: u32,
    bgra: &mut [u8],
    bgra_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
) {
    ycgco_r_type_ro_rgbx::<
        { YuvSourceChannels::Bgra as u8 },
        { YuvChromaSample::YUV422 as u8 },
        { YCgCoR::YCgCoRo as usize },
    >(
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
    )
}

/// Convert YCgCo-Ro 444 planar format to RGBA format.
///
/// This function takes YCgCo-Ro 444 data with 8-bit precision,
/// and converts it to RGBA format with 8-bit per channel precision.
///
/// # Arguments
///
/// * `y_plane` - A slice to load the Y (luminance) plane data.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `cg_plane` - A slice to load the U (chrominance) plane data.
/// * `cg_stride` - The stride (bytes per row) for the U plane.
/// * `co_plane` - A slice to load the V (chrominance) plane data.
/// * `co_stride` - The stride (bytes per row) for the V plane.
/// * `width` - The width of the YUV image.
/// * `height` - The height of the YUV image.
/// * `rgba_data` - A mutable slice to store the converted RGBA data.
/// * `range` - The YUV range (limited or full).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGRA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn ycgcoro444_to_rgba(
    y_plane: &[u16],
    y_stride: u32,
    cg_plane: &[u16],
    cg_stride: u32,
    co_plane: &[u16],
    co_stride: u32,
    rgba: &mut [u8],
    rgba_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
) {
    ycgco_r_type_ro_rgbx::<
        { YuvSourceChannels::Rgba as u8 },
        { YuvChromaSample::YUV444 as u8 },
        { YCgCoR::YCgCoRo as usize },
    >(
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
    )
}

/// Convert YCgCo-Ro 444 planar format to BGRA format.
///
/// This function takes YCgCo-Ro 444 data with 8-bit precision,
/// and converts it to BGRA format with 8-bit per channel precision.
///
/// # Arguments
///
/// * `y_plane` - A slice to load the Y (luminance) plane data.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `cg_plane` - A slice to load the U (chrominance) plane data.
/// * `cg_stride` - The stride (bytes per row) for the U plane.
/// * `co_plane` - A slice to load the V (chrominance) plane data.
/// * `co_stride` - The stride (bytes per row) for the V plane.
/// * `width` - The width of the YUV image.
/// * `height` - The height of the YUV image.
/// * `bgra_data` - A mutable slice to store the converted BGRA data.
/// * `range` - The YUV range (limited or full).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGRA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn ycgcoro444_to_bgra(
    y_plane: &[u16],
    y_stride: u32,
    cg_plane: &[u16],
    cg_stride: u32,
    co_plane: &[u16],
    co_stride: u32,
    bgra: &mut [u8],
    bgra_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
) {
    ycgco_r_type_ro_rgbx::<
        { YuvSourceChannels::Bgra as u8 },
        { YuvChromaSample::YUV444 as u8 },
        { YCgCoR::YCgCoRo as usize },
    >(
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
    )
}

/// Convert YCgCo-Ro 444 planar format to RGB format.
///
/// This function takes YCgCo-Ro 444 data with 8-bit precision,
/// and converts it to RGB format with 8-bit per channel precision.
///
/// # Arguments
///
/// * `y_plane` - A slice to load the Y (luminance) plane data.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `cg_plane` - A slice to load the U (chrominance) plane data.
/// * `cg_stride` - The stride (bytes per row) for the U plane.
/// * `co_plane` - A slice to load the V (chrominance) plane data.
/// * `co_stride` - The stride (bytes per row) for the V plane.
/// * `width` - The width of the YUV image.
/// * `height` - The height of the YUV image.
/// * `rgb_data` - A mutable slice to store the converted RGB data.
/// * `range` - The YUV range (limited or full).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGRA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn ycgcoro444_to_rgb(
    y_plane: &[u16],
    y_stride: u32,
    cg_plane: &[u16],
    cg_stride: u32,
    co_plane: &[u16],
    co_stride: u32,
    rgb: &mut [u8],
    rgb_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
) {
    ycgco_r_type_ro_rgbx::<
        { YuvSourceChannels::Rgb as u8 },
        { YuvChromaSample::YUV444 as u8 },
        { YCgCoR::YCgCoRo as usize },
    >(
        y_plane, y_stride, cg_plane, cg_stride, co_plane, co_stride, rgb, rgb_stride, width,
        height, range,
    )
}
