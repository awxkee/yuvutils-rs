/*
 * // Copyright (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */
use crate::yuv_support::{
    get_inverse_transform, get_kr_kb, get_yuv_range, YuvSourceChannels,
    Yuy2Description,
};
use crate::{YuvRange, YuvStandardMatrix};

fn yuy2_to_rgb_impl<const DESTINATION_CHANNELS: u8, const YUY2_SOURCE: usize>(
    yuy2_store: &[u8],
    yuy2_stride: u32,
    rgb_store: &mut [u8],
    rgb_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) {
    let yuy2_target: Yuy2Description = YUY2_SOURCE.into();

    let dst_chans: YuvSourceChannels = DESTINATION_CHANNELS.into();
    let channels = dst_chans.get_channels_count();
    let range = get_yuv_range(8, range);
    let kr_kb = get_kr_kb(matrix);
    let transform = get_inverse_transform(255, range.range_y, range.range_uv, kr_kb.kr, kr_kb.kb);
    let inverse_transform = transform.to_integers(6);
    let cr_coef = inverse_transform.cr_coef;
    let cb_coef = inverse_transform.cb_coef;
    let y_coef = inverse_transform.y_coef;
    let g_coef_1 = inverse_transform.g_coeff_1;
    let g_coef_2 = inverse_transform.g_coeff_2;
    let bias_y = range.bias_y as i32;
    let bias_uv = range.bias_uv as i32;

    let mut rgb_offset = 0usize;
    let mut yuy_offset = 0usize;

    for _ in 0..height as usize {
        let mut _cx = 0usize;
        let mut _yuy2_x = 0usize;

        for x in _yuy2_x..width.saturating_sub(1) as usize / 2 {
            let rgb_pos = rgb_offset + _cx * channels;
            let dst_offset = yuy_offset + x * 4;

            let yuy2_plane_shifted = unsafe { yuy2_store.get_unchecked(dst_offset..) };

            let first_y =
                unsafe { *yuy2_plane_shifted.get_unchecked(yuy2_target.get_first_y_position()) };
            let second_y =
                unsafe { *yuy2_plane_shifted.get_unchecked(yuy2_target.get_second_y_position()) };
            let u_value =
                unsafe { *yuy2_plane_shifted.get_unchecked(yuy2_target.get_u_position()) };
            let v_value =
                unsafe { *yuy2_plane_shifted.get_unchecked(yuy2_target.get_v_position()) };

            let cb = u_value as i32 - bias_uv;
            let cr = v_value as i32 - bias_uv;
            let f_y = (first_y as i32 - bias_y) * y_coef;
            let s_y = (second_y as i32 - bias_y) * y_coef;

            unsafe {
                let dst0 = rgb_store.get_unchecked_mut(rgb_pos..);
                let r0 = ((f_y + cr_coef * cr) >> 6).clamp(0, 255);
                let b0 = ((f_y + cb_coef * cb) >> 6).clamp(0, 255);
                let g0 = ((f_y - g_coef_1 * cr - g_coef_2 * cb) >> 6).clamp(0, 255);
                *dst0.get_unchecked_mut(dst_chans.get_r_channel_offset()) = r0 as u8;
                *dst0.get_unchecked_mut(dst_chans.get_g_channel_offset()) = g0 as u8;
                *dst0.get_unchecked_mut(dst_chans.get_b_channel_offset()) = b0 as u8;
                if dst_chans.has_alpha() {
                    *dst0.get_unchecked_mut(dst_chans.get_a_channel_offset()) = 255;
                }

                let dst1 = rgb_store.get_unchecked_mut((rgb_pos + channels)..);

                let r1 = ((s_y + cr_coef * cr) >> 6).clamp(0, 255);
                let b1 = ((s_y + cb_coef * cb) >> 6).clamp(0, 255);
                let g1 = ((s_y - g_coef_1 * cr - g_coef_2 * cb) >> 6).clamp(0, 255);
                *dst1.get_unchecked_mut(dst_chans.get_r_channel_offset()) = r1 as u8;
                *dst1.get_unchecked_mut(dst_chans.get_g_channel_offset()) = g1 as u8;
                *dst1.get_unchecked_mut(dst_chans.get_b_channel_offset()) = b1 as u8;
                if dst_chans.has_alpha() {
                    *dst1.get_unchecked_mut(dst_chans.get_a_channel_offset()) = 255;
                }
            }

            _cx += 2;
        }

        if width & 1 == 1 {
            let rgb_pos = rgb_offset + _cx * channels;
            let dst_offset = yuy_offset + ((width as usize - 1) / 2) * 4;

            let yuy2_plane_shifted = unsafe { yuy2_store.get_unchecked(dst_offset..) };

            let first_y =
                unsafe { *yuy2_plane_shifted.get_unchecked(yuy2_target.get_first_y_position()) };
            let u_value =
                unsafe { *yuy2_plane_shifted.get_unchecked(yuy2_target.get_u_position()) };
            let v_value =
                unsafe { *yuy2_plane_shifted.get_unchecked(yuy2_target.get_v_position()) };

            let cb = u_value as i32 - bias_uv;
            let cr = v_value as i32 - bias_uv;
            let f_y = (first_y as i32 - bias_y) * y_coef;

            unsafe {
                let dst0 = rgb_store.get_unchecked_mut(rgb_pos..);
                let r0 = ((f_y + cr_coef * cr) >> 6).clamp(0, 255);
                let b0 = ((f_y + cb_coef * cb) >> 6).clamp(0, 255);
                let g0 = ((f_y - g_coef_1 * cr - g_coef_2 * cb) >> 6).clamp(0, 255);
                *dst0.get_unchecked_mut(dst_chans.get_r_channel_offset()) = r0 as u8;
                *dst0.get_unchecked_mut(dst_chans.get_g_channel_offset()) = g0 as u8;
                *dst0.get_unchecked_mut(dst_chans.get_b_channel_offset()) = b0 as u8;
                if dst_chans.has_alpha() {
                    *dst0.get_unchecked_mut(dst_chans.get_a_channel_offset()) = 255;
                }
            }
        }

        rgb_offset += rgb_stride as usize;
        yuy_offset += yuy2_stride as usize;
    }
}

/// Convert YUYV format to RGB image.
///
/// This function takes YUYV (4:2:2) format data with 8-bit precision,
/// and converts it to RGB with 8-bit per channel precision.
///
/// # Arguments
///
/// * `yuy2_store` - A slice to store the converted YUYV data.
/// * `yuy2_stride` - The stride (bytes per row) for the YUYV plane.
/// * `rgb` - A mutable slice to store the converted RGB data.
/// * `rgb_stride` - The stride (bytes per row) for the RGB image data.
/// * `width` - The width of the image.
/// * `height` - The height of the image.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input YUYV data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuyv422_to_rgb(
    yuy2_store: &[u8],
    yuy2_stride: u32,
    rgb: &mut [u8],
    rgb_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) {
    yuy2_to_rgb_impl::<{ YuvSourceChannels::Rgb as u8 }, { Yuy2Description::YUYV as usize }>(
        yuy2_store,
        yuy2_stride,
        rgb,
        rgb_stride,
        width,
        height,
        range,
        matrix,
    );
}

/// Convert YUYV format to RGBA image.
///
/// This function takes YUYV (4:2:2) format data with 8-bit precision,
/// and converts it to RGBA with 8-bit per channel precision.
///
/// # Arguments
///
/// * `yuy2_store` - A slice to store the converted YUYV data.
/// * `yuy2_stride` - The stride (bytes per row) for the YUYV plane.
/// * `rgba` - A mutable slice to store the converted RGBA data.
/// * `rgba_stride` - The stride (bytes per row) for the RGBA image data.
/// * `width` - The width of the image.
/// * `height` - The height of the image.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input YUYV data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuyv422_to_rgba(
    yuy2_store: &[u8],
    yuy2_stride: u32,
    rgba: &mut [u8],
    rgba_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) {
    yuy2_to_rgb_impl::<{ YuvSourceChannels::Rgba as u8 }, { Yuy2Description::YUYV as usize }>(
        yuy2_store,
        yuy2_stride,
        rgba,
        rgba_stride,
        width,
        height,
        range,
        matrix,
    );
}

/// Convert YUYV format to BGR image.
///
/// This function takes YUYV (4:2:2) format data with 8-bit precision,
/// and converts it to BGR with 8-bit per channel precision.
///
/// # Arguments
///
/// * `yuy2_store` - A slice to store the converted YUYV data.
/// * `yuy2_stride` - The stride (bytes per row) for the YUYV plane.
/// * `bgr` - A mutable slice to store the converted BGR data.
/// * `bgr_stride` - The stride (bytes per row) for the BGR image data.
/// * `width` - The width of the image.
/// * `height` - The height of the image.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input YUYV data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuyv422_to_bgr(
    yuy2_store: &[u8],
    yuy2_stride: u32,
    bgr: &mut [u8],
    bgr_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) {
    yuy2_to_rgb_impl::<{ YuvSourceChannels::Bgr as u8 }, { Yuy2Description::YUYV as usize }>(
        yuy2_store,
        yuy2_stride,
        bgr,
        bgr_stride,
        width,
        height,
        range,
        matrix,
    );
}

/// Convert YUYV format to BGR image.
///
/// This function takes YUYV (4:2:2) format data with 8-bit precision,
/// and converts it to BGRA with 8-bit per channel precision.
///
/// # Arguments
///
/// * `yuy2_store` - A slice to store the converted YUYV data.
/// * `yuy2_stride` - The stride (bytes per row) for the YUYV plane.
/// * `bgra` - A mutable slice to store the converted BGRA data.
/// * `bgra_stride` - The stride (bytes per row) for the BGRA image data.
/// * `width` - The width of the image.
/// * `height` - The height of the image.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input YUYV data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuyv422_to_bgra(
    yuy2_store: &[u8],
    yuy2_stride: u32,
    bgra: &mut [u8],
    bgra_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) {
    yuy2_to_rgb_impl::<{ YuvSourceChannels::Bgra as u8 }, { Yuy2Description::YUYV as usize }>(
        yuy2_store,
        yuy2_stride,
        bgra,
        bgra_stride,
        width,
        height,
        range,
        matrix,
    );
}

/// Convert UYVY format to RGB image.
///
/// This function takes UYVY (4:2:2) format data with 8-bit precision,
/// and converts it to RGB with 8-bit per channel precision.
///
/// # Arguments
///
/// * `yuy2_store` - A slice to store the converted UYVY data.
/// * `yuy2_stride` - The stride (bytes per row) for the UYVY plane.
/// * `rgb` - A mutable slice to store the converted RGB data.
/// * `rgb_stride` - The stride (bytes per row) for the RGB image data.
/// * `width` - The width of the image.
/// * `height` - The height of the image.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input UYVY data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn uyvy422_to_rgb(
    yuy2_store: &[u8],
    yuy2_stride: u32,
    rgb: &mut [u8],
    rgb_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) {
    yuy2_to_rgb_impl::<{ YuvSourceChannels::Rgb as u8 }, { Yuy2Description::UYVY as usize }>(
        yuy2_store,
        yuy2_stride,
        rgb,
        rgb_stride,
        width,
        height,
        range,
        matrix,
    );
}

/// Convert UYVY format to RGBA image.
///
/// This function takes UYVY (4:2:2) format data with 8-bit precision,
/// and converts it to RGBA with 8-bit per channel precision.
///
/// # Arguments
///
/// * `yuy2_store` - A slice to store the converted UYVY data.
/// * `yuy2_stride` - The stride (bytes per row) for the UYVY plane.
/// * `rgba` - A mutable slice to store the converted RGBA data.
/// * `rgba_stride` - The stride (bytes per row) for the RGBA image data.
/// * `width` - The width of the image.
/// * `height` - The height of the image.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input UYVY data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn uyvy422_to_rgba(
    yuy2_store: &[u8],
    yuy2_stride: u32,
    rgba: &mut [u8],
    rgba_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) {
    yuy2_to_rgb_impl::<{ YuvSourceChannels::Rgba as u8 }, { Yuy2Description::UYVY as usize }>(
        yuy2_store,
        yuy2_stride,
        rgba,
        rgba_stride,
        width,
        height,
        range,
        matrix,
    );
}

/// Convert UYVY format to BGR image.
///
/// This function takes UYVY (4:2:2) format data with 8-bit precision,
/// and converts it to BGR with 8-bit per channel precision.
///
/// # Arguments
///
/// * `yuy2_store` - A slice to store the converted UYVY data.
/// * `yuy2_stride` - The stride (bytes per row) for the UYVY plane.
/// * `bgr` - A mutable slice to store the converted BGR data.
/// * `bgr_stride` - The stride (bytes per row) for the BGR image data.
/// * `width` - The width of the image.
/// * `height` - The height of the image.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input UYVY data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn uyvy422_to_bgr(
    yuy2_store: &[u8],
    yuy2_stride: u32,
    bgr: &mut [u8],
    bgr_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) {
    yuy2_to_rgb_impl::<{ YuvSourceChannels::Bgr as u8 }, { Yuy2Description::UYVY as usize }>(
        yuy2_store,
        yuy2_stride,
        bgr,
        bgr_stride,
        width,
        height,
        range,
        matrix,
    );
}

/// Convert UYVY format to BGRA image.
///
/// This function takes UYVY (4:2:2) format data with 8-bit precision,
/// and converts it to BGRA with 8-bit per channel precision.
///
/// # Arguments
///
/// * `yuy2_store` - A slice to store the converted UYVY data.
/// * `yuy2_stride` - The stride (bytes per row) for the UYVY plane.
/// * `bgra` - A mutable slice to store the converted BGRA data.
/// * `bgra_stride` - The stride (bytes per row) for the BGRA image data.
/// * `width` - The width of the image.
/// * `height` - The height of the image.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input UYVY data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn uyvy422_to_bgra(
    yuy2_store: &[u8],
    yuy2_stride: u32,
    bgra: &mut [u8],
    bgra_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) {
    yuy2_to_rgb_impl::<{ YuvSourceChannels::Bgra as u8 }, { Yuy2Description::UYVY as usize }>(
        yuy2_store,
        yuy2_stride,
        bgra,
        bgra_stride,
        width,
        height,
        range,
        matrix,
    );
}


/// Convert YVYU format to RGB image.
///
/// This function takes YVYU (4:2:2) format data with 8-bit precision,
/// and converts it to RGB with 8-bit per channel precision.
///
/// # Arguments
///
/// * `yuy2_store` - A slice to store the converted YVYU data.
/// * `yuy2_stride` - The stride (bytes per row) for the YVYU plane.
/// * `rgb` - A mutable slice to store the converted RGB data.
/// * `rgb_stride` - The stride (bytes per row) for the RGB image data.
/// * `width` - The width of the image.
/// * `height` - The height of the image.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input YVYU data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yvyu422_to_rgb(
    yuy2_store: &[u8],
    yuy2_stride: u32,
    rgb: &mut [u8],
    rgb_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) {
    yuy2_to_rgb_impl::<{ YuvSourceChannels::Rgb as u8 }, { Yuy2Description::YVYU as usize }>(
        yuy2_store,
        yuy2_stride,
        rgb,
        rgb_stride,
        width,
        height,
        range,
        matrix,
    );
}

/// Convert YVYU format to RGBA image.
///
/// This function takes YVYU (4:2:2) format data with 8-bit precision,
/// and converts it to RGBA with 8-bit per channel precision.
///
/// # Arguments
///
/// * `yuy2_store` - A slice to store the converted YVYU data.
/// * `yuy2_stride` - The stride (bytes per row) for the YVYU plane.
/// * `rgba` - A mutable slice to store the converted RGBA data.
/// * `rgba_stride` - The stride (bytes per row) for the RGBA image data.
/// * `width` - The width of the image.
/// * `height` - The height of the image.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input YVYU data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yvyu422_to_rgba(
    yuy2_store: &[u8],
    yuy2_stride: u32,
    rgba: &mut [u8],
    rgba_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) {
    yuy2_to_rgb_impl::<{ YuvSourceChannels::Rgba as u8 }, { Yuy2Description::YVYU as usize }>(
        yuy2_store,
        yuy2_stride,
        rgba,
        rgba_stride,
        width,
        height,
        range,
        matrix,
    );
}

/// Convert YVYU format to BGR image.
///
/// This function takes YVYU (4:2:2) format data with 8-bit precision,
/// and converts it to BGR with 8-bit per channel precision.
///
/// # Arguments
///
/// * `yuy2_store` - A slice to store the converted YVYU data.
/// * `yuy2_stride` - The stride (bytes per row) for the YVYU plane.
/// * `bgr` - A mutable slice to store the converted BGR data.
/// * `bgr_stride` - The stride (bytes per row) for the BGR image data.
/// * `width` - The width of the image.
/// * `height` - The height of the image.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input YVYU data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yvyu422_to_bgr(
    yuy2_store: &[u8],
    yuy2_stride: u32,
    bgr: &mut [u8],
    bgr_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) {
    yuy2_to_rgb_impl::<{ YuvSourceChannels::Bgr as u8 }, { Yuy2Description::YVYU as usize }>(
        yuy2_store,
        yuy2_stride,
        bgr,
        bgr_stride,
        width,
        height,
        range,
        matrix,
    );
}

/// Convert YVYU format to BGRA image.
///
/// This function takes YVYU (4:2:2) format data with 8-bit precision,
/// and converts it to BGRA with 8-bit per channel precision.
///
/// # Arguments
///
/// * `yuy2_store` - A slice to store the converted YVYU data.
/// * `yuy2_stride` - The stride (bytes per row) for the YVYU plane.
/// * `bgra` - A mutable slice to store the converted BGRA data.
/// * `bgra_stride` - The stride (bytes per row) for the BGRA image data.
/// * `width` - The width of the image.
/// * `height` - The height of the image.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input YVYU data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yvyu422_to_bgra(
    yuy2_store: &[u8],
    yuy2_stride: u32,
    bgra: &mut [u8],
    bgra_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) {
    yuy2_to_rgb_impl::<{ YuvSourceChannels::Bgra as u8 }, { Yuy2Description::YVYU as usize }>(
        yuy2_store,
        yuy2_stride,
        bgra,
        bgra_stride,
        width,
        height,
        range,
        matrix,
    );
}

/// Convert VYUY format to RGB image.
///
/// This function takes VYUY (4:2:2) format data with 8-bit precision,
/// and converts it to RGB with 8-bit per channel precision.
///
/// # Arguments
///
/// * `yuy2_store` - A slice to store the converted VYUY data.
/// * `yuy2_stride` - The stride (bytes per row) for the VYUY plane.
/// * `rgb` - A mutable slice to store the converted RGB data.
/// * `rgb_stride` - The stride (bytes per row) for the RGB image data.
/// * `width` - The width of the image.
/// * `height` - The height of the image.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input VYUY data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn vyuy422_to_rgb(
    yuy2_store: &[u8],
    yuy2_stride: u32,
    rgb: &mut [u8],
    rgb_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) {
    yuy2_to_rgb_impl::<{ YuvSourceChannels::Rgb as u8 }, { Yuy2Description::VYUY as usize }>(
        yuy2_store,
        yuy2_stride,
        rgb,
        rgb_stride,
        width,
        height,
        range,
        matrix,
    );
}


/// Convert VYUY format to RGBA image.
///
/// This function takes VYUY (4:2:2) format data with 8-bit precision,
/// and converts it to RGBA with 8-bit per channel precision.
///
/// # Arguments
///
/// * `yuy2_store` - A slice to store the converted VYUY data.
/// * `yuy2_stride` - The stride (bytes per row) for the VYUY plane.
/// * `rgba` - A mutable slice to store the converted RGBA data.
/// * `rgba_stride` - The stride (bytes per row) for the RGBA image data.
/// * `width` - The width of the image.
/// * `height` - The height of the image.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input VYUY data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn vyuy422_to_rgba(
    yuy2_store: &[u8],
    yuy2_stride: u32,
    rgba: &mut [u8],
    rgba_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) {
    yuy2_to_rgb_impl::<{ YuvSourceChannels::Rgba as u8 }, { Yuy2Description::VYUY as usize }>(
        yuy2_store,
        yuy2_stride,
        rgba,
        rgba_stride,
        width,
        height,
        range,
        matrix,
    );
}

/// Convert VYUY format to BGR image.
///
/// This function takes VYUY (4:2:2) format data with 8-bit precision,
/// and converts it to BGR with 8-bit per channel precision.
///
/// # Arguments
///
/// * `yuy2_store` - A slice to store the converted VYUY data.
/// * `yuy2_stride` - The stride (bytes per row) for the VYUY plane.
/// * `bgr` - A mutable slice to store the converted BGR data.
/// * `bgr_stride` - The stride (bytes per row) for the BGR image data.
/// * `width` - The width of the image.
/// * `height` - The height of the image.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input VYUY data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn vyuy422_to_bgr(
    yuy2_store: &[u8],
    yuy2_stride: u32,
    bgr: &mut [u8],
    bgr_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) {
    yuy2_to_rgb_impl::<{ YuvSourceChannels::Bgr as u8 }, { Yuy2Description::VYUY as usize }>(
        yuy2_store,
        yuy2_stride,
        bgr,
        bgr_stride,
        width,
        height,
        range,
        matrix,
    );
}

/// Convert VYUY format to BGRA image.
///
/// This function takes VYUY (4:2:2) format data with 8-bit precision,
/// and converts it to BGRA with 8-bit per channel precision.
///
/// # Arguments
///
/// * `yuy2_store` - A slice to store the converted VYUY data.
/// * `yuy2_stride` - The stride (bytes per row) for the VYUY plane.
/// * `bgra` - A mutable slice to store the converted BGRA data.
/// * `bgra_stride` - The stride (bytes per row) for the BGRA image data.
/// * `width` - The width of the image.
/// * `height` - The height of the image.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input VYUY data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn vyuy422_to_bgra(
    yuy2_store: &[u8],
    yuy2_stride: u32,
    bgra: &mut [u8],
    bgra_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) {
    yuy2_to_rgb_impl::<{ YuvSourceChannels::Bgra as u8 }, { Yuy2Description::VYUY as usize }>(
        yuy2_store,
        yuy2_stride,
        bgra,
        bgra_stride,
        width,
        height,
        range,
        matrix,
    );
}
