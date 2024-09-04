/*
 * // Copyright (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

use crate::sharpyuv::SharpYuvGammaTransfer;
use crate::yuv_support::*;

fn rgbx_to_sharp_yuv<const ORIGIN_CHANNELS: u8, const SAMPLING: u8>(
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
    sharp_yuv_gamma_transfer: SharpYuvGammaTransfer,
) {
    let chroma_subsampling: YuvChromaSample = SAMPLING.into();
    let src_chans: YuvSourceChannels = ORIGIN_CHANNELS.into();

    let linearize_func = sharp_yuv_gamma_transfer.get_linearize_function();
    let gamma_func = sharp_yuv_gamma_transfer.get_gamma_function();

    let mut linear_map_table = [0u16; 256];
    let mut gamma_map_table = [0u8; u16::MAX as usize + 1];

    let linear_scale = (1. / 255.) as f32;
    let gamma_scale = 1. / u16::MAX as f32;

    for i in 0..256 {
        unsafe {
            let linear = linearize_func(i as f32 * linear_scale);
            *linear_map_table.get_unchecked_mut(i as usize) =
                (linear * u16::MAX as f32).round() as u16;
        }
    }

    for i in 0..(u16::MAX as usize + 1) {
        unsafe {
            let gamma = gamma_func(i as f32 * gamma_scale);
            *gamma_map_table.get_unchecked_mut(i) = (gamma * 255.).round() as u8;
        }
    }

    // Always using 3 Channels ( RGB etc. ) layout since we do not need a alpha channel
    let mut rgb_layout: Vec<u16> = vec![0u16; width as usize * height as usize * 3];

    let rgb_layout_stride_len = width as usize * 3;

    for y in 0..height as usize {
        unsafe {
            let rgb_layout_cast = rgb_layout.get_unchecked_mut((y * rgb_layout_stride_len)..);
            let src_layout = rgba.get_unchecked((y * rgba_stride as usize)..);
            for x in 0..width as usize {
                let dst_layout = rgb_layout_cast.get_unchecked_mut((x * 3)..);
                let src = src_layout.get_unchecked((x * src_chans.get_channels_count())..);
                *dst_layout.get_unchecked_mut(0) =
                    *linear_map_table.get_unchecked((*src.get_unchecked(0)) as usize);
                *dst_layout.get_unchecked_mut(1) =
                    *linear_map_table.get_unchecked((*src.get_unchecked(1)) as usize);
                *dst_layout.get_unchecked_mut(2) =
                    *linear_map_table.get_unchecked((*src.get_unchecked(2)) as usize);
            }
        }
    }

    let channels = src_chans.get_channels_count();
    let range = get_yuv_range(8, range);
    let kr_kb = get_kr_kb(matrix);
    let max_range_p8 = (1u32 << 8u32) - 1u32;
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

    let i_bias_y = range.bias_y as i32;
    let i_cap_y = range.range_y as i32 + i_bias_y;
    let i_cap_uv = i_bias_y + range.range_uv as i32;

    let iterator_step = match chroma_subsampling {
        YuvChromaSample::YUV420 => 2usize,
        YuvChromaSample::YUV422 => 2usize,
        YuvChromaSample::YUV444 => 1usize,
    };

    let mut y_offset = 0usize;
    let mut u_offset = 0usize;
    let mut v_offset = 0usize;
    let mut rgba_offset = 0usize;
    let mut rgb_layout_offset = 0usize;

    for y in 0..height as usize {
        let mut _cx = 0usize;
        let mut _ux = 0usize;

        for x in (_cx..width as usize).step_by(iterator_step) {
            unsafe {
                let px = x * channels;
                let rgba_shift = rgba_offset + px;
                let src = rgba.get_unchecked(rgba_shift..);
                let r = *src.get_unchecked(src_chans.get_r_channel_offset()) as i32;
                let g = *src.get_unchecked(src_chans.get_g_channel_offset()) as i32;
                let b = *src.get_unchecked(src_chans.get_b_channel_offset()) as i32;

                let rgb_layout_src = rgb_layout.get_unchecked((rgb_layout_offset + x * 3)..);

                let sharp_r_c = *rgb_layout_src.get_unchecked(src_chans.get_r_channel_offset());
                let sharp_g_c = *rgb_layout_src.get_unchecked(src_chans.get_g_channel_offset());
                let sharp_b_c = *rgb_layout_src.get_unchecked(src_chans.get_b_channel_offset());

                let mut sharp_r_next = sharp_r_c;
                let mut sharp_g_next = sharp_g_c;
                let mut sharp_b_next = sharp_b_c;

                let y_0 = (r * transform.yr + g * transform.yg + b * transform.yb + bias_y) >> 8;

                *y_plane.get_unchecked_mut(y_offset + x) = y_0.clamp(i_bias_y, i_cap_y) as u8;

                match chroma_subsampling {
                    YuvChromaSample::YUV420 | YuvChromaSample::YUV422 => {
                        if x + 1 < width as usize {
                            let next_px = (x + 1) * channels;
                            let rgba_shift = rgba_offset + next_px;
                            let src = rgba.get_unchecked(rgba_shift..);
                            let r = *src.get_unchecked(src_chans.get_r_channel_offset()) as i32;
                            let g = *src.get_unchecked(src_chans.get_g_channel_offset()) as i32;
                            let b = *src.get_unchecked(src_chans.get_b_channel_offset()) as i32;

                            let rgb_layout_src_next = rgb_layout_src.get_unchecked(3..);

                            sharp_r_next = *rgb_layout_src_next
                                .get_unchecked(src_chans.get_r_channel_offset());
                            sharp_g_next = *rgb_layout_src_next
                                .get_unchecked(src_chans.get_g_channel_offset());
                            sharp_b_next = *rgb_layout_src_next
                                .get_unchecked(src_chans.get_b_channel_offset());

                            let y_1 =
                                (r * transform.yr + g * transform.yg + b * transform.yb + bias_y)
                                    >> 8;
                            *y_plane.get_unchecked_mut(y_offset + x + 1) =
                                y_1.clamp(i_bias_y, i_cap_y) as u8;
                        }
                    }
                    _ => {}
                }

                let mut sharp_r_c_next_row = sharp_r_c;
                let mut sharp_g_c_next_row = sharp_g_c;
                let mut sharp_b_c_next_row = sharp_b_c;

                let mut sharp_r_next_row = sharp_r_c;
                let mut sharp_g_next_row = sharp_g_c;
                let mut sharp_b_next_row = sharp_b_c;

                if y + 1 < height as usize {
                    let mut rgb_layout_src_next_row =
                        rgb_layout_src.get_unchecked(rgb_layout_stride_len..);
                    sharp_r_c_next_row =
                        *rgb_layout_src_next_row.get_unchecked(src_chans.get_r_channel_offset());
                    sharp_g_c_next_row =
                        *rgb_layout_src_next_row.get_unchecked(src_chans.get_g_channel_offset());
                    sharp_b_c_next_row =
                        *rgb_layout_src_next_row.get_unchecked(src_chans.get_b_channel_offset());

                    if x + 1 < width as usize {
                        rgb_layout_src_next_row = rgb_layout_src_next_row.get_unchecked(3..);
                        sharp_r_next_row = *rgb_layout_src_next_row
                            .get_unchecked(src_chans.get_r_channel_offset());
                        sharp_g_next_row = *rgb_layout_src_next_row
                            .get_unchecked(src_chans.get_g_channel_offset());
                        sharp_b_next_row = *rgb_layout_src_next_row
                            .get_unchecked(src_chans.get_b_channel_offset());
                    }
                }

                let interpolated_r = ((sharp_r_c as i32 * 9
                    + sharp_r_next as i32 * 3
                    + sharp_r_c_next_row as i32 * 3
                    + sharp_r_next_row as i32)
                    >> 4) as u16;
                let interpolated_g = ((sharp_g_c as i32 * 9
                    + sharp_g_next as i32 * 3
                    + sharp_g_c_next_row as i32 * 3
                    + sharp_g_next_row as i32)
                    >> 4) as u16;
                let interpolated_b = ((sharp_b_c as i32 * 9
                    + sharp_b_next as i32 * 3
                    + sharp_b_c_next_row as i32 * 3
                    + sharp_b_next_row as i32)
                    >> 4) as u16;

                let corrected_r = *gamma_map_table.get_unchecked(interpolated_r as usize) as i32;
                let corrected_g = *gamma_map_table.get_unchecked(interpolated_g as usize) as i32;
                let corrected_b = *gamma_map_table.get_unchecked(interpolated_b as usize) as i32;

                let cb = (corrected_r * transform.cb_r
                    + corrected_g * transform.cb_g
                    + corrected_b * transform.cb_b
                    + bias_uv)
                    >> 8;
                let cr = (corrected_r * transform.cr_r
                    + corrected_g * transform.cr_g
                    + corrected_b * transform.cr_b
                    + bias_uv)
                    >> 8;

                let u_pos = match chroma_subsampling {
                    YuvChromaSample::YUV420 | YuvChromaSample::YUV422 => u_offset + _ux,
                    YuvChromaSample::YUV444 => u_offset + _ux,
                };
                *u_plane.get_unchecked_mut(u_pos) = cb.clamp(i_bias_y, i_cap_uv) as u8;
                let v_pos = match chroma_subsampling {
                    YuvChromaSample::YUV420 | YuvChromaSample::YUV422 => v_offset + _ux,
                    YuvChromaSample::YUV444 => v_offset + _ux,
                };
                *v_plane.get_unchecked_mut(v_pos) = cr.clamp(i_bias_y, i_cap_uv) as u8;
            }
            _ux += 1;
        }

        y_offset += y_stride as usize;
        rgba_offset += rgba_stride as usize;
        rgb_layout_offset += rgb_layout_stride_len;
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

/// Convert RGB image data to YUV 422 planar format using bi-linear interpolation and gamma correction ( sharp YUV algorithm ).
///
/// This function performs RGB to YUV conversion using bi-linear interpolation and gamma correction and stores the result in YUV422 planar format using bi-linear interpolation and gamma correction,
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
pub fn rgb_to_sharp_yuv422(
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
    gamma_transfer: SharpYuvGammaTransfer,
) {
    rgbx_to_sharp_yuv::<{ YuvSourceChannels::Rgb as u8 }, { YuvChromaSample::YUV422 as u8 }>(
        y_plane,
        y_stride,
        u_plane,
        u_stride,
        v_plane,
        v_stride,
        rgb,
        rgb_stride,
        width,
        height,
        range,
        matrix,
        gamma_transfer,
    );
}

/// Convert BGR image data to YUV 422 planar format using bi-linear interpolation and gamma correction ( sharp YUV algorithm ).
///
/// This function performs BGR to YUV conversion using bi-linear interpolation and gamma correction and stores the result in YUV422 planar format using bi-linear interpolation and gamma correction,
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
/// * `bgr` - The input BGR image data slice.
/// * `bgr_stride` - The stride (bytes per row) for the BGR image data.
/// * `width` - The width of the image in pixels.
/// * `height` - The height of the image in pixels.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGR data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn bgr_to_sharp_yuv422(
    y_plane: &mut [u8],
    y_stride: u32,
    u_plane: &mut [u8],
    u_stride: u32,
    v_plane: &mut [u8],
    v_stride: u32,
    bgr: &[u8],
    bgr_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    gamma_transfer: SharpYuvGammaTransfer,
) {
    rgbx_to_sharp_yuv::<{ YuvSourceChannels::Bgr as u8 }, { YuvChromaSample::YUV422 as u8 }>(
        y_plane,
        y_stride,
        u_plane,
        u_stride,
        v_plane,
        v_stride,
        bgr,
        bgr_stride,
        width,
        height,
        range,
        matrix,
        gamma_transfer,
    );
}

/// Convert RGBA image data to YUV 422 planar format using bi-linear interpolation and gamma correction ( sharp YUV algorithm ).
///
/// This function performs RGBA to YUV conversion using bi-linear interpolation and gamma correction and stores the result in YUV422 planar format using bi-linear interpolation and gamma correction,
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
pub fn rgba_to_sharp_yuv422(
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
    gamma_transfer: SharpYuvGammaTransfer,
) {
    rgbx_to_sharp_yuv::<{ YuvSourceChannels::Rgba as u8 }, { YuvChromaSample::YUV422 as u8 }>(
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
        gamma_transfer,
    );
}

/// Convert BGRA image data to YUV 422 planar format using bi-linear interpolation and gamma correction ( sharp YUV algorithm ).
///
/// This function performs BGRA to YUV conversion using bi-linear interpolation and gamma correction and stores the result in YUV422 planar format using bi-linear interpolation and gamma correction,
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
pub fn bgra_to_sharp_yuv422(
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
    gamma_transfer: SharpYuvGammaTransfer,
) {
    rgbx_to_sharp_yuv::<{ YuvSourceChannels::Bgra as u8 }, { YuvChromaSample::YUV422 as u8 }>(
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
        gamma_transfer,
    );
}

/// Convert RGB image data to YUV 420 planar format using bi-linear interpolation and gamma correction ( sharp YUV algorithm ).
///
/// This function performs RGB to YUV conversion using bi-linear interpolation and gamma correction and stores the result in YUV420 planar format using bi-linear interpolation and gamma correction,
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
pub fn rgb_to_sharp_yuv420(
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
    gamma_transfer: SharpYuvGammaTransfer,
) {
    rgbx_to_sharp_yuv::<{ YuvSourceChannels::Rgb as u8 }, { YuvChromaSample::YUV420 as u8 }>(
        y_plane,
        y_stride,
        u_plane,
        u_stride,
        v_plane,
        v_stride,
        rgb,
        rgb_stride,
        width,
        height,
        range,
        matrix,
        gamma_transfer,
    );
}

/// Convert BGR image data to YUV 420 planar format using bi-linear interpolation and gamma correction ( sharp YUV algorithm ).
///
/// This function performs BGR to YUV conversion using bi-linear interpolation and gamma correction and stores the result in YUV420 planar format using bi-linear interpolation and gamma correction,
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
/// * `bgr` - The input BGR image data slice.
/// * `bgr_stride` - The stride (bytes per row) for the BGR image data.
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
pub fn bgr_to_sharp_yuv420(
    y_plane: &mut [u8],
    y_stride: u32,
    u_plane: &mut [u8],
    u_stride: u32,
    v_plane: &mut [u8],
    v_stride: u32,
    bgr: &[u8],
    bgr_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    gamma_transfer: SharpYuvGammaTransfer,
) {
    rgbx_to_sharp_yuv::<{ YuvSourceChannels::Bgr as u8 }, { YuvChromaSample::YUV420 as u8 }>(
        y_plane,
        y_stride,
        u_plane,
        u_stride,
        v_plane,
        v_stride,
        bgr,
        bgr_stride,
        width,
        height,
        range,
        matrix,
        gamma_transfer,
    );
}

/// Convert RGBA image data to YUV 420 planar format using bi-linear interpolation and gamma correction ( sharp YUV algorithm ).
///
/// This function performs RGBA to YUV conversion using bi-linear interpolation and gamma correction and stores the result in YUV420 planar format using bi-linear interpolation and gamma correction,
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
pub fn rgba_to_sharp_yuv420(
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
    gamma_transfer: SharpYuvGammaTransfer,
) {
    rgbx_to_sharp_yuv::<{ YuvSourceChannels::Rgba as u8 }, { YuvChromaSample::YUV420 as u8 }>(
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
        gamma_transfer,
    );
}

/// Convert BGRA image data to YUV 420 planar format using bi-linear interpolation and gamma correction ( sharp YUV algorithm ).
///
/// This function performs BGRA to YUV conversion using bi-linear interpolation and gamma correction and stores the result in YUV420 planar format using bi-linear interpolation and gamma correction,
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
pub fn bgra_to_sharp_yuv420(
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
    gamma_transfer: SharpYuvGammaTransfer,
) {
    rgbx_to_sharp_yuv::<{ YuvSourceChannels::Bgra as u8 }, { YuvChromaSample::YUV420 as u8 }>(
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
        gamma_transfer,
    );
}
