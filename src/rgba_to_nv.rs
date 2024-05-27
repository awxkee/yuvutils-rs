/*
 * // Copyright (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use std::arch::aarch64::*;

use crate::yuv_support::*;

fn rgbx_to_nv<const ORIGIN_CHANNELS: u8, const UV_ORDER: u8, const SAMPLING: u8>(
    y_plane: &mut [u8],
    y_stride: u32,
    uv_plane: &mut [u8],
    uv_stride: u32,
    rgba: &[u8],
    rgba_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) {
    let order: YuvNVOrder = UV_ORDER.into();
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
    let precision_scale = (1i32 << 8i32) as f32;
    let bias_y = ((range.bias_y as f32 + 0.5f32) * precision_scale) as i32;
    let bias_uv = ((range.bias_uv as f32 + 0.5f32) * precision_scale) as i32;

    let iterator_step = match chroma_subsampling {
        YuvChromaSample::YUV420 => 2usize,
        YuvChromaSample::YUV422 => 2usize,
        YuvChromaSample::YUV444 => 1usize,
    };

    let mut y_offset = 0usize;
    let mut uv_offset = 0usize;
    let mut rgba_offset = 0usize;

    for y in 0..height as usize {
        #[allow(unused_variables)]
        #[allow(unused_mut)]
        let mut cx = 0usize;
        let mut ux = 0usize;

        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        unsafe {
            let y_ptr = y_plane.as_mut_ptr();
            let uv_ptr = uv_plane.as_mut_ptr();
            let rgba_ptr = rgba.as_ptr();

            let y_bias = vdupq_n_s32(bias_y);
            let uv_bias = vdupq_n_s32(bias_uv);
            let v_yr = vdupq_n_s16(transform.yr as i16);
            let v_yg = vdupq_n_s16(transform.yg as i16);
            let v_yb = vdupq_n_s16(transform.yb as i16);
            let v_cb_r = vdupq_n_s16(transform.cb_r as i16);
            let v_cb_g = vdupq_n_s16(transform.cb_g as i16);
            let v_cb_b = vdupq_n_s16(transform.cb_b as i16);
            let v_cr_r = vdupq_n_s16(transform.cr_r as i16);
            let v_cr_g = vdupq_n_s16(transform.cr_g as i16);
            let v_cr_b = vdupq_n_s16(transform.cr_b as i16);
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

                let mut cb_h_high = vmlal_high_s16(uv_bias, r_high, v_cb_r);
                cb_h_high = vmlal_high_s16(cb_h_high, g_high, v_cb_g);
                cb_h_high = vmlal_high_s16(cb_h_high, b_high, v_cb_b);

                let mut cb_h_low = vmlal_s16(uv_bias, r_h_low, vget_low_s16(v_cb_r));
                cb_h_low = vmlal_s16(cb_h_low, g_h_low, vget_low_s16(v_cb_g));
                cb_h_low = vmlal_s16(cb_h_low, b_h_low, vget_low_s16(v_cb_b));

                let cb_high =
                    vcombine_u16(vqshrun_n_s32::<8>(cb_h_low), vqshrun_n_s32::<8>(cb_h_high));

                let mut cr_h_high = vmlal_high_s16(uv_bias, r_high, v_cr_r);
                cr_h_high = vmlal_high_s16(cr_h_high, g_high, v_cr_g);
                cr_h_high = vmlal_high_s16(cr_h_high, b_high, v_cr_b);

                let mut cr_h_low = vmlal_s16(uv_bias, r_h_low, vget_low_s16(v_cr_r));
                cr_h_low = vmlal_s16(cr_h_low, g_h_low, vget_low_s16(v_cr_g));
                cr_h_low = vmlal_s16(cr_h_low, b_h_low, vget_low_s16(v_cr_b));

                let cr_high =
                    vcombine_u16(vqshrun_n_s32::<8>(cr_h_low), vqshrun_n_s32::<8>(cr_h_high));

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

                let mut cb_l_high = vmlal_high_s16(uv_bias, r_low, v_cb_r);
                cb_l_high = vmlal_high_s16(cb_l_high, g_low, v_cb_g);
                cb_l_high = vmlal_high_s16(cb_l_high, b_low, v_cb_b);

                let mut cb_l_low = vmlal_s16(uv_bias, r_l_low, vget_low_s16(v_cb_r));
                cb_l_low = vmlal_s16(cb_l_low, g_l_low, vget_low_s16(v_cb_g));
                cb_l_low = vmlal_s16(cb_l_low, b_l_low, vget_low_s16(v_cb_b));

                let cb_low =
                    vcombine_u16(vqshrun_n_s32::<8>(cb_l_low), vqshrun_n_s32::<8>(cb_l_high));

                let mut cr_l_high = vmlal_high_s16(uv_bias, r_low, v_cr_r);
                cr_l_high = vmlal_high_s16(cr_l_high, g_low, v_cr_g);
                cr_l_high = vmlal_high_s16(cr_l_high, b_low, v_cr_b);

                let mut cr_l_low = vmlal_s16(uv_bias, r_l_low, vget_low_s16(v_cr_r));
                cr_l_low = vmlal_s16(cr_l_low, g_l_low, vget_low_s16(v_cr_g));
                cr_l_low = vmlal_s16(cr_l_low, b_l_low, vget_low_s16(v_cr_b));

                let cr_low =
                    vcombine_u16(vqshrun_n_s32::<8>(cr_l_low), vqshrun_n_s32::<8>(cr_l_high));

                let y = vcombine_u8(vqmovn_u16(y_low), vqmovn_u16(y_high));
                let cb = vcombine_u8(vqmovn_u16(cb_low), vqmovn_u16(cb_high));
                let cr = vcombine_u8(vqmovn_u16(cr_low), vqmovn_u16(cr_high));
                vst1q_u8(y_ptr.add(y_offset + cx), y);

                match chroma_subsampling {
                    YuvChromaSample::YUV420 | YuvChromaSample::YUV422 => {
                        let cb_s = vshrn_n_u16::<1>(vpaddlq_u8(cb));
                        let cr_s = vshrn_n_u16::<1>(vpaddlq_u8(cr));
                        match order {
                            YuvNVOrder::UV => {
                                let store: uint8x8x2_t = uint8x8x2_t(cb_s, cr_s);
                                vst2_u8(uv_ptr.add(uv_offset + ux), store);
                            }
                            YuvNVOrder::VU => {
                                let store: uint8x8x2_t = uint8x8x2_t(cr_s, cb_s);
                                vst2_u8(uv_ptr.add(uv_offset + ux), store);
                            }
                        }
                        ux += 16;
                    }
                    YuvChromaSample::YUV444 => {
                        match order {
                            YuvNVOrder::UV => {
                                let store: uint8x16x2_t = uint8x16x2_t(cb, cr);
                                vst2q_u8(uv_ptr.add(uv_offset + ux), store);
                            }
                            YuvNVOrder::VU => {
                                let store: uint8x16x2_t = uint8x16x2_t(cr, cb);
                                vst2q_u8(uv_ptr.add(uv_offset + ux), store);
                            }
                        }

                        ux += 32;
                    }
                }

                cx += 16;
            }
        }

        for x in (cx..width as usize).step_by(iterator_step) {
            let px = x * channels;
            let r = rgba[rgba_offset + px + source_channels.get_r_channel_offset()] as i32;
            let g = rgba[rgba_offset + px + source_channels.get_g_channel_offset()] as i32;
            let b = rgba[rgba_offset + px + source_channels.get_b_channel_offset()] as i32;
            let y_0 = (r * transform.yr + g * transform.yg + b * transform.yb + bias_y) >> 8;
            let cb = (r * transform.cb_r + g * transform.cb_g + b * transform.cb_b + bias_uv) >> 8;
            let cr = (r * transform.cr_r + g * transform.cr_g + b * transform.cr_b + bias_uv) >> 8;
            y_plane[y_offset + x] = y_0 as u8;
            let uv_pos = uv_offset + ux;
            match order {
                YuvNVOrder::UV => {
                    uv_plane[uv_pos] = cb as u8;
                    uv_plane[uv_pos + 1] = cr as u8;
                }
                YuvNVOrder::VU => {
                    uv_plane[uv_pos] = cr as u8;
                    uv_plane[uv_pos + 1] = cb as u8;
                }
            }
            match chroma_subsampling {
                YuvChromaSample::YUV420 | YuvChromaSample::YUV422 => {
                    let next_x = x + 1;
                    if next_x < width as usize {
                        let next_px = next_x * channels;
                        let r = rgba[rgba_offset + next_px + source_channels.get_r_channel_offset()]
                            as i32;
                        let g = rgba[rgba_offset + next_px + source_channels.get_g_channel_offset()]
                            as i32;
                        let b = rgba[rgba_offset + next_px + source_channels.get_b_channel_offset()]
                            as i32;
                        let y_1 =
                            (r * transform.yr + g * transform.yg + b * transform.yb + bias_y) >> 8;
                        y_plane[y_offset + next_x] = y_1 as u8;
                    }
                }
                _ => {}
            }

            ux += 2;
        }

        y_offset += y_stride as usize;
        rgba_offset += rgba_stride as usize;
        match chroma_subsampling {
            YuvChromaSample::YUV420 => {
                if y & 1 == 1 {
                    uv_offset += uv_stride as usize;
                }
            }
            YuvChromaSample::YUV444 | YuvChromaSample::YUV422 => {
                uv_offset += uv_stride as usize;
            }
        }
    }
}

/// Convert RGB image data to YUV NV16 bi-planar format.
///
/// This function performs RGB to YUV conversion and stores the result in YUV NV16 bi-planar format,
/// with plane for Y (luminance), and bi-plane UV (chrominance) components.
///
/// # Arguments
///
/// * `y_plane` - A mutable slice to store the Y (luminance) plane data.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `uv_plane` - A mutable slice to store the UV (chrominance) plane data.
/// * `uv_stride` - The stride (bytes per row) for the UV plane.
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
pub fn rgb_to_yuv_nv16(
    y_plane: &mut [u8],
    y_stride: u32,
    uv_plane: &mut [u8],
    uv_stride: u32,
    rgb: &[u8],
    rgb_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) {
    rgbx_to_nv::<
        { YuvSourceChannels::Rgb as u8 },
        { YuvNVOrder::UV as u8 },
        { YuvChromaSample::YUV422 as u8 },
    >(
        y_plane, y_stride, uv_plane, uv_stride, rgb, rgb_stride, width, height, range, matrix,
    );
}

/// Convert RGBA image data to YUV NV16 bi-planar format.
///
/// This function performs RGBA to YUV conversion and stores the result in YUV NV16 bi-planar format,
/// with plane for Y (luminance), and bi-plane UV (chrominance) components.
///
/// # Arguments
///
/// * `y_plane` - A mutable slice to store the Y (luminance) plane data.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `uv_plane` - A mutable slice to store the UV (chrominance) plane data.
/// * `uv_stride` - The stride (bytes per row) for the UV plane.
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
pub fn rgba_to_yuv_nv16(
    y_plane: &mut [u8],
    y_stride: u32,
    uv_plane: &mut [u8],
    uv_stride: u32,
    rgba: &[u8],
    rgba_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) {
    rgbx_to_nv::<
        { YuvSourceChannels::Rgba as u8 },
        { YuvNVOrder::UV as u8 },
        { YuvChromaSample::YUV422 as u8 },
    >(
        y_plane,
        y_stride,
        uv_plane,
        uv_stride,
        rgba,
        rgba_stride,
        width,
        height,
        range,
        matrix,
    );
}

/// Convert BGRA image data to YUV NV16 bi-planar format.
///
/// This function performs BGRA to YUV conversion and stores the result in YUV NV16 bi-planar format,
/// with plane for Y (luminance), and bi-plane UV (chrominance) components.
///
/// # Arguments
///
/// * `y_plane` - A mutable slice to store the Y (luminance) plane data.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `uv_plane` - A mutable slice to store the UV (chrominance) plane data.
/// * `uv_stride` - The stride (bytes per row) for the UV plane.
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
pub fn bgra_to_yuv_nv16(
    y_plane: &mut [u8],
    y_stride: u32,
    uv_plane: &mut [u8],
    uv_stride: u32,
    bgra: &[u8],
    bgra_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) {
    rgbx_to_nv::<
        { YuvSourceChannels::Bgra as u8 },
        { YuvNVOrder::UV as u8 },
        { YuvChromaSample::YUV422 as u8 },
    >(
        y_plane,
        y_stride,
        uv_plane,
        uv_stride,
        bgra,
        bgra_stride,
        width,
        height,
        range,
        matrix,
    );
}

/// Convert RGB image data to YUV NV12 bi-planar format.
///
/// This function performs RGB to YUV conversion and stores the result in YUV NV12 bi-planar format,
/// with plane for Y (luminance), and bi-plane UV (chrominance) components.
///
/// # Arguments
///
/// * `y_plane` - A mutable slice to store the Y (luminance) plane data.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `uv_plane` - A mutable slice to store the UV (chrominance) plane data.
/// * `uv_stride` - The stride (bytes per row) for the UV plane.
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
pub fn rgb_to_yuv_nv12(
    y_plane: &mut [u8],
    y_stride: u32,
    uv_plane: &mut [u8],
    uv_stride: u32,
    rgb: &[u8],
    rgb_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) {
    rgbx_to_nv::<
        { YuvSourceChannels::Rgb as u8 },
        { YuvNVOrder::UV as u8 },
        { YuvChromaSample::YUV420 as u8 },
    >(
        y_plane, y_stride, uv_plane, uv_stride, rgb, rgb_stride, width, height, range, matrix,
    );
}

/// Convert RGBA image data to YUV NV12 bi-planar format.
///
/// This function performs RGBA to YUV conversion and stores the result in YUV NV12 bi-planar format,
/// with plane for Y (luminance), and bi-plane UV (chrominance) components.
///
/// # Arguments
///
/// * `y_plane` - A mutable slice to store the Y (luminance) plane data.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `uv_plane` - A mutable slice to store the UV (chrominance) plane data.
/// * `uv_stride` - The stride (bytes per row) for the UV plane.
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
pub fn rgba_to_yuv_nv12(
    y_plane: &mut [u8],
    y_stride: u32,
    uv_plane: &mut [u8],
    uv_stride: u32,
    rgba: &[u8],
    rgba_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) {
    rgbx_to_nv::<
        { YuvSourceChannels::Rgba as u8 },
        { YuvNVOrder::UV as u8 },
        { YuvChromaSample::YUV420 as u8 },
    >(
        y_plane,
        y_stride,
        uv_plane,
        uv_stride,
        rgba,
        rgba_stride,
        width,
        height,
        range,
        matrix,
    );
}

/// Convert BGRA image data to YUV NV12 bi-planar format.
///
/// This function performs BGRA to YUV conversion and stores the result in YUV NV12 bi-planar format,
/// with plane for Y (luminance), and bi-plane UV (chrominance) components.
///
/// # Arguments
///
/// * `y_plane` - A mutable slice to store the Y (luminance) plane data.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `uv_plane` - A mutable slice to store the UV (chrominance) plane data.
/// * `uv_stride` - The stride (bytes per row) for the UV plane.
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
pub fn bgra_to_yuv_nv12(
    y_plane: &mut [u8],
    y_stride: u32,
    uv_plane: &mut [u8],
    uv_stride: u32,
    bgra: &[u8],
    bgra_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) {
    rgbx_to_nv::<
        { YuvSourceChannels::Bgra as u8 },
        { YuvNVOrder::UV as u8 },
        { YuvChromaSample::YUV420 as u8 },
    >(
        y_plane,
        y_stride,
        uv_plane,
        uv_stride,
        bgra,
        bgra_stride,
        width,
        height,
        range,
        matrix,
    );
}

/// Convert RGB image data to YUV NV24 bi-planar format.
///
/// This function performs RGB to YUV conversion and stores the result in YUV NV24 bi-planar format,
/// with plane for Y (luminance), and bi-plane UV (chrominance) components.
///
/// # Arguments
///
/// * `y_plane` - A mutable slice to store the Y (luminance) plane data.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `uv_plane` - A mutable slice to store the UV (chrominance) plane data.
/// * `uv_stride` - The stride (bytes per row) for the UV plane.
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
pub fn rgb_to_yuv_nv24(
    y_plane: &mut [u8],
    y_stride: u32,
    uv_plane: &mut [u8],
    uv_stride: u32,
    rgb: &[u8],
    rgb_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) {
    rgbx_to_nv::<
        { YuvSourceChannels::Rgb as u8 },
        { YuvNVOrder::UV as u8 },
        { YuvChromaSample::YUV444 as u8 },
    >(
        y_plane, y_stride, uv_plane, uv_stride, rgb, rgb_stride, width, height, range, matrix,
    );
}

/// Convert RGBA image data to YUV NV24 bi-planar format.
///
/// This function performs RGBA to YUV conversion and stores the result in YUV NV24 bi-planar format,
/// with plane for Y (luminance), and bi-plane UV (chrominance) components.
///
/// # Arguments
///
/// * `y_plane` - A mutable slice to store the Y (luminance) plane data.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `uv_plane` - A mutable slice to store the UV (chrominance) plane data.
/// * `uv_stride` - The stride (bytes per row) for the UV plane.
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
pub fn rgba_to_yuv_nv24(
    y_plane: &mut [u8],
    y_stride: u32,
    uv_plane: &mut [u8],
    uv_stride: u32,
    rgba: &[u8],
    rgba_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) {
    rgbx_to_nv::<
        { YuvSourceChannels::Rgba as u8 },
        { YuvNVOrder::UV as u8 },
        { YuvChromaSample::YUV444 as u8 },
    >(
        y_plane,
        y_stride,
        uv_plane,
        uv_stride,
        rgba,
        rgba_stride,
        width,
        height,
        range,
        matrix,
    );
}

/// Convert BGRA image data to YUV NV24 bi-planar format.
///
/// This function performs BGRA to YUV conversion and stores the result in YUV NV24 bi-planar format,
/// with plane for Y (luminance), and bi-plane UV (chrominance) components.
///
/// # Arguments
///
/// * `y_plane` - A mutable slice to store the Y (luminance) plane data.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `uv_plane` - A mutable slice to store the UV (chrominance) plane data.
/// * `uv_stride` - The stride (bytes per row) for the UV plane.
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
pub fn bgra_to_yuv_nv24(
    y_plane: &mut [u8],
    y_stride: u32,
    uv_plane: &mut [u8],
    uv_stride: u32,
    bgra: &[u8],
    bgra_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) {
    rgbx_to_nv::<
        { YuvSourceChannels::Bgra as u8 },
        { YuvNVOrder::UV as u8 },
        { YuvChromaSample::YUV444 as u8 },
    >(
        y_plane,
        y_stride,
        uv_plane,
        uv_stride,
        bgra,
        bgra_stride,
        width,
        height,
        range,
        matrix,
    );
}
