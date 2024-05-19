#[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
#[cfg(target_feature = "neon")]
use std::arch::aarch64::{
    uint8x16_t, vcombine_u16, vcombine_u8, vdupq_n_s16, vdupq_n_s32, vget_low_s16, vget_low_u8,
    vld3q_u8, vld4q_u8, vmaxq_s32, vmlal_high_s16, vmlal_s16, vmovl_high_u8, vmovl_u8, vqmovn_u16,
    vqshrun_n_s32, vreinterpretq_s16_u16, vst1q_u8,
};

use crate::yuv_support::{
    get_forward_transform, get_kr_kb, get_yuv_range, ToIntegerTransform, YuvRange,
    YuvSourceChannels, YuvStandardMatrix,
};

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

    let mut y_offset = 0usize;
    let mut rgba_offset = 0usize;

    for _ in 0..height as usize {
        #[allow(unused_variables)]
        #[allow(unused_mut)]
        let mut cx = 0usize;

        #[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
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
            let r = rgba[dst_offset + source_channels.get_r_channel_offset()] as i32;
            let g = rgba[dst_offset + source_channels.get_g_channel_offset()] as i32;
            let b = rgba[dst_offset + source_channels.get_b_channel_offset()] as i32;
            let y = (r * transform.yr + g * transform.yg + b * transform.yb + bias_y) >> 8;
            y_plane[y_offset + x] = y as u8;
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
