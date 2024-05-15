#[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
#[cfg(target_feature = "neon")]
use std::arch::aarch64::{
    uint8x16x2_t, uint8x16x3_t, uint8x16x4_t, uint8x8_t, uint8x8x2_t, vcombine_u8, vdup_n_u8,
    vdupq_n_s16, vdupq_n_u8, vget_high_u8, vget_low_u8, vld1q_u8, vld2_u8, vld2q_u8, vmaxq_s16,
    vmovl_u8, vmull_high_u8, vmull_u8, vmulq_s16, vqaddq_s16, vqshrun_n_s16, vreinterpretq_s16_u16,
    vst3q_u8, vst4q_u8, vsubq_s16, vsubq_u8, vzip1_u8, vzip2_u8,
};

use crate::yuv_support::{
    get_inverse_transform, get_kr_kb, get_yuv_range, YuvChromaSample, YuvNVOrder, YuvRange,
    YuvSourceChannels, YuvStandardMatrix,
};

fn yuv_nv12_to_rgbx<
    const UV_ORDER: u8,
    const DESTINATION_CHANNELS: u8,
    const YUV_CHROMA_SAMPLING: u8,
>(
    y_plane: &[u8],
    y_stride: u32,
    uv_plane: &[u8],
    uv_stride: u32,
    bgra: &mut [u8],
    bgra_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) {
    let order: YuvNVOrder = UV_ORDER.into();
    let range = get_yuv_range(8, range);
    let destination_channels: YuvSourceChannels = DESTINATION_CHANNELS.into();
    let chroma_subsampling: YuvChromaSample = YUV_CHROMA_SAMPLING.into();
    let channels = destination_channels.get_channels_count();
    let kr_kb = get_kr_kb(matrix);
    let transform = get_inverse_transform(255, range.range_y, range.range_uv, kr_kb.kr, kr_kb.kb);
    let i_transform = transform.to_integers(6u32);
    let cr_coef = i_transform.cr_coef;
    let cb_coef = i_transform.cb_coef;
    let y_coef = i_transform.y_coef;
    let g_coef_1 = i_transform.g_coeff_1;
    let g_coef_2 = i_transform.g_coeff_2;

    let bias_y = range.bias_y as i32;
    let bias_uv = range.bias_uv as i32;

    let mut y_offset = 0usize;
    let mut uv_offset = 0usize;
    let mut dst_offset = 0usize;

    for y in 0..height as usize {
        let mut x = 0usize;

        #[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
        #[cfg(target_feature = "neon")]
        unsafe {
            let y_ptr = y_plane.as_ptr();
            let uv_ptr = uv_plane.as_ptr();
            let bgra_ptr = bgra.as_mut_ptr();

            let y_corr = vdupq_n_u8(bias_y as u8);
            let uv_corr = vdupq_n_s16(bias_uv as i16);
            let v_luma_coeff = vdupq_n_u8(y_coef as u8);
            let v_luma_coeff_8 = vdup_n_u8(y_coef as u8);
            let v_cr_coeff = vdupq_n_s16(cr_coef as i16);
            let v_cb_coeff = vdupq_n_s16(cb_coef as i16);
            let v_min_values = vdupq_n_s16(0i16);
            let v_g_coeff_1 = vdupq_n_s16(-1i16 * (g_coef_1 as i16));
            let v_g_coeff_2 = vdupq_n_s16(-1i16 * (g_coef_2 as i16));
            let v_alpha = vdupq_n_u8(255u8);
            while x + 16 < width as usize {
                let y_values = vsubq_u8(vld1q_u8(y_ptr.add(y_offset + x)), y_corr);

                let u_high_u8: uint8x8_t;
                let v_high_u8: uint8x8_t;
                let u_low_u8: uint8x8_t;
                let v_low_u8: uint8x8_t;

                match chroma_subsampling {
                    YuvChromaSample::YUV420 | YuvChromaSample::YUV422 => {
                        let mut uv_values = vld2_u8(uv_ptr.add(uv_offset + x));
                        if order == YuvNVOrder::VU {
                            uv_values = uint8x8x2_t(uv_values.1, uv_values.0);
                        }

                        u_high_u8 = vzip2_u8(uv_values.0, uv_values.0);
                        v_high_u8 = vzip2_u8(uv_values.1, uv_values.1);
                        u_low_u8 = vzip1_u8(uv_values.0, uv_values.0);
                        v_low_u8 = vzip1_u8(uv_values.1, uv_values.1);
                    }
                    YuvChromaSample::YUV444 => {
                        let mut uv_values = vld2q_u8(uv_ptr.add(uv_offset + x * 2));
                        if order == YuvNVOrder::VU {
                            uv_values = uint8x16x2_t(uv_values.1, uv_values.0);
                        }
                        u_high_u8 = vget_high_u8(uv_values.0);
                        v_high_u8 = vget_high_u8(uv_values.1);
                        u_low_u8 = vget_low_u8(uv_values.0);
                        v_low_u8 = vget_low_u8(uv_values.1);
                    }
                }

                let u_high = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(u_high_u8)), uv_corr);
                let v_high = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(v_high_u8)), uv_corr);
                let y_high = vreinterpretq_s16_u16(vmull_high_u8(y_values, v_luma_coeff));

                let r_high = vqshrun_n_s16::<6>(vmaxq_s16(
                    vqaddq_s16(y_high, vmulq_s16(v_high, v_cr_coeff)),
                    v_min_values,
                ));
                let b_high = vqshrun_n_s16::<6>(vmaxq_s16(
                    vqaddq_s16(y_high, vmulq_s16(u_high, v_cb_coeff)),
                    v_min_values,
                ));
                let g_high = vqshrun_n_s16::<6>(vmaxq_s16(
                    vqaddq_s16(
                        y_high,
                        vqaddq_s16(
                            vmulq_s16(v_high, v_g_coeff_1),
                            vmulq_s16(u_high, v_g_coeff_2),
                        ),
                    ),
                    v_min_values,
                ));

                let u_low = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(u_low_u8)), uv_corr);
                let v_low = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(v_low_u8)), uv_corr);
                let y_low = vreinterpretq_s16_u16(vmull_u8(vget_low_u8(y_values), v_luma_coeff_8));

                let r_low = vqshrun_n_s16::<6>(vmaxq_s16(
                    vqaddq_s16(y_low, vmulq_s16(v_low, v_cr_coeff)),
                    v_min_values,
                ));
                let b_low = vqshrun_n_s16::<6>(vmaxq_s16(
                    vqaddq_s16(y_low, vmulq_s16(u_low, v_cb_coeff)),
                    v_min_values,
                ));
                let g_low = vqshrun_n_s16::<6>(vmaxq_s16(
                    vqaddq_s16(
                        y_low,
                        vqaddq_s16(vmulq_s16(v_low, v_g_coeff_1), vmulq_s16(u_low, v_g_coeff_2)),
                    ),
                    v_min_values,
                ));

                let r_values = vcombine_u8(r_low, r_high);
                let g_values = vcombine_u8(g_low, g_high);
                let b_values = vcombine_u8(b_low, b_high);

                let dst_shift = dst_offset + x * channels;

                match destination_channels {
                    YuvSourceChannels::Rgb => {
                        let dst_pack: uint8x16x3_t = uint8x16x3_t(r_values, g_values, b_values);
                        vst3q_u8(bgra_ptr.add(dst_shift), dst_pack);
                    }
                    YuvSourceChannels::Rgba => {
                        let dst_pack: uint8x16x4_t =
                            uint8x16x4_t(b_values, g_values, r_values, v_alpha);
                        vst4q_u8(bgra_ptr.add(dst_shift), dst_pack);
                    }
                    YuvSourceChannels::Bgra => {
                        let dst_pack: uint8x16x4_t =
                            uint8x16x4_t(r_values, g_values, b_values, v_alpha);
                        vst4q_u8(bgra_ptr.add(dst_shift), dst_pack);
                    }
                }

                x += 16;
            }
        }

        while x < width as usize {
            let y_value = (y_plane[y_offset + x] as i32 - bias_y) * y_coef;
            let cb_value: i32;
            let cr_value: i32;
            let cb_pos = uv_offset + x;
            let cr_pos = uv_offset + x + 1;

            match order {
                YuvNVOrder::UV => {
                    cb_value = uv_plane[cb_pos] as i32 - bias_uv;
                    cr_value = uv_plane[cr_pos] as i32 - bias_uv;
                }
                YuvNVOrder::VU => {
                    cb_value = uv_plane[cr_pos] as i32 - bias_uv;
                    cr_value = uv_plane[cb_pos] as i32 - bias_uv;
                }
            }

            let r = ((y_value + cr_coef * cr_value) >> 6).min(255).max(0);
            let b = ((y_value + cb_coef * cb_value) >> 6).min(255).max(0);
            let g = ((y_value - g_coef_1 * cr_value - g_coef_2 * cb_value) >> 6)
                .min(255)
                .max(0);

            let px = x * channels;

            let dst_shift = dst_offset + px;

            bgra[dst_shift + destination_channels.get_b_channel_offset()] = b as u8;
            bgra[dst_shift + destination_channels.get_g_channel_offset()] = g as u8;
            bgra[dst_shift + destination_channels.get_r_channel_offset()] = r as u8;
            if destination_channels.has_alpha() {
                bgra[dst_shift + destination_channels.get_a_channel_offset()] = 255;
            }

            if chroma_subsampling == YuvChromaSample::YUV422
                || chroma_subsampling == YuvChromaSample::YUV420
            {
                x += 1;
                if x + 1 < width as usize {
                    let y_value = (y_plane[y_offset + x + 1] as i32 - bias_y) * y_coef;

                    let r = ((y_value + cr_coef * cr_value) >> 6).min(255).max(0);
                    let b = ((y_value + cb_coef * cb_value) >> 6).min(255).max(0);
                    let g = ((y_value - g_coef_1 * cr_value - g_coef_2 * cb_value) >> 6)
                        .min(255)
                        .max(0);

                    let next_px = x * channels;
                    let dst_shift = dst_offset + next_px;
                    bgra[dst_shift + destination_channels.get_b_channel_offset()] = b as u8;
                    bgra[dst_shift + destination_channels.get_g_channel_offset()] = g as u8;
                    bgra[dst_shift + destination_channels.get_r_channel_offset()] = r as u8;
                    if destination_channels.has_alpha() {
                        bgra[dst_shift + destination_channels.get_a_channel_offset()] = 255;
                    }
                }
            }

            x += 1;
        }

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

        dst_offset += bgra_stride as usize;
        y_offset += y_stride as usize;
    }
}

/// Convert YUV NV12 format to BGRA format.
///
/// This function takes YUV NV12 data with 8-bit precision,
/// and converts it to BGRA format with 8-bit per channel precision.
///
/// # Arguments
///
/// * `y_plane` - A slice to load the Y (luminance) plane data.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `uv_plane` - A slice to load the UV (chrominance) plane data.
/// * `uv_stride` - The stride (bytes per row) for the UV plane.
/// * `width` - The width of the YUV image.
/// * `height` - The height of the YUV image.
/// * `bgra_data` - A mutable slice to store the converted BGRA data.
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGRA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv_nv12_to_bgra(
    y_plane: &[u8],
    y_stride: u32,
    uv_plane: &[u8],
    uv_stride: u32,
    bgra: &mut [u8],
    bgra_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) {
    yuv_nv12_to_rgbx::<
        { YuvNVOrder::UV as u8 },
        { YuvSourceChannels::Bgra as u8 },
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
    )
}

/// Convert YUV NV21 format to BGRA format.
///
/// This function takes YUV NV12 data with 8-bit precision,
/// and converts it to BGRA format with 8-bit per channel precision.
///
/// # Arguments
///
/// * `y_plane` - A slice to load the Y (luminance) plane data.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `vu_plane` - A slice to load the VU (chrominance) plane data.
/// * `vu_stride` - The stride (bytes per row) for the VU plane.
/// * `width` - The width of the YUV image.
/// * `height` - The height of the YUV image.
/// * `bgra_data` - A mutable slice to store the converted BGRA data.
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGRA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv_nv21_to_bgra(
    y_plane: &[u8],
    y_stride: u32,
    vu_plane: &[u8],
    vu_stride: u32,
    bgra: &mut [u8],
    bgra_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) {
    yuv_nv12_to_rgbx::<
        { YuvNVOrder::VU as u8 },
        { YuvSourceChannels::Bgra as u8 },
        { YuvChromaSample::YUV420 as u8 },
    >(
        y_plane,
        y_stride,
        vu_plane,
        vu_stride,
        bgra,
        bgra_stride,
        width,
        height,
        range,
        matrix,
    )
}

/// Convert YUV NV12 format to RGBA format.
///
/// This function takes YUV NV12 data with 8-bit precision,
/// and converts it to RGBA format with 8-bit per channel precision.
///
/// # Arguments
///
/// * `y_plane` - A slice to load the Y (luminance) plane data.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `uv_plane` - A slice to load the UV (chrominance) plane data.
/// * `uv_stride` - The stride (bytes per row) for the UV plane.
/// * `width` - The width of the YUV image.
/// * `height` - The height of the YUV image.
/// * `rgba_data` - A mutable slice to store the converted RGBA data.
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGRA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv_nv12_to_rgba(
    y_plane: &[u8],
    y_stride: u32,
    uv_plane: &[u8],
    uv_stride: u32,
    rgba: &mut [u8],
    rgba_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) {
    yuv_nv12_to_rgbx::<
        { YuvNVOrder::UV as u8 },
        { YuvSourceChannels::Rgba as u8 },
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
    )
}

/// Convert YUV NV21 format to RGBA format.
///
/// This function takes YUV NV21 data with 8-bit precision,
/// and converts it to RGBA format with 8-bit per channel precision.
///
/// # Arguments
///
/// * `y_plane` - A slice to load the Y (luminance) plane data.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `vu_plane` - A slice to load the VU (chrominance) plane data.
/// * `vu_stride` - The stride (bytes per row) for the VU plane.
/// * `width` - The width of the YUV image.
/// * `height` - The height of the YUV image.
/// * `rgba_data` - A mutable slice to store the converted RGBA data.
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGRA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv_nv21_to_rgba(
    y_plane: &[u8],
    y_stride: u32,
    vu_plane: &[u8],
    vu_stride: u32,
    rgba: &mut [u8],
    rgba_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) {
    yuv_nv12_to_rgbx::<
        { YuvNVOrder::VU as u8 },
        { YuvSourceChannels::Rgba as u8 },
        { YuvChromaSample::YUV420 as u8 },
    >(
        y_plane,
        y_stride,
        vu_plane,
        vu_stride,
        rgba,
        rgba_stride,
        width,
        height,
        range,
        matrix,
    )
}

/// Convert YUV NV12 format to RGB format.
///
/// This function takes YUV NV12 data with 8-bit precision,
/// and converts it to RGB format with 8-bit per channel precision.
///
/// # Arguments
///
/// * `y_plane` - A slice to load the Y (luminance) plane data.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `uv_plane` - A slice to load the UV (chrominance) plane data.
/// * `uv_stride` - The stride (bytes per row) for the UV plane.
/// * `width` - The width of the YUV image.
/// * `height` - The height of the YUV image.
/// * `rgb_data` - A mutable slice to store the converted RGB data.
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGRA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv_nv12_to_rgb(
    y_plane: &[u8],
    y_stride: u32,
    uv_plane: &[u8],
    uv_stride: u32,
    rgb: &mut [u8],
    rgb_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) {
    yuv_nv12_to_rgbx::<
        { YuvNVOrder::UV as u8 },
        { YuvSourceChannels::Rgb as u8 },
        { YuvChromaSample::YUV420 as u8 },
    >(
        y_plane,
        y_stride,
        uv_plane,
        uv_stride,
        rgb,
        rgb_stride,
        width,
        height,
        range,
        matrix,
    )
}

/// Convert YUV NV21 format to RGB format.
///
/// This function takes YUV NV21 data with 8-bit precision,
/// and converts it to RGB format with 8-bit per channel precision.
///
/// # Arguments
///
/// * `y_plane` - A slice to load the Y (luminance) plane data.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `vu_plane` - A slice to load the VU (chrominance) plane data.
/// * `vu_stride` - The stride (bytes per row) for the VU plane.
/// * `width` - The width of the YUV image.
/// * `height` - The height of the YUV image.
/// * `rgb_data` - A mutable slice to store the converted RGB data.
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGRA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv_nv21_to_rgb(
    y_plane: &[u8],
    y_stride: u32,
    vu_plane: &[u8],
    vu_stride: u32,
    rgb: &mut [u8],
    rgb_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) {
    yuv_nv12_to_rgbx::<
        { YuvNVOrder::VU as u8 },
        { YuvSourceChannels::Rgb as u8 },
        { YuvChromaSample::YUV420 as u8 },
    >(
        y_plane,
        y_stride,
        vu_plane,
        vu_stride,
        rgb,
        rgb_stride,
        width,
        height,
        range,
        matrix,
    )
}

