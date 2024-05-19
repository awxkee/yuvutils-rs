#[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
#[cfg(target_feature = "neon")]
use std::arch::aarch64::{
    int16x4_t, int16x8_t, uint8x8x3_t, uint8x8x4_t, vcombine_s16, vdup_n_s16, vdup_n_u8,
    vdupq_n_s16, vget_low_s16, vld1_u16, vld1q_u16, vmaxq_s16, vmlal_s16, vmull_high_s16,
    vmull_s16, vqshrun_n_s16, vreinterpret_s16_u16, vreinterpret_u16_u8, vreinterpret_u8_u16,
    vreinterpretq_s16_u16, vreinterpretq_u16_u8, vreinterpretq_u8_u16, vrev16_u8, vrev16q_u8,
    vshr_n_u16, vshrn_n_s32, vshrq_n_u16, vst3_u8, vst4_u8, vsub_s16, vsubq_s16, vzip1_s16,
    vzip2_s16,
};
use std::slice;

use crate::yuv_support::{
    get_inverse_transform, get_kr_kb, get_yuv_range, YuvBytesPosition, YuvChromaSample, YuvEndian,
    YuvRange, YuvSourceChannels, YuvStandardMatrix,
};

fn yuv_p10_to_rgbx_impl<
    const DESTINATION_CHANNELS: u8,
    const SAMPLING: u8,
    const ENDIANNESS: u8,
    const BYTES_POSITION: u8,
>(
    y_plane: &[u16],
    y_stride: u32,
    u_plane: &[u16],
    u_stride: u32,
    v_plane: &[u16],
    v_stride: u32,
    rgba: &mut [u8],
    rgba_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) {
    let destination_channels: YuvSourceChannels = DESTINATION_CHANNELS.into();
    let channels = destination_channels.get_channels_count();
    let chroma_subsampling: YuvChromaSample = SAMPLING.into();
    let endianness: YuvEndian = ENDIANNESS.into();
    let bytes_position: YuvBytesPosition = BYTES_POSITION.into();
    let range = get_yuv_range(10, range);
    let kr_kb = get_kr_kb(matrix);
    let max_range_p10 = (2f32.powi(10) - 1f32) as u32;
    let transform = get_inverse_transform(
        max_range_p10,
        range.range_y,
        range.range_uv,
        kr_kb.kr,
        kr_kb.kb,
    );
    let i_transform = transform.to_integers(6u32);
    let cr_coef = i_transform.cr_coef;
    let cb_coef = i_transform.cb_coef;
    let y_coef = i_transform.y_coef;
    let g_coef_1 = i_transform.g_coeff_1;
    let g_coef_2 = i_transform.g_coeff_2;

    let bias_y = range.bias_y as i32;
    let bias_uv = range.bias_uv as i32;

    let mut y_offset = 0usize;
    let mut u_offset = 0usize;
    let mut v_offset = 0usize;
    let mut dst_offset = 0usize;

    let y_src_ptr = y_plane.as_ptr() as *const u8;
    let u_src_ptr = u_plane.as_ptr() as *const u8;
    let v_src_ptr = v_plane.as_ptr() as *const u8;

    for y in 0..height as usize {
        let mut x = 0usize;
        let mut cx = 0usize;

        let y_ld_ptr = unsafe { y_src_ptr.offset(y_offset as isize) as *const u16 };
        let y_ld = unsafe { slice::from_raw_parts(y_ld_ptr, width as usize) };
        let u_ld_ptr = unsafe { u_src_ptr.offset(u_offset as isize) as *const u16 };
        let u_ld = unsafe { slice::from_raw_parts(u_ld_ptr, width as usize) };
        let v_ld_ptr = unsafe { v_src_ptr.offset(v_offset as isize) as *const u16 };
        let v_ld = unsafe { slice::from_raw_parts(v_ld_ptr, width as usize) };

        #[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
        #[cfg(target_feature = "neon")]
        unsafe {
            let dst_ptr = rgba.as_mut_ptr();

            let y_corr = vdupq_n_s16(bias_y as i16);
            let uv_corr = vdup_n_s16(bias_uv as i16);
            let v_luma_coeff = vdupq_n_s16(y_coef as i16);
            let v_luma_coeff_4 = vdup_n_s16(y_coef as i16);
            let v_cr_coeff = vdup_n_s16(cr_coef as i16);
            let v_cb_coeff = vdup_n_s16(cb_coef as i16);
            let v_min_values = vdupq_n_s16(0i16);
            let v_g_coeff_1 = vdup_n_s16(-1i16 * (g_coef_1 as i16));
            let v_g_coeff_2 = vdup_n_s16(-1i16 * (g_coef_2 as i16));
            let v_alpha = vdup_n_u8(255u8);

            while x + 8 < width as usize {
                let y_values: int16x8_t;

                let u_values_c: int16x4_t;
                let v_values_c: int16x4_t;

                let u_values_l = vld1_u16(u_ld_ptr.add(cx));
                let v_values_l = vld1_u16(v_ld_ptr.add(cx));

                match endianness {
                    YuvEndian::BigEndian => {
                        let mut y_u_values = vreinterpretq_u16_u8(vrev16q_u8(
                            vreinterpretq_u8_u16(vld1q_u16(y_ld_ptr.add(x))),
                        ));
                        if bytes_position == YuvBytesPosition::MostSignificantBytes {
                            y_u_values = vshrq_n_u16::<6>(y_u_values);
                        }
                        y_values = vsubq_s16(vreinterpretq_s16_u16(y_u_values), y_corr);

                        let mut u_v =
                            vreinterpret_u16_u8(vrev16_u8(vreinterpret_u8_u16(u_values_l)));
                        let mut v_v =
                            vreinterpret_u16_u8(vrev16_u8(vreinterpret_u8_u16(v_values_l)));
                        if bytes_position == YuvBytesPosition::MostSignificantBytes {
                            u_v = vshr_n_u16::<6>(u_v);
                            v_v = vshr_n_u16::<6>(v_v);
                        }
                        u_values_c = vsub_s16(vreinterpret_s16_u16(u_v), uv_corr);
                        v_values_c = vsub_s16(vreinterpret_s16_u16(v_v), uv_corr);
                    }
                    YuvEndian::LittleEndian => {
                        let mut y_vl = vld1q_u16(y_ld_ptr.add(x));
                        if bytes_position == YuvBytesPosition::MostSignificantBytes {
                            y_vl = vshrq_n_u16::<6>(y_vl);
                        }
                        y_values = vsubq_s16(vreinterpretq_s16_u16(y_vl), y_corr);

                        let mut u_vl = u_values_l;
                        let mut v_vl = v_values_l;
                        if bytes_position == YuvBytesPosition::MostSignificantBytes {
                            u_vl = vshr_n_u16::<6>(u_vl);
                            v_vl = vshr_n_u16::<6>(v_vl);
                        }
                        u_values_c = vsub_s16(vreinterpret_s16_u16(u_vl), uv_corr);
                        v_values_c = vsub_s16(vreinterpret_s16_u16(v_vl), uv_corr);
                    }
                }

                let u_high = vzip2_s16(u_values_c, u_values_c);
                let v_high = vzip2_s16(v_values_c, v_values_c);

                let y_high = vmull_high_s16(y_values, v_luma_coeff);

                let r_high = vshrn_n_s32::<6>(vmlal_s16(y_high, v_high, v_cr_coeff));
                let b_high = vshrn_n_s32::<6>(vmlal_s16(y_high, u_high, v_cb_coeff));
                let g_high = vshrn_n_s32::<6>(vmlal_s16(
                    vmlal_s16(y_high, v_high, v_g_coeff_1),
                    u_high,
                    v_g_coeff_2,
                ));

                let y_low = vmull_s16(vget_low_s16(y_values), v_luma_coeff_4);
                let u_low = vzip1_s16(u_values_c, u_values_c);
                let v_low = vzip1_s16(v_values_c, v_values_c);

                let r_low = vshrn_n_s32::<6>(vmlal_s16(y_low, v_low, v_cr_coeff));
                let b_low = vshrn_n_s32::<6>(vmlal_s16(y_low, u_low, v_cb_coeff));
                let g_low = vshrn_n_s32::<6>(vmlal_s16(
                    vmlal_s16(y_low, v_low, v_g_coeff_1),
                    u_low,
                    v_g_coeff_2,
                ));

                let r_values =
                    vqshrun_n_s16::<2>(vmaxq_s16(vcombine_s16(r_low, r_high), v_min_values));
                let g_values =
                    vqshrun_n_s16::<2>(vmaxq_s16(vcombine_s16(g_low, g_high), v_min_values));
                let b_values =
                    vqshrun_n_s16::<2>(vmaxq_s16(vcombine_s16(b_low, b_high), v_min_values));

                match destination_channels {
                    YuvSourceChannels::Rgb => {
                        let dst_pack: uint8x8x3_t = uint8x8x3_t(r_values, g_values, b_values);
                        vst3_u8(dst_ptr.add(dst_offset + x * channels), dst_pack);
                    }
                    YuvSourceChannels::Rgba => {
                        let dst_pack: uint8x8x4_t =
                            uint8x8x4_t(r_values, g_values, b_values, v_alpha);
                        vst4_u8(dst_ptr.add(dst_offset + x * channels), dst_pack);
                    }
                    YuvSourceChannels::Bgra => {
                        let dst_pack: uint8x8x4_t =
                            uint8x8x4_t(b_values, g_values, r_values, v_alpha);
                        vst4_u8(dst_ptr.add(dst_offset + x * channels), dst_pack);
                    }
                }

                x += 8;
                cx += 4;
            }
        }

        while x < width as usize {
            let y_value: i32;
            let cb_value: i32;
            let cr_value: i32;
            match endianness {
                YuvEndian::BigEndian => {
                    let mut y_vl = u16::from_be(y_ld[x]) as i32;
                    let mut cb_vl = u16::from_be(u_ld[cx]) as i32;
                    let mut cr_vl = u16::from_be(v_ld[cx]) as i32;
                    if bytes_position == YuvBytesPosition::MostSignificantBytes {
                        y_vl = y_vl >> 6;
                        cb_vl = cb_vl >> 6;
                        cr_vl = cr_vl >> 6;
                    }
                    y_value = (y_vl - bias_y) * y_coef;

                    cb_value = cb_vl - bias_uv;
                    cr_value = cr_vl - bias_uv;
                }
                YuvEndian::LittleEndian => {
                    let mut y_vl = u16::from_le(y_ld[x]) as i32;
                    let mut cb_vl = u16::from_le(u_ld[cx]) as i32;
                    let mut cr_vl = u16::from_le(v_ld[cx]) as i32;
                    if bytes_position == YuvBytesPosition::MostSignificantBytes {
                        y_vl = y_vl >> 6;
                        cb_vl = cb_vl >> 6;
                        cr_vl = cr_vl >> 6;
                    }
                    y_value = (y_vl - bias_y) * y_coef;

                    cb_value = cb_vl - bias_uv;
                    cr_value = cr_vl - bias_uv;
                }
            }

            // shift right 8 due to we want to make it 8 bit instead of 10

            let r_u16 = (y_value + cr_coef * cr_value) >> 8;
            let b_u16 = (y_value + cb_coef * cb_value) >> 8;
            let g_u16 = (y_value - g_coef_1 * cr_value - g_coef_2 * cb_value) >> 8;

            let r = r_u16.min(255).max(0);
            let b = b_u16.min(255).max(0);
            let g = g_u16.min(255).max(0);

            let px = x * channels;

            let rgb_offset = dst_offset + px;

            rgba[rgb_offset + destination_channels.get_b_channel_offset()] = b as u8;
            rgba[rgb_offset + destination_channels.get_g_channel_offset()] = g as u8;
            rgba[rgb_offset + destination_channels.get_r_channel_offset()] = r as u8;
            if destination_channels.has_alpha() {
                rgba[rgb_offset + destination_channels.get_a_channel_offset()] = 255;
            }

            x += 1;

            if x + 1 < width as usize {
                let y_value: i32;
                match endianness {
                    YuvEndian::BigEndian => {
                        let mut y_vl = u16::from_be(y_ld[x]) as i32;
                        if bytes_position == YuvBytesPosition::MostSignificantBytes {
                            y_vl = y_vl >> 6;
                        }
                        y_value = (y_vl - bias_y) * y_coef;
                    }
                    YuvEndian::LittleEndian => {
                        let mut y_vl = u16::from_le(y_ld[x]) as i32;
                        if bytes_position == YuvBytesPosition::MostSignificantBytes {
                            y_vl = y_vl >> 6;
                        }
                        y_value = (y_vl - bias_y) * y_coef;
                    }
                }

                let r_u16 = (y_value + cr_coef * cr_value) >> 8;
                let b_u16 = (y_value + cb_coef * cb_value) >> 8;
                let g_u16 = (y_value - g_coef_1 * cr_value - g_coef_2 * cb_value) >> 8;

                let r = r_u16.min(255).max(0);
                let b = b_u16.min(255).max(0);
                let g = g_u16.min(255).max(0);

                let px = x * channels;
                let rgb_offset = dst_offset + px;
                rgba[rgb_offset + destination_channels.get_b_channel_offset()] = b as u8;
                rgba[rgb_offset + destination_channels.get_g_channel_offset()] = g as u8;
                rgba[rgb_offset + destination_channels.get_r_channel_offset()] = r as u8;
                if destination_channels.has_alpha() {
                    rgba[rgb_offset + destination_channels.get_a_channel_offset()] = 255;
                }
            }

            x += 1;
            cx += 1;
        }

        match chroma_subsampling {
            YuvChromaSample::YUV420 => {
                if y & 1 == 1 {
                    u_offset += u_stride as usize;
                    v_offset += v_stride as usize;
                }
            }
            YuvChromaSample::YUV422 | YuvChromaSample::YUV444 => {
                u_offset += u_stride as usize;
                v_offset += v_stride as usize;
            }
        }

        dst_offset += rgba_stride as usize;
        y_offset += y_stride as usize;
    }
}

/// Convert YUV 420 planar format with 10-bit (Little-Endian) pixel format to BGRA format.
///
/// This function takes YUV 420 planar data with 10-bit precision stored in Little-Endian.
/// and converts it to BGRA format with 8-bit precision per channel
///
/// # Arguments
///
/// * `y_plane` -  A slice containing Y (luminance) with 10 bit depth (Little-Endian).
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `u_plane` - A slice to load the U (chrominance) with 10 bit depth (Little-Endian).
/// * `u_stride` - The stride (bytes per row) for the U plane.
/// * `v_plane` - A slice to load the V (chrominance) with 10 bit depth (Little-Endian).
/// * `v_stride` - The stride (bytes per row) for the U plane.
/// * `width` - The width of the YUV image.
/// * `height` - The height of the YUV image.
/// * `bgra_data` - A mutable slice to store the converted BGRA data.
/// * `bgra_stride` - The stride (bytes per row) for BGRA data.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGRA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv420_p10_to_bgra(
    y_plane: &[u16],
    y_stride: u32,
    u_plane: &[u16],
    u_stride: u32,
    v_plane: &[u16],
    v_stride: u32,
    bgra: &mut [u8],
    bgra_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) {
    yuv_p10_to_rgbx_impl::<
        { YuvSourceChannels::Bgra as u8 },
        { YuvChromaSample::YUV420 as u8 },
        { YuvEndian::LittleEndian as u8 },
        { YuvBytesPosition::LeastSignificantBytes as u8 },
    >(
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

/// Convert YUV 422 format with 10-bit (Little-Endian) pixel format to BGRA format .
///
/// This function takes YUV 422 data with 10-bit precision stored in Little-Endian.
/// and converts it to BGRA format with 8-bit precision per channel.
///
/// # Arguments
///
/// * `y_plane` -  A slice containing Y (luminance) with 10 bit depth (Little-Endian).
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `u_plane` - A slice to load the U (chrominance) with 10 bit depth (Little-Endian).
/// * `u_stride` - The stride (bytes per row) for the U plane.
/// * `v_plane` - A slice to load the V (chrominance) with 10 bit depth (Little-Endian).
/// * `v_stride` - The stride (bytes per row) for the U plane
/// * `width` - The width of the YUV image.
/// * `height` - The height of the YUV image.
/// * `bgra_data` - A mutable slice to store the converted BGRA data.
/// * `bgra_stride` - The stride (bytes per row) for BGRA data.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGRA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv422_p10_to_bgra(
    y_plane: &[u16],
    y_stride: u32,
    u_plane: &[u16],
    u_stride: u32,
    v_plane: &[u16],
    v_stride: u32,
    bgra: &mut [u8],
    bgra_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) {
    yuv_p10_to_rgbx_impl::<
        { YuvSourceChannels::Bgra as u8 },
        { YuvChromaSample::YUV422 as u8 },
        { YuvEndian::LittleEndian as u8 },
        { YuvBytesPosition::LeastSignificantBytes as u8 },
    >(
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

/// Convert YUV 420 format with 10-bit (Big-Endian) pixel format to BGRA format.
///
/// This function takes YUV 420 data with 10-bit precision stored in Big-Endian.
/// and converts it to BGRA format with 8-bit precision per channel.
///
/// # Arguments
///
/// * `y_plane` -  A slice containing Y (luminance) with 10 bit depth (Big-Endian).
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `u_plane` - A slice to load the U (chrominance) with 10 bit depth (Big-Endian).
/// * `u_stride` - The stride (bytes per row) for the U plane.
/// * `v_plane` - A slice to load the V (chrominance) with 10 bit depth (Big-Endian).
/// * `v_stride` - The stride (bytes per row) for the U plane
/// * `width` - The width of the YUV image.
/// * `height` - The height of the YUV image.
/// * `bgra_data` - A mutable slice to store the converted BGRA data.
/// * `bgra_stride` - The stride (bytes per row) for BGRA data.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGRA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv420_p10_to_bgra_be(
    y_plane: &[u16],
    y_stride: u32,
    u_plane: &[u16],
    u_stride: u32,
    v_plane: &[u16],
    v_stride: u32,
    bgra: &mut [u8],
    bgra_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) {
    yuv_p10_to_rgbx_impl::<
        { YuvSourceChannels::Bgra as u8 },
        { YuvChromaSample::YUV420 as u8 },
        { YuvEndian::BigEndian as u8 },
        { YuvBytesPosition::LeastSignificantBytes as u8 },
    >(
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

/// Convert YUV 422 format with 10-bit (Big-Endian) pixel format to BGRA format.
///
/// This function takes YUV 422 data with 10-bit precision stored in Big-Endian.
/// and converts it to BGRA format with 8-bit precision per channel.
///
/// # Arguments
///
/// * `y_plane` -  A slice containing Y (luminance) with 10 bit depth (Big-Endian).
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `u_plane` - A slice to load the U (chrominance) with 10 bit depth (Big-Endian).
/// * `u_stride` - The stride (bytes per row) for the U plane.
/// * `v_plane` - A slice to load the V (chrominance) with 10 bit depth (Big-Endian).
/// * `v_stride` - The stride (bytes per row) for the U plane
/// * `width` - The width of the YUV image.
/// * `height` - The height of the YUV image.
/// * `bgra_data` - A mutable slice to store the converted BGRA data.
/// * `bgra_stride` - The stride (bytes per row) for BGRA data.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGRA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv422_p10_to_bgra_be(
    y_plane: &[u16],
    y_stride: u32,
    u_plane: &[u16],
    u_stride: u32,
    v_plane: &[u16],
    v_stride: u32,
    bgra: &mut [u8],
    bgra_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) {
    yuv_p10_to_rgbx_impl::<
        { YuvSourceChannels::Bgra as u8 },
        { YuvChromaSample::YUV422 as u8 },
        { YuvEndian::BigEndian as u8 },
        { YuvBytesPosition::LeastSignificantBytes as u8 },
    >(
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

/// Convert YUV 420 planar format with 10-bit (Little-Endian) pixel format to RGBA format.
///
/// This function takes YUV 420 planar data with 10-bit precision stored in Little-Endian.
/// and converts it to RGBA format with 8-bit precision per channel
///
/// # Arguments
///
/// * `y_plane` -  A slice containing Y (luminance) with 10 bit depth (Little-Endian).
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `u_plane` - A slice to load the U (chrominance) with 10 bit depth (Little-Endian).
/// * `u_stride` - The stride (bytes per row) for the U plane.
/// * `v_plane` - A slice to load the V (chrominance) with 10 bit depth (Little-Endian).
/// * `v_stride` - The stride (bytes per row) for the U plane.
/// * `width` - The width of the YUV image.
/// * `height` - The height of the YUV image.
/// * `rgba_data` - A mutable slice to store the converted BGRA data.
/// * `rgba_stride` - The stride (bytes per row) for RGBA data.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input RGBA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv420_p10_to_rgba(
    y_plane: &[u16],
    y_stride: u32,
    u_plane: &[u16],
    u_stride: u32,
    v_plane: &[u16],
    v_stride: u32,
    rgba: &mut [u8],
    rgba_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) {
    yuv_p10_to_rgbx_impl::<
        { YuvSourceChannels::Rgba as u8 },
        { YuvChromaSample::YUV420 as u8 },
        { YuvEndian::LittleEndian as u8 },
        { YuvBytesPosition::LeastSignificantBytes as u8 },
    >(
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

/// Convert YUV 422 format with 10-bit (Little-Endian) pixel format to RGBA format .
///
/// This function takes YUV 422 data with 10-bit precision stored in Little-Endian.
/// and converts it to RGBA format with 8-bit precision per channel.
///
/// # Arguments
///
/// * `y_plane` -  A slice containing Y (luminance) with 10 bit depth (Little-Endian).
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `u_plane` - A slice to load the U (chrominance) with 10 bit depth (Little-Endian).
/// * `u_stride` - The stride (bytes per row) for the U plane.
/// * `v_plane` - A slice to load the V (chrominance) with 10 bit depth (Little-Endian).
/// * `v_stride` - The stride (bytes per row) for the U plane
/// * `width` - The width of the YUV image.
/// * `height` - The height of the YUV image.
/// * `rgba_data` - A mutable slice to store the converted RGBA data.
/// * `rgba_stride` - The stride (bytes per row) for RGBA data.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input RGBA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv422_p10_to_rgba(
    y_plane: &[u16],
    y_stride: u32,
    u_plane: &[u16],
    u_stride: u32,
    v_plane: &[u16],
    v_stride: u32,
    rgba: &mut [u8],
    rgba_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) {
    yuv_p10_to_rgbx_impl::<
        { YuvSourceChannels::Rgba as u8 },
        { YuvChromaSample::YUV422 as u8 },
        { YuvEndian::LittleEndian as u8 },
        { YuvBytesPosition::LeastSignificantBytes as u8 },
    >(
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

/// Convert YUV 420 planar format with 10-bit (Big-Endian) pixel format to RGBA format.
///
/// This function takes YUV 420 planar data with 10-bit precision stored in Big-Endian.
/// and converts it to RGBA format with 8-bit precision per channel
///
/// # Arguments
///
/// * `y_plane` -  A slice containing Y (luminance) with 10 bit depth (Big-Endian).
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `u_plane` - A slice to load the U (chrominance) with 10 bit depth (Big-Endian).
/// * `u_stride` - The stride (bytes per row) for the U plane.
/// * `v_plane` - A slice to load the V (chrominance) with 10 bit depth (Big-Endian).
/// * `v_stride` - The stride (bytes per row) for the U plane.
/// * `width` - The width of the YUV image.
/// * `height` - The height of the YUV image.
/// * `rgba_data` - A mutable slice to store the converted RGBA data.
/// * `rgba_stride` - The stride (bytes per row) for RGBA data.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input RGBA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv420_p10_to_rgba_be(
    y_plane: &[u16],
    y_stride: u32,
    u_plane: &[u16],
    u_stride: u32,
    v_plane: &[u16],
    v_stride: u32,
    rgba: &mut [u8],
    rgba_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) {
    yuv_p10_to_rgbx_impl::<
        { YuvSourceChannels::Rgba as u8 },
        { YuvChromaSample::YUV420 as u8 },
        { YuvEndian::BigEndian as u8 },
        { YuvBytesPosition::LeastSignificantBytes as u8 },
    >(
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

/// Convert YUV 422 format with 10-bit (Big-Endian) pixel format to RGBA format .
///
/// This function takes YUV 422 data with 10-bit precision stored in Big-Endian.
/// and converts it to RGBA format with 8-bit precision per channel.
///
/// # Arguments
///
/// * `y_plane` -  A slice containing Y (luminance) with 10 bit depth (Big-Endian).
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `u_plane` - A slice to load the U (chrominance) with 10 bit depth (Big-Endian).
/// * `u_stride` - The stride (bytes per row) for the U plane.
/// * `v_plane` - A slice to load the V (chrominance) with 10 bit depth (Big-Endian).
/// * `v_stride` - The stride (bytes per row) for the U plane
/// * `width` - The width of the YUV image.
/// * `height` - The height of the YUV image.
/// * `rgba_data` - A mutable slice to store the converted RGBA data.
/// * `rgba_stride` - The stride (bytes per row) for RGBA data.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input RGBA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv422_p10_to_rgba_be(
    y_plane: &[u16],
    y_stride: u32,
    u_plane: &[u16],
    u_stride: u32,
    v_plane: &[u16],
    v_stride: u32,
    rgba: &mut [u8],
    rgba_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) {
    yuv_p10_to_rgbx_impl::<
        { YuvSourceChannels::Rgba as u8 },
        { YuvChromaSample::YUV422 as u8 },
        { YuvEndian::BigEndian as u8 },
        { YuvBytesPosition::LeastSignificantBytes as u8 },
    >(
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

/// Convert YUV 444 planar format with 10-bit (Little-Endian) pixel format to RGBA format.
///
/// This function takes YUV 444 planar data with 10-bit precision stored in Little-Endian.
/// and converts it to RGBA format with 8-bit precision per channel
///
/// # Arguments
///
/// * `y_plane` -  A slice containing Y (luminance) with 10 bit depth (Little-Endian).
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `u_plane` - A slice to load the U (chrominance) with 10 bit depth (Little-Endian).
/// * `u_stride` - The stride (bytes per row) for the U plane.
/// * `v_plane` - A slice to load the V (chrominance) with 10 bit depth (Little-Endian).
/// * `v_stride` - The stride (bytes per row) for the U plane.
/// * `width` - The width of the YUV image.
/// * `height` - The height of the YUV image.
/// * `rgba_data` - A mutable slice to store the converted RGBA data.
/// * `rgba_stride` - The stride (bytes per row) for RGBA data.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input RGBA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv444_p10_to_rgba(
    y_plane: &[u16],
    y_stride: u32,
    u_plane: &[u16],
    u_stride: u32,
    v_plane: &[u16],
    v_stride: u32,
    rgba: &mut [u8],
    rgba_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) {
    yuv_p10_to_rgbx_impl::<
        { YuvSourceChannels::Rgba as u8 },
        { YuvChromaSample::YUV444 as u8 },
        { YuvEndian::LittleEndian as u8 },
        { YuvBytesPosition::LeastSignificantBytes as u8 },
    >(
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

/// Convert YUV 444 format with 10-bit (Big-Endian) pixel format to RGBA format .
///
/// This function takes YUV 444 data with 10-bit precision stored in Big-Endian.
/// and converts it to RGBA format with 8-bit precision per channel.
///
/// # Arguments
///
/// * `y_plane` -  A slice containing Y (luminance) with 10 bit depth (Big-Endian).
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `u_plane` - A slice to load the U (chrominance) with 10 bit depth (Big-Endian).
/// * `u_stride` - The stride (bytes per row) for the U plane.
/// * `v_plane` - A slice to load the V (chrominance) with 10 bit depth (Big-Endian).
/// * `v_stride` - The stride (bytes per row) for the U plane
/// * `width` - The width of the YUV image.
/// * `height` - The height of the YUV image.
/// * `rgba_data` - A mutable slice to store the converted RGBA data.
/// * `rgba_stride` - The stride (bytes per row) for RGBA data.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input RGBA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv444_p10_to_rgba_be(
    y_plane: &[u16],
    y_stride: u32,
    u_plane: &[u16],
    u_stride: u32,
    v_plane: &[u16],
    v_stride: u32,
    rgba: &mut [u8],
    rgba_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) {
    yuv_p10_to_rgbx_impl::<
        { YuvSourceChannels::Rgba as u8 },
        { YuvChromaSample::YUV444 as u8 },
        { YuvEndian::BigEndian as u8 },
        { YuvBytesPosition::LeastSignificantBytes as u8 },
    >(
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

/// Convert YUV 444 planar format with 10-bit (Little-Endian) pixel format to BGRA format.
///
/// This function takes YUV 444 planar data with 10-bit precision stored in Little-Endian.
/// and converts it to BGRA format with 8-bit precision per channel
///
/// # Arguments
///
/// * `y_plane` -  A slice containing Y (luminance) with 10 bit depth (Little-Endian).
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `u_plane` - A slice to load the U (chrominance) with 10 bit depth (Little-Endian).
/// * `u_stride` - The stride (bytes per row) for the U plane.
/// * `v_plane` - A slice to load the V (chrominance) with 10 bit depth (Little-Endian).
/// * `v_stride` - The stride (bytes per row) for the U plane.
/// * `width` - The width of the YUV image.
/// * `height` - The height of the YUV image.
/// * `rgba_data` - A mutable slice to store the converted BGRA data.
/// * `rgba_stride` - The stride (bytes per row) for BGRA data.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGRA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv444_p10_to_bgra(
    y_plane: &[u16],
    y_stride: u32,
    u_plane: &[u16],
    u_stride: u32,
    v_plane: &[u16],
    v_stride: u32,
    rgba: &mut [u8],
    rgba_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) {
    yuv_p10_to_rgbx_impl::<
        { YuvSourceChannels::Bgra as u8 },
        { YuvChromaSample::YUV444 as u8 },
        { YuvEndian::LittleEndian as u8 },
        { YuvBytesPosition::LeastSignificantBytes as u8 },
    >(
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

/// Convert YUV 444 format with 10-bit (Big-Endian) pixel format to BGRA format .
///
/// This function takes YUV 444 data with 10-bit precision stored in Big-Endian.
/// and converts it to BGRA format with 8-bit precision per channel.
///
/// # Arguments
///
/// * `y_plane` -  A slice containing Y (luminance) with 10 bit depth (Big-Endian).
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `u_plane` - A slice to load the U (chrominance) with 10 bit depth (Big-Endian).
/// * `u_stride` - The stride (bytes per row) for the U plane.
/// * `v_plane` - A slice to load the V (chrominance) with 10 bit depth (Big-Endian).
/// * `v_stride` - The stride (bytes per row) for the U plane
/// * `width` - The width of the YUV image.
/// * `height` - The height of the YUV image.
/// * `rgba_data` - A mutable slice to store the converted BGRA data.
/// * `rgba_stride` - The stride (bytes per row) for BGRA data.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGRA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv444_p10_to_bgra_be(
    y_plane: &[u16],
    y_stride: u32,
    u_plane: &[u16],
    u_stride: u32,
    v_plane: &[u16],
    v_stride: u32,
    rgba: &mut [u8],
    rgba_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) {
    yuv_p10_to_rgbx_impl::<
        { YuvSourceChannels::Bgra as u8 },
        { YuvChromaSample::YUV444 as u8 },
        { YuvEndian::BigEndian as u8 },
        { YuvBytesPosition::LeastSignificantBytes as u8 },
    >(
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
