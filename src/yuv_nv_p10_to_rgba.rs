#[cfg(target_arch = "aarch64")]
#[cfg(target_feature = "neon")]
use std::arch::aarch64::*;
use std::slice;

use crate::yuv_support::{
    get_inverse_transform, get_kr_kb, get_yuv_range, YuvBytesPosition, YuvChromaSample, YuvEndian,
    YuvNVOrder, YuvRange, YuvSourceChannels, YuvStandardMatrix,
};

fn yuv_nv12_p10_to_bgra_impl<
    const DESTINATION_CHANNELS: u8,
    const NV_ORDER: u8,
    const SAMPLING: u8,
    const ENDIANNESS: u8,
    const BYTES_POSITION: u8,
>(
    y_plane: &[u16],
    y_stride: u32,
    uv_plane: &[u16],
    uv_stride: u32,
    bgra: &mut [u8],
    bgra_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) {
    let destination_channels: YuvSourceChannels = DESTINATION_CHANNELS.into();
    let channels = destination_channels.get_channels_count();
    let uv_order: YuvNVOrder = NV_ORDER.into();
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
    let mut uv_offset = 0usize;
    let mut dst_offset = 0usize;

    let y_src_ptr = y_plane.as_ptr() as *const u8;
    let uv_src_ptr = uv_plane.as_ptr() as *const u8;

    let iterator_step = match chroma_subsampling {
        YuvChromaSample::YUV420 => 2usize,
        YuvChromaSample::YUV422 => 2usize,
        YuvChromaSample::YUV444 => 1usize,
    };

    for y in 0..height as usize {
        #[allow(unused_variables)]
        #[allow(unused_mut)]
        let mut cx = 0usize;

        #[allow(unused_variables)]
        #[allow(unused_mut)]
        let mut ux = 0usize;

        let y_ld_ptr = unsafe { y_src_ptr.offset(y_offset as isize) as *const u16 };
        let y_ld = unsafe { slice::from_raw_parts(y_ld_ptr, width as usize) };
        let uv_ld_ptr = unsafe { uv_src_ptr.offset(uv_offset as isize) as *const u16 };
        let uv_length = if chroma_subsampling == YuvChromaSample::YUV444 {
            width as usize * 2usize
        } else {
            width as usize
        };
        let uv_ld = unsafe { slice::from_raw_parts(uv_ld_ptr, uv_length) };

        #[cfg(target_arch = "aarch64")]
        #[cfg(target_feature = "neon")]
        unsafe {
            let dst_ptr = bgra.as_mut_ptr();

            let y_corr = vdupq_n_s16(bias_y as i16);
            let uv_corr = vdup_n_s16(bias_uv as i16);
            let uv_corr_q = vdupq_n_s16(bias_uv as i16);
            let v_luma_coeff = vdupq_n_s16(y_coef as i16);
            let v_luma_coeff_4 = vdup_n_s16(y_coef as i16);
            let v_cr_coeff = vdup_n_s16(cr_coef as i16);
            let v_cb_coeff = vdup_n_s16(cb_coef as i16);
            let v_min_values = vdupq_n_s16(0i16);
            let v_g_coeff_1 = vdup_n_s16(-1i16 * (g_coef_1 as i16));
            let v_g_coeff_2 = vdup_n_s16(-1i16 * (g_coef_2 as i16));
            let v_alpha = vdup_n_u8(255u8);

            while cx + 8 < width as usize {
                let y_values: int16x8_t;

                let u_high: int16x4_t;
                let v_high: int16x4_t;
                let u_low: int16x4_t;
                let v_low: int16x4_t;

                let mut y_vl = vld1q_u16(y_ld_ptr.add(cx));
                if endianness == YuvEndian::BigEndian {
                    y_vl = vreinterpretq_u16_u8(vrev16q_u8(vreinterpretq_u8_u16(y_vl)));
                }
                if bytes_position == YuvBytesPosition::MostSignificantBytes {
                    y_vl = vshrq_n_u16::<6>(y_vl);
                }
                y_values = vsubq_s16(vreinterpretq_s16_u16(y_vl), y_corr);

                match chroma_subsampling {
                    YuvChromaSample::YUV420 | YuvChromaSample::YUV422 => {
                        let mut uv_values_u = vld2_u16(uv_ld_ptr.add(ux));

                        if uv_order == YuvNVOrder::VU {
                            uv_values_u = uint16x4x2_t(uv_values_u.1, uv_values_u.0);
                        }

                        let mut u_vl = uv_values_u.0;
                        if endianness == YuvEndian::BigEndian {
                            u_vl = vreinterpret_u16_u8(vrev16_u8(vreinterpret_u8_u16(u_vl)));
                        }
                        let mut v_vl = uv_values_u.1;
                        if endianness == YuvEndian::BigEndian {
                            v_vl = vreinterpret_u16_u8(vrev16_u8(vreinterpret_u8_u16(v_vl)));
                        }
                        if bytes_position == YuvBytesPosition::MostSignificantBytes {
                            u_vl = vshr_n_u16::<6>(u_vl);
                            v_vl = vshr_n_u16::<6>(v_vl);
                        }
                        let u_values_c = vsub_s16(vreinterpret_s16_u16(u_vl), uv_corr);
                        let v_values_c = vsub_s16(vreinterpret_s16_u16(v_vl), uv_corr);

                        u_high = vzip2_s16(u_values_c, u_values_c);
                        v_high = vzip2_s16(v_values_c, v_values_c);
                        u_low = vzip1_s16(u_values_c, u_values_c);
                        v_low = vzip1_s16(v_values_c, v_values_c);
                    }
                    YuvChromaSample::YUV444 => {
                        let mut uv_values_u = vld2q_u16(uv_ld_ptr.add(ux));

                        if uv_order == YuvNVOrder::VU {
                            uv_values_u = uint16x8x2_t(uv_values_u.1, uv_values_u.0);
                        }
                        let mut u_vl = uv_values_u.0;
                        if endianness == YuvEndian::BigEndian {
                            u_vl = vreinterpretq_u16_u8(vrev16q_u8(vreinterpretq_u8_u16(u_vl)));
                        }
                        let mut v_vl = uv_values_u.1;
                        if endianness == YuvEndian::BigEndian {
                            v_vl = vreinterpretq_u16_u8(vrev16q_u8(vreinterpretq_u8_u16(v_vl)));
                        }
                        if bytes_position == YuvBytesPosition::MostSignificantBytes {
                            u_vl = vshrq_n_u16::<6>(u_vl);
                            v_vl = vshrq_n_u16::<6>(v_vl);
                        }
                        let u_values_c = vsubq_s16(vreinterpretq_s16_u16(u_vl), uv_corr_q);
                        let v_values_c = vsubq_s16(vreinterpretq_s16_u16(v_vl), uv_corr_q);
                        u_high = vget_high_s16(u_values_c);
                        v_high = vget_high_s16(v_values_c);
                        u_low = vget_low_s16(u_values_c);
                        v_low = vget_low_s16(v_values_c);
                    }
                }

                let y_high = vmull_high_s16(y_values, v_luma_coeff);

                let r_high = vshrn_n_s32::<6>(vmlal_s16(y_high, v_high, v_cr_coeff));
                let b_high = vshrn_n_s32::<6>(vmlal_s16(y_high, u_high, v_cb_coeff));
                let g_high = vshrn_n_s32::<6>(vmlal_s16(
                    vmlal_s16(y_high, v_high, v_g_coeff_1),
                    u_high,
                    v_g_coeff_2,
                ));

                let y_low = vmull_s16(vget_low_s16(y_values), v_luma_coeff_4);

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
                        vst3_u8(dst_ptr.add(dst_offset + cx * channels), dst_pack);
                    }
                    YuvSourceChannels::Rgba => {
                        let dst_pack: uint8x8x4_t =
                            uint8x8x4_t(r_values, g_values, b_values, v_alpha);
                        vst4_u8(dst_ptr.add(dst_offset + cx * channels), dst_pack);
                    }
                    YuvSourceChannels::Bgra => {
                        let dst_pack: uint8x8x4_t =
                            uint8x8x4_t(b_values, g_values, r_values, v_alpha);
                        vst4_u8(dst_ptr.add(dst_offset + cx * channels), dst_pack);
                    }
                }

                cx += 8;

                match chroma_subsampling {
                    YuvChromaSample::YUV420 | YuvChromaSample::YUV422 => {
                        ux += 8;
                    }
                    YuvChromaSample::YUV444 => {
                        ux += 16;
                    }
                }
            }
        }

        for x in (cx..width as usize).step_by(iterator_step) {
            let y_value: i32;
            let mut cb_value: i32;
            let mut cr_value: i32;
            match endianness {
                YuvEndian::BigEndian => {
                    let mut y_vl = u16::from_be(y_ld[x]) as i32;
                    let mut cb_vl = u16::from_be(uv_ld[ux]) as i32;
                    let mut cr_vl = u16::from_be(uv_ld[ux + 1]) as i32;
                    if bytes_position == YuvBytesPosition::MostSignificantBytes {
                        y_vl = y_vl >> 6;
                        cb_vl = cb_vl >> 6;
                        cr_vl = cr_vl >> 6;
                    }
                    y_value = (y_vl - bias_y) * y_coef;

                    cb_value = cb_vl;
                    cr_value = cr_vl;
                }
                YuvEndian::LittleEndian => {
                    let mut y_vl = u16::from_le(y_ld[x]) as i32;
                    let mut cb_vl = u16::from_le(uv_ld[ux]) as i32;
                    let mut cr_vl = u16::from_le(uv_ld[ux + 1]) as i32;
                    if bytes_position == YuvBytesPosition::MostSignificantBytes {
                        y_vl = y_vl >> 6;
                        cb_vl = cb_vl >> 6;
                        cr_vl = cr_vl >> 6;
                    }
                    y_value = (y_vl - bias_y) * y_coef;

                    cb_value = cb_vl;
                    cr_value = cr_vl;
                }
            }

            match uv_order {
                YuvNVOrder::UV => {
                    cb_value = cb_value - bias_uv;
                    cr_value = cr_value - bias_uv;
                }
                YuvNVOrder::VU => {
                    cr_value = cb_value - bias_uv;
                    cb_value = cr_value - bias_uv;
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

            bgra[rgb_offset + destination_channels.get_b_channel_offset()] = b as u8;
            bgra[rgb_offset + destination_channels.get_g_channel_offset()] = g as u8;
            bgra[rgb_offset + destination_channels.get_r_channel_offset()] = r as u8;
            if destination_channels.has_alpha() {
                bgra[rgb_offset + destination_channels.get_a_channel_offset()] = 255;
            }

            if chroma_subsampling == YuvChromaSample::YUV422
                || chroma_subsampling == YuvChromaSample::YUV420
            {
                let next_px = x + 1;
                if next_px < width as usize {
                    let y_value: i32;
                    match endianness {
                        YuvEndian::BigEndian => {
                            let mut y_vl = u16::from_be(y_ld[next_px]) as i32;
                            if bytes_position == YuvBytesPosition::MostSignificantBytes {
                                y_vl = y_vl >> 6;
                            }
                            y_value = (y_vl - bias_y) * y_coef;
                        }
                        YuvEndian::LittleEndian => {
                            let mut y_vl = u16::from_le(y_ld[next_px]) as i32;
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

                    let px = next_px * channels;
                    let rgb_offset = dst_offset + px;
                    bgra[rgb_offset + destination_channels.get_b_channel_offset()] = b as u8;
                    bgra[rgb_offset + destination_channels.get_g_channel_offset()] = g as u8;
                    bgra[rgb_offset + destination_channels.get_r_channel_offset()] = r as u8;
                    if destination_channels.has_alpha() {
                        bgra[rgb_offset + destination_channels.get_a_channel_offset()] = 255;
                    }
                }
            }

            ux += 2;
        }

        match chroma_subsampling {
            YuvChromaSample::YUV420 => {
                if y & 1 == 1 {
                    uv_offset += uv_stride as usize;
                }
            }
            YuvChromaSample::YUV422 | YuvChromaSample::YUV444 => {
                uv_offset += uv_stride as usize;
            }
        }

        dst_offset += bgra_stride as usize;
        y_offset += y_stride as usize;
    }
}

/// Convert YUV NV12 format with 10-bit (Little-Endian) pixel format to BGRA format.
///
/// This function takes YUV NV16 data with 10-bit precision stored in Little-Endian.
/// and converts it to BGRA format with 8-bit precision.
///
/// # Arguments
///
/// * `y_plane` -  A slice containing Y (luminance) with 10 bit depth (Little-Endian).
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `uv_plane` - A slice to load the UV (chrominance) with 10 bit depth (Little-Endian).
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
pub fn yuv_nv12_p10_to_bgra(
    y_plane: &[u16],
    y_stride: u32,
    uv_plane: &[u16],
    uv_stride: u32,
    bgra: &mut [u8],
    bgra_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) {
    yuv_nv12_p10_to_bgra_impl::<
        { YuvSourceChannels::Bgra as u8 },
        { YuvNVOrder::UV as u8 },
        { YuvChromaSample::YUV420 as u8 },
        { YuvEndian::LittleEndian as u8 },
        { YuvBytesPosition::LeastSignificantBytes as u8 },
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

/// Convert YUV NV16 format with 10-bit (Little-Endian) pixel format to BGRA format .
///
/// This function takes YUV NV16 data with 10-bit precision stored in Little-Endian.
/// and converts it to BGRA format with 8-bit precision.
///
/// # Arguments
///
/// * `y_plane` -  A slice containing Y (luminance) with 10 bit depth (Little-Endian).
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `uv_plane` - A slice to load the UV (chrominance) with 10 bit depth (Little-Endian).
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
pub fn yuv_nv16_p10_to_bgra(
    y_plane: &[u16],
    y_stride: u32,
    uv_plane: &[u16],
    uv_stride: u32,
    bgra: &mut [u8],
    bgra_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) {
    yuv_nv12_p10_to_bgra_impl::<
        { YuvSourceChannels::Bgra as u8 },
        { YuvNVOrder::UV as u8 },
        { YuvChromaSample::YUV422 as u8 },
        { YuvEndian::LittleEndian as u8 },
        { YuvBytesPosition::LeastSignificantBytes as u8 },
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

/// Convert YUV NV12 format with 10-bit (Big-Endian) pixel format to BGRA format.
///
/// This function takes YUV NV16 data with 10-bit precision stored in Big-Endian.
/// and converts it to BGRA format with 8-bit precision.
///
/// # Arguments
///
/// * `y_plane` -  A slice containing Y (luminance) with 10 bit depth (Big-Endian).
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `uv_plane` - A slice to load the UV (chrominance) with 10 bit depth (Big-Endian).
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
pub fn yuv_nv12_p10_be_to_bgra(
    y_plane: &[u16],
    y_stride: u32,
    uv_plane: &[u16],
    uv_stride: u32,
    bgra: &mut [u8],
    bgra_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) {
    yuv_nv12_p10_to_bgra_impl::<
        { YuvSourceChannels::Bgra as u8 },
        { YuvNVOrder::UV as u8 },
        { YuvChromaSample::YUV420 as u8 },
        { YuvEndian::BigEndian as u8 },
        { YuvBytesPosition::LeastSignificantBytes as u8 },
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

/// Convert YUV NV16 format with 10-bit (Big-Endian) pixel format to BGRA format.
///
/// This function takes YUV NV16 data with 10-bit precision stored in Big-Endian.
/// and converts it to BGRA format with 8-bit precision.
///
/// # Arguments
///
/// * `y_plane` -  A slice containing Y (luminance) with 10 bit depth (Big-Endian).
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `uv_plane` - A slice to load the UV (chrominance) with 10 bit depth (Big-Endian).
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
pub fn yuv_nv16_p10_be_to_bgra(
    y_plane: &[u16],
    y_stride: u32,
    uv_plane: &[u16],
    uv_stride: u32,
    bgra: &mut [u8],
    bgra_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) {
    yuv_nv12_p10_to_bgra_impl::<
        { YuvSourceChannels::Bgra as u8 },
        { YuvNVOrder::UV as u8 },
        { YuvChromaSample::YUV422 as u8 },
        { YuvEndian::BigEndian as u8 },
        { YuvBytesPosition::LeastSignificantBytes as u8 },
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

/// Convert YUV NV12 format with 10-bit pixel format (MSB) to BGRA format.
///
/// This function takes YUV NV16 data with 10-bit precision and MSB ordering,
/// and converts it to BGRA format with 8-bit precision.
/// This format is used by Apple and corresponds to kCVPixelFormatType_420YpCbCr10BiPlanarFullRange/kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange
///
/// # Arguments
///
/// * `y_plane` -  A slice containing Y (luminance) with 10 bit depth stored in Most Significant Bytes of u16.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `uv_plane` - A slice to load the UV (chrominance) with 10 bit depth stored in Most Significant Bytes of u16.
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
pub fn yuv_nv12_p10_msb_to_bgra(
    y_plane: &[u16],
    y_stride: u32,
    uv_plane: &[u16],
    uv_stride: u32,
    bgra: &mut [u8],
    bgra_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) {
    yuv_nv12_p10_to_bgra_impl::<
        { YuvSourceChannels::Bgra as u8 },
        { YuvNVOrder::UV as u8 },
        { YuvChromaSample::YUV420 as u8 },
        { YuvEndian::LittleEndian as u8 },
        { YuvBytesPosition::MostSignificantBytes as u8 },
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

/// Convert YUV NV16 format with 10-bit pixel format (MSB) to BGRA format.
///
/// This function takes YUV NV16 data with 10-bit precision and MSB ordering,
/// and converts it to BGRA format with 8-bit precision.
/// This format is used by Apple and corresponds to kCVPixelFormatType_422YpCbCr10BiPlanarFullRange/kCVPixelFormatType_422YpCbCr10BiPlanarVideoRange
///
/// # Arguments
///
/// * `y_plane` -  A slice containing Y (luminance) with 10 bit depth stored in Most Significant Bytes of u16.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `uv_plane` - A slice to load the UV (chrominance) with 10 bit depth stored in Most Significant Bytes of u16.
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
pub fn yuv_nv16_p10_msb_to_bgra(
    y_plane: &[u16],
    y_stride: u32,
    uv_plane: &[u16],
    uv_stride: u32,
    bgra: &mut [u8],
    bgra_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) {
    yuv_nv12_p10_to_bgra_impl::<
        { YuvSourceChannels::Bgra as u8 },
        { YuvNVOrder::UV as u8 },
        { YuvChromaSample::YUV422 as u8 },
        { YuvEndian::LittleEndian as u8 },
        { YuvBytesPosition::MostSignificantBytes as u8 },
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


/// Convert YUV NV12 format with 10-bit pixel format (MSB) to RGBA format.
///
/// This function takes YUV NV16 data with 10-bit precision and MSB ordering,
/// and converts it to RGBA format with 8-bit precision.
/// This format is used by Apple and corresponds to kCVPixelFormatType_420YpCbCr10BiPlanarFullRange/kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange
///
/// # Arguments
///
/// * `y_plane` -  A slice containing Y (luminance) with 10 bit depth stored in Most Significant Bytes of u16.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `uv_plane` - A slice to load the UV (chrominance) with 10 bit depth stored in Most Significant Bytes of u16.
/// * `uv_stride` - The stride (bytes per row) for the UV plane.
/// * `width` - The width of the YUV image.
/// * `height` - The height of the YUV image.
/// * `bgra_data` - A mutable slice to store the converted RGBA data.
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input RGBA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv_nv12_p10_msb_to_rgba(
    y_plane: &[u16],
    y_stride: u32,
    uv_plane: &[u16],
    uv_stride: u32,
    bgra: &mut [u8],
    bgra_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) {
    yuv_nv12_p10_to_bgra_impl::<
        { YuvSourceChannels::Rgba as u8 },
        { YuvNVOrder::UV as u8 },
        { YuvChromaSample::YUV420 as u8 },
        { YuvEndian::LittleEndian as u8 },
        { YuvBytesPosition::MostSignificantBytes as u8 },
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

/// Convert YUV NV16 format with 10-bit pixel format (MSB) to RGBA format.
///
/// This function takes YUV NV16 data with 10-bit precision and MSB ordering,
/// and converts it to RGBA format with 8-bit precision.
/// This format is used by Apple and corresponds to kCVPixelFormatType_422YpCbCr10BiPlanarFullRange/kCVPixelFormatType_422YpCbCr10BiPlanarVideoRange
///
/// # Arguments
///
/// * `y_plane` -  A slice containing Y (luminance) with 10 bit depth stored in Most Significant Bytes of u16.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `uv_plane` - A slice to load the UV (chrominance) with 10 bit depth stored in Most Significant Bytes of u16.
/// * `uv_stride` - The stride (bytes per row) for the UV plane.
/// * `width` - The width of the YUV image.
/// * `height` - The height of the YUV image.
/// * `bgra_data` - A mutable slice to store the converted RGBA data.
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input RGBA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv_nv16_p10_msb_to_rgba(
    y_plane: &[u16],
    y_stride: u32,
    uv_plane: &[u16],
    uv_stride: u32,
    bgra: &mut [u8],
    bgra_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) {
    yuv_nv12_p10_to_bgra_impl::<
        { YuvSourceChannels::Rgba as u8 },
        { YuvNVOrder::UV as u8 },
        { YuvChromaSample::YUV422 as u8 },
        { YuvEndian::LittleEndian as u8 },
        { YuvBytesPosition::MostSignificantBytes as u8 },
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
