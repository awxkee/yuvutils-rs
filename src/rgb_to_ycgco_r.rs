use crate::ycgcor_support::YCgCoR;
use crate::yuv_support::{get_yuv_range, YuvChromaSample, YuvSourceChannels};
use crate::YuvRange;

fn rgbx_to_ycgco_type_r<const ORIGIN_CHANNELS: u8, const SAMPLING: u8, const R_TYPE: usize>(
    y_plane: &mut [u16],
    y_stride: u32,
    cg_plane: &mut [u16],
    cg_stride: u32,
    co_plane: &mut [u16],
    co_stride: u32,
    rgba: &[u8],
    rgba_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
) {
    let r_type: YCgCoR = R_TYPE.into();
    let y_bit_depth = match r_type {
        YCgCoR::YCgCoRo => 8 + 1,
        YCgCoR::YCgCoRe => 8 + 2,
    };
    let chroma_subsampling: YuvChromaSample = SAMPLING.into();
    let source_channels: YuvSourceChannels = ORIGIN_CHANNELS.into();
    let channels = source_channels.get_channels_count();
    let range = get_yuv_range(y_bit_depth, range);
    let precision_scale = (1 << 8) as f32;
    let bias_y = ((range.bias_y as f32 + 0.5f32) * precision_scale) as i32;
    let max_colors = 2i32.pow(8) - 1i32;

    let range_reduction =
        (range.range_y as f32 / max_colors as f32 * precision_scale).round() as i32;

    let iterator_step = match chroma_subsampling {
        YuvChromaSample::YUV420 => 2usize,
        YuvChromaSample::YUV422 => 2usize,
        YuvChromaSample::YUV444 => 1usize,
    };

    let mut y_offset = 0usize;
    let mut cg_offset = 0usize;
    let mut co_offset = 0usize;
    let mut rgba_offset = 0usize;

    for y in 0..height as usize {
        let mut _cx = 0usize;
        let mut _ux = 0usize;

        for x in (_cx..width as usize).step_by(iterator_step) {
            let px = x * channels;
            let rgba_shift = rgba_offset + px;
            let mut r =
                unsafe { *rgba.get_unchecked(rgba_shift + source_channels.get_r_channel_offset()) }
                    as i32;
            let mut g =
                unsafe { *rgba.get_unchecked(rgba_shift + source_channels.get_g_channel_offset()) }
                    as i32;
            let mut b =
                unsafe { *rgba.get_unchecked(rgba_shift + source_channels.get_b_channel_offset()) }
                    as i32;
            r *= range_reduction;
            g *= range_reduction;
            b *= range_reduction;
            let co = r - b;
            let t = b + (co >> 1);
            let cg = g - t;
            let y_0 = ((t + (cg >> 1)) + bias_y) >> 8;
            unsafe { *y_plane.get_unchecked_mut(y_offset + x) = y_0 as u16 };
            let u_pos = match chroma_subsampling {
                YuvChromaSample::YUV420 | YuvChromaSample::YUV422 => cg_offset + _ux,
                YuvChromaSample::YUV444 => cg_offset + _ux,
            };
            let corrected_cg = (cg + bias_y) >> 8;
            unsafe { *cg_plane.get_unchecked_mut(u_pos) = corrected_cg as u16 };
            let v_pos = match chroma_subsampling {
                YuvChromaSample::YUV420 | YuvChromaSample::YUV422 => co_offset + _ux,
                YuvChromaSample::YUV444 => co_offset + _ux,
            };
            let corrected_co = (co + bias_y) >> 8;
            unsafe { *co_plane.get_unchecked_mut(v_pos) = corrected_co as u16 };
            match chroma_subsampling {
                YuvChromaSample::YUV420 | YuvChromaSample::YUV422 => {
                    if x + 1 < width as usize {
                        let next_px = (x + 1) * channels;
                        let rgba_shift = rgba_offset + next_px;
                        let mut r = unsafe {
                            *rgba.get_unchecked(rgba_shift + source_channels.get_r_channel_offset())
                        } as i32;
                        let mut g = unsafe {
                            *rgba.get_unchecked(rgba_shift + source_channels.get_g_channel_offset())
                        } as i32;
                        let mut b = unsafe {
                            *rgba.get_unchecked(rgba_shift + source_channels.get_b_channel_offset())
                        } as i32;
                        r *= range_reduction;
                        g *= range_reduction;
                        b *= range_reduction;
                        let co = r - b;
                        let t = b + (co >> 1);
                        let cg = g - t;
                        let y_1 = ((t + (cg >> 1)) + bias_y) >> 8;
                        unsafe { *y_plane.get_unchecked_mut(y_offset + x + 1) = y_1 as u16 };
                    }
                }
                _ => {}
            }

            _ux += 1;
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
/// This function performs RGB to YCgCo-Ro conversion and stores the result in YUV422 planar format,
/// with separate planes for Y (luminance), Cg (chrominance), and Co (chrominance) components.
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
pub fn rgb_to_ycgco_ro422(
    y_plane: &mut [u16],
    y_stride: u32,
    cg_plane: &mut [u16],
    cg_stride: u32,
    co_plane: &mut [u16],
    co_stride: u32,
    rgb: &[u8],
    rgb_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
) {
    rgbx_to_ycgco_type_r::<
        { YuvSourceChannels::Rgb as u8 },
        { YuvChromaSample::YUV422 as u8 },
        { YCgCoR::YCgCoRo as usize },
    >(
        y_plane, y_stride, cg_plane, cg_stride, co_plane, co_stride, rgb, rgb_stride, width,
        height, range,
    );
}

/// Convert RGBA image data to YCgCo-Ro 422 planar format.
///
/// This function performs RGBA to YCgCo-Ro conversion and stores the result in YUV422 planar format,
/// with separate planes for Y (luminance), Cg (chrominance), and Co (chrominance) components.
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
pub fn rgba_to_ycgco_ro422(
    y_plane: &mut [u16],
    y_stride: u32,
    cg_plane: &mut [u16],
    cg_stride: u32,
    co_plane: &mut [u16],
    co_stride: u32,
    rgba: &[u8],
    rgba_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
) {
    rgbx_to_ycgco_type_r::<
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
    );
}

/// Convert BGRA image data to YCgCo-Ro 422 planar format.
///
/// This function performs BGRA to YCgCo-Ro conversion and stores the result in YUV422 planar format,
/// with separate planes for Y (luminance), Cg (chrominance), and Co (chrominance) components.
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
pub fn bgra_to_ycgco_ro422(
    y_plane: &mut [u16],
    y_stride: u32,
    cg_plane: &mut [u16],
    cg_stride: u32,
    co_plane: &mut [u16],
    co_stride: u32,
    bgra: &[u8],
    bgra_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
) {
    rgbx_to_ycgco_type_r::<
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
    );
}

/// Convert RGB image data to YCgCo-Ro 420 planar format.
///
/// This function performs RGB to YCgCo-Ro conversion and stores the result in YUV420 planar format,
/// with separate planes for Y (luminance), Cg (chrominance), and Co (chrominance) components.
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
pub fn rgb_to_ycgco_ro420(
    y_plane: &mut [u16],
    y_stride: u32,
    cg_plane: &mut [u16],
    cg_stride: u32,
    co_plane: &mut [u16],
    co_stride: u32,
    rgb: &[u8],
    rgb_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
) {
    rgbx_to_ycgco_type_r::<
        { YuvSourceChannels::Rgb as u8 },
        { YuvChromaSample::YUV420 as u8 },
        { YCgCoR::YCgCoRo as usize },
    >(
        y_plane, y_stride, cg_plane, cg_stride, co_plane, co_stride, rgb, rgb_stride, width,
        height, range,
    );
}

/// Convert RGBA image data to YCgCo-Ro 420 planar format.
///
/// This function performs RGBA to YCgCo-Ro conversion and stores the result in YUV420 planar format,
/// with separate planes for Y (luminance), Cg (chrominance), and Co (chrominance) components.
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
pub fn rgba_to_ycgco_ro420(
    y_plane: &mut [u16],
    y_stride: u32,
    cg_plane: &mut [u16],
    cg_stride: u32,
    co_plane: &mut [u16],
    co_stride: u32,
    rgba: &[u8],
    rgba_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
) {
    rgbx_to_ycgco_type_r::<
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
    );
}

/// Convert BGRA image data to YCgCo-Ro 420 planar format.
///
/// This function performs BGRA to YCgCo-Ro conversion and stores the result in YUV420 planar format,
/// with separate planes for Y (luminance), Cg (chrominance), and Co (chrominance) components.
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
pub fn bgra_to_ycgco_ro420(
    y_plane: &mut [u16],
    y_stride: u32,
    cg_plane: &mut [u16],
    cg_stride: u32,
    co_plane: &mut [u16],
    co_stride: u32,
    bgra: &[u8],
    bgra_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
) {
    rgbx_to_ycgco_type_r::<
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
    );
}

/// Convert RGB image data to YCgCo-Ro 444 planar format.
///
/// This function performs RGB to YCgCo-Ro conversion and stores the result in YUV444 planar format,
/// with separate planes for Y (luminance), Cg (chrominance), and Co (chrominance) components.
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
pub fn rgb_to_ycgco_ro444(
    y_plane: &mut [u16],
    y_stride: u32,
    cg_plane: &mut [u16],
    cg_stride: u32,
    co_plane: &mut [u16],
    co_stride: u32,
    rgb: &[u8],
    rgb_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
) {
    rgbx_to_ycgco_type_r::<
        { YuvSourceChannels::Rgb as u8 },
        { YuvChromaSample::YUV444 as u8 },
        { YCgCoR::YCgCoRo as usize },
    >(
        y_plane, y_stride, cg_plane, cg_stride, co_plane, co_stride, rgb, rgb_stride, width,
        height, range,
    );
}

/// Convert RGBA image data to YCgCo-Ro 444 planar format.
///
/// This function performs RGBA to YCgCo-Ro conversion and stores the result in YUV444 planar format,
/// with separate planes for Y (luminance), Cg (chrominance), and Co (chrominance) components.
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
pub fn rgba_to_ycgco_ro444(
    y_plane: &mut [u16],
    y_stride: u32,
    cg_plane: &mut [u16],
    cg_stride: u32,
    co_plane: &mut [u16],
    co_stride: u32,
    rgba: &[u8],
    rgba_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
) {
    rgbx_to_ycgco_type_r::<
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
    );
}

/// Convert BGRA image data to YCgCo-Ro 444 planar format.
///
/// This function performs BGRA to YCgCo-Ro conversion and stores the result in YUV444 planar format,
/// with separate planes for Y (luminance), Cg (chrominance), and Co (chrominance) components.
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
pub fn bgra_to_ycgco_ro444(
    y_plane: &mut [u16],
    y_stride: u32,
    cg_plane: &mut [u16],
    cg_stride: u32,
    co_plane: &mut [u16],
    co_stride: u32,
    bgra: &[u8],
    bgra_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
) {
    rgbx_to_ycgco_type_r::<
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
    );
}

/// Convert RGB image data to YCgCo-Re 422 planar format.
///
/// This function performs RGB to YCgCo-Re conversion and stores the result in YUV422 planar format,
/// with separate planes for Y (luminance), Cg (chrominance), and Co (chrominance) components.
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
pub fn rgb_to_ycgco_re422(
    y_plane: &mut [u16],
    y_stride: u32,
    cg_plane: &mut [u16],
    cg_stride: u32,
    co_plane: &mut [u16],
    co_stride: u32,
    rgb: &[u8],
    rgb_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
) {
    rgbx_to_ycgco_type_r::<
        { YuvSourceChannels::Rgb as u8 },
        { YuvChromaSample::YUV422 as u8 },
        { YCgCoR::YCgCoRe as usize },
    >(
        y_plane, y_stride, cg_plane, cg_stride, co_plane, co_stride, rgb, rgb_stride, width,
        height, range,
    );
}

/// Convert RGBA image data to YCgCo-Re 422 planar format.
///
/// This function performs RGBA to YCgCo-Re conversion and stores the result in YUV422 planar format,
/// with separate planes for Y (luminance), Cg (chrominance), and Co (chrominance) components.
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
pub fn rgba_to_ycgco_re422(
    y_plane: &mut [u16],
    y_stride: u32,
    cg_plane: &mut [u16],
    cg_stride: u32,
    co_plane: &mut [u16],
    co_stride: u32,
    rgba: &[u8],
    rgba_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
) {
    rgbx_to_ycgco_type_r::<
        { YuvSourceChannels::Rgba as u8 },
        { YuvChromaSample::YUV422 as u8 },
        { YCgCoR::YCgCoRe as usize },
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
    );
}

/// Convert BGRA image data to YCgCo-Re 422 planar format.
///
/// This function performs BGRA to YCgCo-Re conversion and stores the result in YUV422 planar format,
/// with separate planes for Y (luminance), Cg (chrominance), and Co (chrominance) components.
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
pub fn bgra_to_ycgco_re422(
    y_plane: &mut [u16],
    y_stride: u32,
    cg_plane: &mut [u16],
    cg_stride: u32,
    co_plane: &mut [u16],
    co_stride: u32,
    bgra: &[u8],
    bgra_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
) {
    rgbx_to_ycgco_type_r::<
        { YuvSourceChannels::Bgra as u8 },
        { YuvChromaSample::YUV422 as u8 },
        { YCgCoR::YCgCoRe as usize },
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
    );
}

/// Convert RGB image data to YCgCo-Re 420 planar format.
///
/// This function performs RGB to YCgCo-Re conversion and stores the result in YUV420 planar format,
/// with separate planes for Y (luminance), Cg (chrominance), and Co (chrominance) components.
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
pub fn rgb_to_ycgco_re420(
    y_plane: &mut [u16],
    y_stride: u32,
    cg_plane: &mut [u16],
    cg_stride: u32,
    co_plane: &mut [u16],
    co_stride: u32,
    rgb: &[u8],
    rgb_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
) {
    rgbx_to_ycgco_type_r::<
        { YuvSourceChannels::Rgb as u8 },
        { YuvChromaSample::YUV420 as u8 },
        { YCgCoR::YCgCoRe as usize },
    >(
        y_plane, y_stride, cg_plane, cg_stride, co_plane, co_stride, rgb, rgb_stride, width,
        height, range,
    );
}

/// Convert RGBA image data to YCgCo-Re 420 planar format.
///
/// This function performs RGBA to YCgCo-Re conversion and stores the result in YUV420 planar format,
/// with separate planes for Y (luminance), Cg (chrominance), and Co (chrominance) components.
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
pub fn rgba_to_ycgco_re420(
    y_plane: &mut [u16],
    y_stride: u32,
    cg_plane: &mut [u16],
    cg_stride: u32,
    co_plane: &mut [u16],
    co_stride: u32,
    rgba: &[u8],
    rgba_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
) {
    rgbx_to_ycgco_type_r::<
        { YuvSourceChannels::Rgba as u8 },
        { YuvChromaSample::YUV420 as u8 },
        { YCgCoR::YCgCoRe as usize },
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
    );
}

/// Convert BGRA image data to YCgCo-Re 420 planar format.
///
/// This function performs BGRA to YCgCo-Re conversion and stores the result in YUV420 planar format,
/// with separate planes for Y (luminance), Cg (chrominance), and Co (chrominance) components.
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
pub fn bgra_to_ycgco_re420(
    y_plane: &mut [u16],
    y_stride: u32,
    cg_plane: &mut [u16],
    cg_stride: u32,
    co_plane: &mut [u16],
    co_stride: u32,
    bgra: &[u8],
    bgra_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
) {
    rgbx_to_ycgco_type_r::<
        { YuvSourceChannels::Bgra as u8 },
        { YuvChromaSample::YUV420 as u8 },
        { YCgCoR::YCgCoRe as usize },
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
    );
}

/// Convert RGB image data to YCgCo-Re 444 planar format.
///
/// This function performs RGB to YCgCo-Re conversion and stores the result in YUV444 planar format,
/// with separate planes for Y (luminance), Cg (chrominance), and Co (chrominance) components.
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
pub fn rgb_to_ycgco_re444(
    y_plane: &mut [u16],
    y_stride: u32,
    cg_plane: &mut [u16],
    cg_stride: u32,
    co_plane: &mut [u16],
    co_stride: u32,
    rgb: &[u8],
    rgb_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
) {
    rgbx_to_ycgco_type_r::<
        { YuvSourceChannels::Rgb as u8 },
        { YuvChromaSample::YUV444 as u8 },
        { YCgCoR::YCgCoRe as usize },
    >(
        y_plane, y_stride, cg_plane, cg_stride, co_plane, co_stride, rgb, rgb_stride, width,
        height, range,
    );
}

/// Convert RGBA image data to YCgCo-Re 444 planar format.
///
/// This function performs RGBA to YCgCo-Re conversion and stores the result in YUV444 planar format,
/// with separate planes for Y (luminance), Cg (chrominance), and Co (chrominance) components.
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
pub fn rgba_to_ycgco_re444(
    y_plane: &mut [u16],
    y_stride: u32,
    cg_plane: &mut [u16],
    cg_stride: u32,
    co_plane: &mut [u16],
    co_stride: u32,
    rgba: &[u8],
    rgba_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
) {
    rgbx_to_ycgco_type_r::<
        { YuvSourceChannels::Rgba as u8 },
        { YuvChromaSample::YUV444 as u8 },
        { YCgCoR::YCgCoRe as usize },
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
    );
}

/// Convert BGRA image data to YCgCo-Re 444 planar format.
///
/// This function performs BGRA to YCgCo-Re conversion and stores the result in YUV444 planar format,
/// with separate planes for Y (luminance), Cg (chrominance), and Co (chrominance) components.
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
pub fn bgra_to_ycgco_re444(
    y_plane: &mut [u16],
    y_stride: u32,
    cg_plane: &mut [u16],
    cg_stride: u32,
    co_plane: &mut [u16],
    co_stride: u32,
    bgra: &[u8],
    bgra_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
) {
    rgbx_to_ycgco_type_r::<
        { YuvSourceChannels::Bgra as u8 },
        { YuvChromaSample::YUV444 as u8 },
        { YCgCoR::YCgCoRe as usize },
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
    );
}
