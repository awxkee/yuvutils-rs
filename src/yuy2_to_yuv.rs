/*
 * // Copyright (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use crate::avx2::yuy2_to_yuv_avx;
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use crate::neon::yuy2_to_yuv_neon_impl;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use crate::sse::yuy2_to_yuv_sse_impl;
use crate::yuv_support::{YuvChromaSample, Yuy2Description};
#[allow(unused_imports)]
use crate::yuv_to_yuy2::YuvToYuy2Navigation;

fn yuy2_to_yuv_impl<const SAMPLING: u8, const YUY2_TARGET: usize>(
    y_plane: &mut [u8],
    y_stride: u32,
    u_plane: &mut [u8],
    u_stride: u32,
    v_plane: &mut [u8],
    v_stride: u32,
    yuy2_store: &[u8],
    yuy2_stride: u32,
    width: u32,
    height: u32,
) {
    let yuy2_target: Yuy2Description = YUY2_TARGET.into();
    let chroma_subsampling: YuvChromaSample = SAMPLING.into();

    let mut y_offset = 0usize;
    let mut u_offset = 0usize;
    let mut v_offset = 0usize;
    let mut yuy_offset = 0usize;

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    let mut _use_sse = std::arch::is_x86_feature_detected!("sse4.1");
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    let mut _use_avx2 = std::arch::is_x86_feature_detected!("avx2");

    for y in 0..height as usize {
        let mut _cx = 0usize;
        let mut _uv_x = 0usize;
        let mut _yuy2_x = 0usize;

        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        {
            let processed = yuy2_to_yuv_neon_impl::<SAMPLING, YUY2_TARGET>(
                y_plane,
                y_offset,
                u_plane,
                u_offset,
                v_plane,
                v_offset,
                yuy2_store,
                yuy_offset,
                width,
                YuvToYuy2Navigation::new(_cx, _uv_x, _yuy2_x),
            );
            _cx = processed.cx;
            _uv_x = processed.uv_x;
            _yuy2_x = processed.x;
        }

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        unsafe {
            if _use_avx2 {
                let processed = yuy2_to_yuv_avx::<SAMPLING, YUY2_TARGET>(
                    y_plane,
                    y_offset,
                    u_plane,
                    u_offset,
                    v_plane,
                    v_offset,
                    yuy2_store,
                    yuy_offset,
                    width,
                    YuvToYuy2Navigation::new(_cx, _uv_x, _yuy2_x),
                );
                _cx = processed.cx;
                _uv_x = processed.uv_x;
                _yuy2_x = processed.x;
            }
            if _use_sse {
                let processed = yuy2_to_yuv_sse_impl::<SAMPLING, YUY2_TARGET>(
                    y_plane,
                    y_offset,
                    u_plane,
                    u_offset,
                    v_plane,
                    v_offset,
                    yuy2_store,
                    yuy_offset,
                    width,
                    YuvToYuy2Navigation::new(_cx, _uv_x, _yuy2_x),
                );
                _cx = processed.cx;
                _uv_x = processed.uv_x;
                _yuy2_x = processed.x;
            }
        }

        for x in _yuy2_x..width.saturating_sub(1) as usize / 2 {
            let u_pos = u_offset + _uv_x;
            let v_pos = v_offset + _uv_x;
            let y_pos = y_offset + _cx;
            let dst_offset = yuy_offset + x * 4;

            let yuy2_plane_shifted = unsafe { yuy2_store.get_unchecked(dst_offset..) };

            let first_y_position =
                unsafe { *yuy2_plane_shifted.get_unchecked(yuy2_target.get_first_y_position()) };
            let second_y_position =
                unsafe { *yuy2_plane_shifted.get_unchecked(yuy2_target.get_second_y_position()) };
            let u_value =
                unsafe { *yuy2_plane_shifted.get_unchecked(yuy2_target.get_u_position()) };
            let v_value =
                unsafe { *yuy2_plane_shifted.get_unchecked(yuy2_target.get_v_position()) };

            unsafe {
                *y_plane.get_unchecked_mut(y_pos) = first_y_position;
                *y_plane.get_unchecked_mut(y_pos + 1) = second_y_position;
                *u_plane.get_unchecked_mut(u_pos) = u_value;
                *v_plane.get_unchecked_mut(v_pos) = v_value;
                if chroma_subsampling == YuvChromaSample::YUV444 {
                    *u_plane.get_unchecked_mut(u_pos + 1) = u_value;
                    *v_plane.get_unchecked_mut(v_pos + 1) = v_value;
                }
            }

            _uv_x += match chroma_subsampling {
                YuvChromaSample::YUV420 | YuvChromaSample::YUV422 => 1,
                YuvChromaSample::YUV444 => 2,
            };
            _cx += 2;
        }

        if width & 1 == 1 {
            let u_pos = u_offset + _uv_x;
            let v_pos = v_offset + _uv_x;
            let y_pos = y_offset + _cx;
            let dst_offset = yuy_offset + ((width as usize - 1) / 2) * 4;

            let yuy2_plane_shifted = unsafe { yuy2_store.get_unchecked(dst_offset..) };

            let first_y_position =
                unsafe { *yuy2_plane_shifted.get_unchecked(yuy2_target.get_first_y_position()) };
            let u_value =
                unsafe { *yuy2_plane_shifted.get_unchecked(yuy2_target.get_u_position()) };
            let v_value =
                unsafe { *yuy2_plane_shifted.get_unchecked(yuy2_target.get_v_position()) };

            unsafe {
                *y_plane.get_unchecked_mut(y_pos) = first_y_position;
                *u_plane.get_unchecked_mut(u_pos) = u_value;
                *v_plane.get_unchecked_mut(v_pos) = v_value;
            }
        }

        y_offset += y_stride as usize;
        yuy_offset += yuy2_stride as usize;
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

/// Convert YUYV format to YUV 444 planar format.
///
/// This function takes YUYV (4:2:2) format data with 8-bit precision,
/// and converts it to YUV 444 planar format with 8-bit per channel precision.
///
/// # Arguments
///
/// * `y_plane` - A mutable slice to load the Y (luminance) plane data.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `u_plane` - A mutable slice to load the U (chrominance) plane data.
/// * `u_stride` - The stride (bytes per row) for the U plane.
/// * `v_plane` - A mutable slice to load the V (chrominance) plane data.
/// * `v_stride` - The stride (bytes per row) for the V plane.
/// * `width` - The width of the YUV image.
/// * `height` - The height of the YUV image.
/// * `yuy2_store` - A slice to store the converted YUYV data.
/// * `yuy2_stride` - The stride (bytes per row) for the YUYV plane.
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input YUYV data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuyv422_to_yuv444(
    y_plane: &mut [u8],
    y_stride: u32,
    u_plane: &mut [u8],
    u_stride: u32,
    v_plane: &mut [u8],
    v_stride: u32,
    yuy2_store: &[u8],
    yuy2_stride: u32,
    width: u32,
    height: u32,
) {
    yuy2_to_yuv_impl::<{ YuvChromaSample::YUV444 as u8 }, { Yuy2Description::YUYV as usize }>(
        y_plane,
        y_stride,
        u_plane,
        u_stride,
        v_plane,
        v_stride,
        yuy2_store,
        yuy2_stride,
        width,
        height,
    );
}

/// Convert YUYV format to YUV 420 planar format.
///
/// This function takes YUYV (4:2:2) format data with 8-bit precision,
/// and converts it to YUV 444 planar format with 8-bit per channel precision.
///
/// # Arguments
///
/// * `y_plane` - A mutable slice to load the Y (luminance) plane data.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `u_plane` - A mutable slice to load the U (chrominance) plane data.
/// * `u_stride` - The stride (bytes per row) for the U plane.
/// * `v_plane` - A mutable slice to load the V (chrominance) plane data.
/// * `v_stride` - The stride (bytes per row) for the V plane.
/// * `width` - The width of the YUV image.
/// * `height` - The height of the YUV image.
/// * `yuy2_store` - A slice to store the converted YUYV data.
/// * `yuy2_stride` - The stride (bytes per row) for the YUYV plane.
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input YUYV data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuyv422_to_yuv420(
    y_plane: &mut [u8],
    y_stride: u32,
    u_plane: &mut [u8],
    u_stride: u32,
    v_plane: &mut [u8],
    v_stride: u32,
    yuy2_store: &[u8],
    yuy2_stride: u32,
    width: u32,
    height: u32,
) {
    yuy2_to_yuv_impl::<{ YuvChromaSample::YUV420 as u8 }, { Yuy2Description::YUYV as usize }>(
        y_plane,
        y_stride,
        u_plane,
        u_stride,
        v_plane,
        v_stride,
        yuy2_store,
        yuy2_stride,
        width,
        height,
    );
}

/// Convert YVYU format to YUV 422 planar format.
///
/// This function takes YUYV (4:2:2) format data with 8-bit precision,
/// and converts it to YUV 444 planar format with 8-bit per channel precision.
///
/// # Arguments
///
/// * `y_plane` - A mutable slice to load the Y (luminance) plane data.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `u_plane` - A mutable slice to load the U (chrominance) plane data.
/// * `u_stride` - The stride (bytes per row) for the U plane.
/// * `v_plane` - A mutable slice to load the V (chrominance) plane data.
/// * `v_stride` - The stride (bytes per row) for the V plane.
/// * `width` - The width of the YUV image.
/// * `height` - The height of the YUV image.
/// * `yuy2_store` - A slice to store the converted YUYV data.
/// * `yuy2_stride` - The stride (bytes per row) for the YUYV plane.
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input YUYV data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuyv422_to_yuv422(
    y_plane: &mut [u8],
    y_stride: u32,
    u_plane: &mut [u8],
    u_stride: u32,
    v_plane: &mut [u8],
    v_stride: u32,
    yuy2_store: &[u8],
    yuy2_stride: u32,
    width: u32,
    height: u32,
) {
    yuy2_to_yuv_impl::<{ YuvChromaSample::YUV422 as u8 }, { Yuy2Description::YUYV as usize }>(
        y_plane,
        y_stride,
        u_plane,
        u_stride,
        v_plane,
        v_stride,
        yuy2_store,
        yuy2_stride,
        width,
        height,
    );
}

/// Convert YVYU format to YUV 444 planar format.
///
/// This function takes YVYU (4:2:2) format data with 8-bit precision,
/// and converts it to YUV 444 planar format with 8-bit per channel precision.
///
/// # Arguments
///
/// * `y_plane` - A mutable slice to load the Y (luminance) plane data.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `u_plane` - A mutable slice to load the U (chrominance) plane data.
/// * `u_stride` - The stride (bytes per row) for the U plane.
/// * `v_plane` - A mutable slice to load the V (chrominance) plane data.
/// * `v_stride` - The stride (bytes per row) for the V plane.
/// * `width` - The width of the YUV image.
/// * `height` - The height of the YUV image.
/// * `yuy2_store` - A slice to store the converted YVYU data.
/// * `yuy2_stride` - The stride (bytes per row) for the YVYU plane.
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input YVYU data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yvyu422_to_yuv444(
    y_plane: &mut [u8],
    y_stride: u32,
    u_plane: &mut [u8],
    u_stride: u32,
    v_plane: &mut [u8],
    v_stride: u32,
    yuy2_store: &[u8],
    yuy2_stride: u32,
    width: u32,
    height: u32,
) {
    yuy2_to_yuv_impl::<{ YuvChromaSample::YUV444 as u8 }, { Yuy2Description::YVYU as usize }>(
        y_plane,
        y_stride,
        u_plane,
        u_stride,
        v_plane,
        v_stride,
        yuy2_store,
        yuy2_stride,
        width,
        height,
    );
}

/// Convert YVYU format to YUV 420 planar format.
///
/// This function takes YVYU (4:2:2) format data with 8-bit precision,
/// and converts it to YUV 444 planar format with 8-bit per channel precision.
///
/// # Arguments
///
/// * `y_plane` - A mutable slice to load the Y (luminance) plane data.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `u_plane` - A mutable slice to load the U (chrominance) plane data.
/// * `u_stride` - The stride (bytes per row) for the U plane.
/// * `v_plane` - A mutable slice to load the V (chrominance) plane data.
/// * `v_stride` - The stride (bytes per row) for the V plane.
/// * `width` - The width of the YUV image.
/// * `height` - The height of the YUV image.
/// * `yuy2_store` - A slice to store the converted YVYU data.
/// * `yuy2_stride` - The stride (bytes per row) for the YVYU plane.
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input YVYU data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yvyu422_to_yuv420(
    y_plane: &mut [u8],
    y_stride: u32,
    u_plane: &mut [u8],
    u_stride: u32,
    v_plane: &mut [u8],
    v_stride: u32,
    yuy2_store: &[u8],
    yuy2_stride: u32,
    width: u32,
    height: u32,
) {
    yuy2_to_yuv_impl::<{ YuvChromaSample::YUV420 as u8 }, { Yuy2Description::YVYU as usize }>(
        y_plane,
        y_stride,
        u_plane,
        u_stride,
        v_plane,
        v_stride,
        yuy2_store,
        yuy2_stride,
        width,
        height,
    );
}

/// Convert YVYU format to YUV 422 planar format.
///
/// This function takes YVYU (4:2:2) format data with 8-bit precision,
/// and converts it to YUV 444 planar format with 8-bit per channel precision.
///
/// # Arguments
///
/// * `y_plane` - A mutable slice to load the Y (luminance) plane data.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `u_plane` - A mutable slice to load the U (chrominance) plane data.
/// * `u_stride` - The stride (bytes per row) for the U plane.
/// * `v_plane` - A mutable slice to load the V (chrominance) plane data.
/// * `v_stride` - The stride (bytes per row) for the V plane.
/// * `width` - The width of the YUV image.
/// * `height` - The height of the YUV image.
/// * `yuy2_store` - A slice to store the converted YVYU data.
/// * `yuy2_stride` - The stride (bytes per row) for the YVYU plane.
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input YVYU data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yvyu422_to_yuv422(
    y_plane: &mut [u8],
    y_stride: u32,
    u_plane: &mut [u8],
    u_stride: u32,
    v_plane: &mut [u8],
    v_stride: u32,
    yuy2_store: &[u8],
    yuy2_stride: u32,
    width: u32,
    height: u32,
) {
    yuy2_to_yuv_impl::<{ YuvChromaSample::YUV422 as u8 }, { Yuy2Description::YVYU as usize }>(
        y_plane,
        y_stride,
        u_plane,
        u_stride,
        v_plane,
        v_stride,
        yuy2_store,
        yuy2_stride,
        width,
        height,
    );
}

/// Convert VYUY format to YUV 444 planar format.
///
/// This function takes VYUY (4:2:2) format data with 8-bit precision,
/// and converts it to YUV 444 planar format with 8-bit per channel precision.
///
/// # Arguments
///
/// * `y_plane` - A mutable slice to load the Y (luminance) plane data.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `u_plane` - A mutable slice to load the U (chrominance) plane data.
/// * `u_stride` - The stride (bytes per row) for the U plane.
/// * `v_plane` - A mutable slice to load the V (chrominance) plane data.
/// * `v_stride` - The stride (bytes per row) for the V plane.
/// * `width` - The width of the YUV image.
/// * `height` - The height of the YUV image.
/// * `yuy2_store` - A slice to store the converted VYUY data.
/// * `yuy2_stride` - The stride (bytes per row) for the VYUY plane.
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input VYUY data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn vyuy422_to_yuv444(
    y_plane: &mut [u8],
    y_stride: u32,
    u_plane: &mut [u8],
    u_stride: u32,
    v_plane: &mut [u8],
    v_stride: u32,
    yuy2_store: &[u8],
    yuy2_stride: u32,
    width: u32,
    height: u32,
) {
    yuy2_to_yuv_impl::<{ YuvChromaSample::YUV444 as u8 }, { Yuy2Description::VYUY as usize }>(
        y_plane,
        y_stride,
        u_plane,
        u_stride,
        v_plane,
        v_stride,
        yuy2_store,
        yuy2_stride,
        width,
        height,
    );
}

/// Convert VYUY format to YUV 420 planar format.
///
/// This function takes VYUY (4:2:2) format data with 8-bit precision,
/// and converts it to YUV 444 planar format with 8-bit per channel precision.
///
/// # Arguments
///
/// * `y_plane` - A mutable slice to load the Y (luminance) plane data.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `u_plane` - A mutable slice to load the U (chrominance) plane data.
/// * `u_stride` - The stride (bytes per row) for the U plane.
/// * `v_plane` - A mutable slice to load the V (chrominance) plane data.
/// * `v_stride` - The stride (bytes per row) for the V plane.
/// * `width` - The width of the YUV image.
/// * `height` - The height of the YUV image.
/// * `yuy2_store` - A slice to store the converted VYUY data.
/// * `yuy2_stride` - The stride (bytes per row) for the VYUY plane.
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input VYUY data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn vyuy422_to_yuv420(
    y_plane: &mut [u8],
    y_stride: u32,
    u_plane: &mut [u8],
    u_stride: u32,
    v_plane: &mut [u8],
    v_stride: u32,
    yuy2_store: &[u8],
    yuy2_stride: u32,
    width: u32,
    height: u32,
) {
    yuy2_to_yuv_impl::<{ YuvChromaSample::YUV420 as u8 }, { Yuy2Description::VYUY as usize }>(
        y_plane,
        y_stride,
        u_plane,
        u_stride,
        v_plane,
        v_stride,
        yuy2_store,
        yuy2_stride,
        width,
        height,
    );
}

/// Convert VYUY format to YUV 422 planar format.
///
/// This function takes VYUY (4:2:2) format data with 8-bit precision,
/// and converts it to YUV 444 planar format with 8-bit per channel precision.
///
/// # Arguments
///
/// * `y_plane` - A mutable slice to load the Y (luminance) plane data.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `u_plane` - A mutable slice to load the U (chrominance) plane data.
/// * `u_stride` - The stride (bytes per row) for the U plane.
/// * `v_plane` - A mutable slice to load the V (chrominance) plane data.
/// * `v_stride` - The stride (bytes per row) for the V plane.
/// * `width` - The width of the YUV image.
/// * `height` - The height of the YUV image.
/// * `yuy2_store` - A slice to store the converted VYUY data.
/// * `yuy2_stride` - The stride (bytes per row) for the VYUY plane.
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input VYUY data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn vyuy422_to_yuv422(
    y_plane: &mut [u8],
    y_stride: u32,
    u_plane: &mut [u8],
    u_stride: u32,
    v_plane: &mut [u8],
    v_stride: u32,
    yuy2_store: &[u8],
    yuy2_stride: u32,
    width: u32,
    height: u32,
) {
    yuy2_to_yuv_impl::<{ YuvChromaSample::YUV422 as u8 }, { Yuy2Description::VYUY as usize }>(
        y_plane,
        y_stride,
        u_plane,
        u_stride,
        v_plane,
        v_stride,
        yuy2_store,
        yuy2_stride,
        width,
        height,
    );
}

/// Convert UYVY format to YUV 444 planar format.
///
/// This function takes UYVY (4:2:2) format data with 8-bit precision,
/// and converts it to YUV 444 planar format with 8-bit per channel precision.
///
/// # Arguments
///
/// * `y_plane` - A mutable slice to load the Y (luminance) plane data.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `u_plane` - A mutable slice to load the U (chrominance) plane data.
/// * `u_stride` - The stride (bytes per row) for the U plane.
/// * `v_plane` - A mutable slice to load the V (chrominance) plane data.
/// * `v_stride` - The stride (bytes per row) for the V plane.
/// * `width` - The width of the YUV image.
/// * `height` - The height of the YUV image.
/// * `yuy2_store` - A slice to store the converted UYVY data.
/// * `yuy2_stride` - The stride (bytes per row) for the UYVY plane.
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input UYVY data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn uyvy422_to_yuv444(
    y_plane: &mut [u8],
    y_stride: u32,
    u_plane: &mut [u8],
    u_stride: u32,
    v_plane: &mut [u8],
    v_stride: u32,
    yuy2_store: &[u8],
    yuy2_stride: u32,
    width: u32,
    height: u32,
) {
    yuy2_to_yuv_impl::<{ YuvChromaSample::YUV444 as u8 }, { Yuy2Description::UYVY as usize }>(
        y_plane,
        y_stride,
        u_plane,
        u_stride,
        v_plane,
        v_stride,
        yuy2_store,
        yuy2_stride,
        width,
        height,
    );
}

/// Convert UYVY format to YUV 420 planar format.
///
/// This function takes UYVY (4:2:2) format data with 8-bit precision,
/// and converts it to YUV 444 planar format with 8-bit per channel precision.
///
/// # Arguments
///
/// * `y_plane` - A mutable slice to load the Y (luminance) plane data.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `u_plane` - A mutable slice to load the U (chrominance) plane data.
/// * `u_stride` - The stride (bytes per row) for the U plane.
/// * `v_plane` - A mutable slice to load the V (chrominance) plane data.
/// * `v_stride` - The stride (bytes per row) for the V plane.
/// * `width` - The width of the YUV image.
/// * `height` - The height of the YUV image.
/// * `yuy2_store` - A slice to store the converted UYVY data.
/// * `yuy2_stride` - The stride (bytes per row) for the UYVY plane.
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input UYVY data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn uyvy422_to_yuv420(
    y_plane: &mut [u8],
    y_stride: u32,
    u_plane: &mut [u8],
    u_stride: u32,
    v_plane: &mut [u8],
    v_stride: u32,
    yuy2_store: &[u8],
    yuy2_stride: u32,
    width: u32,
    height: u32,
) {
    yuy2_to_yuv_impl::<{ YuvChromaSample::YUV420 as u8 }, { Yuy2Description::UYVY as usize }>(
        y_plane,
        y_stride,
        u_plane,
        u_stride,
        v_plane,
        v_stride,
        yuy2_store,
        yuy2_stride,
        width,
        height,
    );
}

/// Convert UYVY format to YUV 422 planar format.
///
/// This function takes UYVY (4:2:2) format data with 8-bit precision,
/// and converts it to YUV 444 planar format with 8-bit per channel precision.
///
/// # Arguments
///
/// * `y_plane` - A mutable slice to load the Y (luminance) plane data.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `u_plane` - A mutable slice to load the U (chrominance) plane data.
/// * `u_stride` - The stride (bytes per row) for the U plane.
/// * `v_plane` - A mutable slice to load the V (chrominance) plane data.
/// * `v_stride` - The stride (bytes per row) for the V plane.
/// * `width` - The width of the YUV image.
/// * `height` - The height of the YUV image.
/// * `yuy2_store` - A slice to store the converted UYVY data.
/// * `yuy2_stride` - The stride (bytes per row) for the UYVY plane.
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input UYVY data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn uyvy422_to_yuv422(
    y_plane: &mut [u8],
    y_stride: u32,
    u_plane: &mut [u8],
    u_stride: u32,
    v_plane: &mut [u8],
    v_stride: u32,
    yuy2_store: &[u8],
    yuy2_stride: u32,
    width: u32,
    height: u32,
) {
    yuy2_to_yuv_impl::<{ YuvChromaSample::YUV422 as u8 }, { Yuy2Description::UYVY as usize }>(
        y_plane,
        y_stride,
        u_plane,
        u_stride,
        v_plane,
        v_stride,
        yuy2_store,
        yuy2_stride,
        width,
        height,
    );
}
