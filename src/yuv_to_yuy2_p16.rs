/*
 * // Copyright (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */
use crate::yuv_support::{YuvChromaSample, Yuy2Description};

fn yuv_to_yuy2_impl_p16<const SAMPLING: u8, const YUY2_TARGET: usize>(
    y_plane: &[u16],
    y_stride: u32,
    u_plane: &[u16],
    u_stride: u32,
    v_plane: &[u16],
    v_stride: u32,
    yuy2_store: &mut [u16],
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

    for y in 0..height as usize {
        let mut _cx = 0usize;
        let mut _uv_x = 0usize;
        let mut _yuy2_x = 0usize;

        for x in _yuy2_x..width as usize / 2 {
            unsafe {
                let u_pos = _uv_x;
                let v_pos = _uv_x;
                let y_pos = _cx;

                let y_src_ptr =
                    ((y_plane.as_ptr() as *const u8).add(y_offset) as *const u16).add(y_pos);
                let u_src_ptr =
                    ((u_plane.as_ptr() as *const u8).add(u_offset) as *const u16).add(u_pos);
                let v_src_ptr =
                    ((v_plane.as_ptr() as *const u8).add(v_offset) as *const u16).add(v_pos);

                let (u_value, v_value);

                if chroma_subsampling == YuvChromaSample::YUV444 {
                    u_value = ((u_src_ptr.read_unaligned() as u32
                        + u_src_ptr.add(1).read_unaligned() as u32) + 1
                        >> 1) as u16;
                    v_value = ((v_src_ptr.read_unaligned() as u32
                        + v_src_ptr.add(1).read_unaligned() as u32) + 1
                        >> 1) as u16;
                } else {
                    u_value = u_src_ptr.read_unaligned();
                    v_value = v_src_ptr.read_unaligned();
                }

                let first_y_value = y_src_ptr.read_unaligned();
                let second_y_value = y_src_ptr.add(1).read_unaligned();

                let dst_ptr =
                    ((yuy2_store.as_ptr() as *mut u8).add(yuy_offset) as *mut u16).add(x * 4);

                dst_ptr
                    .add(yuy2_target.get_first_y_position())
                    .write_unaligned(first_y_value);
                dst_ptr
                    .add(yuy2_target.get_u_position())
                    .write_unaligned(u_value);
                dst_ptr
                    .add(yuy2_target.get_second_y_position())
                    .write_unaligned(second_y_value);
                dst_ptr
                    .add(yuy2_target.get_v_position())
                    .write_unaligned(v_value);
            }

            _uv_x += match chroma_subsampling {
                YuvChromaSample::YUV420 | YuvChromaSample::YUV422 => 1,
                YuvChromaSample::YUV444 => 2,
            };
            _cx += 2;
        }

        if width & 1 == 1 {
            unsafe {
                let u_pos = _uv_x;
                let v_pos = _uv_x;
                let y_pos = _cx;

                let y_src_ptr =
                    ((y_plane.as_ptr() as *const u8).add(y_offset) as *const u16).add(y_pos);
                let u_src_ptr =
                    ((u_plane.as_ptr() as *const u8).add(u_offset) as *const u16).add(u_pos);
                let v_src_ptr =
                    ((v_plane.as_ptr() as *const u8).add(v_offset) as *const u16).add(v_pos);

                let u_value = u_src_ptr.read_unaligned();
                let v_value = v_src_ptr.read_unaligned();

                let first_y_value = y_src_ptr.read_unaligned();

                let dst_ptr = ((yuy2_store.as_ptr() as *mut u8).add(yuy_offset) as *mut u16)
                    .add(((width as usize - 1) / 2) * 4);
                dst_ptr
                    .add(yuy2_target.get_first_y_position())
                    .write_unaligned(first_y_value);
                dst_ptr
                    .add(yuy2_target.get_u_position())
                    .write_unaligned(u_value);
                dst_ptr
                    .add(yuy2_target.get_second_y_position())
                    .write_unaligned(0);
                dst_ptr
                    .add(yuy2_target.get_v_position())
                    .write_unaligned(v_value);
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

/// Convert YUV 444 planar format to YUYV ( YUV Packed ) format.
///
/// This function takes YUV 444 planar format data with 8-16 bit precision,
/// and converts it to YUYV format with 8-16 bit per channel precision.
/// Do not forget about odd alignment, use (width + 1) for buffers.
///
/// # Arguments
///
/// * `y_plane` - A slice to load the Y (luminance) plane data.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `u_plane` - A slice to load the U (chrominance) plane data.
/// * `u_stride` - The stride (bytes per row) for the U plane.
/// * `v_plane` - A slice to load the V (chrominance) plane data.
/// * `v_stride` - The stride (bytes per row) for the V plane.
/// * `width` - The width of the YUV image.
/// * `height` - The height of the YUV image.
/// * `yuy2_store` - A mutable slice to store the converted YUYV data.
/// * `yuy2_stride` - The stride (bytes per row) for the YUYV plane.
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input YUYV data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
/// Panic will be received if buffer doesn't expand with (width + 1) size for odd width
///
pub fn yuv444_to_yuyv422_p16(
    y_plane: &[u16],
    y_stride: u32,
    u_plane: &[u16],
    u_stride: u32,
    v_plane: &[u16],
    v_stride: u32,
    yuy2_store: &mut [u16],
    yuy2_stride: u32,
    width: u32,
    height: u32,
) {
    yuv_to_yuy2_impl_p16::<{ YuvChromaSample::YUV444 as u8 }, { Yuy2Description::YUYV as usize }>(
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

/// Convert YUV 422 planar format to YUYV ( YUV Packed ) format.
///
/// This function takes YUV 422 planar format data with 8-16 bit precision,
/// and converts it to YUYV format with 8-16 bit per channel precision.
/// Do not forget about odd alignment, use (width + 1) for buffers.
///
/// # Arguments
///
/// * `y_plane` - A slice to load the Y (luminance) plane data.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `u_plane` - A slice to load the U (chrominance) plane data.
/// * `u_stride` - The stride (bytes per row) for the U plane.
/// * `v_plane` - A slice to load the V (chrominance) plane data.
/// * `v_stride` - The stride (bytes per row) for the V plane.
/// * `width` - The width of the YUV image.
/// * `height` - The height of the YUV image.
/// * `yuy2_store` - A mutable slice to store the converted YUYV data.
/// * `yuy2_stride` - The stride (bytes per row) for the YUYV plane.
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input YUYV data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
/// Panic will be received if buffer doesn't expand with (width + 1) size for odd width
///
pub fn yuv422_to_yuyv422_p16(
    y_plane: &[u16],
    y_stride: u32,
    u_plane: &[u16],
    u_stride: u32,
    v_plane: &[u16],
    v_stride: u32,
    yuy2_store: &mut [u16],
    yuy2_stride: u32,
    width: u32,
    height: u32,
) {
    yuv_to_yuy2_impl_p16::<{ YuvChromaSample::YUV422 as u8 }, { Yuy2Description::YUYV as usize }>(
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

/// Convert YUV 420 planar format to YUYV ( YUV Packed ) format.
///
/// This function takes YUV 420 planar format data with 8-16 bit precision,
/// and converts it to YUYV format with 8-16 bit per channel precision.
/// Do not forget about odd alignment, use (width + 1) for buffers.
///
/// # Arguments
///
/// * `y_plane` - A slice to load the Y (luminance) plane data.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `u_plane` - A slice to load the U (chrominance) plane data.
/// * `u_stride` - The stride (bytes per row) for the U plane.
/// * `v_plane` - A slice to load the V (chrominance) plane data.
/// * `v_stride` - The stride (bytes per row) for the V plane.
/// * `width` - The width of the YUV image.
/// * `height` - The height of the YUV image.
/// * `yuy2_store` - A mutable slice to store the converted YUYV data.
/// * `yuy2_stride` - The stride (bytes per row) for the YUYV plane.
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input YUYV data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
/// Panic will be received if buffer doesn't expand with (width + 1) size for odd width
///
pub fn yuv420_to_yuyv422_p16(
    y_plane: &[u16],
    y_stride: u32,
    u_plane: &[u16],
    u_stride: u32,
    v_plane: &[u16],
    v_stride: u32,
    yuy2_store: &mut [u16],
    yuy2_stride: u32,
    width: u32,
    height: u32,
) {
    yuv_to_yuy2_impl_p16::<{ YuvChromaSample::YUV420 as u8 }, { Yuy2Description::YUYV as usize }>(
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

/// Convert YUV 444 planar format to YVYU ( YUV Packed ) format.
///
/// This function takes YUV 444 planar format data with 8-16 bit precision,
/// and converts it to YVYU format with 8-16 bit per channel precision.
/// Do not forget about odd alignment, use (width + 1) for buffers.
///
/// # Arguments
///
/// * `y_plane` - A slice to load the Y (luminance) plane data.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `u_plane` - A slice to load the U (chrominance) plane data.
/// * `u_stride` - The stride (bytes per row) for the U plane.
/// * `v_plane` - A slice to load the V (chrominance) plane data.
/// * `v_stride` - The stride (bytes per row) for the V plane.
/// * `width` - The width of the YUV image.
/// * `height` - The height of the YUV image.
/// * `yuy2_store` - A mutable slice to store the converted YVYU data.
/// * `yuy2_stride` - The stride (bytes per row) for the YVYU plane.
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input YVYU data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
/// Panic will be received if buffer doesn't expand with (width + 1) size for odd width
///
pub fn yuv444_to_yvyu422_p16(
    y_plane: &[u16],
    y_stride: u32,
    u_plane: &[u16],
    u_stride: u32,
    v_plane: &[u16],
    v_stride: u32,
    yuy2_store: &mut [u16],
    yuy2_stride: u32,
    width: u32,
    height: u32,
) {
    yuv_to_yuy2_impl_p16::<{ YuvChromaSample::YUV444 as u8 }, { Yuy2Description::YVYU as usize }>(
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

/// Convert YUV 422 planar format to YVYU ( YUV Packed ) format.
///
/// This function takes YUV 422 planar format data with 8-16 bit precision,
/// and converts it to YVYU format with 8-16 bit per channel precision.
/// Do not forget about odd alignment, use (width + 1) for buffers.
///
/// # Arguments
///
/// * `y_plane` - A slice to load the Y (luminance) plane data.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `u_plane` - A slice to load the U (chrominance) plane data.
/// * `u_stride` - The stride (bytes per row) for the U plane.
/// * `v_plane` - A slice to load the V (chrominance) plane data.
/// * `v_stride` - The stride (bytes per row) for the V plane.
/// * `width` - The width of the YUV image.
/// * `height` - The height of the YUV image.
/// * `yuy2_store` - A mutable slice to store the converted YVYU data.
/// * `yuy2_stride` - The stride (bytes per row) for the YVYU plane.
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input YVYU data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
/// Panic will be received if buffer doesn't expand with (width + 1) size for odd width
///
pub fn yuv422_to_yvyu422_p16(
    y_plane: &[u16],
    y_stride: u32,
    u_plane: &[u16],
    u_stride: u32,
    v_plane: &[u16],
    v_stride: u32,
    yuy2_store: &mut [u16],
    yuy2_stride: u32,
    width: u32,
    height: u32,
) {
    yuv_to_yuy2_impl_p16::<{ YuvChromaSample::YUV422 as u8 }, { Yuy2Description::YVYU as usize }>(
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

/// Convert YUV 420 planar format to YVYU ( YUV Packed ) format.
///
/// This function takes YUV 420 planar format data with 8-16 bit precision,
/// and converts it to YVYU format with 8-16 bit per channel precision.
/// Do not forget about odd alignment, use (width + 1) for buffers.
///
/// # Arguments
///
/// * `y_plane` - A slice to load the Y (luminance) plane data.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `u_plane` - A slice to load the U (chrominance) plane data.
/// * `u_stride` - The stride (bytes per row) for the U plane.
/// * `v_plane` - A slice to load the V (chrominance) plane data.
/// * `v_stride` - The stride (bytes per row) for the V plane.
/// * `width` - The width of the YUV image.
/// * `height` - The height of the YUV image.
/// * `yuy2_store` - A mutable slice to store the converted YVYU data.
/// * `yuy2_stride` - The stride (bytes per row) for the YVYU plane.
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input YVYU data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
/// Panic will be received if buffer doesn't expand with (width + 1) size for odd width
///
pub fn yuv420_to_yvyu422_p16(
    y_plane: &[u16],
    y_stride: u32,
    u_plane: &[u16],
    u_stride: u32,
    v_plane: &[u16],
    v_stride: u32,
    yuy2_store: &mut [u16],
    yuy2_stride: u32,
    width: u32,
    height: u32,
) {
    yuv_to_yuy2_impl_p16::<{ YuvChromaSample::YUV420 as u8 }, { Yuy2Description::YVYU as usize }>(
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

/// Convert YUV 444 planar format to VYUY ( YUV Packed ) format.
///
/// This function takes YUV 444 planar format data with 8-16 bit precision,
/// and converts it to VYUY format with 8-16 bit per channel precision.
/// Do not forget about odd alignment, use (width + 1) for buffers.
///
/// # Arguments
///
/// * `y_plane` - A slice to load the Y (luminance) plane data.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `u_plane` - A slice to load the U (chrominance) plane data.
/// * `u_stride` - The stride (bytes per row) for the U plane.
/// * `v_plane` - A slice to load the V (chrominance) plane data.
/// * `v_stride` - The stride (bytes per row) for the V plane.
/// * `width` - The width of the YUV image.
/// * `height` - The height of the YUV image.
/// * `yuy2_store` - A mutable slice to store the converted VYUY data.
/// * `yuy2_stride` - The stride (bytes per row) for the VYUY plane.
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input VYUY data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
/// Panic will be received if buffer doesn't expand with (width + 1) size for odd width
///
pub fn yuv444_to_vyuy422_p16(
    y_plane: &[u16],
    y_stride: u32,
    u_plane: &[u16],
    u_stride: u32,
    v_plane: &[u16],
    v_stride: u32,
    yuy2_store: &mut [u16],
    yuy2_stride: u32,
    width: u32,
    height: u32,
) {
    yuv_to_yuy2_impl_p16::<{ YuvChromaSample::YUV444 as u8 }, { Yuy2Description::VYUY as usize }>(
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

/// Convert YUV 422 planar format to VYUY ( YUV Packed ) format.
///
/// This function takes YUV 422 planar format data with 8-16 bit precision,
/// and converts it to VYUY format with 8-16 bit per channel precision.
/// Do not forget about odd alignment, use (width + 1) for buffers.
///
/// # Arguments
///
/// * `y_plane` - A slice to load the Y (luminance) plane data.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `u_plane` - A slice to load the U (chrominance) plane data.
/// * `u_stride` - The stride (bytes per row) for the U plane.
/// * `v_plane` - A slice to load the V (chrominance) plane data.
/// * `v_stride` - The stride (bytes per row) for the V plane.
/// * `width` - The width of the YUV image.
/// * `height` - The height of the YUV image.
/// * `yuy2_store` - A mutable slice to store the converted VYUY data.
/// * `yuy2_stride` - The stride (bytes per row) for the VYUY plane.
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input VYUY data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
/// Panic will be received if buffer doesn't expand with (width + 1) size for odd width
///
pub fn yuv422_to_vyuy422_p16(
    y_plane: &[u16],
    y_stride: u32,
    u_plane: &[u16],
    u_stride: u32,
    v_plane: &[u16],
    v_stride: u32,
    yuy2_store: &mut [u16],
    yuy2_stride: u32,
    width: u32,
    height: u32,
) {
    yuv_to_yuy2_impl_p16::<{ YuvChromaSample::YUV422 as u8 }, { Yuy2Description::VYUY as usize }>(
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

/// Convert YUV 420 planar format to VYUY ( YUV Packed ) format.
///
/// This function takes YUV 420 planar format data with 8-16 bit precision,
/// and converts it to VYUY format with 8-16 bit per channel precision.
/// Do not forget about odd alignment, use (width + 1) for buffers.
///
/// # Arguments
///
/// * `y_plane` - A slice to load the Y (luminance) plane data.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `u_plane` - A slice to load the U (chrominance) plane data.
/// * `u_stride` - The stride (bytes per row) for the U plane.
/// * `v_plane` - A slice to load the V (chrominance) plane data.
/// * `v_stride` - The stride (bytes per row) for the V plane.
/// * `width` - The width of the YUV image.
/// * `height` - The height of the YUV image.
/// * `yuy2_store` - A mutable slice to store the converted VYUY data.
/// * `yuy2_stride` - The stride (bytes per row) for the VYUY plane.
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input VYUY data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
/// Panic will be received if buffer doesn't expand with (width + 1) size for odd width
///
pub fn yuv420_to_vyuy422_p16(
    y_plane: &[u16],
    y_stride: u32,
    u_plane: &[u16],
    u_stride: u32,
    v_plane: &[u16],
    v_stride: u32,
    yuy2_store: &mut [u16],
    yuy2_stride: u32,
    width: u32,
    height: u32,
) {
    yuv_to_yuy2_impl_p16::<{ YuvChromaSample::YUV420 as u8 }, { Yuy2Description::VYUY as usize }>(
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

/// Convert YUV 444 planar format to UYVY ( YUV Packed ) format.
///
/// This function takes YUV 444 planar format data with 8-16 bit precision,
/// and converts it to UYVY format with 8-16 bit per channel precision.
/// Do not forget about odd alignment, use (width + 1) for buffers.
///
/// # Arguments
///
/// * `y_plane` - A slice to load the Y (luminance) plane data.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `u_plane` - A slice to load the U (chrominance) plane data.
/// * `u_stride` - The stride (bytes per row) for the U plane.
/// * `v_plane` - A slice to load the V (chrominance) plane data.
/// * `v_stride` - The stride (bytes per row) for the V plane.
/// * `width` - The width of the YUV image.
/// * `height` - The height of the YUV image.
/// * `yuy2_store` - A mutable slice to store the converted UYVY data.
/// * `yuy2_stride` - The stride (bytes per row) for the UYVY plane.
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input UYVY data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
/// Panic will be received if buffer doesn't expand with (width + 1) size for odd width
///
pub fn yuv444_to_uyvy422_p16(
    y_plane: &[u16],
    y_stride: u32,
    u_plane: &[u16],
    u_stride: u32,
    v_plane: &[u16],
    v_stride: u32,
    yuy2_store: &mut [u16],
    yuy2_stride: u32,
    width: u32,
    height: u32,
) {
    yuv_to_yuy2_impl_p16::<{ YuvChromaSample::YUV444 as u8 }, { Yuy2Description::UYVY as usize }>(
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

/// Convert YUV 422 planar format to UYVY ( YUV Packed ) format.
///
/// This function takes YUV 422 planar format data with 8-16 bit precision,
/// and converts it to UYVY format with 8-16 bit per channel precision.
/// Do not forget about odd alignment, use (width + 1) for buffers.
///
/// # Arguments
///
/// * `y_plane` - A slice to load the Y (luminance) plane data.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `u_plane` - A slice to load the U (chrominance) plane data.
/// * `u_stride` - The stride (bytes per row) for the U plane.
/// * `v_plane` - A slice to load the V (chrominance) plane data.
/// * `v_stride` - The stride (bytes per row) for the V plane.
/// * `width` - The width of the YUV image.
/// * `height` - The height of the YUV image.
/// * `yuy2_store` - A mutable slice to store the converted UYVY data.
/// * `yuy2_stride` - The stride (bytes per row) for the UYVY plane.
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input UYVY data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
/// Panic will be received if buffer doesn't expand with (width + 1) size for odd width
///
pub fn yuv422_to_uyvy422_p16(
    y_plane: &[u16],
    y_stride: u32,
    u_plane: &[u16],
    u_stride: u32,
    v_plane: &[u16],
    v_stride: u32,
    yuy2_store: &mut [u16],
    yuy2_stride: u32,
    width: u32,
    height: u32,
) {
    yuv_to_yuy2_impl_p16::<{ YuvChromaSample::YUV422 as u8 }, { Yuy2Description::UYVY as usize }>(
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

/// Convert YUV 420 planar format to UYVY ( YUV Packed ) format.
///
/// This function takes YUV 420 planar format data with 8-16 bit precision,
/// and converts it to UYVY format with 8-16 bit per channel precision.
/// Do not forget about odd alignment, use (width + 1) for buffers.
///
/// # Arguments
///
/// * `y_plane` - A slice to load the Y (luminance) plane data.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `u_plane` - A slice to load the U (chrominance) plane data.
/// * `u_stride` - The stride (bytes per row) for the U plane.
/// * `v_plane` - A slice to load the V (chrominance) plane data.
/// * `v_stride` - The stride (bytes per row) for the V plane.
/// * `width` - The width of the YUV image.
/// * `height` - The height of the YUV image.
/// * `yuy2_store` - A mutable slice to store the converted UYVY data.
/// * `yuy2_stride` - The stride (bytes per row) for the UYVY plane.
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input UYVY data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
/// Panic will be received if buffer doesn't expand with (width + 1) size for odd width
///
pub fn yuv420_to_uyvy422_p16(
    y_plane: &[u16],
    y_stride: u32,
    u_plane: &[u16],
    u_stride: u32,
    v_plane: &[u16],
    v_stride: u32,
    yuy2_store: &mut [u16],
    yuy2_stride: u32,
    width: u32,
    height: u32,
) {
    yuv_to_yuy2_impl_p16::<{ YuvChromaSample::YUV420 as u8 }, { Yuy2Description::UYVY as usize }>(
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
