/*
 * Copyright (c) Radzivon Bartoshyk, 10/2024. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 * 1.  Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2.  Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3.  Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
use crate::yuv_support::{YuvChromaSample, Yuy2Description};

fn yuy2_to_yuv_impl<const SAMPLING: u8, const YUY2_TARGET: usize>(
    y_plane: &mut [u16],
    y_stride: u32,
    u_plane: &mut [u16],
    u_stride: u32,
    v_plane: &mut [u16],
    v_stride: u32,
    yuy2_store: &[u16],
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

                let mut y_dst_ptr = (y_plane.as_mut_ptr() as *mut u8).add(y_offset) as *mut u16;
                y_dst_ptr = y_dst_ptr.add(y_pos);
                let mut u_dst_ptr = (u_plane.as_mut_ptr() as *mut u8).add(u_offset) as *mut u16;
                u_dst_ptr = u_dst_ptr.add(u_pos);
                let mut v_dst_ptr = (v_plane.as_mut_ptr() as *mut u8).add(v_offset) as *mut u16;
                v_dst_ptr = v_dst_ptr.add(v_pos);

                let mut yuy2_ptr = (yuy2_store.as_ptr() as *const u8).add(yuy_offset) as *const u16;
                yuy2_ptr = yuy2_ptr.add(x * 4);

                let first_y_position = yuy2_ptr
                    .add(yuy2_target.get_first_y_position())
                    .read_unaligned();
                let second_y_position = yuy2_ptr
                    .add(yuy2_target.get_second_y_position())
                    .read_unaligned();
                let u_value = u_dst_ptr.add(yuy2_target.get_u_position()).read_unaligned();
                let v_value = v_dst_ptr.add(yuy2_target.get_v_position()).read_unaligned();

                y_dst_ptr.write_unaligned(first_y_position);
                y_dst_ptr.add(1).write_unaligned(second_y_position);
                u_dst_ptr.write_unaligned(u_value);
                v_dst_ptr.write_unaligned(v_value);
                if chroma_subsampling == YuvChromaSample::YUV444 {
                    u_dst_ptr.add(1).write_unaligned(u_value);
                    v_dst_ptr.add(1).write_unaligned(v_value);
                }
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
                let yuy2_x = ((width as usize - 1) / 2) * 4;

                let mut y_dst_ptr = (y_plane.as_mut_ptr() as *mut u8).add(y_offset) as *mut u16;
                y_dst_ptr = y_dst_ptr.add(y_pos);
                let mut u_dst_ptr = (u_plane.as_mut_ptr() as *mut u8).add(u_offset) as *mut u16;
                u_dst_ptr = u_dst_ptr.add(u_pos);
                let mut v_dst_ptr = (v_plane.as_mut_ptr() as *mut u8).add(v_offset) as *mut u16;
                v_dst_ptr = v_dst_ptr.add(v_pos);

                let mut yuy2_ptr = (yuy2_store.as_ptr() as *const u8).add(yuy_offset) as *const u16;
                yuy2_ptr = yuy2_ptr.add(yuy2_x);

                let first_y_position = yuy2_ptr
                    .add(yuy2_target.get_first_y_position())
                    .read_unaligned();
                let u_value = yuy2_ptr.add(yuy2_target.get_u_position()).read_unaligned();
                let v_value = yuy2_ptr.add(yuy2_target.get_v_position()).read_unaligned();

                y_dst_ptr.write_unaligned(first_y_position);
                u_dst_ptr.write_unaligned(u_value);
                v_dst_ptr.write_unaligned(v_value);
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

/// Convert YUYV (YUV Packed) format to YUV 444 planar format.
///
/// This function takes YUYV (4:2:2) format data with 8-16 bit precision,
/// and converts it to YUV 444 planar format with 8-16 bit per channel precision.
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
pub fn yuyv422_to_yuv444_p16(
    y_plane: &mut [u16],
    y_stride: u32,
    u_plane: &mut [u16],
    u_stride: u32,
    v_plane: &mut [u16],
    v_stride: u32,
    yuy2_store: &[u16],
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

/// Convert YUYV (YUV Packed) format to YUV 420 planar format.
///
/// This function takes YUYV (4:2:2) format data with 8-16 bit precision,
/// and converts it to YUV 444 planar format with 8-16 bit per channel precision.
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
pub fn yuyv422_to_yuv420_p16(
    y_plane: &mut [u16],
    y_stride: u32,
    u_plane: &mut [u16],
    u_stride: u32,
    v_plane: &mut [u16],
    v_stride: u32,
    yuy2_store: &[u16],
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

/// Convert YVYU (YUV Packed) format to YUV 422 planar format.
///
/// This function takes YUYV (4:2:2) format data with 8-16 bit precision,
/// and converts it to YUV 444 planar format with 8-16 bit per channel precision.
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
pub fn yuyv422_to_yuv422_p16(
    y_plane: &mut [u16],
    y_stride: u32,
    u_plane: &mut [u16],
    u_stride: u32,
    v_plane: &mut [u16],
    v_stride: u32,
    yuy2_store: &[u16],
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

/// Convert YVYU (YUV Packed) format to YUV 444 planar format.
///
/// This function takes YVYU (4:2:2) format data with 8-16 bit precision,
/// and converts it to YUV 444 planar format with 8-16 bit per channel precision.
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
pub fn yvyu422_to_yuv444_p16(
    y_plane: &mut [u16],
    y_stride: u32,
    u_plane: &mut [u16],
    u_stride: u32,
    v_plane: &mut [u16],
    v_stride: u32,
    yuy2_store: &[u16],
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

/// Convert YVYU (YUV Packed) format to YUV 420 planar format.
///
/// This function takes YVYU (4:2:2) format data with 8-16 bit precision,
/// and converts it to YUV 444 planar format with 8-16 bit per channel precision.
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
pub fn yvyu422_to_yuv420_p16(
    y_plane: &mut [u16],
    y_stride: u32,
    u_plane: &mut [u16],
    u_stride: u32,
    v_plane: &mut [u16],
    v_stride: u32,
    yuy2_store: &[u16],
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

/// Convert YVYU (YUV Packed) format to YUV 422 planar format.
///
/// This function takes YVYU (4:2:2) format data with 8-16 bit precision,
/// and converts it to YUV 444 planar format with 8-16 bit per channel precision.
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
pub fn yvyu422_to_yuv422_p16(
    y_plane: &mut [u16],
    y_stride: u32,
    u_plane: &mut [u16],
    u_stride: u32,
    v_plane: &mut [u16],
    v_stride: u32,
    yuy2_store: &[u16],
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

/// Convert VYUY (YUV Packed) format to YUV 444 planar format.
///
/// This function takes VYUY (4:2:2) format data with 8-16 bit precision,
/// and converts it to YUV 444 planar format with 8-16 bit per channel precision.
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
pub fn vyuy422_to_yuv444_p16(
    y_plane: &mut [u16],
    y_stride: u32,
    u_plane: &mut [u16],
    u_stride: u32,
    v_plane: &mut [u16],
    v_stride: u32,
    yuy2_store: &[u16],
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

/// Convert VYUY (YUV Packed) format to YUV 420 planar format.
///
/// This function takes VYUY (4:2:2) format data with 8-16 bit precision,
/// and converts it to YUV 444 planar format with 8-16 bit per channel precision.
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
pub fn vyuy422_to_yuv420_p16(
    y_plane: &mut [u16],
    y_stride: u32,
    u_plane: &mut [u16],
    u_stride: u32,
    v_plane: &mut [u16],
    v_stride: u32,
    yuy2_store: &[u16],
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

/// Convert VYUY (YUV Packed) format to YUV 422 planar format.
///
/// This function takes VYUY (4:2:2) format data with 8-16 bit precision,
/// and converts it to YUV 444 planar format with 8-16 bit per channel precision.
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
pub fn vyuy422_to_yuv422_p16(
    y_plane: &mut [u16],
    y_stride: u32,
    u_plane: &mut [u16],
    u_stride: u32,
    v_plane: &mut [u16],
    v_stride: u32,
    yuy2_store: &[u16],
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

/// Convert UYVY (YUV Packed) format to YUV 444 planar format.
///
/// This function takes UYVY (4:2:2) format data with 8-16 bit precision,
/// and converts it to YUV 444 planar format with 8-16 bit per channel precision.
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
pub fn uyvy422_to_yuv444_p16(
    y_plane: &mut [u16],
    y_stride: u32,
    u_plane: &mut [u16],
    u_stride: u32,
    v_plane: &mut [u16],
    v_stride: u32,
    yuy2_store: &[u16],
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

/// Convert UYVY (YUV Packed) format to YUV 420 planar format.
///
/// This function takes UYVY (4:2:2) format data with 8-16 bit precision,
/// and converts it to YUV 444 planar format with 8-16 bit per channel precision.
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
pub fn uyvy422_to_yuv420_p16(
    y_plane: &mut [u16],
    y_stride: u32,
    u_plane: &mut [u16],
    u_stride: u32,
    v_plane: &mut [u16],
    v_stride: u32,
    yuy2_store: &[u16],
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

/// Convert UYVY (YUV Packed) format to YUV 422 planar format.
///
/// This function takes UYVY (4:2:2) format data with 8-16 bit precision,
/// and converts it to YUV 444 planar format with 8-16 bit per channel precision.
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
pub fn uyvy422_to_yuv422_p16(
    y_plane: &mut [u16],
    y_stride: u32,
    u_plane: &mut [u16],
    u_stride: u32,
    v_plane: &mut [u16],
    v_stride: u32,
    yuy2_store: &[u16],
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
