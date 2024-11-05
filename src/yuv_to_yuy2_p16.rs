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
use crate::yuv_support::{YuvChromaSubsample, Yuy2Description};
use crate::{YuvError, YuvPlanarImage};

fn yuv_to_yuy2_impl_p16<const SAMPLING: u8, const YUY2_TARGET: usize>(
    planar_image: &YuvPlanarImage<u16>,
    yuy2_store: &mut [u16],
    yuy2_stride: u32,
) -> Result<(), YuvError> {
    let yuy2_target: Yuy2Description = YUY2_TARGET.into();
    let chroma_subsampling: YuvChromaSubsample = SAMPLING.into();

    planar_image.check_constraints(chroma_subsampling)?;

    let mut y_offset = 0usize;
    let mut u_offset = 0usize;
    let mut v_offset = 0usize;
    let mut yuy_offset = 0usize;

    let height = planar_image.height;
    let width = planar_image.width;
    let y_stride = planar_image.y_stride * 2;
    let u_stride = planar_image.u_stride * 2;
    let v_stride = planar_image.v_stride * 2;
    let y_plane = planar_image.y_plane;
    let v_plane = planar_image.v_plane;
    let u_plane = planar_image.u_plane;

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

                if chroma_subsampling == YuvChromaSubsample::Yuv444 {
                    u_value = (((u_src_ptr.read_unaligned() as u32
                        + u_src_ptr.add(1).read_unaligned() as u32)
                        + 1)
                        >> 1) as u16;
                    v_value = (((v_src_ptr.read_unaligned() as u32
                        + v_src_ptr.add(1).read_unaligned() as u32)
                        + 1)
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
                YuvChromaSubsample::Yuv420 | YuvChromaSubsample::Yuv422 => 1,
                YuvChromaSubsample::Yuv444 => 2,
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
            YuvChromaSubsample::Yuv420 => {
                if y & 1 == 1 {
                    u_offset += u_stride as usize;
                    v_offset += v_stride as usize;
                }
            }
            YuvChromaSubsample::Yuv444 | YuvChromaSubsample::Yuv422 => {
                u_offset += u_stride as usize;
                v_offset += v_stride as usize;
            }
        }
    }

    Ok(())
}

/// Convert YUV 444 planar format to YUYV ( YUV Packed ) format.
///
/// This function takes YUV 444 planar format data with 8-16 bit precision,
/// and converts it to YUYV format with 8-16 bit per channel precision.
/// Do not forget about odd alignment, use (width + 1) for buffers.
///
/// # Arguments
///
/// * `planar_image` - Source planar image.
/// * `yuy2_store` - A mutable slice to store the converted YUYV data.
/// * `yuy2_stride` - The stride (components per row) for the YUYV plane.
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input YUYV data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
/// Panic will be received if buffer doesn't expand with (width + 1) size for odd width.
///
pub fn yuv444_to_yuyv422_p16(
    planar_image: &YuvPlanarImage<u16>,
    yuy2_store: &mut [u16],
    yuy2_stride: u32,
) -> Result<(), YuvError> {
    yuv_to_yuy2_impl_p16::<{ YuvChromaSubsample::Yuv444 as u8 }, { Yuy2Description::YUYV as usize }>(
        planar_image,
        yuy2_store,
        yuy2_stride,
    )
}

/// Convert YUV 422 planar format to YUYV ( YUV Packed ) format.
///
/// This function takes YUV 422 planar format data with 8-16 bit precision,
/// and converts it to YUYV format with 8-16 bit per channel precision.
/// Do not forget about odd alignment, use (width + 1) for buffers.
///
/// # Arguments
///
/// * `planar_image` - Source planar image.
/// * `yuy2_store` - A mutable slice to store the converted YUYV data.
/// * `yuy2_stride` - The stride (components per row) for the YUYV plane.
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input YUYV data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
/// Panic will be received if buffer doesn't expand with (width + 1) size for odd width.
///
pub fn yuv422_to_yuyv422_p16(
    planar_image: &YuvPlanarImage<u16>,
    yuy2_store: &mut [u16],
    yuy2_stride: u32,
) -> Result<(), YuvError> {
    yuv_to_yuy2_impl_p16::<{ YuvChromaSubsample::Yuv422 as u8 }, { Yuy2Description::YUYV as usize }>(
        planar_image,
        yuy2_store,
        yuy2_stride,
    )
}

/// Convert YUV 420 planar format to YUYV ( YUV Packed ) format.
///
/// This function takes YUV 420 planar format data with 8-16 bit precision,
/// and converts it to YUYV format with 8-16 bit per channel precision.
/// Do not forget about odd alignment, use (width + 1) for buffers.
///
/// # Arguments
///
/// * `planar_image` - Source planar image.
/// * `yuy2_store` - A mutable slice to store the converted YUYV data.
/// * `yuy2_stride` - The stride (components per row) for the YUYV plane.
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input YUYV data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
/// Panic will be received if buffer doesn't expand with (width + 1) size for odd width.
///
pub fn yuv420_to_yuyv422_p16(
    planar_image: &YuvPlanarImage<u16>,
    yuy2_store: &mut [u16],
    yuy2_stride: u32,
) -> Result<(), YuvError> {
    yuv_to_yuy2_impl_p16::<{ YuvChromaSubsample::Yuv420 as u8 }, { Yuy2Description::YUYV as usize }>(
        planar_image,
        yuy2_store,
        yuy2_stride,
    )
}

/// Convert YUV 444 planar format to YVYU ( YUV Packed ) format.
///
/// This function takes YUV 444 planar format data with 8-16 bit precision,
/// and converts it to YVYU format with 8-16 bit per channel precision.
/// Do not forget about odd alignment, use (width + 1) for buffers.
///
/// # Arguments
///
/// * `planar_image` - Source planar image.
/// * `yuy2_store` - A mutable slice to store the converted YVYU data.
/// * `yuy2_stride` - The stride (components per row) for the YVYU plane.
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input YVYU data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
/// Panic will be received if buffer doesn't expand with (width + 1) size for odd width.
///
pub fn yuv444_to_yvyu422_p16(
    planar_image: &YuvPlanarImage<u16>,
    yuy2_store: &mut [u16],
    yuy2_stride: u32,
) -> Result<(), YuvError> {
    yuv_to_yuy2_impl_p16::<{ YuvChromaSubsample::Yuv444 as u8 }, { Yuy2Description::YVYU as usize }>(
        planar_image,
        yuy2_store,
        yuy2_stride,
    )
}

/// Convert YUV 422 planar format to YVYU ( YUV Packed ) format.
///
/// This function takes YUV 422 planar format data with 8-16 bit precision,
/// and converts it to YVYU format with 8-16 bit per channel precision.
/// Do not forget about odd alignment, use (width + 1) for buffers.
///
/// # Arguments
///
/// * `planar_image` - Source planar image.
/// * `yuy2_store` - A mutable slice to store the converted YVYU data.
/// * `yuy2_stride` - The stride (components per row) for the YVYU plane.
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input YVYU data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
/// Panic will be received if buffer doesn't expand with (width + 1) size for odd width.
///
pub fn yuv422_to_yvyu422_p16(
    planar_image: &YuvPlanarImage<u16>,
    yuy2_store: &mut [u16],
    yuy2_stride: u32,
) -> Result<(), YuvError> {
    yuv_to_yuy2_impl_p16::<{ YuvChromaSubsample::Yuv422 as u8 }, { Yuy2Description::YVYU as usize }>(
        planar_image,
        yuy2_store,
        yuy2_stride,
    )
}

/// Convert YUV 420 planar format to YVYU ( YUV Packed ) format.
///
/// This function takes YUV 420 planar format data with 8-16 bit precision,
/// and converts it to YVYU format with 8-16 bit per channel precision.
/// Do not forget about odd alignment, use (width + 1) for buffers.
///
/// # Arguments
///
/// * `planar_image` - Source planar image.
/// * `yuy2_store` - A mutable slice to store the converted YVYU data.
/// * `yuy2_stride` - The stride (components per row) for the YVYU plane.
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input YVYU data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
/// Panic will be received if buffer doesn't expand with (width + 1) size for odd width.
///
pub fn yuv420_to_yvyu422_p16(
    planar_image: &YuvPlanarImage<u16>,
    yuy2_store: &mut [u16],
    yuy2_stride: u32,
) -> Result<(), YuvError> {
    yuv_to_yuy2_impl_p16::<{ YuvChromaSubsample::Yuv420 as u8 }, { Yuy2Description::YVYU as usize }>(
        planar_image,
        yuy2_store,
        yuy2_stride,
    )
}

/// Convert YUV 444 planar format to VYUY ( YUV Packed ) format.
///
/// This function takes YUV 444 planar format data with 8-16 bit precision,
/// and converts it to VYUY format with 8-16 bit per channel precision.
/// Do not forget about odd alignment, use (width + 1) for buffers.
///
/// # Arguments
///
/// * `planar_image` - Source planar image.
/// * `yuy2_store` - A mutable slice to store the converted VYUY data.
/// * `yuy2_stride` - The stride (components per row) for the VYUY plane.
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input VYUY data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
/// Panic will be received if buffer doesn't expand with (width + 1) size for odd width.
///
pub fn yuv444_to_vyuy422_p16(
    planar_image: &YuvPlanarImage<u16>,
    yuy2_store: &mut [u16],
    yuy2_stride: u32,
) -> Result<(), YuvError> {
    yuv_to_yuy2_impl_p16::<{ YuvChromaSubsample::Yuv444 as u8 }, { Yuy2Description::VYUY as usize }>(
        planar_image,
        yuy2_store,
        yuy2_stride,
    )
}

/// Convert YUV 422 planar format to VYUY ( YUV Packed ) format.
///
/// This function takes YUV 422 planar format data with 8-16 bit precision,
/// and converts it to VYUY format with 8-16 bit per channel precision.
/// Do not forget about odd alignment, use (width + 1) for buffers.
///
/// # Arguments
///
/// * `planar_image` - Source planar image.
/// * `yuy2_store` - A mutable slice to store the converted VYUY data.
/// * `yuy2_stride` - The stride (components per row) for the VYUY plane.
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input VYUY data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
/// Panic will be received if buffer doesn't expand with (width + 1) size for odd width.
///
pub fn yuv422_to_vyuy422_p16(
    planar_image: &YuvPlanarImage<u16>,
    yuy2_store: &mut [u16],
    yuy2_stride: u32,
) -> Result<(), YuvError> {
    yuv_to_yuy2_impl_p16::<{ YuvChromaSubsample::Yuv422 as u8 }, { Yuy2Description::VYUY as usize }>(
        planar_image,
        yuy2_store,
        yuy2_stride,
    )
}

/// Convert YUV 420 planar format to VYUY ( YUV Packed ) format.
///
/// This function takes YUV 420 planar format data with 8-16 bit precision,
/// and converts it to VYUY format with 8-16 bit per channel precision.
/// Do not forget about odd alignment, use (width + 1) for buffers.
///
/// # Arguments
///
/// * `planar_image` - Source planar image.
/// * `yuy2_store` - A mutable slice to store the converted VYUY data.
/// * `yuy2_stride` - The stride (components per row) for the VYUY plane.
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input VYUY data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
/// Panic will be received if buffer doesn't expand with (width + 1) size for odd width.
///
pub fn yuv420_to_vyuy422_p16(
    planar_image: &YuvPlanarImage<u16>,
    yuy2_store: &mut [u16],
    yuy2_stride: u32,
) -> Result<(), YuvError> {
    yuv_to_yuy2_impl_p16::<{ YuvChromaSubsample::Yuv420 as u8 }, { Yuy2Description::VYUY as usize }>(
        planar_image,
        yuy2_store,
        yuy2_stride,
    )
}

/// Convert YUV 444 planar format to UYVY ( YUV Packed ) format.
///
/// This function takes YUV 444 planar format data with 8-16 bit precision,
/// and converts it to UYVY format with 8-16 bit per channel precision.
/// Do not forget about odd alignment, use (width + 1) for buffers.
///
/// # Arguments
///
/// * `planar_image` - Source planar image.
/// * `yuy2_store` - A mutable slice to store the converted UYVY data.
/// * `yuy2_stride` - The stride (components per row) for the UYVY plane.
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input UYVY data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
/// Panic will be received if buffer doesn't expand with (width + 1) size for odd width.
///
pub fn yuv444_to_uyvy422_p16(
    planar_image: &YuvPlanarImage<u16>,
    yuy2_store: &mut [u16],
    yuy2_stride: u32,
) -> Result<(), YuvError> {
    yuv_to_yuy2_impl_p16::<{ YuvChromaSubsample::Yuv444 as u8 }, { Yuy2Description::UYVY as usize }>(
        planar_image,
        yuy2_store,
        yuy2_stride,
    )
}

/// Convert YUV 422 planar format to UYVY ( YUV Packed ) format.
///
/// This function takes YUV 422 planar format data with 8-16 bit precision,
/// and converts it to UYVY format with 8-16 bit per channel precision.
/// Do not forget about odd alignment, use (width + 1) for buffers.
///
/// # Arguments
///
/// * `planar_image` - Source planar image.
/// * `yuy2_store` - A mutable slice to store the converted UYVY data.
/// * `yuy2_stride` - The stride (components per row) for the UYVY plane.
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input UYVY data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
/// Panic will be received if buffer doesn't expand with (width + 1) size for odd width.
///
pub fn yuv422_to_uyvy422_p16(
    planar_image: &YuvPlanarImage<u16>,
    yuy2_store: &mut [u16],
    yuy2_stride: u32,
) -> Result<(), YuvError> {
    yuv_to_yuy2_impl_p16::<{ YuvChromaSubsample::Yuv422 as u8 }, { Yuy2Description::UYVY as usize }>(
        planar_image,
        yuy2_store,
        yuy2_stride,
    )
}

/// Convert YUV 420 planar format to UYVY ( YUV Packed ) format.
///
/// This function takes YUV 420 planar format data with 8-16 bit precision,
/// and converts it to UYVY format with 8-16 bit per channel precision.
/// Do not forget about odd alignment, use (width + 1) for buffers.
///
/// # Arguments
///
/// * `planar_image` - Source planar image.
/// * `yuy2_store` - A mutable slice to store the converted UYVY data.
/// * `yuy2_stride` - The stride (components per row) for the UYVY plane.
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input UYVY data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
/// Panic will be received if buffer doesn't expand with (width + 1) size for odd width.
///
pub fn yuv420_to_uyvy422_p16(
    planar_image: &YuvPlanarImage<u16>,
    yuy2_store: &mut [u16],
    yuy2_stride: u32,
) -> Result<(), YuvError> {
    yuv_to_yuy2_impl_p16::<{ YuvChromaSubsample::Yuv420 as u8 }, { Yuy2Description::UYVY as usize }>(
        planar_image,
        yuy2_store,
        yuy2_stride,
    )
}
