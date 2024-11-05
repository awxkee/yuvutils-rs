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
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use crate::avx2::yuv_to_yuy2_avx2_row;
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use crate::neon::yuv_to_yuy2_neon_impl;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use crate::sse::yuv_to_yuy2_sse_impl;
use crate::yuv_support::{YuvChromaSubsample, Yuy2Description};
use crate::{YuvError, YuvPlanarImage};
#[cfg(feature = "rayon")]
use rayon::iter::{IndexedParallelIterator, ParallelIterator};
#[cfg(feature = "rayon")]
use rayon::prelude::ParallelSliceMut;

#[allow(dead_code)]
#[derive(Debug, Copy, Clone, Ord, PartialOrd, Eq, PartialEq)]
pub struct YuvToYuy2Navigation {
    pub cx: usize,
    pub uv_x: usize,
    pub x: usize,
}

impl YuvToYuy2Navigation {
    #[allow(dead_code)]
    pub const fn new(cx: usize, uv_x: usize, x: usize) -> YuvToYuy2Navigation {
        YuvToYuy2Navigation { cx, uv_x, x }
    }
}

fn yuv_to_yuy2_impl<const SAMPLING: u8, const YUY2_TARGET: usize>(
    planar_image: &YuvPlanarImage<u8>,
    yuy2_store: &mut [u8],
    yuy2_stride: u32,
) -> Result<(), YuvError> {
    let yuy2_target: Yuy2Description = YUY2_TARGET.into();
    let chroma_subsampling: YuvChromaSubsample = SAMPLING.into();

    planar_image.check_constraints(chroma_subsampling)?;

    let yuy_offset = 0usize;

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    let mut _use_sse = is_x86_feature_detected!("sse4.1");
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    let mut _use_avx2 = is_x86_feature_detected!("avx2");

    let iter;
    #[cfg(feature = "rayon")]
    {
        iter = yuy2_store.par_chunks_exact_mut(yuy2_stride as usize);
    }
    #[cfg(not(feature = "rayon"))]
    {
        iter = yuy2_store.chunks_exact_mut(yuy2_stride as usize);
    }

    let y_stride = planar_image.y_stride;
    let u_stride = planar_image.u_stride;
    let v_stride = planar_image.v_stride;
    let y_plane = planar_image.y_plane;
    let v_plane = planar_image.v_plane;
    let u_plane = planar_image.u_plane;
    let width = planar_image.width;

    iter.enumerate().for_each(|(y, yuy2_store)| unsafe {
        let y_offset = y * (y_stride as usize);
        let u_offset = if chroma_subsampling == YuvChromaSubsample::Yuv420 {
            (y >> 1) * (u_stride as usize)
        } else {
            y * (u_stride as usize)
        };
        let v_offset = if chroma_subsampling == YuvChromaSubsample::Yuv420 {
            (y >> 1) * (v_stride as usize)
        } else {
            y * (v_stride as usize)
        };

        let mut _cx = 0usize;
        let mut _uv_x = 0usize;
        let mut _yuy2_x = 0usize;

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if _use_avx2 {
                let processed = yuv_to_yuy2_avx2_row::<SAMPLING, YUY2_TARGET>(
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
                let processed = yuv_to_yuy2_sse_impl::<SAMPLING, YUY2_TARGET>(
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

        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        {
            let processed = yuv_to_yuy2_neon_impl::<SAMPLING, YUY2_TARGET>(
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

        for x in _yuy2_x..width as usize / 2 {
            let u_pos = u_offset + _uv_x;
            let v_pos = v_offset + _uv_x;
            let y_pos = y_offset + _cx;

            let (u_value, v_value);

            if chroma_subsampling == YuvChromaSubsample::Yuv444 {
                u_value = (((*u_plane.get_unchecked(u_pos) as u32
                    + *u_plane.get_unchecked(u_pos + 1) as u32)
                    + 1)
                    >> 1) as u8;
                v_value = (((*v_plane.get_unchecked(v_pos) as u32
                    + *v_plane.get_unchecked(v_pos + 1) as u32)
                    + 1)
                    >> 1) as u8;
            } else {
                u_value = *u_plane.get_unchecked(u_pos);
                v_value = *v_plane.get_unchecked(v_pos);
            }

            let first_y_value = *y_plane.get_unchecked(y_pos);
            let second_y_value = *y_plane.get_unchecked(y_pos + 1);

            let dst_offset = yuy_offset + x * 4;
            let dst_store = yuy2_store.get_unchecked_mut(dst_offset..);
            *dst_store.get_unchecked_mut(yuy2_target.get_first_y_position()) = first_y_value;
            *dst_store.get_unchecked_mut(yuy2_target.get_u_position()) = u_value;
            *dst_store.get_unchecked_mut(yuy2_target.get_second_y_position()) = second_y_value;
            *dst_store.get_unchecked_mut(yuy2_target.get_v_position()) = v_value;

            _uv_x += match chroma_subsampling {
                YuvChromaSubsample::Yuv420 | YuvChromaSubsample::Yuv422 => 1,
                YuvChromaSubsample::Yuv444 => 2,
            };
            _cx += 2;
        }

        if width & 1 == 1 {
            let u_pos = u_offset + _uv_x;
            let v_pos = v_offset + _uv_x;
            let y_pos = y_offset + _cx;

            let u_value = *u_plane.get_unchecked(u_pos);
            let v_value = *v_plane.get_unchecked(v_pos);

            let first_y_value = *y_plane.get_unchecked(y_pos);

            let dst_offset = yuy_offset + ((width as usize - 1) / 2) * 4;
            let dst_store = yuy2_store.get_unchecked_mut(dst_offset..);
            *dst_store.get_unchecked_mut(yuy2_target.get_first_y_position()) = first_y_value;
            *dst_store.get_unchecked_mut(yuy2_target.get_u_position()) = u_value;
            *dst_store.get_unchecked_mut(yuy2_target.get_second_y_position()) = 0;
            *dst_store.get_unchecked_mut(yuy2_target.get_v_position()) = v_value;
        }
    });
    Ok(())
}

/// Convert YUV 444 planar format to YUYV ( YUV Packed ) format.
///
/// This function takes YUV 444 planar format data with 8-bit precision,
/// and converts it to YUYV format with 8-bit per channel precision.
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
/// Panic will be received if buffer doesn't expand with (width + 1) size for odd width
///
pub fn yuv444_to_yuyv422(
    planar_image: &YuvPlanarImage<u8>,
    yuy2_store: &mut [u8],
    yuy2_stride: u32,
) -> Result<(), YuvError> {
    yuv_to_yuy2_impl::<{ YuvChromaSubsample::Yuv444 as u8 }, { Yuy2Description::YUYV as usize }>(
        planar_image,
        yuy2_store,
        yuy2_stride,
    )
}

/// Convert YUV 422 planar format to YUYV ( YUV Packed ) format.
///
/// This function takes YUV 422 planar format data with 8-bit precision,
/// and converts it to YUYV format with 8-bit per channel precision.
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
/// Panic will be received if buffer doesn't expand with (width + 1) size for odd width
///
pub fn yuv422_to_yuyv422(
    planar_image: &YuvPlanarImage<u8>,
    yuy2_store: &mut [u8],
    yuy2_stride: u32,
) -> Result<(), YuvError> {
    yuv_to_yuy2_impl::<{ YuvChromaSubsample::Yuv422 as u8 }, { Yuy2Description::YUYV as usize }>(
        planar_image,
        yuy2_store,
        yuy2_stride,
    )
}

/// Convert YUV 420 planar format to YUYV ( YUV Packed ) format.
///
/// This function takes YUV 420 planar format data with 8-bit precision,
/// and converts it to YUYV format with 8-bit per channel precision.
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
/// Panic will be received if buffer doesn't expand with (width + 1) size for odd width
///
pub fn yuv420_to_yuyv422(
    planar_image: &YuvPlanarImage<u8>,
    yuy2_store: &mut [u8],
    yuy2_stride: u32,
) -> Result<(), YuvError> {
    yuv_to_yuy2_impl::<{ YuvChromaSubsample::Yuv420 as u8 }, { Yuy2Description::YUYV as usize }>(
        planar_image,
        yuy2_store,
        yuy2_stride,
    )
}

/// Convert YUV 444 planar format to YVYU ( YUV Packed ) format.
///
/// This function takes YUV 444 planar format data with 8-bit precision,
/// and converts it to YVYU format with 8-bit per channel precision.
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
/// Panic will be received if buffer doesn't expand with (width + 1) size for odd width
///
pub fn yuv444_to_yvyu422(
    planar_image: &YuvPlanarImage<u8>,
    yuy2_store: &mut [u8],
    yuy2_stride: u32,
) -> Result<(), YuvError> {
    yuv_to_yuy2_impl::<{ YuvChromaSubsample::Yuv444 as u8 }, { Yuy2Description::YVYU as usize }>(
        planar_image,
        yuy2_store,
        yuy2_stride,
    )
}

/// Convert YUV 422 planar format to YVYU ( YUV Packed ) format.
///
/// This function takes YUV 422 planar format data with 8-bit precision,
/// and converts it to YVYU format with 8-bit per channel precision.
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
/// Panic will be received if buffer doesn't expand with (width + 1) size for odd width
///
pub fn yuv422_to_yvyu422(
    planar_image: &YuvPlanarImage<u8>,
    yuy2_store: &mut [u8],
    yuy2_stride: u32,
) -> Result<(), YuvError> {
    yuv_to_yuy2_impl::<{ YuvChromaSubsample::Yuv422 as u8 }, { Yuy2Description::YVYU as usize }>(
        planar_image,
        yuy2_store,
        yuy2_stride,
    )
}

/// Convert YUV 420 planar format to YVYU ( YUV Packed ) format.
///
/// This function takes YUV 420 planar format data with 8-bit precision,
/// and converts it to YVYU format with 8-bit per channel precision.
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
/// Panic will be received if buffer doesn't expand with (width + 1) size for odd width
///
pub fn yuv420_to_yvyu422(
    planar_image: &YuvPlanarImage<u8>,
    yuy2_store: &mut [u8],
    yuy2_stride: u32,
) -> Result<(), YuvError> {
    yuv_to_yuy2_impl::<{ YuvChromaSubsample::Yuv420 as u8 }, { Yuy2Description::YVYU as usize }>(
        planar_image,
        yuy2_store,
        yuy2_stride,
    )
}

/// Convert YUV 444 planar format to VYUY ( YUV Packed ) format.
///
/// This function takes YUV 444 planar format data with 8-bit precision,
/// and converts it to VYUY format with 8-bit per channel precision.
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
/// Panic will be received if buffer doesn't expand with (width + 1) size for odd width
///
pub fn yuv444_to_vyuy422(
    planar_image: &YuvPlanarImage<u8>,
    yuy2_store: &mut [u8],
    yuy2_stride: u32,
) -> Result<(), YuvError> {
    yuv_to_yuy2_impl::<{ YuvChromaSubsample::Yuv444 as u8 }, { Yuy2Description::VYUY as usize }>(
        planar_image,
        yuy2_store,
        yuy2_stride,
    )
}

/// Convert YUV 422 planar format to VYUY ( YUV Packed ) format.
///
/// This function takes YUV 422 planar format data with 8-bit precision,
/// and converts it to VYUY format with 8-bit per channel precision.
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
/// Panic will be received if buffer doesn't expand with (width + 1) size for odd width
///
pub fn yuv422_to_vyuy422(
    planar_image: &YuvPlanarImage<u8>,
    yuy2_store: &mut [u8],
    yuy2_stride: u32,
) -> Result<(), YuvError> {
    yuv_to_yuy2_impl::<{ YuvChromaSubsample::Yuv422 as u8 }, { Yuy2Description::VYUY as usize }>(
        planar_image,
        yuy2_store,
        yuy2_stride,
    )
}

/// Convert YUV 420 planar format to VYUY ( YUV Packed ) format.
///
/// This function takes YUV 420 planar format data with 8-bit precision,
/// and converts it to VYUY format with 8-bit per channel precision.
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
/// Panic will be received if buffer doesn't expand with (width + 1) size for odd width
///
pub fn yuv420_to_vyuy422(
    planar_image: &YuvPlanarImage<u8>,
    yuy2_store: &mut [u8],
    yuy2_stride: u32,
) -> Result<(), YuvError> {
    yuv_to_yuy2_impl::<{ YuvChromaSubsample::Yuv420 as u8 }, { Yuy2Description::VYUY as usize }>(
        planar_image,
        yuy2_store,
        yuy2_stride,
    )
}

/// Convert YUV 444 planar format to UYVY ( YUV Packed ) format.
///
/// This function takes YUV 444 planar format data with 8-bit precision,
/// and converts it to UYVY format with 8-bit per channel precision.
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
/// Panic will be received if buffer doesn't expand with (width + 1) size for odd width
///
pub fn yuv444_to_uyvy422(
    planar_image: &YuvPlanarImage<u8>,
    yuy2_store: &mut [u8],
    yuy2_stride: u32,
) -> Result<(), YuvError> {
    yuv_to_yuy2_impl::<{ YuvChromaSubsample::Yuv444 as u8 }, { Yuy2Description::UYVY as usize }>(
        planar_image,
        yuy2_store,
        yuy2_stride,
    )
}

/// Convert YUV 422 planar format to UYVY ( YUV Packed ) format.
///
/// This function takes YUV 422 planar format data with 8-bit precision,
/// and converts it to UYVY format with 8-bit per channel precision.
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
/// Panic will be received if buffer doesn't expand with (width + 1) size for odd width
///
pub fn yuv422_to_uyvy422(
    planar_image: &YuvPlanarImage<u8>,
    yuy2_store: &mut [u8],
    yuy2_stride: u32,
) -> Result<(), YuvError> {
    yuv_to_yuy2_impl::<{ YuvChromaSubsample::Yuv422 as u8 }, { Yuy2Description::UYVY as usize }>(
        planar_image,
        yuy2_store,
        yuy2_stride,
    )
}

/// Convert YUV 420 planar format to UYVY ( YUV Packed ) format.
///
/// This function takes YUV 420 planar format data with 8-bit precision,
/// and converts it to UYVY format with 8-bit per channel precision.
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
/// Panic will be received if buffer doesn't expand with (width + 1) size for odd width
///
pub fn yuv420_to_uyvy422(
    planar_image: &YuvPlanarImage<u8>,
    yuy2_store: &mut [u8],
    yuy2_stride: u32,
) -> Result<(), YuvError> {
    yuv_to_yuy2_impl::<{ YuvChromaSubsample::Yuv420 as u8 }, { Yuy2Description::UYVY as usize }>(
        planar_image,
        yuy2_store,
        yuy2_stride,
    )
}
