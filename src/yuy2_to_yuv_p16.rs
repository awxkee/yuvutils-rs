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
use crate::{YuvError, YuvPlanarImageMut};
#[cfg(feature = "rayon")]
use rayon::iter::{IndexedParallelIterator, ParallelIterator};
#[cfg(feature = "rayon")]
use rayon::prelude::{ParallelSlice, ParallelSliceMut};

fn yuy2_to_yuv_impl<const SAMPLING: u8, const YUY2_TARGET: usize>(
    planar_image: &mut YuvPlanarImageMut<u16>,
    yuy2_store: &[u16],
    yuy2_stride: u32,
) -> Result<(), YuvError> {
    let yuy2_target: Yuy2Description = YUY2_TARGET.into();
    let chroma_subsampling: YuvChromaSubsample = SAMPLING.into();

    planar_image.check_constraints(chroma_subsampling)?;

    let width = planar_image.width;
    let y_plane = planar_image.y_plane.borrow_mut();
    let y_stride = planar_image.y_stride;
    let u_plane = planar_image.u_plane.borrow_mut();
    let u_stride = planar_image.u_stride;
    let v_plane = planar_image.v_plane.borrow_mut();
    let v_stride = planar_image.v_stride;

    if chroma_subsampling == YuvChromaSubsample::Yuv444 {
        let iter;
        #[cfg(feature = "rayon")]
        {
            iter = y_plane
                .par_chunks_exact_mut(y_stride as usize)
                .zip(u_plane.par_chunks_exact_mut(u_stride as usize))
                .zip(v_plane.par_chunks_exact_mut(v_stride as usize))
                .zip(yuy2_store.par_chunks_exact(yuy2_stride as usize));
        }
        #[cfg(not(feature = "rayon"))]
        {
            iter = y_plane
                .chunks_exact_mut(y_stride as usize)
                .zip(u_plane.chunks_exact_mut(u_stride as usize))
                .zip(v_plane.chunks_exact_mut(v_stride as usize))
                .zip(yuy2_store.chunks_exact(yuy2_stride as usize));
        }
        iter.for_each(|(((y_dst, u_dst), v_dst), yuy2_src)| {
            for (((y_dst, u_dst), v_dst), yuy2) in y_dst
                .chunks_exact_mut(2)
                .zip(u_dst.chunks_exact_mut(2))
                .zip(v_dst.chunks_exact_mut(2))
                .zip(yuy2_src.chunks_exact(4))
            {
                let first_y_position = yuy2[yuy2_target.get_first_y_position()];
                let second_y_position = yuy2[yuy2_target.get_second_y_position()];
                let u_value = yuy2[yuy2_target.get_u_position()];
                let v_value = yuy2[yuy2_target.get_v_position()];
                y_dst[0] = first_y_position;
                y_dst[1] = second_y_position;
                u_dst[0] = u_value;
                u_dst[1] = u_value;
                v_dst[0] = v_value;
                v_dst[1] = v_value;
            }

            if width & 1 != 0 {
                let y_dst = y_dst.last_mut().unwrap();
                let u_dst = u_dst.last_mut().unwrap();
                let v_dst = v_dst.last_mut().unwrap();
                let yuy2 = yuy2_src.chunks_exact(4).remainder();
                let yuy2 = &yuy2[0..4];
                *y_dst = yuy2[yuy2_target.get_first_y_position()];
                *u_dst = yuy2[yuy2_target.get_u_position()];
                *v_dst = yuy2[yuy2_target.get_v_position()];
            }
        });
    } else if chroma_subsampling == YuvChromaSubsample::Yuv422 {
        let iter;
        #[cfg(feature = "rayon")]
        {
            iter = y_plane
                .par_chunks_exact_mut(y_stride as usize)
                .zip(u_plane.par_chunks_exact_mut(u_stride as usize))
                .zip(v_plane.par_chunks_exact_mut(v_stride as usize))
                .zip(yuy2_store.par_chunks_exact(yuy2_stride as usize));
        }
        #[cfg(not(feature = "rayon"))]
        {
            iter = y_plane
                .chunks_exact_mut(y_stride as usize)
                .zip(u_plane.chunks_exact_mut(u_stride as usize))
                .zip(v_plane.chunks_exact_mut(v_stride as usize))
                .zip(yuy2_store.chunks_exact(yuy2_stride as usize));
        }
        iter.for_each(|(((y_dst, u_dst), v_dst), yuy2_src)| {
            for (((y_dst, u_dst), v_dst), yuy2) in y_dst
                .chunks_exact_mut(2)
                .zip(u_dst.iter_mut())
                .zip(v_dst.iter_mut())
                .zip(yuy2_src.chunks_exact(4))
            {
                let first_y_position = yuy2[yuy2_target.get_first_y_position()];
                let second_y_position = yuy2[yuy2_target.get_second_y_position()];
                let u_value = yuy2[yuy2_target.get_u_position()];
                let v_value = yuy2[yuy2_target.get_v_position()];
                y_dst[0] = first_y_position;
                y_dst[1] = second_y_position;
                *u_dst = u_value;
                *v_dst = v_value;
            }

            if width & 1 != 0 {
                let y_dst = y_dst.last_mut().unwrap();
                let u_dst = u_dst.last_mut().unwrap();
                let v_dst = v_dst.last_mut().unwrap();
                let yuy2 = yuy2_src.chunks_exact(4).remainder();
                let yuy2 = &yuy2[0..4];
                *y_dst = yuy2[yuy2_target.get_first_y_position()];
                *u_dst = yuy2[yuy2_target.get_u_position()];
                *v_dst = yuy2[yuy2_target.get_v_position()];
            }
        });
    } else if chroma_subsampling == YuvChromaSubsample::Yuv420 {
        let iter;
        #[cfg(feature = "rayon")]
        {
            iter = y_plane
                .par_chunks_exact_mut(y_stride as usize * 2)
                .zip(u_plane.par_chunks_exact_mut(u_stride as usize))
                .zip(v_plane.par_chunks_exact_mut(v_stride as usize))
                .zip(yuy2_store.par_chunks_exact(yuy2_stride as usize * 2));
        }
        #[cfg(not(feature = "rayon"))]
        {
            iter = y_plane
                .chunks_exact_mut(y_stride as usize * 2)
                .zip(u_plane.chunks_exact_mut(u_stride as usize))
                .zip(v_plane.chunks_exact_mut(v_stride as usize))
                .zip(yuy2_store.chunks_exact(yuy2_stride as usize * 2));
        }
        iter.for_each(|(((y_dst, u_dst), v_dst), yuy2_src)| {
            for (y, (y_dst, yuy2)) in y_dst
                .chunks_exact_mut(y_stride as usize)
                .zip(yuy2_src.chunks_exact(yuy2_stride as usize))
                .enumerate()
            {
                let process_chroma = y & 1 == 0;

                for (((y_dst, u_dst), v_dst), yuy2) in y_dst
                    .chunks_exact_mut(2)
                    .zip(u_dst.iter_mut())
                    .zip(v_dst.iter_mut())
                    .zip(yuy2.chunks_exact(4))
                {
                    let first_y_position = yuy2[yuy2_target.get_first_y_position()];
                    let second_y_position = yuy2[yuy2_target.get_second_y_position()];
                    y_dst[0] = first_y_position;
                    y_dst[1] = second_y_position;
                    if process_chroma {
                        let u_value = yuy2[yuy2_target.get_u_position()];
                        let v_value = yuy2[yuy2_target.get_v_position()];
                        *u_dst = u_value;
                        *v_dst = v_value;
                    }
                }

                if width & 1 != 0 {
                    let y_dst = y_dst.last_mut().unwrap();
                    let yuy2 = yuy2_src.chunks_exact(4).remainder();
                    let yuy2 = &yuy2[0..4];
                    *y_dst = yuy2[yuy2_target.get_first_y_position()];
                    if process_chroma {
                        let u_dst = u_dst.last_mut().unwrap();
                        let v_dst = v_dst.last_mut().unwrap();
                        *u_dst = yuy2[yuy2_target.get_u_position()];
                        *v_dst = yuy2[yuy2_target.get_v_position()];
                    }
                }
            }
        });
    }

    Ok(())
}

/// Convert YUYV (YUV Packed) format to YUV 444 planar format.
///
/// This function takes YUYV (4:2:2) format data with 8-16 bit precision,
/// and converts it to YUV 444 planar format with 8-16 bit per channel precision.
///
/// # Arguments
///
/// * `planar_image` - Target planar image.
/// * `yuy2_store` - A slice to store the converted YUYV data.
/// * `yuy2_stride` - The stride (elements per row) for the YUYV plane.
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input YUYV data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuyv422_to_yuv444_p16(
    planar_image: &mut YuvPlanarImageMut<u16>,
    yuy2_store: &[u16],
    yuy2_stride: u32,
) -> Result<(), YuvError> {
    yuy2_to_yuv_impl::<{ YuvChromaSubsample::Yuv444 as u8 }, { Yuy2Description::YUYV as usize }>(
        planar_image,
        yuy2_store,
        yuy2_stride,
    )
}

/// Convert YUYV (YUV Packed) format to YUV 420 planar format.
///
/// This function takes YUYV (4:2:2) format data with 8-16 bit precision,
/// and converts it to YUV 444 planar format with 8-16 bit per channel precision.
///
/// # Arguments
///
/// * `planar_image` - Target planar image.
/// * `yuy2_store` - A slice to store the converted YUYV data.
/// * `yuy2_stride` - The stride (elements per row) for the YUYV plane.
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input YUYV data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuyv422_to_yuv420_p16(
    planar_image: &mut YuvPlanarImageMut<u16>,
    yuy2_store: &[u16],
    yuy2_stride: u32,
) -> Result<(), YuvError> {
    yuy2_to_yuv_impl::<{ YuvChromaSubsample::Yuv420 as u8 }, { Yuy2Description::YUYV as usize }>(
        planar_image,
        yuy2_store,
        yuy2_stride,
    )
}

/// Convert YVYU (YUV Packed) format to YUV 422 planar format.
///
/// This function takes YUYV (4:2:2) format data with 8-16 bit precision,
/// and converts it to YUV 444 planar format with 8-16 bit per channel precision.
///
/// # Arguments
///
/// * `planar_image` - Target planar image.
/// * `yuy2_store` - A slice to store the converted YUYV data.
/// * `yuy2_stride` - The stride (elements per row) for the YUYV plane.
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input YUYV data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuyv422_to_yuv422_p16(
    planar_image: &mut YuvPlanarImageMut<u16>,
    yuy2_store: &[u16],
    yuy2_stride: u32,
) -> Result<(), YuvError> {
    yuy2_to_yuv_impl::<{ YuvChromaSubsample::Yuv422 as u8 }, { Yuy2Description::YUYV as usize }>(
        planar_image,
        yuy2_store,
        yuy2_stride,
    )
}

/// Convert YVYU (YUV Packed) format to YUV 444 planar format.
///
/// This function takes YVYU (4:2:2) format data with 8-16 bit precision,
/// and converts it to YUV 444 planar format with 8-16 bit per channel precision.
///
/// # Arguments
///
/// * `planar_image` - Target planar image.
/// * `yuy2_store` - A slice to store the converted YVYU data.
/// * `yuy2_stride` - The stride (elements per row) for the YVYU plane.
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input YVYU data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yvyu422_to_yuv444_p16(
    planar_image: &mut YuvPlanarImageMut<u16>,
    yuy2_store: &[u16],
    yuy2_stride: u32,
) -> Result<(), YuvError> {
    yuy2_to_yuv_impl::<{ YuvChromaSubsample::Yuv444 as u8 }, { Yuy2Description::YVYU as usize }>(
        planar_image,
        yuy2_store,
        yuy2_stride,
    )
}

/// Convert YVYU (YUV Packed) format to YUV 420 planar format.
///
/// This function takes YVYU (4:2:2) format data with 8-16 bit precision,
/// and converts it to YUV 444 planar format with 8-16 bit per channel precision.
///
/// # Arguments
///
/// * `planar_image` - Target planar image.
/// * `yuy2_store` - A slice to store the converted YVYU data.
/// * `yuy2_stride` - The stride (elements per row) for the YVYU plane.
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input YVYU data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yvyu422_to_yuv420_p16(
    planar_image: &mut YuvPlanarImageMut<u16>,
    yuy2_store: &[u16],
    yuy2_stride: u32,
) -> Result<(), YuvError> {
    yuy2_to_yuv_impl::<{ YuvChromaSubsample::Yuv420 as u8 }, { Yuy2Description::YVYU as usize }>(
        planar_image,
        yuy2_store,
        yuy2_stride,
    )
}

/// Convert YVYU (YUV Packed) format to YUV 422 planar format.
///
/// This function takes YVYU (4:2:2) format data with 8-16 bit precision,
/// and converts it to YUV 444 planar format with 8-16 bit per channel precision.
///
/// # Arguments
///
/// * `planar_image` - Target planar image.
/// * `yuy2_store` - A slice to store the converted YVYU data.
/// * `yuy2_stride` - The stride (elements per row) for the YVYU plane.
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input YVYU data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yvyu422_to_yuv422_p16(
    planar_image: &mut YuvPlanarImageMut<u16>,
    yuy2_store: &[u16],
    yuy2_stride: u32,
) -> Result<(), YuvError> {
    yuy2_to_yuv_impl::<{ YuvChromaSubsample::Yuv422 as u8 }, { Yuy2Description::YVYU as usize }>(
        planar_image,
        yuy2_store,
        yuy2_stride,
    )
}

/// Convert VYUY (YUV Packed) format to YUV 444 planar format.
///
/// This function takes VYUY (4:2:2) format data with 8-16 bit precision,
/// and converts it to YUV 444 planar format with 8-16 bit per channel precision.
///
/// # Arguments
///
/// * `planar_image` - Target planar image.
/// * `yuy2_store` - A slice to store the converted VYUY data.
/// * `yuy2_stride` - The stride (elements per row) for the VYUY plane.
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input VYUY data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn vyuy422_to_yuv444_p16(
    planar_image: &mut YuvPlanarImageMut<u16>,
    yuy2_store: &[u16],
    yuy2_stride: u32,
) -> Result<(), YuvError> {
    yuy2_to_yuv_impl::<{ YuvChromaSubsample::Yuv444 as u8 }, { Yuy2Description::VYUY as usize }>(
        planar_image,
        yuy2_store,
        yuy2_stride,
    )
}

/// Convert VYUY (YUV Packed) format to YUV 420 planar format.
///
/// This function takes VYUY (4:2:2) format data with 8-16 bit precision,
/// and converts it to YUV 444 planar format with 8-16 bit per channel precision.
///
/// # Arguments
///
/// * `planar_image` - Target planar image.
/// * `yuy2_store` - A slice to store the converted VYUY data.
/// * `yuy2_stride` - The stride (elements per row) for the VYUY plane.
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input VYUY data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn vyuy422_to_yuv420_p16(
    planar_image: &mut YuvPlanarImageMut<u16>,
    yuy2_store: &[u16],
    yuy2_stride: u32,
) -> Result<(), YuvError> {
    yuy2_to_yuv_impl::<{ YuvChromaSubsample::Yuv420 as u8 }, { Yuy2Description::VYUY as usize }>(
        planar_image,
        yuy2_store,
        yuy2_stride,
    )
}

/// Convert VYUY (YUV Packed) format to YUV 422 planar format.
///
/// This function takes VYUY (4:2:2) format data with 8-16 bit precision,
/// and converts it to YUV 444 planar format with 8-16 bit per channel precision.
///
/// # Arguments
///
/// * `planar_image` - Target planar image.
/// * `yuy2_store` - A slice to store the converted VYUY data.
/// * `yuy2_stride` - The stride (elements per row) for the VYUY plane.
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input VYUY data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn vyuy422_to_yuv422_p16(
    planar_image: &mut YuvPlanarImageMut<u16>,
    yuy2_store: &[u16],
    yuy2_stride: u32,
) -> Result<(), YuvError> {
    yuy2_to_yuv_impl::<{ YuvChromaSubsample::Yuv422 as u8 }, { Yuy2Description::VYUY as usize }>(
        planar_image,
        yuy2_store,
        yuy2_stride,
    )
}

/// Convert UYVY (YUV Packed) format to YUV 444 planar format.
///
/// This function takes UYVY (4:2:2) format data with 8-16 bit precision,
/// and converts it to YUV 444 planar format with 8-16 bit per channel precision.
///
/// # Arguments
///
/// * `planar_image` - Target planar image.
/// * `yuy2_store` - A slice to store the converted UYVY data.
/// * `yuy2_stride` - The stride (elements per row) for the UYVY plane.
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input UYVY data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn uyvy422_to_yuv444_p16(
    planar_image: &mut YuvPlanarImageMut<u16>,
    yuy2_store: &[u16],
    yuy2_stride: u32,
) -> Result<(), YuvError> {
    yuy2_to_yuv_impl::<{ YuvChromaSubsample::Yuv444 as u8 }, { Yuy2Description::UYVY as usize }>(
        planar_image,
        yuy2_store,
        yuy2_stride,
    )
}

/// Convert UYVY (YUV Packed) format to YUV 420 planar format.
///
/// This function takes UYVY (4:2:2) format data with 8-16 bit precision,
/// and converts it to YUV 444 planar format with 8-16 bit per channel precision.
///
/// # Arguments
///
/// * `planar_image` - Target planar image.
/// * `yuy2_store` - A slice to store the converted UYVY data.
/// * `yuy2_stride` - The stride (elements per row) for the UYVY plane.
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input UYVY data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn uyvy422_to_yuv420_p16(
    planar_image: &mut YuvPlanarImageMut<u16>,
    yuy2_store: &[u16],
    yuy2_stride: u32,
) -> Result<(), YuvError> {
    yuy2_to_yuv_impl::<{ YuvChromaSubsample::Yuv420 as u8 }, { Yuy2Description::UYVY as usize }>(
        planar_image,
        yuy2_store,
        yuy2_stride,
    )
}

/// Convert UYVY (YUV Packed) format to YUV 422 planar format.
///
/// This function takes UYVY (4:2:2) format data with 8-16 bit precision,
/// and converts it to YUV 444 planar format with 8-16 bit per channel precision.
///
/// # Arguments
///
/// * `planar_image` - Target planar image.
/// * `yuy2_store` - A slice to store the converted UYVY data.
/// * `yuy2_stride` - The stride (elements per row) for the UYVY plane.
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input UYVY data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn uyvy422_to_yuv422_p16(
    planar_image: &mut YuvPlanarImageMut<u16>,
    yuy2_store: &[u16],
    yuy2_stride: u32,
) -> Result<(), YuvError> {
    yuy2_to_yuv_impl::<{ YuvChromaSubsample::Yuv422 as u8 }, { Yuy2Description::UYVY as usize }>(
        planar_image,
        yuy2_store,
        yuy2_stride,
    )
}
