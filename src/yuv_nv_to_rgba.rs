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
use crate::avx2::avx2_yuv_nv_to_rgba_row;
#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    feature = "nightly_avx512"
))]
use crate::avx512bw::avx512_yuv_nv_to_rgba;
#[allow(unused_imports)]
use crate::internals::*;
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use crate::neon::neon_yuv_nv_to_rgba_row;
use crate::numerics::qrshr;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use crate::sse::sse_yuv_nv_to_rgba;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
use crate::wasm32::wasm_yuv_nv_to_rgba_row;
use crate::yuv_support::*;
use crate::{YuvBiPlanarImage, YuvError};
#[cfg(feature = "rayon")]
use rayon::iter::{IndexedParallelIterator, ParallelIterator};
#[cfg(feature = "rayon")]
use rayon::prelude::ParallelSliceMut;

fn yuv_nv12_to_rgbx<
    const UV_ORDER: u8,
    const DESTINATION_CHANNELS: u8,
    const YUV_CHROMA_SAMPLING: u8,
>(
    bi_planar_image: &YuvBiPlanarImage<u8>,
    bgra: &mut [u8],
    bgra_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) -> Result<(), YuvError> {
    let order: YuvNVOrder = UV_ORDER.into();
    let dst_chans: YuvSourceChannels = DESTINATION_CHANNELS.into();
    let chroma_subsampling: YuvChromaSample = YUV_CHROMA_SAMPLING.into();

    bi_planar_image.check_constraints(chroma_subsampling)?;

    let range = get_yuv_range(8, range);
    let channels = dst_chans.get_channels_count();
    let kr_kb = matrix.get_kr_kb();
    let transform = get_inverse_transform(255, range.range_y, range.range_uv, kr_kb.kr, kr_kb.kb);
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    const PRECISION: i32 = 6;
    #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
    const PRECISION: i32 = 12;
    let inverse_transform = transform.to_integers(PRECISION as u32);
    let cr_coef = inverse_transform.cr_coef;
    let cb_coef = inverse_transform.cb_coef;
    let y_coef = inverse_transform.y_coef;
    let g_coef_1 = inverse_transform.g_coeff_1;
    let g_coef_2 = inverse_transform.g_coeff_2;

    let bias_y = range.bias_y as i32;
    let bias_uv = range.bias_uv as i32;

    let dst_offset = 0usize;

    let iterator_step = match chroma_subsampling {
        YuvChromaSample::Yuv420 => 2usize,
        YuvChromaSample::Yuv422 => 2usize,
        YuvChromaSample::Yuv444 => 1usize,
    };

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    let mut _use_avx2 = std::arch::is_x86_feature_detected!("avx2");
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    let mut _use_sse = std::arch::is_x86_feature_detected!("sse4.1");
    #[cfg(all(
        any(target_arch = "x86", target_arch = "x86_64"),
        feature = "nightly_avx512"
    ))]
    let mut _use_avx512 = std::arch::is_x86_feature_detected!("avx512bw");

    let iter;
    #[cfg(feature = "rayon")]
    {
        iter = bgra.par_chunks_exact_mut(bgra_stride as usize);
    }
    #[cfg(not(feature = "rayon"))]
    {
        iter = bgra.chunks_exact_mut(bgra_stride as usize);
    }

    let y_stride = bi_planar_image.y_stride;
    let uv_stride = bi_planar_image.uv_stride;
    let y_plane = bi_planar_image.y_plane;
    let uv_plane = bi_planar_image.uv_plane;
    let width = bi_planar_image.width;

    iter.enumerate().for_each(|(y, bgra)| unsafe {
        let y_offset = y * (y_stride as usize);
        let uv_offset = if chroma_subsampling == YuvChromaSample::Yuv420 {
            (y >> 1) * (uv_stride as usize)
        } else {
            y * (uv_stride as usize)
        };

        #[allow(unused_variables)]
        #[allow(unused_mut)]
        let mut cx = 0usize;

        #[allow(unused_variables)]
        #[allow(unused_mut)]
        let mut ux = 0usize;

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            #[cfg(feature = "nightly_avx512")]
            if _use_avx512 {
                let processed =
                    avx512_yuv_nv_to_rgba::<UV_ORDER, DESTINATION_CHANNELS, YUV_CHROMA_SAMPLING>(
                        &range,
                        &inverse_transform,
                        y_plane,
                        uv_plane,
                        bgra,
                        cx,
                        ux,
                        y_offset,
                        uv_offset,
                        dst_offset,
                        width as usize,
                    );
                cx = processed.cx;
                ux = processed.ux;
            }

            if _use_avx2 {
                let processed =
                    avx2_yuv_nv_to_rgba_row::<UV_ORDER, DESTINATION_CHANNELS, YUV_CHROMA_SAMPLING>(
                        &range,
                        &inverse_transform,
                        y_plane,
                        uv_plane,
                        bgra,
                        cx,
                        ux,
                        y_offset,
                        uv_offset,
                        dst_offset,
                        width as usize,
                    );
                cx = processed.cx;
                ux = processed.ux;
            }

            if _use_sse {
                let processed =
                    sse_yuv_nv_to_rgba::<UV_ORDER, DESTINATION_CHANNELS, YUV_CHROMA_SAMPLING>(
                        &range,
                        &inverse_transform,
                        y_plane,
                        uv_plane,
                        bgra,
                        cx,
                        ux,
                        y_offset,
                        uv_offset,
                        dst_offset,
                        width as usize,
                    );
                cx = processed.cx;
                ux = processed.ux;
            }
        }

        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        {
            let processed =
                neon_yuv_nv_to_rgba_row::<UV_ORDER, DESTINATION_CHANNELS, YUV_CHROMA_SAMPLING>(
                    &range,
                    &inverse_transform,
                    y_plane,
                    uv_plane,
                    bgra,
                    cx,
                    ux,
                    y_offset,
                    uv_offset,
                    dst_offset,
                    width as usize,
                );
            cx = processed.cx;
            ux = processed.ux;
        }

        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        {
            let processed =
                wasm_yuv_nv_to_rgba_row::<UV_ORDER, DESTINATION_CHANNELS, YUV_CHROMA_SAMPLING>(
                    &range,
                    &inverse_transform,
                    y_plane,
                    uv_plane,
                    bgra,
                    cx,
                    ux,
                    y_offset,
                    uv_offset,
                    dst_offset,
                    width as usize,
                );
            cx = processed.cx;
            ux = processed.ux;
        }

        for x in (cx..width as usize).step_by(iterator_step) {
            let y_value = (*y_plane.get_unchecked(y_offset + x) as i32 - bias_y) * y_coef;
            let cb_pos = uv_offset + ux;
            let cb_value: i32 =
                *uv_plane.get_unchecked(cb_pos + order.get_u_position()) as i32 - bias_uv;
            let cr_value: i32 =
                *uv_plane.get_unchecked(cb_pos + order.get_v_position()) as i32 - bias_uv;

            let r = qrshr::<PRECISION, 8>(y_value + cr_coef * cr_value);
            let b = qrshr::<PRECISION, 8>(y_value + cb_coef * cb_value);
            let g = qrshr::<PRECISION, 8>(y_value - g_coef_1 * cr_value - g_coef_2 * cb_value);

            let px = x * channels;

            let dst_shift = dst_offset + px;

            let dst_slice = bgra.get_unchecked_mut(dst_shift..);
            *dst_slice.get_unchecked_mut(dst_chans.get_b_channel_offset()) = b as u8;
            *dst_slice.get_unchecked_mut(dst_chans.get_g_channel_offset()) = g as u8;
            *dst_slice.get_unchecked_mut(dst_chans.get_r_channel_offset()) = r as u8;
            if dst_chans.has_alpha() {
                *dst_slice.get_unchecked_mut(dst_chans.get_a_channel_offset()) = 255;
            }

            if chroma_subsampling == YuvChromaSample::Yuv422
                || chroma_subsampling == YuvChromaSample::Yuv420
            {
                let next_px = x + 1;
                if next_px < width as usize {
                    let y_value =
                        (*y_plane.get_unchecked(y_offset + next_px) as i32 - bias_y) * y_coef;

                    let r = qrshr::<PRECISION, 8>(y_value + cr_coef * cr_value);
                    let b = qrshr::<PRECISION, 8>(y_value + cb_coef * cb_value);
                    let g =
                        qrshr::<PRECISION, 8>(y_value - g_coef_1 * cr_value - g_coef_2 * cb_value);

                    let next_px = next_px * channels;
                    let dst_shift = dst_offset + next_px;
                    let dst_slice = bgra.get_unchecked_mut(dst_shift..);
                    *dst_slice.get_unchecked_mut(dst_chans.get_b_channel_offset()) = b as u8;
                    *dst_slice.get_unchecked_mut(dst_chans.get_g_channel_offset()) = g as u8;
                    *dst_slice.get_unchecked_mut(dst_chans.get_r_channel_offset()) = r as u8;
                    if dst_chans.has_alpha() {
                        *dst_slice.get_unchecked_mut(dst_chans.get_a_channel_offset()) = 255;
                    }
                }
            }

            ux += 2;
        }
    });

    Ok(())
}

/// Convert YUV NV12 format to BGRA format.
///
/// This function takes YUV NV12 data with 8-bit precision,
/// and converts it to BGRA format with 8-bit per channel precision.
///
/// # Arguments
///
/// * `bi_planar_image` - Source Bi-Planar image.
/// * `bgra` - A mutable slice to store the converted BGRA data.
/// * `bgra_stride` - The stride (bytes per row) for the BGRA image data.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGRA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv_nv12_to_bgra(
    bi_planar_image: &YuvBiPlanarImage<u8>,
    bgra: &mut [u8],
    bgra_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) -> Result<(), YuvError> {
    yuv_nv12_to_rgbx::<
        { YuvNVOrder::UV as u8 },
        { YuvSourceChannels::Bgra as u8 },
        { YuvChromaSample::Yuv420 as u8 },
    >(bi_planar_image, bgra, bgra_stride, range, matrix)
}

/// Convert YUV NV16 format to BGRA format.
///
/// This function takes YUV NV16 data with 8-bit precision,
/// and converts it to BGRA format with 8-bit per channel precision.
///
/// # Arguments
///
/// * `bi_planar_image` - Source Bi-Planar image.
/// * `bgra` - A mutable slice to store the converted BGRA data.
/// * `bgra_stride` - The stride (bytes per row) for the BGRA image data.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGRA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv_nv16_to_bgra(
    bi_planar_image: &YuvBiPlanarImage<u8>,
    bgra: &mut [u8],
    bgra_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) -> Result<(), YuvError> {
    yuv_nv12_to_rgbx::<
        { YuvNVOrder::UV as u8 },
        { YuvSourceChannels::Bgra as u8 },
        { YuvChromaSample::Yuv422 as u8 },
    >(bi_planar_image, bgra, bgra_stride, range, matrix)
}

/// Convert YUV NV61 format to BGRA format.
///
/// This function takes YUV NV61 data with 8-bit precision,
/// and converts it to BGRA format with 8-bit per channel precision.
///
/// # Arguments
///
/// * `bi_planar_image` - Source Bi-Planar image.
/// * `bgra` - A mutable slice to store the converted BGRA data.
/// * `bgra_stride` - The stride (bytes per row) for the BGRA image data.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGRA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv_nv61_to_bgra(
    bi_planar_image: &YuvBiPlanarImage<u8>,
    bgra: &mut [u8],
    bgra_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) -> Result<(), YuvError> {
    yuv_nv12_to_rgbx::<
        { YuvNVOrder::VU as u8 },
        { YuvSourceChannels::Bgra as u8 },
        { YuvChromaSample::Yuv422 as u8 },
    >(bi_planar_image, bgra, bgra_stride, range, matrix)
}

/// Convert YUV NV21 format to BGRA format.
///
/// This function takes YUV NV12 data with 8-bit precision,
/// and converts it to BGRA format with 8-bit per channel precision.
///
/// # Arguments
///
/// * `bi_planar_image` - Source Bi-Planar image.
/// * `bgra` - A mutable slice to store the converted BGRA data.
/// * `bgra_stride` - The stride (bytes per row) for the BGRA image data.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGRA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv_nv21_to_bgra(
    bi_planar_image: &YuvBiPlanarImage<u8>,
    bgra: &mut [u8],
    bgra_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) -> Result<(), YuvError> {
    yuv_nv12_to_rgbx::<
        { YuvNVOrder::VU as u8 },
        { YuvSourceChannels::Bgra as u8 },
        { YuvChromaSample::Yuv420 as u8 },
    >(bi_planar_image, bgra, bgra_stride, range, matrix)
}

/// Convert YUV NV16 format to RGBA format.
///
/// This function takes YUV NV16 data with 8-bit precision,
/// and converts it to RGBA format with 8-bit per channel precision.
///
/// # Arguments
///
/// * `bi_planar_image` - Source Bi-Planar image.
/// * `rgba` - A mutable slice to store the converted RGBA data.
/// * `rgba_stride` - The stride (bytes per row) for the RGBA image data.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGRA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv_nv16_to_rgba(
    bi_planar_image: &YuvBiPlanarImage<u8>,
    rgba: &mut [u8],
    rgba_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) -> Result<(), YuvError> {
    yuv_nv12_to_rgbx::<
        { YuvNVOrder::UV as u8 },
        { YuvSourceChannels::Rgba as u8 },
        { YuvChromaSample::Yuv422 as u8 },
    >(bi_planar_image, rgba, rgba_stride, range, matrix)
}

/// Convert YUV NV61 format to RGBA format.
///
/// This function takes YUV NV61 data with 8-bit precision,
/// and converts it to RGBA format with 8-bit per channel precision.
///
/// # Arguments
///
/// * `bi_planar_image` - Source Bi-Planar image.
/// * `rgba` - A mutable slice to store the converted RGBA data.
/// * `rgba_stride` - The stride (bytes per row) for the RGBA image data.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input RGBA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv_nv61_to_rgba(
    bi_planar_image: &YuvBiPlanarImage<u8>,
    rgba: &mut [u8],
    rgba_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) -> Result<(), YuvError> {
    yuv_nv12_to_rgbx::<
        { YuvNVOrder::VU as u8 },
        { YuvSourceChannels::Rgba as u8 },
        { YuvChromaSample::Yuv422 as u8 },
    >(bi_planar_image, rgba, rgba_stride, range, matrix)
}

/// Convert YUV NV12 format to RGBA format.
///
/// This function takes YUV NV12 data with 8-bit precision,
/// and converts it to RGBA format with 8-bit per channel precision.
///
/// # Arguments
///
/// * `bi_planar_image` - Source Bi-Planar image.
/// * `rgba` - A mutable slice to store the converted RGBA data.
/// * `rgba_stride` - The stride (bytes per row) for the RGBA image data.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input RGBA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv_nv12_to_rgba(
    bi_planar_image: &YuvBiPlanarImage<u8>,
    rgba: &mut [u8],
    rgba_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) -> Result<(), YuvError> {
    yuv_nv12_to_rgbx::<
        { YuvNVOrder::UV as u8 },
        { YuvSourceChannels::Rgba as u8 },
        { YuvChromaSample::Yuv420 as u8 },
    >(bi_planar_image, rgba, rgba_stride, range, matrix)
}

/// Convert YUV NV21 format to RGBA format.
///
/// This function takes YUV NV21 data with 8-bit precision,
/// and converts it to RGBA format with 8-bit per channel precision.
///
/// # Arguments
///
/// * `bi_planar_image` - Source Bi-Planar image.
/// * `rgba` - A mutable slice to store the converted RGBA data.
/// * `rgba_stride` - The stride (bytes per row) for the RGBA image data.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input RGBA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv_nv21_to_rgba(
    bi_planar_image: &YuvBiPlanarImage<u8>,
    rgba: &mut [u8],
    rgba_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) -> Result<(), YuvError> {
    yuv_nv12_to_rgbx::<
        { YuvNVOrder::VU as u8 },
        { YuvSourceChannels::Rgba as u8 },
        { YuvChromaSample::Yuv420 as u8 },
    >(bi_planar_image, rgba, rgba_stride, range, matrix)
}

/// Convert YUV NV12 format to RGB format.
///
/// This function takes YUV NV12 data with 8-bit precision,
/// and converts it to RGB format with 8-bit per channel precision.
///
/// # Arguments
///
/// * `bi_planar_image` - Source Bi-Planar image.
/// * `rgb` - A mutable slice to store the converted RGB data.
/// * `rgb_stride` - The stride (bytes per row) for the RGB image data.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input RGB data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv_nv12_to_rgb(
    bi_planar_image: &YuvBiPlanarImage<u8>,
    rgb: &mut [u8],
    rgb_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) -> Result<(), YuvError> {
    yuv_nv12_to_rgbx::<
        { YuvNVOrder::UV as u8 },
        { YuvSourceChannels::Rgb as u8 },
        { YuvChromaSample::Yuv420 as u8 },
    >(bi_planar_image, rgb, rgb_stride, range, matrix)
}

/// Convert YUV NV12 format to BGR format.
///
/// This function takes YUV NV12 data with 8-bit precision,
/// and converts it to BGR format with 8-bit per channel precision.
///
/// # Arguments
///
/// * `bi_planar_image` - Source Bi-Planar image.
/// * `bgr` - A mutable slice to store the converted BGR data.
/// * `bgr_stride` - The stride (bytes per row) for the BGR image data.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGR data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv_nv12_to_bgr(
    bi_planar_image: &YuvBiPlanarImage<u8>,
    bgr: &mut [u8],
    bgr_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) -> Result<(), YuvError> {
    yuv_nv12_to_rgbx::<
        { YuvNVOrder::UV as u8 },
        { YuvSourceChannels::Bgr as u8 },
        { YuvChromaSample::Yuv420 as u8 },
    >(bi_planar_image, bgr, bgr_stride, range, matrix)
}

/// Convert YUV NV16 format to RGB format.
///
/// This function takes YUV NV16 data with 8-bit precision,
/// and converts it to RGB format with 8-bit per channel precision.
///
/// # Arguments
///
/// * `bi_planar_image` - Source Bi-Planar image.
/// * `rgb` - A mutable slice to store the converted RGB data.
/// * `rgb_stride` - The stride (bytes per row) for the RGB image data.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input RGB data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv_nv16_to_rgb(
    bi_planar_image: &YuvBiPlanarImage<u8>,
    rgb: &mut [u8],
    rgb_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) -> Result<(), YuvError> {
    yuv_nv12_to_rgbx::<
        { YuvNVOrder::UV as u8 },
        { YuvSourceChannels::Rgb as u8 },
        { YuvChromaSample::Yuv422 as u8 },
    >(bi_planar_image, rgb, rgb_stride, range, matrix)
}

/// Convert YUV NV16 format to BGR format.
///
/// This function takes YUV NV16 data with 8-bit precision,
/// and converts it to BGR format with 8-bit per channel precision.
///
/// # Arguments
///
/// * `bi_planar_image` - Source Bi-Planar image.
/// * `bgr` - A mutable slice to store the converted BGR data.
/// * `bgr_stride` - The stride (bytes per row) for the BGR image data.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGR data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv_nv16_to_bgr(
    bi_planar_image: &YuvBiPlanarImage<u8>,
    bgr: &mut [u8],
    bgr_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) -> Result<(), YuvError> {
    yuv_nv12_to_rgbx::<
        { YuvNVOrder::UV as u8 },
        { YuvSourceChannels::Bgr as u8 },
        { YuvChromaSample::Yuv422 as u8 },
    >(bi_planar_image, bgr, bgr_stride, range, matrix)
}

/// Convert YUV NV61 format to RGB format.
///
/// This function takes YUV NV61 data with 8-bit precision,
/// and converts it to RGB format with 8-bit per channel precision.
///
/// # Arguments
///
/// * `bi_planar_image` - Source Bi-Planar image.
/// * `rgb` - A mutable slice to store the converted RGB data.
/// * `rgb_stride` - The stride (bytes per row) for the RGB image data.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input RGB data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv_nv61_to_rgb(
    bi_planar_image: &YuvBiPlanarImage<u8>,
    rgb: &mut [u8],
    rgb_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) -> Result<(), YuvError> {
    yuv_nv12_to_rgbx::<
        { YuvNVOrder::VU as u8 },
        { YuvSourceChannels::Rgb as u8 },
        { YuvChromaSample::Yuv422 as u8 },
    >(bi_planar_image, rgb, rgb_stride, range, matrix)
}

/// Convert YUV NV61 format to BGR format.
///
/// This function takes YUV NV61 data with 8-bit precision,
/// and converts it to BGR format with 8-bit per channel precision.
///
/// # Arguments
///
/// * `bi_planar_image` - Source Bi-Planar image.
/// * `bgr` - A mutable slice to store the converted BGR data.
/// * `bgr_stride` - The stride (bytes per row) for the BGR image data.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGR data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv_nv61_to_bgr(
    bi_planar_image: &YuvBiPlanarImage<u8>,
    bgr: &mut [u8],
    bgr_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) -> Result<(), YuvError> {
    yuv_nv12_to_rgbx::<
        { YuvNVOrder::VU as u8 },
        { YuvSourceChannels::Bgr as u8 },
        { YuvChromaSample::Yuv422 as u8 },
    >(bi_planar_image, bgr, bgr_stride, range, matrix)
}

/// Convert YUV NV21 format to RGB format.
///
/// This function takes YUV NV21 data with 8-bit precision,
/// and converts it to RGB format with 8-bit per channel precision.
///
/// # Arguments
///
/// * `bi_planar_image` - Source Bi-Planar image.
/// * `rgb` - A mutable slice to store the converted RGB data.
/// * `rgb_stride` - The stride (bytes per row) for the RGB image data.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input RGB data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv_nv21_to_rgb(
    bi_planar_image: &YuvBiPlanarImage<u8>,
    rgb: &mut [u8],
    rgb_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) -> Result<(), YuvError> {
    yuv_nv12_to_rgbx::<
        { YuvNVOrder::VU as u8 },
        { YuvSourceChannels::Rgb as u8 },
        { YuvChromaSample::Yuv420 as u8 },
    >(bi_planar_image, rgb, rgb_stride, range, matrix)
}

/// Convert YUV NV21 format to BGR format.
///
/// This function takes YUV NV21 data with 8-bit precision,
/// and converts it to BGR format with 8-bit per channel precision.
///
/// # Arguments
///
/// * `bi_planar_image` - Source Bi-Planar image.
/// * `bgr` - A mutable slice to store the converted BGR data.
/// * `bgr_stride` - The stride (bytes per row) for the BGR image data.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGR data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv_nv21_to_bgr(
    bi_planar_image: &YuvBiPlanarImage<u8>,
    bgr: &mut [u8],
    bgr_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) -> Result<(), YuvError> {
    yuv_nv12_to_rgbx::<
        { YuvNVOrder::VU as u8 },
        { YuvSourceChannels::Bgr as u8 },
        { YuvChromaSample::Yuv420 as u8 },
    >(bi_planar_image, bgr, bgr_stride, range, matrix)
}

/// Convert YUV NV24 format to RGBA format.
///
/// This function takes YUV NV24 data with 8-bit precision,
/// and converts it to RGBA format with 8-bit per channel precision.
///
/// # Arguments
///
/// * `bi_planar_image` - Source Bi-Planar image.
/// * `rgba` - A mutable slice to store the converted RGBA data.
/// * `rgba_stride` - The stride (bytes per row) for the RGBA image data.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGRA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv_nv42_to_rgba(
    bi_planar_image: &YuvBiPlanarImage<u8>,
    rgba: &mut [u8],
    rgba_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) -> Result<(), YuvError> {
    yuv_nv12_to_rgbx::<
        { YuvNVOrder::VU as u8 },
        { YuvSourceChannels::Rgba as u8 },
        { YuvChromaSample::Yuv444 as u8 },
    >(bi_planar_image, rgba, rgba_stride, range, matrix)
}

/// Convert YUV NV24 format to RGB format.
///
/// This function takes YUV NV24 data with 8-bit precision,
/// and converts it to RGB format with 8-bit per channel precision.
///
/// # Arguments
///
/// * `bi_planar_image` - Source Bi-Planar image.
/// * `rgb` - A mutable slice to store the converted RGB data.
/// * `rgb_stride` - The stride (bytes per row) for the RGB image data.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input RGB data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv_nv24_to_rgb(
    bi_planar_image: &YuvBiPlanarImage<u8>,
    rgb: &mut [u8],
    rgb_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) -> Result<(), YuvError> {
    yuv_nv12_to_rgbx::<
        { YuvNVOrder::UV as u8 },
        { YuvSourceChannels::Rgb as u8 },
        { YuvChromaSample::Yuv444 as u8 },
    >(bi_planar_image, rgb, rgb_stride, range, matrix)
}

/// Convert YUV NV24 format to BGR format.
///
/// This function takes YUV NV24 data with 8-bit precision,
/// and converts it to BGR format with 8-bit per channel precision.
///
/// # Arguments
///
/// * `bi_planar_image` - Source Bi-Planar image.
/// * `bgr` - A mutable slice to store the converted BGR data.
/// * `bgr_stride` - The stride (bytes per row) for the BGR image data.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGR data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv_nv24_to_bgr(
    bi_planar_image: &YuvBiPlanarImage<u8>,
    bgr: &mut [u8],
    bgr_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) -> Result<(), YuvError> {
    yuv_nv12_to_rgbx::<
        { YuvNVOrder::UV as u8 },
        { YuvSourceChannels::Bgr as u8 },
        { YuvChromaSample::Yuv444 as u8 },
    >(bi_planar_image, bgr, bgr_stride, range, matrix)
}

/// Convert YUV NV24 format to RGBA format.
///
/// This function takes YUV NV24 data with 8-bit precision,
/// and converts it to RGBA format with 8-bit per channel precision.
///
/// # Arguments
///
/// * `bi_planar_image` - Source Bi-Planar image.
/// * `rgba` - A mutable slice to store the converted RGBA data.
/// * `rgba_stride` - The stride (bytes per row) for the RGBA image data.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input RGBA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv_nv24_to_rgba(
    bi_planar_image: &YuvBiPlanarImage<u8>,
    rgb: &mut [u8],
    rgb_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) -> Result<(), YuvError> {
    yuv_nv12_to_rgbx::<
        { YuvNVOrder::UV as u8 },
        { YuvSourceChannels::Rgba as u8 },
        { YuvChromaSample::Yuv444 as u8 },
    >(bi_planar_image, rgb, rgb_stride, range, matrix)
}

/// Convert YUV NV24 format to BGRA format.
///
/// This function takes YUV NV24 data with 8-bit precision,
/// and converts it to RGBA format with 8-bit per channel precision.
///
/// # Arguments
///
/// * `bi_planar_image` - Source Bi-Planar image.
/// * `bgra` - A mutable slice to store the converted BGRA data.
/// * `bgra_stride` - The stride (bytes per row) for the BGRA image data.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGRA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv_nv24_to_bgra(
    bi_planar_image: &YuvBiPlanarImage<u8>,
    rgb: &mut [u8],
    rgb_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) -> Result<(), YuvError> {
    yuv_nv12_to_rgbx::<
        { YuvNVOrder::UV as u8 },
        { YuvSourceChannels::Bgra as u8 },
        { YuvChromaSample::Yuv444 as u8 },
    >(bi_planar_image, rgb, rgb_stride, range, matrix)
}

/// Convert YUV NV42 format to RGB format.
///
/// This function takes YUV NV42 data with 8-bit precision,
/// and converts it to RGB format with 8-bit per channel precision.
///
/// # Arguments
///
/// * `bi_planar_image` - Source Bi-Planar image.
/// * `rgb` - A mutable slice to store the converted RGB data.
/// * `rgb_stride` - The stride (bytes per row) for the RGB image data.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input RGB data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv_nv42_to_rgb(
    bi_planar_image: &YuvBiPlanarImage<u8>,
    rgb: &mut [u8],
    rgb_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) -> Result<(), YuvError> {
    yuv_nv12_to_rgbx::<
        { YuvNVOrder::VU as u8 },
        { YuvSourceChannels::Rgb as u8 },
        { YuvChromaSample::Yuv444 as u8 },
    >(bi_planar_image, rgb, rgb_stride, range, matrix)
}

/// Convert YUV NV42 format to BGR format.
///
/// This function takes YUV NV42 data with 8-bit precision,
/// and converts it to BGR format with 8-bit per channel precision.
///
/// # Arguments
///
/// * `bi_planar_image` - Source Bi-Planar image.
/// * `bgr` - A mutable slice to store the converted BGR data.
/// * `bgr_stride` - The stride (bytes per row) for the BGR image data.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGR data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv_nv42_to_bgr(
    bi_planar_image: &YuvBiPlanarImage<u8>,
    bgr: &mut [u8],
    bgr_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) -> Result<(), YuvError> {
    yuv_nv12_to_rgbx::<
        { YuvNVOrder::VU as u8 },
        { YuvSourceChannels::Bgr as u8 },
        { YuvChromaSample::Yuv444 as u8 },
    >(bi_planar_image, bgr, bgr_stride, range, matrix)
}

/// Convert YUV NV42 format to BGRA format.
///
/// This function takes YUV NV42 data with 8-bit precision,
/// and converts it to RGB format with 8-bit per channel precision.
///
/// # Arguments
///
/// * `bi_planar_image` - Source Bi-Planar image.
/// * `bgra` - A mutable slice to store the converted BGRA data.
/// * `bgra_stride` - The stride (bytes per row) for the BGRA image data.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGRA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv_nv42_to_bgra(
    bi_planar_image: &YuvBiPlanarImage<u8>,
    rgb: &mut [u8],
    rgb_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) -> Result<(), YuvError> {
    yuv_nv12_to_rgbx::<
        { YuvNVOrder::VU as u8 },
        { YuvSourceChannels::Bgra as u8 },
        { YuvChromaSample::Yuv444 as u8 },
    >(bi_planar_image, rgb, rgb_stride, range, matrix)
}
