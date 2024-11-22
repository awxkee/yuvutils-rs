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
use crate::neon::{neon_yuv_nv_to_rgba_row, neon_yuv_nv_to_rgba_row_rdm};
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
use rayon::prelude::{ParallelSlice, ParallelSliceMut};

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
    let chroma_subsampling: YuvChromaSubsampling = YUV_CHROMA_SAMPLING.into();

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

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    let mut _use_avx2 = std::arch::is_x86_feature_detected!("avx2");
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    let mut _use_sse = std::arch::is_x86_feature_detected!("sse4.1");
    #[cfg(all(
        any(target_arch = "x86", target_arch = "x86_64"),
        feature = "nightly_avx512"
    ))]
    let mut _use_avx512 = std::arch::is_x86_feature_detected!("avx512bw");
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    let is_rdm_available = std::arch::is_aarch64_feature_detected!("rdm");
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    let neon_wide_row = if is_rdm_available {
        neon_yuv_nv_to_rgba_row_rdm::<UV_ORDER, DESTINATION_CHANNELS, YUV_CHROMA_SAMPLING>
    } else {
        neon_yuv_nv_to_rgba_row::<PRECISION, UV_ORDER, DESTINATION_CHANNELS, YUV_CHROMA_SAMPLING>
    };

    let width = bi_planar_image.width;

    let process_wide_row = |_bgra: &mut [u8], _y_plane: &[u8], _uv_plane: &[u8]| {
        let mut _offset = ProcessedOffset { cx: 0, ux: 0 };
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            #[cfg(feature = "nightly_avx512")]
            if _use_avx512 {
                let processed =
                    avx512_yuv_nv_to_rgba::<UV_ORDER, DESTINATION_CHANNELS, YUV_CHROMA_SAMPLING>(
                        &range,
                        &inverse_transform,
                        _y_plane,
                        _uv_plane,
                        _bgra,
                        _offset.cx,
                        _offset.ux,
                        width as usize,
                    );
                _offset = processed;
            }

            if _use_avx2 {
                let processed =
                    avx2_yuv_nv_to_rgba_row::<UV_ORDER, DESTINATION_CHANNELS, YUV_CHROMA_SAMPLING>(
                        &range,
                        &inverse_transform,
                        _y_plane,
                        _uv_plane,
                        _bgra,
                        _offset.cx,
                        _offset.ux,
                        width as usize,
                    );
                _offset = processed;
            }

            if _use_sse {
                let processed =
                    sse_yuv_nv_to_rgba::<UV_ORDER, DESTINATION_CHANNELS, YUV_CHROMA_SAMPLING>(
                        &range,
                        &inverse_transform,
                        _y_plane,
                        _uv_plane,
                        _bgra,
                        _offset.cx,
                        _offset.ux,
                        width as usize,
                    );
                _offset = processed;
            }
        }

        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        {
            unsafe {
                let processed = neon_wide_row(
                    &range,
                    &inverse_transform,
                    _y_plane,
                    _uv_plane,
                    _bgra,
                    _offset.cx,
                    _offset.ux,
                    width as usize,
                );
                _offset = processed;
            }
        }

        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        {
            unsafe {
                let processed =
                    wasm_yuv_nv_to_rgba_row::<UV_ORDER, DESTINATION_CHANNELS, YUV_CHROMA_SAMPLING>(
                        &range,
                        &inverse_transform,
                        _y_plane,
                        _uv_plane,
                        _bgra,
                        _offset.cx,
                        _offset.ux,
                        0,
                        0,
                        0,
                        width as usize,
                    );
                _offset = processed;
            }
        }

        _offset
    };

    let process_halved_chroma_row = |y_src: &[u8], uv_src: &[u8], rgba: &mut [u8]| {
        let processed = process_wide_row(rgba, y_src, uv_src);

        for ((rgba, y_src), uv_src) in rgba
            .chunks_exact_mut(channels * 2)
            .zip(y_src.chunks_exact(2))
            .zip(uv_src.chunks_exact(2))
            .skip(processed.cx / 2)
        {
            let y_vl0 = y_src[0] as i32;
            let mut cb_value = uv_src[order.get_u_position()] as i32;
            let mut cr_value = uv_src[order.get_v_position()] as i32;

            let y_value0: i32 = (y_vl0 - bias_y) * y_coef;

            cb_value -= bias_uv;
            cr_value -= bias_uv;

            let r0 = qrshr::<PRECISION, 8>(y_value0 + cr_coef * cr_value);
            let b0 = qrshr::<PRECISION, 8>(y_value0 + cb_coef * cb_value);
            let g0 = qrshr::<PRECISION, 8>(y_value0 - g_coef_1 * cr_value - g_coef_2 * cb_value);

            rgba[dst_chans.get_b_channel_offset()] = b0 as u8;
            rgba[dst_chans.get_g_channel_offset()] = g0 as u8;
            rgba[dst_chans.get_r_channel_offset()] = r0 as u8;

            if dst_chans.has_alpha() {
                rgba[dst_chans.get_a_channel_offset()] = 255;
            }

            let y_vl1 = y_src[1] as i32;

            let y_value1: i32 = (y_vl1 - bias_y) * y_coef;

            let r1 = qrshr::<PRECISION, 8>(y_value1 + cr_coef * cr_value);
            let b1 = qrshr::<PRECISION, 8>(y_value1 + cb_coef * cb_value);
            let g1 = qrshr::<PRECISION, 8>(y_value1 - g_coef_1 * cr_value - g_coef_2 * cb_value);

            rgba[channels + dst_chans.get_b_channel_offset()] = b1 as u8;
            rgba[channels + dst_chans.get_g_channel_offset()] = g1 as u8;
            rgba[channels + dst_chans.get_r_channel_offset()] = r1 as u8;

            if dst_chans.has_alpha() {
                rgba[channels + dst_chans.get_a_channel_offset()] = 255;
            }
        }

        if width & 1 != 0 {
            let rgba = rgba.chunks_exact_mut(channels * 2).into_remainder();
            let rgba = &mut rgba[0..channels];
            let uv_src = uv_src.chunks_exact(2).last().unwrap();
            let y_src = y_src.chunks_exact(2).remainder();

            let y_vl0 = y_src[0] as i32;
            let y_value0: i32 = (y_vl0 - bias_y) * y_coef;
            let mut cb_value = uv_src[order.get_u_position()] as i32;
            let mut cr_value = uv_src[order.get_v_position()] as i32;

            cb_value -= bias_uv;
            cr_value -= bias_uv;

            let r0 = qrshr::<PRECISION, 8>(y_value0 + cr_coef * cr_value);
            let b0 = qrshr::<PRECISION, 8>(y_value0 + cb_coef * cb_value);
            let g0 = qrshr::<PRECISION, 8>(y_value0 - g_coef_1 * cr_value - g_coef_2 * cb_value);

            rgba[dst_chans.get_b_channel_offset()] = b0 as u8;
            rgba[dst_chans.get_g_channel_offset()] = g0 as u8;
            rgba[dst_chans.get_r_channel_offset()] = r0 as u8;

            if dst_chans.has_alpha() {
                rgba[dst_chans.get_a_channel_offset()] = 255;
            }
        }
    };

    let y_stride = bi_planar_image.y_stride;
    let uv_stride = bi_planar_image.uv_stride;
    let y_plane = bi_planar_image.y_plane;
    let uv_plane = bi_planar_image.uv_plane;

    if chroma_subsampling == YuvChromaSubsampling::Yuv444 {
        let iter;
        #[cfg(feature = "rayon")]
        {
            iter = y_plane
                .par_chunks_exact(y_stride as usize)
                .zip(uv_plane.par_chunks_exact(uv_stride as usize))
                .zip(bgra.par_chunks_exact_mut(bgra_stride as usize));
        }
        #[cfg(not(feature = "rayon"))]
        {
            iter = y_plane
                .chunks_exact(y_stride as usize)
                .zip(uv_plane.chunks_exact(uv_stride as usize))
                .zip(bgra.chunks_exact_mut(bgra_stride as usize));
        }
        iter.for_each(|((y_src, uv_src), rgba)| {
            let processed = process_wide_row(rgba, y_src, uv_src);

            for ((rgba, &y_src), uv_src) in rgba
                .chunks_exact_mut(channels)
                .zip(y_src.iter())
                .zip(uv_src.chunks_exact(2))
                .skip(processed.cx)
            {
                let y_vl = y_src as i32;
                let mut cb_value = uv_src[order.get_u_position()] as i32;
                let mut cr_value = uv_src[order.get_v_position()] as i32;

                let y_value: i32 = (y_vl - bias_y) * y_coef;

                cb_value -= bias_uv;
                cr_value -= bias_uv;

                let r = qrshr::<PRECISION, 8>(y_value + cr_coef * cr_value);
                let b = qrshr::<PRECISION, 8>(y_value + cb_coef * cb_value);
                let g = qrshr::<PRECISION, 8>(y_value - g_coef_1 * cr_value - g_coef_2 * cb_value);

                rgba[dst_chans.get_b_channel_offset()] = b as u8;
                rgba[dst_chans.get_g_channel_offset()] = g as u8;
                rgba[dst_chans.get_r_channel_offset()] = r as u8;

                if dst_chans.has_alpha() {
                    rgba[dst_chans.get_a_channel_offset()] = 255;
                }
            }
        });
    } else if chroma_subsampling == YuvChromaSubsampling::Yuv422 {
        let iter;
        #[cfg(feature = "rayon")]
        {
            iter = y_plane
                .par_chunks_exact(y_stride as usize)
                .zip(uv_plane.par_chunks_exact(uv_stride as usize))
                .zip(bgra.par_chunks_exact_mut(bgra_stride as usize));
        }
        #[cfg(not(feature = "rayon"))]
        {
            iter = y_plane
                .chunks_exact(y_stride as usize)
                .zip(uv_plane.chunks_exact(uv_stride as usize))
                .zip(bgra.chunks_exact_mut(bgra_stride as usize));
        }
        iter.for_each(|((y_src, uv_src), rgba)| {
            process_halved_chroma_row(y_src, uv_src, rgba);
        });
    } else if chroma_subsampling == YuvChromaSubsampling::Yuv420 {
        let iter;
        #[cfg(feature = "rayon")]
        {
            iter = y_plane
                .par_chunks_exact(y_stride as usize * 2)
                .zip(uv_plane.par_chunks_exact(uv_stride as usize))
                .zip(bgra.par_chunks_exact_mut(bgra_stride as usize * 2));
        }
        #[cfg(not(feature = "rayon"))]
        {
            iter = y_plane
                .chunks_exact(y_stride as usize * 2)
                .zip(uv_plane.chunks_exact(uv_stride as usize))
                .zip(bgra.chunks_exact_mut(bgra_stride as usize * 2));
        }
        iter.for_each(|((y_src, uv_src), rgba)| {
            for (y_src, rgba) in y_src
                .chunks_exact(y_stride as usize)
                .zip(rgba.chunks_exact_mut(bgra_stride as usize))
            {
                process_halved_chroma_row(y_src, uv_src, rgba);
            }
        });
        if bi_planar_image.height & 1 != 0 {
            let y_src = y_plane.chunks_exact(y_stride as usize * 2).remainder();
            let uv_src = uv_plane.chunks_exact(uv_stride as usize).last().unwrap();
            let rgba = bgra
                .chunks_exact_mut(bgra_stride as usize * 2)
                .into_remainder();
            process_halved_chroma_row(y_src, uv_src, rgba);
        }
    } else {
        unreachable!();
    }

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
/// * `bgra_stride` - The stride (components per row) for the BGRA image data.
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
        { YuvChromaSubsampling::Yuv420 as u8 },
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
/// * `bgra_stride` - The stride (components per row) for the BGRA image data.
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
        { YuvChromaSubsampling::Yuv422 as u8 },
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
/// * `bgra_stride` - The stride (components per row) for the BGRA image data.
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
        { YuvChromaSubsampling::Yuv422 as u8 },
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
/// * `bgra_stride` - The stride (components per row) for the BGRA image data.
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
        { YuvChromaSubsampling::Yuv420 as u8 },
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
/// * `rgba_stride` - The stride (components per row) for the RGBA image data.
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
        { YuvChromaSubsampling::Yuv422 as u8 },
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
/// * `rgba_stride` - The stride (components per row) for the RGBA image data.
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
        { YuvChromaSubsampling::Yuv422 as u8 },
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
/// * `rgba_stride` - The stride (components per row) for the RGBA image data.
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
        { YuvChromaSubsampling::Yuv420 as u8 },
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
/// * `rgba_stride` - The stride (components per row) for the RGBA image data.
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
        { YuvChromaSubsampling::Yuv420 as u8 },
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
/// * `rgb_stride` - The stride (components per row) for the RGB image data.
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
        { YuvChromaSubsampling::Yuv420 as u8 },
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
/// * `bgr_stride` - The stride (components per row) for the BGR image data.
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
        { YuvChromaSubsampling::Yuv420 as u8 },
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
/// * `rgb_stride` - The stride (components per row) for the RGB image data.
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
        { YuvChromaSubsampling::Yuv422 as u8 },
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
/// * `bgr_stride` - The stride (components per row) for the BGR image data.
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
        { YuvChromaSubsampling::Yuv422 as u8 },
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
/// * `rgb_stride` - The stride (components per row) for the RGB image data.
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
        { YuvChromaSubsampling::Yuv422 as u8 },
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
/// * `bgr_stride` - The stride (components per row) for the BGR image data.
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
        { YuvChromaSubsampling::Yuv422 as u8 },
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
/// * `rgb_stride` - The stride (components per row) for the RGB image data.
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
        { YuvChromaSubsampling::Yuv420 as u8 },
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
/// * `bgr_stride` - The stride (components per row) for the BGR image data.
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
        { YuvChromaSubsampling::Yuv420 as u8 },
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
/// * `rgba_stride` - The stride (components per row) for the RGBA image data.
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
        { YuvChromaSubsampling::Yuv444 as u8 },
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
/// * `rgb_stride` - The stride (components per row) for the RGB image data.
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
        { YuvChromaSubsampling::Yuv444 as u8 },
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
/// * `bgr_stride` - The stride (components per row) for the BGR image data.
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
        { YuvChromaSubsampling::Yuv444 as u8 },
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
/// * `rgba_stride` - The stride (components per row) for the RGBA image data.
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
        { YuvChromaSubsampling::Yuv444 as u8 },
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
/// * `bgra_stride` - The stride (components per row) for the BGRA image data.
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
        { YuvChromaSubsampling::Yuv444 as u8 },
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
/// * `rgb_stride` - The stride (components per row) for the RGB image data.
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
        { YuvChromaSubsampling::Yuv444 as u8 },
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
/// * `bgr_stride` - The stride (components per row) for the BGR image data.
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
        { YuvChromaSubsampling::Yuv444 as u8 },
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
/// * `bgra_stride` - The stride (components per row) for the BGRA image data.
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
        { YuvChromaSubsampling::Yuv444 as u8 },
    >(bi_planar_image, rgb, rgb_stride, range, matrix)
}
