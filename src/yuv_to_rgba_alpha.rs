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
use crate::avx2::avx2_yuv_to_rgba_alpha;
#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    feature = "nightly_avx512"
))]
use crate::avx512bw::avx512_yuv_to_rgba_alpha;
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use crate::neon::neon_yuv_to_rgba_alpha;
use crate::numerics::{div_by_255, qrshr};
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use crate::sse::sse_yuv_to_rgba_alpha_row;
use crate::yuv_error::check_rgba_destination;
#[allow(unused_imports)]
use crate::yuv_support::*;
use crate::{YuvError, YuvPlanarImageWithAlpha, YuvRange, YuvStandardMatrix};
#[cfg(feature = "rayon")]
use rayon::iter::{IndexedParallelIterator, ParallelIterator};
#[cfg(feature = "rayon")]
use rayon::prelude::{ParallelSlice, ParallelSliceMut};

fn yuv_with_alpha_to_rgbx<const DESTINATION_CHANNELS: u8, const SAMPLING: u8>(
    planar_with_alpha: &YuvPlanarImageWithAlpha<u8>,
    rgba: &mut [u8],
    rgba_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    premultiply_alpha: bool,
) -> Result<(), YuvError> {
    let chroma_subsampling: YuvChromaSample = SAMPLING.into();
    let dst_chans: YuvSourceChannels = DESTINATION_CHANNELS.into();
    assert!(
        dst_chans.has_alpha(),
        "yuv_with_alpha_to_rgbx cannot be called on configuration without alpha"
    );
    let channels = dst_chans.get_channels_count();

    check_rgba_destination(
        rgba,
        rgba_stride,
        planar_with_alpha.width,
        planar_with_alpha.height,
        channels,
    )?;
    planar_with_alpha.check_constraints(chroma_subsampling)?;

    let range = get_yuv_range(8, range);
    let kr_kb = matrix.get_kr_kb();
    let transform = get_inverse_transform(255, range.range_y, range.range_uv, kr_kb.kr, kr_kb.kb);
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

    let width = planar_with_alpha.width;
    let y_plane = planar_with_alpha.y_plane;
    let u_plane = planar_with_alpha.u_plane;
    let v_plane = planar_with_alpha.v_plane;
    let y_stride = planar_with_alpha.y_stride;
    let u_stride = planar_with_alpha.u_stride;
    let v_stride = planar_with_alpha.v_stride;
    let a_plane = planar_with_alpha.a_plane;
    let a_stride = planar_with_alpha.a_stride;

    let iter;
    #[cfg(feature = "rayon")]
    {
        iter = rgba
            .par_chunks_exact_mut(rgba_stride as usize)
            .zip(y_plane.par_chunks_exact(y_stride as usize))
            .zip(a_plane.par_chunks_exact(a_stride as usize));
    }
    #[cfg(not(feature = "rayon"))]
    {
        iter = rgba
            .chunks_exact_mut(rgba_stride as usize)
            .zip(y_plane.chunks_exact(y_stride as usize))
            .zip(a_plane.chunks_exact(a_stride as usize));
    }

    iter.enumerate()
        .for_each(|(y, ((rgba, y_plane), a_plane))| {
            let u_offset = if chroma_subsampling == YuvChromaSample::Yuv420 {
                (y >> 1) * (u_stride as usize)
            } else {
                y * (u_stride as usize)
            };
            let v_offset = if chroma_subsampling == YuvChromaSample::Yuv420 {
                (y >> 1) * (v_stride as usize)
            } else {
                y * (v_stride as usize)
            };

            #[allow(unused_variables)]
            #[allow(unused_mut)]
            let mut _cx = 0usize;

            #[allow(unused_variables)]
            #[allow(unused_mut)]
            let mut _uv_x = 0usize;

            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            unsafe {
                #[cfg(feature = "nightly_avx512")]
                {
                    if _use_avx512 {
                        let processed = avx512_yuv_to_rgba_alpha::<DESTINATION_CHANNELS, SAMPLING>(
                            &range,
                            &inverse_transform,
                            y_plane,
                            u_plane,
                            v_plane,
                            a_plane,
                            rgba,
                            _cx,
                            _uv_x,
                            u_offset,
                            v_offset,
                            width as usize,
                            premultiply_alpha,
                        );
                        _cx = processed.cx;
                        _uv_x = processed.ux;
                    }
                }
                if _use_avx2 {
                    let processed = avx2_yuv_to_rgba_alpha::<DESTINATION_CHANNELS, SAMPLING>(
                        &range,
                        &inverse_transform,
                        y_plane,
                        u_plane,
                        v_plane,
                        a_plane,
                        rgba,
                        _cx,
                        _uv_x,
                        u_offset,
                        v_offset,
                        width as usize,
                        premultiply_alpha,
                    );
                    _cx = processed.cx;
                    _uv_x = processed.ux;
                }
                if _use_sse {
                    let processed = sse_yuv_to_rgba_alpha_row::<DESTINATION_CHANNELS, SAMPLING>(
                        &range,
                        &inverse_transform,
                        y_plane,
                        u_plane,
                        v_plane,
                        a_plane,
                        rgba,
                        _cx,
                        _uv_x,
                        u_offset,
                        v_offset,
                        width as usize,
                        premultiply_alpha,
                    );
                    _cx = processed.cx;
                    _uv_x = processed.ux;
                }
            }

            #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
            unsafe {
                let processed = neon_yuv_to_rgba_alpha::<DESTINATION_CHANNELS, SAMPLING>(
                    &range,
                    &inverse_transform,
                    y_plane,
                    u_plane,
                    v_plane,
                    a_plane,
                    rgba,
                    _cx,
                    _uv_x,
                    0,
                    u_offset,
                    v_offset,
                    0,
                    0,
                    width as usize,
                    premultiply_alpha,
                );
                _cx = processed.cx;
                _uv_x = processed.ux;
            }

            if chroma_subsampling == YuvChromaSample::Yuv444 && _cx < width as usize {
                let u_plane = &u_plane[u_offset..];
                let v_plane = &v_plane[v_offset..];

                for ((((rgba, &y_src), &u_src), &v_src), &a_src) in rgba
                    .chunks_exact_mut(channels)
                    .zip(y_plane.iter())
                    .zip(u_plane.iter())
                    .zip(v_plane.iter())
                    .zip(a_plane.iter())
                    .skip(_cx)
                {
                    let y_value = (y_src as i32 - bias_y) * y_coef;
                    let cb_value = u_src as i32 - bias_uv;
                    let cr_value = v_src as i32 - bias_uv;

                    let mut r = qrshr::<PRECISION, 8>(y_value + cr_coef * cr_value);
                    let mut b = qrshr::<PRECISION, 8>(y_value + cb_coef * cb_value);
                    let mut g =
                        qrshr::<PRECISION, 8>(y_value - g_coef_1 * cr_value - g_coef_2 * cb_value);

                    if premultiply_alpha {
                        r = div_by_255(r as u16 * a_src as u16) as i32;
                        g = div_by_255(g as u16 * a_src as u16) as i32;
                        b = div_by_255(b as u16 * a_src as u16) as i32;
                    }

                    rgba[dst_chans.get_r_channel_offset()] = r as u8;
                    rgba[dst_chans.get_g_channel_offset()] = g as u8;
                    rgba[dst_chans.get_b_channel_offset()] = b as u8;
                    if dst_chans.has_alpha() {
                        rgba[dst_chans.get_a_channel_offset()] = a_src;
                    }
                }
            } else if _cx < width as usize {
                let chroma_width = (width as usize + 1) / 2;
                let u_plane = &u_plane[u_offset..(u_offset + chroma_width)];
                let v_plane = &v_plane[v_offset..(v_offset + chroma_width)];

                for ((((rgba, y_src), &u_src), &v_src), a_src) in rgba
                    .chunks_exact_mut(channels * 2)
                    .zip(y_plane.chunks_exact(2))
                    .zip(u_plane.iter())
                    .zip(v_plane.iter())
                    .zip(a_plane.chunks_exact(2))
                    .skip(_cx / 2)
                {
                    let y0_value = (y_src[0] as i32 - bias_y) * y_coef;
                    let cb_value = u_src as i32 - bias_uv;
                    let cr_value = v_src as i32 - bias_uv;

                    let mut r0 = qrshr::<PRECISION, 8>(y0_value + cr_coef * cr_value);
                    let mut b0 = qrshr::<PRECISION, 8>(y0_value + cb_coef * cb_value);
                    let mut g0 =
                        qrshr::<PRECISION, 8>(y0_value - g_coef_1 * cr_value - g_coef_2 * cb_value);

                    if premultiply_alpha {
                        let a0 = a_src[0];
                        r0 = div_by_255(r0 as u16 * a0 as u16) as i32;
                        g0 = div_by_255(g0 as u16 * a0 as u16) as i32;
                        b0 = div_by_255(b0 as u16 * a0 as u16) as i32;
                    }

                    rgba[dst_chans.get_r_channel_offset()] = r0 as u8;
                    rgba[dst_chans.get_g_channel_offset()] = g0 as u8;
                    rgba[dst_chans.get_b_channel_offset()] = b0 as u8;
                    if dst_chans.has_alpha() {
                        rgba[dst_chans.get_a_channel_offset()] = a_src[0];
                    }

                    let y1_value = (y_src[1] as i32 - bias_y) * y_coef;

                    let mut r1 = qrshr::<PRECISION, 8>(y1_value + cr_coef * cr_value);
                    let mut b1 = qrshr::<PRECISION, 8>(y1_value + cb_coef * cb_value);
                    let mut g1 =
                        qrshr::<PRECISION, 8>(y1_value - g_coef_1 * cr_value - g_coef_2 * cb_value);

                    if premultiply_alpha {
                        let a1 = a_src[1];
                        r1 = div_by_255(r1 as u16 * a1 as u16) as i32;
                        g1 = div_by_255(g1 as u16 * a1 as u16) as i32;
                        b1 = div_by_255(b1 as u16 * a1 as u16) as i32;
                    }

                    rgba[dst_chans.get_r_channel_offset() + channels] = r1 as u8;
                    rgba[dst_chans.get_g_channel_offset() + channels] = g1 as u8;
                    rgba[dst_chans.get_b_channel_offset() + channels] = b1 as u8;
                    if dst_chans.has_alpha() {
                        rgba[dst_chans.get_a_channel_offset() + channels] = a_src[1];
                    }
                }

                if width & 1 != 0 {
                    let y0_value = (*y_plane.last().unwrap() as i32 - bias_y) * y_coef;
                    let cb_value = *u_plane.last().unwrap() as i32 - bias_uv;
                    let cr_value = *v_plane.last().unwrap() as i32 - bias_uv;
                    let a_src = a_plane.last().unwrap();
                    let rgba = rgba.chunks_exact_mut(channels * 2).into_remainder();

                    let mut r0 = qrshr::<PRECISION, 8>(y0_value + cr_coef * cr_value);
                    let mut b0 = qrshr::<PRECISION, 8>(y0_value + cb_coef * cb_value);
                    let mut g0 =
                        qrshr::<PRECISION, 8>(y0_value - g_coef_1 * cr_value - g_coef_2 * cb_value);

                    if premultiply_alpha {
                        let a0 = *a_src;
                        r0 = div_by_255(r0 as u16 * a0 as u16) as i32;
                        g0 = div_by_255(g0 as u16 * a0 as u16) as i32;
                        b0 = div_by_255(b0 as u16 * a0 as u16) as i32;
                    }

                    rgba[dst_chans.get_r_channel_offset()] = r0 as u8;
                    rgba[dst_chans.get_g_channel_offset()] = g0 as u8;
                    rgba[dst_chans.get_b_channel_offset()] = b0 as u8;
                    if dst_chans.has_alpha() {
                        rgba[dst_chans.get_a_channel_offset()] = *a_src;
                    }
                }
            }
        });

    Ok(())
}

/// Convert YUV 420 planar format to RGBA format and appends provided alpha channel.
///
/// This function takes YUV 420 planar format data with 8-bit precision,
/// and converts it to RGBA format with 8-bit per channel precision.
///
/// # Arguments
///
/// * `planar_with_alpha` - Source planar image.
/// * `rgba` - A mutable slice to store the converted RGBA data.
/// * `rgba_stride` - Elements per row.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
/// * `premultiply_alpha` - Flag to premultiply alpha or not
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGRA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv420_with_alpha_to_rgba(
    planar_with_alpha: &YuvPlanarImageWithAlpha<u8>,
    rgba: &mut [u8],
    rgba_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    premultiply_alpha: bool,
) -> Result<(), YuvError> {
    yuv_with_alpha_to_rgbx::<{ YuvSourceChannels::Rgba as u8 }, { YuvChromaSample::Yuv420 as u8 }>(
        planar_with_alpha,
        rgba,
        rgba_stride,
        range,
        matrix,
        premultiply_alpha,
    )
}

/// Convert YUV 420 planar format to BGRA format and appends provided alpha channel.
///
/// This function takes YUV 420 planar format data with 8-bit precision,
/// and converts it to BGRA format with 8-bit per channel precision.
///
/// # Arguments
///
/// * `planar_with_alpha` - Source planar image.
/// * `bgra` - A mutable slice to store the converted BGRA data.
/// * `bgra_stride` - Elements per row.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
/// * `premultiply_alpha` - Flag to premultiply alpha or not
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGRA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv420_with_alpha_to_bgra(
    planar_with_alpha: &YuvPlanarImageWithAlpha<u8>,
    bgra: &mut [u8],
    bgra_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    premultiply_alpha: bool,
) -> Result<(), YuvError> {
    yuv_with_alpha_to_rgbx::<{ YuvSourceChannels::Bgra as u8 }, { YuvChromaSample::Yuv420 as u8 }>(
        planar_with_alpha,
        bgra,
        bgra_stride,
        range,
        matrix,
        premultiply_alpha,
    )
}

/// Convert YUV 422 planar format to RGBA format and appends provided alpha channel.
///
/// This function takes YUV 422 planar format data with 8-bit precision,
/// and converts it to RGBA format with 8-bit per channel precision.
///
/// # Arguments
///
/// * `planar_with_alpha` - Source planar image.
/// * `rgba` - A mutable slice to store the converted RGBA data.
/// * `rgba_stride` - Elements per row.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
/// * `premultiply_alpha` - Flag to premultiply alpha or not
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGRA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv422_with_alpha_to_rgba(
    planar_with_alpha: &YuvPlanarImageWithAlpha<u8>,
    rgba: &mut [u8],
    rgba_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    premultiply_alpha: bool,
) -> Result<(), YuvError> {
    yuv_with_alpha_to_rgbx::<{ YuvSourceChannels::Rgba as u8 }, { YuvChromaSample::Yuv422 as u8 }>(
        planar_with_alpha,
        rgba,
        rgba_stride,
        range,
        matrix,
        premultiply_alpha,
    )
}

/// Convert YUV 422 planar format to BGRA format and appends provided alpha channel.
///
/// This function takes YUV 422 planar format data with 8-bit precision,
/// and converts it to BGRA format with 8-bit per channel precision.
///
/// # Arguments
///
/// * `planar_with_alpha` - Source planar image.
/// * `bgra` - A mutable slice to store the converted BGRA data.
/// * `bgra_stride` - Elements per row.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
/// * `premultiply_alpha` - Flag to premultiply alpha or not
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGRA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv422_with_alpha_to_bgra(
    planar_with_alpha: &YuvPlanarImageWithAlpha<u8>,
    bgra: &mut [u8],
    bgra_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    premultiply_alpha: bool,
) -> Result<(), YuvError> {
    yuv_with_alpha_to_rgbx::<{ YuvSourceChannels::Bgra as u8 }, { YuvChromaSample::Yuv422 as u8 }>(
        planar_with_alpha,
        bgra,
        bgra_stride,
        range,
        matrix,
        premultiply_alpha,
    )
}

/// Convert YUV 444 planar format to RGBA format and appends provided alpha channel.
///
/// This function takes YUV 444 planar format data with 8-bit precision,
/// and converts it to RGBA format with 8-bit per channel precision.
///
/// # Arguments
///
/// * `planar_with_alpha` - Source planar image.
/// * `rgba` - A mutable slice to store the converted RGBA data.
/// * `rgba_stride` - Elements per row.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
/// * `premultiply_alpha` - Flag to premultiply alpha or not
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGRA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv444_with_alpha_to_rgba(
    planar_with_alpha: &YuvPlanarImageWithAlpha<u8>,
    rgba: &mut [u8],
    rgba_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    premultiply_alpha: bool,
) -> Result<(), YuvError> {
    yuv_with_alpha_to_rgbx::<{ YuvSourceChannels::Rgba as u8 }, { YuvChromaSample::Yuv444 as u8 }>(
        planar_with_alpha,
        rgba,
        rgba_stride,
        range,
        matrix,
        premultiply_alpha,
    )
}

/// Convert YUV 444 planar format to BGRA format and appends provided alpha channel.
///
/// This function takes YUV 444 planar format data with 8-bit precision,
/// and converts it to BGRA format with 8-bit per channel precision.
///
/// # Arguments
///
/// * `planar_with_alpha` - Source planar image.
/// * `bgra` - A mutable slice to store the converted BGRA data.
/// * `bgra_stride` - Elements per row.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
/// * `premultiply_alpha` - Flag to premultiply alpha or not
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGRA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv444_with_alpha_to_bgra(
    planar_with_alpha: &YuvPlanarImageWithAlpha<u8>,
    bgra: &mut [u8],
    bgra_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    premultiply_alpha: bool,
) -> Result<(), YuvError> {
    yuv_with_alpha_to_rgbx::<{ YuvSourceChannels::Bgra as u8 }, { YuvChromaSample::Yuv444 as u8 }>(
        planar_with_alpha,
        bgra,
        bgra_stride,
        range,
        matrix,
        premultiply_alpha,
    )
}
