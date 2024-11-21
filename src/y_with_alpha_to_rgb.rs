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
use crate::numerics::qrshr;
use crate::yuv_error::check_rgba_destination;
use crate::yuv_support::*;
use crate::{YuvError, YuvGrayAlphaImage};
use num_traits::AsPrimitive;
#[cfg(feature = "rayon")]
use rayon::iter::{IndexedParallelIterator, ParallelIterator};
#[cfg(feature = "rayon")]
use rayon::prelude::{ParallelSlice, ParallelSliceMut};
use std::fmt::Debug;

// Chroma subsampling always assumed as 400
#[inline]
fn y_with_alpha_to_rgbx<
    V: Copy + AsPrimitive<i32> + 'static + Send + Sync + Debug + Default + Clone,
    const DESTINATION_CHANNELS: u8,
    const BIT_DEPTH: usize,
>(
    gray_alpha_image: &YuvGrayAlphaImage<V>,
    rgba: &mut [V],
    rgba_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) -> Result<(), YuvError>
where
    i32: AsPrimitive<V>,
{
    let destination_channels: YuvSourceChannels = DESTINATION_CHANNELS.into();
    let channels = destination_channels.get_channels_count();
    assert!(
        destination_channels.has_alpha(),
        "YUV400 with alpha cannot be called on target image without alpha"
    );
    assert_eq!(
        channels, 4,
        "YUV400 with alpha cannot be called on target image without alpha"
    );
    assert!(
        (8..=16).contains(&BIT_DEPTH),
        "Invalid bit depth is provided"
    );

    check_rgba_destination(
        rgba,
        rgba_stride,
        gray_alpha_image.width,
        gray_alpha_image.height,
        channels,
    )?;
    gray_alpha_image.check_constraints()?;

    let max_colors = (1 << BIT_DEPTH) - 1;

    let chroma_range = get_yuv_range(BIT_DEPTH as u32, range);
    let kr_kb = matrix.get_kr_kb();
    let transform = get_inverse_transform(
        max_colors,
        chroma_range.range_y,
        chroma_range.range_uv,
        kr_kb.kr,
        kr_kb.kb,
    );

    const PRECISION: i32 = 12;
    let inverse_transform = transform.to_integers(PRECISION as u32);
    let y_coef = inverse_transform.y_coef;

    let bias_y = chroma_range.bias_y as i32;

    let iter;
    let y_iter;
    let a_iter;
    #[cfg(feature = "rayon")]
    {
        iter = rgba.par_chunks_exact_mut(rgba_stride as usize);
        y_iter = gray_alpha_image
            .y_plane
            .par_chunks_exact(gray_alpha_image.y_stride as usize);
        a_iter = gray_alpha_image
            .a_plane
            .par_chunks_exact(gray_alpha_image.a_stride as usize);
    }
    #[cfg(not(feature = "rayon"))]
    {
        iter = rgba.chunks_exact_mut(rgba_stride as usize);
        y_iter = gray_alpha_image
            .y_plane
            .chunks_exact(gray_alpha_image.y_stride as usize);
        a_iter = gray_alpha_image
            .a_plane
            .chunks_exact(gray_alpha_image.a_stride as usize);
    }

    if range == YuvRange::Limited {
        iter.zip(y_iter)
            .zip(a_iter)
            .for_each(|((rgba, y_plane), a_plane)| {
                for ((y_src, a_src), rgba) in y_plane
                    .iter()
                    .zip(a_plane)
                    .zip(rgba.chunks_exact_mut(channels))
                {
                    let y_value = (y_src.as_() - bias_y) * y_coef;

                    let r = qrshr::<PRECISION, BIT_DEPTH>(y_value);
                    rgba[destination_channels.get_r_channel_offset()] = r.as_();
                    rgba[destination_channels.get_g_channel_offset()] = r.as_();
                    rgba[destination_channels.get_b_channel_offset()] = r.as_();
                    rgba[destination_channels.get_a_channel_offset()] = *a_src;
                }
            });
    } else {
        iter.zip(y_iter)
            .zip(a_iter)
            .for_each(|((rgba, y_plane), a_plane)| {
                for ((y_src, a_src), rgba) in y_plane
                    .iter()
                    .zip(a_plane)
                    .zip(rgba.chunks_exact_mut(channels))
                {
                    let y_value = *y_src;
                    rgba[destination_channels.get_r_channel_offset()] = y_value;
                    rgba[destination_channels.get_g_channel_offset()] = y_value;
                    rgba[destination_channels.get_b_channel_offset()] = y_value;
                    rgba[destination_channels.get_a_channel_offset()] = *a_src;
                }
            });
    }

    Ok(())
}

/// Convert YUV 400 planar format with alpha plane to RGBA format.
///
/// This function takes YUV 400 planar format data with 8-bit precision,
/// and converts it to RGBA format with 8-bit per channel precision.
///
/// # Arguments
///
/// * `gray_alpha_image` - Source gray image with alpha.
/// * `rgba` - A mutable slice to store the converted RGBA data.
/// * `rgba_stride` - Elements per row.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGRA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv400_with_alpha_to_rgba(
    gray_alpha_image: &YuvGrayAlphaImage<u8>,
    rgba: &mut [u8],
    rgba_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) -> Result<(), YuvError> {
    y_with_alpha_to_rgbx::<u8, { YuvSourceChannels::Rgba as u8 }, 8>(
        gray_alpha_image,
        rgba,
        rgba_stride,
        range,
        matrix,
    )
}

/// Convert YUV 400 planar format with alpha plane to BGRA format.
///
/// This function takes YUV 400 planar format data with 8-bit precision,
/// and converts it to BGRA format with 8-bit per channel precision.
///
/// # Arguments
///
/// * `gray_alpha_image` - Source gray image with alpha.
/// * `bgra` - A mutable slice to store the converted BGRA data.
/// * `bgra_stride` - Elements per row.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGRA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv400_with_alpha_to_bgra(
    gray_alpha_image: &YuvGrayAlphaImage<u8>,
    bgra: &mut [u8],
    bgra_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) -> Result<(), YuvError> {
    y_with_alpha_to_rgbx::<u8, { YuvSourceChannels::Bgra as u8 }, 8>(
        gray_alpha_image,
        bgra,
        bgra_stride,
        range,
        matrix,
    )
}
