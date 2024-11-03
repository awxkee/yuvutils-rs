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
use crate::yuv_error::{check_rgba_destination, check_y8_channel};
use crate::yuv_support::*;
use crate::YuvError;
use num_traits::AsPrimitive;
#[cfg(feature = "rayon")]
use rayon::iter::{IndexedParallelIterator, ParallelIterator};
#[cfg(feature = "rayon")]
use rayon::prelude::{ParallelSlice, ParallelSliceMut};

// Chroma subsampling always assumed as 400
#[inline]
fn y_with_alpha_to_rgbx<
    V: Copy + AsPrimitive<i32> + 'static + Send + Sync,
    const DESTINATION_CHANNELS: u8,
    const BIT_DEPTH: usize,
>(
    y_plane: &[V],
    y_stride: u32,
    a_plane: &[V],
    a_stride: u32,
    rgba: &mut [V],
    rgba_stride: u32,
    width: u32,
    height: u32,
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

    check_rgba_destination(rgba, rgba_stride, width, height, channels)?;
    check_y8_channel(y_plane, y_stride, width, height)?;
    check_y8_channel(a_plane, a_stride, width, height)?;

    let max_colors = (1 << BIT_DEPTH) - 1;

    let range = get_yuv_range(BIT_DEPTH as u32, range);
    let kr_kb = matrix.get_kr_kb();
    let transform = get_inverse_transform(
        max_colors,
        range.range_y,
        range.range_uv,
        kr_kb.kr,
        kr_kb.kb,
    );

    const PRECISION: i32 = 12;
    let inverse_transform = transform.to_integers(PRECISION as u32);
    let y_coef = inverse_transform.y_coef;

    let bias_y = range.bias_y as i32;

    let iter;
    let y_iter;
    let a_iter;
    #[cfg(feature = "rayon")]
    {
        iter = rgba.par_chunks_exact_mut(rgba_stride as usize);
        y_iter = y_plane.par_chunks_exact(y_stride as usize);
        a_iter = a_plane.par_chunks_exact(a_stride as usize);
    }
    #[cfg(not(feature = "rayon"))]
    {
        iter = rgba.chunks_exact_mut(rgba_stride as usize);
        y_iter = y_plane.chunks_exact(y_stride as usize);
        a_iter = a_plane.chunks_exact(a_stride as usize);
    }

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

    Ok(())
}

/// Convert YUV 400 planar format with alpha plane to RGBA format.
///
/// This function takes YUV 400 planar format data with 8-bit precision,
/// and converts it to RGBA format with 8-bit per channel precision.
///
/// # Arguments
///
/// * `y_plane` - A slice to load the Y (luminance) plane data.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `a_plane` - A slice to load alpha plane data
/// * `a_stride` - The stride (bytes per row) for the alpha plane.
/// * `width` - The width of the YUV image.
/// * `height` - The height of the YUV image.
/// * `rgba_data` - A mutable slice to store the converted RGBA data.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGRA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv400_with_alpha_to_rgba(
    y_plane: &[u8],
    y_stride: u32,
    a_plane: &[u8],
    a_stride: u32,
    rgba: &mut [u8],
    rgba_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) -> Result<(), YuvError> {
    y_with_alpha_to_rgbx::<u8, { YuvSourceChannels::Rgba as u8 }, 8>(
        y_plane,
        y_stride,
        a_plane,
        a_stride,
        rgba,
        rgba_stride,
        width,
        height,
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
/// * `y_plane` - A slice to load the Y (luminance) plane data.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `a_plane` - A slice to load alpha plane data
/// * `a_stride` - The stride (bytes per row) for the alpha plane.
/// * `width` - The width of the YUV image.
/// * `height` - The height of the YUV image.
/// * `bgra_data` - A mutable slice to store the converted BGRA data.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGRA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv400_with_alpha_to_bgra(
    y_plane: &[u8],
    y_stride: u32,
    a_plane: &[u8],
    a_stride: u32,
    bgra: &mut [u8],
    bgra_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) -> Result<(), YuvError> {
    y_with_alpha_to_rgbx::<u8, { YuvSourceChannels::Bgra as u8 }, 8>(
        y_plane,
        y_stride,
        a_plane,
        a_stride,
        bgra,
        bgra_stride,
        width,
        height,
        range,
        matrix,
    )
}
