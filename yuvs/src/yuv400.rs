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

use crate::{get_inverse_transform, get_kr_kb, get_yuv_range, YuvRange, YuvStandardMatrix};

/// Converts YUV444 to Rgb
///
/// This support not tightly packed data and crop image using stride in place.
///
/// # Arguments
///
/// * `y_plane`: Luma plane
/// * `y_stride`: Luma stride
/// * `u_plane`: U chroma plane
/// * `u_stride`: U chroma stride, even odd images is supported this always must match `u_stride * height`
/// * `v_plane`: V chroma plane
/// * `v_stride`: V chroma stride, even odd images is supported this always must match `v_stride * height`
/// * `rgb`: RGB image layout
/// * `width`: Image width
/// * `height`: Image height
/// * `range`: see [YuvRange]
/// * `matrix`: see [YuvStandardMatrix]
///
///
pub fn yuv400_to_rgb(
    y_plane: &[u8],
    y_stride: usize,
    rgb: &mut [u8],
    width: usize,
    height: usize,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) -> Result<(), String> {
    if y_plane.len() != y_stride * height {
        return Err(format!(
            "Luma plane expected {} bytes, got {}",
            y_stride * height,
            y_plane.len()
        ));
    }

    let range = get_yuv_range(8, range);
    let kr_kb = get_kr_kb(matrix);
    let transform = get_inverse_transform(255, range.range_y, range.range_uv, kr_kb.kr, kr_kb.kb)?;
    const PRECISION: i32 = 6;
    const ROUNDING_CONST: i32 = 1 << (PRECISION - 1);
    let inverse_transform = transform.to_integers(PRECISION as u32);
    let y_coef = inverse_transform.y_coef;

    let bias_y = range.bias_y as i32;
    const CHANNELS: usize = 3;

    if rgb.len() != width * height * CHANNELS {
        return Err(format!(
            "RGB image layout expected {} bytes, got {}",
            width * height * CHANNELS,
            rgb.len()
        ));
    }

    let max_value = (1 << 8) - 1;

    let rgb_stride = width * CHANNELS;

    let y_iter = y_plane.chunks_exact(y_stride);
    let rgb_iter = rgb.chunks_exact_mut(rgb_stride);

    for (y_src, rgb) in y_iter.zip(rgb_iter) {
        let rgb_chunks = rgb.chunks_exact_mut(CHANNELS);

        for (y_src, rgb_dst) in y_src.iter().zip(rgb_chunks) {
            let y_value = (*y_src as i32 - bias_y) * y_coef;

            let r = ((y_value + ROUNDING_CONST) >> PRECISION).clamp(0, max_value);
            rgb_dst[0] = r as u8;
            rgb_dst[1] = r as u8;
            rgb_dst[2] = r as u8;
        }
    }

    Ok(())
}
