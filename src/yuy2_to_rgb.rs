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
use crate::avx2::yuy2_to_rgb_avx;
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use crate::neon::yuy2_to_rgb_neon;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use crate::sse::yuy2_to_rgb_sse;
use crate::yuv_support::{
    get_inverse_transform, get_yuv_range, YuvSourceChannels, Yuy2Description,
};
#[allow(unused_imports)]
use crate::yuv_to_yuy2::YuvToYuy2Navigation;
use crate::{YuvRange, YuvStandardMatrix};
#[cfg(feature = "rayon")]
use rayon::iter::{IndexedParallelIterator, ParallelIterator};
#[cfg(feature = "rayon")]
use rayon::prelude::{ParallelSlice, ParallelSliceMut};

fn yuy2_to_rgb_impl<const DESTINATION_CHANNELS: u8, const YUY2_SOURCE: usize>(
    yuy2_store: &[u8],
    yuy2_stride: u32,
    rgb_store: &mut [u8],
    rgb_stride: u32,
    width: u32,
    _: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) {
    let yuy2_source: Yuy2Description = YUY2_SOURCE.into();

    let dst_chans: YuvSourceChannels = DESTINATION_CHANNELS.into();
    let channels = dst_chans.get_channels_count();

    let range = get_yuv_range(8, range);
    let kr_kb = matrix.get_kr_kb();
    let transform = get_inverse_transform(255, range.range_y, range.range_uv, kr_kb.kr, kr_kb.kb);
    const PRECISION: i32 = 6;
    const ROUNDING_CONST: i32 = 1 << (PRECISION - 1);
    let inverse_transform = transform.to_integers(PRECISION as u32);
    let cr_coef = inverse_transform.cr_coef;
    let cb_coef = inverse_transform.cb_coef;
    let y_coef = inverse_transform.y_coef;
    let g_coef_1 = inverse_transform.g_coeff_1;
    let g_coef_2 = inverse_transform.g_coeff_2;
    let bias_y = range.bias_y as i32;
    let bias_uv = range.bias_uv as i32;

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    let mut _use_sse = std::arch::is_x86_feature_detected!("sse4.1");
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    let mut _use_avx = std::arch::is_x86_feature_detected!("avx2");

    let rgb_iter;
    let yuy2_iter;
    #[cfg(feature = "rayon")]
    {
        rgb_iter = rgb_store.par_chunks_exact_mut(rgb_stride as usize);
        yuy2_iter = yuy2_store.par_chunks_exact(yuy2_stride as usize);
    }
    #[cfg(not(feature = "rayon"))]
    {
        rgb_iter = rgb_store.chunks_exact_mut(rgb_stride as usize);
        yuy2_iter = yuy2_store.chunks_exact(yuy2_stride as usize);
    }

    rgb_iter
        .zip(yuy2_iter)
        .for_each(|(rgb_store, yuy2_store)| unsafe {
            let rgb_offset = 0usize;
            let yuy_offset = 0usize;

            let mut _cx = 0usize;
            let mut _yuy2_x = 0usize;

            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            {
                if _use_avx {
                    let processed = yuy2_to_rgb_avx::<DESTINATION_CHANNELS, YUY2_SOURCE>(
                        &range,
                        &inverse_transform,
                        yuy2_store,
                        yuy_offset,
                        rgb_store,
                        rgb_offset,
                        width,
                        YuvToYuy2Navigation::new(_cx, 0, _yuy2_x),
                    );
                    _cx = processed.cx;
                    _yuy2_x = processed.x;
                }
                if _use_sse {
                    let processed = yuy2_to_rgb_sse::<DESTINATION_CHANNELS, YUY2_SOURCE>(
                        &range,
                        &inverse_transform,
                        yuy2_store,
                        yuy_offset,
                        rgb_store,
                        rgb_offset,
                        width,
                        YuvToYuy2Navigation::new(_cx, 0, _yuy2_x),
                    );
                    _cx = processed.cx;
                    _yuy2_x = processed.x;
                }
            }

            #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
            {
                let processed = yuy2_to_rgb_neon::<DESTINATION_CHANNELS, YUY2_SOURCE>(
                    &range,
                    &inverse_transform,
                    yuy2_store,
                    yuy_offset,
                    rgb_store,
                    rgb_offset,
                    width,
                    YuvToYuy2Navigation::new(_cx, 0, _yuy2_x),
                );
                _cx = processed.cx;
                _yuy2_x = processed.x;
            }

            let max_iter = width as usize / 2;
            for x in _yuy2_x..max_iter {
                let rgb_pos = rgb_offset + _cx * channels;
                let yuy2_offset = yuy_offset + x * 4;

                let yuy2_plane_shifted = yuy2_store.get_unchecked(yuy2_offset..);

                let first_y = *yuy2_plane_shifted.get_unchecked(yuy2_source.get_first_y_position());
                let second_y =
                    *yuy2_plane_shifted.get_unchecked(yuy2_source.get_second_y_position());
                let u_value = *yuy2_plane_shifted.get_unchecked(yuy2_source.get_u_position());
                let v_value = *yuy2_plane_shifted.get_unchecked(yuy2_source.get_v_position());

                let cb = u_value as i32 - bias_uv;
                let cr = v_value as i32 - bias_uv;
                let f_y = (first_y as i32 - bias_y) * y_coef;
                let s_y = (second_y as i32 - bias_y) * y_coef;

                let dst0 = rgb_store.get_unchecked_mut(rgb_pos..);
                let r0 = ((f_y + cr_coef * cr + ROUNDING_CONST) >> PRECISION).clamp(0, 255);
                let b0 = ((f_y + cb_coef * cb + ROUNDING_CONST) >> PRECISION).clamp(0, 255);
                let g0 = ((f_y - g_coef_1 * cr - g_coef_2 * cb + ROUNDING_CONST) >> PRECISION)
                    .clamp(0, 255);
                *dst0.get_unchecked_mut(dst_chans.get_r_channel_offset()) = r0 as u8;
                *dst0.get_unchecked_mut(dst_chans.get_g_channel_offset()) = g0 as u8;
                *dst0.get_unchecked_mut(dst_chans.get_b_channel_offset()) = b0 as u8;

                if dst_chans.has_alpha() {
                    *dst0.get_unchecked_mut(dst_chans.get_a_channel_offset()) = 255;
                }

                let dst1 = dst0.get_unchecked_mut(channels..);

                let r1 = ((s_y + cr_coef * cr + ROUNDING_CONST) >> PRECISION).clamp(0, 255);
                let b1 = ((s_y + cb_coef * cb + ROUNDING_CONST) >> PRECISION).clamp(0, 255);
                let g1 = ((s_y - g_coef_1 * cr - g_coef_2 * cb + ROUNDING_CONST) >> PRECISION)
                    .clamp(0, 255);
                *dst1.get_unchecked_mut(dst_chans.get_r_channel_offset()) = r1 as u8;
                *dst1.get_unchecked_mut(dst_chans.get_g_channel_offset()) = g1 as u8;
                *dst1.get_unchecked_mut(dst_chans.get_b_channel_offset()) = b1 as u8;
                if dst_chans.has_alpha() {
                    *dst1.get_unchecked_mut(dst_chans.get_a_channel_offset()) = 255;
                }

                _cx += 2;
            }

            if width & 1 == 1 {
                let rgb_pos = rgb_offset + (width as usize - 1) * channels;
                let yuy2_offset = yuy_offset + ((width as usize - 1) / 2) * 4;

                let yuy2_plane_shifted = yuy2_store.get_unchecked(yuy2_offset..);

                let first_y = *yuy2_plane_shifted.get_unchecked(yuy2_source.get_first_y_position());
                let u_value = *yuy2_plane_shifted.get_unchecked(yuy2_source.get_u_position());
                let v_value = *yuy2_plane_shifted.get_unchecked(yuy2_source.get_v_position());

                let cb = u_value as i32 - bias_uv;
                let cr = v_value as i32 - bias_uv;
                let f_y = (first_y as i32 - bias_y) * y_coef;

                let dst0 = rgb_store.get_unchecked_mut(rgb_pos..);
                let r0 = ((f_y + cr_coef * cr + ROUNDING_CONST) >> PRECISION).clamp(0, 255);
                let b0 = ((f_y + cb_coef * cb + ROUNDING_CONST) >> PRECISION).clamp(0, 255);
                let g0 = ((f_y - g_coef_1 * cr - g_coef_2 * cb + ROUNDING_CONST) >> PRECISION)
                    .clamp(0, 255);
                *dst0.get_unchecked_mut(dst_chans.get_r_channel_offset()) = r0 as u8;
                *dst0.get_unchecked_mut(dst_chans.get_g_channel_offset()) = g0 as u8;
                *dst0.get_unchecked_mut(dst_chans.get_b_channel_offset()) = b0 as u8;
                if dst_chans.has_alpha() {
                    *dst0.get_unchecked_mut(dst_chans.get_a_channel_offset()) = 255;
                }
            }
        });
}

/// Convert YUYV (YUV Packed) format to RGB image.
///
/// This function takes YUYV (4:2:2 Packed) format data with 8-bit precision,
/// and converts it to RGB with 8-bit per channel precision.
///
/// # Arguments
///
/// * `yuy2_store` - A slice to store the converted YUYV data.
/// * `yuy2_stride` - The stride (bytes per row) for the YUYV plane.
/// * `rgb` - A mutable slice to store the converted RGB data.
/// * `rgb_stride` - The stride (bytes per row) for the RGB image data.
/// * `width` - The width of the image.
/// * `height` - The height of the image.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input YUYV data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuyv422_to_rgb(
    yuy2_store: &[u8],
    yuy2_stride: u32,
    rgb: &mut [u8],
    rgb_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) {
    yuy2_to_rgb_impl::<{ YuvSourceChannels::Rgb as u8 }, { Yuy2Description::YUYV as usize }>(
        yuy2_store,
        yuy2_stride,
        rgb,
        rgb_stride,
        width,
        height,
        range,
        matrix,
    );
}

/// Convert YUYV (YUV Packed) format to RGBA image.
///
/// This function takes YUYV (4:2:2 Packed) format data with 8-bit precision,
/// and converts it to RGBA with 8-bit per channel precision.
///
/// # Arguments
///
/// * `yuy2_store` - A slice to store the converted YUYV data.
/// * `yuy2_stride` - The stride (bytes per row) for the YUYV plane.
/// * `rgba` - A mutable slice to store the converted RGBA data.
/// * `rgba_stride` - The stride (bytes per row) for the RGBA image data.
/// * `width` - The width of the image.
/// * `height` - The height of the image.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input YUYV data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuyv422_to_rgba(
    yuy2_store: &[u8],
    yuy2_stride: u32,
    rgba: &mut [u8],
    rgba_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) {
    yuy2_to_rgb_impl::<{ YuvSourceChannels::Rgba as u8 }, { Yuy2Description::YUYV as usize }>(
        yuy2_store,
        yuy2_stride,
        rgba,
        rgba_stride,
        width,
        height,
        range,
        matrix,
    );
}

/// Convert YUYV (YUV Packed) format to BGR image.
///
/// This function takes YUYV (4:2:2 Packed) format data with 8-bit precision,
/// and converts it to BGR with 8-bit per channel precision.
///
/// # Arguments
///
/// * `yuy2_store` - A slice to store the converted YUYV data.
/// * `yuy2_stride` - The stride (bytes per row) for the YUYV plane.
/// * `bgr` - A mutable slice to store the converted BGR data.
/// * `bgr_stride` - The stride (bytes per row) for the BGR image data.
/// * `width` - The width of the image.
/// * `height` - The height of the image.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input YUYV data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuyv422_to_bgr(
    yuy2_store: &[u8],
    yuy2_stride: u32,
    bgr: &mut [u8],
    bgr_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) {
    yuy2_to_rgb_impl::<{ YuvSourceChannels::Bgr as u8 }, { Yuy2Description::YUYV as usize }>(
        yuy2_store,
        yuy2_stride,
        bgr,
        bgr_stride,
        width,
        height,
        range,
        matrix,
    );
}

/// Convert YUYV (YUV Packed) format to BGR image.
///
/// This function takes YUYV (4:2:2 Packed) format data with 8-bit precision,
/// and converts it to BGRA with 8-bit per channel precision.
///
/// # Arguments
///
/// * `yuy2_store` - A slice to store the converted YUYV data.
/// * `yuy2_stride` - The stride (bytes per row) for the YUYV plane.
/// * `bgra` - A mutable slice to store the converted BGRA data.
/// * `bgra_stride` - The stride (bytes per row) for the BGRA image data.
/// * `width` - The width of the image.
/// * `height` - The height of the image.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input YUYV data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuyv422_to_bgra(
    yuy2_store: &[u8],
    yuy2_stride: u32,
    bgra: &mut [u8],
    bgra_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) {
    yuy2_to_rgb_impl::<{ YuvSourceChannels::Bgra as u8 }, { Yuy2Description::YUYV as usize }>(
        yuy2_store,
        yuy2_stride,
        bgra,
        bgra_stride,
        width,
        height,
        range,
        matrix,
    );
}

/// Convert UYVY (YUV Packed) format to RGB image.
///
/// This function takes UYVY (4:2:2 Packed) format data with 8-bit precision,
/// and converts it to RGB with 8-bit per channel precision.
///
/// # Arguments
///
/// * `yuy2_store` - A slice to store the converted UYVY data.
/// * `yuy2_stride` - The stride (bytes per row) for the UYVY plane.
/// * `rgb` - A mutable slice to store the converted RGB data.
/// * `rgb_stride` - The stride (bytes per row) for the RGB image data.
/// * `width` - The width of the image.
/// * `height` - The height of the image.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input UYVY data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn uyvy422_to_rgb(
    yuy2_store: &[u8],
    yuy2_stride: u32,
    rgb: &mut [u8],
    rgb_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) {
    yuy2_to_rgb_impl::<{ YuvSourceChannels::Rgb as u8 }, { Yuy2Description::UYVY as usize }>(
        yuy2_store,
        yuy2_stride,
        rgb,
        rgb_stride,
        width,
        height,
        range,
        matrix,
    );
}

/// Convert UYVY (YUV Packed) format to RGBA image.
///
/// This function takes UYVY (4:2:2 Packed) format data with 8-bit precision,
/// and converts it to RGBA with 8-bit per channel precision.
///
/// # Arguments
///
/// * `yuy2_store` - A slice to store the converted UYVY data.
/// * `yuy2_stride` - The stride (bytes per row) for the UYVY plane.
/// * `rgba` - A mutable slice to store the converted RGBA data.
/// * `rgba_stride` - The stride (bytes per row) for the RGBA image data.
/// * `width` - The width of the image.
/// * `height` - The height of the image.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input UYVY data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn uyvy422_to_rgba(
    yuy2_store: &[u8],
    yuy2_stride: u32,
    rgba: &mut [u8],
    rgba_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) {
    yuy2_to_rgb_impl::<{ YuvSourceChannels::Rgba as u8 }, { Yuy2Description::UYVY as usize }>(
        yuy2_store,
        yuy2_stride,
        rgba,
        rgba_stride,
        width,
        height,
        range,
        matrix,
    );
}

/// Convert UYVY (YUV Packed) format to BGR image.
///
/// This function takes UYVY (4:2:2 Packed) format data with 8-bit precision,
/// and converts it to BGR with 8-bit per channel precision.
///
/// # Arguments
///
/// * `yuy2_store` - A slice to store the converted UYVY data.
/// * `yuy2_stride` - The stride (bytes per row) for the UYVY plane.
/// * `bgr` - A mutable slice to store the converted BGR data.
/// * `bgr_stride` - The stride (bytes per row) for the BGR image data.
/// * `width` - The width of the image.
/// * `height` - The height of the image.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input UYVY data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn uyvy422_to_bgr(
    yuy2_store: &[u8],
    yuy2_stride: u32,
    bgr: &mut [u8],
    bgr_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) {
    yuy2_to_rgb_impl::<{ YuvSourceChannels::Bgr as u8 }, { Yuy2Description::UYVY as usize }>(
        yuy2_store,
        yuy2_stride,
        bgr,
        bgr_stride,
        width,
        height,
        range,
        matrix,
    );
}

/// Convert UYVY (YUV Packed) format to BGRA image.
///
/// This function takes UYVY (4:2:2 Packed) format data with 8-bit precision,
/// and converts it to BGRA with 8-bit per channel precision.
///
/// # Arguments
///
/// * `yuy2_store` - A slice to store the converted UYVY data.
/// * `yuy2_stride` - The stride (bytes per row) for the UYVY plane.
/// * `bgra` - A mutable slice to store the converted BGRA data.
/// * `bgra_stride` - The stride (bytes per row) for the BGRA image data.
/// * `width` - The width of the image.
/// * `height` - The height of the image.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input UYVY data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn uyvy422_to_bgra(
    yuy2_store: &[u8],
    yuy2_stride: u32,
    bgra: &mut [u8],
    bgra_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) {
    yuy2_to_rgb_impl::<{ YuvSourceChannels::Bgra as u8 }, { Yuy2Description::UYVY as usize }>(
        yuy2_store,
        yuy2_stride,
        bgra,
        bgra_stride,
        width,
        height,
        range,
        matrix,
    );
}

/// Convert YVYU ( YUV Packed ) format to RGB image.
///
/// This function takes YVYU (4:2:2 Packed) format data with 8-bit precision,
/// and converts it to RGB with 8-bit per channel precision.
///
/// # Arguments
///
/// * `yuy2_store` - A slice to store the converted YVYU data.
/// * `yuy2_stride` - The stride (bytes per row) for the YVYU plane.
/// * `rgb` - A mutable slice to store the converted RGB data.
/// * `rgb_stride` - The stride (bytes per row) for the RGB image data.
/// * `width` - The width of the image.
/// * `height` - The height of the image.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input YVYU data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yvyu422_to_rgb(
    yuy2_store: &[u8],
    yuy2_stride: u32,
    rgb: &mut [u8],
    rgb_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) {
    yuy2_to_rgb_impl::<{ YuvSourceChannels::Rgb as u8 }, { Yuy2Description::YVYU as usize }>(
        yuy2_store,
        yuy2_stride,
        rgb,
        rgb_stride,
        width,
        height,
        range,
        matrix,
    );
}

/// Convert YVYU (YUV Packed) format to RGBA image.
///
/// This function takes YVYU (4:2:2 Packed) format data with 8-bit precision,
/// and converts it to RGBA with 8-bit per channel precision.
///
/// # Arguments
///
/// * `yuy2_store` - A slice to store the converted YVYU data.
/// * `yuy2_stride` - The stride (bytes per row) for the YVYU plane.
/// * `rgba` - A mutable slice to store the converted RGBA data.
/// * `rgba_stride` - The stride (bytes per row) for the RGBA image data.
/// * `width` - The width of the image.
/// * `height` - The height of the image.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input YVYU data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yvyu422_to_rgba(
    yuy2_store: &[u8],
    yuy2_stride: u32,
    rgba: &mut [u8],
    rgba_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) {
    yuy2_to_rgb_impl::<{ YuvSourceChannels::Rgba as u8 }, { Yuy2Description::YVYU as usize }>(
        yuy2_store,
        yuy2_stride,
        rgba,
        rgba_stride,
        width,
        height,
        range,
        matrix,
    );
}

/// Convert YVYU (YUV Packed) format to BGR image.
///
/// This function takes YVYU (4:2:2 Packed) format data with 8-bit precision,
/// and converts it to BGR with 8-bit per channel precision.
///
/// # Arguments
///
/// * `yuy2_store` - A slice to store the converted YVYU data.
/// * `yuy2_stride` - The stride (bytes per row) for the YVYU plane.
/// * `bgr` - A mutable slice to store the converted BGR data.
/// * `bgr_stride` - The stride (bytes per row) for the BGR image data.
/// * `width` - The width of the image.
/// * `height` - The height of the image.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input YVYU data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yvyu422_to_bgr(
    yuy2_store: &[u8],
    yuy2_stride: u32,
    bgr: &mut [u8],
    bgr_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) {
    yuy2_to_rgb_impl::<{ YuvSourceChannels::Bgr as u8 }, { Yuy2Description::YVYU as usize }>(
        yuy2_store,
        yuy2_stride,
        bgr,
        bgr_stride,
        width,
        height,
        range,
        matrix,
    );
}

/// Convert YVYU (YUV Packed) format to BGRA image.
///
/// This function takes YVYU (4:2:2 Packed) format data with 8-bit precision,
/// and converts it to BGRA with 8-bit per channel precision.
///
/// # Arguments
///
/// * `yuy2_store` - A slice to store the converted YVYU data.
/// * `yuy2_stride` - The stride (bytes per row) for the YVYU plane.
/// * `bgra` - A mutable slice to store the converted BGRA data.
/// * `bgra_stride` - The stride (bytes per row) for the BGRA image data.
/// * `width` - The width of the image.
/// * `height` - The height of the image.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input YVYU data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yvyu422_to_bgra(
    yuy2_store: &[u8],
    yuy2_stride: u32,
    bgra: &mut [u8],
    bgra_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) {
    yuy2_to_rgb_impl::<{ YuvSourceChannels::Bgra as u8 }, { Yuy2Description::YVYU as usize }>(
        yuy2_store,
        yuy2_stride,
        bgra,
        bgra_stride,
        width,
        height,
        range,
        matrix,
    );
}

/// Convert VYUY (YUV Packed) format to RGB image.
///
/// This function takes VYUY (4:2:2 Packed) format data with 8-bit precision,
/// and converts it to RGB with 8-bit per channel precision.
///
/// # Arguments
///
/// * `yuy2_store` - A slice to store the converted VYUY data.
/// * `yuy2_stride` - The stride (bytes per row) for the VYUY plane.
/// * `rgb` - A mutable slice to store the converted RGB data.
/// * `rgb_stride` - The stride (bytes per row) for the RGB image data.
/// * `width` - The width of the image.
/// * `height` - The height of the image.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input VYUY data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn vyuy422_to_rgb(
    yuy2_store: &[u8],
    yuy2_stride: u32,
    rgb: &mut [u8],
    rgb_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) {
    yuy2_to_rgb_impl::<{ YuvSourceChannels::Rgb as u8 }, { Yuy2Description::VYUY as usize }>(
        yuy2_store,
        yuy2_stride,
        rgb,
        rgb_stride,
        width,
        height,
        range,
        matrix,
    );
}

/// Convert VYUY (YUV Packed) format to RGBA image.
///
/// This function takes VYUY (4:2:2 Packed) format data with 8-bit precision,
/// and converts it to RGBA with 8-bit per channel precision.
///
/// # Arguments
///
/// * `yuy2_store` - A slice to store the converted VYUY data.
/// * `yuy2_stride` - The stride (bytes per row) for the VYUY plane.
/// * `rgba` - A mutable slice to store the converted RGBA data.
/// * `rgba_stride` - The stride (bytes per row) for the RGBA image data.
/// * `width` - The width of the image.
/// * `height` - The height of the image.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input VYUY data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn vyuy422_to_rgba(
    yuy2_store: &[u8],
    yuy2_stride: u32,
    rgba: &mut [u8],
    rgba_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) {
    yuy2_to_rgb_impl::<{ YuvSourceChannels::Rgba as u8 }, { Yuy2Description::VYUY as usize }>(
        yuy2_store,
        yuy2_stride,
        rgba,
        rgba_stride,
        width,
        height,
        range,
        matrix,
    );
}

/// Convert VYUY (YUV Packed) format to BGR image.
///
/// This function takes VYUY (4:2:2 Packed) format data with 8-bit precision,
/// and converts it to BGR with 8-bit per channel precision.
///
/// # Arguments
///
/// * `yuy2_store` - A slice to store the converted VYUY data.
/// * `yuy2_stride` - The stride (bytes per row) for the VYUY plane.
/// * `bgr` - A mutable slice to store the converted BGR data.
/// * `bgr_stride` - The stride (bytes per row) for the BGR image data.
/// * `width` - The width of the image.
/// * `height` - The height of the image.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input VYUY data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn vyuy422_to_bgr(
    yuy2_store: &[u8],
    yuy2_stride: u32,
    bgr: &mut [u8],
    bgr_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) {
    yuy2_to_rgb_impl::<{ YuvSourceChannels::Bgr as u8 }, { Yuy2Description::VYUY as usize }>(
        yuy2_store,
        yuy2_stride,
        bgr,
        bgr_stride,
        width,
        height,
        range,
        matrix,
    );
}

/// Convert VYUY (YUV Packed) format to BGRA image.
///
/// This function takes VYUY (4:2:2 Packed) format data with 8-bit precision,
/// and converts it to BGRA with 8-bit per channel precision.
///
/// # Arguments
///
/// * `yuy2_store` - A slice to store the converted VYUY data.
/// * `yuy2_stride` - The stride (bytes per row) for the VYUY plane.
/// * `bgra` - A mutable slice to store the converted BGRA data.
/// * `bgra_stride` - The stride (bytes per row) for the BGRA image data.
/// * `width` - The width of the image.
/// * `height` - The height of the image.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input VYUY data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn vyuy422_to_bgra(
    yuy2_store: &[u8],
    yuy2_stride: u32,
    bgra: &mut [u8],
    bgra_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) {
    yuy2_to_rgb_impl::<{ YuvSourceChannels::Bgra as u8 }, { Yuy2Description::VYUY as usize }>(
        yuy2_store,
        yuy2_stride,
        bgra,
        bgra_stride,
        width,
        height,
        range,
        matrix,
    );
}
