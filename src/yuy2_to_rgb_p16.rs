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
use crate::yuv_support::{
    get_inverse_transform, get_yuv_range, YuvSourceChannels, Yuy2Description,
};
use crate::{YuvRange, YuvStandardMatrix};

fn yuy2_to_rgb_impl_p16<const DESTINATION_CHANNELS: u8, const YUY2_SOURCE: usize>(
    yuy2_store: &[u16],
    yuy2_stride: u32,
    rgb_store: &mut [u16],
    rgb_stride: u32,
    bit_depth: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) {
    let yuy2_source: Yuy2Description = YUY2_SOURCE.into();
    const PRECISION: i32 = 6;
    let dst_chans: YuvSourceChannels = DESTINATION_CHANNELS.into();
    let channels = dst_chans.get_channels_count();
    let range = get_yuv_range(bit_depth, range);
    let max_colors = (1 << bit_depth) - 1;
    let kr_kb = matrix.get_kr_kb();
    let transform = get_inverse_transform(
        max_colors as u32,
        range.range_y,
        range.range_uv,
        kr_kb.kr,
        kr_kb.kb,
    );
    const ROUNDING_CONST: i32 = 1 << (PRECISION - 1);
    let inverse_transform = transform.to_integers(PRECISION as u32);
    let cr_coef = inverse_transform.cr_coef;
    let cb_coef = inverse_transform.cb_coef;
    let y_coef = inverse_transform.y_coef;
    let g_coef_1 = inverse_transform.g_coeff_1;
    let g_coef_2 = inverse_transform.g_coeff_2;
    let bias_y = range.bias_y as i32;
    let bias_uv = range.bias_uv as i32;

    let mut rgb_offset = 0usize;
    let mut yuy_offset = 0usize;

    for _ in 0..height as usize {
        let mut _cx = 0usize;
        let mut _yuy2_x = 0usize;

        let max_iter = width as usize / 2;
        for x in _yuy2_x..max_iter {
            unsafe {
                let rgb_pos = _cx * channels;
                let yuy2_offset = x * 4;

                let src_ptr = ((yuy2_store.as_ptr() as *const u8).add(yuy_offset) as *const u16)
                    .add(yuy2_offset);

                let dst_ptr =
                    ((rgb_store.as_mut_ptr() as *mut u8).add(rgb_offset) as *mut u16).add(rgb_pos);

                let first_y = src_ptr
                    .add(yuy2_source.get_first_y_position())
                    .read_unaligned();
                let second_y = src_ptr
                    .add(yuy2_source.get_second_y_position())
                    .read_unaligned();
                let u_value = src_ptr.add(yuy2_source.get_u_position()).read_unaligned();
                let v_value = src_ptr.add(yuy2_source.get_v_position()).read_unaligned();

                let cb = u_value as i32 - bias_uv;
                let cr = v_value as i32 - bias_uv;
                let f_y = (first_y as i32 - bias_y) * y_coef;
                let s_y = (second_y as i32 - bias_y) * y_coef;

                let r0 = ((f_y + cr_coef * cr + ROUNDING_CONST) >> PRECISION).clamp(0, max_colors);
                let b0 = ((f_y + cb_coef * cb + ROUNDING_CONST) >> PRECISION).clamp(0, max_colors);
                let g0 = ((f_y - g_coef_1 * cr - g_coef_2 * cb + ROUNDING_CONST) >> PRECISION)
                    .clamp(0, max_colors);
                dst_ptr
                    .add(dst_chans.get_r_channel_offset())
                    .write_unaligned(r0 as u16);
                dst_ptr
                    .add(dst_chans.get_g_channel_offset())
                    .write_unaligned(g0 as u16);
                dst_ptr
                    .add(dst_chans.get_b_channel_offset())
                    .write_unaligned(b0 as u16);

                if dst_chans.has_alpha() {
                    dst_ptr
                        .add(dst_chans.get_a_channel_offset())
                        .write_unaligned(max_colors as u16);
                }

                let dst1 = dst_ptr.add(channels);

                let r1 = ((s_y + cr_coef * cr + ROUNDING_CONST) >> PRECISION).clamp(0, max_colors);
                let b1 = ((s_y + cb_coef * cb + ROUNDING_CONST) >> PRECISION).clamp(0, max_colors);
                let g1 = ((s_y - g_coef_1 * cr - g_coef_2 * cb + ROUNDING_CONST) >> PRECISION)
                    .clamp(0, max_colors);
                dst1.add(dst_chans.get_r_channel_offset())
                    .write_unaligned(r1 as u16);
                dst1.add(dst_chans.get_g_channel_offset())
                    .write_unaligned(g1 as u16);
                dst1.add(dst_chans.get_b_channel_offset())
                    .write_unaligned(b1 as u16);
                if dst_chans.has_alpha() {
                    dst1.add(dst_chans.get_a_channel_offset())
                        .write_unaligned(max_colors as u16);
                }
            }

            _cx += 2;
        }

        if width & 1 == 1 {
            unsafe {
                let rgb_pos = (width as usize - 1) * channels;
                let yuy2_offset = ((width as usize - 1) / 2) * 4;

                let src_ptr = ((yuy2_store.as_ptr() as *const u8).add(yuy_offset) as *const u16)
                    .add(yuy2_offset);

                let dst_ptr =
                    ((rgb_store.as_mut_ptr() as *mut u8).add(rgb_offset) as *mut u16).add(rgb_pos);

                let first_y = src_ptr
                    .add(yuy2_source.get_first_y_position())
                    .read_unaligned();
                let u_value = src_ptr.add(yuy2_source.get_u_position()).read_unaligned();
                let v_value = src_ptr.add(yuy2_source.get_v_position()).read_unaligned();

                let cb = u_value as i32 - bias_uv;
                let cr = v_value as i32 - bias_uv;
                let f_y = (first_y as i32 - bias_y) * y_coef;

                let r0 = ((f_y + cr_coef * cr + ROUNDING_CONST) >> PRECISION).clamp(0, max_colors);
                let b0 = ((f_y + cb_coef * cb + ROUNDING_CONST) >> PRECISION).clamp(0, max_colors);
                let g0 = ((f_y - g_coef_1 * cr - g_coef_2 * cb + ROUNDING_CONST) >> PRECISION)
                    .clamp(0, max_colors);
                dst_ptr
                    .add(dst_chans.get_r_channel_offset())
                    .write_unaligned(r0 as u16);
                dst_ptr
                    .add(dst_chans.get_g_channel_offset())
                    .write_unaligned(g0 as u16);
                dst_ptr
                    .add(dst_chans.get_b_channel_offset())
                    .write_unaligned(b0 as u16);
                if dst_chans.has_alpha() {
                    dst_ptr
                        .add(dst_chans.get_a_channel_offset())
                        .write_unaligned(max_colors as u16);
                }
            }
        }

        rgb_offset += rgb_stride as usize;
        yuy_offset += yuy2_stride as usize;
    }
}

/// Convert YUYV (YUV Packed) 8+ bit depth format to RGB image.
///
/// This function takes YUYV (4:2:2 Packed) format data with 8-16 bit precision,
/// and converts it to RGB with 8-16 bit per channel precision.
///
/// # Arguments
///
/// * `yuy2_store` - A slice to store the converted YUYV data.
/// * `yuy2_stride` - The stride (bytes per row) for the YUYV plane.
/// * `rgb` - A mutable slice to store the converted RGB data.
/// * `rgb_stride` - The stride (bytes per row) for the RGB image data.
/// * `bit_depth` - YUV and RGB bit depth
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
pub fn yuyv422_to_rgb_p16(
    yuy2_store: &[u16],
    yuy2_stride: u32,
    rgb: &mut [u16],
    rgb_stride: u32,
    bit_depth: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) {
    yuy2_to_rgb_impl_p16::<{ YuvSourceChannels::Rgb as u8 }, { Yuy2Description::YUYV as usize }>(
        yuy2_store,
        yuy2_stride,
        rgb,
        rgb_stride,
        bit_depth,
        width,
        height,
        range,
        matrix,
    );
}

/// Convert YUYV (YUV Packed) 8+ bit depth format to RGBA image.
///
/// This function takes YUYV (4:2:2 Packed) format data with 8-16 bit precision,
/// and converts it to RGBA with 8-16 bit per channel precision.
///
/// # Arguments
///
/// * `yuy2_store` - A slice to store the converted YUYV data.
/// * `yuy2_stride` - The stride (bytes per row) for the YUYV plane.
/// * `rgba` - A mutable slice to store the converted RGBA data.
/// * `rgba_stride` - The stride (bytes per row) for the RGBA image data.
/// * `bit_depth` - YUV and RGB bit depth
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
pub fn yuyv422_to_rgba_p16(
    yuy2_store: &[u16],
    yuy2_stride: u32,
    rgba: &mut [u16],
    rgba_stride: u32,
    bit_depth: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) {
    yuy2_to_rgb_impl_p16::<{ YuvSourceChannels::Rgba as u8 }, { Yuy2Description::YUYV as usize }>(
        yuy2_store,
        yuy2_stride,
        rgba,
        rgba_stride,
        bit_depth,
        width,
        height,
        range,
        matrix,
    );
}

/// Convert YUYV (YUV Packed) 8+ bit depth format to BGR image.
///
/// This function takes YUYV (4:2:2 Packed) format data with 8-16 bit precision,
/// and converts it to BGR with 8-16 bit per channel precision.
///
/// # Arguments
///
/// * `yuy2_store` - A slice to store the converted YUYV data.
/// * `yuy2_stride` - The stride (bytes per row) for the YUYV plane.
/// * `bgr` - A mutable slice to store the converted BGR data.
/// * `bgr_stride` - The stride (bytes per row) for the BGR image data.
/// * `bit_depth` - YUV and RGB bit depth
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
pub fn yuyv422_to_bgr_p16(
    yuy2_store: &[u16],
    yuy2_stride: u32,
    bgr: &mut [u16],
    bgr_stride: u32,
    bit_depth: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) {
    yuy2_to_rgb_impl_p16::<{ YuvSourceChannels::Bgr as u8 }, { Yuy2Description::YUYV as usize }>(
        yuy2_store,
        yuy2_stride,
        bgr,
        bgr_stride,
        bit_depth,
        width,
        height,
        range,
        matrix,
    );
}

/// Convert YUYV (YUV Packed) 8+ bit depth format to BGR image.
///
/// This function takes YUYV (4:2:2 Packed) format data with 8-16 bit precision,
/// and converts it to BGRA with 8-16 bit per channel precision.
///
/// # Arguments
///
/// * `yuy2_store` - A slice to store the converted YUYV data.
/// * `yuy2_stride` - The stride (bytes per row) for the YUYV plane.
/// * `bgra` - A mutable slice to store the converted BGRA data.
/// * `bgra_stride` - The stride (bytes per row) for the BGRA image data.
/// * `bit_depth` - YUV and RGB bit depth
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
pub fn yuyv422_to_bgra_p16(
    yuy2_store: &[u16],
    yuy2_stride: u32,
    bgra: &mut [u16],
    bgra_stride: u32,
    bit_depth: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) {
    yuy2_to_rgb_impl_p16::<{ YuvSourceChannels::Bgra as u8 }, { Yuy2Description::YUYV as usize }>(
        yuy2_store,
        yuy2_stride,
        bgra,
        bgra_stride,
        bit_depth,
        width,
        height,
        range,
        matrix,
    );
}

/// Convert UYVY (YUV Packed) 8+ bit depth format to RGB image.
///
/// This function takes UYVY (4:2:2 Packed) format data with 8-16 bit precision,
/// and converts it to RGB with 8-16 bit per channel precision.
///
/// # Arguments
///
/// * `yuy2_store` - A slice to store the converted UYVY data.
/// * `yuy2_stride` - The stride (bytes per row) for the UYVY plane.
/// * `rgb` - A mutable slice to store the converted RGB data.
/// * `rgb_stride` - The stride (bytes per row) for the RGB image data.
/// * `bit_depth` - YUV and RGB bit depth
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
pub fn uyvy422_to_rgb_p16(
    yuy2_store: &[u16],
    yuy2_stride: u32,
    rgb: &mut [u16],
    rgb_stride: u32,
    bit_depth: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) {
    yuy2_to_rgb_impl_p16::<{ YuvSourceChannels::Rgb as u8 }, { Yuy2Description::UYVY as usize }>(
        yuy2_store,
        yuy2_stride,
        rgb,
        rgb_stride,
        bit_depth,
        width,
        height,
        range,
        matrix,
    );
}

/// Convert UYVY (YUV Packed) 8+ bit depth format to RGBA image.
///
/// This function takes UYVY (4:2:2 Packed) format data with 8-16 bit precision,
/// and converts it to RGBA with 8-16 bit per channel precision.
///
/// # Arguments
///
/// * `yuy2_store` - A slice to store the converted UYVY data.
/// * `yuy2_stride` - The stride (bytes per row) for the UYVY plane.
/// * `rgba` - A mutable slice to store the converted RGBA data.
/// * `rgba_stride` - The stride (bytes per row) for the RGBA image data.
/// * `bit_depth` - YUV and RGB bit depth
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
pub fn uyvy422_to_rgba_p16(
    yuy2_store: &[u16],
    yuy2_stride: u32,
    rgba: &mut [u16],
    rgba_stride: u32,
    bit_depth: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) {
    yuy2_to_rgb_impl_p16::<{ YuvSourceChannels::Rgba as u8 }, { Yuy2Description::UYVY as usize }>(
        yuy2_store,
        yuy2_stride,
        rgba,
        rgba_stride,
        bit_depth,
        width,
        height,
        range,
        matrix,
    );
}

/// Convert UYVY (YUV Packed) 8+ bit depth format to BGR image.
///
/// This function takes UYVY (4:2:2 Packed) format data with 8-16 bit precision,
/// and converts it to BGR with 8-16 bit per channel precision.
///
/// # Arguments
///
/// * `yuy2_store` - A slice to store the converted UYVY data.
/// * `yuy2_stride` - The stride (bytes per row) for the UYVY plane.
/// * `bgr` - A mutable slice to store the converted BGR data.
/// * `bgr_stride` - The stride (bytes per row) for the BGR image data.
/// * `bit_depth` - YUV and RGB bit depth
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
pub fn uyvy422_to_bgr_p16(
    yuy2_store: &[u16],
    yuy2_stride: u32,
    bgr: &mut [u16],
    bgr_stride: u32,
    bit_depth: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) {
    yuy2_to_rgb_impl_p16::<{ YuvSourceChannels::Bgr as u8 }, { Yuy2Description::UYVY as usize }>(
        yuy2_store,
        yuy2_stride,
        bgr,
        bgr_stride,
        bit_depth,
        width,
        height,
        range,
        matrix,
    );
}

/// Convert UYVY (YUV Packed) 8+ bit depth format to BGRA image.
///
/// This function takes UYVY (4:2:2 Packed) format data with 8-16 bit precision,
/// and converts it to BGRA with 8-16 bit per channel precision.
///
/// # Arguments
///
/// * `yuy2_store` - A slice to store the converted UYVY data.
/// * `yuy2_stride` - The stride (bytes per row) for the UYVY plane.
/// * `bgra` - A mutable slice to store the converted BGRA data.
/// * `bgra_stride` - The stride (bytes per row) for the BGRA image data.
/// * `bit_depth` - YUV and RGB bit depth
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
pub fn uyvy422_to_bgra_p16(
    yuy2_store: &[u16],
    yuy2_stride: u32,
    bgra: &mut [u16],
    bgra_stride: u32,
    bit_depth: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) {
    yuy2_to_rgb_impl_p16::<{ YuvSourceChannels::Bgra as u8 }, { Yuy2Description::UYVY as usize }>(
        yuy2_store,
        yuy2_stride,
        bgra,
        bgra_stride,
        bit_depth,
        width,
        height,
        range,
        matrix,
    );
}

/// Convert YVYU ( YUV Packed ) 8+ bit depth format to RGB image.
///
/// This function takes YVYU (4:2:2 Packed) format data with 8-16 bit precision,
/// and converts it to RGB with 8-16 bit per channel precision.
///
/// # Arguments
///
/// * `yuy2_store` - A slice to store the converted YVYU data.
/// * `yuy2_stride` - The stride (bytes per row) for the YVYU plane.
/// * `rgb` - A mutable slice to store the converted RGB data.
/// * `rgb_stride` - The stride (bytes per row) for the RGB image data.
/// * `bit_depth` - YUV and RGB bit depth
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
pub fn yvyu422_to_rgb_p16(
    yuy2_store: &[u16],
    yuy2_stride: u32,
    rgb: &mut [u16],
    rgb_stride: u32,
    bit_depth: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) {
    yuy2_to_rgb_impl_p16::<{ YuvSourceChannels::Rgb as u8 }, { Yuy2Description::YVYU as usize }>(
        yuy2_store,
        yuy2_stride,
        rgb,
        rgb_stride,
        bit_depth,
        width,
        height,
        range,
        matrix,
    );
}

/// Convert YVYU (YUV Packed) 8+ bit depth format to RGBA image.
///
/// This function takes YVYU (4:2:2) format data with 8-16 bit precision,
/// and converts it to RGBA with 8-16 bit per channel precision.
///
/// # Arguments
///
/// * `yuy2_store` - A slice to store the converted YVYU data.
/// * `yuy2_stride` - The stride (bytes per row) for the YVYU plane.
/// * `rgba` - A mutable slice to store the converted RGBA data.
/// * `rgba_stride` - The stride (bytes per row) for the RGBA image data.
/// * `bit_depth` - YUV and RGB bit depth
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
pub fn yvyu422_to_rgba_p16(
    yuy2_store: &[u16],
    yuy2_stride: u32,
    rgba: &mut [u16],
    rgba_stride: u32,
    bit_depth: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) {
    yuy2_to_rgb_impl_p16::<{ YuvSourceChannels::Rgba as u8 }, { Yuy2Description::YVYU as usize }>(
        yuy2_store,
        yuy2_stride,
        rgba,
        rgba_stride,
        bit_depth,
        width,
        height,
        range,
        matrix,
    );
}

/// Convert YVYU (YUV Packed) 8+ bit depth format to BGR image.
///
/// This function takes YVYU (4:2:2 Packed) format data with 8-16 bit precision,
/// and converts it to BGR with 8-16 bit per channel precision.
///
/// # Arguments
///
/// * `yuy2_store` - A slice to store the converted YVYU data.
/// * `yuy2_stride` - The stride (bytes per row) for the YVYU plane.
/// * `bgr` - A mutable slice to store the converted BGR data.
/// * `bgr_stride` - The stride (bytes per row) for the BGR image data.
/// * `bit_depth` - YUV and RGB bit depth
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
pub fn yvyu422_to_bgr_p16(
    yuy2_store: &[u16],
    yuy2_stride: u32,
    bgr: &mut [u16],
    bgr_stride: u32,
    bit_depth: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) {
    yuy2_to_rgb_impl_p16::<{ YuvSourceChannels::Bgr as u8 }, { Yuy2Description::YVYU as usize }>(
        yuy2_store,
        yuy2_stride,
        bgr,
        bgr_stride,
        bit_depth,
        width,
        height,
        range,
        matrix,
    );
}

/// Convert YVYU (YUV Packed) 8+ bit depth format to BGRA image.
///
/// This function takes YVYU (4:2:2 Packed) format data with 8-16 bit precision,
/// and converts it to BGRA with 8-16 bit per channel precision.
///
/// # Arguments
///
/// * `yuy2_store` - A slice to store the converted YVYU data.
/// * `yuy2_stride` - The stride (bytes per row) for the YVYU plane.
/// * `bgra` - A mutable slice to store the converted BGRA data.
/// * `bgra_stride` - The stride (bytes per row) for the BGRA image data.
/// * `bit_depth` - YUV and RGB bit depth
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
pub fn yvyu422_to_bgra_p16(
    yuy2_store: &[u16],
    yuy2_stride: u32,
    bgra: &mut [u16],
    bgra_stride: u32,
    bit_depth: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) {
    yuy2_to_rgb_impl_p16::<{ YuvSourceChannels::Bgra as u8 }, { Yuy2Description::YVYU as usize }>(
        yuy2_store,
        yuy2_stride,
        bgra,
        bgra_stride,
        bit_depth,
        width,
        height,
        range,
        matrix,
    );
}

/// Convert VYUY (YUV Packed) 8+ bit depth format to RGB image.
///
/// This function takes VYUY (4:2:2 Packed) format data with 8-16 bit precision,
/// and converts it to RGB with 8-16 bit per channel precision.
///
/// # Arguments
///
/// * `yuy2_store` - A slice to store the converted VYUY data.
/// * `yuy2_stride` - The stride (bytes per row) for the VYUY plane.
/// * `rgb` - A mutable slice to store the converted RGB data.
/// * `rgb_stride` - The stride (bytes per row) for the RGB image data.
/// * `bit_depth` - YUV and RGB bit depth
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
pub fn vyuy422_to_rgb_p16(
    yuy2_store: &[u16],
    yuy2_stride: u32,
    rgb: &mut [u16],
    rgb_stride: u32,
    bit_depth: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) {
    yuy2_to_rgb_impl_p16::<{ YuvSourceChannels::Rgb as u8 }, { Yuy2Description::VYUY as usize }>(
        yuy2_store,
        yuy2_stride,
        rgb,
        rgb_stride,
        bit_depth,
        width,
        height,
        range,
        matrix,
    );
}

/// Convert VYUY (YUV Packed) 8+ bit depth format to RGBA image.
///
/// This function takes VYUY (4:2:2 Packed) format data with 8-16 bit precision,
/// and converts it to RGBA with 8-16 bit per channel precision.
///
/// # Arguments
///
/// * `yuy2_store` - A slice to store the converted VYUY data.
/// * `yuy2_stride` - The stride (bytes per row) for the VYUY plane.
/// * `rgba` - A mutable slice to store the converted RGBA data.
/// * `rgba_stride` - The stride (bytes per row) for the RGBA image data.
/// * `bit_depth` - YUV and RGB bit depth
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
pub fn vyuy422_to_rgba_p16(
    yuy2_store: &[u16],
    yuy2_stride: u32,
    rgba: &mut [u16],
    rgba_stride: u32,
    bit_depth: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) {
    yuy2_to_rgb_impl_p16::<{ YuvSourceChannels::Rgba as u8 }, { Yuy2Description::VYUY as usize }>(
        yuy2_store,
        yuy2_stride,
        rgba,
        rgba_stride,
        bit_depth,
        width,
        height,
        range,
        matrix,
    );
}

/// Convert VYUY (YUV Packed) 8+ bit depth format to BGR image.
///
/// This function takes VYUY (4:2:2 Packed) format data with 8-16 bit precision,
/// and converts it to BGR with 8-16 bit per channel precision.
///
/// # Arguments
///
/// * `yuy2_store` - A slice to store the converted VYUY data.
/// * `yuy2_stride` - The stride (bytes per row) for the VYUY plane.
/// * `bgr` - A mutable slice to store the converted BGR data.
/// * `bgr_stride` - The stride (bytes per row) for the BGR image data.
/// * `bit_depth` - YUV and RGB bit depth
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
pub fn vyuy422_to_bgr_p16(
    yuy2_store: &[u16],
    yuy2_stride: u32,
    bgr: &mut [u16],
    bgr_stride: u32,
    bit_depth: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) {
    yuy2_to_rgb_impl_p16::<{ YuvSourceChannels::Bgr as u8 }, { Yuy2Description::VYUY as usize }>(
        yuy2_store,
        yuy2_stride,
        bgr,
        bgr_stride,
        bit_depth,
        width,
        height,
        range,
        matrix,
    );
}

/// Convert VYUY (YUV Packed) 8+ bit depth format to BGRA image.
///
/// This function takes VYUY (4:2:2 Packed) format data with 8-16 bit precision,
/// and converts it to BGRA with 8-16 bit per channel precision.
///
/// # Arguments
///
/// * `yuy2_store` - A slice to store the converted VYUY data.
/// * `yuy2_stride` - The stride (bytes per row) for the VYUY plane.
/// * `bgra` - A mutable slice to store the converted BGRA data.
/// * `bgra_stride` - The stride (bytes per row) for the BGRA image data.
/// * `bit_depth` - YUV and RGB bit depth
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
pub fn vyuy422_to_bgra_p16(
    yuy2_store: &[u16],
    yuy2_stride: u32,
    bgra: &mut [u16],
    bgra_stride: u32,
    bit_depth: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) {
    yuy2_to_rgb_impl_p16::<{ YuvSourceChannels::Bgra as u8 }, { Yuy2Description::VYUY as usize }>(
        yuy2_store,
        yuy2_stride,
        bgra,
        bgra_stride,
        bit_depth,
        width,
        height,
        range,
        matrix,
    );
}
