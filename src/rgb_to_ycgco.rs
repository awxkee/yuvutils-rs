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
use crate::avx2::avx2_rgb_to_ycgco_row;
#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    feature = "nightly_avx512"
))]
use crate::avx512bw::avx512_rgb_to_ycgco_row;
#[allow(unused_imports)]
use crate::internals::ProcessedOffset;
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use crate::neon::neon_rgb_to_ycgco_row;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use crate::sse::sse_rgb_to_ycgco_row;
#[allow(unused_imports)]
use crate::yuv_support::*;

fn rgbx_to_ycgco<const ORIGIN_CHANNELS: u8, const SAMPLING: u8>(
    y_plane: &mut [u8],
    y_stride: u32,
    cg_plane: &mut [u8],
    cg_stride: u32,
    co_plane: &mut [u8],
    co_stride: u32,
    rgba: &[u8],
    rgba_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
) {
    let chroma_subsampling: YuvChromaSample = SAMPLING.into();
    let source_channels: YuvSourceChannels = ORIGIN_CHANNELS.into();
    let channels = source_channels.get_channels_count();
    let range = get_yuv_range(8, range);
    let precision_scale = (1 << 8) as f32;
    let bias_y = ((range.bias_y as f32 + 0.5f32) * precision_scale) as i32;
    let bias_uv = ((range.bias_uv as f32 + 0.5f32) * precision_scale) as i32;
    let max_colors = (1 << 8) - 1i32;

    let iterator_step = match chroma_subsampling {
        YuvChromaSample::Yuv420 => 2usize,
        YuvChromaSample::Yuv422 => 2usize,
        YuvChromaSample::Yuv444 => 1usize,
    };

    let range_reduction_y =
        (range.range_y as f32 / max_colors as f32 * precision_scale).round() as i32;
    let range_reduction_uv =
        (range.range_uv as f32 / max_colors as f32 * precision_scale).round() as i32;

    let mut y_offset = 0usize;
    let mut cg_offset = 0usize;
    let mut co_offset = 0usize;
    let mut rgba_offset = 0usize;

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    let mut _use_sse = std::arch::is_x86_feature_detected!("sse4.1");
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    let mut _use_avx = std::arch::is_x86_feature_detected!("avx2");
    #[cfg(all(
        any(target_arch = "x86", target_arch = "x86_64"),
        feature = "nightly_avx512"
    ))]
    let mut _use_avx512 = std::arch::is_x86_feature_detected!("avx512bw");

    for y in 0..height as usize {
        #[allow(unused_variables)]
        #[allow(unused_mut)]
        let mut cx = 0usize;
        #[allow(unused_variables)]
        #[allow(unused_mut)]
        let mut ux = 0usize;

        let compute_uv_row = chroma_subsampling == YuvChromaSample::Yuv444
            || chroma_subsampling == YuvChromaSample::Yuv422
            || y & 1 == 0;

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        unsafe {
            #[cfg(feature = "nightly_avx512")]
            if _use_avx512 {
                let processed_offset = avx512_rgb_to_ycgco_row::<ORIGIN_CHANNELS, SAMPLING>(
                    &range,
                    y_plane.as_mut_ptr(),
                    cg_plane.as_mut_ptr(),
                    co_plane.as_mut_ptr(),
                    rgba,
                    y_offset,
                    cg_offset,
                    co_offset,
                    rgba_offset,
                    cx,
                    ux,
                    width as usize,
                    compute_uv_row,
                );
                cx = processed_offset.cx;
                ux = processed_offset.ux;
            }
            if _use_avx {
                let processed_offset = avx2_rgb_to_ycgco_row::<ORIGIN_CHANNELS, SAMPLING>(
                    &range,
                    y_plane.as_mut_ptr(),
                    cg_plane.as_mut_ptr(),
                    co_plane.as_mut_ptr(),
                    rgba,
                    y_offset,
                    cg_offset,
                    co_offset,
                    rgba_offset,
                    cx,
                    ux,
                    width as usize,
                    compute_uv_row,
                );
                cx = processed_offset.cx;
                ux = processed_offset.ux;
            }
            if _use_sse {
                let processed_offset = sse_rgb_to_ycgco_row::<ORIGIN_CHANNELS, SAMPLING>(
                    &range,
                    y_plane.as_mut_ptr(),
                    cg_plane.as_mut_ptr(),
                    co_plane.as_mut_ptr(),
                    rgba,
                    y_offset,
                    cg_offset,
                    co_offset,
                    rgba_offset,
                    cx,
                    ux,
                    width as usize,
                    compute_uv_row,
                );
                cx = processed_offset.cx;
                ux = processed_offset.ux;
            }
        }

        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        unsafe {
            let processed_offset = neon_rgb_to_ycgco_row::<ORIGIN_CHANNELS, SAMPLING>(
                &range,
                y_plane.as_mut_ptr(),
                cg_plane.as_mut_ptr(),
                co_plane.as_mut_ptr(),
                rgba,
                y_offset,
                cg_offset,
                co_offset,
                rgba_offset,
                cx,
                ux,
                width as usize,
                compute_uv_row,
            );
            cx = processed_offset.cx;
            ux = processed_offset.ux;
        }

        for x in (cx..width as usize).step_by(iterator_step) {
            let px = x * channels;
            let rgba_shift = rgba_offset + px;
            let src0 = unsafe { rgba.get_unchecked(rgba_shift..) };
            let r0 = unsafe { *src0.get_unchecked(source_channels.get_r_channel_offset()) } as i32;
            let g0 = unsafe { *src0.get_unchecked(source_channels.get_g_channel_offset()) } as i32;
            let b0 = unsafe { *src0.get_unchecked(source_channels.get_b_channel_offset()) } as i32;

            let mut r1 = r0;
            let mut g1 = g0;
            let mut b1 = b0;

            let hg = (g0 * range_reduction_y) >> 1;
            let y_0 = (hg + ((r0 * range_reduction_y + b0 * range_reduction_y) >> 2) + bias_y) >> 8;
            unsafe { *y_plane.get_unchecked_mut(y_offset + x) = y_0 as u8 };
            match chroma_subsampling {
                YuvChromaSample::Yuv420 | YuvChromaSample::Yuv422 => {
                    if x + 1 < width as usize {
                        let next_px = (x + 1) * channels;
                        let rgba_shift = rgba_offset + next_px;
                        let src1 = unsafe { rgba.get_unchecked(rgba_shift..) };
                        r1 = unsafe { *src1.get_unchecked(source_channels.get_r_channel_offset()) }
                            as i32;
                        g1 = unsafe { *src1.get_unchecked(source_channels.get_g_channel_offset()) }
                            as i32;
                        b1 = unsafe { *src1.get_unchecked(source_channels.get_b_channel_offset()) }
                            as i32;
                        let hg_1 = (g1 * range_reduction_y) >> 1;
                        let y_1 = (hg_1
                            + ((r1 * range_reduction_y + b1 * range_reduction_y) >> 2)
                            + bias_y)
                            >> 8;
                        unsafe { *y_plane.get_unchecked_mut(y_offset + x + 1) = y_1 as u8 };
                    }
                }
                _ => {}
            }

            if compute_uv_row {
                let mut r = if chroma_subsampling == YuvChromaSample::Yuv444 {
                    r0
                } else {
                    (r0 + r1 + 1) >> 1
                };
                let mut g = if chroma_subsampling == YuvChromaSample::Yuv444 {
                    g0
                } else {
                    (g0 + g1 + 1) >> 1
                };
                let mut b = if chroma_subsampling == YuvChromaSample::Yuv444 {
                    b0
                } else {
                    (b0 + b1 + 1) >> 1
                };
                r *= range_reduction_uv;
                g *= range_reduction_uv;
                b *= range_reduction_uv;
                let cg = (((g >> 1) - ((r + b) >> 2)) + bias_uv) >> 8;
                let co = (((r - b) >> 1) + bias_uv) >> 8;
                let u_pos = match chroma_subsampling {
                    YuvChromaSample::Yuv420 | YuvChromaSample::Yuv422 => cg_offset + ux,
                    YuvChromaSample::Yuv444 => cg_offset + ux,
                };
                unsafe { *cg_plane.get_unchecked_mut(u_pos) = cg as u8 };
                let v_pos = match chroma_subsampling {
                    YuvChromaSample::Yuv420 | YuvChromaSample::Yuv422 => co_offset + ux,
                    YuvChromaSample::Yuv444 => co_offset + ux,
                };
                unsafe { *co_plane.get_unchecked_mut(v_pos) = co as u8 };
            }

            ux += 1;
        }

        y_offset += y_stride as usize;
        rgba_offset += rgba_stride as usize;
        match chroma_subsampling {
            YuvChromaSample::Yuv420 => {
                if y & 1 == 1 {
                    cg_offset += cg_stride as usize;
                    co_offset += co_stride as usize;
                }
            }
            YuvChromaSample::Yuv444 | YuvChromaSample::Yuv422 => {
                cg_offset += cg_stride as usize;
                co_offset += co_stride as usize;
            }
        }
    }
}

/// Convert RGB image data to YCgCo 422 planar format.
///
/// This function performs RGB to YCgCo conversion and stores the result in YUV422 planar format,
/// with separate planes for Y (luminance), Cg (chrominance), and Co (chrominance) components.
/// YCgCo is very fast transformation by its nature. If you just work if intensity (Y channel) and do not require YCbCr prefer this one over YCbCr
///
/// # Arguments
///
/// * `y_plane` - A mutable slice to store the Y (luminance) plane data.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `cg_plane` - A mutable slice to store the Cg (chrominance) plane data.
/// * `cg_stride` - The stride (bytes per row) for the Cg plane.
/// * `co_plane` - A mutable slice to store the Co (chrominance) plane data.
/// * `co_stride` - The stride (bytes per row) for the Co plane.
/// * `rgb` - The input RGB image data slice.
/// * `rgb_stride` - The stride (bytes per row) for the RGB image data.
/// * `width` - The width of the image in pixels.
/// * `height` - The height of the image in pixels.
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input RGB data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn rgb_to_ycgco422(
    y_plane: &mut [u8],
    y_stride: u32,
    cg_plane: &mut [u8],
    cg_stride: u32,
    co_plane: &mut [u8],
    co_stride: u32,
    rgb: &[u8],
    rgb_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
) {
    rgbx_to_ycgco::<{ YuvSourceChannels::Rgb as u8 }, { YuvChromaSample::Yuv422 as u8 }>(
        y_plane, y_stride, cg_plane, cg_stride, co_plane, co_stride, rgb, rgb_stride, width,
        height, range,
    );
}

/// Convert BGR image data to YCgCo 422 planar format.
///
/// This function performs BGR to YCgCo conversion and stores the result in YUV422 planar format,
/// with separate planes for Y (luminance), Cg (chrominance), and Co (chrominance) components.
/// YCgCo is very fast transformation by its nature. If you just work if intensity (Y channel) and do not require YCbCr prefer this one over YCbCr
///
/// # Arguments
///
/// * `y_plane` - A mutable slice to store the Y (luminance) plane data.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `cg_plane` - A mutable slice to store the Cg (chrominance) plane data.
/// * `cg_stride` - The stride (bytes per row) for the Cg plane.
/// * `co_plane` - A mutable slice to store the Co (chrominance) plane data.
/// * `co_stride` - The stride (bytes per row) for the Co plane.
/// * `bgr` - The input BGR image data slice.
/// * `bgr_stride` - The stride (bytes per row) for the BGR image data.
/// * `width` - The width of the image in pixels.
/// * `height` - The height of the image in pixels.
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input RGB data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn bgr_to_ycgco422(
    y_plane: &mut [u8],
    y_stride: u32,
    cg_plane: &mut [u8],
    cg_stride: u32,
    co_plane: &mut [u8],
    co_stride: u32,
    bgr: &[u8],
    bgr_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
) {
    rgbx_to_ycgco::<{ YuvSourceChannels::Bgr as u8 }, { YuvChromaSample::Yuv422 as u8 }>(
        y_plane, y_stride, cg_plane, cg_stride, co_plane, co_stride, bgr, bgr_stride, width,
        height, range,
    );
}

/// Convert RGBA image data to YCgCo 422 planar format.
///
/// This function performs RGBA to YCgCo conversion and stores the result in YUV422 planar format,
/// with separate planes for Y (luminance), Cg (chrominance), and Co (chrominance) components.
/// YCgCo is very fast transformation by its nature. If you just work if intensity (Y channel) and do not require YCbCr prefer this one over YCbCr
///
/// # Arguments
///
/// * `y_plane` - A mutable slice to store the Y (luminance) plane data.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `cg_plane` - A mutable slice to store the Cg (chrominance) plane data.
/// * `cg_stride` - The stride (bytes per row) for the Cg plane.
/// * `co_plane` - A mutable slice to store the Co (chrominance) plane data.
/// * `co_stride` - The stride (bytes per row) for the Co plane.
/// * `rgba` - The input RGBA image data slice.
/// * `rgba_stride` - The stride (bytes per row) for the RGBA image data.
/// * `width` - The width of the image in pixels.
/// * `height` - The height of the image in pixels.
/// * `range` - The YUV range (limited or full).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input RGBA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn rgba_to_ycgco422(
    y_plane: &mut [u8],
    y_stride: u32,
    cg_plane: &mut [u8],
    cg_stride: u32,
    co_plane: &mut [u8],
    co_stride: u32,
    rgba: &[u8],
    rgba_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
) {
    rgbx_to_ycgco::<{ YuvSourceChannels::Rgba as u8 }, { YuvChromaSample::Yuv422 as u8 }>(
        y_plane,
        y_stride,
        cg_plane,
        cg_stride,
        co_plane,
        co_stride,
        rgba,
        rgba_stride,
        width,
        height,
        range,
    );
}

/// Convert BGRA image data to YCgCo 422 planar format.
///
/// This function performs BGRA to YCgCo conversion and stores the result in YUV422 planar format,
/// with separate planes for Y (luminance), Cg (chrominance), and Co (chrominance) components.
/// YCgCo is very fast transformation by its nature. If you just work if intensity (Y channel) and do not require YCbCr prefer this one over YCbCr
///
/// # Arguments
///
/// * `y_plane` - A mutable slice to store the Y (luminance) plane data.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `cg_plane` - A mutable slice to store the Cg (chrominance) plane data.
/// * `cg_stride` - The stride (bytes per row) for the Cg plane.
/// * `co_plane` - A mutable slice to store the Co (chrominance) plane data.
/// * `co_stride` - The stride (bytes per row) for the Co plane.
/// * `bgra` - The input BGRA image data slice.
/// * `bgra_stride` - The stride (bytes per row) for the BGRA image data.
/// * `width` - The width of the image in pixels.
/// * `height` - The height of the image in pixels.
/// * `range` - The YUV range (limited or full).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGRA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn bgra_to_ycgco422(
    y_plane: &mut [u8],
    y_stride: u32,
    cg_plane: &mut [u8],
    cg_stride: u32,
    co_plane: &mut [u8],
    co_stride: u32,
    bgra: &[u8],
    bgra_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
) {
    rgbx_to_ycgco::<{ YuvSourceChannels::Bgra as u8 }, { YuvChromaSample::Yuv422 as u8 }>(
        y_plane,
        y_stride,
        cg_plane,
        cg_stride,
        co_plane,
        co_stride,
        bgra,
        bgra_stride,
        width,
        height,
        range,
    );
}

/// Convert RGB image data to YCgCo 420 planar format.
///
/// This function performs RGB to YCgCo conversion and stores the result in YUV420 planar format,
/// with separate planes for Y (luminance), Cg (chrominance), and Co (chrominance) components.
/// YCgCo is very fast transformation by its nature. If you just work if intensity (Y channel) and do not require YCbCr prefer this one over YCbCr
///
/// # Arguments
///
/// * `y_plane` - A mutable slice to store the Y (luminance) plane data.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `cg_plane` - A mutable slice to store the Cg (chrominance) plane data.
/// * `cg_stride` - The stride (bytes per row) for the Cg plane.
/// * `co_plane` - A mutable slice to store the Co (chrominance) plane data.
/// * `co_stride` - The stride (bytes per row) for the Co plane.
/// * `rgb` - The input RGB image data slice.
/// * `rgb_stride` - The stride (bytes per row) for the RGB image data.
/// * `width` - The width of the image in pixels.
/// * `height` - The height of the image in pixels.
/// * `range` - The YUV range (limited or full).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input RGB data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn rgb_to_ycgco420(
    y_plane: &mut [u8],
    y_stride: u32,
    cg_plane: &mut [u8],
    cg_stride: u32,
    co_plane: &mut [u8],
    co_stride: u32,
    rgb: &[u8],
    rgb_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
) {
    rgbx_to_ycgco::<{ YuvSourceChannels::Rgb as u8 }, { YuvChromaSample::Yuv420 as u8 }>(
        y_plane, y_stride, cg_plane, cg_stride, co_plane, co_stride, rgb, rgb_stride, width,
        height, range,
    );
}

/// Convert BGR image data to YCgCo 420 planar format.
///
/// This function performs BGR to YCgCo conversion and stores the result in YUV420 planar format,
/// with separate planes for Y (luminance), Cg (chrominance), and Co (chrominance) components.
/// YCgCo is very fast transformation by its nature. If you just work if intensity (Y channel) and do not require YCbCr prefer this one over YCbCr
///
/// # Arguments
///
/// * `y_plane` - A mutable slice to store the Y (luminance) plane data.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `cg_plane` - A mutable slice to store the Cg (chrominance) plane data.
/// * `cg_stride` - The stride (bytes per row) for the Cg plane.
/// * `co_plane` - A mutable slice to store the Co (chrominance) plane data.
/// * `co_stride` - The stride (bytes per row) for the Co plane.
/// * `bgr` - The input BGR image data slice.
/// * `bgr_stride` - The stride (bytes per row) for the BGR image data.
/// * `width` - The width of the image in pixels.
/// * `height` - The height of the image in pixels.
/// * `range` - The YUV range (limited or full).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGR data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn bgr_to_ycgco420(
    y_plane: &mut [u8],
    y_stride: u32,
    cg_plane: &mut [u8],
    cg_stride: u32,
    co_plane: &mut [u8],
    co_stride: u32,
    bgr: &[u8],
    bgr_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
) {
    rgbx_to_ycgco::<{ YuvSourceChannels::Bgr as u8 }, { YuvChromaSample::Yuv420 as u8 }>(
        y_plane, y_stride, cg_plane, cg_stride, co_plane, co_stride, bgr, bgr_stride, width,
        height, range,
    );
}

/// Convert RGBA image data to YCgCo 420 planar format.
///
/// This function performs RGBA to YCgCo conversion and stores the result in YUV420 planar format,
/// with separate planes for Y (luminance), Cg (chrominance), and Co (chrominance) components.
/// YCgCo is very fast transformation by its nature. If you just work if intensity (Y channel) and do not require YCbCr prefer this one over YCbCr
///
/// # Arguments
///
/// * `y_plane` - A mutable slice to store the Y (luminance) plane data.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `cg_plane` - A mutable slice to store the Cg (chrominance) plane data.
/// * `cg_stride` - The stride (bytes per row) for the Cg plane.
/// * `co_plane` - A mutable slice to store the Co (chrominance) plane data.
/// * `co_stride` - The stride (bytes per row) for the Co plane.
/// * `rgba` - The input RGBA image data slice.
/// * `rgba_stride` - The stride (bytes per row) for the RGBA image data.
/// * `width` - The width of the image in pixels.
/// * `height` - The height of the image in pixels.
/// * `range` - The YUV range (limited or full).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input RGBA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn rgba_to_ycgco420(
    y_plane: &mut [u8],
    y_stride: u32,
    cg_plane: &mut [u8],
    cg_stride: u32,
    co_plane: &mut [u8],
    co_stride: u32,
    rgba: &[u8],
    rgba_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
) {
    rgbx_to_ycgco::<{ YuvSourceChannels::Rgba as u8 }, { YuvChromaSample::Yuv420 as u8 }>(
        y_plane,
        y_stride,
        cg_plane,
        cg_stride,
        co_plane,
        co_stride,
        rgba,
        rgba_stride,
        width,
        height,
        range,
    );
}

/// Convert BGRA image data to YCgCo 420 planar format.
///
/// This function performs BGRA to YCgCo conversion and stores the result in YUV420 planar format,
/// with separate planes for Y (luminance), Cg (chrominance), and Co (chrominance) components.
/// YCgCo is very fast transformation by its nature. If you just work if intensity (Y channel) and do not require YCbCr prefer this one over YCbCr
///
/// # Arguments
///
/// * `y_plane` - A mutable slice to store the Y (luminance) plane data.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `cg_plane` - A mutable slice to store the Cg (chrominance) plane data.
/// * `cg_stride` - The stride (bytes per row) for the Cg plane.
/// * `co_plane` - A mutable slice to store the Co (chrominance) plane data.
/// * `co_stride` - The stride (bytes per row) for the Co plane.
/// * `bgra` - The input BGRA image data slice.
/// * `bgra_stride` - The stride (bytes per row) for the BGRA image data.
/// * `width` - The width of the image in pixels.
/// * `height` - The height of the image in pixels.
/// * `range` - The YUV range (limited or full).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGRA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn bgra_to_ycgco420(
    y_plane: &mut [u8],
    y_stride: u32,
    cg_plane: &mut [u8],
    cg_stride: u32,
    co_plane: &mut [u8],
    co_stride: u32,
    bgra: &[u8],
    bgra_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
) {
    rgbx_to_ycgco::<{ YuvSourceChannels::Bgra as u8 }, { YuvChromaSample::Yuv420 as u8 }>(
        y_plane,
        y_stride,
        cg_plane,
        cg_stride,
        co_plane,
        co_stride,
        bgra,
        bgra_stride,
        width,
        height,
        range,
    );
}

/// Convert RGB image data to YCgCo 444 planar format.
///
/// This function performs RGB to YCgCo conversion and stores the result in YUV444 planar format,
/// with separate planes for Y (luminance), Cg (chrominance), and Co (chrominance) components.
/// YCgCo is very fast transformation by its nature. If you just work if intensity (Y channel) and do not require YCbCr prefer this one over YCbCr
///
/// # Arguments
///
/// * `y_plane` - A mutable slice to store the Y (luminance) plane data.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `cg_plane` - A mutable slice to store the Cg (chrominance) plane data.
/// * `cg_stride` - The stride (bytes per row) for the Cg plane.
/// * `co_plane` - A mutable slice to store the Co (chrominance) plane data.
/// * `co_stride` - The stride (bytes per row) for the Co plane.
/// * `rgb` - The input RGB image data slice.
/// * `rgb_stride` - The stride (bytes per row) for the RGB image data.
/// * `width` - The width of the image in pixels.
/// * `height` - The height of the image in pixels.
/// * `range` - The YUV range (limited or full).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input RGB data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn rgb_to_ycgco444(
    y_plane: &mut [u8],
    y_stride: u32,
    cg_plane: &mut [u8],
    cg_stride: u32,
    co_plane: &mut [u8],
    co_stride: u32,
    rgb: &[u8],
    rgb_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
) {
    rgbx_to_ycgco::<{ YuvSourceChannels::Rgb as u8 }, { YuvChromaSample::Yuv444 as u8 }>(
        y_plane, y_stride, cg_plane, cg_stride, co_plane, co_stride, rgb, rgb_stride, width,
        height, range,
    );
}

/// Convert BGR image data to YCgCo 444 planar format.
///
/// This function performs BGR to YCgCo conversion and stores the result in YUV444 planar format,
/// with separate planes for Y (luminance), Cg (chrominance), and Co (chrominance) components.
/// YCgCo is very fast transformation by its nature. If you just work if intensity (Y channel) and do not require YCbCr prefer this one over YCbCr
///
/// # Arguments
///
/// * `y_plane` - A mutable slice to store the Y (luminance) plane data.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `cg_plane` - A mutable slice to store the Cg (chrominance) plane data.
/// * `cg_stride` - The stride (bytes per row) for the Cg plane.
/// * `co_plane` - A mutable slice to store the Co (chrominance) plane data.
/// * `co_stride` - The stride (bytes per row) for the Co plane.
/// * `bgr` - The input RGB image data slice.
/// * `bgr_stride` - The stride (bytes per row) for the BGR image data.
/// * `width` - The width of the image in pixels.
/// * `height` - The height of the image in pixels.
/// * `range` - The YUV range (limited or full).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGR data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn bgr_to_ycgco444(
    y_plane: &mut [u8],
    y_stride: u32,
    cg_plane: &mut [u8],
    cg_stride: u32,
    co_plane: &mut [u8],
    co_stride: u32,
    bgr: &[u8],
    bgr_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
) {
    rgbx_to_ycgco::<{ YuvSourceChannels::Bgr as u8 }, { YuvChromaSample::Yuv444 as u8 }>(
        y_plane, y_stride, cg_plane, cg_stride, co_plane, co_stride, bgr, bgr_stride, width,
        height, range,
    );
}

/// Convert RGBA image data to YCgCo 444 planar format.
///
/// This function performs RGBA to YCgCo conversion and stores the result in YUV444 planar format,
/// with separate planes for Y (luminance), Cg (chrominance), and Co (chrominance) components.
/// YCgCo is very fast transformation by its nature. If you just work if intensity (Y channel) and do not require YCbCr prefer this one over YCbCr
///
/// # Arguments
///
/// * `y_plane` - A mutable slice to store the Y (luminance) plane data.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `cg_plane` - A mutable slice to store the Cg (chrominance) plane data.
/// * `cg_stride` - The stride (bytes per row) for the Cg plane.
/// * `co_plane` - A mutable slice to store the Co (chrominance) plane data.
/// * `co_stride` - The stride (bytes per row) for the Co plane.
/// * `rgba` - The input RGBA image data slice.
/// * `rgba_stride` - The stride (bytes per row) for the RGBA image data.
/// * `width` - The width of the image in pixels.
/// * `height` - The height of the image in pixels.
/// * `range` - The YUV range (limited or full).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input RGBA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn rgba_to_ycgco444(
    y_plane: &mut [u8],
    y_stride: u32,
    cg_plane: &mut [u8],
    cg_stride: u32,
    co_plane: &mut [u8],
    co_stride: u32,
    rgba: &[u8],
    rgba_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
) {
    rgbx_to_ycgco::<{ YuvSourceChannels::Rgba as u8 }, { YuvChromaSample::Yuv444 as u8 }>(
        y_plane,
        y_stride,
        cg_plane,
        cg_stride,
        co_plane,
        co_stride,
        rgba,
        rgba_stride,
        width,
        height,
        range,
    );
}

/// Convert BGRA image data to YCgCo 444 planar format.
///
/// This function performs BGRA to YCgCo conversion and stores the result in YUV444 planar format,
/// with separate planes for Y (luminance), Cg (chrominance), and Co (chrominance) components.
/// YCgCo is very fast transformation by its nature. If you just work if intensity (Y channel) and do not require YCbCr prefer this one over YCbCr
///
/// # Arguments
///
/// * `y_plane` - A mutable slice to store the Y (luminance) plane data.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `cg_plane` - A mutable slice to store the Cg (chrominance) plane data.
/// * `cg_stride` - The stride (bytes per row) for the Cg plane.
/// * `co_plane` - A mutable slice to store the Co (chrominance) plane data.
/// * `co_stride` - The stride (bytes per row) for the Co plane.
/// * `bgra` - The input BGRA image data slice.
/// * `bgra_stride` - The stride (bytes per row) for the BGRA image data.
/// * `width` - The width of the image in pixels.
/// * `height` - The height of the image in pixels.
/// * `range` - The YUV range (limited or full).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGRA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn bgra_to_ycgco444(
    y_plane: &mut [u8],
    y_stride: u32,
    cg_plane: &mut [u8],
    cg_stride: u32,
    co_plane: &mut [u8],
    co_stride: u32,
    bgra: &[u8],
    bgra_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
) {
    rgbx_to_ycgco::<{ YuvSourceChannels::Bgra as u8 }, { YuvChromaSample::Yuv444 as u8 }>(
        y_plane,
        y_stride,
        cg_plane,
        cg_stride,
        co_plane,
        co_stride,
        bgra,
        bgra_stride,
        width,
        height,
        range,
    );
}
