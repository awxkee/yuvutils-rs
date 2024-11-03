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
use crate::avx2::avx2_rgba_to_yuv;
#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    feature = "nightly_avx512"
))]
use crate::avx512bw::avx512_rgba_to_yuv;
#[allow(unused_imports)]
use crate::internals::*;
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use crate::neon::neon_rgba_to_yuv;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use crate::sse::sse_rgba_to_yuv_row;
use crate::yuv_error::check_rgba_destination;
#[allow(unused_imports)]
use crate::yuv_support::*;
use crate::{YuvError, YuvPlanarImageMut};

fn rgbx_to_yuv8<const ORIGIN_CHANNELS: u8, const SAMPLING: u8>(
    planar_image: &mut YuvPlanarImageMut<u8>,
    rgba: &[u8],
    rgba_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) -> Result<(), YuvError> {
    let chroma_subsampling: YuvChromaSample = SAMPLING.into();
    let src_chans: YuvSourceChannels = ORIGIN_CHANNELS.into();
    let channels = src_chans.get_channels_count();

    check_rgba_destination(
        rgba,
        rgba_stride,
        planar_image.width,
        planar_image.height,
        channels,
    )?;
    planar_image.check_constraints(chroma_subsampling)?;

    let range = get_yuv_range(8, range);
    let kr_kb = matrix.get_kr_kb();
    const PRECISION: i32 = 12;
    let max_range_p8 = (1u32 << 8u32) - 1u32;
    let transform_precise = get_forward_transform(
        max_range_p8,
        range.range_y,
        range.range_uv,
        kr_kb.kr,
        kr_kb.kb,
    );
    let transform = transform_precise.to_integers(PRECISION as u32);

    const ROUNDING_CONST_BIAS: i32 = 1 << (PRECISION - 1);
    let bias_y = range.bias_y as i32 * (1 << PRECISION) + ROUNDING_CONST_BIAS;
    let bias_uv = range.bias_uv as i32 * (1 << PRECISION) + ROUNDING_CONST_BIAS;

    let iterator_step = match chroma_subsampling {
        YuvChromaSample::YUV420 => 2usize,
        YuvChromaSample::YUV422 => 2usize,
        YuvChromaSample::YUV444 => 1usize,
    };

    let mut y_offset = 0usize;
    let mut u_offset = 0usize;
    let mut v_offset = 0usize;
    let mut rgba_offset = 0usize;

    let i_bias_y = range.bias_y as i32;
    let i_cap_y = range.range_y as i32 + i_bias_y;
    let i_cap_uv = i_bias_y + range.range_uv as i32;

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    let mut _use_sse = std::arch::is_x86_feature_detected!("sse4.1");
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    let mut _use_avx = std::arch::is_x86_feature_detected!("avx2");
    #[cfg(all(
        any(target_arch = "x86", target_arch = "x86_64"),
        feature = "nightly_avx512"
    ))]
    let mut _use_avx512 = std::arch::is_x86_feature_detected!("avx512bw");

    let y_plane = planar_image.y_plane.borrow_mut();
    let u_plane = planar_image.u_plane.borrow_mut();
    let v_plane = planar_image.v_plane.borrow_mut();
    let width = planar_image.width;
    let y_stride = planar_image.y_stride;
    let u_stride = planar_image.u_stride;
    let v_stride = planar_image.v_stride;

    for y in 0..planar_image.height as usize {
        #[allow(unused_variables)]
        #[allow(unused_mut)]
        let mut cx = 0usize;
        #[allow(unused_variables)]
        #[allow(unused_mut)]
        let mut ux = 0usize;

        let compute_uv_row = chroma_subsampling == YuvChromaSample::YUV444
            || chroma_subsampling == YuvChromaSample::YUV422
            || y & 1 == 0;

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        unsafe {
            #[cfg(feature = "nightly_avx512")]
            {
                if _use_avx512 {
                    let processed_offset = avx512_rgba_to_yuv::<ORIGIN_CHANNELS, SAMPLING>(
                        &transform,
                        &range,
                        y_plane.as_mut_ptr().add(y_offset),
                        u_plane.as_mut_ptr().add(u_offset),
                        v_plane.as_mut_ptr().add(v_offset),
                        rgba,
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

            if _use_avx {
                let processed_offset = avx2_rgba_to_yuv::<ORIGIN_CHANNELS, SAMPLING>(
                    &transform,
                    &range,
                    y_plane.as_mut_ptr().add(y_offset),
                    u_plane.as_mut_ptr().add(u_offset),
                    v_plane.as_mut_ptr().add(v_offset),
                    rgba,
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
                let processed_offset = sse_rgba_to_yuv_row::<ORIGIN_CHANNELS, SAMPLING>(
                    &transform,
                    &range,
                    y_plane.as_mut_ptr().add(y_offset),
                    u_plane.as_mut_ptr().add(u_offset),
                    v_plane.as_mut_ptr().add(v_offset),
                    rgba,
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
            let offset = neon_rgba_to_yuv::<ORIGIN_CHANNELS, SAMPLING, PRECISION>(
                &transform,
                &range,
                y_plane.as_mut_ptr().add(y_offset),
                u_plane.as_mut_ptr().add(u_offset),
                v_plane.as_mut_ptr().add(v_offset),
                rgba,
                rgba_offset,
                cx,
                ux,
                width as usize,
                compute_uv_row,
            );
            cx = offset.cx;
            ux = offset.ux;
        }

        for x in (cx..width as usize).step_by(iterator_step) {
            let px = x * channels;
            let rgba_shift = rgba_offset + px;
            let src = unsafe { rgba.get_unchecked(rgba_shift..) };
            let r0 = unsafe { *src.get_unchecked(src_chans.get_r_channel_offset()) } as i32;
            let g0 = unsafe { *src.get_unchecked(src_chans.get_g_channel_offset()) } as i32;
            let b0 = unsafe { *src.get_unchecked(src_chans.get_b_channel_offset()) } as i32;
            let y_0 =
                (r0 * transform.yr + g0 * transform.yg + b0 * transform.yb + bias_y) >> PRECISION;
            unsafe {
                *y_plane.get_unchecked_mut(y_offset + x) = y_0.clamp(i_bias_y, i_cap_y) as u8;
            }
            let mut r1 = r0;
            let mut g1 = g0;
            let mut b1 = b0;
            match chroma_subsampling {
                YuvChromaSample::YUV420 | YuvChromaSample::YUV422 => {
                    if x + 1 < width as usize {
                        let next_px = (x + 1) * channels;
                        let rgba_shift = rgba_offset + next_px;
                        let src = unsafe { rgba.get_unchecked(rgba_shift..) };
                        r1 = unsafe { *src.get_unchecked(src_chans.get_r_channel_offset()) } as i32;
                        g1 = unsafe { *src.get_unchecked(src_chans.get_g_channel_offset()) } as i32;
                        b1 = unsafe { *src.get_unchecked(src_chans.get_b_channel_offset()) } as i32;
                        let y_1 =
                            (r1 * transform.yr + g1 * transform.yg + b1 * transform.yb + bias_y)
                                >> PRECISION;
                        unsafe {
                            *y_plane.get_unchecked_mut(y_offset + x + 1) =
                                y_1.clamp(i_bias_y, i_cap_y) as u8;
                        }
                    }
                }
                _ => {}
            }

            if compute_uv_row {
                let r = if chroma_subsampling == YuvChromaSample::YUV444 {
                    r0
                } else {
                    (r0 + r1 + 1) >> 1
                };
                let g = if chroma_subsampling == YuvChromaSample::YUV444 {
                    g0
                } else {
                    (g0 + g1 + 1) >> 1
                };
                let b = if chroma_subsampling == YuvChromaSample::YUV444 {
                    b0
                } else {
                    (b0 + b1 + 1) >> 1
                };
                let cb = (r * transform.cb_r + g * transform.cb_g + b * transform.cb_b + bias_uv)
                    >> PRECISION;
                let cr = (r * transform.cr_r + g * transform.cr_g + b * transform.cr_b + bias_uv)
                    >> PRECISION;

                let u_pos = match chroma_subsampling {
                    YuvChromaSample::YUV420 | YuvChromaSample::YUV422 => u_offset + ux,
                    YuvChromaSample::YUV444 => u_offset + ux,
                };
                unsafe {
                    *u_plane.get_unchecked_mut(u_pos) = cb.clamp(i_bias_y, i_cap_uv) as u8;
                }
                let v_pos = match chroma_subsampling {
                    YuvChromaSample::YUV420 | YuvChromaSample::YUV422 => v_offset + ux,
                    YuvChromaSample::YUV444 => v_offset + ux,
                };
                unsafe {
                    *v_plane.get_unchecked_mut(v_pos) = cr.clamp(i_bias_y, i_cap_uv) as u8;
                }
            }

            ux += 1;
        }

        y_offset += y_stride as usize;
        rgba_offset += rgba_stride as usize;
        match chroma_subsampling {
            YuvChromaSample::YUV420 => {
                if y & 1 == 1 {
                    u_offset += u_stride as usize;
                    v_offset += v_stride as usize;
                }
            }
            YuvChromaSample::YUV444 | YuvChromaSample::YUV422 => {
                u_offset += u_stride as usize;
                v_offset += v_stride as usize;
            }
        }
    }

    Ok(())
}

/// Convert RGB image data to YUV 422 planar format.
///
/// This function performs RGB to YUV conversion and stores the result in YUV422 planar format,
/// with separate planes for Y (luminance), U (chrominance), and V (chrominance) components.
///
/// # Arguments
///
/// * `planar_image` - Target planar image.
/// * `rgb` - The input RGB image data slice.
/// * `rgb_stride` - The stride (bytes per row) for the RGB image data.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input RGB data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn rgb_to_yuv422(
    planar_image: &mut YuvPlanarImageMut<u8>,
    rgb: &[u8],
    rgb_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) -> Result<(), YuvError> {
    rgbx_to_yuv8::<{ YuvSourceChannels::Rgb as u8 }, { YuvChromaSample::YUV422 as u8 }>(
        planar_image,
        rgb,
        rgb_stride,
        range,
        matrix,
    )
}

/// Convert BGR image data to YUV 422 planar format.
///
/// This function performs BGR to YUV conversion and stores the result in YUV422 planar format,
/// with separate planes for Y (luminance), U (chrominance), and V (chrominance) components.
///
/// # Arguments
///
/// * `planar_image` - Target planar image.
/// * `bgr` - The input BGR image data slice.
/// * `bgr_stride` - The stride (bytes per row) for the BGR image data.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGR data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn bgr_to_yuv422(
    planar_image: &mut YuvPlanarImageMut<u8>,
    bgr: &[u8],
    bgr_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) -> Result<(), YuvError> {
    rgbx_to_yuv8::<{ YuvSourceChannels::Bgr as u8 }, { YuvChromaSample::YUV422 as u8 }>(
        planar_image,
        bgr,
        bgr_stride,
        range,
        matrix,
    )
}

/// Convert RGBA image data to YUV 422 planar format.
///
/// This function performs RGBA to YUV conversion and stores the result in YUV422 planar format,
/// with separate planes for Y (luminance), U (chrominance), and V (chrominance) components.
///
/// # Arguments
///
/// * `planar_image` - Target planar image.
/// * `rgba` - The input RGBA image data slice.
/// * `rgba_stride` - The stride (bytes per row) for the RGBA image data.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input RGBA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn rgba_to_yuv422(
    planar_image: &mut YuvPlanarImageMut<u8>,
    rgba: &[u8],
    rgba_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) -> Result<(), YuvError> {
    rgbx_to_yuv8::<{ YuvSourceChannels::Rgba as u8 }, { YuvChromaSample::YUV422 as u8 }>(
        planar_image,
        rgba,
        rgba_stride,
        range,
        matrix,
    )
}

/// Convert BGRA image data to YUV 422 planar format.
///
/// This function performs BGRA to YUV conversion and stores the result in YUV422 planar format,
/// with separate planes for Y (luminance), U (chrominance), and V (chrominance) components.
///
/// # Arguments
///
/// * `planar_image` - Target planar image.
/// * `bgra` - The input BGRA image data slice.
/// * `bgra_stride` - The stride (bytes per row) for the BGRA image data.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGRA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn bgra_to_yuv422(
    planar_image: &mut YuvPlanarImageMut<u8>,
    bgra: &[u8],
    bgra_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) -> Result<(), YuvError> {
    rgbx_to_yuv8::<{ YuvSourceChannels::Bgra as u8 }, { YuvChromaSample::YUV422 as u8 }>(
        planar_image,
        bgra,
        bgra_stride,
        range,
        matrix,
    )
}

/// Convert RGB image data to YUV 420 planar format.
///
/// This function performs RGB to YUV conversion and stores the result in YUV420 planar format,
/// with separate planes for Y (luminance), U (chrominance), and V (chrominance) components.
///
/// # Arguments
///
/// * `planar_image` - Target planar image.
/// * `rgb` - The input RGB image data slice.
/// * `rgb_stride` - The stride (bytes per row) for the RGB image data.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input RGB data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn rgb_to_yuv420(
    planar_image: &mut YuvPlanarImageMut<u8>,
    rgb: &[u8],
    rgb_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) -> Result<(), YuvError> {
    rgbx_to_yuv8::<{ YuvSourceChannels::Rgb as u8 }, { YuvChromaSample::YUV420 as u8 }>(
        planar_image,
        rgb,
        rgb_stride,
        range,
        matrix,
    )
}

/// Convert BGR image data to YUV 420 planar format.
///
/// This function performs BGR to YUV conversion and stores the result in YUV420 planar format,
/// with separate planes for Y (luminance), U (chrominance), and V (chrominance) components.
///
/// # Arguments
///
/// * `planar_image` - Target planar image.
/// * `bgr` - The input BGR image data slice.
/// * `bgr_stride` - The stride (bytes per row) for the BGR image data.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input RGB data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn bgr_to_yuv420(
    planar_image: &mut YuvPlanarImageMut<u8>,
    bgr: &[u8],
    bgr_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) -> Result<(), YuvError> {
    rgbx_to_yuv8::<{ YuvSourceChannels::Bgr as u8 }, { YuvChromaSample::YUV420 as u8 }>(
        planar_image,
        bgr,
        bgr_stride,
        range,
        matrix,
    )
}

/// Convert RGBA image data to YUV 420 planar format.
///
/// This function performs RGBA to YUV conversion and stores the result in YUV420 planar format,
/// with separate planes for Y (luminance), U (chrominance), and V (chrominance) components.
///
/// # Arguments
///
/// * `planar_image` - Target planar image.
/// * `rgba` - The input RGBA image data slice.
/// * `rgba_stride` - The stride (bytes per row) for the RGBA image data.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input RGBA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn rgba_to_yuv420(
    planar_image: &mut YuvPlanarImageMut<u8>,
    rgba: &[u8],
    rgba_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) -> Result<(), YuvError> {
    rgbx_to_yuv8::<{ YuvSourceChannels::Rgba as u8 }, { YuvChromaSample::YUV420 as u8 }>(
        planar_image,
        rgba,
        rgba_stride,
        range,
        matrix,
    )
}

/// Convert BGRA image data to YUV 420 planar format.
///
/// This function performs BGRA to YUV conversion and stores the result in YUV420 planar format,
/// with separate planes for Y (luminance), U (chrominance), and V (chrominance) components.
///
/// # Arguments
///
/// * `planar_image` - Target planar image.
/// * `bgra` - The input BGRA image data slice.
/// * `bgra_stride` - The stride (bytes per row) for the BGRA image data.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGRA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn bgra_to_yuv420(
    planar_image: &mut YuvPlanarImageMut<u8>,
    bgra: &[u8],
    bgra_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) -> Result<(), YuvError> {
    rgbx_to_yuv8::<{ YuvSourceChannels::Bgra as u8 }, { YuvChromaSample::YUV420 as u8 }>(
        planar_image,
        bgra,
        bgra_stride,
        range,
        matrix,
    )
}

/// Convert RGB image data to YUV 444 planar format.
///
/// This function performs RGB to YUV conversion and stores the result in YUV444 planar format,
/// with separate planes for Y (luminance), U (chrominance), and V (chrominance) components.
///
/// # Arguments
///
/// * `planar_image` - Target planar image.
/// * `rgb` - The input RGB image data slice.
/// * `rgb_stride` - The stride (bytes per row) for the RGB image data.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input RGB data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn rgb_to_yuv444(
    planar_image: &mut YuvPlanarImageMut<u8>,
    rgb: &[u8],
    rgb_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) -> Result<(), YuvError> {
    rgbx_to_yuv8::<{ YuvSourceChannels::Rgb as u8 }, { YuvChromaSample::YUV444 as u8 }>(
        planar_image,
        rgb,
        rgb_stride,
        range,
        matrix,
    )
}

/// Convert BGR image data to YUV 444 planar format.
///
/// This function performs BGR to YUV conversion and stores the result in YUV444 planar format,
/// with separate planes for Y (luminance), U (chrominance), and V (chrominance) components.
///
/// # Arguments
///
/// * `planar_image` - Target planar image.
/// * `bgr` - The input BGR image data slice.
/// * `bgr_stride` - The stride (bytes per row) for the BGR image data.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGR data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn bgr_to_yuv444(
    planar_image: &mut YuvPlanarImageMut<u8>,
    bgr: &[u8],
    bgr_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) -> Result<(), YuvError> {
    rgbx_to_yuv8::<{ YuvSourceChannels::Bgr as u8 }, { YuvChromaSample::YUV444 as u8 }>(
        planar_image,
        bgr,
        bgr_stride,
        range,
        matrix,
    )
}

/// Convert RGBA image data to YUV 444 planar format.
///
/// This function performs RGBA to YUV conversion and stores the result in YUV444 planar format,
/// with separate planes for Y (luminance), U (chrominance), and V (chrominance) components.
///
/// # Arguments
///
/// * `planar_image` - Target planar image.
/// * `rgba` - The input RGBA image data slice.
/// * `rgba_stride` - The stride (bytes per row) for the RGBA image data.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input RGBA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn rgba_to_yuv444(
    planar_image: &mut YuvPlanarImageMut<u8>,
    rgba: &[u8],
    rgba_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) -> Result<(), YuvError> {
    rgbx_to_yuv8::<{ YuvSourceChannels::Rgba as u8 }, { YuvChromaSample::YUV444 as u8 }>(
        planar_image,
        rgba,
        rgba_stride,
        range,
        matrix,
    )
}

/// Convert BGRA image data to YUV 444 planar format.
///
/// This function performs BGRA to YUV conversion and stores the result in YUV444 planar format,
/// with separate planes for Y (luminance), U (chrominance), and V (chrominance) components.
///
/// # Arguments
///
/// * `planar_image` - Target planar image.
/// * `bgra` - The input BGRA image data slice.
/// * `bgra_stride` - The stride (bytes per row) for the BGRA image data.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGRA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn bgra_to_yuv444(
    planar_image: &mut YuvPlanarImageMut<u8>,
    bgra: &[u8],
    bgra_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) -> Result<(), YuvError> {
    rgbx_to_yuv8::<{ YuvSourceChannels::Bgra as u8 }, { YuvChromaSample::YUV444 as u8 }>(
        planar_image,
        bgra,
        bgra_stride,
        range,
        matrix,
    )
}
