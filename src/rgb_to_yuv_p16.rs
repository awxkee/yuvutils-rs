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
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use crate::neon::neon_rgba_to_yuv_p16;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use crate::sse::sse_rgba_to_yuv_p16;
use crate::yuv_support::{
    get_forward_transform, get_yuv_range, ToIntegerTransform, YuvChromaSubsample, YuvSourceChannels,
};
use crate::{
    YuvBytesPacking, YuvEndianness, YuvError, YuvPlanarImageMut, YuvRange, YuvStandardMatrix,
};

#[inline(always)]
fn transform_integer<const ENDIANNESS: u8, const BYTES_POSITION: u8>(
    v: i32,
    bit_depth: u32,
) -> u16 {
    let endianness: YuvEndianness = ENDIANNESS.into();
    let bytes_position: YuvBytesPacking = BYTES_POSITION.into();
    let packing: i32 = 16 - bit_depth as i32;
    let packed_bytes = match bytes_position {
        YuvBytesPacking::MostSignificantBytes => v << packing,
        YuvBytesPacking::LeastSignificantBytes => v,
    } as u16;
    match endianness {
        YuvEndianness::BigEndian => packed_bytes.to_be(),
        YuvEndianness::LittleEndian => packed_bytes.to_le(),
    }
}

fn rgbx_to_yuv_impl<
    const ORIGIN_CHANNELS: u8,
    const SAMPLING: u8,
    const ENDIANNESS: u8,
    const BYTES_POSITION: u8,
>(
    planar_image: &mut YuvPlanarImageMut<u16>,
    rgba: &[u16],
    rgba_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    bit_depth: u32,
) -> Result<(), YuvError> {
    let chroma_subsampling: YuvChromaSubsample = SAMPLING.into();
    let src_chans: YuvSourceChannels = ORIGIN_CHANNELS.into();
    let channels = src_chans.get_channels_count();

    planar_image.check_constraints(chroma_subsampling)?;

    let range = get_yuv_range(bit_depth, range);
    let kr_kb = matrix.get_kr_kb();
    let max_range_p8 = (1u32 << bit_depth) - 1u32;
    let transform_precise = get_forward_transform(
        max_range_p8,
        range.range_y,
        range.range_uv,
        kr_kb.kr,
        kr_kb.kb,
    );
    let transform = transform_precise.to_integers(8);
    const PRECISION: i32 = 8;
    const ROUNDING_CONST_BIAS: i32 = 1 << (PRECISION - 1);
    let bias_y = range.bias_y as i32 * (1 << PRECISION) + ROUNDING_CONST_BIAS;
    let bias_uv = range.bias_uv as i32 * (1 << PRECISION) + ROUNDING_CONST_BIAS;

    let iterator_step = match chroma_subsampling {
        YuvChromaSubsample::Yuv420 => 2usize,
        YuvChromaSubsample::Yuv422 => 2usize,
        YuvChromaSubsample::Yuv444 => 1usize,
    };

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    let mut _use_sse = std::arch::is_x86_feature_detected!("sse4.1");

    let width = planar_image.width;
    let y_stride = planar_image.y_stride;
    let u_stride = planar_image.u_stride;
    let v_stride = planar_image.v_stride;

    for y in 0..planar_image.height as usize {
        let y_offset = y * y_stride as usize;
        let u_offset = if chroma_subsampling == YuvChromaSubsample::Yuv420 {
            (y >> 1) * (u_stride as usize)
        } else {
            y * (u_stride as usize)
        };
        let v_offset = if chroma_subsampling == YuvChromaSubsample::Yuv420 {
            (y >> 1) * (v_stride as usize)
        } else {
            y * (v_stride as usize)
        };
        let rgba_offset = rgba_stride as usize * y;

        let rgba = &rgba[rgba_offset..];
        let y_plane = &mut planar_image.y_plane.borrow_mut()[y_offset..];
        let u_plane = &mut planar_image.u_plane.borrow_mut()[u_offset..];
        let v_plane = &mut planar_image.v_plane.borrow_mut()[v_offset..];

        let mut _cx = 0usize;
        let mut _ux = 0usize;

        let compute_uv_row = chroma_subsampling == YuvChromaSubsample::Yuv444
            || chroma_subsampling == YuvChromaSubsample::Yuv422
            || y & 1 == 0;

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        unsafe {
            if _use_sse {
                let offset =
                    sse_rgba_to_yuv_p16::<ORIGIN_CHANNELS, SAMPLING, ENDIANNESS, BYTES_POSITION>(
                        &transform,
                        &range,
                        y_plane.as_mut_ptr(),
                        u_plane.as_mut_ptr(),
                        v_plane.as_mut_ptr(),
                        rgba.as_ptr(),
                        _cx,
                        _ux,
                        width as usize,
                        compute_uv_row,
                        bit_depth,
                    );
                _cx = offset.cx;
                _ux = offset.ux;
            }
        }

        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        unsafe {
            let offset =
                neon_rgba_to_yuv_p16::<ORIGIN_CHANNELS, SAMPLING, ENDIANNESS, BYTES_POSITION>(
                    &transform,
                    &range,
                    y_plane.as_mut_ptr(),
                    u_plane.as_mut_ptr(),
                    v_plane.as_mut_ptr(),
                    rgba.as_ptr(),
                    _cx,
                    _ux,
                    width as usize,
                    compute_uv_row,
                    bit_depth,
                );
            _cx = offset.cx;
            _ux = offset.ux;
        }

        for x in (_cx..width as usize).step_by(iterator_step) {
            let px = x * channels;
            let src = unsafe { rgba.get_unchecked(px..(px + iterator_step * channels)) };
            let r0 = src[src_chans.get_r_channel_offset()] as i32;
            let g0 = src[src_chans.get_g_channel_offset()] as i32;
            let b0 = src[src_chans.get_b_channel_offset()] as i32;

            let mut r1 = r0;
            let mut g1 = g0;
            let mut b1 = b0;

            let y_0 =
                (r0 * transform.yr + g0 * transform.yg + b0 * transform.yb + bias_y) >> PRECISION;
            unsafe {
                *y_plane.get_unchecked_mut(x) =
                    transform_integer::<ENDIANNESS, BYTES_POSITION>(y_0, bit_depth);
            }

            match chroma_subsampling {
                YuvChromaSubsample::Yuv420 | YuvChromaSubsample::Yuv422 => {
                    if x + 1 < width as usize {
                        r1 = src[channels + src_chans.get_r_channel_offset()] as i32;
                        g1 = src[channels + src_chans.get_g_channel_offset()] as i32;
                        b1 = src[channels + src_chans.get_b_channel_offset()] as i32;
                        let y_1 =
                            (r1 * transform.yr + g1 * transform.yg + b1 * transform.yb + bias_y)
                                >> PRECISION;
                        unsafe {
                            *y_plane.get_unchecked_mut(x + 1) =
                                transform_integer::<ENDIANNESS, BYTES_POSITION>(y_1, bit_depth);
                        }
                    }
                }
                _ => {}
            }

            if compute_uv_row {
                let r = if chroma_subsampling == YuvChromaSubsample::Yuv444 {
                    r0
                } else {
                    (r0 + r1 + 1) >> 1
                };
                let g = if chroma_subsampling == YuvChromaSubsample::Yuv444 {
                    g0
                } else {
                    (g0 + g1 + 1) >> 1
                };
                let b = if chroma_subsampling == YuvChromaSubsample::Yuv444 {
                    b0
                } else {
                    (b0 + b1 + 1) >> 1
                };
                let cb = (r * transform.cb_r + g * transform.cb_g + b * transform.cb_b + bias_uv)
                    >> PRECISION;
                let cr = (r * transform.cr_r + g * transform.cr_g + b * transform.cr_b + bias_uv)
                    >> PRECISION;
                let u_pos = match chroma_subsampling {
                    YuvChromaSubsample::Yuv420 | YuvChromaSubsample::Yuv422 => _ux,
                    YuvChromaSubsample::Yuv444 => _ux,
                };
                unsafe {
                    *u_plane.get_unchecked_mut(u_pos) =
                        transform_integer::<ENDIANNESS, BYTES_POSITION>(cb, bit_depth);
                }
                let v_pos = match chroma_subsampling {
                    YuvChromaSubsample::Yuv420 | YuvChromaSubsample::Yuv422 => _ux,
                    YuvChromaSubsample::Yuv444 => _ux,
                };
                unsafe {
                    *v_plane.get_unchecked_mut(v_pos) =
                        transform_integer::<ENDIANNESS, BYTES_POSITION>(cr, bit_depth);
                }
            }

            _ux += 1;
        }
    }

    Ok(())
}

fn rgbx_to_yuv<
    const ORIGIN_CHANNELS: u8,
    const SAMPLING: u8,
    const ENDIANNESS: u8,
    const BYTES_POSITION: u8,
>(
    planar_image: &mut YuvPlanarImageMut<u16>,
    rgba: &[u16],
    rgba_stride: u32,
    bit_depth: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) -> Result<(), YuvError> {
    rgbx_to_yuv_impl::<ORIGIN_CHANNELS, SAMPLING, ENDIANNESS, BYTES_POSITION>(
        planar_image,
        rgba,
        rgba_stride,
        range,
        matrix,
        bit_depth,
    )
}

/// Convert RGB image data to YUV 422 planar format with 10 or 12 bit depth.
///
/// This function performs RGB to YUV conversion and stores the result in YUV422 planar format,
/// with separate planes for Y (luminance), U (chrominance), and V (chrominance) components.
///
/// # Arguments
///
/// * `planar_image` - Target planar image.
/// * `rgb` - The input RGB image data slice.
/// * `rgb_stride` - The stride (components per row) for the RGB image data.
/// * `bit_depth` - Only 10 or 12 bit-depth is supported
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
/// * `endianness` - The endianness of final YUV
/// * `bytes_packing` - position of significant bytes for YUV ( most significant or least significant ) if it in most significant it should be stated as per Apple *kCVPixelFormatType_422YpCbCr10BiPlanarFullRange/kCVPixelFormatType_422YpCbCr10BiPlanarVideoRange*
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input RGB data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn rgb_to_yuv422_p16(
    planar_image: &mut YuvPlanarImageMut<u16>,
    rgb: &[u16],
    rgb_stride: u32,
    bit_depth: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    endianness: YuvEndianness,
    bytes_packing: YuvBytesPacking,
) -> Result<(), YuvError> {
    assert!(
        bit_depth == 10 || bit_depth == 12,
        "Only 10 and 12 bit depth is supported"
    );
    let dispatcher = match endianness {
        YuvEndianness::BigEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                rgbx_to_yuv::<
                    { YuvSourceChannels::Rgb as u8 },
                    { YuvChromaSubsample::Yuv422 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                rgbx_to_yuv::<
                    { YuvSourceChannels::Rgb as u8 },
                    { YuvChromaSubsample::Yuv422 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
        YuvEndianness::LittleEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                rgbx_to_yuv::<
                    { YuvSourceChannels::Rgb as u8 },
                    { YuvChromaSubsample::Yuv422 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                rgbx_to_yuv::<
                    { YuvSourceChannels::Rgb as u8 },
                    { YuvChromaSubsample::Yuv422 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
    };
    dispatcher(planar_image, rgb, rgb_stride, bit_depth, range, matrix)
}

/// Convert BGR image data to YUV 422 planar format with 10 or 12 bit depth.
///
/// This function performs BGR to YUV conversion and stores the result in YUV422 planar format,
/// with separate planes for Y (luminance), U (chrominance), and V (chrominance) components.
///
/// # Arguments
///
/// * `planar_image` - Target planar image.
/// * `bgr` - The input BGR image data slice.
/// * `bgr_stride` - The stride (components per row) for the BGR image data.
/// * `bit_depth` - Only 10 or 12 bit-depth is supported.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
/// * `endianness` - The endianness of final YUV
/// * `bytes_packing` - position of significant bytes for YUV ( most significant or least significant ) if it in most significant it should be stated as per Apple *kCVPixelFormatType_422YpCbCr10BiPlanarFullRange/kCVPixelFormatType_422YpCbCr10BiPlanarVideoRange*
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGR data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn bgr_to_yuv422_p16(
    planar_image: &mut YuvPlanarImageMut<u16>,
    bgr: &[u16],
    bgr_stride: u32,
    bit_depth: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    endianness: YuvEndianness,
    bytes_packing: YuvBytesPacking,
) -> Result<(), YuvError> {
    assert!(
        bit_depth == 10 || bit_depth == 12,
        "Only 10 and 12 bit depth is supported"
    );
    let dispatcher = match endianness {
        YuvEndianness::BigEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                rgbx_to_yuv::<
                    { YuvSourceChannels::Bgr as u8 },
                    { YuvChromaSubsample::Yuv422 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                rgbx_to_yuv::<
                    { YuvSourceChannels::Bgr as u8 },
                    { YuvChromaSubsample::Yuv422 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
        YuvEndianness::LittleEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                rgbx_to_yuv::<
                    { YuvSourceChannels::Bgr as u8 },
                    { YuvChromaSubsample::Yuv422 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                rgbx_to_yuv::<
                    { YuvSourceChannels::Bgr as u8 },
                    { YuvChromaSubsample::Yuv422 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
    };
    dispatcher(planar_image, bgr, bgr_stride, bit_depth, range, matrix)
}

/// Convert RGBA image data to YUV 422 planar format with 10 or 12 bit depth.
///
/// This function performs RGBA to YUV conversion and stores the result in YUV422 planar format,
/// with separate planes for Y (luminance), U (chrominance), and V (chrominance) components.
///
/// # Arguments
///
/// * `planar_image` - Target planar image.
/// * `rgba` - The input RGBA image data slice.
/// * `rgba_stride` - The stride (components per row) for the RGBA image data.
/// * `bit_depth` - Only 10 or 12 bit-depth is supported.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
/// * `endianness` - The endianness of final YUV
/// * `bytes_packing` - position of significant bytes for YUV ( most significant or least significant ) if it in most significant it should be stated as per Apple *kCVPixelFormatType_422YpCbCr10BiPlanarFullRange/kCVPixelFormatType_422YpCbCr10BiPlanarVideoRange*
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input RGBA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn rgba_to_yuv422_p16(
    planar_image: &mut YuvPlanarImageMut<u16>,
    rgba: &[u16],
    rgba_stride: u32,
    bit_depth: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    endianness: YuvEndianness,
    bytes_packing: YuvBytesPacking,
) -> Result<(), YuvError> {
    assert!(
        bit_depth == 10 || bit_depth == 12,
        "Only 10 and 12 bit depth is supported"
    );
    let dispatcher = match endianness {
        YuvEndianness::BigEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                rgbx_to_yuv::<
                    { YuvSourceChannels::Rgba as u8 },
                    { YuvChromaSubsample::Yuv422 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                rgbx_to_yuv::<
                    { YuvSourceChannels::Rgba as u8 },
                    { YuvChromaSubsample::Yuv422 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
        YuvEndianness::LittleEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                rgbx_to_yuv::<
                    { YuvSourceChannels::Rgba as u8 },
                    { YuvChromaSubsample::Yuv422 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                rgbx_to_yuv::<
                    { YuvSourceChannels::Rgba as u8 },
                    { YuvChromaSubsample::Yuv422 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
    };
    dispatcher(planar_image, rgba, rgba_stride, bit_depth, range, matrix)
}

/// Convert BGRA image data to YUV 422 planar format with 10 or 12 bit depth.
///
/// This function performs BGRA to YUV conversion and stores the result in YUV422 planar format,
/// with separate planes for Y (luminance), U (chrominance), and V (chrominance) components.
///
/// # Arguments
///
/// * `planar_image` - Target planar image.
/// * `bgra` - The input BGRA image data slice.
/// * `bgra_stride` - The stride (components per row) for the BGRA image data.
/// * `bit_depth` - Only 10 or 12 bit-depth is supported.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
/// * `endianness` - The endianness of final YUV
/// * `bytes_packing` - position of significant bytes for YUV ( most significant or least significant ) if it in most significant it should be stated as per Apple *kCVPixelFormatType_422YpCbCr10BiPlanarFullRange/kCVPixelFormatType_422YpCbCr10BiPlanarVideoRange*
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGRA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn bgra_to_yuv422_p16(
    planar_image: &mut YuvPlanarImageMut<u16>,
    bgra: &[u16],
    bgra_stride: u32,
    bit_depth: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    endianness: YuvEndianness,
    bytes_packing: YuvBytesPacking,
) -> Result<(), YuvError> {
    assert!(
        bit_depth == 10 || bit_depth == 12,
        "Only 10 and 12 bit depth is supported"
    );
    let dispatcher = match endianness {
        YuvEndianness::BigEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                rgbx_to_yuv::<
                    { YuvSourceChannels::Bgra as u8 },
                    { YuvChromaSubsample::Yuv422 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                rgbx_to_yuv::<
                    { YuvSourceChannels::Bgra as u8 },
                    { YuvChromaSubsample::Yuv422 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
        YuvEndianness::LittleEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                rgbx_to_yuv::<
                    { YuvSourceChannels::Bgra as u8 },
                    { YuvChromaSubsample::Yuv422 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                rgbx_to_yuv::<
                    { YuvSourceChannels::Bgra as u8 },
                    { YuvChromaSubsample::Yuv422 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
    };
    dispatcher(planar_image, bgra, bgra_stride, bit_depth, range, matrix)
}

/// Convert RGB image data to YUV 420 planar format with 10 or 12 bit depth.
///
/// This function performs RGB to YUV conversion and stores the result in YUV420 planar format,
/// with separate planes for Y (luminance), U (chrominance), and V (chrominance) components.
///
/// # Arguments
///
/// * `planar_image` - Target planar image.
/// * `rgb` - The input RGB image data slice.
/// * `rgb_stride` - The stride (components per row) for the RGB image data.
/// * `bit_depth` - Only 10 or 12 bit-depth is supported.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
/// * `endianness` - The endianness of final YUV
/// * `bytes_packing` - position of significant bytes for YUV ( most significant or least significant ) if it in most significant it should be stated as per Apple *kCVPixelFormatType_422YpCbCr10BiPlanarFullRange/kCVPixelFormatType_422YpCbCr10BiPlanarVideoRange*
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input RGB data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn rgb_to_yuv420_p16(
    planar_image: &mut YuvPlanarImageMut<u16>,
    rgb: &[u16],
    rgb_stride: u32,
    bit_depth: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    endianness: YuvEndianness,
    bytes_packing: YuvBytesPacking,
) -> Result<(), YuvError> {
    assert!(
        bit_depth == 10 || bit_depth == 12,
        "Only 10 and 12 bit depth is supported"
    );
    let dispatcher = match endianness {
        YuvEndianness::BigEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                rgbx_to_yuv::<
                    { YuvSourceChannels::Rgb as u8 },
                    { YuvChromaSubsample::Yuv420 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                rgbx_to_yuv::<
                    { YuvSourceChannels::Rgb as u8 },
                    { YuvChromaSubsample::Yuv420 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
        YuvEndianness::LittleEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                rgbx_to_yuv::<
                    { YuvSourceChannels::Rgb as u8 },
                    { YuvChromaSubsample::Yuv420 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                rgbx_to_yuv::<
                    { YuvSourceChannels::Rgb as u8 },
                    { YuvChromaSubsample::Yuv420 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
    };
    dispatcher(planar_image, rgb, rgb_stride, bit_depth, range, matrix)
}

/// Convert BGR image data to YUV 420 planar format with 10 or 12 bit depth.
///
/// This function performs BGR to YUV conversion and stores the result in YUV420 planar format,
/// with separate planes for Y (luminance), U (chrominance), and V (chrominance) components.
///
/// # Arguments
///
/// * `planar_image` - Target planar image.
/// * `bgr` - The input BGR image data slice.
/// * `bgr_stride` - The stride (components per row) for the BGR image data.
/// * `bit_depth` - Only 10 or 12 bit-depth is supported.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
/// * `endianness` - The endianness of final YUV
/// * `bytes_packing` - position of significant bytes for YUV ( most significant or least significant ) if it in most significant it should be stated as per Apple *kCVPixelFormatType_422YpCbCr10BiPlanarFullRange/kCVPixelFormatType_422YpCbCr10BiPlanarVideoRange*
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input RGB data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn bgr_to_yuv420_p16(
    planar_image: &mut YuvPlanarImageMut<u16>,
    bgr: &[u16],
    bgr_stride: u32,
    bit_depth: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    endianness: YuvEndianness,
    bytes_packing: YuvBytesPacking,
) -> Result<(), YuvError> {
    assert!(
        bit_depth == 10 || bit_depth == 12,
        "Only 10 and 12 bit depth is supported"
    );
    let dispatcher = match endianness {
        YuvEndianness::BigEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                rgbx_to_yuv::<
                    { YuvSourceChannels::Bgr as u8 },
                    { YuvChromaSubsample::Yuv420 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                rgbx_to_yuv::<
                    { YuvSourceChannels::Bgr as u8 },
                    { YuvChromaSubsample::Yuv420 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
        YuvEndianness::LittleEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                rgbx_to_yuv::<
                    { YuvSourceChannels::Bgr as u8 },
                    { YuvChromaSubsample::Yuv420 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                rgbx_to_yuv::<
                    { YuvSourceChannels::Bgr as u8 },
                    { YuvChromaSubsample::Yuv420 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
    };
    dispatcher(planar_image, bgr, bgr_stride, bit_depth, range, matrix)
}

/// Convert RGBA image data to YUV 420 planar format with 10 or 12 bit depth.
///
/// This function performs RGBA to YUV conversion and stores the result in YUV420 planar format,
/// with separate planes for Y (luminance), U (chrominance), and V (chrominance) components.
///
/// # Arguments
///
/// * `planar_image` - Target planar image.
/// * `rgba` - The input RGBA image data slice.
/// * `rgba_stride` - The stride (components per row) for the RGBA image data.
/// * `bit_depth` - Only 10 or 12 bit-depth is supported.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
/// * `endianness` - The endianness of final YUV
/// * `bytes_packing` - position of significant bytes for YUV ( most significant or least significant ) if it in most significant it should be stated as per Apple *kCVPixelFormatType_422YpCbCr10BiPlanarFullRange/kCVPixelFormatType_422YpCbCr10BiPlanarVideoRange*
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input RGBA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn rgba_to_yuv420_p16(
    planar_image: &mut YuvPlanarImageMut<u16>,
    rgba: &[u16],
    rgba_stride: u32,
    bit_depth: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    endianness: YuvEndianness,
    bytes_packing: YuvBytesPacking,
) -> Result<(), YuvError> {
    assert!(
        bit_depth == 10 || bit_depth == 12,
        "Only 10 and 12 bit depth is supported"
    );
    let dispatcher = match endianness {
        YuvEndianness::BigEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                rgbx_to_yuv::<
                    { YuvSourceChannels::Rgba as u8 },
                    { YuvChromaSubsample::Yuv420 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                rgbx_to_yuv::<
                    { YuvSourceChannels::Rgba as u8 },
                    { YuvChromaSubsample::Yuv420 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
        YuvEndianness::LittleEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                rgbx_to_yuv::<
                    { YuvSourceChannels::Rgba as u8 },
                    { YuvChromaSubsample::Yuv420 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                rgbx_to_yuv::<
                    { YuvSourceChannels::Rgba as u8 },
                    { YuvChromaSubsample::Yuv420 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
    };
    dispatcher(planar_image, rgba, rgba_stride, bit_depth, range, matrix)
}

/// Convert BGRA image data to YUV 420 planar format with 10 or 12 bit depth.
///
/// This function performs BGRA to YUV conversion and stores the result in YUV420 planar format,
/// with separate planes for Y (luminance), U (chrominance), and V (chrominance) components.
///
/// # Arguments
///
/// * `planar_image` - Target planar image.
/// * `bgra` - The input BGRA image data slice.
/// * `bgra_stride` - The stride (components per row) for the BGRA image data.
/// * `bit_depth` - Only 10 or 12 bit-depth is supported.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
/// * `endianness` - The endianness of final YUV
/// * `bytes_packing` - position of significant bytes for YUV ( most significant or least significant ) if it in most significant it should be stated as per Apple *kCVPixelFormatType_422YpCbCr10BiPlanarFullRange/kCVPixelFormatType_422YpCbCr10BiPlanarVideoRange*
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGRA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn bgra_to_yuv420_p16(
    planar_image: &mut YuvPlanarImageMut<u16>,
    bgra: &[u16],
    bgra_stride: u32,
    bit_depth: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    endianness: YuvEndianness,
    bytes_packing: YuvBytesPacking,
) -> Result<(), YuvError> {
    assert!(
        bit_depth == 10 || bit_depth == 12,
        "Only 10 and 12 bit depth is supported"
    );
    let dispatcher = match endianness {
        YuvEndianness::BigEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                rgbx_to_yuv::<
                    { YuvSourceChannels::Bgra as u8 },
                    { YuvChromaSubsample::Yuv420 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                rgbx_to_yuv::<
                    { YuvSourceChannels::Bgra as u8 },
                    { YuvChromaSubsample::Yuv420 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
        YuvEndianness::LittleEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                rgbx_to_yuv::<
                    { YuvSourceChannels::Bgra as u8 },
                    { YuvChromaSubsample::Yuv420 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                rgbx_to_yuv::<
                    { YuvSourceChannels::Bgra as u8 },
                    { YuvChromaSubsample::Yuv420 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
    };
    dispatcher(planar_image, bgra, bgra_stride, bit_depth, range, matrix)
}

/// Convert RGB image data to YUV 444 planar format with 10 or 12 bit depth.
///
/// This function performs RGB to YUV conversion and stores the result in YUV444 planar format,
/// with separate planes for Y (luminance), U (chrominance), and V (chrominance) components.
///
/// # Arguments
///
/// * `planar_image` - Target planar image.
/// * `rgb` - The input RGB image data slice.
/// * `rgb_stride` - The stride (components per row) for the RGB image data.
/// * `bit_depth` - Only 10 or 12 bit-depth is supported.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
/// * `endianness` - The endianness of final YUV
/// * `bytes_packing` - position of significant bytes for YUV ( most significant or least significant ) if it in most significant it should be stated as per Apple *kCVPixelFormatType_422YpCbCr10BiPlanarFullRange/kCVPixelFormatType_422YpCbCr10BiPlanarVideoRange*
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input RGB data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn rgb_to_yuv444_p16(
    planar_image: &mut YuvPlanarImageMut<u16>,
    rgb: &[u16],
    rgb_stride: u32,
    bit_depth: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    endianness: YuvEndianness,
    bytes_packing: YuvBytesPacking,
) -> Result<(), YuvError> {
    assert!(
        bit_depth == 10 || bit_depth == 12,
        "Only 10 and 12 bit depth is supported"
    );
    let dispatcher = match endianness {
        YuvEndianness::BigEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                rgbx_to_yuv::<
                    { YuvSourceChannels::Rgb as u8 },
                    { YuvChromaSubsample::Yuv444 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                rgbx_to_yuv::<
                    { YuvSourceChannels::Rgb as u8 },
                    { YuvChromaSubsample::Yuv444 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
        YuvEndianness::LittleEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                rgbx_to_yuv::<
                    { YuvSourceChannels::Rgb as u8 },
                    { YuvChromaSubsample::Yuv444 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                rgbx_to_yuv::<
                    { YuvSourceChannels::Rgb as u8 },
                    { YuvChromaSubsample::Yuv444 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
    };
    dispatcher(planar_image, rgb, rgb_stride, bit_depth, range, matrix)
}

/// Convert BGR image data to YUV 444 planar format with 10 or 12 bit depth.
///
/// This function performs BGR to YUV conversion and stores the result in YUV444 planar format,
/// with separate planes for Y (luminance), U (chrominance), and V (chrominance) components.
///
/// # Arguments
///
/// * `planar_image` - Target planar image.
/// * `bgr` - The input BGR image data slice.
/// * `bgr_stride` - The stride (components per row) for the BGR image data.
/// * `bit_depth` - Only 10 or 12 bit-depth is supported.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
/// * `endianness` - The endianness of final YUV
/// * `bytes_packing` - position of significant bytes for YUV ( most significant or least significant ) if it in most significant it should be stated as per Apple *kCVPixelFormatType_422YpCbCr10BiPlanarFullRange/kCVPixelFormatType_422YpCbCr10BiPlanarVideoRange*
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGR data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn bgr_to_yuv444_p16(
    planar_image: &mut YuvPlanarImageMut<u16>,
    bgr: &[u16],
    bgr_stride: u32,
    bit_depth: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    endianness: YuvEndianness,
    bytes_packing: YuvBytesPacking,
) -> Result<(), YuvError> {
    assert!(
        bit_depth == 10 || bit_depth == 12,
        "Only 10 and 12 bit depth is supported"
    );
    let dispatcher = match endianness {
        YuvEndianness::BigEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                rgbx_to_yuv::<
                    { YuvSourceChannels::Bgr as u8 },
                    { YuvChromaSubsample::Yuv444 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                rgbx_to_yuv::<
                    { YuvSourceChannels::Bgr as u8 },
                    { YuvChromaSubsample::Yuv444 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
        YuvEndianness::LittleEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                rgbx_to_yuv::<
                    { YuvSourceChannels::Bgr as u8 },
                    { YuvChromaSubsample::Yuv444 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                rgbx_to_yuv::<
                    { YuvSourceChannels::Bgr as u8 },
                    { YuvChromaSubsample::Yuv444 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
    };
    dispatcher(planar_image, bgr, bgr_stride, bit_depth, range, matrix)
}

/// Convert RGBA image data to YUV 444 planar format with 10 or 12 bit depth.
///
/// This function performs RGBA to YUV conversion and stores the result in YUV444 planar format,
/// with separate planes for Y (luminance), U (chrominance), and V (chrominance) components.
///
/// # Arguments
///
/// * `planar_image` - Target planar image.
/// * `rgba` - The input RGBA image data slice.
/// * `rgba_stride` - The stride (components per row) for the RGBA image data.
/// * `bit_depth` - Only 10 or 12 bit-depth is supported.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
/// * `endianness` - The endianness of final YUV
/// * `bytes_packing` - position of significant bytes for YUV ( most significant or least significant ) if it in most significant it should be stated as per Apple *kCVPixelFormatType_422YpCbCr10BiPlanarFullRange/kCVPixelFormatType_422YpCbCr10BiPlanarVideoRange*
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input RGBA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn rgba_to_yuv444_p16(
    planar_image: &mut YuvPlanarImageMut<u16>,
    rgba: &[u16],
    rgba_stride: u32,
    bit_depth: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    endianness: YuvEndianness,
    bytes_packing: YuvBytesPacking,
) -> Result<(), YuvError> {
    assert!(
        bit_depth == 10 || bit_depth == 12,
        "Only 10 and 12 bit depth is supported"
    );
    let dispatcher = match endianness {
        YuvEndianness::BigEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                rgbx_to_yuv::<
                    { YuvSourceChannels::Rgba as u8 },
                    { YuvChromaSubsample::Yuv444 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                rgbx_to_yuv::<
                    { YuvSourceChannels::Rgba as u8 },
                    { YuvChromaSubsample::Yuv444 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
        YuvEndianness::LittleEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                rgbx_to_yuv::<
                    { YuvSourceChannels::Rgba as u8 },
                    { YuvChromaSubsample::Yuv444 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                rgbx_to_yuv::<
                    { YuvSourceChannels::Rgba as u8 },
                    { YuvChromaSubsample::Yuv444 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
    };
    dispatcher(planar_image, rgba, rgba_stride, bit_depth, range, matrix)
}

/// Convert BGRA image data to YUV 444 planar format with 10 or 12 bit depth.
///
/// This function performs BGRA to YUV conversion and stores the result in YUV444 planar format,
/// with separate planes for Y (luminance), U (chrominance), and V (chrominance) components.
///
/// # Arguments
///
/// * `planar_image` - Target planar image.
/// * `bgra` - The input BGRA image data slice.
/// * `bgra_stride` - The stride (components per row) for the BGRA image data.
/// * `bit_depth` - Only 10 or 12 bit-depth is supported.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
/// * `endianness` - The endianness of final YUV
/// * `bytes_packing` - position of significant bytes for YUV ( most significant or least significant ) if it in most significant it should be stated as per Apple *kCVPixelFormatType_422YpCbCr10BiPlanarFullRange/kCVPixelFormatType_422YpCbCr10BiPlanarVideoRange*
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGRA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn bgra_to_yuv444_p16(
    planar_image: &mut YuvPlanarImageMut<u16>,
    bgra: &[u16],
    bgra_stride: u32,
    bit_depth: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    endianness: YuvEndianness,
    bytes_packing: YuvBytesPacking,
) -> Result<(), YuvError> {
    assert!(
        bit_depth == 10 || bit_depth == 12,
        "Only 10 and 12 bit depth is supported"
    );
    let dispatcher = match endianness {
        YuvEndianness::BigEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                rgbx_to_yuv::<
                    { YuvSourceChannels::Bgra as u8 },
                    { YuvChromaSubsample::Yuv444 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                rgbx_to_yuv::<
                    { YuvSourceChannels::Bgra as u8 },
                    { YuvChromaSubsample::Yuv444 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
        YuvEndianness::LittleEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                rgbx_to_yuv::<
                    { YuvSourceChannels::Bgra as u8 },
                    { YuvChromaSubsample::Yuv444 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                rgbx_to_yuv::<
                    { YuvSourceChannels::Bgra as u8 },
                    { YuvChromaSubsample::Yuv444 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
    };
    dispatcher(planar_image, bgra, bgra_stride, bit_depth, range, matrix)
}
