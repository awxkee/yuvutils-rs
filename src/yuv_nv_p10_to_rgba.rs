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
use crate::internals::{ProcessedOffset, RowDBiPlanarInversionHandler};
use crate::numerics::{qrshr, to_ne};
use crate::yuv_support::*;
use crate::{YuvBiPlanarImage, YuvError};
#[cfg(feature = "rayon")]
use rayon::iter::{IndexedParallelIterator, ParallelIterator};
#[cfg(feature = "rayon")]
use rayon::prelude::{ParallelSlice, ParallelSliceMut};

type RowHandlerFn = unsafe fn(
    y_plane: &[u16],
    uv_plane: &[u16],
    bgra: &mut [u8],
    width: u32,
    range: &YuvChromaRange,
    transform: &CbCrInverseTransform<i32>,
    start_cx: usize,
    start_ux: usize,
) -> ProcessedOffset;

struct RowHandlerBalanced<
    const DESTINATION_CHANNELS: u8,
    const NV_ORDER: u8,
    const SAMPLING: u8,
    const ENDIANNESS: u8,
    const BYTES_POSITION: u8,
    const PRECISION: i32,
    const BIT_DEPTH: usize,
> {
    handler: Option<RowHandlerFn>,
}

#[cfg(feature = "professional_mode")]
struct RowHandlerProfessional<
    const DESTINATION_CHANNELS: u8,
    const NV_ORDER: u8,
    const SAMPLING: u8,
    const ENDIANNESS: u8,
    const BYTES_POSITION: u8,
    const PRECISION: i32,
    const BIT_DEPTH: usize,
> {
    handler: Option<RowHandlerFn>,
}

impl<
        const DESTINATION_CHANNELS: u8,
        const NV_ORDER: u8,
        const SAMPLING: u8,
        const ENDIANNESS: u8,
        const BYTES_POSITION: u8,
        const PRECISION: i32,
        const BIT_DEPTH: usize,
    > Default
    for RowHandlerBalanced<
        DESTINATION_CHANNELS,
        NV_ORDER,
        SAMPLING,
        ENDIANNESS,
        BYTES_POSITION,
        PRECISION,
        BIT_DEPTH,
    >
{
    fn default() -> Self {
        if PRECISION == 13 {
            assert_eq!(PRECISION, 13);
            #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
            {
                use crate::neon::neon_yuv_nv12_p10_to_rgba_row;
                return Self {
                    handler: Some(
                        neon_yuv_nv12_p10_to_rgba_row::<
                            DESTINATION_CHANNELS,
                            NV_ORDER,
                            SAMPLING,
                            ENDIANNESS,
                            BYTES_POSITION,
                            PRECISION,
                            BIT_DEPTH,
                        >,
                    ),
                };
            }
        }
        Self { handler: None }
    }
}

#[cfg(feature = "professional_mode")]
impl<
        const DESTINATION_CHANNELS: u8,
        const NV_ORDER: u8,
        const SAMPLING: u8,
        const ENDIANNESS: u8,
        const BYTES_POSITION: u8,
        const PRECISION: i32,
        const BIT_DEPTH: usize,
    > Default
    for RowHandlerProfessional<
        DESTINATION_CHANNELS,
        NV_ORDER,
        SAMPLING,
        ENDIANNESS,
        BYTES_POSITION,
        PRECISION,
        BIT_DEPTH,
    >
{
    fn default() -> Self {
        if PRECISION == 14 {
            assert_eq!(PRECISION, 14);
            #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
            {
                use crate::neon::neon_yuv_nv12_p10_to_rgba_row_prof;
                return Self {
                    handler: Some(
                        neon_yuv_nv12_p10_to_rgba_row_prof::<
                            DESTINATION_CHANNELS,
                            NV_ORDER,
                            SAMPLING,
                            ENDIANNESS,
                            BYTES_POSITION,
                            BIT_DEPTH,
                        >,
                    ),
                };
            }
        }
        Self { handler: None }
    }
}

macro_rules! impl_row_handler {
    ($struct_name:ident) => {
        impl<
                const DESTINATION_CHANNELS: u8,
                const NV_ORDER: u8,
                const SAMPLING: u8,
                const ENDIANNESS: u8,
                const BYTES_POSITION: u8,
                const PRECISION: i32,
                const BIT_DEPTH: usize,
            > RowDBiPlanarInversionHandler<u16, u8, i32>
            for $struct_name<
                DESTINATION_CHANNELS,
                NV_ORDER,
                SAMPLING,
                ENDIANNESS,
                BYTES_POSITION,
                PRECISION,
                BIT_DEPTH,
            >
        {
            fn handle_row(
                &self,
                y_plane: &[u16],
                uv_plane: &[u16],
                rgba: &mut [u8],
                width: u32,
                chroma: YuvChromaRange,
                transform: &CbCrInverseTransform<i32>,
            ) -> ProcessedOffset {
                if let Some(handler) = self.handler {
                    unsafe {
                        return handler(y_plane, uv_plane, rgba, width, &chroma, transform, 0, 0);
                    }
                }
                ProcessedOffset { cx: 0, ux: 0 }
            }
        }
    };
}

impl_row_handler!(RowHandlerBalanced);
#[cfg(feature = "professional_mode")]
impl_row_handler!(RowHandlerProfessional);

fn yuv_nv_p10_to_image_impl_d<
    const DESTINATION_CHANNELS: u8,
    const NV_ORDER: u8,
    const SAMPLING: u8,
    const ENDIANNESS: u8,
    const BYTES_POSITION: u8,
    const PRECISION: i32,
    const V_R_SHR: i32,
>(
    image: &YuvBiPlanarImage<u16>,
    bgra: &mut [u8],
    bgra_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    row_handler: impl RowDBiPlanarInversionHandler<u16, u8, i32> + Send + Sync,
) -> Result<(), YuvError> {
    let dst_chans: YuvSourceChannels = DESTINATION_CHANNELS.into();
    let channels = dst_chans.get_channels_count();
    let uv_order: YuvNVOrder = NV_ORDER.into();
    let chroma_subsampling: YuvChromaSubsampling = SAMPLING.into();

    const BIT_DEPTH: usize = 10;

    image.check_constraints(chroma_subsampling)?;

    let chroma_range = get_yuv_range(BIT_DEPTH as u32, range);
    let kr_kb = matrix.get_kr_kb();
    let i_transform = search_inverse_transform(
        PRECISION,
        BIT_DEPTH as u32,
        range,
        matrix,
        chroma_range,
        kr_kb,
    );
    let cr_coef = i_transform.cr_coef;
    let cb_coef = i_transform.cb_coef;
    let y_coef = i_transform.y_coef;
    let g_coef_1 = i_transform.g_coeff_1;
    let g_coef_2 = i_transform.g_coeff_2;

    let bias_y = chroma_range.bias_y as i32;
    let bias_uv = chroma_range.bias_uv as i32;

    let msb_shift = 16 - BIT_DEPTH as i32;
    let width = image.width;

    let process_halved_chroma_row = |y_src: &[u16], uv_src: &[u16], rgba: &mut [u8]| {
        let processed =
            row_handler.handle_row(y_src, uv_src, rgba, image.width, chroma_range, &i_transform);
        if processed.cx != image.width as usize {
            for ((rgba, y_src), uv_src) in rgba
                .chunks_exact_mut(channels * 2)
                .zip(y_src.chunks_exact(2))
                .zip(uv_src.chunks_exact(2))
                .skip(processed.cx / 2)
            {
                let y_vl0 = to_ne::<ENDIANNESS, BYTES_POSITION>(y_src[0], msb_shift) as i32;
                let mut cb_value = to_ne::<ENDIANNESS, BYTES_POSITION>(
                    uv_src[uv_order.get_u_position()],
                    msb_shift,
                ) as i32;
                let mut cr_value = to_ne::<ENDIANNESS, BYTES_POSITION>(
                    uv_src[uv_order.get_v_position()],
                    msb_shift,
                ) as i32;

                let y_value0: i32 = (y_vl0 - bias_y) * y_coef;

                cb_value -= bias_uv;
                cr_value -= bias_uv;

                let r_p0 = qrshr::<V_R_SHR, 8>(y_value0 + cr_coef * cr_value);
                let b_p0 = qrshr::<V_R_SHR, 8>(y_value0 + cb_coef * cb_value);
                let g_p0 =
                    qrshr::<V_R_SHR, 8>(y_value0 - g_coef_1 * cr_value - g_coef_2 * cb_value);

                rgba[dst_chans.get_b_channel_offset()] = b_p0 as u8;
                rgba[dst_chans.get_g_channel_offset()] = g_p0 as u8;
                rgba[dst_chans.get_r_channel_offset()] = r_p0 as u8;

                if dst_chans.has_alpha() {
                    rgba[dst_chans.get_a_channel_offset()] = 255u8;
                }

                let y_vl1 = to_ne::<ENDIANNESS, BYTES_POSITION>(y_src[1], msb_shift) as i32;

                let y_value1: i32 = (y_vl1 - bias_y) * y_coef;

                let r_p1 = qrshr::<V_R_SHR, 8>(y_value1 + cr_coef * cr_value);
                let b_p1 = qrshr::<V_R_SHR, 8>(y_value1 + cb_coef * cb_value);
                let g_p1 =
                    qrshr::<V_R_SHR, 8>(y_value1 - g_coef_1 * cr_value - g_coef_2 * cb_value);

                rgba[channels + dst_chans.get_b_channel_offset()] = b_p1 as u8;
                rgba[channels + dst_chans.get_g_channel_offset()] = g_p1 as u8;
                rgba[channels + dst_chans.get_r_channel_offset()] = r_p1 as u8;

                if dst_chans.has_alpha() {
                    rgba[channels + dst_chans.get_a_channel_offset()] = 255;
                }
            }

            if width & 1 != 0 {
                let rgba = rgba.chunks_exact_mut(channels * 2).into_remainder();
                let rgba = &mut rgba[0..channels];
                let uv_src = uv_src.chunks_exact(2).last().unwrap();
                let y_src = y_src.chunks_exact(2).remainder();

                let y_vl0 = to_ne::<ENDIANNESS, BYTES_POSITION>(y_src[0], msb_shift) as i32;
                let y_value0: i32 = (y_vl0 - bias_y) * y_coef;
                let mut cb_value = to_ne::<ENDIANNESS, BYTES_POSITION>(
                    uv_src[uv_order.get_u_position()],
                    msb_shift,
                ) as i32;
                let mut cr_value = to_ne::<ENDIANNESS, BYTES_POSITION>(
                    uv_src[uv_order.get_v_position()],
                    msb_shift,
                ) as i32;

                cb_value -= bias_uv;
                cr_value -= bias_uv;

                let r_p0 = qrshr::<V_R_SHR, 8>(y_value0 + cr_coef * cr_value);
                let b_p0 = qrshr::<V_R_SHR, 8>(y_value0 + cb_coef * cb_value);
                let g_p0 =
                    qrshr::<V_R_SHR, 8>(y_value0 - g_coef_1 * cr_value - g_coef_2 * cb_value);

                rgba[dst_chans.get_b_channel_offset()] = b_p0 as u8;
                rgba[dst_chans.get_g_channel_offset()] = g_p0 as u8;
                rgba[dst_chans.get_r_channel_offset()] = r_p0 as u8;

                if dst_chans.has_alpha() {
                    rgba[dst_chans.get_a_channel_offset()] = 255u8;
                }
            }
        }
    };

    let y_stride = image.y_stride;
    let uv_stride = image.uv_stride;
    let y_plane = image.y_plane;
    let uv_plane = image.uv_plane;

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
            let y_src = &y_src[0..image.width as usize];
            let processed = row_handler.handle_row(
                y_src,
                uv_src,
                rgba,
                image.width,
                chroma_range,
                &i_transform,
            );
            if processed.cx != image.width as usize {
                for ((rgba, &y_src), uv_src) in rgba
                    .chunks_exact_mut(channels)
                    .zip(y_src.iter())
                    .zip(uv_src.chunks_exact(2))
                    .skip(processed.cx)
                {
                    let y_vl = to_ne::<ENDIANNESS, BYTES_POSITION>(y_src, msb_shift) as i32;
                    let mut cb_value = to_ne::<ENDIANNESS, BYTES_POSITION>(
                        uv_src[uv_order.get_u_position()],
                        msb_shift,
                    ) as i32;
                    let mut cr_value = to_ne::<ENDIANNESS, BYTES_POSITION>(
                        uv_src[uv_order.get_v_position()],
                        msb_shift,
                    ) as i32;

                    let y_value: i32 = (y_vl - bias_y) * y_coef;

                    cb_value -= bias_uv;
                    cr_value -= bias_uv;

                    let r_p16 = qrshr::<V_R_SHR, 8>(y_value + cr_coef * cr_value);
                    let b_p16 = qrshr::<V_R_SHR, 8>(y_value + cb_coef * cb_value);
                    let g_p16 =
                        qrshr::<V_R_SHR, 8>(y_value - g_coef_1 * cr_value - g_coef_2 * cb_value);

                    rgba[dst_chans.get_b_channel_offset()] = b_p16 as u8;
                    rgba[dst_chans.get_g_channel_offset()] = g_p16 as u8;
                    rgba[dst_chans.get_r_channel_offset()] = r_p16 as u8;

                    if dst_chans.has_alpha() {
                        rgba[dst_chans.get_a_channel_offset()] = 255u8;
                    }
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
            process_halved_chroma_row(
                &y_src[0..image.width as usize],
                &uv_src[0..(image.width as usize).div_ceil(2) * 2],
                &mut rgba[0..image.width as usize * channels],
            );
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
                process_halved_chroma_row(
                    &y_src[0..image.width as usize],
                    &uv_src[0..(image.width as usize).div_ceil(2) * 2],
                    &mut rgba[0..image.width as usize * channels],
                );
            }
        });
        if image.height & 1 != 0 {
            let y_src = y_plane.chunks_exact(y_stride as usize * 2).remainder();
            let uv_src = uv_plane.chunks_exact(uv_stride as usize).last().unwrap();
            let rgba = bgra
                .chunks_exact_mut(bgra_stride as usize * 2)
                .into_remainder();
            process_halved_chroma_row(
                &y_src[0..image.width as usize],
                &uv_src[0..(image.width as usize).div_ceil(2) * 2],
                &mut rgba[0..image.width as usize * channels],
            );
        }
    } else {
        unreachable!();
    }

    Ok(())
}

#[inline]
fn yuv_nv_p10_to_image_impl<
    const DESTINATION_CHANNELS: u8,
    const NV_ORDER: u8,
    const SAMPLING: u8,
    const ENDIANNESS: u8,
    const BYTES_POSITION: u8,
>(
    image: &YuvBiPlanarImage<u16>,
    bgra: &mut [u8],
    bgra_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    mode: YuvConversionMode,
) -> Result<(), YuvError> {
    match mode {
        #[cfg(feature = "fast_mode")]
        YuvConversionMode::Fast => yuv_nv_p10_to_image_impl_d::<
            DESTINATION_CHANNELS,
            NV_ORDER,
            SAMPLING,
            ENDIANNESS,
            BYTES_POSITION,
            13,
            15,
        >(
            image,
            bgra,
            bgra_stride,
            range,
            matrix,
            RowHandlerBalanced::<
                DESTINATION_CHANNELS,
                NV_ORDER,
                SAMPLING,
                ENDIANNESS,
                BYTES_POSITION,
                13,
                10,
            >::default(),
        ),
        YuvConversionMode::Balanced => yuv_nv_p10_to_image_impl_d::<
            DESTINATION_CHANNELS,
            NV_ORDER,
            SAMPLING,
            ENDIANNESS,
            BYTES_POSITION,
            13,
            15,
        >(
            image,
            bgra,
            bgra_stride,
            range,
            matrix,
            RowHandlerBalanced::<
                DESTINATION_CHANNELS,
                NV_ORDER,
                SAMPLING,
                ENDIANNESS,
                BYTES_POSITION,
                13,
                10,
            >::default(),
        ),
        #[cfg(feature = "professional_mode")]
        YuvConversionMode::Professional => yuv_nv_p10_to_image_impl_d::<
            DESTINATION_CHANNELS,
            NV_ORDER,
            SAMPLING,
            ENDIANNESS,
            BYTES_POSITION,
            14,
            16,
        >(
            image,
            bgra,
            bgra_stride,
            range,
            matrix,
            RowHandlerProfessional::<
                DESTINATION_CHANNELS,
                NV_ORDER,
                SAMPLING,
                ENDIANNESS,
                BYTES_POSITION,
                14,
                10,
            >::default(),
        ),
    }
}

/// Convert YUV NV12 format with 10-bit pixel format to BGRA format.
///
/// This function takes YUV NV12 data with 10-bit precision
/// and converts it to BGRA format with 8-bit precision.
///
/// # Arguments
///
/// * `bi_planar_image` - Source Bi-Planar 10-bit image.
/// * `bgra` - A mutable slice to store the converted BGRA data.
/// * `bgra_stride` - The stride (components per row) for the BGRA image data.
/// * `endianness` - The endianness of stored bytes
/// * `bytes_packing` - see [YuvBytesPacking] for more info.
/// * `mode` - See [YuvConversionMode] for more info.
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGRA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv_nv12_p10_to_bgra(
    bi_planar_image: &YuvBiPlanarImage<u16>,
    bgra: &mut [u8],
    bgra_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    endianness: YuvEndianness,
    bytes_packing: YuvBytesPacking,
    mode: YuvConversionMode,
) -> Result<(), YuvError> {
    let dispatcher = match endianness {
        #[cfg(feature = "big_endian")]
        YuvEndianness::BigEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Bgra as u8 },
                    { YuvNVOrder::UV as u8 },
                    { YuvChromaSubsampling::Yuv420 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Bgra as u8 },
                    { YuvNVOrder::UV as u8 },
                    { YuvChromaSubsampling::Yuv420 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
        YuvEndianness::LittleEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Bgra as u8 },
                    { YuvNVOrder::UV as u8 },
                    { YuvChromaSubsampling::Yuv420 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Bgra as u8 },
                    { YuvNVOrder::UV as u8 },
                    { YuvChromaSubsampling::Yuv420 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
    };
    dispatcher(bi_planar_image, bgra, bgra_stride, range, matrix, mode)
}

/// Convert YUV NV12 format with 10-bit pixel format to RGBA format.
///
/// This function takes YUV NV12 data with 10-bit precision
/// and converts it to RGBA format with 8-bit precision.
///
/// # Arguments
///
/// * `bi_planar_image` - Source Bi-Planar 10-bit image.
/// * `rgba` - A mutable slice to store the converted RGBA data.
/// * `rgba_stride` - The stride (components per row) for the RGBA image data.
/// * `endianness` - The endianness of stored bytes
/// * `bytes_packing` - see [YuvBytesPacking] for more info.
/// * `mode` - See [YuvConversionMode] for more info.
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input RGBA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv_nv12_p10_to_rgba(
    bi_planar_image: &YuvBiPlanarImage<u16>,
    rgba: &mut [u8],
    rgba_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    endianness: YuvEndianness,
    bytes_packing: YuvBytesPacking,
    mode: YuvConversionMode,
) -> Result<(), YuvError> {
    let dispatcher = match endianness {
        #[cfg(feature = "big_endian")]
        YuvEndianness::BigEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Rgba as u8 },
                    { YuvNVOrder::UV as u8 },
                    { YuvChromaSubsampling::Yuv420 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Rgba as u8 },
                    { YuvNVOrder::UV as u8 },
                    { YuvChromaSubsampling::Yuv420 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
        },
        YuvEndianness::LittleEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Rgba as u8 },
                    { YuvNVOrder::UV as u8 },
                    { YuvChromaSubsampling::Yuv420 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Rgba as u8 },
                    { YuvNVOrder::UV as u8 },
                    { YuvChromaSubsampling::Yuv420 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
    };
    dispatcher(bi_planar_image, rgba, rgba_stride, range, matrix, mode)
}

/// Convert YUV NV12 format with 10-bit pixel format to BGR format.
///
/// This function takes YUV NV12 data with 10-bit precision
/// and converts it to BGR format with 8-bit precision.
///
/// # Arguments
///
/// * `bi_planar_image` - Source Bi-Planar 10-bit image.
/// * `bgr` - A mutable slice to store the converted BGR data.
/// * `bgr_stride` - The stride (components per row) for the BGR image data.
/// * `endianness` - The endianness of stored bytes
/// * `bytes_packing` - see [YuvBytesPacking] for more info.
/// * `mode` - See [YuvConversionMode] for more info.
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGR data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv_nv12_p10_to_bgr(
    bi_planar_image: &YuvBiPlanarImage<u16>,
    bgr: &mut [u8],
    bgr_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    endianness: YuvEndianness,
    bytes_packing: YuvBytesPacking,
    mode: YuvConversionMode,
) -> Result<(), YuvError> {
    let dispatcher = match endianness {
        #[cfg(feature = "big_endian")]
        YuvEndianness::BigEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Bgr as u8 },
                    { YuvNVOrder::UV as u8 },
                    { YuvChromaSubsampling::Yuv420 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Bgr as u8 },
                    { YuvNVOrder::UV as u8 },
                    { YuvChromaSubsampling::Yuv420 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
        YuvEndianness::LittleEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Bgr as u8 },
                    { YuvNVOrder::UV as u8 },
                    { YuvChromaSubsampling::Yuv420 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Bgr as u8 },
                    { YuvNVOrder::UV as u8 },
                    { YuvChromaSubsampling::Yuv420 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
    };
    dispatcher(bi_planar_image, bgr, bgr_stride, range, matrix, mode)
}

/// Convert YUV NV12 format with 10-bit pixel format to RGB format.
///
/// This function takes YUV NV12 data with 10-bit precision
/// and converts it to RGB format with 8-bit precision.
///
/// # Arguments
///
/// * `bi_planar_image` - Source Bi-Planar 10-bit image.
/// * `rgb` - A mutable slice to store the converted RGB data.
/// * `rgb_stride` - The stride (components per row) for the RGB image data.
/// * `endianness` - The endianness of stored bytes
/// * `bytes_packing` - see [YuvBytesPacking] for more info.
/// * `mode` - See [YuvConversionMode] for more info.
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input RGB data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv_nv12_p10_to_rgb(
    bi_planar_image: &YuvBiPlanarImage<u16>,
    rgb: &mut [u8],
    rgb_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    endianness: YuvEndianness,
    bytes_packing: YuvBytesPacking,
    mode: YuvConversionMode,
) -> Result<(), YuvError> {
    let dispatcher = match endianness {
        #[cfg(feature = "big_endian")]
        YuvEndianness::BigEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Rgb as u8 },
                    { YuvNVOrder::UV as u8 },
                    { YuvChromaSubsampling::Yuv420 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Rgb as u8 },
                    { YuvNVOrder::UV as u8 },
                    { YuvChromaSubsampling::Yuv420 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
        YuvEndianness::LittleEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Rgb as u8 },
                    { YuvNVOrder::UV as u8 },
                    { YuvChromaSubsampling::Yuv420 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Rgb as u8 },
                    { YuvNVOrder::UV as u8 },
                    { YuvChromaSubsampling::Yuv420 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
    };
    dispatcher(bi_planar_image, rgb, rgb_stride, range, matrix, mode)
}

/// Convert YUV NV16 format with 10-bit pixel format to BGRA format.
///
/// This function takes YUV NV16 data with 10-bit precision.
/// and converts it to BGRA format with 8-bit precision.
///
/// # Arguments
///
/// * `bi_planar_image` - Source Bi-Planar 10-bit image.
/// * `bgra` - A mutable slice to store the converted BGRA data.
/// * `bgra_stride` - The stride (components per row) for the BGRA image data.
/// * `endianness` - The endianness of stored bytes
/// * `bytes_packing` - see [YuvBytesPacking] for more info.
/// * `mode` - See [YuvConversionMode] for more info.
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGRA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv_nv16_p10_to_bgra(
    bi_planar_image: &YuvBiPlanarImage<u16>,
    bgra: &mut [u8],
    bgra_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    endianness: YuvEndianness,
    bytes_packing: YuvBytesPacking,
    mode: YuvConversionMode,
) -> Result<(), YuvError> {
    let dispatcher = match endianness {
        #[cfg(feature = "big_endian")]
        YuvEndianness::BigEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Bgra as u8 },
                    { YuvNVOrder::UV as u8 },
                    { YuvChromaSubsampling::Yuv422 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Bgra as u8 },
                    { YuvNVOrder::UV as u8 },
                    { YuvChromaSubsampling::Yuv422 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
        YuvEndianness::LittleEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Bgra as u8 },
                    { YuvNVOrder::UV as u8 },
                    { YuvChromaSubsampling::Yuv422 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Bgra as u8 },
                    { YuvNVOrder::UV as u8 },
                    { YuvChromaSubsampling::Yuv422 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
    };
    dispatcher(bi_planar_image, bgra, bgra_stride, range, matrix, mode)
}

/// Convert YUV NV61 format with 10-bit pixel format to BGRA format.
///
/// This function takes YUV NV61 data with 10-bit precision.
/// and converts it to BGRA format with 8-bit precision.
///
/// # Arguments
///
/// * `bi_planar_image` - Source Bi-Planar 10-bit image.
/// * `bgra` - A mutable slice to store the converted BGRA data.
/// * `bgra_stride` - The stride (components per row) for the BGRA image data.
/// * `endianness` - The endianness of stored bytes
/// * `bytes_packing` - see [YuvBytesPacking] for more info.
/// * `mode` - See [YuvConversionMode] for more info.
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGRA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv_nv61_p10_to_bgra(
    bi_planar_image: &YuvBiPlanarImage<u16>,
    bgra: &mut [u8],
    bgra_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    endianness: YuvEndianness,
    bytes_packing: YuvBytesPacking,
    mode: YuvConversionMode,
) -> Result<(), YuvError> {
    let dispatcher = match endianness {
        #[cfg(feature = "big_endian")]
        YuvEndianness::BigEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Bgra as u8 },
                    { YuvNVOrder::VU as u8 },
                    { YuvChromaSubsampling::Yuv422 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Bgra as u8 },
                    { YuvNVOrder::VU as u8 },
                    { YuvChromaSubsampling::Yuv422 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
        YuvEndianness::LittleEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Bgra as u8 },
                    { YuvNVOrder::VU as u8 },
                    { YuvChromaSubsampling::Yuv422 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Bgra as u8 },
                    { YuvNVOrder::VU as u8 },
                    { YuvChromaSubsampling::Yuv422 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
    };
    dispatcher(bi_planar_image, bgra, bgra_stride, range, matrix, mode)
}

/// Convert YUV NV16 format with 10-bit pixel format to BGR format.
///
/// This function takes YUV NV16 data with 10-bit precision.
/// and converts it to BGR format with 8-bit precision.
///
/// # Arguments
///
/// * `bi_planar_image` - Source Bi-Planar 10-bit image.
/// * `bgra` - A mutable slice to store the converted BGR data.
/// * `bgra_stride` - The stride (components per row) for the BGR image data.
/// * `endianness` - The endianness of stored bytes
/// * `bytes_packing` - see [YuvBytesPacking] for more info.
/// * `mode` - See [YuvConversionMode] for more info.
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGR data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv_nv16_p10_to_bgr(
    bi_planar_image: &YuvBiPlanarImage<u16>,
    bgra: &mut [u8],
    bgra_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    endianness: YuvEndianness,
    bytes_packing: YuvBytesPacking,
    mode: YuvConversionMode,
) -> Result<(), YuvError> {
    let dispatcher = match endianness {
        #[cfg(feature = "big_endian")]
        YuvEndianness::BigEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Bgr as u8 },
                    { YuvNVOrder::UV as u8 },
                    { YuvChromaSubsampling::Yuv422 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Bgr as u8 },
                    { YuvNVOrder::UV as u8 },
                    { YuvChromaSubsampling::Yuv422 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
        YuvEndianness::LittleEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Bgr as u8 },
                    { YuvNVOrder::UV as u8 },
                    { YuvChromaSubsampling::Yuv422 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Bgr as u8 },
                    { YuvNVOrder::UV as u8 },
                    { YuvChromaSubsampling::Yuv422 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
    };
    dispatcher(bi_planar_image, bgra, bgra_stride, range, matrix, mode)
}

/// Convert YUV NV61 format with 10-bit pixel format to BGR format.
///
/// This function takes YUV NV61 data with 10-bit precision.
/// and converts it to BGR format with 8-bit precision.
///
/// # Arguments
///
/// * `bi_planar_image` - Source Bi-Planar 10-bit image.
/// * `bgra` - A mutable slice to store the converted BGR data.
/// * `bgra_stride` - The stride (components per row) for the BGR image data.
/// * `endianness` - The endianness of stored bytes
/// * `bytes_packing` - see [YuvBytesPacking] for more info.
/// * `mode` - See [YuvConversionMode] for more info.
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGR data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv_nv61_p10_to_bgr(
    bi_planar_image: &YuvBiPlanarImage<u16>,
    bgra: &mut [u8],
    bgra_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    endianness: YuvEndianness,
    bytes_packing: YuvBytesPacking,
    mode: YuvConversionMode,
) -> Result<(), YuvError> {
    let dispatcher = match endianness {
        #[cfg(feature = "big_endian")]
        YuvEndianness::BigEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Bgr as u8 },
                    { YuvNVOrder::VU as u8 },
                    { YuvChromaSubsampling::Yuv422 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Bgr as u8 },
                    { YuvNVOrder::VU as u8 },
                    { YuvChromaSubsampling::Yuv422 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
        YuvEndianness::LittleEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Bgr as u8 },
                    { YuvNVOrder::VU as u8 },
                    { YuvChromaSubsampling::Yuv422 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Bgr as u8 },
                    { YuvNVOrder::VU as u8 },
                    { YuvChromaSubsampling::Yuv422 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
    };
    dispatcher(bi_planar_image, bgra, bgra_stride, range, matrix, mode)
}

/// Convert YUV NV16 format with 10-bit pixel format to RGB format.
///
/// This function takes YUV NV16 data with 10-bit precision.
/// and converts it to RGB format with 8-bit precision.
///
/// # Arguments
///
/// * `bi_planar_image` - Source Bi-Planar 10-bit image.
/// * `bgra` - A mutable slice to store the converted RGB data.
/// * `bgra_stride` - The stride (components per row) for the RGB image data.
/// * `endianness` - The endianness of stored bytes
/// * `bytes_packing` - see [YuvBytesPacking] for more info.
/// * `mode` - See [YuvConversionMode] for more info.
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input RGB data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv_nv16_p10_to_rgb(
    bi_planar_image: &YuvBiPlanarImage<u16>,
    bgra: &mut [u8],
    bgra_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    endianness: YuvEndianness,
    bytes_packing: YuvBytesPacking,
    mode: YuvConversionMode,
) -> Result<(), YuvError> {
    let dispatcher = match endianness {
        #[cfg(feature = "big_endian")]
        YuvEndianness::BigEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Rgb as u8 },
                    { YuvNVOrder::UV as u8 },
                    { YuvChromaSubsampling::Yuv422 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Rgb as u8 },
                    { YuvNVOrder::UV as u8 },
                    { YuvChromaSubsampling::Yuv422 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
        YuvEndianness::LittleEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Rgb as u8 },
                    { YuvNVOrder::UV as u8 },
                    { YuvChromaSubsampling::Yuv422 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Rgb as u8 },
                    { YuvNVOrder::UV as u8 },
                    { YuvChromaSubsampling::Yuv422 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
    };
    dispatcher(bi_planar_image, bgra, bgra_stride, range, matrix, mode)
}

/// Convert YUV NV61 format with 10-bit pixel format to RGB format.
///
/// This function takes YUV NV61 data with 10-bit precision.
/// and converts it to RGB format with 8-bit precision.
///
/// # Arguments
///
/// * `bi_planar_image` - Source Bi-Planar 10-bit image.
/// * `bgra` - A mutable slice to store the converted RGB data.
/// * `bgra_stride` - The stride (components per row) for the RGB image data.
/// * `endianness` - The endianness of stored bytes
/// * `bytes_packing` - see [YuvBytesPacking] for more info.
/// * `mode` - See [YuvConversionMode] for more info.
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input RGB data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv_nv61_p10_to_rgb(
    bi_planar_image: &YuvBiPlanarImage<u16>,
    bgra: &mut [u8],
    bgra_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    endianness: YuvEndianness,
    bytes_packing: YuvBytesPacking,
    mode: YuvConversionMode,
) -> Result<(), YuvError> {
    let dispatcher = match endianness {
        #[cfg(feature = "big_endian")]
        YuvEndianness::BigEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Rgb as u8 },
                    { YuvNVOrder::VU as u8 },
                    { YuvChromaSubsampling::Yuv422 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Rgb as u8 },
                    { YuvNVOrder::VU as u8 },
                    { YuvChromaSubsampling::Yuv422 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
        YuvEndianness::LittleEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Rgb as u8 },
                    { YuvNVOrder::VU as u8 },
                    { YuvChromaSubsampling::Yuv422 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Rgb as u8 },
                    { YuvNVOrder::VU as u8 },
                    { YuvChromaSubsampling::Yuv422 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
    };
    dispatcher(bi_planar_image, bgra, bgra_stride, range, matrix, mode)
}

/// Convert YUV NV16 format with 10-bit pixel format to RGB format.
///
/// This function takes YUV NV16 data with 10-bit precision.
/// and converts it to RGBA format with 8-bit precision.
///
/// # Arguments
///
/// * `bi_planar_image` - Source Bi-Planar 10-bit image.
/// * `bgra` - A mutable slice to store the converted RGBA data.
/// * `bgra_stride` - The stride (components per row) for the RGBA image data.
/// * `endianness` - The endianness of stored bytes
/// * `bytes_packing` - see [YuvBytesPacking] for more info.
/// * `mode` - See [YuvConversionMode] for more info.
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input RGBA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv_nv16_p10_to_rgba(
    bi_planar_image: &YuvBiPlanarImage<u16>,
    bgra: &mut [u8],
    bgra_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    endianness: YuvEndianness,
    bytes_packing: YuvBytesPacking,
    mode: YuvConversionMode,
) -> Result<(), YuvError> {
    let dispatcher = match endianness {
        #[cfg(feature = "big_endian")]
        YuvEndianness::BigEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Rgba as u8 },
                    { YuvNVOrder::UV as u8 },
                    { YuvChromaSubsampling::Yuv422 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Rgba as u8 },
                    { YuvNVOrder::UV as u8 },
                    { YuvChromaSubsampling::Yuv422 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
        YuvEndianness::LittleEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Rgba as u8 },
                    { YuvNVOrder::UV as u8 },
                    { YuvChromaSubsampling::Yuv422 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Rgba as u8 },
                    { YuvNVOrder::UV as u8 },
                    { YuvChromaSubsampling::Yuv422 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
    };
    dispatcher(bi_planar_image, bgra, bgra_stride, range, matrix, mode)
}

/// Convert YUV NV61 format with 10-bit pixel format to RGB format.
///
/// This function takes YUV NV61 data with 10-bit precision.
/// and converts it to RGBA format with 8-bit precision.
///
/// # Arguments
///
/// * `bi_planar_image` - Source Bi-Planar 10-bit image.
/// * `rgba` - A mutable slice to store the converted RGBA data.
/// * `rgba_stride` - The stride (components per row) for the RGBA image data.
/// * `endianness` - The endianness of stored bytes
/// * `bytes_packing` - see [YuvBytesPacking] for more info.
/// * `mode` - See [YuvConversionMode] for more info.
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input RGBA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv_nv61_p10_to_rgba(
    bi_planar_image: &YuvBiPlanarImage<u16>,
    rgba: &mut [u8],
    rgba_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    endianness: YuvEndianness,
    bytes_packing: YuvBytesPacking,
    mode: YuvConversionMode,
) -> Result<(), YuvError> {
    let dispatcher = match endianness {
        #[cfg(feature = "big_endian")]
        YuvEndianness::BigEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Rgba as u8 },
                    { YuvNVOrder::VU as u8 },
                    { YuvChromaSubsampling::Yuv422 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Rgba as u8 },
                    { YuvNVOrder::VU as u8 },
                    { YuvChromaSubsampling::Yuv422 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
        YuvEndianness::LittleEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Rgba as u8 },
                    { YuvNVOrder::VU as u8 },
                    { YuvChromaSubsampling::Yuv422 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Rgba as u8 },
                    { YuvNVOrder::VU as u8 },
                    { YuvChromaSubsampling::Yuv422 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
    };
    dispatcher(bi_planar_image, rgba, rgba_stride, range, matrix, mode)
}

/// Convert YUV NV21 format with 10-bit pixel format to BGR format.
///
/// This function takes YUV NV21 data with 10-bit precision
/// and converts it to BGR format with 8-bit precision.
///
/// # Arguments
///
/// * `bi_planar_image` - Source Bi-Planar 10-bit image.
/// * `bgr` - A mutable slice to store the converted BGR data.
/// * `bgr_stride` - The stride (components per row) for the BGR image data.
/// * `endianness` - The endianness of stored bytes
/// * `bytes_packing` - see [YuvBytesPacking] for more info.
/// * `mode` - See [YuvConversionMode] for more info.
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGR data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv_nv21_p10_to_bgr(
    bi_planar_image: &YuvBiPlanarImage<u16>,
    bgr: &mut [u8],
    bgr_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    endianness: YuvEndianness,
    bytes_packing: YuvBytesPacking,
    mode: YuvConversionMode,
) -> Result<(), YuvError> {
    let dispatcher = match endianness {
        #[cfg(feature = "big_endian")]
        YuvEndianness::BigEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Bgr as u8 },
                    { YuvNVOrder::VU as u8 },
                    { YuvChromaSubsampling::Yuv420 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Bgr as u8 },
                    { YuvNVOrder::VU as u8 },
                    { YuvChromaSubsampling::Yuv420 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
        YuvEndianness::LittleEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Bgr as u8 },
                    { YuvNVOrder::VU as u8 },
                    { YuvChromaSubsampling::Yuv420 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Bgr as u8 },
                    { YuvNVOrder::VU as u8 },
                    { YuvChromaSubsampling::Yuv420 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
    };
    dispatcher(bi_planar_image, bgr, bgr_stride, range, matrix, mode)
}

/// Convert YUV NV21 format with 10-bit pixel format to BGRA format.
///
/// This function takes YUV NV21 data with 10-bit precision
/// and converts it to BGRA format with 8-bit precision.
///
/// # Arguments
///
/// * `bi_planar_image` - Source Bi-Planar 10-bit image.
/// * `bgra` - A mutable slice to store the converted BGRA data.
/// * `bgra_stride` - The stride (components per row) for the BGRA image data.
/// * `endianness` - The endianness of stored bytes
/// * `bytes_packing` - see [YuvBytesPacking] for more info.
/// * `mode` - See [YuvConversionMode] for more info.
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGRA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv_nv21_p10_to_bgra(
    bi_planar_image: &YuvBiPlanarImage<u16>,
    bgra: &mut [u8],
    bgra_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    endianness: YuvEndianness,
    bytes_packing: YuvBytesPacking,
    mode: YuvConversionMode,
) -> Result<(), YuvError> {
    let dispatcher = match endianness {
        #[cfg(feature = "big_endian")]
        YuvEndianness::BigEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Bgra as u8 },
                    { YuvNVOrder::VU as u8 },
                    { YuvChromaSubsampling::Yuv420 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Bgra as u8 },
                    { YuvNVOrder::VU as u8 },
                    { YuvChromaSubsampling::Yuv420 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
        YuvEndianness::LittleEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Bgra as u8 },
                    { YuvNVOrder::VU as u8 },
                    { YuvChromaSubsampling::Yuv420 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Bgra as u8 },
                    { YuvNVOrder::VU as u8 },
                    { YuvChromaSubsampling::Yuv420 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
    };
    dispatcher(bi_planar_image, bgra, bgra_stride, range, matrix, mode)
}

/// Convert YUV NV21 format with 10-bit pixel format to RGB format.
///
/// This function takes YUV NV21 data with 10-bit precision
/// and converts it to RGB format with 8-bit precision.
///
/// # Arguments
///
/// * `bi_planar_image` - Source Bi-Planar 10-bit image.
/// * `rgb` - A mutable slice to store the converted RGB data.
/// * `rgb_stride` - The stride (components per row) for the RGB image data.
/// * `endianness` - The endianness of stored bytes
/// * `bytes_packing` - see [YuvBytesPacking] for more info.
/// * `mode` - See [YuvConversionMode] for more info.
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input RGB data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv_nv21_p10_to_rgb(
    bi_planar_image: &YuvBiPlanarImage<u16>,
    rgb: &mut [u8],
    rgb_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    endianness: YuvEndianness,
    bytes_packing: YuvBytesPacking,
    mode: YuvConversionMode,
) -> Result<(), YuvError> {
    let dispatcher = match endianness {
        #[cfg(feature = "big_endian")]
        YuvEndianness::BigEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Rgb as u8 },
                    { YuvNVOrder::VU as u8 },
                    { YuvChromaSubsampling::Yuv420 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Rgb as u8 },
                    { YuvNVOrder::VU as u8 },
                    { YuvChromaSubsampling::Yuv420 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
        YuvEndianness::LittleEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Rgb as u8 },
                    { YuvNVOrder::VU as u8 },
                    { YuvChromaSubsampling::Yuv420 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Rgb as u8 },
                    { YuvNVOrder::VU as u8 },
                    { YuvChromaSubsampling::Yuv420 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
    };
    dispatcher(bi_planar_image, rgb, rgb_stride, range, matrix, mode)
}

/// Convert YUV NV21 format with 10-bit pixel format to RGBA format.
///
/// This function takes YUV NV21 data with 10-bit precision
/// and converts it to RGBA format with 8-bit precision.
///
/// # Arguments
///
/// * `bi_planar_image` - Source Bi-Planar 10-bit image.
/// * `rgba` - A mutable slice to store the converted RGBA data.
/// * `rgba_stride` - The stride (components per row) for the RGBA image data.
/// * `endianness` - The endianness of stored bytes
/// * `bytes_packing` - see [YuvBytesPacking] for more info.
/// * `mode` - See [YuvConversionMode] for more info.
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input RGBA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv_nv21_p10_to_rgba(
    bi_planar_image: &YuvBiPlanarImage<u16>,
    rgba: &mut [u8],
    rgba_stride: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    endianness: YuvEndianness,
    bytes_packing: YuvBytesPacking,
    mode: YuvConversionMode,
) -> Result<(), YuvError> {
    let dispatcher = match endianness {
        #[cfg(feature = "big_endian")]
        YuvEndianness::BigEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Rgba as u8 },
                    { YuvNVOrder::VU as u8 },
                    { YuvChromaSubsampling::Yuv420 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Rgba as u8 },
                    { YuvNVOrder::VU as u8 },
                    { YuvChromaSubsampling::Yuv420 as u8 },
                    { YuvEndianness::BigEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
        YuvEndianness::LittleEndian => match bytes_packing {
            YuvBytesPacking::MostSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Rgba as u8 },
                    { YuvNVOrder::VU as u8 },
                    { YuvChromaSubsampling::Yuv420 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::MostSignificantBytes as u8 },
                >
            }
            YuvBytesPacking::LeastSignificantBytes => {
                yuv_nv_p10_to_image_impl::<
                    { YuvSourceChannels::Rgba as u8 },
                    { YuvNVOrder::VU as u8 },
                    { YuvChromaSubsampling::Yuv420 as u8 },
                    { YuvEndianness::LittleEndian as u8 },
                    { YuvBytesPacking::LeastSignificantBytes as u8 },
                >
            }
        },
    };
    dispatcher(bi_planar_image, rgba, rgba_stride, range, matrix, mode)
}
