/*
 * Copyright (c) Radzivon Bartoshyk, 2/2025. All rights reserved.
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
#![allow(clippy::excessive_precision)]

use crate::internals::ProcessedOffset;
use crate::numerics::qrshr;
use crate::yuv_error::check_rgba_destination;
use crate::yuv_support::{CbCrForwardTransform, CbCrInverseTransform, YuvChromaRange};
use crate::{YuvChromaSubsampling, YuvError, YuvPlanarImage, YuvPlanarImageMut, YuvRange};
use std::fmt::{Display, Formatter};

#[repr(u8)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub(crate) enum RdpChannels {
    Rgb = 0,
    Rgba = 1,
    Bgra = 2,
    Bgr = 3,
    Abgr = 4,
    Argb = 5,
}

impl Display for RdpChannels {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            RdpChannels::Rgb => f.write_str("RdpChannels::Rgb"),
            RdpChannels::Rgba => f.write_str("RdpChannels::Rgba"),
            RdpChannels::Bgra => f.write_str("RdpChannels::Bgra"),
            RdpChannels::Bgr => f.write_str("RdpChannels::Bgr"),
            RdpChannels::Abgr => f.write_str("RdpChannels::Abgr"),
            RdpChannels::Argb => f.write_str("RdpChannels::Argb"),
        }
    }
}

impl From<u8> for RdpChannels {
    #[inline(always)]
    fn from(value: u8) -> Self {
        match value {
            0 => RdpChannels::Rgb,
            1 => RdpChannels::Rgba,
            2 => RdpChannels::Bgra,
            3 => RdpChannels::Bgr,
            4 => RdpChannels::Abgr,
            5 => RdpChannels::Argb,
            _ => {
                unimplemented!("Unknown value")
            }
        }
    }
}

impl RdpChannels {
    #[inline(always)]
    pub const fn get_channels_count(&self) -> usize {
        match self {
            RdpChannels::Rgb | RdpChannels::Bgr => 3,
            RdpChannels::Rgba | RdpChannels::Bgra | RdpChannels::Abgr | RdpChannels::Argb => 4,
        }
    }

    #[inline(always)]
    pub const fn has_alpha(&self) -> bool {
        match self {
            RdpChannels::Rgb | RdpChannels::Bgr => false,
            RdpChannels::Rgba | RdpChannels::Bgra | RdpChannels::Abgr | RdpChannels::Argb => true,
        }
    }
}

impl RdpChannels {
    #[inline(always)]
    pub const fn get_r_channel_offset(&self) -> usize {
        match self {
            RdpChannels::Rgb => 0,
            RdpChannels::Rgba => 0,
            RdpChannels::Bgra => 2,
            RdpChannels::Bgr => 2,
            RdpChannels::Abgr => 3,
            RdpChannels::Argb => 1,
        }
    }

    #[inline(always)]
    pub const fn get_g_channel_offset(&self) -> usize {
        match self {
            RdpChannels::Rgb | RdpChannels::Bgr => 1,
            RdpChannels::Rgba | RdpChannels::Bgra => 1,
            RdpChannels::Abgr | RdpChannels::Argb => 2,
        }
    }

    #[inline(always)]
    pub const fn get_b_channel_offset(&self) -> usize {
        match self {
            RdpChannels::Rgb => 2,
            RdpChannels::Rgba => 2,
            RdpChannels::Bgra => 0,
            RdpChannels::Bgr => 0,
            RdpChannels::Abgr => 1,
            RdpChannels::Argb => 3,
        }
    }
    #[inline(always)]
    pub const fn get_a_channel_offset(&self) -> usize {
        match self {
            RdpChannels::Rgb | RdpChannels::Bgr => 0,
            RdpChannels::Rgba | RdpChannels::Bgra => 3,
            RdpChannels::Abgr | RdpChannels::Argb => 0,
        }
    }
}

type RgbEncoderHandler = Option<
    unsafe fn(
        transform: &CbCrForwardTransform<i32>,
        y_plane: &mut [i16],
        u_plane: &mut [i16],
        v_plane: &mut [i16],
        rgba: &[u8],
        width: usize,
    ) -> ProcessedOffset,
>;

struct RgbEncoder<const ORIGIN_CHANNELS: u8, const Q: i32> {
    handler: RgbEncoderHandler,
}

impl<const ORIGIN_CHANNELS: u8, const Q: i32> Default for RgbEncoder<ORIGIN_CHANNELS, Q> {
    fn default() -> Self {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            #[cfg(feature = "avx")]
            {
                let use_avx = std::arch::is_x86_feature_detected!("avx2");
                if use_avx {
                    use crate::avx2::rdp_avx2_rgba_to_yuv;
                    return RgbEncoder {
                        handler: Some(rdp_avx2_rgba_to_yuv::<ORIGIN_CHANNELS, Q>),
                    };
                }
            }
        }
        RgbEncoder { handler: None }
    }
}

pub(crate) trait WideRdpRowForwardHandler<V, T, K> {
    fn handle_row(
        &self,
        y_plane: &mut [V],
        u_plane: &mut [V],
        v_plane: &mut [V],
        rgba: &[T],
        width: u32,
        chroma: YuvChromaRange,
        transform: &CbCrForwardTransform<K>,
    ) -> ProcessedOffset;
}

impl<const ORIGIN_CHANNELS: u8, const Q: i32> WideRdpRowForwardHandler<i16, u8, i32>
    for RgbEncoder<ORIGIN_CHANNELS, Q>
{
    fn handle_row(
        &self,
        y_plane: &mut [i16],
        u_plane: &mut [i16],
        v_plane: &mut [i16],
        rgba: &[u8],
        width: u32,
        _: YuvChromaRange,
        transform: &CbCrForwardTransform<i32>,
    ) -> ProcessedOffset {
        if let Some(handler) = self.handler {
            unsafe {
                return handler(transform, y_plane, u_plane, v_plane, rgba, width as usize);
            }
        }
        ProcessedOffset { cx: 0, ux: 0 }
    }
}

fn to_rdp_yuv<const ORIGIN_CHANNELS: u8>(
    planar_image: &mut YuvPlanarImageMut<i16>,
    rgba: &[u8],
    rgba_stride: u32,
) -> Result<(), YuvError> {
    let ch: RdpChannels = ORIGIN_CHANNELS.into();
    let channels = ch.get_channels_count();
    planar_image.check_constraints(YuvChromaSubsampling::Yuv444)?;
    check_rgba_destination(
        rgba,
        rgba_stride,
        planar_image.width,
        planar_image.height,
        channels,
    )?;

    let y_plane = planar_image.y_plane.borrow_mut();
    let u_plane = planar_image.u_plane.borrow_mut();
    let v_plane = planar_image.v_plane.borrow_mut();

    const PRECISION: i32 = 15;
    const SCALE: f32 = (1 << PRECISION) as f32;
    const Y_R: i32 = (0.299 * SCALE) as i32;
    const Y_G: i32 = (0.587 * SCALE) as i32;
    const Y_B: i32 = (0.114 * SCALE) as i32;
    const CB_R: i32 = -(0.168_935 * SCALE) as i32;
    const CB_G: i32 = -(0.331_665 * SCALE) as i32;
    const CB_B: i32 = (0.500_59 * SCALE) as i32;
    const CR_R: i32 = (0.499_813 * SCALE) as i32;
    const CR_G: i32 = -(0.418_531 * SCALE) as i32;
    const CR_B: i32 = -(0.081_282 * SCALE) as i32;

    let b_transform = CbCrForwardTransform {
        yr: Y_R,
        yg: Y_G,
        yb: Y_B,
        cb_r: CB_R,
        cb_g: CB_G,
        cb_b: CB_B,
        cr_r: CR_R,
        cr_g: CR_G,
        cr_b: CR_B,
    };

    let handler = RgbEncoder::<ORIGIN_CHANNELS, 10>::default();

    let iter = y_plane
        .chunks_exact_mut(planar_image.y_stride as usize)
        .zip(u_plane.chunks_exact_mut(planar_image.u_stride as usize))
        .zip(v_plane.chunks_exact_mut(planar_image.v_stride as usize))
        .zip(rgba.chunks_exact(rgba_stride as usize));

    iter.for_each(|(((y_dst, u_dst), v_dst), rgba)| {
        let offset = handler.handle_row(
            y_dst,
            u_dst,
            v_dst,
            rgba,
            planar_image.width,
            YuvChromaRange {
                bias_y: 4096,
                bias_uv: 4096,
                range: YuvRange::Full,
                range_uv: 0,
                range_y: 0,
            },
            &b_transform,
        );

        for (((y_dst, u_dst), v_dst), rgba) in y_dst
            .iter_mut()
            .zip(u_dst.iter_mut())
            .zip(v_dst.iter_mut())
            .zip(rgba.chunks_exact(channels))
            .take(planar_image.width as usize)
            .skip(offset.cx)
        {
            let r = rgba[ch.get_r_channel_offset()] as i32;
            let g = rgba[ch.get_g_channel_offset()] as i32;
            let b = rgba[ch.get_b_channel_offset()] as i32;
            const Q: i32 = 10;
            const RND: i32 = (1 << (Q - 1)) - 1;
            const Y_RND: i32 = 4096 * (1 << Q);

            let y0 =
                (r * b_transform.yr + g * b_transform.yg + b * b_transform.yb - Y_RND + RND) >> Q;
            let u = (r * b_transform.cb_r + g * b_transform.cb_g + b * b_transform.cb_b + RND) >> Q;
            let v = (r * b_transform.cr_r + g * b_transform.cr_g + b * b_transform.cr_b + RND) >> Q;

            *y_dst = y0 as i16;
            *u_dst = u as i16;
            *v_dst = v as i16;
        }
    });

    Ok(())
}

macro_rules! d_forward {
    ($method: ident, $cn: expr, $name: ident, $stride_name: ident) => {
        pub fn $method(
            planar_image: &mut YuvPlanarImageMut<i16>,
            $name: &[u8],
            $stride_name: u32,
        ) -> Result<(), YuvError> {
            to_rdp_yuv::<{ $cn as u8 }>(planar_image, $name, $stride_name)
        }
    };
}

d_forward!(rdp_rgb_to_yuv444, RdpChannels::Rgb, rgb, rgb_stride);
d_forward!(rdp_rgba_to_yuv444, RdpChannels::Rgba, rgba, rgba_stride);
d_forward!(rdp_bgra_to_yuv444, RdpChannels::Bgra, bgra, bgra_stride);
d_forward!(rdp_abgr_to_yuv444, RdpChannels::Abgr, abgr, abgr_stride);
d_forward!(rdp_bgr_to_yuv444, RdpChannels::Bgr, bgr, bgr_stride);
d_forward!(rdp_argb_to_yuv444, RdpChannels::Argb, argb, argb_stride);

fn rdp_yuv_to_rgb<const ORIGIN_CHANNELS: u8>(
    planar_image: &YuvPlanarImage<i16>,
    rgba: &mut [u8],
    rgba_stride: u32,
) -> Result<(), YuvError> {
    let ch: RdpChannels = ORIGIN_CHANNELS.into();
    let channels = ch.get_channels_count();
    planar_image.check_constraints(YuvChromaSubsampling::Yuv444)?;
    check_rgba_destination(
        rgba,
        rgba_stride,
        planar_image.width,
        planar_image.height,
        channels,
    )?;

    let y_plane = planar_image.y_plane;
    let u_plane = planar_image.u_plane;
    let v_plane = planar_image.v_plane;

    const PRECISION: i32 = 16;
    const Y_SCALE: i32 = 1 << PRECISION;
    const B_Y: i32 = (1.402525f32 * Y_SCALE as f32) as i32;
    const B_G_1: i32 = (0.343730f32 * Y_SCALE as f32) as i32;
    const B_G_2: i32 = (0.714401f32 * Y_SCALE as f32) as i32;
    const B_B_1: i32 = (1.769905 * Y_SCALE as f32) as i32;

    let b_transform = CbCrInverseTransform::<i32> {
        y_coef: Y_SCALE,
        cr_coef: B_Y,
        cb_coef: B_B_1,
        g_coeff_1: B_G_1,
        g_coeff_2: B_G_2,
    };

    let iter = y_plane
        .chunks_exact(planar_image.y_stride as usize)
        .zip(u_plane.chunks_exact(planar_image.u_stride as usize))
        .zip(v_plane.chunks_exact(planar_image.v_stride as usize))
        .zip(rgba.chunks_exact_mut(rgba_stride as usize));

    iter.for_each(|(((y_dst, u_dst), v_dst), rgba)| {
        let mut _cx = 0;

        let mut _offset = ProcessedOffset { cx: 0, ux: 0 };

        _cx = _offset.cx;

        for (((&y_0, &u), &v), rgba) in y_dst
            .iter()
            .zip(u_dst.iter())
            .zip(v_dst.iter())
            .zip(rgba.chunks_exact_mut(channels))
            .take(planar_image.width as usize)
            .skip(_cx)
        {
            let y = y_0;
            let yy = ((y + 4096) as i32) * Y_SCALE;
            let r = qrshr::<21, 8>(yy + b_transform.cr_coef * v as i32);
            let g = qrshr::<21, 8>(
                yy - b_transform.g_coeff_2 * v as i32 - b_transform.g_coeff_1 * u as i32,
            );
            let b = qrshr::<21, 8>(yy + b_transform.cb_coef * u as i32);

            rgba[ch.get_r_channel_offset()] = r as u8;
            rgba[ch.get_g_channel_offset()] = g as u8;
            rgba[ch.get_b_channel_offset()] = b as u8;
            if ch.has_alpha() {
                rgba[ch.get_a_channel_offset()] = 255;
            }
        }
    });
    Ok(())
}

macro_rules! d_backward {
    ($method: ident, $cn: expr, $name: ident, $stride_name: ident) => {
        pub fn $method(
            planar_image: &YuvPlanarImage<i16>,
            $name: &mut [u8],
            $stride_name: u32,
        ) -> Result<(), YuvError> {
            rdp_yuv_to_rgb::<{ $cn as u8 }>(planar_image, $name, $stride_name)
        }
    };
}

d_backward!(rdp_yuv444_to_rgb, RdpChannels::Rgb, rgb, rgb_stride);
d_backward!(rdp_yuv444_to_rgba, RdpChannels::Rgba, rgba, rgba_stride);
d_backward!(rdp_yuv444_to_bgra, RdpChannels::Bgra, bgra, bgra_stride);
d_backward!(rdp_yuv444_to_abgr, RdpChannels::Abgr, abgr, abgr_stride);
d_backward!(rdp_yuv444_to_bgr, RdpChannels::Bgr, bgr, bgr_stride);
d_backward!(rdp_yuv444_to_argb, RdpChannels::Argb, argb, argb_stride);
