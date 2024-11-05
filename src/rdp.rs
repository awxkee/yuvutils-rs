/*
 * Copyright (c) Radzivon Bartoshyk, 11/2024. All rights reserved.
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

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use crate::avx2::rdp_avx2_rgba_to_yuv;
use crate::internals::ProcessedOffset;
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use crate::neon::{rdp_neon_rgba_to_yuv, rdp_neon_yuv_to_rgba_row};
use crate::numerics::qrshr;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use crate::sse::{rdp_sse_yuv_to_rgba_row, sse_rdp_rgba_to_yuv_row};
use crate::yuv_error::check_rgba_destination;
use crate::yuv_support::{CbCrForwardTransform, CbCrInverseTransform, YuvSourceChannels};
use crate::{YuvChromaSubsample, YuvError, YuvPlanarImage, YuvPlanarImageMut};

#[inline(always)]
fn rescale_fp_12<const PRECISION: i32>(v: i32) -> u16 {
    let shift = PRECISION - 5;
    let rounding: i32 = 1 << (shift - 1);
    let new_v = (v + rounding) >> shift;
    new_v.clamp(-4096, 4095) as u16
}

#[inline(always)]
fn rescale_y_fp_12<const PRECISION: i32>(v: i32) -> u16 {
    let shift = PRECISION - 5;
    let rounding: i32 = 1 << (shift - 1);
    let new_v = (v + rounding) >> shift;
    (new_v - 4096).clamp(-4096, 4095) as u16
}

fn to_rdp_yuv<const ORIGIN_CHANNELS: u8>(
    planar_image: &mut YuvPlanarImageMut<u16>,
    rgba: &[u8],
    rgba_stride: u32,
) -> Result<(), YuvError> {
    let ch: YuvSourceChannels = ORIGIN_CHANNELS.into();
    let channels = ch.get_channels_count();
    planar_image.check_constraints(YuvChromaSubsample::Yuv444)?;
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

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    const PRECISION: i32 = 13;
    #[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
    const PRECISION: i32 = 14;
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

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    let use_sse = std::arch::is_x86_feature_detected!("sse4.1");
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    let use_avx = std::arch::is_x86_feature_detected!("avx2");

    for (((y_dst, u_dst), v_dst), rgba) in y_plane
        .chunks_exact_mut(planar_image.y_stride as usize)
        .zip(u_plane.chunks_exact_mut(planar_image.u_stride as usize))
        .zip(v_plane.chunks_exact_mut(planar_image.v_stride as usize))
        .zip(rgba.chunks_exact(rgba_stride as usize))
    {
        let mut _cx = 0;

        let mut _offset = ProcessedOffset { cx: 0, ux: 0 };

        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        {
            unsafe {
                _offset = rdp_neon_rgba_to_yuv::<ORIGIN_CHANNELS, PRECISION>(
                    &b_transform,
                    y_dst.as_mut_ptr(),
                    u_dst.as_mut_ptr(),
                    v_dst.as_mut_ptr(),
                    rgba,
                    _offset.cx,
                    _offset.ux,
                    planar_image.width as usize,
                );
            }
        }

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            unsafe {
                if use_avx {
                    let offset = rdp_avx2_rgba_to_yuv::<ORIGIN_CHANNELS>(
                        &b_transform,
                        y_dst,
                        u_dst,
                        v_dst,
                        rgba,
                        _offset.cx,
                        planar_image.width as usize,
                    );
                    _offset = offset;
                }
            }
            unsafe {
                if use_sse {
                    let offset = sse_rdp_rgba_to_yuv_row::<ORIGIN_CHANNELS>(
                        &b_transform,
                        y_dst.as_mut_ptr(),
                        u_dst.as_mut_ptr(),
                        v_dst.as_mut_ptr(),
                        rgba,
                        _offset.cx,
                        _offset.ux,
                        planar_image.width as usize,
                    );
                    _offset = offset;
                }
            }
        }

        _cx = _offset.cx;

        for (((y_dst, u_dst), v_dst), rgba) in y_dst
            .iter_mut()
            .zip(u_dst.iter_mut())
            .zip(v_dst.iter_mut())
            .zip(rgba.chunks_exact(channels))
            .skip(_cx)
        {
            let r = rgba[ch.get_r_channel_offset()] as i32;
            let g = rgba[ch.get_g_channel_offset()] as i32;
            let b = rgba[ch.get_b_channel_offset()] as i32;

            let y0 = rescale_y_fp_12::<PRECISION>(
                r * b_transform.yr + g * b_transform.yg + b * b_transform.yb,
            );
            let u = rescale_fp_12::<PRECISION>(
                r * b_transform.cb_r + g * b_transform.cb_g + b * b_transform.cb_b,
            );
            let v = rescale_fp_12::<PRECISION>(
                r * b_transform.cr_r + g * b_transform.cr_g + b * b_transform.cr_b,
            );

            *y_dst = y0;
            *u_dst = u;
            *v_dst = v;
        }
    }

    Ok(())
}

pub fn rdp_rgb_to_yuv444(
    planar_image: &mut YuvPlanarImageMut<u16>,
    rgba: &[u8],
    rgba_stride: u32,
) -> Result<(), YuvError> {
    to_rdp_yuv::<{ YuvSourceChannels::Rgb as u8 }>(planar_image, rgba, rgba_stride)
}

pub fn rdp_rgba_to_yuv444(
    planar_image: &mut YuvPlanarImageMut<u16>,
    rgba: &[u8],
    rgba_stride: u32,
) -> Result<(), YuvError> {
    to_rdp_yuv::<{ YuvSourceChannels::Rgba as u8 }>(planar_image, rgba, rgba_stride)
}

fn rdp_yuv_to_rgb<const ORIGIN_CHANNELS: u8>(
    planar_image: &YuvPlanarImage<u16>,
    rgba: &mut [u8],
    rgba_stride: u32,
) -> Result<(), YuvError> {
    let ch: YuvSourceChannels = ORIGIN_CHANNELS.into();
    let channels = ch.get_channels_count();
    planar_image.check_constraints(YuvChromaSubsample::Yuv444)?;
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

    const PRECISION: i32 = 12;
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

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    let use_sse = std::arch::is_x86_feature_detected!("sse4.1");

    for (((y_dst, u_dst), v_dst), rgba) in y_plane
        .chunks_exact(planar_image.y_stride as usize)
        .zip(u_plane.chunks_exact(planar_image.u_stride as usize))
        .zip(v_plane.chunks_exact(planar_image.v_stride as usize))
        .zip(rgba.chunks_exact_mut(rgba_stride as usize))
    {
        let mut _cx = 0;

        let mut _offset = ProcessedOffset { cx: 0, ux: 0 };

        #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
        {
            unsafe {
                _offset = rdp_neon_yuv_to_rgba_row::<ORIGIN_CHANNELS>(
                    &b_transform,
                    y_dst,
                    u_dst,
                    v_dst,
                    rgba,
                    _offset.cx,
                    _offset.ux,
                    planar_image.width as usize,
                );
            }
        }

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            unsafe {
                if use_sse {
                    let offset = rdp_sse_yuv_to_rgba_row::<ORIGIN_CHANNELS>(
                        &b_transform,
                        y_dst,
                        u_dst,
                        v_dst,
                        rgba,
                        _offset.cx,
                        planar_image.width as usize,
                    );
                    _offset = offset;
                }
            }
        }

        _cx = _offset.cx;
        
        for (((&y_0, &u), &v), rgba) in y_dst
            .iter()
            .zip(u_dst.iter())
            .zip(v_dst.iter())
            .zip(rgba.chunks_exact_mut(channels))
            .skip(_cx)
        {
            let y = y_0 as i16;
            let u = u as i16;
            let v = v as i16;
            let yy = (y + 4096) as i32 * Y_SCALE;
            let r = qrshr::<17, 8>(yy + b_transform.cr_coef * v as i32);
            let g = qrshr::<17, 8>(
                yy - b_transform.g_coeff_2 * v as i32 - b_transform.g_coeff_1 * u as i32,
            );
            let b = qrshr::<17, 8>(yy + b_transform.cb_coef * u as i32);

            rgba[ch.get_r_channel_offset()] = r as u8;
            rgba[ch.get_g_channel_offset()] = g as u8;
            rgba[ch.get_b_channel_offset()] = b as u8;
            if ch.has_alpha() {
                rgba[ch.get_a_channel_offset()] = 255;
            }
        }
    }

    Ok(())
}

pub fn rdp_yuv444_to_rgb(
    planar_image: &YuvPlanarImage<u16>,
    rgba: &mut [u8],
    rgba_stride: u32,
) -> Result<(), YuvError> {
    rdp_yuv_to_rgb::<{ YuvSourceChannels::Rgb as u8 }>(planar_image, rgba, rgba_stride)
}

pub fn rdp_yuv444_to_rgba(
    planar_image: &YuvPlanarImage<u16>,
    rgba: &mut [u8],
    rgba_stride: u32,
) -> Result<(), YuvError> {
    rdp_yuv_to_rgb::<{ YuvSourceChannels::Rgba as u8 }>(planar_image, rgba, rgba_stride)
}
