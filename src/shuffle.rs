/*
 * Copyright (c) Radzivon Bartoshyk, 12/2024. All rights reserved.
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
use crate::yuv_error::check_rgba_destination;
use crate::yuv_support::YuvSourceChannels;
use crate::YuvError;

pub(crate) trait ShuffleConverter<V: Copy, const SRC: u8, const DST: u8> {
    fn convert(&self, src: &[V], dst: &mut [V], width: usize);
}

trait ShuffleConverterFactory<V> {
    fn make_converter<const SRC: u8, const DST: u8>() -> Box<dyn ShuffleConverter<V, SRC, DST>>;
}

struct Rgba8DefaultConverter<const SRC: u8, const DST: u8> {}

impl<const SRC: u8, const DST: u8> Default for Rgba8DefaultConverter<SRC, DST> {
    fn default() -> Self {
        Rgba8DefaultConverter {}
    }
}

impl<const SRC: u8, const DST: u8> ShuffleConverter<u8, SRC, DST>
    for Rgba8DefaultConverter<SRC, DST>
{
    fn convert(&self, src: &[u8], dst: &mut [u8], _: usize) {
        let src_channels: YuvSourceChannels = SRC.into();
        let dst_channels: YuvSourceChannels = DST.into();
        for (dst, src) in dst
            .chunks_exact_mut(dst_channels.get_channels_count())
            .zip(src.chunks_exact(src_channels.get_channels_count()))
        {
            dst[dst_channels.get_r_channel_offset()] = src[src_channels.get_r_channel_offset()];
            dst[dst_channels.get_g_channel_offset()] = src[src_channels.get_g_channel_offset()];
            dst[dst_channels.get_b_channel_offset()] = src[src_channels.get_b_channel_offset()];
            if dst_channels.has_alpha() {
                let a = if src_channels.has_alpha() {
                    src[src_channels.get_a_channel_offset()]
                } else {
                    255
                };
                dst[dst_channels.get_a_channel_offset()] = a;
            }
        }
    }
}

impl ShuffleConverterFactory<u8> for u8 {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    fn make_converter<const SRC: u8, const DST: u8>() -> Box<dyn ShuffleConverter<u8, SRC, DST>> {
        use crate::sse::{ShuffleConverterSse, ShuffleQTableConverterSse};
        let mut converter: Box<dyn ShuffleConverter<u8, SRC, DST>> =
            Box::new(Rgba8DefaultConverter::default());
        let src_channels: YuvSourceChannels = SRC.into();
        let dst_channels: YuvSourceChannels = DST.into();
        if std::arch::is_x86_feature_detected!("sse4.1") {
            if src_channels.get_channels_count() == 4 && dst_channels.get_channels_count() == 4 {
                converter = Box::new(ShuffleQTableConverterSse::<SRC, DST>::create());
            } else {
                converter = Box::new(ShuffleConverterSse::<SRC, DST>::default());
            }
        }
        converter
    }

    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    fn make_converter<const SRC: u8, const DST: u8>() -> Box<dyn ShuffleConverter<u8, SRC, DST>> {
        use crate::neon::ShuffleConverterNeon;
        Box::new(ShuffleConverterNeon::<SRC, DST>::default())
    }

    #[cfg(not(any(
        all(target_arch = "aarch64", target_feature = "neon"),
        any(target_arch = "x86", target_arch = "x86_64")
    )))]
    fn make_converter<const SRC: u8, const DST: u8>() -> Box<dyn ShuffleConverter<u8, SRC, DST>> {
        Box::new(Rgba8DefaultConverter::<SRC, DST>::default())
    }
}

/// Channel reshuffling implementation
fn shuffle_impl<
    V: Copy + ShuffleConverterFactory<V>,
    const SRC: u8,
    const DST: u8,
    const BIT_DEPTH: usize,
>(
    src: &[V],
    src_stride: u32,
    dst: &mut [V],
    dst_stride: u32,
    width: u32,
    height: u32,
) -> Result<(), YuvError> {
    let src_channels: YuvSourceChannels = SRC.into();
    let dst_channels: YuvSourceChannels = DST.into();
    check_rgba_destination(
        src,
        src_stride,
        width,
        height,
        src_channels.get_channels_count(),
    )?;
    check_rgba_destination(
        dst,
        dst_stride,
        width,
        height,
        dst_channels.get_channels_count(),
    )?;

    let converter = V::make_converter::<SRC, DST>();

    for (dst, src) in dst
        .chunks_exact_mut(dst_stride as usize)
        .zip(src.chunks_exact(src_stride as usize))
    {
        let dst = &mut dst[0..dst_channels.get_channels_count() * width as usize];
        let src = &src[0..src_channels.get_channels_count() * width as usize];
        converter.convert(src, dst, dst_stride as usize);
    }

    Ok(())
}

/// Converts RGBA8 to BGRA8
///
/// # Arguments
///
/// * `src`: Source slice
/// * `src_stride`: Source slice stride
/// * `dst`: Destination slice
/// * `dst_stride`: Destination slice stride
/// * `width`: Image width
/// * `height`: Image height
///
/// returns: Result<(), YuvError>
///
pub fn rgba_to_bgra(
    src: &[u8],
    src_stride: u32,
    dst: &mut [u8],
    dst_stride: u32,
    width: u32,
    height: u32,
) -> Result<(), YuvError> {
    shuffle_impl::<u8, { YuvSourceChannels::Rgba as u8 }, { YuvSourceChannels::Bgra as u8 }, 8>(
        src, src_stride, dst, dst_stride, width, height,
    )
}

/// Converts RGBA8 to BGR8
///
/// # Arguments
///
/// * `src`: Source slice
/// * `src_stride`: Source slice stride
/// * `dst`: Destination slice
/// * `dst_stride`: Destination slice stride
/// * `width`: Image width
/// * `height`: Image height
///
/// returns: Result<(), YuvError>
///
pub fn rgba_to_bgr(
    src: &[u8],
    src_stride: u32,
    dst: &mut [u8],
    dst_stride: u32,
    width: u32,
    height: u32,
) -> Result<(), YuvError> {
    shuffle_impl::<u8, { YuvSourceChannels::Rgba as u8 }, { YuvSourceChannels::Bgr as u8 }, 8>(
        src, src_stride, dst, dst_stride, width, height,
    )
}

/// Converts RGBA8 to RGB8
///
/// # Arguments
///
/// * `src`: Source slice
/// * `src_stride`: Source slice stride
/// * `dst`: Destination slice
/// * `dst_stride`: Destination slice stride
/// * `width`: Image width
/// * `height`: Image height
///
/// returns: Result<(), YuvError>
///
pub fn rgba_to_rgb(
    src: &[u8],
    src_stride: u32,
    dst: &mut [u8],
    dst_stride: u32,
    width: u32,
    height: u32,
) -> Result<(), YuvError> {
    shuffle_impl::<u8, { YuvSourceChannels::Rgba as u8 }, { YuvSourceChannels::Rgb as u8 }, 8>(
        src, src_stride, dst, dst_stride, width, height,
    )
}

/// Converts RGB8 to BGBA8
///
/// # Arguments
///
/// * `src`: Source slice
/// * `src_stride`: Source slice stride
/// * `dst`: Destination slice
/// * `dst_stride`: Destination slice stride
/// * `width`: Image width
/// * `height`: Image height
///
/// returns: Result<(), YuvError>
///
pub fn rgb_to_bgra(
    src: &[u8],
    src_stride: u32,
    dst: &mut [u8],
    dst_stride: u32,
    width: u32,
    height: u32,
) -> Result<(), YuvError> {
    shuffle_impl::<u8, { YuvSourceChannels::Rgb as u8 }, { YuvSourceChannels::Bgra as u8 }, 8>(
        src, src_stride, dst, dst_stride, width, height,
    )
}

/// Converts RGB8 to BGB8
///
/// # Arguments
///
/// * `src`: Source slice
/// * `src_stride`: Source slice stride
/// * `dst`: Destination slice
/// * `dst_stride`: Destination slice stride
/// * `width`: Image width
/// * `height`: Image height
///
/// returns: Result<(), YuvError>
///
pub fn rgb_to_bgr(
    src: &[u8],
    src_stride: u32,
    dst: &mut [u8],
    dst_stride: u32,
    width: u32,
    height: u32,
) -> Result<(), YuvError> {
    shuffle_impl::<u8, { YuvSourceChannels::Rgb as u8 }, { YuvSourceChannels::Bgr as u8 }, 8>(
        src, src_stride, dst, dst_stride, width, height,
    )
}

/// Converts BGR8 to RGB8
///
/// # Arguments
///
/// * `src`: Source slice
/// * `src_stride`: Source slice stride
/// * `dst`: Destination slice
/// * `dst_stride`: Destination slice stride
/// * `width`: Image width
/// * `height`: Image height
///
/// returns: Result<(), YuvError>
///
pub fn bgr_to_rgb(
    src: &[u8],
    src_stride: u32,
    dst: &mut [u8],
    dst_stride: u32,
    width: u32,
    height: u32,
) -> Result<(), YuvError> {
    shuffle_impl::<u8, { YuvSourceChannels::Bgr as u8 }, { YuvSourceChannels::Rgb as u8 }, 8>(
        src, src_stride, dst, dst_stride, width, height,
    )
}

/// Converts BGR8 to RGBA8
///
/// # Arguments
///
/// * `src`: Source slice
/// * `src_stride`: Source slice stride
/// * `dst`: Destination slice
/// * `dst_stride`: Destination slice stride
/// * `width`: Image width
/// * `height`: Image height
///
/// returns: Result<(), YuvError>
///
pub fn bgr_to_rgba(
    src: &[u8],
    src_stride: u32,
    dst: &mut [u8],
    dst_stride: u32,
    width: u32,
    height: u32,
) -> Result<(), YuvError> {
    shuffle_impl::<u8, { YuvSourceChannels::Bgr as u8 }, { YuvSourceChannels::Rgba as u8 }, 8>(
        src, src_stride, dst, dst_stride, width, height,
    )
}

/// Converts BGR8 to BGRA8
///
/// # Arguments
///
/// * `src`: Source slice
/// * `src_stride`: Source slice stride
/// * `dst`: Destination slice
/// * `dst_stride`: Destination slice stride
/// * `width`: Image width
/// * `height`: Image height
///
/// returns: Result<(), YuvError>
///
pub fn bgr_to_bgra(
    src: &[u8],
    src_stride: u32,
    dst: &mut [u8],
    dst_stride: u32,
    width: u32,
    height: u32,
) -> Result<(), YuvError> {
    shuffle_impl::<u8, { YuvSourceChannels::Bgr as u8 }, { YuvSourceChannels::Bgra as u8 }, 8>(
        src, src_stride, dst, dst_stride, width, height,
    )
}

/// Converts RGB8 to RGBA8
///
/// # Arguments
///
/// * `src`: Source slice
/// * `src_stride`: Source slice stride
/// * `dst`: Destination slice
/// * `dst_stride`: Destination slice stride
/// * `width`: Image width
/// * `height`: Image height
///
/// returns: Result<(), YuvError>
///
pub fn rgb_to_rgba(
    src: &[u8],
    src_stride: u32,
    dst: &mut [u8],
    dst_stride: u32,
    width: u32,
    height: u32,
) -> Result<(), YuvError> {
    shuffle_impl::<u8, { YuvSourceChannels::Rgb as u8 }, { YuvSourceChannels::Rgba as u8 }, 8>(
        src, src_stride, dst, dst_stride, width, height,
    )
}

/// Converts BGRA to RGBA8
///
/// # Arguments
///
/// * `src`: Source slice
/// * `src_stride`: Source slice stride
/// * `dst`: Destination slice
/// * `dst_stride`: Destination slice stride
/// * `width`: Image width
/// * `height`: Image height
///
/// returns: Result<(), YuvError>
///
pub fn bgra_to_rgba(
    src: &[u8],
    src_stride: u32,
    dst: &mut [u8],
    dst_stride: u32,
    width: u32,
    height: u32,
) -> Result<(), YuvError> {
    shuffle_impl::<u8, { YuvSourceChannels::Bgra as u8 }, { YuvSourceChannels::Rgba as u8 }, 8>(
        src, src_stride, dst, dst_stride, width, height,
    )
}

/// Converts BGRA to RGB8
///
/// # Arguments
///
/// * `src`: Source slice
/// * `src_stride`: Source slice stride
/// * `dst`: Destination slice
/// * `dst_stride`: Destination slice stride
/// * `width`: Image width
/// * `height`: Image height
///
/// returns: Result<(), YuvError>
///
pub fn bgra_to_rgb(
    src: &[u8],
    src_stride: u32,
    dst: &mut [u8],
    dst_stride: u32,
    width: u32,
    height: u32,
) -> Result<(), YuvError> {
    shuffle_impl::<u8, { YuvSourceChannels::Bgra as u8 }, { YuvSourceChannels::Rgb as u8 }, 8>(
        src, src_stride, dst, dst_stride, width, height,
    )
}

/// Converts BGRA to RGB8
///
/// # Arguments
///
/// * `src`: Source slice
/// * `src_stride`: Source slice stride
/// * `dst`: Destination slice
/// * `dst_stride`: Destination slice stride
/// * `width`: Image width
/// * `height`: Image height
///
/// returns: Result<(), YuvError>
///
pub fn bgra_to_bgr(
    src: &[u8],
    src_stride: u32,
    dst: &mut [u8],
    dst_stride: u32,
    width: u32,
    height: u32,
) -> Result<(), YuvError> {
    shuffle_impl::<u8, { YuvSourceChannels::Bgra as u8 }, { YuvSourceChannels::Bgr as u8 }, 8>(
        src, src_stride, dst, dst_stride, width, height,
    )
}
