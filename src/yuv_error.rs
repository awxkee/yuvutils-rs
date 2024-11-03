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
use crate::yuv_support::YuvChromaSample;
use std::error::Error;
use std::fmt::{Display, Formatter};

#[derive(Debug, Copy, Clone, Ord, PartialOrd, Eq, PartialEq)]
pub struct MismatchedSize {
    pub expected: usize,
    pub received: usize,
}

#[derive(Debug)]
pub enum YuvError {
    DestinationSizeMismatch(MismatchedSize),
    MinimumDestinationSizeMismatch(MismatchedSize),
    PointerOverflow,
    ZeroBaseSize,
    LumaPlaneSizeMismatch(MismatchedSize),
    LumaPlaneMinimumSizeMismatch(MismatchedSize),
}

impl Display for YuvError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            YuvError::LumaPlaneSizeMismatch(size) => f.write_fmt(format_args!(
                "Luma plane have invalid size, it must be {}, but it was {}",
                size.expected, size.received
            )),
            YuvError::LumaPlaneMinimumSizeMismatch(size) => f.write_fmt(format_args!(
                "Luma plane have invalid size, it must be at least {}, but it was {}",
                size.expected, size.received
            )),
            YuvError::PointerOverflow => f.write_str("Image size overflow pointer capabilities"),
            YuvError::ZeroBaseSize => f.write_str("Zero sized images is not supported"),
            YuvError::DestinationSizeMismatch(size) => f.write_fmt(format_args!(
                "Destination size mismatch: expected={}, received={}",
                size.expected, size.received
            )),
            YuvError::MinimumDestinationSizeMismatch(size) => f.write_fmt(format_args!(
                "Destination must have size at least {} but it is {}",
                size.expected, size.received
            )),
        }
    }
}

impl Error for YuvError {}

#[inline]
pub(crate) fn check_overflow_v2(v0: usize, v1: usize) -> Result<(), YuvError> {
    let (_, overflow) = v0.overflowing_mul(v1);
    if overflow {
        return Err(YuvError::PointerOverflow);
    }
    Ok(())
}

#[inline]
pub(crate) fn check_overflow_v3(v0: usize, v1: usize, v2: usize) -> Result<(), YuvError> {
    let (product0, overflow) = v0.overflowing_mul(v1);
    if overflow {
        return Err(YuvError::PointerOverflow);
    }
    let (_, overflow) = product0.overflowing_mul(v2);
    if overflow {
        return Err(YuvError::PointerOverflow);
    }
    Ok(())
}

#[inline]
pub(crate) fn check_rgba_destination<V>(
    arr: &[V],
    rgba_stride: u32,
    width: u32,
    height: u32,
    channels: usize,
) -> Result<(), YuvError> {
    check_overflow_v3(width as usize, height as usize, channels)?;
    if arr.len() != rgba_stride as usize * height as usize {
        return Err(YuvError::DestinationSizeMismatch(MismatchedSize {
            expected: rgba_stride as usize * height as usize,
            received: arr.len(),
        }));
    }
    if (rgba_stride as usize * height as usize) < (width as usize * height as usize * channels) {
        return Err(YuvError::MinimumDestinationSizeMismatch(MismatchedSize {
            expected: width as usize * height as usize * channels,
            received: rgba_stride as usize * height as usize,
        }));
    }
    Ok(())
}

#[inline]
pub(crate) fn check_y8_channel<V>(
    data: &[V],
    stride: u32,
    width: u32,
    height: u32,
) -> Result<(), YuvError> {
    check_overflow_v2(stride as usize, height as usize)?;
    check_overflow_v2(width as usize, height as usize)?;
    if (stride as usize * height as usize) < (width as usize * height as usize) {
        return Err(YuvError::LumaPlaneMinimumSizeMismatch(MismatchedSize {
            expected: width as usize * height as usize,
            received: stride as usize * height as usize,
        }));
    }
    if stride as usize * height as usize != data.len() {
        return Err(YuvError::LumaPlaneSizeMismatch(MismatchedSize {
            expected: stride as usize * height as usize,
            received: data.len(),
        }));
    }
    Ok(())
}

#[inline]
pub(crate) fn check_chroma_channel(
    data: &[u8],
    stride: u32,
    image_width: u32,
    image_height: u32,
    sampling: YuvChromaSample,
) -> Result<(), YuvError> {
    let chroma_min_width = match sampling {
        YuvChromaSample::YUV420 | YuvChromaSample::YUV422 => (image_width + 1) / 2,
        YuvChromaSample::YUV444 => image_width,
    };
    let chroma_height = match sampling {
        YuvChromaSample::YUV420 => (image_height + 1) / 2,
        YuvChromaSample::YUV422 | YuvChromaSample::YUV444 => image_height,
    };
    check_overflow_v2(stride as usize, chroma_height as usize)?;
    check_overflow_v2(chroma_min_width as usize, chroma_height as usize)?;
    if (stride as usize * chroma_height as usize)
        < (chroma_min_width as usize * chroma_height as usize)
    {
        return Err(YuvError::LumaPlaneMinimumSizeMismatch(MismatchedSize {
            expected: chroma_min_width as usize * chroma_height as usize,
            received: stride as usize * chroma_height as usize,
        }));
    }
    if stride as usize * chroma_height as usize != data.len() {
        return Err(YuvError::LumaPlaneSizeMismatch(MismatchedSize {
            expected: stride as usize * chroma_height as usize,
            received: data.len(),
        }));
    }
    Ok(())
}
