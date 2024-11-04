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
use crate::yuv_error::{check_chroma_channel, check_interleaved_chroma_channel, check_y8_channel};
use crate::yuv_support::YuvChromaSample;
use crate::YuvError;
use std::fmt::Debug;

#[derive(Debug)]
pub enum BufferStoreMut<'a, T: Copy + Debug> {
    Borrowed(&'a mut [T]),
    Owned(Vec<T>),
}

impl<T: Copy + Debug> BufferStoreMut<'_, T> {
    pub fn borrow(&self) -> &[T] {
        match self {
            Self::Borrowed(p_ref) => p_ref,
            Self::Owned(vec) => vec,
        }
    }

    pub fn as_mut(&mut self) -> &mut [T] {
        match self {
            Self::Borrowed(p_ref) => p_ref,
            Self::Owned(vec) => vec,
        }
    }
}

#[derive(Debug, Clone)]
/// Non representation of Bi-Planar YUV image
pub struct YuvBiPlanarImage<'a, T>
where
    T: Copy + Debug,
{
    pub y_plane: &'a [T],
    /// Stride here always means Elements per row.
    pub y_stride: u32,
    pub uv_plane: &'a [T],
    /// Stride here always means Elements per row.
    pub uv_stride: u32,
    pub width: u32,
    pub height: u32,
}

impl<T> YuvBiPlanarImage<'_, T>
where
    T: Copy + Debug,
{
    pub fn check_constraints(&self, subsampling: YuvChromaSample) -> Result<(), YuvError> {
        check_y8_channel(self.y_plane, self.y_stride, self.width, self.height)?;
        check_interleaved_chroma_channel(
            self.uv_plane,
            self.uv_stride,
            self.width,
            self.height,
            subsampling,
        )?;
        Ok(())
    }
}

#[derive(Debug)]
/// Mutable representation of Bi-Planar YUV image
pub struct YuvBiPlanarImageMut<'a, T>
where
    T: Copy + Debug,
{
    pub y_plane: BufferStoreMut<'a, T>,
    /// Stride here always means Elements per row.
    pub y_stride: u32,
    pub uv_plane: BufferStoreMut<'a, T>,
    /// Stride here always means Elements per row.
    pub uv_stride: u32,
    pub width: u32,
    pub height: u32,
}

impl<T> YuvBiPlanarImageMut<'_, T>
where
    T: Copy + Debug,
{
    pub fn check_constraints(&self, subsampling: YuvChromaSample) -> Result<(), YuvError> {
        check_y8_channel(
            self.y_plane.borrow(),
            self.y_stride,
            self.width,
            self.height,
        )?;
        check_interleaved_chroma_channel(
            self.uv_plane.borrow(),
            self.uv_stride,
            self.width,
            self.height,
            subsampling,
        )?;
        Ok(())
    }
}

impl<'a, T> YuvBiPlanarImageMut<'a, T>
where
    T: Default + Clone + Copy + Debug,
{
    /// Allocates mutable target Bi-Planar image with required chroma subsampling
    pub fn alloc(width: u32, height: u32, subsampling: YuvChromaSample) -> Self {
        let chroma_width = match subsampling {
            YuvChromaSample::YUV420 | YuvChromaSample::YUV422 => ((width as usize + 1) / 2) * 2,
            YuvChromaSample::YUV444 => width as usize * 2,
        };
        let chroma_height = match subsampling {
            YuvChromaSample::YUV420 => (height as usize + 1) / 2,
            YuvChromaSample::YUV422 | YuvChromaSample::YUV444 => height as usize,
        };
        let y_target = vec![T::default(); width as usize * height as usize];
        let chroma_target = vec![T::default(); chroma_width * chroma_height];
        YuvBiPlanarImageMut {
            y_plane: BufferStoreMut::Owned(y_target),
            y_stride: width,
            uv_plane: BufferStoreMut::Owned(chroma_target),
            uv_stride: chroma_width as u32,
            width,
            height,
        }
    }

    pub fn to_fixed(&'a self) -> YuvBiPlanarImage<'a, T> {
        YuvBiPlanarImage {
            y_plane: self.y_plane.borrow(),
            y_stride: self.y_stride,
            uv_plane: self.uv_plane.borrow(),
            uv_stride: self.uv_stride,
            width: self.width,
            height: self.height,
        }
    }
}

impl<'a, T> YuvBiPlanarImage<'a, T>
where
    T: Default + Clone + Copy + Debug,
{
    pub fn from_mut(bi_planar_mut: &'a YuvBiPlanarImageMut<T>) -> Self {
        YuvBiPlanarImage::<'a, T> {
            y_plane: bi_planar_mut.y_plane.borrow(),
            y_stride: bi_planar_mut.y_stride,
            uv_plane: bi_planar_mut.uv_plane.borrow(),
            uv_stride: bi_planar_mut.uv_stride,
            width: bi_planar_mut.width,
            height: bi_planar_mut.height,
        }
    }
}

#[derive(Debug)]
/// Represents YUV gray non-mutable image
pub struct YuvGrayImage<'a, T>
where
    T: Copy + Debug,
{
    pub y_plane: &'a [T],
    /// Stride here always means Elements per row.
    pub y_stride: u32,
    pub width: u32,
    pub height: u32,
}

impl<T> YuvGrayImage<'_, T>
where
    T: Copy + Debug,
{
    pub fn check_constraints(&self) -> Result<(), YuvError> {
        check_y8_channel(self.y_plane, self.y_stride, self.width, self.height)?;
        Ok(())
    }
}

#[derive(Debug)]
/// Represents YUV gray mutable image
pub struct YuvGrayImageMut<'a, T>
where
    T: Copy + Debug,
{
    pub y_plane: BufferStoreMut<'a, T>,
    /// Stride here always means Elements per row.
    pub y_stride: u32,
    pub width: u32,
    pub height: u32,
}

impl<'a, T> YuvGrayImageMut<'a, T>
where
    T: Copy + Debug,
{
    pub fn check_constraints(&self) -> Result<(), YuvError> {
        check_y8_channel(
            self.y_plane.borrow(),
            self.y_stride,
            self.width,
            self.height,
        )?;
        Ok(())
    }

    pub fn to_fixed(&'a self) -> YuvGrayImage<'a, T> {
        YuvGrayImage {
            y_plane: self.y_plane.borrow(),
            y_stride: self.y_stride,
            width: self.width,
            height: self.height,
        }
    }
}

impl<T> YuvGrayImageMut<'_, T>
where
    T: Copy + Debug + Clone + Default,
{
    /// Allocates mutable target gray image
    pub fn alloc(width: u32, height: u32) -> Self {
        let y_target = vec![T::default(); width as usize * height as usize];
        Self {
            y_plane: BufferStoreMut::Owned(y_target),
            y_stride: width,
            width,
            height,
        }
    }
}

#[derive(Debug)]
/// Represents YUV gray with alpha non-mutable image
pub struct YuvGrayAlphaImage<'a, T>
where
    T: Copy + Debug,
{
    pub y_plane: &'a [T],
    /// Stride here always means Elements per row.
    pub y_stride: u32,
    pub a_plane: &'a [T],
    /// Stride here always means Elements per row.
    pub a_stride: u32,
    pub width: u32,
    pub height: u32,
}

impl<T> YuvGrayAlphaImage<'_, T>
where
    T: Copy + Debug,
{
    pub fn check_constraints(&self) -> Result<(), YuvError> {
        check_y8_channel(self.y_plane, self.y_stride, self.width, self.height)?;
        check_y8_channel(self.a_plane, self.a_stride, self.width, self.height)?;
        Ok(())
    }
}

#[derive(Debug, Clone)]
/// Non-mutable representation of Bi-Planar YUV image
pub struct YuvPlanarImage<'a, T>
where
    T: Copy + Debug,
{
    pub y_plane: &'a [T],
    /// Stride here always means Elements per row.
    pub y_stride: u32,
    pub u_plane: &'a [T],
    /// Stride here always means Elements per row.
    pub u_stride: u32,
    pub v_plane: &'a [T],
    /// Stride here always means Elements per row.
    pub v_stride: u32,
    pub width: u32,
    pub height: u32,
}

impl<T> YuvPlanarImage<'_, T>
where
    T: Copy + Debug,
{
    pub fn check_constraints(&self, subsampling: YuvChromaSample) -> Result<(), YuvError> {
        check_y8_channel(self.y_plane, self.y_stride, self.width, self.height)?;
        check_chroma_channel(
            self.u_plane,
            self.u_stride,
            self.width,
            self.height,
            subsampling,
        )?;
        check_chroma_channel(
            self.v_plane,
            self.v_stride,
            self.width,
            self.height,
            subsampling,
        )?;
        Ok(())
    }
}

#[derive(Debug)]
/// Mutable of Bi-Planar YUV image
pub struct YuvPlanarImageMut<'a, T>
where
    T: Copy + Debug,
{
    pub y_plane: BufferStoreMut<'a, T>,
    /// Stride here always means Elements per row.
    pub y_stride: u32,
    pub u_plane: BufferStoreMut<'a, T>,
    /// Stride here always means Elements per row.
    pub u_stride: u32,
    pub v_plane: BufferStoreMut<'a, T>,
    /// Stride here always means Elements per row.
    pub v_stride: u32,
    pub width: u32,
    pub height: u32,
}

impl<T> YuvPlanarImageMut<'_, T>
where
    T: Copy + Debug,
{
    pub fn check_constraints(&self, subsampling: YuvChromaSample) -> Result<(), YuvError> {
        check_y8_channel(
            self.y_plane.borrow(),
            self.y_stride,
            self.width,
            self.height,
        )?;
        check_chroma_channel(
            self.u_plane.borrow(),
            self.u_stride,
            self.width,
            self.height,
            subsampling,
        )?;
        check_chroma_channel(
            self.v_plane.borrow(),
            self.v_stride,
            self.width,
            self.height,
            subsampling,
        )?;
        Ok(())
    }
}

impl<'a, T> YuvPlanarImageMut<'a, T>
where
    T: Default + Clone + Copy + Debug,
{
    /// Allocates mutable target Bi-Planar image with required chroma subsampling
    pub fn alloc(width: u32, height: u32, subsampling: YuvChromaSample) -> Self {
        let chroma_width = match subsampling {
            YuvChromaSample::YUV420 | YuvChromaSample::YUV422 => (width as usize + 1) / 2,
            YuvChromaSample::YUV444 => width as usize,
        };
        let chroma_height = match subsampling {
            YuvChromaSample::YUV420 => (height as usize + 1) / 2,
            YuvChromaSample::YUV422 | YuvChromaSample::YUV444 => height as usize,
        };
        let y_target = vec![T::default(); width as usize * height as usize];
        let u_target = vec![T::default(); chroma_width * chroma_height];
        let v_target = vec![T::default(); chroma_width * chroma_height];
        Self {
            y_plane: BufferStoreMut::Owned(y_target),
            y_stride: width,
            u_plane: BufferStoreMut::Owned(u_target),
            u_stride: chroma_width as u32,
            v_plane: BufferStoreMut::Owned(v_target),
            v_stride: chroma_width as u32,
            width,
            height,
        }
    }

    pub fn to_fixed(&'a self) -> YuvPlanarImage<'a, T> {
        YuvPlanarImage {
            y_plane: self.y_plane.borrow(),
            y_stride: self.y_stride,
            u_plane: self.u_plane.borrow(),
            u_stride: self.u_stride,
            v_plane: self.v_plane.borrow(),
            v_stride: self.v_stride,
            width: self.width,
            height: self.height,
        }
    }
}
