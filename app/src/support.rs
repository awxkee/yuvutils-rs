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
use std::fs::File;
use std::io::{Error, Read, Write};
use std::path::Path;
use yuv::{BufferStoreMut, YuvPlanarImageMut};

pub(crate) fn save_yuy2_image(
    filename: &str,
    width: usize,
    height: usize,
    yuy2_data: &[u8],
) -> std::io::Result<()> {
    let mut file = File::create(filename)?;
    file.write_all(yuy2_data)?;
    Ok(())
}

pub fn read_yuv420_16bit<P: AsRef<Path>>(
    path: P,
    width: usize,
    height: usize,
) -> Result<YuvPlanarImageMut<'static, u16>, Error> {
    let mut file = File::open(path)?;
    let frame_size = width * height;
    let chroma_size = (width / 2) * (height / 2);

    let mut y_buf = vec![0u8; frame_size * 2];
    let mut u_buf = vec![0u8; chroma_size * 2];
    let mut v_buf = vec![0u8; chroma_size * 2];

    file.read_exact(&mut y_buf)?;
    file.read_exact(&mut u_buf)?;
    file.read_exact(&mut v_buf)?;

    let y = y_buf
        .chunks_exact(2)
        .map(|b| u16::from_le_bytes([b[0], b[1]]))
        .collect::<Vec<u16>>();

    let u = u_buf
        .chunks_exact(2)
        .map(|b| u16::from_le_bytes([b[0], b[1]]))
        .collect::<Vec<u16>>();

    let v = v_buf
        .chunks_exact(2)
        .map(|b| u16::from_le_bytes([b[0], b[1]]))
        .collect::<Vec<u16>>();

    Ok(YuvPlanarImageMut {
        y_plane: BufferStoreMut::Owned(y),
        y_stride: width as u32,
        u_plane: BufferStoreMut::Owned(u),
        u_stride: (width as u32).div_ceil(2),
        v_plane: BufferStoreMut::Owned(v),
        v_stride: (width as u32).div_ceil(2),
        width: width as u32,
        height: height as u32,
    })
}
