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

#![no_main]

use libfuzzer_sys::fuzz_target;
use yuv::{bgr_to_rgb, rgba_to_bgr, rgba_to_bgra};

fuzz_target!(|data: (u8, u8, bool, bool)| {
    fuzz_shuffler(data.0, data.1, data.2, data.3);
});

fn fuzz_shuffler(i_width: u8, i_height: u8, src_rgba: bool, dst_rgba: bool) {
    if i_height == 0 || i_width == 0 {
        return;
    }
    let src_chans = if src_rgba { 4 } else { 3 };
    let dst_chans = if dst_rgba { 4 } else { 3 };
    let src_data = vec![126u8; src_chans * i_width as usize * i_height as usize];
    let mut dst_data = vec![50u8; dst_chans * i_width as usize * i_height as usize];

    let shuffler = if src_rgba && dst_rgba {
        rgba_to_bgra
    } else if src_rgba {
        rgba_to_bgr
    } else {
        bgr_to_rgb
    };
    shuffler(
        &src_data,
        src_chans as u32 * i_width as u32,
        &mut dst_data,
        dst_chans as u32 * i_width as u32,
        i_width as u32,
        i_height as u32,
    )
    .unwrap();
}
