/*
 * Copyright (c) Radzivon Bartoshyk, 09/2026. All rights reserved.
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

use crate::shuffle::ShuffleConverter;
use crate::yuv_support::YuvSourceChannels;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Shuffle implementation between 4-channel images only
pub(crate) struct ShuffleQTableConverterAvx512<const SRC: u8, const DST: u8> {
    q_table_avx: [u8; 16],
}

const RGBA_TO_BGRA_TABLE: [u8; 16] = [
    2,
    1,
    0,
    3,
    2 + 4,
    1 + 4,
    4,
    3 + 4,
    2 + 8,
    1 + 8,
    8,
    3 + 8,
    2 + 12,
    1 + 12,
    12,
    3 + 12,
];

const IDENTITY_TABLE: [u8; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];

impl<const SRC: u8, const DST: u8> ShuffleQTableConverterAvx512<SRC, DST> {
    pub(crate) fn create() -> Self {
        let src_channels: YuvSourceChannels = SRC.into();
        let dst_channels: YuvSourceChannels = DST.into();
        if src_channels.get_channels_count() != 4 || dst_channels.get_channels_count() != 4 {
            unimplemented!("Shuffle table implemented only for 4 channels");
        }
        let new_table_avx: [u8; 16] = match src_channels {
            YuvSourceChannels::Rgb => unreachable!(),
            YuvSourceChannels::Rgba => match dst_channels {
                YuvSourceChannels::Rgb => unreachable!(),
                YuvSourceChannels::Rgba => IDENTITY_TABLE,
                YuvSourceChannels::Bgra => RGBA_TO_BGRA_TABLE,
                YuvSourceChannels::Bgr => unreachable!(),
            },
            YuvSourceChannels::Bgra => match dst_channels {
                YuvSourceChannels::Rgb => unreachable!(),
                YuvSourceChannels::Rgba => RGBA_TO_BGRA_TABLE,
                YuvSourceChannels::Bgra => IDENTITY_TABLE,
                YuvSourceChannels::Bgr => unreachable!(),
            },
            YuvSourceChannels::Bgr => unreachable!(),
        };
        ShuffleQTableConverterAvx512 {
            q_table_avx: new_table_avx,
        }
    }
}

impl<const SRC: u8, const DST: u8> ShuffleConverter<u8, SRC, DST>
    for ShuffleQTableConverterAvx512<SRC, DST>
{
    fn convert(&self, src: &[u8], dst: &mut [u8], width: usize) {
        unsafe { shuffle_qtable_channels8_avx_impl::<SRC, DST>(src, dst, width, self.q_table_avx) }
    }
}

#[target_feature(enable = "avx512bw")]
unsafe fn shuffle_qtable_channels8_avx_impl<const SRC: u8, const DST: u8>(
    src: &[u8],
    dst: &mut [u8],
    _: usize,
    vq_table_avx: [u8; 16],
) {
    let src_channels: YuvSourceChannels = SRC.into();
    let dst_channels: YuvSourceChannels = DST.into();
    assert_eq!(src_channels.get_channels_count(), 4);
    assert_eq!(dst_channels.get_channels_count(), 4);

    let q_table_avx = _mm512_broadcast_i32x4(_mm_loadu_si128(vq_table_avx.as_ptr() as *const _));

    for (src, dst) in src.chunks_exact(64 * 4).zip(dst.chunks_exact_mut(64 * 4)) {
        let mut row_1 = _mm512_loadu_si512(src.as_ptr() as *const __m512i);
        let mut row_2 = _mm512_loadu_si512(src.as_ptr().add(64) as *const __m512i);
        let mut row_3 = _mm512_loadu_si512(src.as_ptr().add(128) as *const __m512i);
        let mut row_4 = _mm512_loadu_si512(src.as_ptr().add(192) as *const __m512i);

        row_1 = _mm512_shuffle_epi8(row_1, q_table_avx);
        row_2 = _mm512_shuffle_epi8(row_2, q_table_avx);
        row_3 = _mm512_shuffle_epi8(row_3, q_table_avx);
        row_4 = _mm512_shuffle_epi8(row_4, q_table_avx);

        _mm512_storeu_si512(dst.as_mut_ptr() as *mut __m512i, row_1);
        _mm512_storeu_si512(dst.as_mut_ptr().add(64) as *mut __m512i, row_2);
        _mm512_storeu_si512(dst.as_mut_ptr().add(128) as *mut __m512i, row_3);
        _mm512_storeu_si512(dst.as_mut_ptr().add(192) as *mut __m512i, row_4);
    }

    let src = src.chunks_exact(64 * 4).remainder();
    let dst = dst.chunks_exact_mut(64 * 4).into_remainder();

    for (src, dst) in src.chunks_exact(64).zip(dst.chunks_exact_mut(64)) {
        let mut row_1 = _mm512_loadu_si512(src.as_ptr() as *const __m512i);

        row_1 = _mm512_shuffle_epi8(row_1, q_table_avx);

        _mm512_storeu_si512(dst.as_mut_ptr() as *mut __m512i, row_1);
    }

    let src = src.chunks_exact(64).remainder();
    let dst = dst.chunks_exact_mut(64).into_remainder();

    if !src.is_empty() && !dst.is_empty() {
        assert!(src.len() < 64);
        assert!(dst.len() < 64);

        let mut row_1 = _mm512_maskz_loadu_epi8((1u64 << src.len()) - 1, src.as_ptr() as *const i8);
        row_1 = _mm512_shuffle_epi8(row_1, q_table_avx);
        _mm512_mask_storeu_epi8(dst.as_mut_ptr() as *mut _, (1u64 << dst.len()) - 1, row_1);
    }
}
