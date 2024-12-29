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
use crate::shuffle::ShuffleConverter;
use crate::sse::{
    _mm_load_deinterleave_half_rgbx, _mm_load_deinterleave_rgbx,
    _mm_store_interleave_half_rgb_for_yuv, _mm_store_interleave_rgb_for_yuv, _xx_load_si64,
};
use crate::yuv_support::YuvSourceChannels;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// This is default shuffling with interleaving and de-interleaving.
///
/// For the same channels count there is more fast approach on x86 with reshuffling table
pub(crate) struct ShuffleConverterSse<const SRC: u8, const DST: u8> {}

impl<const SRC: u8, const DST: u8> Default for ShuffleConverterSse<SRC, DST> {
    fn default() -> Self {
        ShuffleConverterSse {}
    }
}

impl<const SRC: u8, const DST: u8> ShuffleConverter<u8, SRC, DST>
    for ShuffleConverterSse<SRC, DST>
{
    fn convert(&self, src: &[u8], dst: &mut [u8], width: usize) {
        unsafe { shuffle_channels8_impl::<SRC, DST>(src, dst, width) }
    }
}

#[target_feature(enable = "sse4.1")]
unsafe fn shuffle_channels8_impl<const SRC: u8, const DST: u8>(
    src: &[u8],
    dst: &mut [u8],
    _: usize,
) {
    let src_channels: YuvSourceChannels = SRC.into();
    let dst_channels: YuvSourceChannels = DST.into();
    for (src, dst) in src
        .chunks_exact(16 * src_channels.get_channels_count())
        .zip(dst.chunks_exact_mut(16 * dst_channels.get_channels_count()))
    {
        let (a0, b0, c0, d0) = _mm_load_deinterleave_rgbx::<SRC>(src.as_ptr());
        _mm_store_interleave_rgb_for_yuv::<DST>(dst.as_mut_ptr(), a0, b0, c0, d0);
    }

    let src = src
        .chunks_exact(16 * src_channels.get_channels_count())
        .remainder();
    let dst = dst
        .chunks_exact_mut(16 * dst_channels.get_channels_count())
        .into_remainder();

    for (src, dst) in src
        .chunks_exact(8 * src_channels.get_channels_count())
        .zip(dst.chunks_exact_mut(8 * dst_channels.get_channels_count()))
    {
        let (a0, b0, c0, d0) = _mm_load_deinterleave_half_rgbx::<SRC>(src.as_ptr());
        _mm_store_interleave_half_rgb_for_yuv::<DST>(dst.as_mut_ptr(), a0, b0, c0, d0);
    }

    let src = src
        .chunks_exact(8 * src_channels.get_channels_count())
        .remainder();
    let dst = dst
        .chunks_exact_mut(8 * dst_channels.get_channels_count())
        .into_remainder();

    if src.len() > 0 && dst.len() > 0 {
        assert!(src.len() < 64);
        assert!(dst.len() < 64);
        let mut transient_src: [u8; 64] = [0; 64];
        let mut transient_dst: [u8; 64] = [0; 64];
        std::ptr::copy_nonoverlapping(src.as_ptr(), transient_src.as_mut_ptr(), src.len());
        let (a0, b0, c0, d0) = _mm_load_deinterleave_half_rgbx::<SRC>(transient_src.as_ptr());
        _mm_store_interleave_half_rgb_for_yuv::<DST>(transient_dst.as_mut_ptr(), a0, b0, c0, d0);
        std::ptr::copy_nonoverlapping(transient_dst.as_ptr(), dst.as_mut_ptr(), dst.len());
    }
}

/// This is shuffling only for 4 channels image
///
/// This is more fast method that just swaps channel positions
pub(crate) struct ShuffleQTableConverterSse<const SRC: u8, const DST: u8> {
    q_table: [u8; 16],
}

const RGBA_TO_BGRA_TABLE: [u8; 16] = [
    2,
    1,
    0,
    3,
    2 + 4,
    1 + 4,
    0 + 4,
    3 + 4,
    2 + 8,
    1 + 8,
    0 + 8,
    3 + 8,
    2 + 12,
    1 + 12,
    0 + 12,
    3 + 12,
];

impl<const SRC: u8, const DST: u8> ShuffleQTableConverterSse<SRC, DST> {
    pub(crate) fn create() -> Self {
        let src_channels: YuvSourceChannels = SRC.into();
        let dst_channels: YuvSourceChannels = DST.into();
        if src_channels.get_channels_count() != 4 || dst_channels.get_channels_count() != 4 {
            unimplemented!("Shuffle table implemented only for 4 channels");
        }
        let new_table: [u8; 16] = match src_channels {
            YuvSourceChannels::Rgb => unreachable!(),
            YuvSourceChannels::Rgba => match dst_channels {
                YuvSourceChannels::Rgb => unreachable!(),
                YuvSourceChannels::Rgba => RGBA_TO_BGRA_TABLE,
                YuvSourceChannels::Bgra => RGBA_TO_BGRA_TABLE,
                YuvSourceChannels::Bgr => unreachable!(),
            },
            YuvSourceChannels::Bgra => match dst_channels {
                YuvSourceChannels::Rgb => unreachable!(),
                YuvSourceChannels::Rgba => RGBA_TO_BGRA_TABLE,
                YuvSourceChannels::Bgra => RGBA_TO_BGRA_TABLE,
                YuvSourceChannels::Bgr => unreachable!(),
            },
            YuvSourceChannels::Bgr => unreachable!(),
        };
        ShuffleQTableConverterSse { q_table: new_table }
    }
}

impl<const SRC: u8, const DST: u8> ShuffleConverter<u8, SRC, DST>
    for ShuffleQTableConverterSse<SRC, DST>
{
    fn convert(&self, src: &[u8], dst: &mut [u8], width: usize) {
        unsafe { shuffle_qtable_channels8_impl::<SRC, DST>(src, dst, width, self.q_table) }
    }
}

#[target_feature(enable = "sse4.1")]
unsafe fn shuffle_qtable_channels8_impl<const SRC: u8, const DST: u8>(
    src: &[u8],
    dst: &mut [u8],
    _: usize,
    vq_table: [u8; 16],
) {
    let src_channels: YuvSourceChannels = SRC.into();
    let dst_channels: YuvSourceChannels = DST.into();
    assert_eq!(src_channels.get_channels_count(), 4);
    assert_eq!(dst_channels.get_channels_count(), 4);

    let q_table = _mm_loadu_si128(vq_table.as_ptr() as *const _);

    for (src, dst) in src.chunks_exact(16 * 4).zip(dst.chunks_exact_mut(16 * 4)) {
        let mut row_1 = _mm_loadu_si128(src.as_ptr() as *const __m128i);
        let mut row_2 = _mm_loadu_si128(src.as_ptr().add(16) as *const __m128i);
        let mut row_3 = _mm_loadu_si128(src.as_ptr().add(32) as *const __m128i);
        let mut row_4 = _mm_loadu_si128(src.as_ptr().add(48) as *const __m128i);

        row_1 = _mm_shuffle_epi8(row_1, q_table);
        row_2 = _mm_shuffle_epi8(row_2, q_table);
        row_3 = _mm_shuffle_epi8(row_3, q_table);
        row_4 = _mm_shuffle_epi8(row_4, q_table);

        _mm_storeu_si128(dst.as_mut_ptr() as *mut __m128i, row_1);
        _mm_storeu_si128(dst.as_mut_ptr().add(16) as *mut __m128i, row_2);
        _mm_storeu_si128(dst.as_mut_ptr().add(32) as *mut __m128i, row_3);
        _mm_storeu_si128(dst.as_mut_ptr().add(48) as *mut __m128i, row_4);
    }

    let src = src.chunks_exact(16 * 4).remainder();
    let dst = dst.chunks_exact_mut(16 * 4).into_remainder();

    for (src, dst) in src.chunks_exact(16).zip(dst.chunks_exact_mut(16)) {
        let mut row_1 = _mm_loadu_si128(src.as_ptr() as *const __m128i);
        row_1 = _mm_shuffle_epi8(row_1, q_table);
        _mm_storeu_si128(dst.as_mut_ptr() as *mut __m128i, row_1);
    }

    let src = src.chunks_exact(16).remainder();
    let dst = dst.chunks_exact_mut(16).into_remainder();

    for (src, dst) in src.chunks_exact(8).zip(dst.chunks_exact_mut(8)) {
        let mut row_1 = _xx_load_si64(src.as_ptr());
        row_1 = _mm_shuffle_epi8(row_1, q_table);
        _mm_storeu_si64(dst.as_mut_ptr(), row_1);
    }

    let src = src.chunks_exact(8).remainder();
    let dst = dst.chunks_exact_mut(8).into_remainder();

    if src.len() > 0 && dst.len() > 0 {
        assert!(src.len() < 16);
        assert!(dst.len() < 16);
        let mut transient_src: [u8; 16] = [0; 16];
        let mut transient_dst: [u8; 16] = [0; 16];
        std::ptr::copy_nonoverlapping(src.as_ptr(), transient_src.as_mut_ptr(), src.len());
        let mut row_1 = _xx_load_si64(transient_src.as_ptr());
        row_1 = _mm_shuffle_epi8(row_1, q_table);
        _mm_storeu_si64(transient_dst.as_mut_ptr(), row_1);
        std::ptr::copy_nonoverlapping(transient_dst.as_ptr(), dst.as_mut_ptr(), dst.len());
    }
}
