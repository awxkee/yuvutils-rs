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
use crate::yuv_support::{CbCrForwardTransform, CbCrInverseTransform, YuvChromaRange};

/// Handles the tail (remainder) of a 420 SIMD encode loop.
///
/// Allocates stack buffers, copies remaining source pixels in (replicating the
/// last pixel for odd-width subsampling), invokes the caller's encode block,
/// then copies the Y/U/V results back to the destination planes.
///
/// Uses `MaybeUninit` for all buffers:
/// - Source buffers are zeroed because SIMD reads the full vector width.
/// - Output buffers are left uninitialized because the encode writes all bytes.
#[allow(unused_macros)]
macro_rules! tail_420 {
    (
        $stride:expr,
        $cx:expr, $ux:expr, $width:expr, $channels:expr,
        $rgba0:expr, $rgba1:expr,
        $y_plane0:expr, $y_plane1:expr, $u_plane:expr, $v_plane:expr,
        |$sb0:ident, $sb1:ident, $yb0:ident, $yb1:ident, $ub:ident, $vb:ident| $encode:expr
    ) => {
        if $cx < $width {
            let diff = $width - $cx;
            debug_assert!(diff <= $stride);

            // SAFETY: Zeroed via `MaybeUninit::zeroed()`. SIMD reads the full STRIDE
            // even when fewer than STRIDE pixels remain; zeroed padding ensures no UB.
            // Actual pixel data is overwritten by `copy_nonoverlapping`.
            let mut src_buf0: [std::mem::MaybeUninit<u8>; $stride * 4] =
                [std::mem::MaybeUninit::zeroed(); $stride * 4];
            let mut src_buf1: [std::mem::MaybeUninit<u8>; $stride * 4] =
                [std::mem::MaybeUninit::zeroed(); $stride * 4];

            // SAFETY: Left uninitialized. The encode function writes all STRIDE bytes
            // via SIMD stores. Only `diff` (Y) or `diff.div_ceil(2)` (UV) bytes are
            // read back by `copy_nonoverlapping`, all of which were written.
            let mut y_buf0: [std::mem::MaybeUninit<u8>; $stride] =
                [std::mem::MaybeUninit::uninit(); $stride];
            let mut y_buf1: [std::mem::MaybeUninit<u8>; $stride] =
                [std::mem::MaybeUninit::uninit(); $stride];
            let mut u_buf: [std::mem::MaybeUninit<u8>; $stride] =
                [std::mem::MaybeUninit::uninit(); $stride];
            let mut v_buf: [std::mem::MaybeUninit<u8>; $stride] =
                [std::mem::MaybeUninit::uninit(); $stride];

            std::ptr::copy_nonoverlapping(
                $rgba0.get_unchecked($cx * $channels..).as_ptr(),
                src_buf0.as_mut_ptr().cast::<u8>(),
                diff * $channels,
            );
            std::ptr::copy_nonoverlapping(
                $rgba1.get_unchecked($cx * $channels..).as_ptr(),
                src_buf1.as_mut_ptr().cast::<u8>(),
                diff * $channels,
            );

            if diff % 2 != 0 {
                let lst = ($width - 1) * $channels;
                let last0 = $rgba0.get_unchecked(lst..(lst + $channels));
                let last1 = $rgba1.get_unchecked(lst..(lst + $channels));
                let dvb = diff * $channels;
                let d0 = std::slice::from_raw_parts_mut(
                    src_buf0.as_mut_ptr().add(dvb).cast::<u8>(),
                    $channels,
                );
                let d1 = std::slice::from_raw_parts_mut(
                    src_buf1.as_mut_ptr().add(dvb).cast::<u8>(),
                    $channels,
                );
                for (d, s) in d0.iter_mut().zip(last0) {
                    *d = *s;
                }
                for (d, s) in d1.iter_mut().zip(last1) {
                    *d = *s;
                }
            }

            let $sb0 = std::slice::from_raw_parts(src_buf0.as_ptr().cast::<u8>(), $stride * 4);
            let $sb1 = std::slice::from_raw_parts(src_buf1.as_ptr().cast::<u8>(), $stride * 4);
            let $yb0 = std::slice::from_raw_parts_mut(y_buf0.as_mut_ptr().cast::<u8>(), $stride);
            let $yb1 = std::slice::from_raw_parts_mut(y_buf1.as_mut_ptr().cast::<u8>(), $stride);
            let $ub = std::slice::from_raw_parts_mut(u_buf.as_mut_ptr().cast::<u8>(), $stride);
            let $vb = std::slice::from_raw_parts_mut(v_buf.as_mut_ptr().cast::<u8>(), $stride);

            $encode;

            std::ptr::copy_nonoverlapping(
                y_buf0.as_ptr().cast::<u8>(),
                $y_plane0.get_unchecked_mut($cx..).as_mut_ptr(),
                diff,
            );
            std::ptr::copy_nonoverlapping(
                y_buf1.as_ptr().cast::<u8>(),
                $y_plane1.get_unchecked_mut($cx..).as_mut_ptr(),
                diff,
            );

            $cx += diff;

            let hv = diff.div_ceil(2);
            std::ptr::copy_nonoverlapping(
                u_buf.as_ptr().cast::<u8>(),
                $u_plane.get_unchecked_mut($ux..).as_mut_ptr(),
                hv,
            );
            std::ptr::copy_nonoverlapping(
                v_buf.as_ptr().cast::<u8>(),
                $v_plane.get_unchecked_mut($ux..).as_mut_ptr(),
                hv,
            );

            $ux += hv;
        }
    };
}
#[allow(unused_imports)]
pub(crate) use tail_420;

#[allow(dead_code)]
#[derive(Debug, Copy, Clone, Ord, PartialOrd, Eq, PartialEq)]
pub(crate) struct ProcessedOffset {
    pub(crate) cx: usize,
    pub(crate) ux: usize,
}

pub(crate) trait WideRowInversionHandler<V, K> {
    fn handle_row(
        &self,
        y_plane: &[V],
        u_plane: &[V],
        v_plane: &[V],
        rgba: &mut [V],
        width: u32,
        chroma: YuvChromaRange,
        transform: &CbCrInverseTransform<K>,
    ) -> ProcessedOffset;
}

pub(crate) trait WideRowAlphaInversionHandler<V, T, K> {
    fn handle_row(
        &self,
        y_plane: &[V],
        u_plane: &[V],
        v_plane: &[V],
        a_plane: &[V],
        rgba: &mut [T],
        width: u32,
        chroma: YuvChromaRange,
        transform: &CbCrInverseTransform<K>,
        use_premultiplied_alpha: bool,
    ) -> ProcessedOffset;
}

#[cfg(feature = "nightly_f16")]
pub(crate) trait WideDRowInversionHandler<V, T, K> {
    fn handle_row(
        &self,
        y_plane: &[V],
        u_plane: &[V],
        v_plane: &[V],
        rgba: &mut [T],
        width: u32,
        chroma: YuvChromaRange,
        transform: &CbCrInverseTransform<K>,
    ) -> ProcessedOffset;
}

#[cfg(feature = "nightly_f16")]
pub(crate) trait WideDAlphaRowInversionHandler<V, T, K> {
    fn handle_row(
        &self,
        y_plane: &[V],
        u_plane: &[V],
        v_plane: &[V],
        a_plane: &[V],
        rgba: &mut [T],
        width: u32,
        chroma: YuvChromaRange,
        transform: &CbCrInverseTransform<K>,
    ) -> ProcessedOffset;
}

pub(crate) trait WideRow420InversionHandler<V, K> {
    fn handle_row(
        &self,
        y0_plane: &[V],
        y1_plane: &[V],
        u_plane: &[V],
        v_plane: &[V],
        rgba0: &mut [V],
        rgba1: &mut [V],
        width: u32,
        chroma: YuvChromaRange,
        transform: &CbCrInverseTransform<K>,
    ) -> ProcessedOffset;
}

pub(crate) trait RowBiPlanarInversionHandler<V, K> {
    fn handle_row(
        &self,
        y_plane: &[V],
        uv_plane: &[V],
        rgba: &mut [V],
        width: u32,
        chroma: YuvChromaRange,
        transform: &CbCrInverseTransform<K>,
    ) -> ProcessedOffset;
}

pub(crate) trait RowDBiPlanarInversionHandler<V, T, K> {
    fn handle_row(
        &self,
        y_plane: &[V],
        uv_plane: &[V],
        rgba: &mut [T],
        width: u32,
        chroma: YuvChromaRange,
        transform: &CbCrInverseTransform<K>,
    ) -> ProcessedOffset;
}

pub(crate) trait RowBiPlanarInversion420Handler<V, K> {
    fn handle_row(
        &self,
        y_plane0: &[V],
        y_plane1: &[V],
        uv_plane: &[V],
        rgba0: &mut [V],
        rgba1: &mut [V],
        width: u32,
        chroma: YuvChromaRange,
        transform: &CbCrInverseTransform<K>,
    ) -> ProcessedOffset;
}

pub(crate) trait WideRowForwardHandler<V, K> {
    fn handle_row(
        &self,
        y_plane: &mut [V],
        u_plane: &mut [V],
        v_plane: &mut [V],
        rgba: &[V],
        width: u32,
        chroma: YuvChromaRange,
        transform: &CbCrForwardTransform<K>,
    ) -> ProcessedOffset;
}

pub(crate) trait WideRowForward420Handler<V, K> {
    fn handle_row(
        &self,
        y_plane0: &mut [V],
        y_plane1: &mut [V],
        u_plane: &mut [V],
        v_plane: &mut [V],
        rgba0: &[V],
        rgba1: &[V],
        width: u32,
        chroma: YuvChromaRange,
        transform: &CbCrForwardTransform<K>,
    ) -> ProcessedOffset;
}

pub(crate) trait WideRowForwardBiPlanar420Handler<V, K> {
    fn handle_rows(
        &self,
        rgba0: &[V],
        rgba1: &[V],
        y_plane0: &mut [V],
        y_plane1: &mut [V],
        uv_plane: &mut [V],
        width: u32,
        chroma: YuvChromaRange,
        transform: &CbCrForwardTransform<K>,
    ) -> ProcessedOffset;
}

pub(crate) trait WideRowForwardBiPlanarHandler<V, K> {
    fn handle_row(
        &self,
        rgba: &[V],
        y_plane: &mut [V],
        uv_plane: &mut [V],
        width: u32,
        chroma: YuvChromaRange,
        transform: &CbCrForwardTransform<K>,
    ) -> ProcessedOffset;
}
