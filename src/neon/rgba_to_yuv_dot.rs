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

use crate::internals::ProcessedOffset;
use crate::yuv_support::{
    CbCrForwardTransform, YuvChromaRange, YuvChromaSubsampling, YuvSourceChannels,
};
use std::arch::aarch64::*;

#[target_feature(enable = "i8mm")]
pub(crate) unsafe fn neon_rgba_to_yuv_dot_rgba<const ORIGIN_CHANNELS: u8, const SAMPLING: u8>(
    transform: &CbCrForwardTransform<i32>,
    range: &YuvChromaRange,
    y_plane: &mut [u8],
    u_plane: &mut [u8],
    v_plane: &mut [u8],
    rgba: &[u8],
    start_cx: usize,
    start_ux: usize,
    width: usize,
) -> ProcessedOffset {
    let chroma_subsampling: YuvChromaSubsampling = SAMPLING.into();
    let source_channels: YuvSourceChannels = ORIGIN_CHANNELS.into();
    assert!(
        source_channels == YuvSourceChannels::Rgba || source_channels == YuvSourceChannels::Bgra
    );
    let channels = source_channels.get_channels_count();

    let y_ptr = y_plane;
    let u_ptr = u_plane;
    let v_ptr = v_plane;

    const A_E: i32 = 7;
    let y_bias = vdupq_n_s32(range.bias_y as i32 * (1 << A_E) + (1 << (A_E - 1)) - 1);
    let uv_bias = vdupq_n_s32(range.bias_uv as i32 * (1 << A_E) + (1 << (A_E - 1)) - 1);

    let weights_yr: [i8; 16] = [
        transform.yr as i8,
        transform.yg as i8,
        transform.yb as i8,
        0,
        transform.yr as i8,
        transform.yg as i8,
        transform.yb as i8,
        0,
        transform.yr as i8,
        transform.yg as i8,
        transform.yb as i8,
        0,
        transform.yr as i8,
        transform.yg as i8,
        transform.yb as i8,
        0,
    ];
    let weights_yr_bgra: [i8; 16] = [
        transform.yb as i8,
        transform.yg as i8,
        transform.yr as i8,
        0,
        transform.yb as i8,
        transform.yg as i8,
        transform.yr as i8,
        0,
        transform.yb as i8,
        transform.yg as i8,
        transform.yr as i8,
        0,
        transform.yb as i8,
        transform.yg as i8,
        transform.yr as i8,
        0,
    ];

    let weights_cb_rgba: [i8; 16] = [
        transform.cb_r as i8,
        transform.cb_g as i8,
        transform.cb_b as i8,
        0,
        transform.cb_r as i8,
        transform.cb_g as i8,
        transform.cb_b as i8,
        0,
        transform.cb_r as i8,
        transform.cb_g as i8,
        transform.cb_b as i8,
        0,
        transform.cb_r as i8,
        transform.cb_g as i8,
        transform.cb_b as i8,
        0,
    ];

    let weights_cb_bgra: [i8; 16] = [
        transform.cb_b as i8,
        transform.cb_g as i8,
        transform.cb_r as i8,
        0,
        transform.cb_b as i8,
        transform.cb_g as i8,
        transform.cb_r as i8,
        0,
        transform.cb_b as i8,
        transform.cb_g as i8,
        transform.cb_r as i8,
        0,
        transform.cb_b as i8,
        transform.cb_g as i8,
        transform.cb_r as i8,
        0,
    ];

    let weights_cr_rgba: [i8; 16] = [
        transform.cr_r as i8,
        transform.cr_g as i8,
        transform.cr_b as i8,
        0,
        transform.cr_r as i8,
        transform.cr_g as i8,
        transform.cr_b as i8,
        0,
        transform.cr_r as i8,
        transform.cr_g as i8,
        transform.cr_b as i8,
        0,
        transform.cr_r as i8,
        transform.cr_g as i8,
        transform.cr_b as i8,
        0,
    ];
    let weights_cr_bgra: [i8; 16] = [
        transform.cr_b as i8,
        transform.cr_g as i8,
        transform.cr_r as i8,
        0,
        transform.cr_b as i8,
        transform.cr_g as i8,
        transform.cr_r as i8,
        0,
        transform.cr_b as i8,
        transform.cr_g as i8,
        transform.cr_r as i8,
        0,
        transform.cr_b as i8,
        transform.cr_g as i8,
        transform.cr_r as i8,
        0,
    ];

    let y_weights = vld1q_s8(if source_channels == YuvSourceChannels::Rgba {
        weights_yr.as_ptr()
    } else {
        weights_yr_bgra.as_ptr()
    });
    let cb_weights = vld1q_s8(if source_channels == YuvSourceChannels::Rgba {
        weights_cb_rgba.as_ptr()
    } else {
        weights_cb_bgra.as_ptr()
    });
    let cr_weights = vld1q_s8(if source_channels == YuvSourceChannels::Rgba {
        weights_cr_rgba.as_ptr()
    } else {
        weights_cr_bgra.as_ptr()
    });

    let v422_shuffle_table: [u8; 16] = [0, 1, 2, 3, 8, 9, 10, 11, 4, 5, 6, 7, 12, 13, 14, 15];
    let v422_shuffle = vld1q_u8(v422_shuffle_table.as_ptr());

    let mut cx = start_cx;
    let mut ux = start_ux;

    while cx + 16 < width {
        let src = rgba.get_unchecked(cx * channels..).as_ptr();

        let v0 = vld1q_u8(src);
        let v1 = vld1q_u8(src.add(16));
        let v2 = vld1q_u8(src.add(32));
        let v3 = vld1q_u8(src.add(48));

        let y0 = vusdotq_s32(y_bias, v0, y_weights);
        let y1 = vusdotq_s32(y_bias, v1, y_weights);
        let y2 = vusdotq_s32(y_bias, v2, y_weights);
        let y3 = vusdotq_s32(y_bias, v3, y_weights);

        let v0_s = if chroma_subsampling != YuvChromaSubsampling::Yuv444 {
            vqtbl1q_u8(v0, v422_shuffle)
        } else {
            vdupq_n_u8(0)
        };
        let v1_s = if chroma_subsampling != YuvChromaSubsampling::Yuv444 {
            vqtbl1q_u8(v1, v422_shuffle)
        } else {
            vdupq_n_u8(0)
        };
        let v2_s = if chroma_subsampling != YuvChromaSubsampling::Yuv444 {
            vqtbl1q_u8(v2, v422_shuffle)
        } else {
            vdupq_n_u8(0)
        };
        let v3_s = if chroma_subsampling != YuvChromaSubsampling::Yuv444 {
            vqtbl1q_u8(v3, v422_shuffle)
        } else {
            vdupq_n_u8(0)
        };

        let yn_0 = vqshrun_n_s32::<A_E>(y0);
        let yn_1 = vqshrun_n_s32::<A_E>(y1);
        let yn_2 = vqshrun_n_s32::<A_E>(y2);
        let yn_3 = vqshrun_n_s32::<A_E>(y3);

        let y_vl = vcombine_u8(
            vqmovn_u16(vcombine_u16(yn_0, yn_1)),
            vqmovn_u16(vcombine_u16(yn_2, yn_3)),
        );

        vst1q_u8(y_ptr.get_unchecked_mut(cx..).as_mut_ptr(), y_vl);

        if chroma_subsampling == YuvChromaSubsampling::Yuv444 {
            let cb0 = vusdotq_s32(uv_bias, v0, cb_weights);
            let cb1 = vusdotq_s32(uv_bias, v1, cb_weights);
            let cb2 = vusdotq_s32(uv_bias, v2, cb_weights);
            let cb3 = vusdotq_s32(uv_bias, v3, cb_weights);

            let cr0 = vusdotq_s32(uv_bias, v0, cr_weights);
            let cr1 = vusdotq_s32(uv_bias, v1, cr_weights);
            let cr2 = vusdotq_s32(uv_bias, v2, cr_weights);
            let cr3 = vusdotq_s32(uv_bias, v3, cr_weights);

            let cb_0 = vqshrun_n_s32::<A_E>(cb0);
            let cb_1 = vqshrun_n_s32::<A_E>(cb1);
            let cb_2 = vqshrun_n_s32::<A_E>(cb2);
            let cb_3 = vqshrun_n_s32::<A_E>(cb3);

            let cr_0 = vqshrun_n_s32::<A_E>(cr0);
            let cr_1 = vqshrun_n_s32::<A_E>(cr1);
            let cr_2 = vqshrun_n_s32::<A_E>(cr2);
            let cr_3 = vqshrun_n_s32::<A_E>(cr3);

            let cb_vl = vcombine_u8(
                vqmovn_u16(vcombine_u16(cb_0, cb_1)),
                vqmovn_u16(vcombine_u16(cb_2, cb_3)),
            );

            let cr_vl = vcombine_u8(
                vqmovn_u16(vcombine_u16(cr_0, cr_1)),
                vqmovn_u16(vcombine_u16(cr_2, cr_3)),
            );

            vst1q_u8(u_ptr.get_unchecked_mut(ux..).as_mut_ptr(), cb_vl);
            vst1q_u8(v_ptr.get_unchecked_mut(ux..).as_mut_ptr(), cr_vl);

            ux += 16;
        } else if (chroma_subsampling == YuvChromaSubsampling::Yuv420)
            || (chroma_subsampling == YuvChromaSubsampling::Yuv422)
        {
            let v0 = vhadd_u8(vget_low_u8(v0_s), vget_high_u8(v0_s));
            let v1 = vhadd_u8(vget_low_u8(v1_s), vget_high_u8(v1_s));
            let v2 = vhadd_u8(vget_low_u8(v2_s), vget_high_u8(v2_s));
            let v3 = vhadd_u8(vget_low_u8(v3_s), vget_high_u8(v3_s));

            let v0_f = vcombine_u8(v0, v1);
            let v1_f = vcombine_u8(v2, v3);

            let cb0 = vusdotq_s32(uv_bias, v0_f, cb_weights);
            let cb1 = vusdotq_s32(uv_bias, v1_f, cb_weights);

            let cr0 = vusdotq_s32(uv_bias, v0_f, cr_weights);
            let cr1 = vusdotq_s32(uv_bias, v1_f, cr_weights);

            let cb_0 = vqshrun_n_s32::<A_E>(cb0);
            let cb_1 = vqshrun_n_s32::<A_E>(cb1);

            let cr_0 = vqshrun_n_s32::<A_E>(cr0);
            let cr_1 = vqshrun_n_s32::<A_E>(cr1);

            let cb_vl = vqmovn_u16(vcombine_u16(cb_0, cb_1));
            let cr_vl = vqmovn_u16(vcombine_u16(cr_0, cr_1));

            vst1_u8(u_ptr.get_unchecked_mut(ux..).as_mut_ptr(), cb_vl);
            vst1_u8(v_ptr.get_unchecked_mut(ux..).as_mut_ptr(), cr_vl);

            ux += 8;
        }

        cx += 16;
    }

    if cx < width {
        let diff = width - cx;
        assert!(diff <= 16);

        let mut src_buffer: [u8; 16 * 4] = [0; 16 * 4];
        let mut y_buffer: [u8; 16] = [0; 16];
        let mut u_buffer: [u8; 16] = [0; 16];
        let mut v_buffer: [u8; 16] = [0; 16];

        std::ptr::copy_nonoverlapping(
            rgba.get_unchecked(cx * channels..).as_ptr(),
            src_buffer.as_mut_ptr(),
            diff * channels,
        );

        // Replicate last item to one more position for subsampling
        if chroma_subsampling != YuvChromaSubsampling::Yuv444 && diff % 2 != 0 {
            let lst = (width - 1) * channels;
            let last_items = rgba.get_unchecked(lst..(lst + channels));
            let dvb = diff * channels;
            let dst = src_buffer.get_unchecked_mut(dvb..(dvb + channels));
            for (dst, src) in dst.iter_mut().zip(last_items) {
                *dst = *src;
            }
        }

        let v0 = vld1q_u8(src_buffer.as_ptr());
        let v1 = vld1q_u8(src_buffer.as_ptr().add(16));
        let v2 = vld1q_u8(src_buffer.as_ptr().add(32));
        let v3 = vld1q_u8(src_buffer.as_ptr().add(48));

        let y0 = vusdotq_s32(y_bias, v0, y_weights);
        let y1 = vusdotq_s32(y_bias, v1, y_weights);
        let y2 = vusdotq_s32(y_bias, v2, y_weights);
        let y3 = vusdotq_s32(y_bias, v3, y_weights);

        let v0_s = if chroma_subsampling != YuvChromaSubsampling::Yuv444 {
            vqtbl1q_u8(v0, v422_shuffle)
        } else {
            vdupq_n_u8(0)
        };
        let v1_s = if chroma_subsampling != YuvChromaSubsampling::Yuv444 {
            vqtbl1q_u8(v1, v422_shuffle)
        } else {
            vdupq_n_u8(0)
        };
        let v2_s = if chroma_subsampling != YuvChromaSubsampling::Yuv444 {
            vqtbl1q_u8(v2, v422_shuffle)
        } else {
            vdupq_n_u8(0)
        };
        let v3_s = if chroma_subsampling != YuvChromaSubsampling::Yuv444 {
            vqtbl1q_u8(v3, v422_shuffle)
        } else {
            vdupq_n_u8(0)
        };

        let yn_0 = vqshrun_n_s32::<A_E>(y0);
        let yn_1 = vqshrun_n_s32::<A_E>(y1);
        let yn_2 = vqshrun_n_s32::<A_E>(y2);
        let yn_3 = vqshrun_n_s32::<A_E>(y3);

        let y_vl = vcombine_u8(
            vqmovn_u16(vcombine_u16(yn_0, yn_1)),
            vqmovn_u16(vcombine_u16(yn_2, yn_3)),
        );

        vst1q_u8(y_buffer.as_mut_ptr(), y_vl);

        if chroma_subsampling == YuvChromaSubsampling::Yuv444 {
            let cb0 = vusdotq_s32(uv_bias, v0, cb_weights);
            let cb1 = vusdotq_s32(uv_bias, v1, cb_weights);
            let cb2 = vusdotq_s32(uv_bias, v2, cb_weights);
            let cb3 = vusdotq_s32(uv_bias, v3, cb_weights);

            let cr0 = vusdotq_s32(uv_bias, v0, cr_weights);
            let cr1 = vusdotq_s32(uv_bias, v1, cr_weights);
            let cr2 = vusdotq_s32(uv_bias, v2, cr_weights);
            let cr3 = vusdotq_s32(uv_bias, v3, cr_weights);

            let cb_0 = vqshrun_n_s32::<A_E>(cb0);
            let cb_1 = vqshrun_n_s32::<A_E>(cb1);
            let cb_2 = vqshrun_n_s32::<A_E>(cb2);
            let cb_3 = vqshrun_n_s32::<A_E>(cb3);

            let cr_0 = vqshrun_n_s32::<A_E>(cr0);
            let cr_1 = vqshrun_n_s32::<A_E>(cr1);
            let cr_2 = vqshrun_n_s32::<A_E>(cr2);
            let cr_3 = vqshrun_n_s32::<A_E>(cr3);

            let cb_vl = vcombine_u8(
                vqmovn_u16(vcombine_u16(cb_0, cb_1)),
                vqmovn_u16(vcombine_u16(cb_2, cb_3)),
            );

            let cr_vl = vcombine_u8(
                vqmovn_u16(vcombine_u16(cr_0, cr_1)),
                vqmovn_u16(vcombine_u16(cr_2, cr_3)),
            );

            vst1q_u8(u_buffer.as_mut_ptr(), cb_vl);
            vst1q_u8(v_buffer.as_mut_ptr(), cr_vl);
        } else if (chroma_subsampling == YuvChromaSubsampling::Yuv420)
            || (chroma_subsampling == YuvChromaSubsampling::Yuv422)
        {
            let v0 = vhadd_u8(vget_low_u8(v0_s), vget_high_u8(v0_s));
            let v1 = vhadd_u8(vget_low_u8(v1_s), vget_high_u8(v1_s));
            let v2 = vhadd_u8(vget_low_u8(v2_s), vget_high_u8(v2_s));
            let v3 = vhadd_u8(vget_low_u8(v3_s), vget_high_u8(v3_s));

            let v0_f = vcombine_u8(v0, v1);
            let v1_f = vcombine_u8(v2, v3);

            let cb0 = vusdotq_s32(uv_bias, v0_f, cb_weights);
            let cb1 = vusdotq_s32(uv_bias, v1_f, cb_weights);

            let cr0 = vusdotq_s32(uv_bias, v0_f, cr_weights);
            let cr1 = vusdotq_s32(uv_bias, v1_f, cr_weights);

            let cb_0 = vqshrun_n_s32::<A_E>(cb0);
            let cb_1 = vqshrun_n_s32::<A_E>(cb1);

            let cr_0 = vqshrun_n_s32::<A_E>(cr0);
            let cr_1 = vqshrun_n_s32::<A_E>(cr1);

            let cb_vl = vqmovn_u16(vcombine_u16(cb_0, cb_1));
            let cr_vl = vqmovn_u16(vcombine_u16(cr_0, cr_1));

            vst1_u8(u_buffer.as_mut_ptr(), cb_vl);
            vst1_u8(v_buffer.as_mut_ptr(), cr_vl);
        }
        std::ptr::copy_nonoverlapping(
            y_buffer.as_ptr(),
            y_ptr.get_unchecked_mut(cx..).as_mut_ptr(),
            diff,
        );

        cx += diff;

        if chroma_subsampling == YuvChromaSubsampling::Yuv444 {
            std::ptr::copy_nonoverlapping(
                u_buffer.as_ptr(),
                u_ptr.get_unchecked_mut(ux..).as_mut_ptr(),
                diff,
            );
            std::ptr::copy_nonoverlapping(
                v_buffer.as_ptr(),
                v_ptr.get_unchecked_mut(ux..).as_mut_ptr(),
                diff,
            );

            ux += diff;
        } else if (chroma_subsampling == YuvChromaSubsampling::Yuv420)
            || (chroma_subsampling == YuvChromaSubsampling::Yuv422)
        {
            let hv = diff.div_ceil(2);
            std::ptr::copy_nonoverlapping(
                u_buffer.as_ptr(),
                u_ptr.get_unchecked_mut(ux..).as_mut_ptr(),
                hv,
            );
            std::ptr::copy_nonoverlapping(
                v_buffer.as_ptr(),
                v_ptr.get_unchecked_mut(ux..).as_mut_ptr(),
                hv,
            );

            ux += hv;
        }
    }

    ProcessedOffset { cx, ux }
}
