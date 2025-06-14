/*
 * Copyright (c) Radzivon Bartoshyk, 6/2025. All rights reserved.
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

#[inline(always)]
/// Saturating rounding shift right against bit depth
pub(crate) fn qrshr<const PRECISION: i32, const BIT_DEPTH: usize>(val: i32) -> i32 {
    let rounding: i32 = (1 << (PRECISION - 1)) - 1;
    let max_value: i32 = (1 << BIT_DEPTH) - 1;
    ((val + rounding) >> PRECISION).min(max_value).max(0)
}

// Q_FRACTION and Q fixed point format should match to work correctly;
const Y_BIAS: i16 = 0;
const UV_BIAS: i16 = 0;
const Y_COEF: i16 = 0;
const CR_COEF: i16 = 0;
const CB_COEF: i16 = 0;
const G_COEF_1: i16 = 0;
const G_COEF_2: i16 = 0;

fn interpolate_1_row<const CN: usize, const Q_FRACTION: i32>(
    y_plane: &[u8],
    u_plane: &[u8],
    v_plane: &[u8],
    rgba: &mut [u8],
) {
    const BIT_DEPTH: usize = 8;

    // Bilinear upscaling weights in Q0.4
    // x = x0 * 0.75 + x1 * 0.25 = (x0 * 3 + x1 + 1) >> 2

    for (((rgba, y_src), u_src), v_src) in rgba
        .chunks_exact_mut(CN * 2)
        .zip(y_plane.chunks_exact(2))
        .zip(u_plane.windows(2))
        .zip(v_plane.windows(2))
    {
        let cb_0 = (u_src[0] as u16 * 3 + u_src[1] as u16 + 1) >> 2;
        let cr_0 = (v_src[0] as u16 * 3 + v_src[1] as u16 + 1) >> 2;

        let cb_1 = (u_src[0] as u16 + u_src[1] as u16 * 3 + 1) >> 2;
        let cr_1 = (v_src[0] as u16 + v_src[1] as u16 * 3 + 1) >> 2;

        let y_value0 = (y_src[0] as i32 - Y_BIAS as i32) * Y_COEF as i32;
        let cb_value0 = cb_0 as i16 - UV_BIAS;
        let cr_value0 = cr_0 as i16 - UV_BIAS;

        let r0 = qrshr::<Q_FRACTION, BIT_DEPTH>(y_value0 + CR_COEF as i32 * cr_value0 as i32);
        let b0 = qrshr::<Q_FRACTION, BIT_DEPTH>(y_value0 + CB_COEF as i32 * cb_value0 as i32);
        let g0 = qrshr::<Q_FRACTION, BIT_DEPTH>(
            y_value0 - G_COEF_1 as i32 * cr_value0 as i32 - G_COEF_2 as i32 * cb_value0 as i32,
        );

        let rgba0 = &mut rgba[..CN];

        rgba0[0] = r0 as u8;
        rgba0[1] = g0 as u8;
        rgba0[2] = b0 as u8;
        if CN == 4 {
            rgba0[3] = 255u8;
        }

        let y_value1 = (y_src[1] as i32 - Y_BIAS as i32) * Y_COEF as i32;
        let cb_value1 = cb_1 as i16 - UV_BIAS;
        let cr_value1 = cr_1 as i16 - UV_BIAS;

        let r0 = qrshr::<Q_FRACTION, BIT_DEPTH>(y_value1 + CR_COEF as i32 * cr_value1 as i32);
        let b0 = qrshr::<Q_FRACTION, BIT_DEPTH>(y_value1 + CB_COEF as i32 * cb_value1 as i32);
        let g0 = qrshr::<Q_FRACTION, BIT_DEPTH>(
            y_value1 - G_COEF_1 as i32 * cr_value1 as i32 - G_COEF_2 as i32 * cb_value1 as i32,
        );

        let rgba1 = &mut rgba[CN..CN * 2];

        rgba1[0] = r0 as u8;
        rgba1[1] = g0 as u8;
        rgba1[2] = b0 as u8;
        if CN == 4 {
            rgba1[3] = 255u8;
        }
    }

    let y_chunks = y_plane.chunks_exact(2);
    let y_remainder = y_chunks.remainder();
    let rgba_chunks = rgba.chunks_exact_mut(CN * 2);
    let rgba_remainder = rgba_chunks.into_remainder();

    if let ([last_y], [rgba @ ..]) = (y_remainder, rgba_remainder) {
        let y_value0 = (*last_y as i32 - Y_BIAS as i32) * Y_COEF as i32;
        let cb_value = *u_plane.last().unwrap() as i32 - UV_BIAS as i32;
        let cr_value = *v_plane.last().unwrap() as i32 - UV_BIAS as i32;
        let rgba0 = &mut rgba[..CN];

        let r0 = qrshr::<Q_FRACTION, BIT_DEPTH>(y_value0 + CR_COEF as i32 * cr_value);
        let b0 = qrshr::<Q_FRACTION, BIT_DEPTH>(y_value0 + CB_COEF as i32 * cb_value);
        let g0 = qrshr::<Q_FRACTION, BIT_DEPTH>(
            y_value0 - G_COEF_1 as i32 * cr_value - G_COEF_2 as i32 * cb_value,
        );
        rgba0[0] = r0 as u8;
        rgba0[1] = g0 as u8;
        rgba0[2] = b0 as u8;
        if CN == 4 {
            rgba0[3] = 255;
        }
    }
}

fn interpolate_2_rows<const CN: usize, const Q_FRACTION: i32>(
    y_plane: &[u8],
    u_plane0: &[u8],
    u_plane1: &[u8],
    v_plane0: &[u8],
    v_plane1: &[u8],
    rgba: &mut [u8],
) {
    const BIT_DEPTH: usize = 8;

    // Bilinear upscaling weights in Q0.4
    // x = x0y0 * 0.5625 + x1y0 * 0.1875 + x0y1 * 0.1875 * x1y1 * 0.0625 = (x0y0 * 9 + x1y0 * 3 + x0y1 * 3 + x1y1 + (1 << 3)) >> 4

    for (((((rgba0, y_src0), u_src), u_src_next), v_src), v_src_next) in rgba
        .chunks_exact_mut(CN * 2)
        .zip(y_plane.chunks_exact(2))
        .zip(u_plane0.windows(2))
        .zip(u_plane1.windows(2))
        .zip(v_plane0.windows(2))
        .zip(v_plane1.windows(2))
    {
        let cb_0 = (u_src[0] as u16 * 9
            + u_src[1] as u16 * 3
            + u_src_next[0] as u16 * 3
            + u_src_next[1] as u16
            + (1 << 3))
            >> 4;
        let cr_0 = (v_src[0] as u16 * 9
            + v_src[1] as u16 * 3
            + v_src_next[0] as u16 * 3
            + v_src_next[1] as u16
            + (1 << 3))
            >> 4;

        let cb_1 = (u_src[0] as u16 * 3
            + u_src[1] as u16 * 9
            + u_src_next[0] as u16
            + u_src_next[1] as u16 * 3
            + (1 << 3))
            >> 4;
        let cr_1 = (v_src[0] as u16 * 3
            + v_src[1] as u16 * 9
            + v_src_next[0] as u16
            + v_src_next[1] as u16 * 3
            + (1 << 3))
            >> 4;

        let y_value0 = (y_src0[0] as i32 - Y_BIAS as i32) * Y_COEF as i32;
        let cb_value0 = cb_0 as i16 - UV_BIAS;
        let cr_value0 = cr_0 as i16 - UV_BIAS;

        let g_built_coeff0 =
            -G_COEF_1 as i32 * cr_value0 as i32 - G_COEF_2 as i32 * cb_value0 as i32;

        let r0 = qrshr::<Q_FRACTION, BIT_DEPTH>(y_value0 + CR_COEF as i32 * cr_value0 as i32);
        let b0 = qrshr::<Q_FRACTION, BIT_DEPTH>(y_value0 + CB_COEF as i32 * cb_value0 as i32);
        let g0 = qrshr::<Q_FRACTION, BIT_DEPTH>(y_value0 + g_built_coeff0);

        let rgba00 = &mut rgba0[..CN];

        rgba00[0] = r0 as u8;
        rgba00[1] = g0 as u8;
        rgba00[2] = b0 as u8;
        if CN == 4 {
            rgba00[3] = 255u8;
        }

        let y_value1 = (y_src0[1] as i32 - Y_BIAS as i32) * Y_COEF as i32;
        let cb_value1 = cb_1 as i16 - UV_BIAS;
        let cr_value1 = cr_1 as i16 - UV_BIAS;

        let g_built_coeff1 =
            -G_COEF_1 as i32 * cr_value1 as i32 - G_COEF_2 as i32 * cb_value1 as i32;

        let r1 = qrshr::<Q_FRACTION, BIT_DEPTH>(y_value1 + CR_COEF as i32 * cr_value1 as i32);
        let b1 = qrshr::<Q_FRACTION, BIT_DEPTH>(y_value1 + CB_COEF as i32 * cb_value1 as i32);
        let g1 = qrshr::<Q_FRACTION, BIT_DEPTH>(y_value1 + g_built_coeff1);

        let rgba01 = &mut rgba0[CN..CN * 2];

        rgba01[0] = r1 as u8;
        rgba01[1] = g1 as u8;
        rgba01[2] = b1 as u8;
        if CN == 4 {
            rgba01[3] = 255u8;
        }
    }

    let y_chunks = y_plane.chunks_exact(2);
    let y_remainder = y_chunks.remainder();
    let rgba_chunks = rgba.chunks_exact_mut(CN * 2);
    let rgba_remainder = rgba_chunks.into_remainder();

    if let ([last_y], [rgba @ ..]) = (y_remainder, rgba_remainder) {
        let y_value0 = (*last_y as i32 - Y_BIAS as i32) * Y_COEF as i32;

        let cb_0 = (*u_plane0.last().unwrap() as u16 * 12
            + *u_plane1.last().unwrap() as u16 * 4
            + (1 << 3))
            >> 4;
        let cr_0 = (*v_plane0.last().unwrap() as u16 * 12
            + (*v_plane1.last().unwrap()) as u16 * 4
            + (1 << 3))
            >> 4;

        let cb_value = cb_0 as i16 - UV_BIAS;
        let cr_value = cr_0 as i16 - UV_BIAS;
        let rgba0 = &mut rgba[..CN];

        let g_built_coeff = -G_COEF_1 * cr_value - G_COEF_2 * cb_value;

        let r0 = qrshr::<Q_FRACTION, BIT_DEPTH>(y_value0 + CR_COEF as i32 * cr_value as i32);
        let b0 = qrshr::<Q_FRACTION, BIT_DEPTH>(y_value0 + CB_COEF as i32 * cb_value as i32);
        let g0 = qrshr::<Q_FRACTION, BIT_DEPTH>(y_value0 + g_built_coeff as i32);

        rgba0[0] = r0 as u8;
        rgba0[1] = g0 as u8;
        rgba0[2] = b0 as u8;
        if CN == 4 {
            rgba0[3] = 255;
        }
    }
}

#[derive(Copy, Clone, Debug, Ord, PartialOrd, Eq, PartialEq)]
enum Sampling {
    S420,
    S422,
}

fn yuv_to_rgbx_impl_bilinear<
    const CN: usize, // CHANNELS count 3 or 4
    const Q_FRACTION: i32,
>(
    y_plane: &[u8],
    y_stride: usize,
    u_plane: &[u8],
    u_stride: usize,
    v_plane: &[u8],
    v_stride: usize,
    rgba: &mut [u8],
    rgba_stride: u32,
    width: usize,
    height: usize,
    sampling: Sampling,
) {
    if sampling == Sampling::S422 {
        let iter = rgba
            .chunks_exact_mut(rgba_stride as usize)
            .zip(y_plane.chunks_exact(y_stride))
            .zip(u_plane.chunks_exact(u_stride))
            .zip(v_plane.chunks_exact(v_stride));
        iter.for_each(|(((rgba, y_plane), u_plane), v_plane)| {
            interpolate_1_row::<CN, Q_FRACTION>(
                &y_plane[..width],
                &u_plane[..width.div_ceil(2)],
                &v_plane[..width.div_ceil(2)],
                &mut rgba[..width * CN],
            );
        });
    } else if sampling == Sampling::S420 {
        let iter = rgba
            .chunks_exact_mut(rgba_stride as usize * 2)
            .zip(y_plane.chunks_exact(y_stride * 2))
            .zip(u_plane.windows(u_stride * 2).step_by(u_stride))
            .zip(v_plane.windows(v_stride * 2).step_by(v_stride));
        iter.for_each(|(((rgba, y_plane), u_plane), v_plane)| {
            let (y_plane0, y_plane1) = y_plane.split_at(y_stride);
            let (rgba0, rgba1) = rgba.split_at_mut(rgba_stride as usize);
            let (u_plane0, u_plane1) = u_plane.split_at(u_stride);
            let (v_plane0, v_plane1) = v_plane.split_at(v_stride);
            interpolate_2_rows::<CN, Q_FRACTION>(
                &y_plane0[..width],
                &u_plane0[..width.div_ceil(2)],
                &u_plane1[..width.div_ceil(2)],
                &v_plane0[..width.div_ceil(2)],
                &v_plane1[..width.div_ceil(2)],
                &mut rgba0[..width * CN],
            );
            interpolate_2_rows::<CN, Q_FRACTION>(
                &y_plane1[..width],
                &u_plane1[..width.div_ceil(2)],
                &u_plane0[..width.div_ceil(2)],
                &v_plane1[..width.div_ceil(2)],
                &v_plane0[..width.div_ceil(2)],
                &mut rgba1[..width * CN],
            );
        });

        if height & 1 != 0 {
            let rgba = rgba.chunks_exact_mut(rgba_stride as usize).last().unwrap();
            let u_plane = u_plane.chunks_exact(u_stride).last().unwrap();
            let v_plane = v_plane.chunks_exact(v_stride).last().unwrap();
            let y_plane = y_plane.chunks_exact(y_stride).last().unwrap();
            interpolate_1_row::<CN, Q_FRACTION>(
                &y_plane[..width],
                &u_plane[..width.div_ceil(2)],
                &v_plane[..width.div_ceil(2)],
                &mut rgba[..width * CN],
            );
        }
    } else {
        unreachable!();
    }
}
