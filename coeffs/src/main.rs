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
use rug::float::Round;
use rug::Float;

const BITS: u32 = 150;

#[derive(Debug, Copy, Clone, PartialOrd, PartialEq)]
pub struct YuvChromaRange {
    pub bias_y: u32,
    pub bias_uv: u32,
    pub range_y: u32,
    pub range_uv: u32,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialOrd, PartialEq)]
/// Declares YUV range TV (limited) or Full
pub enum YuvRange {
    /// Limited range Y ∈ [16 << (depth - 8), 16 << (depth - 8) + 224 << (depth - 8)], UV ∈ [-1 << (depth - 1), -1 << (depth - 1) + 1 << (depth - 1)]
    Limited,
    /// Full range Y ∈ [0, 2^bit_depth - 1], UV ∈ [-1 << (depth - 1), -1 << (depth - 1) + 2^bit_depth - 1]
    Full,
}

pub const fn get_yuv_range(depth: u32, range: YuvRange) -> YuvChromaRange {
    match range {
        YuvRange::Limited => YuvChromaRange {
            bias_y: 16 << (depth - 8),
            bias_uv: 1 << (depth - 1),
            range_y: 219 << (depth - 8),
            range_uv: 224 << (depth - 8),
        },
        YuvRange::Full => YuvChromaRange {
            bias_y: 0,
            bias_uv: 1 << (depth - 1),
            range_uv: (1 << depth) - 1,
            range_y: (1 << depth) - 1,
        },
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialOrd, PartialEq)]
pub struct CbCrForwardTransform<T> {
    pub yr: T,
    pub yg: T,
    pub yb: T,
    pub cb_r: T,
    pub cb_g: T,
    pub cb_b: T,
    pub cr_r: T,
    pub cr_g: T,
    pub cr_b: T,
}

fn get_forward_coeffs(
    kr: f32,
    kb: f32,
    bit_depth: u32,
    range: YuvRange,
) -> CbCrForwardTransform<f32> {
    let get_kg = || -> Float {
        let kg =
            Float::with_val(BITS, 1.0f32) - Float::with_val(BITS, kr) - Float::with_val(BITS, kb);
        kg
    };

    let range = get_yuv_range(bit_depth, range);

    let yr = kr * Float::with_val(BITS, range.range_y as f32)
        / Float::with_val(BITS, ((1 << bit_depth) - 1) as f32);
    let yg = get_kg() * Float::with_val(BITS, range.range_y as f32)
        / Float::with_val(BITS, ((1 << bit_depth) - 1) as f32);
    let yb = kb * Float::with_val(BITS, range.range_y as f32)
        / Float::with_val(BITS, ((1 << bit_depth) - 1) as f32);

    let cb_r = Float::with_val(BITS, -0.5f32) * Float::with_val(BITS, kr)
        / (Float::with_val(BITS, 1f32) - Float::with_val(BITS, kb))
        * Float::with_val(BITS, range.range_uv as f32)
        / Float::with_val(BITS, ((1 << bit_depth) - 1) as f32);
    let cb_g = Float::with_val(BITS, -0.5f32) * get_kg() / (Float::with_val(BITS, 1f32) - kb)
        * Float::with_val(BITS, range.range_uv as f32)
        / Float::with_val(BITS, ((1 << bit_depth) - 1) as f32);
    let cb_b = Float::with_val(BITS, 0.5f32) * Float::with_val(BITS, range.range_uv as f32)
        / Float::with_val(BITS, ((1 << bit_depth) - 1) as f32);

    let cr_r = Float::with_val(BITS, 0.5f32) * Float::with_val(BITS, range.range_uv as f32)
        / Float::with_val(BITS, ((1 << bit_depth) - 1) as f32);
    let cr_g = Float::with_val(BITS, -0.5f32) * get_kg()
        / (Float::with_val(BITS, 1f32) - Float::with_val(BITS, kr))
        * Float::with_val(BITS, range.range_uv as f32)
        / Float::with_val(BITS, ((1 << bit_depth) - 1) as f32);
    let cr_b = Float::with_val(BITS, -0.5f32) * Float::with_val(BITS, kb)
        / (1f32 - Float::with_val(BITS, kr))
        * Float::with_val(BITS, range.range_uv as f32)
        / Float::with_val(BITS, ((1 << bit_depth) - 1) as f32);

    CbCrForwardTransform {
        yr: yr.to_f32(),
        yg: yg.to_f32(),
        yb: yb.to_f32(),
        cb_r: cb_r.to_f32(),
        cb_g: cb_g.to_f32(),
        cb_b: cb_b.to_f32(),
        cr_r: cr_r.to_f32(),
        cr_g: cr_g.to_f32(),
        cr_b: cr_b.to_f32(),
    }
}

fn get_forward_coeffs_integral(
    kr: f32,
    kb: f32,
    bit_depth: u32,
    range: YuvRange,
    precision: u32,
) -> CbCrForwardTransform<i32> {
    let get_kg = || -> Float {
        let kg =
            Float::with_val(BITS, 1.0f32) - Float::with_val(BITS, kr) - Float::with_val(BITS, kb);
        kg
    };

    let range = get_yuv_range(bit_depth, range);

    let yr = kr * Float::with_val(BITS, range.range_y as f32)
        / Float::with_val(BITS, ((1 << bit_depth) - 1) as f32);
    let yg = get_kg() * Float::with_val(BITS, range.range_y as f32)
        / Float::with_val(BITS, ((1 << bit_depth) - 1) as f32);
    let yb = kb * Float::with_val(BITS, range.range_y as f32)
        / Float::with_val(BITS, ((1 << bit_depth) - 1) as f32);

    let cb_r = Float::with_val(BITS, -0.5f32) * Float::with_val(BITS, kr)
        / (Float::with_val(BITS, 1f32) - Float::with_val(BITS, kb))
        * Float::with_val(BITS, range.range_uv as f32)
        / Float::with_val(BITS, ((1 << bit_depth) - 1) as f32);
    let cb_g = Float::with_val(BITS, -0.5f32) * get_kg() / (Float::with_val(BITS, 1f32) - kb)
        * Float::with_val(BITS, range.range_uv as f32)
        / Float::with_val(BITS, ((1 << bit_depth) - 1) as f32);
    let cb_b = Float::with_val(BITS, 0.5f32) * Float::with_val(BITS, range.range_uv as f32)
        / Float::with_val(BITS, ((1 << bit_depth) - 1) as f32);

    let cr_r = Float::with_val(BITS, 0.5f32) * Float::with_val(BITS, range.range_uv as f32)
        / Float::with_val(BITS, ((1 << bit_depth) - 1) as f32);
    let cr_g = Float::with_val(BITS, -0.5f32) * get_kg()
        / (Float::with_val(BITS, 1f32) - Float::with_val(BITS, kr))
        * Float::with_val(BITS, range.range_uv as f32)
        / Float::with_val(BITS, ((1 << bit_depth) - 1) as f32);
    let cr_b = Float::with_val(BITS, -0.5f32) * Float::with_val(BITS, kb)
        / (1f32 - Float::with_val(BITS, kr))
        * Float::with_val(BITS, range.range_uv as f32)
        / Float::with_val(BITS, ((1 << bit_depth) - 1) as f32);

    let prec = (1 << precision) as f32;

    CbCrForwardTransform {
        yr: (yr * Float::with_val(BITS, prec))
            .to_i32_saturating_round(Round::Nearest)
            .unwrap(),
        yg: (yg * Float::with_val(BITS, prec))
            .to_i32_saturating_round(Round::Nearest)
            .unwrap(),
        yb: (yb * Float::with_val(BITS, prec))
            .to_i32_saturating_round(Round::Nearest)
            .unwrap(),
        cb_r: (cb_r * Float::with_val(BITS, prec))
            .to_i32_saturating_round(Round::Nearest)
            .unwrap(),
        cb_g: (cb_g * Float::with_val(BITS, prec))
            .to_i32_saturating_round(Round::Nearest)
            .unwrap(),
        cb_b: (cb_b * Float::with_val(BITS, prec))
            .to_i32_saturating_round(Round::Nearest)
            .unwrap(),
        cr_r: (cr_r * Float::with_val(BITS, prec))
            .to_i32_saturating_round(Round::Nearest)
            .unwrap(),
        cr_g: (cr_g * Float::with_val(BITS, prec))
            .to_i32_saturating_round(Round::Nearest)
            .unwrap(),
        cr_b: (cr_b * Float::with_val(BITS, prec))
            .to_i32_saturating_round(Round::Nearest)
            .unwrap(),
    }
}

#[derive(Debug, Copy, Clone)]
pub struct CbCrInverseTransform<T> {
    pub y_coef: T,
    pub cr_coef: T,
    pub cb_coef: T,
    pub g_coeff_1: T,
    pub g_coeff_2: T,
}

impl<T> CbCrInverseTransform<T> {
    pub fn new(
        y_coef: T,
        cr_coef: T,
        cb_coef: T,
        g_coeff_1: T,
        g_coeff_2: T,
    ) -> CbCrInverseTransform<T> {
        CbCrInverseTransform {
            y_coef,
            cr_coef,
            cb_coef,
            g_coeff_1,
            g_coeff_2,
        }
    }
}

pub fn get_inverse_transform(
    kr: f32,
    kb: f32,
    bit_depth: u32,
    range: YuvRange,
) -> CbCrInverseTransform<Float> {
    let get_kg = || -> Float {
        let kg =
            Float::with_val(BITS, 1.0f32) - Float::with_val(BITS, kr) - Float::with_val(BITS, kb);
        kg
    };
    let range = get_yuv_range(bit_depth, range);
    let range_uv = || {
        Float::with_val(BITS, ((1 << bit_depth) - 1) as f32)
            / Float::with_val(BITS, range.range_uv as f32)
    };
    let y_coef = Float::with_val(BITS, ((1 << bit_depth) - 1) as f32)
        / Float::with_val(BITS, range.range_y as f32);
    let cr_coeff = (2f32 * (Float::with_val(BITS, 1f32) - Float::with_val(BITS, kr))) * range_uv();
    let cb_coeff = (2f32 * (Float::with_val(BITS, 1f32) - Float::with_val(BITS, kb))) * range_uv();
    let g_coeff_1 = (2f32
        * ((Float::with_val(BITS, 1f32) - Float::with_val(BITS, kr)) * Float::with_val(BITS, kr)
            / get_kg()))
        * range_uv();
    let g_coeff_2 = (2f32
        * ((Float::with_val(BITS, 1f32) - Float::with_val(BITS, kb)) * Float::with_val(BITS, kb)
            / get_kg()))
        * range_uv();
    CbCrInverseTransform::new(y_coef, cr_coeff, cb_coeff, g_coeff_1, g_coeff_2)
}

pub fn get_inverse_transform_integral(
    kr: f32,
    kb: f32,
    bit_depth: u32,
    prec: u32,
    range: YuvRange,
) -> CbCrInverseTransform<i32> {
    let get_kg = || -> Float {
        let kg =
            Float::with_val(BITS, 1.0f32) - Float::with_val(BITS, kr) - Float::with_val(BITS, kb);
        kg
    };
    let range = get_yuv_range(bit_depth, range);
    let range_uv = || {
        Float::with_val(BITS, ((1 << bit_depth) - 1) as f32)
            / Float::with_val(BITS, range.range_uv as f32)
    };
    let y_coef = Float::with_val(BITS, ((1 << bit_depth) - 1) as f32)
        / Float::with_val(BITS, range.range_y as f32);
    let cr_coeff = (2f32 * (Float::with_val(BITS, 1f32) - Float::with_val(BITS, kr))) * range_uv();
    let cb_coeff = (2f32 * (Float::with_val(BITS, 1f32) - Float::with_val(BITS, kb))) * range_uv();
    let g_coeff_1 = (2f32
        * ((Float::with_val(BITS, 1f32) - Float::with_val(BITS, kr)) * Float::with_val(BITS, kr)
            / get_kg()))
        * range_uv();
    let g_coeff_2 = (2f32
        * ((Float::with_val(BITS, 1f32) - Float::with_val(BITS, kb)) * Float::with_val(BITS, kb)
            / get_kg()))
        * range_uv();
    let prec = (1 << prec) as f32;
    let y_coeff = y_coef * Float::with_val(BITS, prec);
    let cr_coeff = cr_coeff * Float::with_val(BITS, prec);
    let cb_coeff = cb_coeff * Float::with_val(BITS, prec);
    let g_coeff_1 = g_coeff_1 * Float::with_val(BITS, prec);
    let g_coeff_2 = g_coeff_2 * Float::with_val(BITS, prec);
    CbCrInverseTransform::new(
        y_coeff.to_i32_saturating_round(Round::Nearest).unwrap(),
        cr_coeff.to_i32_saturating_round(Round::Nearest).unwrap(),
        cb_coeff.to_i32_saturating_round(Round::Nearest).unwrap(),
        g_coeff_1.to_i32_saturating_round(Round::Nearest).unwrap(),
        g_coeff_2.to_i32_saturating_round(Round::Nearest).unwrap(),
    )
}

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialOrd, PartialEq)]
/// Declares standard prebuilt YUV conversion matrices, check [ITU-R](https://www.itu.int/rec/T-REC-H.273/en) information for more info
/// JPEG YUV Matrix corresponds Bt.601 + Full Range
pub enum YuvStandardMatrix {
    /// If you want to encode/decode JPEG YUV use Bt.601 + Full Range
    Bt601,
    Bt709,
    Bt2020,
    Smpte240,
    Bt470_6,
    /// Custom parameters first goes for kr, second for kb.
    /// Methods will *panic* if 1.0f32 - kr - kb == 0
    Custom(f32, f32),
}

#[derive(Debug, Copy, Clone, PartialOrd, PartialEq)]
pub struct YuvBias {
    pub kr: f32,
    pub kb: f32,
}

impl YuvStandardMatrix {
    pub const fn get_kr_kb(self) -> YuvBias {
        match self {
            YuvStandardMatrix::Bt601 => YuvBias {
                kr: 0.299f32,
                kb: 0.114f32,
            },
            YuvStandardMatrix::Bt709 => YuvBias {
                kr: 0.2126f32,
                kb: 0.0722f32,
            },
            YuvStandardMatrix::Bt2020 => YuvBias {
                kr: 0.2627f32,
                kb: 0.0593f32,
            },
            YuvStandardMatrix::Smpte240 => YuvBias {
                kr: 0.087f32,
                kb: 0.212f32,
            },
            YuvStandardMatrix::Bt470_6 => YuvBias {
                kr: 0.2220f32,
                kb: 0.0713f32,
            },
            YuvStandardMatrix::Custom(kr, kb) => YuvBias { kr, kb },
        }
    }
}

fn main() {
    let kr_kb = YuvStandardMatrix::Bt2020.get_kr_kb();
    let range = YuvRange::Full;
    let bit_depth = 12;
    let transform = get_forward_coeffs(kr_kb.kr, kr_kb.kb, bit_depth, range);
    println!("Precise {:?};", transform);
    let integral = get_forward_coeffs_integral(kr_kb.kr, kr_kb.kb, bit_depth, range, 14);
    println!("Integral {:?};", integral);

    let inverse = get_inverse_transform(kr_kb.kr, kr_kb.kb, bit_depth, range);
    println!("Inverse {:?}", inverse);
    let inverse_integral = get_inverse_transform_integral(kr_kb.kr, kr_kb.kb, bit_depth, 13, range);
    println!("Inverse Integral {:?};", inverse_integral);
}
