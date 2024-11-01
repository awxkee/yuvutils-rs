/*
 * // Copyright (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

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

impl CbCrInverseTransform<f32> {
    /// Integral transformation adds an error not less than 1%
    pub fn to_integers(self, precision: u32) -> CbCrInverseTransform<i32> {
        let precision_scale: i32 = 1i32 << (precision as i32);
        let cr_coef = (self.cr_coef * precision_scale as f32) as i32;
        let cb_coef = (self.cb_coef * precision_scale as f32) as i32;
        let y_coef = (self.y_coef * precision_scale as f32) as i32;
        let g_coef_1 = (self.g_coeff_1 * precision_scale as f32) as i32;
        let g_coef_2 = (self.g_coeff_2 * precision_scale as f32) as i32;
        CbCrInverseTransform::<i32> {
            y_coef,
            cr_coef,
            cb_coef,
            g_coeff_1: g_coef_1,
            g_coeff_2: g_coef_2,
        }
    }
}

/// Transformation RGB to YUV with coefficients as specified in [ITU-R](https://www.itu.int/rec/T-REC-H.273/en)
pub fn get_inverse_transform(
    range_bgra: u32,
    range_y: u32,
    range_uv: u32,
    kr: f32,
    kb: f32,
    precision: u32,
) -> Result<CbCrInverseTransform<i32>, String> {
    let range_uv = range_bgra as f32 / range_uv as f32;
    let y_coef = range_bgra as f32 / range_y as f32;
    let cr_coeff = (2f32 * (1f32 - kr)) * range_uv;
    let cb_coeff = (2f32 * (1f32 - kb)) * range_uv;
    let kg = 1.0f32 - kr - kb;
    if kg == 0f32 {
        return Err("1.0f - kr - kg must not be 0".parse().unwrap());
    }
    let g_coeff_1 = (2f32 * ((1f32 - kr) * kr / kg)) * range_uv;
    let g_coeff_2 = (2f32 * ((1f32 - kb) * kb / kg)) * range_uv;
    let exact_transform =
        CbCrInverseTransform::new(y_coef, cr_coeff, cb_coeff, g_coeff_1, g_coeff_2);
    Ok(exact_transform.to_integers(precision))
}

/// Transformation RGB to YUV with coefficients as specified in [ITU-R](https://www.itu.int/rec/T-REC-H.273/en)
pub fn get_exact_inverse_transform(
    range_bgra: u32,
    range_y: u32,
    range_uv: u32,
    kr: f32,
    kb: f32,
) -> Result<CbCrInverseTransform<f32>, String> {
    let range_uv = range_bgra as f32 / range_uv as f32;
    let y_coef = range_bgra as f32 / range_y as f32;
    let cr_coeff = (2f32 * (1f32 - kr)) * range_uv;
    let cb_coeff = (2f32 * (1f32 - kb)) * range_uv;
    let kg = 1.0f32 - kr - kb;
    if kg == 0f32 {
        return Err("1.0f - kr - kg must not be 0".parse().unwrap());
    }
    let g_coeff_1 = (2f32 * ((1f32 - kr) * kr / kg)) * range_uv;
    let g_coeff_2 = (2f32 * ((1f32 - kb) * kb / kg)) * range_uv;
    let exact_transform =
        CbCrInverseTransform::new(y_coef, cr_coeff, cb_coeff, g_coeff_1, g_coeff_2);
    Ok(exact_transform)
}

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialOrd, PartialEq)]
/// Declares YUV range TV (limited) or Full, more info [ITU-R](https://www.itu.int/rec/T-REC-H.273/en)
pub enum YuvRange {
    /// Limited range Y ∈ [16 << (depth - 8), 16 << (depth - 8) + 224 << (depth - 8)], UV ∈ [-1 << (depth - 1), -1 << (depth - 1) + 1 << (depth - 1)]
    Tv,
    /// Full range Y ∈ [0, 2^bit_depth - 1], UV ∈ [-1 << (depth - 1), -1 << (depth - 1) + 2^bit_depth - 1]
    Pc,
}

#[derive(Debug, Copy, Clone, PartialOrd, PartialEq)]
pub struct YuvChromaRange {
    pub bias_y: u32,
    pub bias_uv: u32,
    pub range_y: u32,
    pub range_uv: u32,
    pub range: YuvRange,
}

pub const fn get_yuv_range(depth: u32, range: YuvRange) -> YuvChromaRange {
    match range {
        YuvRange::Tv => YuvChromaRange {
            bias_y: 16 << (depth - 8),
            bias_uv: 1 << (depth - 1),
            range_y: 219 << (depth - 8),
            range_uv: 224 << (depth - 8),
            range,
        },
        YuvRange::Pc => YuvChromaRange {
            bias_y: 0,
            bias_uv: 1 << (depth - 1),
            range_uv: (1 << depth) - 1,
            range_y: (1 << depth) - 1,
            range,
        },
    }
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

pub const fn get_kr_kb(matrix: YuvStandardMatrix) -> YuvBias {
    match matrix {
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
