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
use crate::yuv_support::{CbCrForwardTransform, CbCrInverseTransform};
use crate::{YuvRange, YuvStandardMatrix};

static FORWARD_BT601_FULL_8_13PREC: CbCrForwardTransform<i32> = CbCrForwardTransform {
    yr: 2449,
    yg: 4808,
    yb: 933,
    cb_r: -1383,
    cb_g: -2714,
    cb_b: 4096,
    cr_r: 4096,
    cr_g: -3430,
    cr_b: -667,
};

static FORWARD_BT601_LIMITED_8_13PREC: CbCrForwardTransform<i32> = CbCrForwardTransform {
    yr: 2103,
    yg: 4129,
    yb: 802,
    cb_r: -1215,
    cb_g: -2384,
    cb_b: 3598,
    cr_r: 3598,
    cr_g: -3013,
    cr_b: -586,
};

static FORWARD_BT709_LIMITED_8_13PREC: CbCrForwardTransform<i32> = CbCrForwardTransform {
    yr: 1495,
    yg: 5031,
    yb: 507,
    cb_r: -825,
    cb_g: -2774,
    cb_b: 3598,
    cr_r: 3598,
    cr_g: -3269,
    cr_b: -330,
};

static FORWARD_BT709_FULL_8_13PREC: CbCrForwardTransform<i32> = CbCrForwardTransform {
    yr: 1741,
    yg: 5858,
    yb: 591,
    cb_r: -939,
    cb_g: -3158,
    cb_b: 4096,
    cr_r: 4096,
    cr_g: -3721,
    cr_b: -376,
};

static FORWARD_BT2020_LIMITED_8_13PREC: CbCrForwardTransform<i32> = CbCrForwardTransform {
    yr: 1848,
    yg: 4770,
    yb: 417,
    cb_r: -1005,
    cb_g: -2594,
    cb_b: 3598,
    cr_r: 3598,
    cr_g: -3309,
    cr_b: -290,
};

static FORWARD_BT2020_FULL_8_13PREC: CbCrForwardTransform<i32> = CbCrForwardTransform {
    yr: 2152,
    yg: 5554,
    yb: 485,
    cb_r: -1144,
    cb_g: -2953,
    cb_b: 4096,
    cr_r: 4096,
    cr_g: -3767,
    cr_b: -330,
};

static FORWARD_SMPTE240_LIMITED_8_13PREC: CbCrForwardTransform<i32> = CbCrForwardTransform {
    yr: 612,
    yg: 4931,
    yb: 1491,
    cb_r: -398,
    cb_g: -3201,
    cb_b: 3598,
    cr_r: 3598,
    cr_g: -2763,
    cr_b: -836,
};

static FORWARD_SMPTE240_FULL_8_13PREC: CbCrForwardTransform<i32> = CbCrForwardTransform {
    yr: 712,
    yg: 5742,
    yb: 1736,
    cb_r: -453,
    cb_g: -3644,
    cb_b: 4096,
    cr_r: 4096,
    cr_g: -3145,
    cr_b: -952,
};

static FORWARD_BT470_FULL_8_13PREC: CbCrForwardTransform<i32> = CbCrForwardTransform {
    yr: 1818,
    yg: 5789,
    yb: 584,
    cb_r: -980,
    cb_g: -3117,
    cb_b: 4096,
    cr_r: 4096,
    cr_g: -3721,
    cr_b: -376,
};

static FORWARD_BT470_LIMITED_8_13PREC: CbCrForwardTransform<i32> = CbCrForwardTransform {
    yr: 1561,
    yg: 4971,
    yb: 501,
    cb_r: -861,
    cb_g: -2738,
    cb_b: 3598,
    cr_r: 3598,
    cr_g: -3269,
    cr_b: -330,
};

pub(crate) fn get_built_forward_transform(
    prec: u32,
    bit_depth: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) -> Option<CbCrForwardTransform<i32>> {
    if prec != 13 {
        return None;
    }
    if bit_depth == 8 {
        if matrix == YuvStandardMatrix::Bt601 {
            return match range {
                YuvRange::Limited => Some(FORWARD_BT601_LIMITED_8_13PREC),
                YuvRange::Full => Some(FORWARD_BT601_FULL_8_13PREC),
            };
        } else if matrix == YuvStandardMatrix::Bt709 {
            return match range {
                YuvRange::Full => Some(FORWARD_BT709_FULL_8_13PREC),
                YuvRange::Limited => Some(FORWARD_BT709_LIMITED_8_13PREC),
            };
        } else if matrix == YuvStandardMatrix::Bt2020 {
            return match range {
                YuvRange::Full => Some(FORWARD_BT2020_FULL_8_13PREC),
                YuvRange::Limited => Some(FORWARD_BT2020_LIMITED_8_13PREC),
            };
        } else if matrix == YuvStandardMatrix::Smpte240 {
            return match range {
                YuvRange::Full => Some(FORWARD_SMPTE240_FULL_8_13PREC),
                YuvRange::Limited => Some(FORWARD_SMPTE240_LIMITED_8_13PREC),
            };
        } else if matrix == YuvStandardMatrix::Bt470_6 {
            return match range {
                YuvRange::Full => Some(FORWARD_BT470_FULL_8_13PREC),
                YuvRange::Limited => Some(FORWARD_BT470_LIMITED_8_13PREC),
            };
        }
    }
    None
}

static INVERSE_BT601_LIMITED_8_PREC13: CbCrInverseTransform<i32> = CbCrInverseTransform {
    y_coef: 9538,
    cr_coef: 13074,
    cb_coef: 16525,
    g_coeff_1: 6659,
    g_coeff_2: 3209,
};

static INVERSE_BT601_FULL_8_PREC13: CbCrInverseTransform<i32> = CbCrInverseTransform {
    y_coef: 8192,
    cr_coef: 11485,
    cb_coef: 14516,
    g_coeff_1: 5850,
    g_coeff_2: 2819,
};

static INVERSE_BT709_LIMITED_8_PREC13: CbCrInverseTransform<i32> = CbCrInverseTransform {
    y_coef: 9538,
    cr_coef: 14686,
    cb_coef: 17304,
    g_coeff_1: 4365,
    g_coeff_2: 1746,
};

static INVERSE_BT709_FULL_8_PREC13: CbCrInverseTransform<i32> = CbCrInverseTransform {
    y_coef: 8192,
    cr_coef: 12900,
    cb_coef: 15201,
    g_coeff_1: 3834,
    g_coeff_2: 1534,
};

static INVERSE_BT2020_LIMITED_8_PREC13: CbCrInverseTransform<i32> = CbCrInverseTransform {
    y_coef: 9538,
    cr_coef: 13751,
    cb_coef: 17545,
    g_coeff_1: 5328,
    g_coeff_2: 1534,
};

static INVERSE_BT2020_FULL_8_PREC13: CbCrInverseTransform<i32> = CbCrInverseTransform {
    y_coef: 8192,
    cr_coef: 12079,
    cb_coef: 15412,
    g_coeff_1: 4680,
    g_coeff_2: 1348,
};

static INVERSE_SMPTE240_LIMITED_8_PREC13: CbCrInverseTransform<i32> = CbCrInverseTransform {
    y_coef: 9538,
    cr_coef: 17028,
    cb_coef: 14697,
    g_coeff_1: 2113,
    g_coeff_2: 4444,
};

static INVERSE_SMPTE240_FULL_8_PREC13: CbCrInverseTransform<i32> = CbCrInverseTransform {
    y_coef: 8192,
    cr_coef: 14958,
    cb_coef: 12910,
    g_coeff_1: 1856,
    g_coeff_2: 3904,
};

static INVERSE_BT470_LIMITED_8_PREC13: CbCrInverseTransform<i32> = CbCrInverseTransform {
    y_coef: 9538,
    cr_coef: 14510,
    cb_coef: 17321,
    g_coeff_1: 4558,
    g_coeff_2: 1747,
};

static INVERSE_BT470_FULL_8_PREC13: CbCrInverseTransform<i32> = CbCrInverseTransform {
    y_coef: 8192,
    cr_coef: 12746,
    cb_coef: 15215,
    g_coeff_1: 4004,
    g_coeff_2: 1535,
};

pub(crate) fn get_built_inverse_transform(
    prec: u32,
    bit_depth: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) -> Option<CbCrInverseTransform<i32>> {
    if prec != 13 {
        return None;
    }
    if bit_depth == 8 {
        if matrix == YuvStandardMatrix::Bt601 {
            return match range {
                YuvRange::Limited => Some(INVERSE_BT601_LIMITED_8_PREC13),
                YuvRange::Full => Some(INVERSE_BT601_FULL_8_PREC13),
            };
        } else if matrix == YuvStandardMatrix::Bt709 {
            return match range {
                YuvRange::Limited => Some(INVERSE_BT709_LIMITED_8_PREC13),
                YuvRange::Full => Some(INVERSE_BT709_FULL_8_PREC13),
            };
        } else if matrix == YuvStandardMatrix::Bt2020 {
            return match range {
                YuvRange::Limited => Some(INVERSE_BT2020_LIMITED_8_PREC13),
                YuvRange::Full => Some(INVERSE_BT2020_FULL_8_PREC13),
            };
        } else if matrix == YuvStandardMatrix::Smpte240 {
            return match range {
                YuvRange::Limited => Some(INVERSE_SMPTE240_LIMITED_8_PREC13),
                YuvRange::Full => Some(INVERSE_SMPTE240_FULL_8_PREC13),
            };
        } else if matrix == YuvStandardMatrix::Bt470_6 {
            return match range {
                YuvRange::Limited => Some(INVERSE_BT470_LIMITED_8_PREC13),
                YuvRange::Full => Some(INVERSE_BT470_FULL_8_PREC13),
            };
        }
    }
    None
}
