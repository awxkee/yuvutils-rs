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
    yg: 4809,
    yb: 934,
    cb_r: -1382,
    cb_g: -2714,
    cb_b: 4096,
    cr_r: 4096,
    cr_g: -3430,
    cr_b: -666,
};

static FORWARD_BT601_LIMITED_8_13PREC: CbCrForwardTransform<i32> = CbCrForwardTransform {
    yr: 2104,
    yg: 4130,
    yb: 802,
    cb_r: -1214,
    cb_g: -2384,
    cb_b: 3598,
    cr_r: 3598,
    cr_g: -3013,
    cr_b: -585,
};

static FORWARD_BT709_LIMITED_8_13PREC: CbCrForwardTransform<i32> = CbCrForwardTransform {
    yr: 1496,
    yg: 5032,
    yb: 508,
    cb_r: -824,
    cb_g: -2774,
    cb_b: 3598,
    cr_r: 3598,
    cr_g: -3268,
    cr_b: -330,
};

static FORWARD_BT709_FULL_8_13PREC: CbCrForwardTransform<i32> = CbCrForwardTransform {
    yr: 1742,
    yg: 5859,
    yb: 591,
    cb_r: -939,
    cb_g: -3157,
    cb_b: 4096,
    cr_r: 4096,
    cr_g: -3720,
    cr_b: -376,
};

static FORWARD_BT2020_LIMITED_8_13PREC: CbCrForwardTransform<i32> = CbCrForwardTransform {
    yr: 1848,
    yg: 4770,
    yb: 417,
    cb_r: -1005,
    cb_g: -2593,
    cb_b: 3598,
    cr_r: 3598,
    cr_g: -3309,
    cr_b: -289,
};

static FORWARD_BT2020_FULL_8_13PREC: CbCrForwardTransform<i32> = CbCrForwardTransform {
    yr: 2152,
    yg: 5554,
    yb: 486,
    cb_r: -1144,
    cb_g: -2952,
    cb_b: 4096,
    cr_r: 4096,
    cr_g: -3767,
    cr_b: -329,
};

static FORWARD_SMPTE240_LIMITED_8_13PREC: CbCrForwardTransform<i32> = CbCrForwardTransform {
    yr: 612,
    yg: 4932,
    yb: 1492,
    cb_r: -397,
    cb_g: -3201,
    cb_b: 3598,
    cr_r: 3598,
    cr_g: -2763,
    cr_b: -835,
};

static FORWARD_SMPTE240_FULL_8_13PREC: CbCrForwardTransform<i32> = CbCrForwardTransform {
    yr: 713,
    yg: 5743,
    yb: 1737,
    cb_r: -452,
    cb_g: -3644,
    cb_b: 4096,
    cr_r: 4096,
    cr_g: -3145,
    cr_b: -951,
};

static FORWARD_BT470_FULL_8_13PREC: CbCrForwardTransform<i32> = CbCrForwardTransform {
    yr: 1819,
    yg: 5789,
    yb: 584,
    cb_r: -979,
    cb_g: -3117,
    cb_b: 4096,
    cr_r: 4096,
    cr_g: -3721,
    cr_b: -375,
};

static FORWARD_BT470_LIMITED_8_13PREC: CbCrForwardTransform<i32> = CbCrForwardTransform {
    yr: 1562,
    yg: 4972,
    yb: 502,
    cb_r: -860,
    cb_g: -2738,
    cb_b: 3598,
    cr_r: 3598,
    cr_g: -3268,
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
    y_coef: 9539,
    cr_coef: 13075,
    cb_coef: 16525,
    g_coeff_1: 6660,
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
    y_coef: 9539,
    cr_coef: 14686,
    cb_coef: 17305,
    g_coeff_1: 4366,
    g_coeff_2: 1747,
};

static INVERSE_BT709_FULL_8_PREC13: CbCrInverseTransform<i32> = CbCrInverseTransform {
    y_coef: 8192,
    cr_coef: 12901,
    cb_coef: 15201,
    g_coeff_1: 3835,
    g_coeff_2: 1535,
};

static INVERSE_BT2020_LIMITED_8_PREC13: CbCrInverseTransform<i32> = CbCrInverseTransform {
    y_coef: 9539,
    cr_coef: 13752,
    cb_coef: 17545,
    g_coeff_1: 5328,
    g_coeff_2: 1535,
};

static INVERSE_BT2020_FULL_8_PREC13: CbCrInverseTransform<i32> = CbCrInverseTransform {
    y_coef: 8192,
    cr_coef: 12080,
    cb_coef: 15412,
    g_coeff_1: 4681,
    g_coeff_2: 1348,
};

static INVERSE_SMPTE240_LIMITED_8_PREC13: CbCrInverseTransform<i32> = CbCrInverseTransform {
    y_coef: 9539,
    cr_coef: 17029,
    cb_coef: 14697,
    g_coeff_1: 2113,
    g_coeff_2: 4445,
};

static INVERSE_SMPTE240_FULL_8_PREC13: CbCrInverseTransform<i32> = CbCrInverseTransform {
    y_coef: 8192,
    cr_coef: 14959,
    cb_coef: 12911,
    g_coeff_1: 1856,
    g_coeff_2: 3904,
};

static INVERSE_BT470_LIMITED_8_PREC13: CbCrInverseTransform<i32> = CbCrInverseTransform {
    y_coef: 9539,
    cr_coef: 14511,
    cb_coef: 17322,
    g_coeff_1: 4558,
    g_coeff_2: 1748,
};

static INVERSE_BT470_FULL_8_PREC13: CbCrInverseTransform<i32> = CbCrInverseTransform {
    y_coef: 8192,
    cr_coef: 12747,
    cb_coef: 15216,
    g_coeff_1: 4004,
    g_coeff_2: 1535,
};

static INVERSE_BT601_LIMITED_10_PREC13: CbCrInverseTransform<i32> = CbCrInverseTransform {
    y_coef: 9567,
    cr_coef: 13113,
    cb_coef: 16574,
    g_coeff_1: 6679,
    g_coeff_2: 3219,
};

static INVERSE_BT601_FULL_10_PREC13: CbCrInverseTransform<i32> = CbCrInverseTransform {
    y_coef: 8192,
    cr_coef: 11485,
    cb_coef: 14516,
    g_coeff_1: 5850,
    g_coeff_2: 2819,
};

static INVERSE_BT709_LIMITED_10_PREC13: CbCrInverseTransform<i32> = CbCrInverseTransform {
    y_coef: 9567,
    cr_coef: 14729,
    cb_coef: 17356,
    g_coeff_1: 4378,
    g_coeff_2: 1752,
};

static INVERSE_BT709_FULL_10_PREC13: CbCrInverseTransform<i32> = CbCrInverseTransform {
    y_coef: 8192,
    cr_coef: 12901,
    cb_coef: 15201,
    g_coeff_1: 3835,
    g_coeff_2: 1535,
};

static INVERSE_BT2020_LIMITED_10_PREC13: CbCrInverseTransform<i32> = CbCrInverseTransform {
    y_coef: 9567,
    cr_coef: 13792,
    cb_coef: 17597,
    g_coeff_1: 5344,
    g_coeff_2: 1539,
};

static INVERSE_BT2020_FULL_10_PREC13: CbCrInverseTransform<i32> = CbCrInverseTransform {
    y_coef: 8192,
    cr_coef: 12080,
    cb_coef: 15412,
    g_coeff_1: 4681,
    g_coeff_2: 1348,
};

static INVERSE_BT601_LIMITED_12_PREC13: CbCrInverseTransform<i32> = CbCrInverseTransform {
    y_coef: 9574,
    cr_coef: 13123,
    cb_coef: 16586,
    g_coeff_1: 6684,
    g_coeff_2: 3221,
};

static INVERSE_BT601_FULL_12_PREC13: CbCrInverseTransform<i32> = CbCrInverseTransform {
    y_coef: 8192,
    cr_coef: 11485,
    cb_coef: 14516,
    g_coeff_1: 5850,
    g_coeff_2: 2819,
};

static INVERSE_BT709_LIMITED_12_PREC13: CbCrInverseTransform<i32> = CbCrInverseTransform {
    y_coef: 9574,
    cr_coef: 14740,
    cb_coef: 17368,
    g_coeff_1: 4382,
    g_coeff_2: 1753,
};

static INVERSE_BT709_FULL_12_PREC13: CbCrInverseTransform<i32> = CbCrInverseTransform {
    y_coef: 8192,
    cr_coef: 12901,
    cb_coef: 15201,
    g_coeff_1: 3835,
    g_coeff_2: 1535,
};

static INVERSE_BT2020_LIMITED_12_PREC13: CbCrInverseTransform<i32> = CbCrInverseTransform {
    y_coef: 9574,
    cr_coef: 13802,
    cb_coef: 17610,
    g_coeff_1: 5348,
    g_coeff_2: 1540,
};

static INVERSE_BT2020_FULL_12_PREC13: CbCrInverseTransform<i32> = CbCrInverseTransform {
    y_coef: 8192,
    cr_coef: 12080,
    cb_coef: 15412,
    g_coeff_1: 4681,
    g_coeff_2: 1348,
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
    } else if bit_depth == 10 {
        if matrix == YuvStandardMatrix::Bt601 {
            return match range {
                YuvRange::Limited => Some(INVERSE_BT601_LIMITED_10_PREC13),
                YuvRange::Full => Some(INVERSE_BT601_FULL_10_PREC13),
            };
        } else if matrix == YuvStandardMatrix::Bt709 {
            return match range {
                YuvRange::Limited => Some(INVERSE_BT709_LIMITED_10_PREC13),
                YuvRange::Full => Some(INVERSE_BT709_FULL_10_PREC13),
            };
        } else if matrix == YuvStandardMatrix::Bt2020 {
            return match range {
                YuvRange::Limited => Some(INVERSE_BT2020_LIMITED_10_PREC13),
                YuvRange::Full => Some(INVERSE_BT2020_FULL_10_PREC13),
            };
        }
    } else if bit_depth == 12 {
        if matrix == YuvStandardMatrix::Bt601 {
            return match range {
                YuvRange::Limited => Some(INVERSE_BT601_LIMITED_12_PREC13),
                YuvRange::Full => Some(INVERSE_BT601_FULL_12_PREC13),
            };
        } else if matrix == YuvStandardMatrix::Bt709 {
            return match range {
                YuvRange::Limited => Some(INVERSE_BT709_LIMITED_12_PREC13),
                YuvRange::Full => Some(INVERSE_BT709_FULL_12_PREC13),
            };
        } else if matrix == YuvStandardMatrix::Bt2020 {
            return match range {
                YuvRange::Limited => Some(INVERSE_BT2020_LIMITED_12_PREC13),
                YuvRange::Full => Some(INVERSE_BT2020_FULL_12_PREC13),
            };
        }
    }
    None
}

// #[cfg(test)]
// mod tests {
//     use super::*;
//
//     fn sqrdml(l: i32, p: i32) -> i32 {
//         ((l << 2) * p + (1 << 14)) >> 15
//     }
//
//     #[test]
//     fn check_forward_y_convergence() {
//         let r = 255;
//         let g = 255;
//         let b = 255;
//         let weights_full = [
//             FORWARD_BT601_FULL_8_13PREC,
//             FORWARD_BT709_FULL_8_13PREC,
//             FORWARD_BT2020_FULL_8_13PREC,
//         ];
//         for weights in weights_full {
//             let default_mul = (r * weights.yr + g * weights.yg + b * weights.yb + (1 << 12)) >> 13;
//             assert_eq!(default_mul, 255, "Failed on weights {:?}", weights);
//         }
//
//         let weights_limited = [
//             FORWARD_BT601_LIMITED_8_13PREC,
//             FORWARD_BT709_LIMITED_8_13PREC,
//             FORWARD_BT2020_LIMITED_8_13PREC,
//         ];
//
//         for weights in weights_limited {
//             let default_mul =
//                 (r * weights.yr + g * weights.yg + b * weights.yb + (1 << 12) + 16 * (1 << 13))
//                     >> 13;
//             assert_eq!(default_mul, 235, "Failed on weights {:?}", weights);
//         }
//     }
// }
