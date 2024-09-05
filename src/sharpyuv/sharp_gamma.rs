/*
 * // Copyright 2024 (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

#[inline]
/// Linear transfer function for sRGB
pub fn srgb_to_linear(gamma: f32) -> f32 {
    if gamma < 0f32 {
        0f32
    } else if gamma < 12.92f32 * 0.003_041_282_5_f32 {
        gamma * (1f32 / 12.92f32)
    } else if gamma < 1.0f32 {
        ((gamma + 0.055_010_717_f32) / 1.055_010_7_f32).powf(2.4f32)
    } else {
        1.0f32
    }
}

#[inline]
/// Gamma transfer function for sRGB
pub fn srgb_from_linear(linear: f32) -> f32 {
    if linear < 0.0f32 {
        0.0f32
    } else if linear < 0.003_041_282_5_f32 {
        linear * 12.92f32
    } else if linear < 1.0f32 {
        1.055_010_7_f32 * linear.powf(1.0f32 / 2.4f32) - 0.055_010_717_f32
    } else {
        1.0f32
    }
}

#[inline]
/// Linear transfer function for Rec.709
pub fn rec709_to_linear(gamma: f32) -> f32 {
    if gamma < 0.0f32 {
        0.0f32
    } else if gamma < 4.5f32 * 0.018_053_97_f32 {
        gamma * (1f32 / 4.5f32)
    } else if gamma < 1.0f32 {
        ((gamma + 0.099_296_82_f32) / 1.099_296_8_f32).powf(1.0f32 / 0.45f32)
    } else {
        1.0f32
    }
}

#[inline]
/// Gamma transfer function for Rec.709
pub fn rec709_from_linear(linear: f32) -> f32 {
    if linear < 0.0f32 {
        0.0f32
    } else if linear < 0.018_053_97_f32 {
        linear * 4.5f32
    } else if linear < 1.0f32 {
        1.099_296_8_f32 * linear.powf(0.45f32) - 0.099_296_82_f32
    } else {
        1.0f32
    }
}

#[inline(always)]
/// Pure gamma transfer function for gamma 2.2
pub fn pure_gamma_function(x: f32, gamma: f32) -> f32 {
    if x <= 0f32 {
        0f32
    } else if x >= 1f32 {
        return 1f32;
    } else {
        return x.powf(gamma);
    }
}

#[inline]
/// Pure gamma transfer function for gamma 2.2
pub fn gamma2p2_from_linear(linear: f32) -> f32 {
    pure_gamma_function(linear, 1f32 / 2.2f32)
}

#[inline]
/// Linear transfer function for gamma 2.2
pub fn gamma2p2_to_linear(gamma: f32) -> f32 {
    pure_gamma_function(gamma, 2.2f32)
}

#[inline]
/// Pure gamma transfer function for gamma 2.8
pub fn gamma2p8_from_linear(linear: f32) -> f32 {
    pure_gamma_function(linear, 1f32 / 2.8f32)
}

#[inline]
/// Linear transfer function for gamma 2.8
pub fn gamma2p8_to_linear(gamma: f32) -> f32 {
    pure_gamma_function(gamma, 2.8f32)
}

#[derive(Debug, Copy, Clone, Ord, PartialOrd, Eq, PartialEq)]
/// Declares transfer function for transfer components into a linear colorspace and its inverse
pub enum SharpYuvGammaTransfer {
    /// sRGB Transfer function
    Srgb,
    /// Rec.709 Transfer function
    Rec709,
    /// Pure gamma 2.2 Transfer function
    Gamma2p2,
    /// Pure gamma 2.8 Transfer function
    Gamma2p8,
}

impl From<u8> for SharpYuvGammaTransfer {
    fn from(value: u8) -> Self {
        match value {
            0 => SharpYuvGammaTransfer::Srgb,
            1 => SharpYuvGammaTransfer::Rec709,
            2 => SharpYuvGammaTransfer::Gamma2p2,
            3 => SharpYuvGammaTransfer::Gamma2p8,
            _ => SharpYuvGammaTransfer::Srgb,
        }
    }
}

impl SharpYuvGammaTransfer {
    #[inline]
    pub fn get_linearize_function(&self) -> fn(f32) -> f32 {
        match self {
            SharpYuvGammaTransfer::Srgb => srgb_to_linear,
            SharpYuvGammaTransfer::Rec709 => rec709_to_linear,
            SharpYuvGammaTransfer::Gamma2p8 => gamma2p8_to_linear,
            SharpYuvGammaTransfer::Gamma2p2 => gamma2p2_to_linear,
        }
    }

    #[inline]
    pub fn get_gamma_function(&self) -> fn(f32) -> f32 {
        match self {
            SharpYuvGammaTransfer::Srgb => srgb_from_linear,
            SharpYuvGammaTransfer::Rec709 => rec709_from_linear,
            SharpYuvGammaTransfer::Gamma2p2 => gamma2p2_from_linear,
            SharpYuvGammaTransfer::Gamma2p8 => gamma2p8_from_linear,
        }
    }
}
