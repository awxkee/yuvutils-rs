#[derive(Copy, Clone)]
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
        return CbCrInverseTransform {
            y_coef,
            cr_coef,
            cb_coef,
            g_coeff_1,
            g_coeff_2,
        };
    }
}

impl CbCrInverseTransform<f32> {
    pub fn to_integers(&self, precision: u32) -> CbCrInverseTransform<i32> {
        let precision_scale: i32 = 1i32 << (precision as i32);
        let cr_coef = (self.cr_coef * precision_scale as f32).round() as i32;
        let cb_coef = (self.cb_coef * precision_scale as f32).round() as i32;
        let y_coef = (self.y_coef * precision_scale as f32).round() as i32;
        let g_coef_1 = (self.g_coeff_1 * precision_scale as f32).round() as i32;
        let g_coef_2 = (self.g_coeff_2 * precision_scale as f32).round() as i32;
        CbCrInverseTransform::<i32> {
            y_coef,
            cr_coef,
            cb_coef,
            g_coeff_1: g_coef_1,
            g_coeff_2: g_coef_2,
        }
    }
}

pub fn get_inverse_transform(
    range_bgra: u32,
    range_y: u32,
    range_uv: u32,
    kr: f32,
    kb: f32,
) -> CbCrInverseTransform<f32> {
    let range_uv = range_bgra as f32 / range_uv as f32;
    let y_coef = range_bgra as f32 / range_y as f32;
    let cr_coeff = (2f32 * (1f32 - kr)) * range_uv;
    let cb_coeff = (2f32 * (1f32 - kb)) * range_uv;
    let kg = 1.0f32 - kr - kb;
    if kg == 0f32 {
        panic!("1.0f - kr - kg must not be 0");
    }
    let g_coeff_1 = (2f32 * ((1f32 - kr) * kr / kg)) * range_uv;
    let g_coeff_2 = (2f32 * ((1f32 - kb) * kb / kg)) * range_uv;
    return CbCrInverseTransform::new(y_coef, cr_coeff, cb_coeff, g_coeff_1, g_coeff_2);
}

#[repr(C)]
#[derive(Copy, Clone, PartialOrd, PartialEq)]
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

pub trait ToIntegerTransform {
    fn to_integers(&self, precision: u32) -> CbCrForwardTransform<i32>;
}

impl ToIntegerTransform for CbCrForwardTransform<f32> {
    fn to_integers(&self, precision: u32) -> CbCrForwardTransform<i32> {
        let scale = (1 << precision) as f32;
        return CbCrForwardTransform::<i32> {
            yr: (self.yr * scale).round() as i32,
            yg: (self.yg * scale).round() as i32,
            yb: (self.yb * scale).round() as i32,
            cb_r: (self.cb_r * scale).round() as i32,
            cb_g: (self.cb_g * scale).round() as i32,
            cb_b: (self.cb_b * scale).round() as i32,
            cr_r: (self.cr_r * scale).round() as i32,
            cr_g: (self.cr_g * scale).round() as i32,
            cr_b: (self.cr_b * scale).round() as i32,
        };
    }
}

pub fn get_forward_transform(
    range_bgra: u32,
    range_y: u32,
    range_uv: u32,
    kr: f32,
    kb: f32,
) -> CbCrForwardTransform<f32> {
    let kg = 1.0f32 - kr - kb;
    if kg == 0f32 {
        panic!("1.0f - kr - kg must not be 0");
    }

    let yr = kr * range_y as f32 / range_bgra as f32;
    let yg = kg * range_y as f32 / range_bgra as f32;
    let yb = kb * range_y as f32 / range_bgra as f32;

    let cb_r = -0.5f32 * kr / (1f32 - kb) * range_uv as f32 / range_bgra as f32;
    let cb_g = -0.5f32 * kg / (1f32 - kb) * range_uv as f32 / range_bgra as f32;
    let cb_b = 0.5f32 * range_uv as f32 / range_bgra as f32;

    let cr_r = 0.5f32 * range_uv as f32 / range_bgra as f32;
    let cr_g = -0.5f32 * kg / (1f32 - kr) * range_uv as f32 / range_bgra as f32;
    let cr_b = -0.5f32 * kb / (1f32 - kr) * range_uv as f32 / range_bgra as f32;
    return CbCrForwardTransform {
        yr,
        yg,
        yb,
        cb_r,
        cb_g,
        cb_b,
        cr_r,
        cr_g,
        cr_b,
    };
}

#[repr(C)]
#[derive(Copy, Clone, PartialOrd, PartialEq)]
pub enum YuvRange {
    TV,
    Full,
}

#[derive(Copy, Clone, PartialOrd, PartialEq)]
pub struct YuvChromaRange {
    pub bias_y: u32,
    pub bias_uv: u32,
    pub range_y: u32,
    pub range_uv: u32,
    pub range: YuvRange,
}

pub fn get_yuv_range(depth: u32, range: YuvRange) -> YuvChromaRange {
    return match range {
        YuvRange::TV => YuvChromaRange {
            bias_y: 16 << (depth - 8),
            bias_uv: 1 << (depth - 1),
            range_y: 219 << (depth - 8),
            range_uv: 224 << (depth - 8),
            range,
        },
        YuvRange::Full => YuvChromaRange {
            bias_y: 0,
            bias_uv: 1 << (depth - 1),
            range_uv: 2f32.powi(depth as i32) as u32 - 1,
            range_y: 2f32.powi(depth as i32) as u32 - 1,
            range,
        },
    };
}

#[repr(C)]
#[derive(Copy, Clone, PartialOrd, PartialEq)]
pub enum YuvStandardMatrix {
    Bt601,
    Bt709,
    Bt2020,
    Smpte240,
}

#[derive(Copy, Clone)]
pub struct YuvBias {
    pub kr: f32,
    pub kb: f32,
}

pub const fn get_kr_kb(matrix: YuvStandardMatrix) -> YuvBias {
    return match matrix {
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
    };
}

#[repr(u8)]
#[derive(Copy, Clone, PartialEq, Eq)]
pub enum YuvNVOrder {
    UV = 0,
    VU = 1,
}

impl From<u8> for YuvNVOrder {
    #[inline(always)]
    fn from(value: u8) -> Self {
        match value {
            0 => YuvNVOrder::UV,
            1 => YuvNVOrder::VU,
            _ => {
                panic!("Unknown value")
            }
        }
    }
}

#[repr(u8)]
#[derive(Copy, Clone, PartialEq, Eq)]
pub enum YuvChromaSample {
    YUV420 = 0,
    YUV422 = 1,
    YUV444 = 2,
}

impl From<u8> for YuvChromaSample {
    #[inline(always)]
    fn from(value: u8) -> Self {
        match value {
            0 => YuvChromaSample::YUV420,
            1 => YuvChromaSample::YUV422,
            2 => YuvChromaSample::YUV444,
            _ => {
                panic!("Unknown value")
            }
        }
    }
}

#[repr(u8)]
#[derive(Copy, Clone, PartialEq, Eq)]
pub enum YuvEndian {
    BigEndian = 0,
    LittleEndian = 1,
}

impl From<u8> for YuvEndian {
    #[inline(always)]
    fn from(value: u8) -> Self {
        match value {
            0 => YuvEndian::BigEndian,
            1 => YuvEndian::LittleEndian,
            _ => {
                panic!("Unknown value")
            }
        }
    }
}

#[repr(u8)]
#[derive(Copy, Clone, PartialEq, Eq)]
pub enum YuvBytesPosition {
    MostSignificantBytes = 0,
    LeastSignificantBytes = 1,
}

impl From<u8> for YuvBytesPosition {
    #[inline(always)]
    fn from(value: u8) -> Self {
        match value {
            0 => YuvBytesPosition::MostSignificantBytes,
            1 => YuvBytesPosition::LeastSignificantBytes,
            _ => {
                panic!("Unknown value")
            }
        }
    }
}

#[repr(u8)]
#[derive(Copy, Clone, PartialEq, Eq)]
pub enum YuvSourceChannels {
    Rgb = 0,
    Rgba = 1,
    Bgra = 2,
}

impl From<u8> for YuvSourceChannels {
    #[inline(always)]
    fn from(value: u8) -> Self {
        match value {
            0 => YuvSourceChannels::Rgb,
            1 => YuvSourceChannels::Rgba,
            2 => YuvSourceChannels::Bgra,
            _ => {
                panic!("Unknown value")
            }
        }
    }
}

impl YuvSourceChannels {
    #[inline(always)]
    pub fn get_channels_count(&self) -> usize {
        match self {
            YuvSourceChannels::Rgb => 3,
            YuvSourceChannels::Rgba | YuvSourceChannels::Bgra => 4,
        }
    }

    #[inline(always)]
    pub fn has_alpha(&self) -> bool {
        match self {
            YuvSourceChannels::Rgb => false,
            YuvSourceChannels::Rgba | YuvSourceChannels::Bgra => true,
        }
    }
}

impl YuvSourceChannels {
    #[inline(always)]
    pub fn get_r_channel_offset(&self) -> usize {
        match self {
            YuvSourceChannels::Rgb => 0,
            YuvSourceChannels::Rgba => 0,
            YuvSourceChannels::Bgra => 2,
        }
    }

    #[inline(always)]
    pub fn get_g_channel_offset(&self) -> usize {
        match self {
            YuvSourceChannels::Rgb => 1,
            YuvSourceChannels::Rgba | YuvSourceChannels::Bgra => 1,
        }
    }

    #[inline(always)]
    pub fn get_b_channel_offset(&self) -> usize {
        match self {
            YuvSourceChannels::Rgb => 2,
            YuvSourceChannels::Rgba => 2,
            YuvSourceChannels::Bgra => 0,
        }
    }
    #[inline(always)]
    pub fn get_a_channel_offset(&self) -> usize {
        match self {
            YuvSourceChannels::Rgb => 0,
            YuvSourceChannels::Rgba | YuvSourceChannels::Bgra => 3,
        }
    }
}
