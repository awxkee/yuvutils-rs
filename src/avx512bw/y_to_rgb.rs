use crate::avx512bw::avx512_utils::{avx512_pack_u16, avx512_rgb_u8, avx512_rgba_u8};
use crate::yuv_support::{CbCrInverseTransform, YuvChromaRange, YuvSourceChannels};
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[cfg(feature = "nightly_avx512")]
#[inline(always)]
#[allow(dead_code)]
pub unsafe fn avx512_y_to_rgb_row<const DESTINATION_CHANNELS: u8>(
    range: &YuvChromaRange,
    transform: &CbCrInverseTransform<i32>,
    y_plane: &[u8],
    rgba: &mut [u8],
    start_cx: usize,
    y_offset: usize,
    rgba_offset: usize,
    width: usize,
) -> usize {
    let destination_channels: YuvSourceChannels = DESTINATION_CHANNELS.into();
    let channels = destination_channels.get_channels_count();

    let mut cx = start_cx;
    let y_ptr = y_plane.as_ptr();
    let rgba_ptr = rgba.as_mut_ptr();

    let y_corr = _mm512_set1_epi8(range.bias_y as i8);
    let v_luma_coeff = _mm512_set1_epi16(transform.y_coef as i16);
    let v_min_values = _mm512_setzero_si512();
    let v_alpha = _mm512_set1_epi8(255u8 as i8);

    while cx + 64 < width {
        let y_values = _mm512_subs_epi8(
            _mm512_loadu_si512(y_ptr.add(y_offset + cx) as *const i32),
            y_corr,
        );

        let y_high = _mm512_mullo_epi16(
            _mm512_cvtepu8_epi16(_mm512_extracti64x4_epi64::<1>(y_values)),
            v_luma_coeff,
        );

        let r_high = _mm512_srli_epi16::<6>(_mm512_max_epi16(y_high, v_min_values));

        let y_low = _mm512_mullo_epi16(
            _mm512_cvtepu8_epi16(_mm512_castsi512_si256(y_values)),
            v_luma_coeff,
        );

        let r_low = _mm512_srli_epi16::<6>(_mm512_max_epi16(y_low, v_min_values));

        let r_values = avx512_pack_u16(r_low, r_high);

        let dst_shift = rgba_offset + cx * channels;

        match destination_channels {
            YuvSourceChannels::Rgb => {
                // We need always to write 104 bytes, however 32 initial offset is safe only for 96, then if there are some exceed it is required to use transient buffer
                let ptr = rgba_ptr.add(dst_shift);
                avx512_rgb_u8(ptr, r_values, r_values, r_values);
            }
            YuvSourceChannels::Rgba => {
                avx512_rgba_u8(
                    rgba_ptr.add(dst_shift),
                    r_values,
                    r_values,
                    r_values,
                    v_alpha,
                );
            }
            YuvSourceChannels::Bgra => {
                avx512_rgba_u8(
                    rgba_ptr.add(dst_shift),
                    r_values,
                    r_values,
                    r_values,
                    v_alpha,
                );
            }
        }

        cx += 64;
    }

    return cx;
}
