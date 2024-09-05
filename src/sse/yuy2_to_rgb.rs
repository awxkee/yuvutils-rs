/*
 * // Copyright (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */
use crate::sse::{sse_deinterleave_rgba, sse_interleave_rgb, sse_interleave_rgba};
use crate::yuv_support::{
    CbCrInverseTransform, YuvChromaRange, YuvSourceChannels, Yuy2Description,
};
use crate::yuv_to_yuy2::YuvToYuy2Navigation;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

pub fn yuy2_to_rgb_sse<const DST_CHANNELS: u8, const YUY2_TARGET: usize>(
    range: &YuvChromaRange,
    transform: &CbCrInverseTransform<i32>,
    yuy2_store: &[u8],
    yuy2_offset: usize,
    rgb: &mut [u8],
    rgb_offset: usize,
    width: u32,
    nav: YuvToYuy2Navigation,
) -> YuvToYuy2Navigation {
    let yuy2_source: Yuy2Description = YUY2_TARGET.into();
    let dst_chans: YuvSourceChannels = DST_CHANNELS.into();

    let mut _cx = nav.cx;
    let mut _yuy2_x = nav.x;

    unsafe {
        let max_x_16 = (width as usize / 2).saturating_sub(16);
        let max_x_8 = (width as usize / 2).saturating_sub(8);

        let y_corr = _mm_set1_epi8(range.bias_y as i8);
        let uv_corr = _mm_set1_epi16(range.bias_uv as i16);
        let v_luma_coeff = _mm_set1_epi16(transform.y_coef as i16);
        let v_cr_coeff = _mm_set1_epi16(transform.cr_coef as i16);
        let v_cb_coeff = _mm_set1_epi16(transform.cb_coef as i16);
        let v_g_coeff_1 = _mm_set1_epi16(-(transform.g_coeff_1 as i16));
        let v_g_coeff_2 = _mm_set1_epi16(-(transform.g_coeff_2 as i16));
        let v_alpha = _mm_set1_epi8(255u8 as i8);
        let rounding_const = _mm_set1_epi16(1 << 5);

        let zeros = _mm_setzero_si128();

        for x in (_yuy2_x..max_x_16).step_by(16) {
            let yuy2_offset = yuy2_offset + x * 4;
            let dst_pos = rgb_offset + _cx * dst_chans.get_channels_count();
            let dst_ptr = rgb.as_mut_ptr().add(dst_pos);

            let yuy2_ptr = yuy2_store.as_ptr().add(yuy2_offset);

            let j0 = _mm_loadu_si128(yuy2_ptr as *const __m128i);
            let j1 = _mm_loadu_si128(yuy2_ptr.add(16) as *const __m128i);
            let j2 = _mm_loadu_si128(yuy2_ptr.add(32) as *const __m128i);
            let j3 = _mm_loadu_si128(yuy2_ptr.add(48) as *const __m128i);

            let pixel_set = sse_deinterleave_rgba(j0, j1, j2, j3);
            let mut y_first = match yuy2_source {
                Yuy2Description::YUYV | Yuy2Description::YVYU => pixel_set.0,
                Yuy2Description::UYVY | Yuy2Description::VYUY => pixel_set.1,
            };
            let mut y_second = match yuy2_source {
                Yuy2Description::YUYV | Yuy2Description::YVYU => pixel_set.2,
                Yuy2Description::UYVY | Yuy2Description::VYUY => pixel_set.3,
            };

            let y_first_reconstructed = _mm_unpacklo_epi8(y_first, y_second);
            let y_second_reconstructed = _mm_unpackhi_epi8(y_first, y_second);
            y_first = y_first_reconstructed;
            y_second = y_second_reconstructed;

            let u_value = match yuy2_source {
                Yuy2Description::YUYV => pixel_set.1,
                Yuy2Description::UYVY => pixel_set.0,
                Yuy2Description::YVYU => pixel_set.3,
                Yuy2Description::VYUY => pixel_set.2,
            };
            let v_value = match yuy2_source {
                Yuy2Description::YUYV => pixel_set.3,
                Yuy2Description::UYVY => pixel_set.2,
                Yuy2Description::YVYU => pixel_set.1,
                Yuy2Description::VYUY => pixel_set.0,
            };

            let low_u_value = _mm_unpacklo_epi8(u_value, u_value);
            let high_u_value = _mm_unpackhi_epi8(u_value, u_value);
            let low_v_value = _mm_unpacklo_epi8(v_value, v_value);
            let high_v_value = _mm_unpackhi_epi8(v_value, v_value);

            y_first = _mm_subs_epu8(y_first, y_corr);
            y_second = _mm_subs_epu8(y_second, y_corr);

            let u_l_h = _mm_sub_epi16(_mm_unpackhi_epi8(low_u_value, zeros), uv_corr);
            let v_l_h = _mm_sub_epi16(_mm_unpackhi_epi8(low_v_value, zeros), uv_corr);
            let y_l_h = _mm_mullo_epi16(_mm_unpackhi_epi8(y_first, zeros), v_luma_coeff);

            let r_l_h = _mm_srai_epi16::<6>(_mm_adds_epi16(
                _mm_max_epi16(
                    _mm_adds_epi16(y_l_h, _mm_mullo_epi16(v_l_h, v_cr_coeff)),
                    zeros,
                ),
                rounding_const,
            ));
            let b_l_h = _mm_srai_epi16::<6>(_mm_adds_epi16(
                _mm_max_epi16(
                    _mm_adds_epi16(y_l_h, _mm_mullo_epi16(u_l_h, v_cb_coeff)),
                    zeros,
                ),
                rounding_const,
            ));
            let g_l_h = _mm_srai_epi16::<6>(_mm_adds_epi16(
                _mm_max_epi16(
                    _mm_adds_epi16(
                        y_l_h,
                        _mm_adds_epi16(
                            _mm_mullo_epi16(v_l_h, v_g_coeff_1),
                            _mm_mullo_epi16(u_l_h, v_g_coeff_2),
                        ),
                    ),
                    zeros,
                ),
                rounding_const,
            ));

            let u_low = _mm_sub_epi16(_mm_unpacklo_epi8(low_u_value, zeros), uv_corr);
            let v_low = _mm_sub_epi16(_mm_unpacklo_epi8(low_v_value, zeros), uv_corr);
            let y_low = _mm_mullo_epi16(_mm_unpacklo_epi8(y_first, zeros), v_luma_coeff);

            let r_l_l = _mm_srai_epi16::<6>(_mm_adds_epi16(
                _mm_max_epi16(
                    _mm_adds_epi16(y_low, _mm_mullo_epi16(v_low, v_cr_coeff)),
                    zeros,
                ),
                rounding_const,
            ));
            let b_l_l = _mm_srai_epi16::<6>(_mm_adds_epi16(
                _mm_max_epi16(
                    _mm_adds_epi16(y_low, _mm_mullo_epi16(u_low, v_cb_coeff)),
                    zeros,
                ),
                rounding_const,
            ));
            let g_l_l = _mm_srai_epi16::<6>(_mm_adds_epi16(
                _mm_max_epi16(
                    _mm_adds_epi16(
                        y_low,
                        _mm_adds_epi16(
                            _mm_mullo_epi16(v_low, v_g_coeff_1),
                            _mm_mullo_epi16(u_low, v_g_coeff_2),
                        ),
                    ),
                    zeros,
                ),
                rounding_const,
            ));

            let r_l = _mm_packus_epi16(r_l_l, r_l_h);
            let g_l = _mm_packus_epi16(g_l_l, g_l_h);
            let b_l = _mm_packus_epi16(b_l_l, b_l_h);

            match dst_chans {
                YuvSourceChannels::Rgb => {
                    let packed = sse_interleave_rgb(r_l, g_l, b_l);
                    _mm_storeu_si128(dst_ptr as *mut __m128i, packed.0);
                    _mm_storeu_si128(dst_ptr.add(16) as *mut __m128i, packed.1);
                    _mm_storeu_si128(dst_ptr.add(32) as *mut __m128i, packed.2);
                }
                YuvSourceChannels::Rgba => {
                    let packed = sse_interleave_rgba(r_l, g_l, b_l, v_alpha);
                    _mm_storeu_si128(dst_ptr as *mut __m128i, packed.0);
                    _mm_storeu_si128(dst_ptr.add(16) as *mut __m128i, packed.1);
                    _mm_storeu_si128(dst_ptr.add(32) as *mut __m128i, packed.2);
                    _mm_storeu_si128(dst_ptr.add(48) as *mut __m128i, packed.3);
                }
                YuvSourceChannels::Bgra => {
                    let packed = sse_interleave_rgba(b_l, g_l, r_l, v_alpha);
                    _mm_storeu_si128(dst_ptr as *mut __m128i, packed.0);
                    _mm_storeu_si128(dst_ptr.add(16) as *mut __m128i, packed.1);
                    _mm_storeu_si128(dst_ptr.add(32) as *mut __m128i, packed.2);
                    _mm_storeu_si128(dst_ptr.add(48) as *mut __m128i, packed.3);
                }
                YuvSourceChannels::Bgr => {
                    let packed = sse_interleave_rgb(b_l, g_l, r_l);
                    _mm_storeu_si128(dst_ptr as *mut __m128i, packed.0);
                    _mm_storeu_si128(dst_ptr.add(16) as *mut __m128i, packed.1);
                    _mm_storeu_si128(dst_ptr.add(32) as *mut __m128i, packed.2);
                }
            }

            let u_h_h = _mm_sub_epi16(_mm_unpackhi_epi8(high_u_value, zeros), uv_corr);
            let v_h_h = _mm_sub_epi16(_mm_unpackhi_epi8(high_v_value, zeros), uv_corr);
            let y_h_h = _mm_mullo_epi16(_mm_unpackhi_epi8(y_second, zeros), v_luma_coeff);

            let r_h_h = _mm_srai_epi16::<6>(_mm_adds_epi16(
                _mm_max_epi16(
                    _mm_adds_epi16(y_h_h, _mm_mullo_epi16(v_h_h, v_cr_coeff)),
                    zeros,
                ),
                rounding_const,
            ));
            let b_h_h = _mm_srai_epi16::<6>(_mm_adds_epi16(
                _mm_max_epi16(
                    _mm_adds_epi16(y_h_h, _mm_mullo_epi16(u_h_h, v_cb_coeff)),
                    zeros,
                ),
                rounding_const,
            ));
            let g_h_h = _mm_srai_epi16::<6>(_mm_adds_epi16(
                _mm_max_epi16(
                    _mm_adds_epi16(
                        y_h_h,
                        _mm_adds_epi16(
                            _mm_mullo_epi16(v_h_h, v_g_coeff_1),
                            _mm_mullo_epi16(u_h_h, v_g_coeff_2),
                        ),
                    ),
                    zeros,
                ),
                rounding_const,
            ));

            let u_h_l = _mm_sub_epi16(_mm_unpacklo_epi8(high_u_value, zeros), uv_corr);
            let v_h_l = _mm_sub_epi16(_mm_unpacklo_epi8(high_v_value, zeros), uv_corr);
            let y_h_l = _mm_mullo_epi16(_mm_unpacklo_epi8(y_second, zeros), v_luma_coeff);

            let r_h_l = _mm_srai_epi16::<6>(_mm_adds_epi16(
                _mm_max_epi16(
                    _mm_adds_epi16(y_h_l, _mm_mullo_epi16(v_h_l, v_cr_coeff)),
                    zeros,
                ),
                rounding_const,
            ));
            let b_h_l = _mm_srai_epi16::<6>(_mm_adds_epi16(
                _mm_max_epi16(
                    _mm_adds_epi16(y_h_l, _mm_mullo_epi16(u_h_l, v_cb_coeff)),
                    zeros,
                ),
                rounding_const,
            ));
            let g_h_l = _mm_srai_epi16::<6>(_mm_adds_epi16(
                _mm_max_epi16(
                    _mm_adds_epi16(
                        y_h_l,
                        _mm_adds_epi16(
                            _mm_mullo_epi16(v_h_l, v_g_coeff_1),
                            _mm_mullo_epi16(u_h_l, v_g_coeff_2),
                        ),
                    ),
                    zeros,
                ),
                rounding_const,
            ));

            let r_h = _mm_packus_epi16(r_h_l, r_h_h);
            let g_h = _mm_packus_epi16(g_h_l, g_h_h);
            let b_h = _mm_packus_epi16(b_h_l, b_h_h);

            match dst_chans {
                YuvSourceChannels::Rgb => {
                    let packed = sse_interleave_rgb(r_h, g_h, b_h);
                    let v_dst = dst_ptr.add(16 * dst_chans.get_channels_count());
                    _mm_storeu_si128(v_dst as *mut __m128i, packed.0);
                    _mm_storeu_si128(v_dst.add(16) as *mut __m128i, packed.1);
                    _mm_storeu_si128(v_dst.add(32) as *mut __m128i, packed.2);
                }
                YuvSourceChannels::Rgba => {
                    let packed = sse_interleave_rgba(r_h, g_h, b_h, v_alpha);
                    let v_dst = dst_ptr.add(16 * dst_chans.get_channels_count());
                    _mm_storeu_si128(v_dst as *mut __m128i, packed.0);
                    _mm_storeu_si128(v_dst.add(16) as *mut __m128i, packed.1);
                    _mm_storeu_si128(v_dst.add(32) as *mut __m128i, packed.2);
                    _mm_storeu_si128(v_dst.add(48) as *mut __m128i, packed.3);
                }
                YuvSourceChannels::Bgra => {
                    let packed = sse_interleave_rgba(b_h, g_h, r_h, v_alpha);
                    let v_dst = dst_ptr.add(16 * dst_chans.get_channels_count());
                    _mm_storeu_si128(v_dst as *mut __m128i, packed.0);
                    _mm_storeu_si128(v_dst.add(16) as *mut __m128i, packed.1);
                    _mm_storeu_si128(v_dst.add(32) as *mut __m128i, packed.2);
                    _mm_storeu_si128(v_dst.add(48) as *mut __m128i, packed.3);
                }
                YuvSourceChannels::Bgr => {
                    let packed = sse_interleave_rgb(b_h, g_h, r_h);
                    let v_dst = dst_ptr.add(16 * dst_chans.get_channels_count());
                    _mm_storeu_si128(v_dst as *mut __m128i, packed.0);
                    _mm_storeu_si128(v_dst.add(16) as *mut __m128i, packed.1);
                    _mm_storeu_si128(v_dst.add(32) as *mut __m128i, packed.2);
                }
            }

            _yuy2_x = x;
            if x + 16 < max_x_16 {
                _cx += 32;
            }
        }

        for x in (_yuy2_x..max_x_8).step_by(8) {
            let yuy2_offset = yuy2_offset + x * 4;
            let dst_pos = rgb_offset + _cx * dst_chans.get_channels_count();
            let dst_ptr = rgb.as_mut_ptr().add(dst_pos);

            let yuy2_ptr = yuy2_store.as_ptr().add(yuy2_offset);

            let j0 = _mm_loadu_si128(yuy2_ptr as *const __m128i);
            let j1 = _mm_loadu_si128(yuy2_ptr.add(16) as *const __m128i);

            let pixel_set = sse_deinterleave_rgba(j0, j1, zeros, zeros);

            let mut y_first = match yuy2_source {
                Yuy2Description::YUYV | Yuy2Description::YVYU => pixel_set.0,
                Yuy2Description::UYVY | Yuy2Description::VYUY => pixel_set.1,
            };
            let mut y_second = match yuy2_source {
                Yuy2Description::YUYV | Yuy2Description::YVYU => pixel_set.2,
                Yuy2Description::UYVY | Yuy2Description::VYUY => pixel_set.3,
            };

            let u_value = match yuy2_source {
                Yuy2Description::YUYV => pixel_set.1,
                Yuy2Description::UYVY => pixel_set.0,
                Yuy2Description::YVYU => pixel_set.3,
                Yuy2Description::VYUY => pixel_set.2,
            };
            let v_value = match yuy2_source {
                Yuy2Description::YUYV => pixel_set.3,
                Yuy2Description::UYVY => pixel_set.2,
                Yuy2Description::YVYU => pixel_set.1,
                Yuy2Description::VYUY => pixel_set.0,
            };

            y_first = _mm_subs_epu8(y_first, y_corr);
            y_second = _mm_subs_epu8(y_second, y_corr);

            let y_reconstructed = _mm_unpacklo_epi8(y_first, y_second);

            let u_l = _mm_unpacklo_epi8(u_value, zeros);
            let v_l = _mm_unpacklo_epi8(v_value, zeros);

            let low_u_value = _mm_unpacklo_epi16(u_l, u_l);
            let high_u_value = _mm_unpackhi_epi16(u_l, u_l);
            let low_v_value = _mm_unpacklo_epi16(v_l, v_l);
            let high_v_value = _mm_unpackhi_epi16(v_l, v_l);

            let u_h = _mm_sub_epi16(high_u_value, uv_corr);
            let v_h = _mm_sub_epi16(high_v_value, uv_corr);
            let y_h = _mm_mullo_epi16(_mm_unpackhi_epi8(y_reconstructed, zeros), v_luma_coeff);

            let r_h = _mm_srai_epi16::<6>(_mm_adds_epi16(
                _mm_max_epi16(_mm_adds_epi16(y_h, _mm_mullo_epi16(v_h, v_cr_coeff)), zeros),
                rounding_const,
            ));
            let b_h = _mm_srai_epi16::<6>(_mm_adds_epi16(
                _mm_max_epi16(_mm_adds_epi16(y_h, _mm_mullo_epi16(u_h, v_cb_coeff)), zeros),
                rounding_const,
            ));
            let g_h = _mm_srai_epi16::<6>(_mm_adds_epi16(
                _mm_max_epi16(
                    _mm_adds_epi16(
                        y_h,
                        _mm_adds_epi16(
                            _mm_mullo_epi16(v_h, v_g_coeff_1),
                            _mm_mullo_epi16(u_h, v_g_coeff_2),
                        ),
                    ),
                    zeros,
                ),
                rounding_const,
            ));

            let u_low = _mm_sub_epi16(low_u_value, uv_corr);
            let v_low = _mm_sub_epi16(low_v_value, uv_corr);
            let y_low = _mm_mullo_epi16(_mm_unpacklo_epi8(y_reconstructed, zeros), v_luma_coeff);

            let r_l = _mm_srai_epi16::<6>(_mm_adds_epi16(
                _mm_max_epi16(
                    _mm_adds_epi16(y_low, _mm_mullo_epi16(v_low, v_cr_coeff)),
                    zeros,
                ),
                rounding_const,
            ));
            let b_l = _mm_srai_epi16::<6>(_mm_adds_epi16(
                _mm_max_epi16(
                    _mm_adds_epi16(y_low, _mm_mullo_epi16(u_low, v_cb_coeff)),
                    zeros,
                ),
                rounding_const,
            ));
            let g_l = _mm_srai_epi16::<6>(_mm_adds_epi16(
                _mm_max_epi16(
                    _mm_adds_epi16(
                        y_low,
                        _mm_adds_epi16(
                            _mm_mullo_epi16(v_low, v_g_coeff_1),
                            _mm_mullo_epi16(u_low, v_g_coeff_2),
                        ),
                    ),
                    zeros,
                ),
                rounding_const,
            ));

            let r_v = _mm_packus_epi16(r_l, r_h);
            let g_v = _mm_packus_epi16(g_l, g_h);
            let b_v = _mm_packus_epi16(b_l, b_h);

            match dst_chans {
                YuvSourceChannels::Rgb => {
                    let packed = sse_interleave_rgb(r_v, g_v, b_v);
                    _mm_storeu_si128(dst_ptr as *mut __m128i, packed.0);
                    _mm_storeu_si128(dst_ptr.add(16) as *mut __m128i, packed.1);
                    _mm_storeu_si128(dst_ptr.add(32) as *mut __m128i, packed.2);
                }
                YuvSourceChannels::Rgba => {
                    let packed = sse_interleave_rgba(r_v, g_v, b_v, v_alpha);
                    _mm_storeu_si128(dst_ptr as *mut __m128i, packed.0);
                    _mm_storeu_si128(dst_ptr.add(16) as *mut __m128i, packed.1);
                    _mm_storeu_si128(dst_ptr.add(32) as *mut __m128i, packed.2);
                    _mm_storeu_si128(dst_ptr.add(48) as *mut __m128i, packed.3);
                }
                YuvSourceChannels::Bgra => {
                    let packed = sse_interleave_rgba(b_v, g_v, r_v, v_alpha);
                    _mm_storeu_si128(dst_ptr as *mut __m128i, packed.0);
                    _mm_storeu_si128(dst_ptr.add(16) as *mut __m128i, packed.1);
                    _mm_storeu_si128(dst_ptr.add(32) as *mut __m128i, packed.2);
                    _mm_storeu_si128(dst_ptr.add(48) as *mut __m128i, packed.3);
                }
                YuvSourceChannels::Bgr => {
                    let packed = sse_interleave_rgb(b_v, g_v, r_v);
                    _mm_storeu_si128(dst_ptr as *mut __m128i, packed.0);
                    _mm_storeu_si128(dst_ptr.add(16) as *mut __m128i, packed.1);
                    _mm_storeu_si128(dst_ptr.add(32) as *mut __m128i, packed.2);
                }
            }

            _yuy2_x = x;
            if x + 8 < max_x_8 {
                _cx += 16;
            }
        }
    }

    YuvToYuy2Navigation {
        cx: _cx,
        uv_x: 0,
        x: _yuy2_x,
    }
}
