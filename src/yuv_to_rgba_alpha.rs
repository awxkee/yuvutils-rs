#[cfg(
    all(
        any(target_arch = "x86", target_arch = "x86_64"),
        target_feature = "avx2"
    )
)]
use core::arch::x86_64::*;
#[cfg(target_arch = "aarch64")]
#[cfg(target_feature = "neon")]
use std::arch::aarch64::{
    uint8x16_t, uint8x16x4_t, uint8x8_t, vcombine_u8, vdup_n_u8, vdupq_n_s16, vdupq_n_u16,
    vdupq_n_u8, vget_high_u8, vget_low_u8, vld1_u8, vld1q_u8, vmaxq_s16, vmovl_u8, vmull_high_u8,
    vmull_u8, vmulq_s16, vqaddq_s16, vqaddq_u16, vqshrn_n_u16, vqshrun_n_s16,
    vreinterpretq_s16_u16, vst4q_u8, vsubq_s16, vsubq_u8, vzip1_u8, vzip2_u8,
};

use crate::yuv_support::{
    get_inverse_transform, get_kr_kb, get_yuv_range, YuvChromaSample, YuvSourceChannels,
};
use crate::{YuvRange, YuvStandardMatrix};

#[cfg(target_arch = "aarch64")]
#[cfg(target_feature = "neon")]
#[inline(always)]
unsafe fn premutiply_vector(v: uint8x16_t, a_values: uint8x16_t) -> uint8x16_t {
    let initial = vdupq_n_u16(127);
    let acc_hi = vqaddq_u16(initial, vmull_high_u8(v, a_values));
    let acc_lo = vqaddq_u16(initial, vmull_u8(vget_low_u8(v), vget_low_u8(a_values)));
    let hi = vqshrn_n_u16::<8>(acc_hi);
    let lo = vqshrn_n_u16::<8>(acc_lo);
    vcombine_u8(lo, hi)
}

fn yuv_with_alpha_to_rgbx<const DESTINATION_CHANNELS: u8, const SAMPLING: u8>(
    y_plane: &[u8],
    y_stride: u32,
    u_plane: &[u8],
    u_stride: u32,
    v_plane: &[u8],
    v_stride: u32,
    a_plane: &[u8],
    a_stride: u32,
    rgba: &mut [u8],
    rgba_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    premultiply_alpha: bool,
) {
    let chroma_subsampling: YuvChromaSample = SAMPLING.into();
    let destination_channels: YuvSourceChannels = DESTINATION_CHANNELS.into();
    if !destination_channels.has_alpha() {
        panic!("yuv_with_alpha_to_rgbx cannot be called on configuration without alpha");
    }
    let channels = destination_channels.get_channels_count();
    let range = get_yuv_range(8, range);
    let kr_kb = get_kr_kb(matrix);
    let transform = get_inverse_transform(255, range.range_y, range.range_uv, kr_kb.kr, kr_kb.kb);
    let precision_scale: i32 = 1i32 << 6i32;
    let cr_coef = (transform.cr_coef * precision_scale as f32).round() as i32;
    let cb_coef = (transform.cb_coef * precision_scale as f32).round() as i32;
    let y_coef = (transform.y_coef * precision_scale as f32).round() as i32;
    let g_coef_1 = (transform.g_coeff_1 * precision_scale as f32).round() as i32;
    let g_coef_2 = (transform.g_coeff_2 * precision_scale as f32).round() as i32;

    let bias_y = range.bias_y as i32;
    let bias_uv = range.bias_uv as i32;

    let mut y_offset = 0usize;
    let mut u_offset = 0usize;
    let mut v_offset = 0usize;
    let mut a_offset = 0usize;
    let mut rgba_offset = 0usize;

    let iterator_step = match chroma_subsampling {
        YuvChromaSample::YUV420 => 2usize,
        YuvChromaSample::YUV422 => 2usize,
        YuvChromaSample::YUV444 => 1usize,
    };

    for y in 0..height as usize {
        #[allow(unused_variables)]
        #[allow(unused_mut)]
        let mut cx = 0usize;

        #[allow(unused_variables)]
        #[allow(unused_mut)]
        let mut uv_x = 0usize;

        #[cfg(
            all(
                any(target_arch = "x86", target_arch = "x86_64"),
                target_feature = "avx2"
            )
        )]
        unsafe {
            let y_ptr = y_plane.as_ptr();
            let u_ptr = u_plane.as_ptr();
            let v_ptr = v_plane.as_ptr();
            let rgba_ptr = rgba.as_mut_ptr();

            let y_corr = _mm256_set1_epi8(bias_y as i8);
            let uv_corr = _mm256_set1_epi16(bias_uv as i16);
            let v_luma_coeff = _mm256_set1_epi8(y_coef as i8);
            let v_luma_coeff_16 = _mm256_set1_epi16(y_coef as i16);
            let v_cr_coeff = _mm256_set1_epi16(cr_coef as i16);
            let v_cb_coeff = _mm256_set1_epi16(cb_coef as i16);
            let v_min_values = _mm256_set1_epi16(0);
            let v_g_coeff_1 = _mm256_set1_epi16(-1 * g_coef_1 as i16);
            let v_g_coeff_2 = _mm256_set1_epi16(-1 * g_coef_2 as i16);
            let v_alpha = _mm256_set1_epi8(255u8 as i8);

            while cx + 32 < width as usize {
                let y_values = _mm256_sub_epi8(
                    _mm256_loadu_si256(y_ptr.add(y_offset + cx) as *const __m256i),
                    y_corr,
                );

                let (u_high_u8, v_high_u8, u_low_u8, v_low_u8);

                match chroma_subsampling {
                    YuvChromaSample::YUV420 | YuvChromaSample::YUV422 => {
                        let u_values =
                            _mm_loadu_si128(u_ptr.add(u_offset + uv_x) as *const __m128i);
                        let v_values =
                            _mm_loadu_si128(v_ptr.add(v_offset + uv_x) as *const __m128i);

                        u_high_u8 = _mm_unpackhi_epi8(u_values, u_values);
                        v_high_u8 = _mm_unpackhi_epi8(v_values, v_values);
                        u_low_u8 = _mm_unpacklo_epi8(u_values, u_values);
                        v_low_u8 = _mm_unpacklo_epi8(v_values, v_values);
                    }
                    YuvChromaSample::YUV444 => {
                        let u_values =
                            _mm256_loadu_si256(u_ptr.add(u_offset + uv_x) as *const __m256i);
                        let v_values =
                            _mm256_loadu_si256(v_ptr.add(v_offset + uv_x) as *const __m256i);

                        u_high_u8 = _mm256_extracti128_si256::<1>(u_values);
                        v_high_u8 = _mm256_extracti128_si256::<1>(v_values);
                        u_low_u8 = _mm256_castsi256_si128(u_values);
                        v_low_u8 = _mm256_castsi256_si128(v_values);
                    }
                }

                let u_high = _mm256_sub_epi16(_mm256_cvtepu8_epi16(u_high_u8), uv_corr);
                let v_high = _mm256_sub_epi16(_mm256_cvtepu8_epi16(v_high_u8), uv_corr);
                let y_high = _mm256_mulhi_epi16(
                    _mm256_cvtepu8_epi16(_mm256_extracti128_si256::<1>(y_values)),
                    v_luma_coeff,
                );

                let r_high = _mm256_srai_epi16::<6>(_mm256_max_epi16(
                    _mm256_adds_epi16(y_high, _mm256_mullo_epi16(v_high, v_cr_coeff)),
                    v_min_values,
                ));
                let b_high = _mm256_srai_epi16::<6>(_mm256_max_epi16(
                    _mm256_adds_epi16(y_high, _mm256_mullo_epi16(u_high, v_cb_coeff)),
                    v_min_values,
                ));
                let g_high = _mm256_srai_epi16::<6>(_mm256_max_epi16(
                    _mm256_adds_epi16(
                        y_high,
                        _mm256_adds_epi16(
                            _mm256_mullo_epi16(v_high, v_g_coeff_1),
                            _mm256_mullo_epi16(u_high, v_g_coeff_2),
                        ),
                    ),
                    v_min_values,
                ));

                let u_low = _mm256_sub_epi16(_mm256_cvtepu8_epi16(u_low_u8), uv_corr);
                let v_low = _mm256_sub_epi16(_mm256_cvtepu8_epi16(v_low_u8), uv_corr);
                let y_low = _mm256_mullo_epi16(
                    _mm256_cvtepu8_epi16(_mm256_castsi256_si128(y_values)),
                    v_luma_coeff_16,
                );

                let r_low = _mm256_srai_epi16::<6>(
                    _mm256_max_epi16(
                        _mm256_adds_epi16(y_low, _mm256_mullo_epi16(v_low, v_cr_coeff)),
                        v_min_values,
                    ),
                );
                let b_low = _mm256_srai_epi16::<6>(
                    _mm256_max_epi16(
                        _mm256_adds_epi16(y_low, _mm256_mullo_epi16(u_low, v_cb_coeff)),
                        v_min_values,
                    ),
                );
                let g_low = _mm256_srai_epi16::<6>(
                    _mm256_max_epi16(
                        _mm256_adds_epi16(
                            y_low,
                            _mm256_adds_epi16(
                                _mm256_mullo_epi16(v_low, v_g_coeff_1),
                                _mm256_mullo_epi16(u_low, v_g_coeff_2),
                            ),
                        ),
                        v_min_values,
                    ),
                );

                let r_values = _mm256_packus_epi16(r_low, r_high);
                let g_values = _mm256_packus_epi16(g_low, g_high);
                let b_values = _mm256_packus_epi16(b_low, b_high);

                let dst_shift = rgba_offset + cx * channels;

                match destination_channels {
                    YuvSourceChannels::Rgb => {
                        let rg_low = _mm256_unpacklo_epi32(r_values, g_values); // [r0, g0, r1, g1, r2, g2, r3, g3]
                        let rg_high = _mm256_unpackhi_epi32(r_values, g_values); // [r4, g4, r5, g5, r6, g6, r7, g7]
                        let b0 = _mm256_unpacklo_epi32(b_values, _mm256_setzero_si256()); // [b0, 0, b1, 0, b2, 0, b3, 0]
                        let b1 = _mm256_unpackhi_epi32(b_values, _mm256_setzero_si256()); // [b4, 0, b5, 0, b6, 0, b7, 0]

                        // Step 2: Unpack 16-bit integers to 8-bit integers (low and high parts)
                        let rgb0 = _mm256_unpacklo_epi16(rg_low, b0); // [r0, g0, b0, 0, r1, g1, b1, 0]
                        let rgb1 = _mm256_unpackhi_epi16(rg_low, b0); // [r2, g2, b2, 0, r3, g3, b3, 0]
                        let rgb2 = _mm256_unpacklo_epi16(rg_high, b1); // [r4, g4, b4, 0, r5, g5, b5, 0]
                        let rgb3 = _mm256_unpackhi_epi16(rg_high, b1); // [r6, g6, b6, 0, r7, g7, b7, 0]

                        // Pack the result into RGB format, stripping the zeroed bytes
                        let result0 = _mm256_packus_epi16(rgb0, rgb1); // [r0, g0, b0, r1, g1, b1, r2, g2, b2, r3, g3, b3, r4, g4, b4, r5, g5, b5, r6, g6, b6, r7, g7, b7]
                        let result1 = _mm256_packus_epi16(rgb2, rgb3); // similarly packed second half

                        // Store the interleaved result in memory
                        _mm256_storeu_si256(rgba_ptr.add(dst_shift) as *mut __m256i, result0);
                        _mm256_storeu_si256(rgba_ptr.add(dst_shift + 32) as *mut __m256i, result1);
                    }
                    YuvSourceChannels::Rgba => {
                        let rg_low = _mm256_unpacklo_epi32(r_values, g_values); // [r0, g0, r1, g1, r2, g2, r3, g3]
                        let rg_high = _mm256_unpackhi_epi32(r_values, g_values); // [r4, g4, r5, g5, r6, g6, r7, g7]
                        let ba_low = _mm256_unpacklo_epi32(b_values, v_alpha); // [b0, a0, b1, a1, b2, a2, b3, a3]
                        let ba_high = _mm256_unpackhi_epi32(b_values, v_alpha); // [b4, a4, b5, a5, b6, a6, b7, a7]

                        // Step 2: Unpack 16-bit integers to 8-bit integers (low and high parts)
                        let rgba0 = _mm256_unpacklo_epi16(rg_low, ba_low); // [r0, g0, b0, a0, r1, g1, b1, a1]
                        let rgba1 = _mm256_unpackhi_epi16(rg_low, ba_low); // [r2, g2, b2, a2, r3, g3, b3, a3]
                        let rgba2 = _mm256_unpacklo_epi16(rg_high, ba_high); // [r4, g4, b4, a4, r5, g5, b5, a5]
                        let rgba3 = _mm256_unpackhi_epi16(rg_high, ba_high); //

                        _mm256_storeu_si256(rgba_ptr.add(dst_shift) as *mut __m256i, rgba0);
                        _mm256_storeu_si256(
                            rgba_ptr.add(dst_shift + 32) as *mut __m256i,
                            rgba1,
                        );
                        _mm256_storeu_si256(
                            rgba_ptr.add(dst_shift + 64) as *mut __m256i,
                            rgba2,
                        );
                        _mm256_storeu_si256(
                            rgba_ptr.add(dst_shift + 96) as *mut __m256i,
                            rgba3,
                        );
                    }
                    YuvSourceChannels::Bgra => {

                    }
                }

                cx += 32;

                match chroma_subsampling {
                    YuvChromaSample::YUV420 | YuvChromaSample::YUV422 => {
                        uv_x += 16;
                    }
                    YuvChromaSample::YUV444 => {
                        uv_x += 32;
                    }
                }
            }
        }

        #[cfg(target_arch = "aarch64")]
        #[cfg(target_feature = "neon")]
        unsafe {
            let y_ptr = y_plane.as_ptr();
            let u_ptr = u_plane.as_ptr();
            let v_ptr = v_plane.as_ptr();
            let a_ptr = a_plane.as_ptr();
            let rgba_ptr = rgba.as_mut_ptr();

            let y_corr = vdupq_n_u8(bias_y as u8);
            let uv_corr = vdupq_n_s16(bias_uv as i16);
            let v_luma_coeff = vdupq_n_u8(y_coef as u8);
            let v_luma_coeff_8 = vdup_n_u8(y_coef as u8);
            let v_cr_coeff = vdupq_n_s16(cr_coef as i16);
            let v_cb_coeff = vdupq_n_s16(cb_coef as i16);
            let v_min_values = vdupq_n_s16(0i16);
            let v_g_coeff_1 = vdupq_n_s16(-1i16 * g_coef_1 as i16);
            let v_g_coeff_2 = vdupq_n_s16(-1i16 * g_coef_2 as i16);

            while cx + 16 < width as usize {
                let y_values = vsubq_u8(vld1q_u8(y_ptr.add(y_offset + cx)), y_corr);
                let a_values = vld1q_u8(a_ptr.add(a_offset + cx));

                let u_high_u8: uint8x8_t;
                let v_high_u8: uint8x8_t;
                let u_low_u8: uint8x8_t;
                let v_low_u8: uint8x8_t;

                match chroma_subsampling {
                    YuvChromaSample::YUV420 | YuvChromaSample::YUV422 => {
                        let u_values = vld1_u8(u_ptr.add(u_offset + uv_x));
                        let v_values = vld1_u8(v_ptr.add(v_offset + uv_x));

                        u_high_u8 = vzip2_u8(u_values, u_values);
                        v_high_u8 = vzip2_u8(v_values, v_values);
                        u_low_u8 = vzip1_u8(u_values, u_values);
                        v_low_u8 = vzip1_u8(v_values, v_values);
                    }
                    YuvChromaSample::YUV444 => {
                        let u_values = vld1q_u8(u_ptr.add(u_offset + uv_x));
                        let v_values = vld1q_u8(v_ptr.add(v_offset + uv_x));

                        u_high_u8 = vget_high_u8(u_values);
                        v_high_u8 = vget_high_u8(v_values);
                        u_low_u8 = vget_low_u8(u_values);
                        v_low_u8 = vget_low_u8(v_values);
                    }
                }

                let u_high = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(u_high_u8)), uv_corr);
                let v_high = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(v_high_u8)), uv_corr);
                let y_high = vreinterpretq_s16_u16(vmull_high_u8(y_values, v_luma_coeff));

                let r_high = vqshrun_n_s16::<6>(vmaxq_s16(
                    vqaddq_s16(y_high, vmulq_s16(v_high, v_cr_coeff)),
                    v_min_values,
                ));
                let b_high = vqshrun_n_s16::<6>(vmaxq_s16(
                    vqaddq_s16(y_high, vmulq_s16(u_high, v_cb_coeff)),
                    v_min_values,
                ));
                let g_high = vqshrun_n_s16::<6>(vmaxq_s16(
                    vqaddq_s16(
                        y_high,
                        vqaddq_s16(
                            vmulq_s16(v_high, v_g_coeff_1),
                            vmulq_s16(u_high, v_g_coeff_2),
                        ),
                    ),
                    v_min_values,
                ));

                let u_low = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(u_low_u8)), uv_corr);
                let v_low = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(v_low_u8)), uv_corr);
                let y_low = vreinterpretq_s16_u16(vmull_u8(vget_low_u8(y_values), v_luma_coeff_8));

                let r_low = vqshrun_n_s16::<6>(vmaxq_s16(
                    vqaddq_s16(y_low, vmulq_s16(v_low, v_cr_coeff)),
                    v_min_values,
                ));
                let b_low = vqshrun_n_s16::<6>(vmaxq_s16(
                    vqaddq_s16(y_low, vmulq_s16(u_low, v_cb_coeff)),
                    v_min_values,
                ));
                let g_low = vqshrun_n_s16::<6>(vmaxq_s16(
                    vqaddq_s16(
                        y_low,
                        vqaddq_s16(vmulq_s16(v_low, v_g_coeff_1), vmulq_s16(u_low, v_g_coeff_2)),
                    ),
                    v_min_values,
                ));

                let mut r_values = vcombine_u8(r_low, r_high);
                let mut g_values = vcombine_u8(g_low, g_high);
                let mut b_values = vcombine_u8(b_low, b_high);

                let dst_shift = rgba_offset + cx * channels;

                if premultiply_alpha {
                    r_values = premutiply_vector(r_values, a_values);
                    g_values = premutiply_vector(g_values, a_values);
                    b_values = premutiply_vector(b_values, a_values);
                }

                match destination_channels {
                    YuvSourceChannels::Rgb => {
                        panic!("Should not be reached");
                    }
                    YuvSourceChannels::Rgba => {
                        let dst_pack: uint8x16x4_t =
                            uint8x16x4_t(r_values, g_values, b_values, a_values);
                        vst4q_u8(rgba_ptr.add(dst_shift), dst_pack);
                    }
                    YuvSourceChannels::Bgra => {
                        let dst_pack: uint8x16x4_t =
                            uint8x16x4_t(b_values, g_values, r_values, a_values);
                        vst4q_u8(rgba_ptr.add(dst_shift), dst_pack);
                    }
                }

                cx += 16;

                match chroma_subsampling {
                    YuvChromaSample::YUV420 | YuvChromaSample::YUV422 => {
                        uv_x += 8;
                    }
                    YuvChromaSample::YUV444 => {
                        uv_x += 16;
                    }
                }
            }
        }

        for x in (cx..width as usize).step_by(iterator_step) {
            let y_value = (y_plane[y_offset + x] as i32 - bias_y) * y_coef;

            let u_pos = match chroma_subsampling {
                YuvChromaSample::YUV420 | YuvChromaSample::YUV422 => u_offset + uv_x,
                YuvChromaSample::YUV444 => u_offset + uv_x,
            };

            let cb_value = u_plane[u_pos] as i32 - bias_uv;

            let v_pos = match chroma_subsampling {
                YuvChromaSample::YUV420 | YuvChromaSample::YUV422 => v_offset + uv_x,
                YuvChromaSample::YUV444 => v_offset + uv_x,
            };

            let cr_value = v_plane[v_pos] as i32 - bias_uv;

            let mut r = ((y_value + cr_coef * cr_value) >> 6).min(255).max(0);
            let mut b = ((y_value + cb_coef * cb_value) >> 6).min(255).max(0);
            let mut g = ((y_value - g_coef_1 * cr_value - g_coef_2 * cb_value) >> 6)
                .min(255)
                .max(0);

            let px = x * channels;

            let rgba_shift = rgba_offset + px;

            let a_value = a_plane[a_offset + x];
            if premultiply_alpha {
                r = (r * a_value as i32 + 127i32) >> 8;
                g = (g * a_value as i32 + 127i32) >> 8;
                b = (b * a_value as i32 + 127i32) >> 8;
            }

            rgba[rgba_shift + destination_channels.get_r_channel_offset()] = r as u8;
            rgba[rgba_shift + destination_channels.get_g_channel_offset()] = g as u8;
            rgba[rgba_shift + destination_channels.get_b_channel_offset()] = b as u8;
            rgba[rgba_shift + destination_channels.get_a_channel_offset()] = a_value;

            if chroma_subsampling == YuvChromaSample::YUV420
                || chroma_subsampling == YuvChromaSample::YUV422
            {
                if x + 1 < width as usize {
                    let y_value = (y_plane[y_offset + x + 1] as i32 - bias_y) * y_coef;

                    let mut r = ((y_value + cr_coef * cr_value) >> 6).min(255).max(0);
                    let mut b = ((y_value + cb_coef * cb_value) >> 6).min(255).max(0);
                    let mut g = ((y_value - g_coef_1 * cr_value - g_coef_2 * cb_value) >> 6)
                        .min(255)
                        .max(0);

                    let next_px = (x + 1) * channels;

                    let rgba_shift = rgba_offset + next_px;

                    let a_value = a_plane[a_offset + next_px];
                    if premultiply_alpha {
                        r = (r * a_value as i32 + 127i32) >> 8;
                        g = (g * a_value as i32 + 127i32) >> 8;
                        b = (b * a_value as i32 + 127i32) >> 8;
                    }

                    rgba[rgba_shift + destination_channels.get_r_channel_offset()] = r as u8;
                    rgba[rgba_shift + destination_channels.get_g_channel_offset()] = g as u8;
                    rgba[rgba_shift + destination_channels.get_b_channel_offset()] = b as u8;
                    rgba[rgba_shift + destination_channels.get_a_channel_offset()] = a_value;
                }
            }

            uv_x += 1;
        }

        y_offset += y_stride as usize;
        rgba_offset += rgba_stride as usize;
        a_offset += a_stride as usize;
        match chroma_subsampling {
            YuvChromaSample::YUV420 => {
                if y & 1 == 1 {
                    u_offset += u_stride as usize;
                    v_offset += v_stride as usize;
                }
            }
            YuvChromaSample::YUV444 | YuvChromaSample::YUV422 => {
                u_offset += u_stride as usize;
                v_offset += v_stride as usize;
            }
        }
    }
}

/// Convert YUV 420 planar format to RGBA format and appends provided alpha channel.
///
/// This function takes YUV 420 planar format data with 8-bit precision,
/// and converts it to RGBA format with 8-bit per channel precision.
///
/// # Arguments
///
/// * `y_plane` - A slice to load the Y (luminance) plane data.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `u_plane` - A slice to load the U (chrominance) plane data.
/// * `u_stride` - The stride (bytes per row) for the U plane.
/// * `v_plane` - A slice to load the V (chrominance) plane data.
/// * `v_stride` - The stride (bytes per row) for the V plane.
/// * `a_plane` - A slice to load alpha plane to append to result.
/// * `a_stride` - The stride (bytes per row) for the alpha plane.
/// * `width` - The width of the YUV image.
/// * `height` - The height of the YUV image.
/// * `rgba_data` - A mutable slice to store the converted RGBA data.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
/// * `premultiply_alpha` - Flag to premultiply alpha or not
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGRA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv420_with_alpha_to_rgba(
    y_plane: &[u8],
    y_stride: u32,
    u_plane: &[u8],
    u_stride: u32,
    v_plane: &[u8],
    v_stride: u32,
    a_plane: &[u8],
    a_stride: u32,
    rgba: &mut [u8],
    rgba_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    premultiply_alpha: bool,
) {
    yuv_with_alpha_to_rgbx::<{ YuvSourceChannels::Rgba as u8 }, { YuvChromaSample::YUV420 as u8 }>(
        y_plane,
        y_stride,
        u_plane,
        u_stride,
        v_plane,
        v_stride,
        a_plane,
        a_stride,
        rgba,
        rgba_stride,
        width,
        height,
        range,
        matrix,
        premultiply_alpha,
    )
}

/// Convert YUV 420 planar format to BGRA format and appends provided alpha channel.
///
/// This function takes YUV 420 planar format data with 8-bit precision,
/// and converts it to BGRA format with 8-bit per channel precision.
///
/// # Arguments
///
/// * `y_plane` - A slice to load the Y (luminance) plane data.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `u_plane` - A slice to load the U (chrominance) plane data.
/// * `u_stride` - The stride (bytes per row) for the U plane.
/// * `v_plane` - A slice to load the V (chrominance) plane data.
/// * `v_stride` - The stride (bytes per row) for the V plane.
/// * `a_plane` - A slice to load alpha plane to append to result.
/// * `a_stride` - The stride (bytes per row) for the alpha plane.
/// * `width` - The width of the YUV image.
/// * `height` - The height of the YUV image.
/// * `bgra_data` - A mutable slice to store the converted BGRA data.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
/// * `premultiply_alpha` - Flag to premultiply alpha or not
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGRA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv420_with_alpha_to_bgra(
    y_plane: &[u8],
    y_stride: u32,
    u_plane: &[u8],
    u_stride: u32,
    v_plane: &[u8],
    v_stride: u32,
    a_plane: &[u8],
    a_stride: u32,
    bgra: &mut [u8],
    bgra_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    premultiply_alpha: bool,
) {
    yuv_with_alpha_to_rgbx::<{ YuvSourceChannels::Bgra as u8 }, { YuvChromaSample::YUV420 as u8 }>(
        y_plane,
        y_stride,
        u_plane,
        u_stride,
        v_plane,
        v_stride,
        a_plane,
        a_stride,
        bgra,
        bgra_stride,
        width,
        height,
        range,
        matrix,
        premultiply_alpha,
    )
}

/// Convert YUV 422 planar format to RGBA format and appends provided alpha channel.
///
/// This function takes YUV 422 planar format data with 8-bit precision,
/// and converts it to RGBA format with 8-bit per channel precision.
///
/// # Arguments
///
/// * `y_plane` - A slice to load the Y (luminance) plane data.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `u_plane` - A slice to load the U (chrominance) plane data.
/// * `u_stride` - The stride (bytes per row) for the U plane.
/// * `v_plane` - A slice to load the V (chrominance) plane data.
/// * `v_stride` - The stride (bytes per row) for the V plane.
/// * `a_plane` - A slice to load alpha plane to append to result.
/// * `a_stride` - The stride (bytes per row) for the alpha plane.
/// * `width` - The width of the YUV image.
/// * `height` - The height of the YUV image.
/// * `rgba_data` - A mutable slice to store the converted RGBA data.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
/// * `premultiply_alpha` - Flag to premultiply alpha or not
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGRA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv422_with_alpha_to_rgba(
    y_plane: &[u8],
    y_stride: u32,
    u_plane: &[u8],
    u_stride: u32,
    v_plane: &[u8],
    v_stride: u32,
    a_plane: &[u8],
    a_stride: u32,
    rgba: &mut [u8],
    rgba_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    premultiply_alpha: bool,
) {
    yuv_with_alpha_to_rgbx::<{ YuvSourceChannels::Rgba as u8 }, { YuvChromaSample::YUV422 as u8 }>(
        y_plane,
        y_stride,
        u_plane,
        u_stride,
        v_plane,
        v_stride,
        a_plane,
        a_stride,
        rgba,
        rgba_stride,
        width,
        height,
        range,
        matrix,
        premultiply_alpha,
    )
}

/// Convert YUV 422 planar format to BGRA format and appends provided alpha channel.
///
/// This function takes YUV 422 planar format data with 8-bit precision,
/// and converts it to BGRA format with 8-bit per channel precision.
///
/// # Arguments
///
/// * `y_plane` - A slice to load the Y (luminance) plane data.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `u_plane` - A slice to load the U (chrominance) plane data.
/// * `u_stride` - The stride (bytes per row) for the U plane.
/// * `v_plane` - A slice to load the V (chrominance) plane data.
/// * `v_stride` - The stride (bytes per row) for the V plane.
/// * `a_plane` - A slice to load alpha plane to append to result.
/// * `a_stride` - The stride (bytes per row) for the alpha plane.
/// * `width` - The width of the YUV image.
/// * `height` - The height of the YUV image.
/// * `bgra_data` - A mutable slice to store the converted BGRA data.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
/// * `premultiply_alpha` - Flag to premultiply alpha or not
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGRA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv422_with_alpha_to_bgra(
    y_plane: &[u8],
    y_stride: u32,
    u_plane: &[u8],
    u_stride: u32,
    v_plane: &[u8],
    v_stride: u32,
    a_plane: &[u8],
    a_stride: u32,
    bgra: &mut [u8],
    bgra_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    premultiply_alpha: bool,
) {
    yuv_with_alpha_to_rgbx::<{ YuvSourceChannels::Bgra as u8 }, { YuvChromaSample::YUV422 as u8 }>(
        y_plane,
        y_stride,
        u_plane,
        u_stride,
        v_plane,
        v_stride,
        a_plane,
        a_stride,
        bgra,
        bgra_stride,
        width,
        height,
        range,
        matrix,
        premultiply_alpha,
    )
}

/// Convert YUV 444 planar format to RGBA format and appends provided alpha channel.
///
/// This function takes YUV 444 planar format data with 8-bit precision,
/// and converts it to RGBA format with 8-bit per channel precision.
///
/// # Arguments
///
/// * `y_plane` - A slice to load the Y (luminance) plane data.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `u_plane` - A slice to load the U (chrominance) plane data.
/// * `u_stride` - The stride (bytes per row) for the U plane.
/// * `v_plane` - A slice to load the V (chrominance) plane data.
/// * `v_stride` - The stride (bytes per row) for the V plane.
/// * `a_plane` - A slice to load alpha plane to append to result.
/// * `a_stride` - The stride (bytes per row) for the alpha plane.
/// * `width` - The width of the YUV image.
/// * `height` - The height of the YUV image.
/// * `rgba_data` - A mutable slice to store the converted RGBA data.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
/// * `premultiply_alpha` - Flag to premultiply alpha or not
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGRA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv444_with_alpha_to_rgba(
    y_plane: &[u8],
    y_stride: u32,
    u_plane: &[u8],
    u_stride: u32,
    v_plane: &[u8],
    v_stride: u32,
    a_plane: &[u8],
    a_stride: u32,
    rgba: &mut [u8],
    rgba_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    premultiply_alpha: bool,
) {
    yuv_with_alpha_to_rgbx::<{ YuvSourceChannels::Rgba as u8 }, { YuvChromaSample::YUV444 as u8 }>(
        y_plane,
        y_stride,
        u_plane,
        u_stride,
        v_plane,
        v_stride,
        a_plane,
        a_stride,
        rgba,
        rgba_stride,
        width,
        height,
        range,
        matrix,
        premultiply_alpha,
    )
}

/// Convert YUV 444 planar format to BGRA format and appends provided alpha channel.
///
/// This function takes YUV 444 planar format data with 8-bit precision,
/// and converts it to BGRA format with 8-bit per channel precision.
///
/// # Arguments
///
/// * `y_plane` - A slice to load the Y (luminance) plane data.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `u_plane` - A slice to load the U (chrominance) plane data.
/// * `u_stride` - The stride (bytes per row) for the U plane.
/// * `v_plane` - A slice to load the V (chrominance) plane data.
/// * `v_stride` - The stride (bytes per row) for the V plane.
/// * `a_plane` - A slice to load alpha plane to append to result.
/// * `a_stride` - The stride (bytes per row) for the alpha plane.
/// * `width` - The width of the YUV image.
/// * `height` - The height of the YUV image.
/// * `bgra_data` - A mutable slice to store the converted BGRA data.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
/// * `premultiply_alpha` - Flag to premultiply alpha or not
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGRA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv444_with_alpha_to_bgra(
    y_plane: &[u8],
    y_stride: u32,
    u_plane: &[u8],
    u_stride: u32,
    v_plane: &[u8],
    v_stride: u32,
    a_plane: &[u8],
    a_stride: u32,
    bgra: &mut [u8],
    bgra_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
    premultiply_alpha: bool,
) {
    yuv_with_alpha_to_rgbx::<{ YuvSourceChannels::Bgra as u8 }, { YuvChromaSample::YUV444 as u8 }>(
        y_plane,
        y_stride,
        u_plane,
        u_stride,
        v_plane,
        v_stride,
        a_plane,
        a_stride,
        bgra,
        bgra_stride,
        width,
        height,
        range,
        matrix,
        premultiply_alpha,
    )
}
