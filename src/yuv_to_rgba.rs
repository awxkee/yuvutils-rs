#[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
#[cfg(target_feature = "neon")]
use std::arch::aarch64::{
    uint8x16x3_t, uint8x16x4_t, uint8x8_t, vcombine_u8, vdup_n_u8, vdupq_n_s16, vdupq_n_u8,
    vget_high_u8, vget_low_u8, vld1_u8, vld1q_u8, vmaxq_s16, vmovl_u8, vmull_high_u8, vmull_u8,
    vmulq_s16, vqaddq_s16, vqshrun_n_s16, vreinterpretq_s16_u16, vst3q_u8, vst4q_u8, vsubq_s16,
    vsubq_u8, vzip1_u8, vzip2_u8,
};
// #[cfg(
//     all(
//         any(target_arch = "x86", target_arch = "x86_64"),
//         target_feature = "avx2"
//     )
// )]
use std::arch::x86_64::*;

use crate::yuv_support::{
    get_inverse_transform, get_kr_kb, get_yuv_range, CbCrInverseTransform, YuvChromaRange,
    YuvChromaSample, YuvRange, YuvSourceChannels, YuvStandardMatrix,
};

#[allow(dead_code)]
struct ProcessedOffset {
    pub cx: usize,
    pub ux: usize,
}

// #[cfg(any(
//     target_arch = "x86_64",
//     target_feature = "avx2"
// ))]
#[inline(always)]
#[allow(dead_code)]
unsafe fn avx2_process_row(
    range: &YuvChromaRange,
    transform: &CbCrInverseTransform<i32>,
    chroma_subsampling: &YuvChromaSample,
    y_plane: &[u8],
    u_plane: &[u8],
    v_plane: &[u8],
    rgba: &mut [u8],
    start_cx: usize,
    start_ux: usize,
    y_offset: usize,
    u_offset: usize,
    v_offset: usize,
    rgba_offset: usize,
    channels: usize,
    destination_channels: YuvSourceChannels,
    width: usize,
) -> ProcessedOffset {
    let mut cx = start_cx;
    let mut uv_x = start_ux;
    let y_ptr = y_plane.as_ptr();
    let u_ptr = u_plane.as_ptr();
    let v_ptr = v_plane.as_ptr();
    let rgba_ptr = rgba.as_mut_ptr();

    let y_corr = _mm256_set1_epi8(range.bias_y as i8);
    let uv_corr = _mm256_set1_epi16(range.bias_uv as i16);
    let v_luma_coeff = _mm256_set1_epi16(transform.y_coef as i16);
    let v_cr_coeff = _mm256_set1_epi16(transform.cr_coef as i16);
    let v_cb_coeff = _mm256_set1_epi16(transform.cb_coef as i16);
    let v_min_values = _mm256_set1_epi16(0);
    let v_g_coeff_1 = _mm256_set1_epi16(-1 * transform.g_coeff_1 as i16);
    let v_g_coeff_2 = _mm256_set1_epi16(-1 * transform.g_coeff_2 as i16);
    let v_alpha = _mm256_set1_epi8(255u8 as i8);

    while cx + 32 < width {
        let y_values = _mm256_sub_epi8(
            _mm256_loadu_si256(y_ptr.add(y_offset + cx) as *const __m256i),
            y_corr,
        );

        let (u_high_u8, v_high_u8, u_low_u8, v_low_u8);

        match chroma_subsampling {
            YuvChromaSample::YUV420 | YuvChromaSample::YUV422 => {
                let u_values = _mm_loadu_si128(u_ptr.add(u_offset + uv_x) as *const __m128i);
                let v_values = _mm_loadu_si128(v_ptr.add(v_offset + uv_x) as *const __m128i);

                u_high_u8 = _mm_unpackhi_epi8(u_values, u_values);
                v_high_u8 = _mm_unpackhi_epi8(v_values, v_values);
                u_low_u8 = _mm_unpacklo_epi8(u_values, u_values);
                v_low_u8 = _mm_unpacklo_epi8(v_values, v_values);
            }
            YuvChromaSample::YUV444 => {
                let u_values = _mm256_loadu_si256(u_ptr.add(u_offset + uv_x) as *const __m256i);
                let v_values = _mm256_loadu_si256(v_ptr.add(v_offset + uv_x) as *const __m256i);

                u_high_u8 = _mm256_extracti128_si256::<1>(u_values);
                v_high_u8 = _mm256_extracti128_si256::<1>(v_values);
                u_low_u8 = _mm256_castsi256_si128(u_values);
                v_low_u8 = _mm256_castsi256_si128(v_values);
            }
        }

        let u_high = _mm256_sub_epi16(_mm256_cvtepu8_epi16(u_high_u8), uv_corr);
        let v_high = _mm256_sub_epi16(_mm256_cvtepu8_epi16(v_high_u8), uv_corr);
        let y_high = _mm256_mullo_epi16(
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
            v_luma_coeff,
        );

        let r_low = _mm256_srai_epi16::<6>(_mm256_max_epi16(
            _mm256_adds_epi16(y_low, _mm256_mullo_epi16(v_low, v_cr_coeff)),
            v_min_values,
        ));
        let b_low = _mm256_srai_epi16::<6>(_mm256_max_epi16(
            _mm256_adds_epi16(y_low, _mm256_mullo_epi16(u_low, v_cb_coeff)),
            v_min_values,
        ));
        let g_low = _mm256_srai_epi16::<6>(_mm256_max_epi16(
            _mm256_adds_epi16(
                y_low,
                _mm256_adds_epi16(
                    _mm256_mullo_epi16(v_low, v_g_coeff_1),
                    _mm256_mullo_epi16(u_low, v_g_coeff_2),
                ),
            ),
            v_min_values,
        ));

        let r_values = _mm256_packus_epi16(r_low, r_high);
        let g_values = _mm256_packus_epi16(g_low, g_high);
        let b_values = _mm256_packus_epi16(b_low, b_high);

        let dst_shift = rgba_offset + cx * channels;

        match destination_channels {
            YuvSourceChannels::Rgb => {
                let rg_lo = _mm256_unpacklo_epi8(r_values, g_values);
                let rg_hi = _mm256_unpackhi_epi8(r_values, g_values);
                let zero = _mm256_setzero_si256();
                let b0_lo = _mm256_unpacklo_epi8(b_values, zero);
                let b0_hi = _mm256_unpackhi_epi8(b_values, zero);

                let rgb0_lo = _mm256_unpacklo_epi16(rg_lo, b0_lo);
                let rgb0_hi = _mm256_unpackhi_epi16(rg_lo, b0_lo);
                let rgb1_lo = _mm256_unpacklo_epi16(rg_hi, b0_hi);
                let rgb1_hi = _mm256_unpackhi_epi16(rg_hi, b0_hi);

                let shuffle_mask = _mm256_setr_epi8(
                    0, 1, 2,
                    4, 5, 6,
                    8, 9, 10,
                    12, 13, 14,
                    16, 17, 18,
                    20, 21, 22,
                    24,25, 26,
                    28, 29, 30,
                    -1, -1, -1,
                    -1,-1,-1,
                    -1,-1,
                );

                let rgb0 = _mm256_shuffle_epi8(rgb0_lo, shuffle_mask);
                let rgb1 = _mm256_shuffle_epi8(rgb0_hi, shuffle_mask);
                let rgb2 = _mm256_shuffle_epi8(rgb1_lo, shuffle_mask);
                let rgb3 = _mm256_shuffle_epi8(rgb1_hi, shuffle_mask);

                _mm256_storeu_si256(rgba_ptr.add(dst_shift) as *mut __m256i, rgb0);
                _mm256_storeu_si256(rgba_ptr.add(dst_shift + 24) as *mut __m256i, rgb1);
                _mm256_storeu_si256(rgba_ptr.add(dst_shift + 48) as *mut __m256i, rgb2);
                // We need always to write 104 bytes, however 32 initial offset is safe only for 96, then if there are some exceed it is required to use transient buffer
                if cx + 35 < width {
                    _mm256_storeu_si256(rgba_ptr.add(dst_shift + 72) as *mut __m256i, rgb3);
                } else {
                    let mut transient: [u8; 32] = [0u8; 32];
                    _mm256_storeu_si256(transient.as_mut_ptr() as *mut __m256i, rgb3);
                    std::ptr::copy_nonoverlapping(transient.as_ptr(), rgba_ptr.add(dst_shift + 72), 24);
                }
            }
            YuvSourceChannels::Rgba => {
                let rg_low = _mm256_unpacklo_epi8(r_values, g_values); // [r0, g0, r1, g1, r2, g2, r3, g3]
                let rg_high = _mm256_unpackhi_epi8(r_values, g_values); // [r4, g4, r5, g5, r6, g6, r7, g7]
                let ba_low = _mm256_unpacklo_epi8(b_values, v_alpha); // [b0, a0, b1, a1, b2, a2, b3, a3]
                let ba_high = _mm256_unpackhi_epi8(b_values, v_alpha); // [b4, a4, b5, a5, b6, a6, b7, a7]

                // Step 2: Unpack 16-bit integers to 8-bit integers (low and high parts)
                let rgba0 = _mm256_unpacklo_epi16(rg_low, ba_low); // [r0, g0, b0, a0, r1, g1, b1, a1]
                let rgba1 = _mm256_unpackhi_epi16(rg_low, ba_low); // [r2, g2, b2, a2, r3, g3, b3, a3]
                let rgba2 = _mm256_unpacklo_epi16(rg_high, ba_high); // [r4, g4, b4, a4, r5, g5, b5, a5]
                let rgba3 = _mm256_unpackhi_epi16(rg_high, ba_high); //

                _mm256_storeu_si256(rgba_ptr.add(dst_shift) as *mut __m256i, rgba0);
                _mm256_storeu_si256(rgba_ptr.add(dst_shift + 32) as *mut __m256i, rgba1);
                _mm256_storeu_si256(rgba_ptr.add(dst_shift + 64) as *mut __m256i, rgba2);
                _mm256_storeu_si256(rgba_ptr.add(dst_shift + 96) as *mut __m256i, rgba3);
            }
            YuvSourceChannels::Bgra => {
                let rg_low = _mm256_unpacklo_epi8(b_values, g_values); // [r0, g0, r1, g1, r2, g2, r3, g3]
                let rg_high = _mm256_unpackhi_epi8(b_values, g_values); // [r4, g4, r5, g5, r6, g6, r7, g7]
                let ba_low = _mm256_unpacklo_epi8(r_values, v_alpha); // [b0, a0, b1, a1, b2, a2, b3, a3]
                let ba_high = _mm256_unpackhi_epi8(r_values, v_alpha); // [b4, a4, b5, a5, b6, a6, b7, a7]

                // Step 2: Unpack 16-bit integers to 8-bit integers (low and high parts)
                let rgba0 = _mm256_unpacklo_epi16(rg_low, ba_low); // [r0, g0, b0, a0, r1, g1, b1, a1]
                let rgba1 = _mm256_unpackhi_epi16(rg_low, ba_low); // [r2, g2, b2, a2, r3, g3, b3, a3]
                let rgba2 = _mm256_unpacklo_epi16(rg_high, ba_high); // [r4, g4, b4, a4, r5, g5, b5, a5]
                let rgba3 = _mm256_unpackhi_epi16(rg_high, ba_high); //

                _mm256_storeu_si256(rgba_ptr.add(dst_shift) as *mut __m256i, rgba0);
                _mm256_storeu_si256(rgba_ptr.add(dst_shift + 32) as *mut __m256i, rgba1);
                _mm256_storeu_si256(rgba_ptr.add(dst_shift + 64) as *mut __m256i, rgba2);
                _mm256_storeu_si256(rgba_ptr.add(dst_shift + 96) as *mut __m256i, rgba3);
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

    return ProcessedOffset {
        cx: cx,
        ux: uv_x,
    };
}

fn yuv_to_rgbx<const DESTINATION_CHANNELS: u8, const SAMPLING: u8>(
    y_plane: &[u8],
    y_stride: u32,
    u_plane: &[u8],
    u_stride: u32,
    v_plane: &[u8],
    v_stride: u32,
    rgba: &mut [u8],
    rgba_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) {
    let chroma_subsampling: YuvChromaSample = SAMPLING.into();
    let destination_channels: YuvSourceChannels = DESTINATION_CHANNELS.into();
    let channels = destination_channels.get_channels_count();
    let range = get_yuv_range(8, range);
    let kr_kb = get_kr_kb(matrix);
    let transform = get_inverse_transform(255, range.range_y, range.range_uv, kr_kb.kr, kr_kb.kb);
    let inverse_transform = transform.to_integers(6);
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

        // #[cfg(all(
        //     any(target_arch = "x86", target_arch = "x86_64"),
        //     target_feature = "avx2"
        // ))]
        unsafe {
            let processed = avx2_process_row(
                    &range,
                    &inverse_transform,
                    &chroma_subsampling,
                    y_plane,
                    u_plane,
                    v_plane,
                    rgba,
                    cx,
                    uv_x,
                    y_offset,
                    u_offset,
                    v_offset,
                    rgba_offset,
                    channels,
                    destination_channels,
                    width as usize,
                );
            cx += processed.cx;
            uv_x += processed.ux;
        }

        #[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
        #[cfg(target_feature = "neon")]
        unsafe {
            let y_ptr = y_plane.as_ptr();
            let u_ptr = u_plane.as_ptr();
            let v_ptr = v_plane.as_ptr();
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
            let v_alpha = vdupq_n_u8(255u8);

            while cx + 16 < width as usize {
                let y_values = vsubq_u8(vld1q_u8(y_ptr.add(y_offset + cx)), y_corr);

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

                let r_values = vcombine_u8(r_low, r_high);
                let g_values = vcombine_u8(g_low, g_high);
                let b_values = vcombine_u8(b_low, b_high);

                let dst_shift = rgba_offset + cx * channels;

                match destination_channels {
                    YuvSourceChannels::Rgb => {
                        let dst_pack: uint8x16x3_t = uint8x16x3_t(r_values, g_values, b_values);
                        vst3q_u8(rgba_ptr.add(dst_shift), dst_pack);
                    }
                    YuvSourceChannels::Rgba => {
                        let dst_pack: uint8x16x4_t =
                            uint8x16x4_t(r_values, g_values, b_values, v_alpha);
                        vst4q_u8(rgba_ptr.add(dst_shift), dst_pack);
                    }
                    YuvSourceChannels::Bgra => {
                        let dst_pack: uint8x16x4_t =
                            uint8x16x4_t(b_values, g_values, r_values, v_alpha);
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

            let r = ((y_value + cr_coef * cr_value) >> 6).min(255).max(0);
            let b = ((y_value + cb_coef * cb_value) >> 6).min(255).max(0);
            let g = ((y_value - g_coef_1 * cr_value - g_coef_2 * cb_value) >> 6)
                .min(255)
                .max(0);

            let px = x * channels;

            let rgba_shift = rgba_offset + px;

            rgba[rgba_shift + destination_channels.get_r_channel_offset()] = r as u8;
            rgba[rgba_shift + destination_channels.get_g_channel_offset()] = g as u8;
            rgba[rgba_shift + destination_channels.get_b_channel_offset()] = b as u8;
            if destination_channels.has_alpha() {
                rgba[rgba_shift + destination_channels.get_a_channel_offset()] = 255;
            }

            if chroma_subsampling == YuvChromaSample::YUV420
                || chroma_subsampling == YuvChromaSample::YUV422
            {
                if x + 1 < width as usize {
                    let y_value = (y_plane[y_offset + x + 1] as i32 - bias_y) * y_coef;

                    let r = ((y_value + cr_coef * cr_value) >> 6).min(255).max(0);
                    let b = ((y_value + cb_coef * cb_value) >> 6).min(255).max(0);
                    let g = ((y_value - g_coef_1 * cr_value - g_coef_2 * cb_value) >> 6)
                        .min(255)
                        .max(0);

                    let next_px = (x + 1) * channels;

                    let rgba_shift = rgba_offset + next_px;

                    rgba[rgba_shift + destination_channels.get_r_channel_offset()] = r as u8;
                    rgba[rgba_shift + destination_channels.get_g_channel_offset()] = g as u8;
                    rgba[rgba_shift + destination_channels.get_b_channel_offset()] = b as u8;
                    if destination_channels.has_alpha() {
                        rgba[rgba_shift + destination_channels.get_a_channel_offset()] = 255;
                    }
                }
            }

            uv_x += 1;
        }

        y_offset += y_stride as usize;
        rgba_offset += rgba_stride as usize;
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

/// Convert YUV 420 planar format to RGB format.
///
/// This function takes YUV 420 planar format data with 8-bit precision,
/// and converts it to RGB format with 8-bit per channel precision.
///
/// # Arguments
///
/// * `y_plane` - A slice to load the Y (luminance) plane data.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `u_plane` - A slice to load the U (chrominance) plane data.
/// * `u_stride` - The stride (bytes per row) for the U plane.
/// * `v_plane` - A slice to load the V (chrominance) plane data.
/// * `v_stride` - The stride (bytes per row) for the V plane.
/// * `width` - The width of the YUV image.
/// * `height` - The height of the YUV image.
/// * `rgb_data` - A mutable slice to store the converted RGB data.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGRA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv420_to_rgb(
    y_plane: &[u8],
    y_stride: u32,
    u_plane: &[u8],
    u_stride: u32,
    v_plane: &[u8],
    v_stride: u32,
    rgb: &mut [u8],
    rgb_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) {
    yuv_to_rgbx::<{ YuvSourceChannels::Rgb as u8 }, { YuvChromaSample::YUV420 as u8 }>(
        y_plane, y_stride, u_plane, u_stride, v_plane, v_stride, rgb, rgb_stride, width, height,
        range, matrix,
    )
}

/// Convert YUV 420 planar format to RGBA format.
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
/// * `width` - The width of the YUV image.
/// * `height` - The height of the YUV image.
/// * `rgba_data` - A mutable slice to store the converted RGBA data.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGRA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv420_to_rgba(
    y_plane: &[u8],
    y_stride: u32,
    u_plane: &[u8],
    u_stride: u32,
    v_plane: &[u8],
    v_stride: u32,
    rgba: &mut [u8],
    rgba_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) {
    yuv_to_rgbx::<{ YuvSourceChannels::Rgba as u8 }, { YuvChromaSample::YUV420 as u8 }>(
        y_plane,
        y_stride,
        u_plane,
        u_stride,
        v_plane,
        v_stride,
        rgba,
        rgba_stride,
        width,
        height,
        range,
        matrix,
    )
}

/// Convert YUV 420 planar format to BGRA format.
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
/// * `width` - The width of the YUV image.
/// * `height` - The height of the YUV image.
/// * `bgra_data` - A mutable slice to store the converted BGRA data.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGRA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv420_to_bgra(
    y_plane: &[u8],
    y_stride: u32,
    u_plane: &[u8],
    u_stride: u32,
    v_plane: &[u8],
    v_stride: u32,
    bgra: &mut [u8],
    bgra_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) {
    yuv_to_rgbx::<{ YuvSourceChannels::Bgra as u8 }, { YuvChromaSample::YUV420 as u8 }>(
        y_plane,
        y_stride,
        u_plane,
        u_stride,
        v_plane,
        v_stride,
        bgra,
        bgra_stride,
        width,
        height,
        range,
        matrix,
    )
}

/// Convert YUV 422 planar format to RGB format.
///
/// This function takes YUV 422 data with 8-bit precision,
/// and converts it to RGB format with 8-bit per channel precision.
///
/// # Arguments
///
/// * `y_plane` - A slice to load the Y (luminance) plane data.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `u_plane` - A slice to load the U (chrominance) plane data.
/// * `u_stride` - The stride (bytes per row) for the U plane.
/// * `v_plane` - A slice to load the V (chrominance) plane data.
/// * `v_stride` - The stride (bytes per row) for the V plane.
/// * `width` - The width of the YUV image.
/// * `height` - The height of the YUV image.
/// * `rgb_data` - A mutable slice to store the converted RGB data.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGRA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv422_to_rgb(
    y_plane: &[u8],
    y_stride: u32,
    u_plane: &[u8],
    u_stride: u32,
    v_plane: &[u8],
    v_stride: u32,
    rgb: &mut [u8],
    rgb_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) {
    yuv_to_rgbx::<{ YuvSourceChannels::Rgb as u8 }, { YuvChromaSample::YUV422 as u8 }>(
        y_plane, y_stride, u_plane, u_stride, v_plane, v_stride, rgb, rgb_stride, width, height,
        range, matrix,
    )
}

/// Convert YUV 422 planar format to RGBA format.
///
/// This function takes YUV 422 data with 8-bit precision,
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
/// * `width` - The width of the YUV image.
/// * `height` - The height of the YUV image.
/// * `bgra_data` - A mutable slice to store the converted RGBA data.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGRA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv422_to_rgba(
    y_plane: &[u8],
    y_stride: u32,
    u_plane: &[u8],
    u_stride: u32,
    v_plane: &[u8],
    v_stride: u32,
    rgba: &mut [u8],
    rgba_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) {
    yuv_to_rgbx::<{ YuvSourceChannels::Rgba as u8 }, { YuvChromaSample::YUV422 as u8 }>(
        y_plane,
        y_stride,
        u_plane,
        u_stride,
        v_plane,
        v_stride,
        rgba,
        rgba_stride,
        width,
        height,
        range,
        matrix,
    )
}

/// Convert YUV 422 planar format to BGRA format.
///
/// This function takes YUV 422 data with 8-bit precision,
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
/// * `width` - The width of the YUV image.
/// * `height` - The height of the YUV image.
/// * `bgra_data` - A mutable slice to store the converted BGRA data.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGRA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv422_to_bgra(
    y_plane: &[u8],
    y_stride: u32,
    u_plane: &[u8],
    u_stride: u32,
    v_plane: &[u8],
    v_stride: u32,
    bgra: &mut [u8],
    bgra_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) {
    yuv_to_rgbx::<{ YuvSourceChannels::Bgra as u8 }, { YuvChromaSample::YUV422 as u8 }>(
        y_plane,
        y_stride,
        u_plane,
        u_stride,
        v_plane,
        v_stride,
        bgra,
        bgra_stride,
        width,
        height,
        range,
        matrix,
    )
}

/// Convert YUV 444 planar format to RGBA format.
///
/// This function takes YUV 444 data with 8-bit precision,
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
/// * `width` - The width of the YUV image.
/// * `height` - The height of the YUV image.
/// * `rgba_data` - A mutable slice to store the converted RGBA data.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGRA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv444_to_rgba(
    y_plane: &[u8],
    y_stride: u32,
    u_plane: &[u8],
    u_stride: u32,
    v_plane: &[u8],
    v_stride: u32,
    rgba: &mut [u8],
    rgba_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) {
    yuv_to_rgbx::<{ YuvSourceChannels::Rgba as u8 }, { YuvChromaSample::YUV444 as u8 }>(
        y_plane,
        y_stride,
        u_plane,
        u_stride,
        v_plane,
        v_stride,
        rgba,
        rgba_stride,
        width,
        height,
        range,
        matrix,
    )
}

/// Convert YUV 444 planar format to BGRA format.
///
/// This function takes YUV 444 data with 8-bit precision,
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
/// * `width` - The width of the YUV image.
/// * `height` - The height of the YUV image.
/// * `bgra_data` - A mutable slice to store the converted BGRA data.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGRA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv444_to_bgra(
    y_plane: &[u8],
    y_stride: u32,
    u_plane: &[u8],
    u_stride: u32,
    v_plane: &[u8],
    v_stride: u32,
    bgra: &mut [u8],
    bgra_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) {
    yuv_to_rgbx::<{ YuvSourceChannels::Bgra as u8 }, { YuvChromaSample::YUV444 as u8 }>(
        y_plane,
        y_stride,
        u_plane,
        u_stride,
        v_plane,
        v_stride,
        bgra,
        bgra_stride,
        width,
        height,
        range,
        matrix,
    )
}

/// Convert YUV 444 planar format to RGB format.
///
/// This function takes YUV 444 data with 8-bit precision,
/// and converts it to RGB format with 8-bit per channel precision.
///
/// # Arguments
///
/// * `y_plane` - A slice to load the Y (luminance) plane data.
/// * `y_stride` - The stride (bytes per row) for the Y plane.
/// * `u_plane` - A slice to load the U (chrominance) plane data.
/// * `u_stride` - The stride (bytes per row) for the U plane.
/// * `v_plane` - A slice to load the V (chrominance) plane data.
/// * `v_stride` - The stride (bytes per row) for the V plane.
/// * `width` - The width of the YUV image.
/// * `height` - The height of the YUV image.
/// * `rgb_data` - A mutable slice to store the converted RGB data.
/// * `range` - The YUV range (limited or full).
/// * `matrix` - The YUV standard matrix (BT.601 or BT.709 or BT.2020 or other).
///
/// # Panics
///
/// This function panics if the lengths of the planes or the input BGRA data are not valid based
/// on the specified width, height, and strides, or if invalid YUV range or matrix is provided.
///
pub fn yuv444_to_rgb(
    y_plane: &[u8],
    y_stride: u32,
    u_plane: &[u8],
    u_stride: u32,
    v_plane: &[u8],
    v_stride: u32,
    rgb: &mut [u8],
    rgb_stride: u32,
    width: u32,
    height: u32,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) {
    yuv_to_rgbx::<{ YuvSourceChannels::Rgb as u8 }, { YuvChromaSample::YUV444 as u8 }>(
        y_plane, y_stride, u_plane, u_stride, v_plane, v_stride, rgb, rgb_stride, width, height,
        range, matrix,
    )
}
