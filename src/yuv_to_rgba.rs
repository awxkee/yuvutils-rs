#[cfg(target_arch = "x86_64")]
#[allow(unused_imports)]
use crate::intel_simd_support::*;
#[allow(unused_imports)]
use crate::internals::ProcessedOffset;
#[allow(unused_imports)]
use crate::yuv_support::*;
#[cfg(target_arch = "aarch64")]
#[cfg(target_feature = "neon")]
use std::arch::aarch64::*;
#[cfg(target_arch = "x86_64")]
#[allow(unused_imports)]
use std::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
#[inline(always)]
#[allow(dead_code)]
unsafe fn avx2_process_row<const DESTINATION_CHANNELS: u8, const SAMPLING: u8>(
    range: &YuvChromaRange,
    transform: &CbCrInverseTransform<i32>,
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
    width: usize,
) -> ProcessedOffset {
    let chroma_subsampling: YuvChromaSample = SAMPLING.into();
    let destination_channels: YuvSourceChannels = DESTINATION_CHANNELS.into();
    let channels = destination_channels.get_channels_count();

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
    let v_min_values = _mm256_setzero_si256();
    let v_g_coeff_1 = _mm256_set1_epi16(-1 * transform.g_coeff_1 as i16);
    let v_g_coeff_2 = _mm256_set1_epi16(-1 * transform.g_coeff_2 as i16);
    let v_alpha = _mm256_set1_epi8(255u8 as i8);

    while cx + 32 < width {
        let y_values = _mm256_subs_epi8(
            _mm256_loadu_si256(y_ptr.add(y_offset + cx) as *const __m256i),
            y_corr,
        );

        let (u_high_u8, v_high_u8, u_low_u8, v_low_u8);

        match chroma_subsampling {
            YuvChromaSample::YUV420 | YuvChromaSample::YUV422 => {
                let u_values = _mm_loadu_si128(u_ptr.add(u_offset + uv_x) as *const __m128i);
                let v_values = _mm_loadu_si128(v_ptr.add(v_offset + uv_x) as *const __m128i);

                u_high_u8 = sse_interleave_even(_mm_unpackhi_epi8(u_values, u_values));
                v_high_u8 = sse_interleave_odd(_mm_unpackhi_epi8(v_values, v_values));
                u_low_u8 = sse_interleave_even(_mm_unpacklo_epi8(u_values, u_values));
                v_low_u8 = sse_interleave_odd(_mm_unpacklo_epi8(v_values, v_values));
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

        let u_high = _mm256_subs_epi16(_mm256_cvtepu8_epi16(u_high_u8), uv_corr);
        let v_high = _mm256_subs_epi16(_mm256_cvtepu8_epi16(v_high_u8), uv_corr);
        let y_high = _mm256_mullo_epi16(
            _mm256_cvtepu8_epi16(_mm256_extracti128_si256::<1>(y_values)),
            v_luma_coeff,
        );

        let r_high = _mm256_srli_epi16::<6>(_mm256_max_epi16(
            _mm256_adds_epi16(y_high, _mm256_mullo_epi16(v_high, v_cr_coeff)),
            v_min_values,
        ));
        let b_high = _mm256_srli_epi16::<6>(_mm256_max_epi16(
            _mm256_adds_epi16(y_high, _mm256_mullo_epi16(u_high, v_cb_coeff)),
            v_min_values,
        ));
        let g_high = _mm256_srli_epi16::<6>(_mm256_max_epi16(
            _mm256_adds_epi16(
                y_high,
                _mm256_adds_epi16(
                    _mm256_mullo_epi16(v_high, v_g_coeff_1),
                    _mm256_mullo_epi16(u_high, v_g_coeff_2),
                ),
            ),
            v_min_values,
        ));

        let u_low = _mm256_subs_epi16(_mm256_cvtepu8_epi16(u_low_u8), uv_corr);
        let v_low = _mm256_subs_epi16(_mm256_cvtepu8_epi16(v_low_u8), uv_corr);
        let y_low = _mm256_mullo_epi16(
            _mm256_cvtepu8_epi16(_mm256_castsi256_si128(y_values)),
            v_luma_coeff,
        );

        let r_low = _mm256_srli_epi16::<6>(_mm256_max_epi16(
            _mm256_adds_epi16(y_low, _mm256_mullo_epi16(v_low, v_cr_coeff)),
            v_min_values,
        ));
        let b_low = _mm256_srli_epi16::<6>(_mm256_max_epi16(
            _mm256_adds_epi16(y_low, _mm256_mullo_epi16(u_low, v_cb_coeff)),
            v_min_values,
        ));
        let g_low = _mm256_srli_epi16::<6>(_mm256_max_epi16(
            _mm256_adds_epi16(
                y_low,
                _mm256_adds_epi16(
                    _mm256_mullo_epi16(v_low, v_g_coeff_1),
                    _mm256_mullo_epi16(u_low, v_g_coeff_2),
                ),
            ),
            v_min_values,
        ));

        let r_values = demote_i16_to_u8(r_low, r_high);
        let g_values = demote_i16_to_u8(g_low, g_high);
        let b_values = demote_i16_to_u8(b_low, b_high);

        let dst_shift = rgba_offset + cx * channels;

        match destination_channels {
            YuvSourceChannels::Rgb => {
                // We need always to write 104 bytes, however 32 initial offset is safe only for 96, then if there are some exceed it is required to use transient buffer
                let ptr = rgba_ptr.add(dst_shift);
                avx2_store_u8_rgb(ptr, r_values, g_values, b_values);
            }
            YuvSourceChannels::Rgba => {
                avx2_store_u8_rgba(
                    rgba_ptr.add(dst_shift),
                    r_values,
                    g_values,
                    b_values,
                    v_alpha,
                );
            }
            YuvSourceChannels::Bgra => {
                avx2_store_u8_rgba(
                    rgba_ptr.add(dst_shift),
                    b_values,
                    g_values,
                    r_values,
                    v_alpha,
                );
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

    return ProcessedOffset { cx, ux: uv_x };
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
#[allow(dead_code)]
unsafe fn sse42_process_row<const DESTINATION_CHANNELS: u8, const SAMPLING: u8>(
    range: &YuvChromaRange,
    transform: &CbCrInverseTransform<i32>,
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
    width: usize,
) -> ProcessedOffset {
    let chroma_subsampling: YuvChromaSample = SAMPLING.into();
    let destination_channels: YuvSourceChannels = DESTINATION_CHANNELS.into();
    let channels = destination_channels.get_channels_count();

    let mut cx = start_cx;
    let mut uv_x = start_ux;

    let y_ptr = y_plane.as_ptr();
    let u_ptr = u_plane.as_ptr();
    let v_ptr = v_plane.as_ptr();
    let rgba_ptr = rgba.as_mut_ptr();

    let y_corr = _mm_set1_epi8(range.bias_y as i8);
    let uv_corr = _mm_set1_epi16(range.bias_uv as i16);
    let v_luma_coeff = _mm_set1_epi16(transform.y_coef as i16);
    let v_cr_coeff = _mm_set1_epi16(transform.cr_coef as i16);
    let v_cb_coeff = _mm_set1_epi16(transform.cb_coef as i16);
    let v_min_values = _mm_setzero_si128();
    let v_g_coeff_1 = _mm_set1_epi16(-1 * transform.g_coeff_1 as i16);
    let v_g_coeff_2 = _mm_set1_epi16(-1 * transform.g_coeff_2 as i16);
    let v_alpha = _mm_set1_epi8(255u8 as i8);

    while cx + 16 < width {
        let y_values = _mm_subs_epi8(
            _mm_loadu_si128(y_ptr.add(y_offset + cx) as *const __m128i),
            y_corr,
        );

        let (u_high_u8, v_high_u8, u_low_u8, v_low_u8);

        match chroma_subsampling {
            YuvChromaSample::YUV420 | YuvChromaSample::YUV422 => {
                let (u_values, v_values);
                if cx + 24 < width {
                    u_values = _mm_loadu_si128(u_ptr.add(u_offset + uv_x) as *const __m128i);
                    v_values = _mm_loadu_si128(v_ptr.add(v_offset + uv_x) as *const __m128i);
                } else {
                    let mut transient_u: [u8; 16] = [0u8; 16];
                    let mut transient_v: [u8; 16] = [0u8; 16];
                    std::ptr::copy_nonoverlapping(
                        u_ptr.add(u_offset + uv_x),
                        transient_u.as_mut_ptr(),
                        8,
                    );
                    std::ptr::copy_nonoverlapping(
                        v_ptr.add(u_offset + uv_x),
                        transient_v.as_mut_ptr(),
                        8,
                    );
                    u_values = _mm_loadu_si128(transient_u.as_ptr() as *const __m128i);
                    v_values = _mm_loadu_si128(transient_v.as_ptr() as *const __m128i);
                }

                u_high_u8 = _mm_unpackhi_epi8(u_values, u_values);
                v_high_u8 = _mm_unpackhi_epi8(v_values, v_values);
                u_low_u8 = _mm_unpacklo_epi8(u_values, u_values);
                v_low_u8 = _mm_unpacklo_epi8(v_values, v_values);
            }
            YuvChromaSample::YUV444 => {
                let u_values = _mm_loadu_si128(u_ptr.add(u_offset + uv_x) as *const __m128i);
                let v_values = _mm_loadu_si128(v_ptr.add(v_offset + uv_x) as *const __m128i);

                u_high_u8 = _mm_unpackhi_epi8(u_values, u_values);
                v_high_u8 = _mm_unpackhi_epi8(v_values, v_values);
                u_low_u8 = _mm_unpacklo_epi8(u_values, u_values);
                v_low_u8 = _mm_unpacklo_epi8(v_values, v_values);
            }
        }

        let u_high = _mm_subs_epi16(_mm_cvtepu8_epi16(u_high_u8), uv_corr);
        let v_high = _mm_subs_epi16(_mm_cvtepu8_epi16(v_high_u8), uv_corr);
        let y_high = _mm_mullo_epi16(
            _mm_cvtepu8_epi16(_mm_srli_si128::<8>(y_values)),
            v_luma_coeff,
        );

        let r_high = _mm_srai_epi16::<6>(_mm_max_epi16(
            _mm_adds_epi16(y_high, _mm_mullo_epi16(v_high, v_cr_coeff)),
            v_min_values,
        ));
        let b_high = _mm_srai_epi16::<6>(_mm_max_epi16(
            _mm_adds_epi16(y_high, _mm_mullo_epi16(u_high, v_cb_coeff)),
            v_min_values,
        ));
        let g_high = _mm_srai_epi16::<6>(_mm_max_epi16(
            _mm_adds_epi16(
                y_high,
                _mm_adds_epi16(
                    _mm_mullo_epi16(v_high, v_g_coeff_1),
                    _mm_mullo_epi16(u_high, v_g_coeff_2),
                ),
            ),
            v_min_values,
        ));

        let u_low = _mm_sub_epi16(_mm_cvtepu8_epi16(u_low_u8), uv_corr);
        let v_low = _mm_sub_epi16(_mm_cvtepu8_epi16(v_low_u8), uv_corr);
        let y_low = _mm_mullo_epi16(
            _mm_cvtepu8_epi16(y_values),
            v_luma_coeff,
        );

        let r_low = _mm_srai_epi16::<6>(_mm_max_epi16(
            _mm_adds_epi16(y_low, _mm_mullo_epi16(v_low, v_cr_coeff)),
            v_min_values,
        ));
        let b_low = _mm_srai_epi16::<6>(_mm_max_epi16(
            _mm_adds_epi16(y_low, _mm_mullo_epi16(u_low, v_cb_coeff)),
            v_min_values,
        ));
        let g_low = _mm_srai_epi16::<6>(_mm_max_epi16(
            _mm_adds_epi16(
                y_low,
                _mm_adds_epi16(
                    _mm_mullo_epi16(v_low, v_g_coeff_1),
                    _mm_mullo_epi16(u_low, v_g_coeff_2),
                ),
            ),
            v_min_values,
        ));

        let r_values = _mm_packus_epi16(r_low, r_high);
        let g_values = _mm_packus_epi16(g_low, g_high);
        let b_values = _mm_packus_epi16(b_low, b_high);

        let dst_shift = rgba_offset + cx * channels;

        match destination_channels {
            YuvSourceChannels::Rgb => {
                sse_store_rgb_u8(
                    rgba_ptr.add(dst_shift),
                    r_values,
                    g_values,
                    b_values,
                );
            }
            YuvSourceChannels::Rgba => {
                sse_store_rgba(
                    rgba_ptr.add(dst_shift),
                    r_values,
                    g_values,
                    b_values,
                    v_alpha,
                );
            }
            YuvSourceChannels::Bgra => {
                sse_store_rgba(
                    rgba_ptr.add(dst_shift),
                    b_values,
                    g_values,
                    r_values,
                    v_alpha,
                );
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

    return ProcessedOffset { cx, ux: uv_x };
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
    let cr_coef = inverse_transform.cr_coef;
    let cb_coef = inverse_transform.cb_coef;
    let y_coef = inverse_transform.y_coef;
    let g_coef_1 = inverse_transform.g_coeff_1;
    let g_coef_2 = inverse_transform.g_coeff_2;

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

    #[cfg(target_arch = "x86_64")]
    let mut use_avx2 = false;
    #[cfg(target_arch = "x86_64")]
    let mut use_sse = false;

    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            use_avx2 = true;
        } else if std::arch::is_x86_feature_detected!("sse4.1") {
            use_sse = true;
        }
    }

    for y in 0..height as usize {
        #[allow(unused_variables)]
        #[allow(unused_mut)]
        let mut cx = 0usize;

        #[allow(unused_variables)]
        #[allow(unused_mut)]
        let mut uv_x = 0usize;

        #[cfg(all(target_arch = "x86_64"))]
        unsafe {
            if use_avx2 {
                let processed = avx2_process_row::<DESTINATION_CHANNELS, SAMPLING>(
                    &range,
                    &inverse_transform,
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
                    width as usize,
                );
                cx += processed.cx;
                uv_x += processed.ux;
            } else if use_sse {
                let processed = sse42_process_row::<DESTINATION_CHANNELS, SAMPLING>(
                    &range,
                    &inverse_transform,
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
                    width as usize,
                );
                cx += processed.cx;
                uv_x += processed.ux;
            }
        }

        #[cfg(target_arch = "aarch64")]
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
                let next_x = x + 1;
                if next_x < width as usize {
                    let y_value = (y_plane[y_offset + next_x] as i32 - bias_y) * y_coef;

                    let r = ((y_value + cr_coef * cr_value) >> 6).min(255).max(0);
                    let b = ((y_value + cb_coef * cb_value) >> 6).min(255).max(0);
                    let g = ((y_value - g_coef_1 * cr_value - g_coef_2 * cb_value) >> 6)
                        .min(255)
                        .max(0);

                    let next_px = next_x * channels;

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
