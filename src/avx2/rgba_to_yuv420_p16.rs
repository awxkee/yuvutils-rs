//! AVX2 RGBA/RGB to YUV420 at precision 16 with split green coefficient.
//!
//! The BT.601 16-bit `yg = 33059` exceeds `i16::MAX` (32767), preventing
//! direct use of PMADDWD which packs coefficients as i16 pairs. We decompose
//! `yg = yg_a + yg_b` where both halves fit i16, then use two PMADDWD calls
//! for the Y green channel and sum the results. UV coefficients all fit i16
//! and use the standard `_mm256_affine_dot` path unchanged.

use crate::avx2::avx2_utils::{
    _mm256_affine_dot, _mm256_interleave_epi16, _mm256_load_deinterleave_rgb_for_yuv,
    avx2_pack_u16, avx2_pack_u32, avx_pairwise_avg_epi16_epi8_j,
};
use crate::internals::{tail_420, ProcessedOffset};
use crate::yuv_support::{CbCrForwardTransform, YuvChromaRange, YuvSourceChannels};
#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Pack two i16 values into a single i32: lo in bits [0:15], hi in bits [16:31].
#[inline(always)]
const fn interleave_i16_pair(lo: i16, hi: i16) -> i32 {
    (lo as u16 as i32) | ((hi as i32) << 16)
}

#[inline(always)]
unsafe fn encode_32_part_p16<const ORIGIN_CHANNELS: u8>(
    src0: &[u8],
    src1: &[u8],
    y_dst0: &mut [u8],
    y_dst1: &mut [u8],
    u_dst: &mut [u8],
    v_dst: &mut [u8],
    y_bias: __m256i,
    uv_bias: __m256i,
    v_yr_yga: __m256i,
    v_0_ygb: __m256i,
    v_yb: __m256i,
    v_cb_r_g: __m256i,
    v_cb_b: __m256i,
    v_cr_r_g: __m256i,
    v_cr_b: __m256i,
) {
    const PRECISION: i32 = 16;
    let zero = _mm256_setzero_si256();

    // --- Load and deinterleave 32 RGBA pixels from each of the two rows ---
    let (r_values0, g_values0, b_values0) =
        _mm256_load_deinterleave_rgb_for_yuv::<ORIGIN_CHANNELS>(src0.as_ptr());
    let (r_values1, g_values1, b_values1) =
        _mm256_load_deinterleave_rgb_for_yuv::<ORIGIN_CHANNELS>(src1.as_ptr());

    // --- Row 0, low 16 pixels: compute Y = (yr*R + yg*G + yb*B + bias) >> 16 ---
    // Widen u8→u16 for the low 16 pixels of each channel
    let r_low0 = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(r_values0));
    let gl0 = _mm256_unpacklo_epi8(g_values0, zero);
    let bl0 = _mm256_unpacklo_epi8(b_values0, zero);

    // Interleave [R,G] and [B,0] pairs for PMADDWD
    let (rg_lo0, rg_lo1) = _mm256_interleave_epi16(r_low0, gl0);
    let (b_lo0, b_lo1) = _mm256_interleave_epi16(bl0, zero);

    // Split yg multiply: PMADDWD([R,G], [yr, yg_a]) + PMADDWD([R,G], [0, yg_b])
    // This avoids i16 overflow since yg > i16::MAX but yg_a and yg_b each fit
    let rg_a_lo0 = _mm256_madd_epi16(rg_lo0, v_yr_yga);
    let rg_a_lo1 = _mm256_madd_epi16(rg_lo1, v_yr_yga);
    let rg_b_lo0 = _mm256_madd_epi16(rg_lo0, v_0_ygb);
    let rg_b_lo1 = _mm256_madd_epi16(rg_lo1, v_0_ygb);
    let b_w_lo0 = _mm256_madd_epi16(b_lo0, v_yb);
    let b_w_lo1 = _mm256_madd_epi16(b_lo1, v_yb);

    // Sum all components + bias, then right-shift by PRECISION to get Y
    let sum_lo0 = _mm256_add_epi32(
        _mm256_add_epi32(rg_a_lo0, rg_b_lo0),
        _mm256_add_epi32(b_w_lo0, y_bias),
    );
    let sum_lo1 = _mm256_add_epi32(
        _mm256_add_epi32(rg_a_lo1, rg_b_lo1),
        _mm256_add_epi32(b_w_lo1, y_bias),
    );
    // Pack i32→u16 after shifting
    let y00_vl = avx2_pack_u32(
        _mm256_srli_epi32::<PRECISION>(sum_lo0),
        _mm256_srli_epi32::<PRECISION>(sum_lo1),
    );

    // --- Row 0, high 16 pixels: same Y computation for pixels 16-31 ---
    let r_high0 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256::<1>(r_values0));
    let gh0 = _mm256_unpackhi_epi8(g_values0, zero);
    let bh0 = _mm256_unpackhi_epi8(b_values0, zero);

    let (rg_hi0, rg_hi1) = _mm256_interleave_epi16(r_high0, gh0);
    let (b_hi0, b_hi1) = _mm256_interleave_epi16(bh0, zero);

    let rg_a_hi0 = _mm256_madd_epi16(rg_hi0, v_yr_yga);
    let rg_a_hi1 = _mm256_madd_epi16(rg_hi1, v_yr_yga);
    let rg_b_hi0 = _mm256_madd_epi16(rg_hi0, v_0_ygb);
    let rg_b_hi1 = _mm256_madd_epi16(rg_hi1, v_0_ygb);
    let b_w_hi0 = _mm256_madd_epi16(b_hi0, v_yb);
    let b_w_hi1 = _mm256_madd_epi16(b_hi1, v_yb);

    let sum_hi0 = _mm256_add_epi32(
        _mm256_add_epi32(rg_a_hi0, rg_b_hi0),
        _mm256_add_epi32(b_w_hi0, y_bias),
    );
    let sum_hi1 = _mm256_add_epi32(
        _mm256_add_epi32(rg_a_hi1, rg_b_hi1),
        _mm256_add_epi32(b_w_hi1, y_bias),
    );
    let y01_vl = avx2_pack_u32(
        _mm256_srli_epi32::<PRECISION>(sum_hi0),
        _mm256_srli_epi32::<PRECISION>(sum_hi1),
    );

    // Pack u16→u8 and store all 32 Y values for row 0
    let y0 = _mm256_packus_epi16(y00_vl, y01_vl);
    _mm256_storeu_si256(y_dst0.as_mut_ptr().cast::<__m256i>(), y0);

    // --- Row 1: same Y computation for the second row ---
    let r_low1 = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(r_values1));
    let gl1 = _mm256_unpacklo_epi8(g_values1, zero);
    let bl1 = _mm256_unpacklo_epi8(b_values1, zero);

    let (rg_lo10, rg_lo11) = _mm256_interleave_epi16(r_low1, gl1);
    let (b_lo10, b_lo11) = _mm256_interleave_epi16(bl1, zero);

    let rg_a_lo10 = _mm256_madd_epi16(rg_lo10, v_yr_yga);
    let rg_a_lo11 = _mm256_madd_epi16(rg_lo11, v_yr_yga);
    let rg_b_lo10 = _mm256_madd_epi16(rg_lo10, v_0_ygb);
    let rg_b_lo11 = _mm256_madd_epi16(rg_lo11, v_0_ygb);
    let b_w_lo10 = _mm256_madd_epi16(b_lo10, v_yb);
    let b_w_lo11 = _mm256_madd_epi16(b_lo11, v_yb);

    let sum_lo10 = _mm256_add_epi32(
        _mm256_add_epi32(rg_a_lo10, rg_b_lo10),
        _mm256_add_epi32(b_w_lo10, y_bias),
    );
    let sum_lo11 = _mm256_add_epi32(
        _mm256_add_epi32(rg_a_lo11, rg_b_lo11),
        _mm256_add_epi32(b_w_lo11, y_bias),
    );
    let y10_vl = avx2_pack_u32(
        _mm256_srli_epi32::<PRECISION>(sum_lo10),
        _mm256_srli_epi32::<PRECISION>(sum_lo11),
    );

    let r_high1 = _mm256_cvtepu8_epi16(_mm256_extracti128_si256::<1>(r_values1));
    let gh1 = _mm256_unpackhi_epi8(g_values1, zero);
    let bh1 = _mm256_unpackhi_epi8(b_values1, zero);

    let (rg_hi10, rg_hi11) = _mm256_interleave_epi16(r_high1, gh1);
    let (b_hi10, b_hi11) = _mm256_interleave_epi16(bh1, zero);

    let rg_a_hi10 = _mm256_madd_epi16(rg_hi10, v_yr_yga);
    let rg_a_hi11 = _mm256_madd_epi16(rg_hi11, v_yr_yga);
    let rg_b_hi10 = _mm256_madd_epi16(rg_hi10, v_0_ygb);
    let rg_b_hi11 = _mm256_madd_epi16(rg_hi11, v_0_ygb);
    let b_w_hi10 = _mm256_madd_epi16(b_hi10, v_yb);
    let b_w_hi11 = _mm256_madd_epi16(b_hi11, v_yb);

    let sum_hi10 = _mm256_add_epi32(
        _mm256_add_epi32(rg_a_hi10, rg_b_hi10),
        _mm256_add_epi32(b_w_hi10, y_bias),
    );
    let sum_hi11 = _mm256_add_epi32(
        _mm256_add_epi32(rg_a_hi11, rg_b_hi11),
        _mm256_add_epi32(b_w_hi11, y_bias),
    );
    let y11_vl = avx2_pack_u32(
        _mm256_srli_epi32::<PRECISION>(sum_hi10),
        _mm256_srli_epi32::<PRECISION>(sum_hi11),
    );

    // Pack u16→u8 and store all 32 Y values for row 1
    let y1 = _mm256_packus_epi16(y10_vl, y11_vl);
    _mm256_storeu_si256(y_dst1.as_mut_ptr().cast::<__m256i>(), y1);

    // --- Chroma (Cb/Cr): 4:2:0 subsampling by averaging 2x2 pixel blocks ---
    // Vertical average of the two rows
    let r_avg = _mm256_avg_epu8(r_values0, r_values1);
    let g_avg = _mm256_avg_epu8(g_values0, g_values1);
    let b_avg = _mm256_avg_epu8(b_values0, b_values1);

    // Horizontal pairwise average: sums adjacent u8 pairs into u16 (range 0..510)
    // This adds +1 to the effective precision (values are 2x), so precision_uv = 17
    let r1 = avx_pairwise_avg_epi16_epi8_j(r_avg, 1);
    let g1 = avx_pairwise_avg_epi16_epi8_j(g_avg, 1);
    let b1 = avx_pairwise_avg_epi16_epi8_j(b_avg, 1);

    // Interleave for PMADDWD and compute Cb/Cr via affine_dot at precision 17
    let (rhv0, rhv1) = _mm256_interleave_epi16(r1, g1);
    let (bhv0, bhv1) = _mm256_interleave_epi16(b1, zero);

    let cb_s = _mm256_affine_dot::<17, false>(uv_bias, rhv0, rhv1, bhv0, bhv1, v_cb_r_g, v_cb_b);
    let cr_s = _mm256_affine_dot::<17, false>(uv_bias, rhv0, rhv1, bhv0, bhv1, v_cr_r_g, v_cr_b);

    // Pack to u8 and store 16 Cb/Cr values (32 luma pixels → 16 chroma)
    let cb = avx2_pack_u16(cb_s, cb_s);
    let cr = avx2_pack_u16(cr_s, cr_s);

    _mm_storeu_si128(u_dst.as_mut_ptr().cast(), _mm256_castsi256_si128(cb));
    _mm_storeu_si128(v_dst.as_mut_ptr().cast(), _mm256_castsi256_si128(cr));
}

#[target_feature(enable = "avx2")]
unsafe fn avx2_rgba_to_yuv420_p16_impl<const ORIGIN_CHANNELS: u8>(
    transform: &CbCrForwardTransform<i32>,
    range: &YuvChromaRange,
    y_plane0: &mut [u8],
    y_plane1: &mut [u8],
    u_plane: &mut [u8],
    v_plane: &mut [u8],
    rgba0: &[u8],
    rgba1: &[u8],
    start_cx: usize,
    start_ux: usize,
    width: usize,
) -> ProcessedOffset {
    let source_channels: YuvSourceChannels = ORIGIN_CHANNELS.into();
    let channels = source_channels.get_channels_count();

    const PRECISION: i32 = 16;

    // Split yg into two i16-safe halves: yg_a + yg_b = yg
    // Required because yg (e.g. 33059 for BT.601) exceeds i16::MAX (32767)
    let yg_a = (transform.yg / 2) as i16;
    let yg_b = (transform.yg - transform.yg / 2) as i16;
    let yr = transform.yr as i16;

    // Y bias includes the range offset and rounding constant for the final shift
    let rounding_const_y = (1 << (PRECISION - 1)) - 1;
    let y_bias = _mm256_set1_epi32(range.bias_y as i32 * (1 << PRECISION) + rounding_const_y);

    // Broadcast coefficient pairs for PMADDWD:
    // v_yr_yga = [yr, yg_a] repeated — first half of split green
    // v_0_ygb  = [0,  yg_b] repeated — second half of split green
    let v_yr_yga = _mm256_set1_epi32(interleave_i16_pair(yr, yg_a));
    let v_0_ygb = _mm256_set1_epi32(interleave_i16_pair(0, yg_b));
    let v_yb = _mm256_set1_epi32(transform.yb);

    // UV precision is PRECISION+1 because the 2x2 pairwise average sums u8 pairs
    // into u16 range 0..510, effectively doubling the input scale
    let precision_uv = PRECISION + 1;
    let rounding_const_uv = (1 << (precision_uv - 1)) - 1;
    let uv_bias = _mm256_set1_epi32(range.bias_uv as i32 * (1 << precision_uv) + rounding_const_uv);
    let v_cb_r_g = _mm256_set1_epi32(transform._interleaved_cbr_cbg());
    let v_cb_b = _mm256_set1_epi32(transform.cb_b);
    let v_cr_r_g = _mm256_set1_epi32(transform._interleaved_crr_crg());
    let v_cr_b = _mm256_set1_epi32(transform.cr_b);

    let mut cx = start_cx;
    let mut uv_x = start_ux;

    while cx + 32 <= width {
        let px = cx * channels;

        encode_32_part_p16::<ORIGIN_CHANNELS>(
            rgba0.get_unchecked(px..),
            rgba1.get_unchecked(px..),
            y_plane0.get_unchecked_mut(cx..),
            y_plane1.get_unchecked_mut(cx..),
            u_plane.get_unchecked_mut(uv_x..),
            v_plane.get_unchecked_mut(uv_x..),
            y_bias,
            uv_bias,
            v_yr_yga,
            v_0_ygb,
            v_yb,
            v_cb_r_g,
            v_cb_b,
            v_cr_r_g,
            v_cr_b,
        );

        uv_x += 16;
        cx += 32;
    }

    tail_420!(
        32,
        cx,
        uv_x,
        width,
        channels,
        rgba0,
        rgba1,
        y_plane0,
        y_plane1,
        u_plane,
        v_plane,
        |sb0, sb1, yb0, yb1, ub, vb| {
            encode_32_part_p16::<ORIGIN_CHANNELS>(
                sb0, sb1, yb0, yb1, ub, vb, y_bias, uv_bias, v_yr_yga, v_0_ygb, v_yb, v_cb_r_g,
                v_cb_b, v_cr_r_g, v_cr_b,
            );
        }
    );

    ProcessedOffset { cx, ux: uv_x }
}

/// 420 handler: processes two rows at once with chroma subsampling.
pub(crate) fn avx2_rgba_to_yuv420_p16<const ORIGIN_CHANNELS: u8>(
    transform: &CbCrForwardTransform<i32>,
    range: &YuvChromaRange,
    y_plane0: &mut [u8],
    y_plane1: &mut [u8],
    u_plane: &mut [u8],
    v_plane: &mut [u8],
    rgba0: &[u8],
    rgba1: &[u8],
    start_cx: usize,
    start_ux: usize,
    width: usize,
) -> ProcessedOffset {
    unsafe {
        avx2_rgba_to_yuv420_p16_impl::<ORIGIN_CHANNELS>(
            transform, range, y_plane0, y_plane1, u_plane, v_plane, rgba0, rgba1, start_cx,
            start_ux, width,
        )
    }
}
