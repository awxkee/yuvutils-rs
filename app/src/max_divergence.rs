/*
 * Copyright (c) Radzivon Bartoshyk, 1/2025. All rights reserved.
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
use core::f16;
use rand::Rng;
use yuv::{
    i410_to_rgb_f16, rgb10_to_i410, YuvChromaSubsampling, YuvConversionMode, YuvPlanarImageMut,
    YuvRange, YuvStandardMatrix,
};

fn matrix(
    mode: YuvConversionMode,
    or: u16,
    og: u16,
    ob: u16,
    range: YuvRange,
    matrix: YuvStandardMatrix,
) -> (u32, u32, u32) {
    let image_width = 32usize;
    let image_height = 32usize;

    let random_point_x = rand::rng().random_range(0..image_width);
    let random_point_y = rand::rng().random_range(0..image_height);

    const CHANNELS: usize = 3;

    let pixel_points = [
        [0, 0],
        [image_width - 1, image_height - 1],
        [image_width - 1, 0],
        [0, image_height - 1],
        [(image_width - 1) / 2, (image_height - 1) / 2],
        [image_width / 5, image_height / 5],
        [0, image_height / 5],
        [image_width / 5, 0],
        [image_width / 5 * 3, image_height / 5],
        [image_width / 5 * 3, image_height / 5 * 3],
        [image_width / 5, image_height / 5 * 3],
        [random_point_x, random_point_y],
    ];

    let mut source_rgb = vec![0u16; image_width * image_height * CHANNELS];

    let or = or as u16;
    let og = og as u16;
    let ob = ob as u16;

    for point in &pixel_points {
        source_rgb[point[0] * CHANNELS + point[1] * image_width * CHANNELS] = or;
        source_rgb[point[0] * CHANNELS + point[1] * image_width * CHANNELS + 1] = og;
        source_rgb[point[0] * CHANNELS + point[1] * image_width * CHANNELS + 2] = ob;

        let nx = (point[0] + 1).min(image_width - 1);
        let ny = point[1].min(image_height - 1);

        source_rgb[nx * CHANNELS + ny * image_width * CHANNELS] = or;
        source_rgb[nx * CHANNELS + ny * image_width * CHANNELS + 1] = og;
        source_rgb[nx * CHANNELS + ny * image_width * CHANNELS + 2] = ob;

        let nx = (point[0] + 1).min(image_width - 1);
        let ny = (point[1] + 1).min(image_height - 1);

        source_rgb[nx * CHANNELS + ny * image_width * CHANNELS] = or;
        source_rgb[nx * CHANNELS + ny * image_width * CHANNELS + 1] = og;
        source_rgb[nx * CHANNELS + ny * image_width * CHANNELS + 2] = ob;

        let nx = point[0].min(image_width - 1);
        let ny = (point[1] + 1).min(image_height - 1);

        source_rgb[nx * CHANNELS + ny * image_width * CHANNELS] = or;
        source_rgb[nx * CHANNELS + ny * image_width * CHANNELS + 1] = og;
        source_rgb[nx * CHANNELS + ny * image_width * CHANNELS + 2] = ob;

        let nx = point[0].saturating_sub(1).min(image_width - 1);
        let ny = point[1].saturating_sub(1).min(image_height - 1);

        source_rgb[nx * CHANNELS + ny * image_width * CHANNELS] = or;
        source_rgb[nx * CHANNELS + ny * image_width * CHANNELS + 1] = og;
        source_rgb[nx * CHANNELS + ny * image_width * CHANNELS + 2] = ob;

        let nx = point[0].min(image_width - 1);
        let ny = point[1].saturating_sub(1).min(image_height - 1);

        source_rgb[nx * CHANNELS + ny * image_width * CHANNELS] = or;
        source_rgb[nx * CHANNELS + ny * image_width * CHANNELS + 1] = og;
        source_rgb[nx * CHANNELS + ny * image_width * CHANNELS + 2] = ob;

        let nx = point[0].saturating_sub(1).min(image_width - 1);
        let ny = point[1].min(image_height - 1);

        source_rgb[nx * CHANNELS + ny * image_width * CHANNELS] = or;
        source_rgb[nx * CHANNELS + ny * image_width * CHANNELS + 1] = og;
        source_rgb[nx * CHANNELS + ny * image_width * CHANNELS + 2] = ob;
    }

    let mut planar_image = YuvPlanarImageMut::<u16>::alloc(
        image_width as u32,
        image_height as u32,
        YuvChromaSubsampling::Yuv444,
    );

    rgb10_to_i410(
        &mut planar_image,
        &source_rgb,
        image_width as u32 * CHANNELS as u32,
        range,
        matrix,
    )
    .unwrap();

    let mut dest_rgb: Vec<f16> = vec![0.; image_width * image_height * CHANNELS];

    let fixed_planar = planar_image.to_fixed();

    i410_to_rgb_f16(
        &fixed_planar,
        &mut dest_rgb,
        image_width as u32 * CHANNELS as u32,
        range,
        matrix,
    )
    .unwrap();

    let mut m_r = u32::MIN;
    let mut m_g = u32::MIN;
    let mut m_b = u32::MIN;

    for point in &pixel_points {
        let x = point[0];
        let y = point[1];
        let px = x * CHANNELS + y * image_width * CHANNELS;

        let r = (dest_rgb[px] as f32 * 1023.).round();
        let g = (dest_rgb[px + 1] as f32 * 1023.).round();
        let b = (dest_rgb[px + 2] as f32 * 1023.).round();

        let diff_r = (r as i32 - or as i32).abs() as u32;
        let diff_g = (g as i32 - og as i32).abs() as u32;
        let diff_b = (b as i32 - ob as i32).abs() as u32;

        m_r = diff_r.max(m_r);
        m_g = diff_g.max(m_g);
        m_b = diff_b.max(m_b);
    }
    (m_r, m_g, m_b)
}

pub(crate) fn search_for_max_divergences(
    mode: YuvConversionMode,
    range: YuvRange,
    yuv_matrix: YuvStandardMatrix,
) -> (u32, u32, u32) {
    let mut m_r = u32::MIN;
    let mut m_g = u32::MIN;
    let mut m_b = u32::MIN;

    for r in 0..255 {
        for g in 0..255 {
            for b in 0..255 {
                let (n_r, n_g, n_b) = matrix(mode, r, g, b, range, yuv_matrix);
                m_r = n_r.max(m_r);
                m_g = n_g.max(m_g);
                m_b = n_b.max(m_b);
            }
        }
    }
    (m_r, m_g, m_b)
}

pub(crate) fn check_div(mode: YuvConversionMode) {
    let max_divergence = search_for_max_divergences(mode, YuvRange::Full, YuvStandardMatrix::Bt601);
    println!("Max Divergence {} Full Bt.601 {:?}", mode, max_divergence);
    let max_divergence =
        search_for_max_divergences(mode, YuvRange::Limited, YuvStandardMatrix::Bt601);
    println!(
        "Max Divergence {} Limited Bt.601 {:?}",
        mode, max_divergence
    );
    let max_divergence = search_for_max_divergences(mode, YuvRange::Full, YuvStandardMatrix::Bt709);
    println!("Max Divergence {} Full Bt.709 {:?}", mode, max_divergence);
    let max_divergence =
        search_for_max_divergences(mode, YuvRange::Limited, YuvStandardMatrix::Bt709);
    println!(
        "Max Divergence {} Limited Bt.709 {:?}",
        mode, max_divergence
    );
    let max_divergence =
        search_for_max_divergences(mode, YuvRange::Full, YuvStandardMatrix::Bt2020);
    println!("Max Divergence {} Full Bt.2020 {:?}", mode, max_divergence);
    let max_divergence =
        search_for_max_divergences(mode, YuvRange::Limited, YuvStandardMatrix::Bt2020);
    println!(
        "Max Divergence {} Limited Bt.2020 {:?}",
        mode, max_divergence
    );
}
