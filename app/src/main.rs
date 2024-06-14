use std::ops::Sub;
use std::time::Instant;
use image::{ColorType, EncodableLayout, GenericImageView};
use image::io::Reader as ImageReader;

use yuvutils_rs::{rgb_to_ycgcoro444, ycgcoro444_to_rgb, YuvRange};

fn main() {
    let img = ImageReader::open("assets/test_image_1.jpg")
        .unwrap()
        .decode()
        .unwrap();
    let dimensions = img.dimensions();

    let width = dimensions.0;
    let height = dimensions.1;

    let src_bytes = img.as_bytes();
    let components = match img.color() {
        ColorType::Rgb8 => 3,
        ColorType::Rgba8 => 4,
        _ => {
            panic!("Not accepted")
        }
    };

    let y_stride = width as usize * std::mem::size_of::<u16>();
    let mut y_plane = vec![0u16; width as usize * height as usize];
    let mut u_plane = vec![0u16; width as usize * height as usize];
    let mut v_plane = vec![0u16; width as usize * height as usize];

    let rgba_stride = width as usize * 3;
    let mut rgba = vec![0u8; height as usize * rgba_stride];

    let start_time = Instant::now();

    rgb_to_ycgcoro444(
        &mut y_plane,
        y_stride as u32,
        &mut u_plane,
        y_stride as u32,
        &mut v_plane,
        y_stride as u32,
        src_bytes,
        width * components,
        width,
        height,
        YuvRange::TV,
    );

    let end_time = Instant::now().sub(start_time);
    println!("Forward time: {:?}", end_time);

    let start_time = Instant::now();
    ycgcoro444_to_rgb(
        &y_plane,
        y_stride as u32,
        &u_plane,
        y_stride as u32,
        &v_plane,
        y_stride as u32,
        &mut rgba,
        rgba_stride as u32,
        width,
        height,
        YuvRange::TV,
    );

    let end_time = Instant::now().sub(start_time);
    println!("Backward time: {:?}", end_time);

    image::save_buffer(
        "converted.jpg",
        rgba.as_bytes(),
        dimensions.0,
        dimensions.1,
        if components == 3 {
            image::ExtendedColorType::Rgb8
        } else {
            image::ExtendedColorType::Rgba8
        },
    )
    .unwrap();
}
