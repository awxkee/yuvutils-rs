use image::io::Reader as ImageReader;
use image::{ColorType, EncodableLayout, GenericImageView};
use std::fs::File;
use std::io::Read;
use std::ops::Sub;
use std::time::Instant;

use yuvutils_rs::{
    rgb_to_ycgcoro444, rgb_to_yuv420, rgb_to_yuv422, rgb_to_yuv444, ycgcoro444_to_rgb,
    yuv420_to_rgb, yuv420_to_yuyv422, yuv422_to_rgb, yuv422_to_yuyv422, yuv444_to_rgb,
    yuv444_to_yuyv422, yuyv422_to_yuv420, yuyv422_to_yuv422, yuyv422_to_yuv444, YuvRange,
    YuvStandardMatrix,
};

fn read_file_bytes(file_path: &str) -> Result<Vec<u8>, String> {
    // Open the file
    let mut file = File::open(file_path).unwrap();

    // Create a buffer to hold the file contents
    let mut buffer = Vec::new();

    // Read the file contents into the buffer
    file.read_to_end(&mut buffer).unwrap();

    // Return the buffer
    Ok(buffer)
}

fn main() {
    let img = ImageReader::open("assets/test_image_3.jpg")
        .unwrap()
        .decode()
        .unwrap();
    let dimensions = img.dimensions();

    let width = dimensions.0;
    let height = dimensions.1;

    let yuyu2 = read_file_bytes("assets/output_video.yuv").unwrap();
    let shift = (width - 1) as usize * 4;
    let drain = &yuyu2[shift..];
    println!("{}", drain[0]);
    println!("{}", drain[1]);
    println!("{}", drain[2]);
    println!("{}", drain[3]);

    let src_bytes = img.as_bytes();
    let components = match img.color() {
        ColorType::Rgb8 => 3,
        ColorType::Rgba8 => 4,
        _ => {
            panic!("Not accepted")
        }
    };

    let y_stride = width as usize * std::mem::size_of::<u8>();
    let mut y_plane = vec![0u8; width as usize * height as usize];
    let mut u_plane = vec![0u8; width as usize * height as usize];
    let mut v_plane = vec![0u8; width as usize * height as usize];

    let rgba_stride = width as usize * 3;
    let mut rgba = vec![0u8; height as usize * rgba_stride];

    let start_time = Instant::now();

    rgb_to_yuv420(
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
        YuvStandardMatrix::Bt709,
    );

    let end_time = Instant::now().sub(start_time);
    println!("Forward time: {:?}", end_time);

    let full_size = if width % 2 == 0 {
        2 * width as usize * height as usize
    } else {
        2 * (width as usize + 1) * height as usize
    };

    println!("Full YUY2 {}", full_size);

    let yuy2_stride = if width % 2 == 0 {
        2 * width as usize
    } else {
        2 * (width as usize + 1)
    };

    let mut yuy2_plane = vec![0u8; full_size];

    let start_time = Instant::now();

    yuv420_to_yuyv422(
        &y_plane,
        y_stride as u32,
        &u_plane,
        y_stride as u32,
        &v_plane,
        y_stride as u32,
        &mut yuy2_plane,
        yuy2_stride as u32,
        width,
        height,
    );

    let end_time = Instant::now().sub(start_time);
    println!("yuv444_to_yuyv422 time: {:?}", end_time);

    let start_time = Instant::now();

    yuyv422_to_yuv420(
        &mut y_plane,
        y_stride as u32,
        &mut u_plane,
        y_stride as u32,
        &mut v_plane,
        y_stride as u32,
        &yuy2_plane,
        yuy2_stride as u32,
        width,
        height,
    );

    let end_time = Instant::now().sub(start_time);
    println!("yuyv422_to_yuv444 time: {:?}", end_time);

    let start_time = Instant::now();
    yuv420_to_rgb(
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
        YuvStandardMatrix::Bt709,
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
