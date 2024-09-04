use image::io::Reader as ImageReader;
use image::{ColorType, EncodableLayout, GenericImageView};
use std::fs::File;
use std::io::Read;
use std::ops::Sub;
use std::time::Instant;

use yuvutils_rs::{rgb_to_sharp_yuv420, rgb_to_yuv420, rgb_to_yuv_nv12_p16, yuv420_to_rgb, yuv420_to_yuyv422, yuv_nv12_p10_to_rgb, yuv_nv12_to_rgb_p16, yuyv422_to_rgb, yuyv422_to_yuv420, SharpYuvGammaTransfer, YuvBytesPacking, YuvEndianness, YuvRange, YuvStandardMatrix};

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
    let img = ImageReader::open("./assets/test_augea.jpg")
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

    let y_stride = width as usize * std::mem::size_of::<u8>();
    let mut y_plane = vec![0u8; width as usize * height as usize];
    let mut u_plane = vec![0u8; width as usize * height as usize];
    let mut v_plane = vec![0u8; width as usize * height as usize];

    let rgba_stride = width as usize * 3;
    let mut rgba = vec![0u8; height as usize * rgba_stride];

    let start_time = Instant::now();

    let mut y_nv_plane = vec![0u16; width as usize * height as usize];
    let mut uv_nv_plane = vec![0u16; width as usize * (height as usize + 1) / 2];

    let mut bytes_16: Vec<u16> = src_bytes.iter().map(|&x| (x as u16) << 2).collect();

    let start_time = Instant::now();
    // rgb_to_yuv_nv12_p16(
    //     &mut y_nv_plane,
    //     width * 2,
    //     &mut uv_nv_plane,
    //     width * 2,
    //     &bytes_16,
    //     rgba_stride as u32 * 2,
    //     10,
    //     width,
    //     height,
    //     YuvRange::Full,
    //     YuvStandardMatrix::Bt709,
    //     YuvEndianness::BigEndian,
    //     YuvBytesPacking::LeastSignificantBytes,
    // );
    bytes_16.fill(0);
    //
    let end_time = Instant::now().sub(start_time);
    println!("rgb_to_yuv_nv12 time: {:?}", end_time);
    let start_time = Instant::now();
    // let vega = Vec::from(&bytes_16[bytes_16.len() / 2..(bytes_16.len() / 2 + 15)]);
    // println!("{:?}", vega);
    // yuv_nv12_to_rgb_p16(
    //     &y_nv_plane,
    //     width * 2,
    //     &uv_nv_plane,
    //     width * 2,
    //     &mut bytes_16,
    //     rgba_stride as u32 * 2,
    //     10,
    //     width,
    //     height,
    //     YuvRange::Full,
    //     YuvStandardMatrix::Bt709,
    //     YuvEndianness::BigEndian,
    //     YuvBytesPacking::LeastSignificantBytes,
    // );
    // yuv_nv12_p10_to_rgb(
    //     &y_nv_plane,
    //     width * 2,
    //     &uv_nv_plane,
    //     width * 2,
    //     &mut rgba,
    //     rgba_stride as u32,
    //     width,
    //     height,
    //     YuvRange::Full,
    //     YuvStandardMatrix::Bt709,
    //     YuvEndiannes::BigEndian,
    //     YuvBytesPacking::LeastSignificantBytes,
    // );
    rgba = bytes_16.iter().map(|&x| (x >> 2) as u8).collect();
    //
    let end_time = Instant::now().sub(start_time);
    println!("yuv_nv12_to_rgb time: {:?}", end_time);
    let start_time = Instant::now();
    rgb_to_sharp_yuv420(
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
        SharpYuvGammaTransfer::Srgb,
    );

    // let mut y_plane_16 = vec![0u16; width as usize * height as usize];
    // let mut u_plane_16 = vec![0u16; width as usize * height as usize];
    // let mut v_plane_16 = vec![0u16; width as usize * height as usize];
    //
    // let start_time = Instant::now();
    // rgb_to_yuv420_u16(
    //     &mut y_plane_16,
    //     y_stride as u32 * 2,
    //     &mut u_plane_16,
    //     y_stride as u32 * 2,
    //     &mut v_plane_16,
    //     y_stride as u32 * 2,
    //     &bytes_16,
    //     width * components * 2,
    //     10,
    //     width,
    //     height,
    //     YuvRange::TV,
    //     YuvStandardMatrix::Bt709,
    //     YuvEndiannes::BigEndian,
    //     YuvBytesPacking::LeastSignificantBytes,
    // );
    // let end_time = Instant::now().sub(start_time);
    // println!("rgb_to_yuv420_u16 time: {:?}", end_time);
    // let start_time = Instant::now();
    // yuv420_p10_to_rgb(
    //     &y_plane_16,
    //     y_stride as u32 * 2,
    //     &u_plane_16,
    //     y_stride as u32 * 2,
    //     &v_plane_16,
    //     y_stride as u32 * 2,
    //     &mut rgba,
    //     width * components,
    //     width,
    //     height,
    //     YuvRange::TV,
    //     YuvStandardMatrix::Bt709,
    //     YuvEndiannes::BigEndian,
    //     YuvBytesPacking::LeastSignificantBytes,
    // );

    let end_time = Instant::now().sub(start_time);
    println!("Forward time: {:?}", end_time);
    //
    // let full_size = if width % 2 == 0 {
    //     2 * width as usize * height as usize
    // } else {
    //     2 * (width as usize + 1) * height as usize
    // };
    //
    // println!("Full YUY2 {}", full_size);

    // let yuy2_stride = if width % 2 == 0 {
    //     2 * width as usize
    // } else {
    //     2 * (width as usize + 1)
    // };
    //
    // let mut yuy2_plane = vec![0u8; full_size];
    //
    // let start_time = Instant::now();
    //
    // yuv420_to_yuyv422(
    //     &y_plane,
    //     y_stride as u32,
    //     &u_plane,
    //     y_stride as u32,
    //     &v_plane,
    //     y_stride as u32,
    //     &mut yuy2_plane,
    //     yuy2_stride as u32,
    //     width,
    //     height,
    // );
    // rgba.fill(0);
    // let end_time = Instant::now().sub(start_time);
    // println!("yuv420_to_yuyv422 time: {:?}", end_time);
    // let start_time = Instant::now();
    // yuyv422_to_rgb(
    //     &yuy2_plane,
    //     yuy2_stride as u32,
    //     &mut rgba,
    //     rgba_stride as u32,
    //     width,
    //     height,
    //     YuvRange::TV,
    //     YuvStandardMatrix::Bt709,
    // );
    //
    // let end_time = Instant::now().sub(start_time);
    // println!("yuyv422_to_rgb time: {:?}", end_time);
    //
    // let start_time = Instant::now();
    //
    // yuyv422_to_yuv420(
    //     &mut y_plane,
    //     y_stride as u32,
    //     &mut u_plane,
    //     y_stride as u32,
    //     &mut v_plane,
    //     y_stride as u32,
    //     &yuy2_plane,
    //     yuy2_stride as u32,
    //     width,
    //     height,
    // );
    //
    // let end_time = Instant::now().sub(start_time);
    // println!("yuyv422_to_yuv444 time: {:?}", end_time);

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
    //
    // let mut gbr = vec![0u8; rgba.len()];
    //
    // let start_time = Instant::now();
    // rgb_to_gbr(
    //     &rgba,
    //     rgba_stride as u32,
    //     &mut gbr,
    //     rgba_stride as u32,
    //     width,
    //     height,
    // );
    //
    // let end_time = Instant::now().sub(start_time);
    // println!("rgb_to_gbr time: {:?}", end_time);
    //
    // let start_time = Instant::now();
    // gbr_to_rgb(
    //     &gbr,
    //     rgba_stride as u32,
    //     &mut rgba,
    //     rgba_stride as u32,
    //     width,
    //     height,
    // );
    // let end_time = Instant::now().sub(start_time);
    // println!("gbr_to_rgb time: {:?}", end_time);
    //
    // rgba = Vec::from(gbr);

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
