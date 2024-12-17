extern crate wee_alloc;
use image::{DynamicImage, EncodableLayout, ImageBuffer, ImageReader};
use js_sys::Uint8Array;
use std::io::Cursor;
use std::panic;
use wasm_bindgen::prelude::wasm_bindgen;
use yuvutils_rs::{rgb_to_yuv420, rgb_to_yuv422, rgb_to_yuv_nv12, yuv420_to_rgb, yuv422_to_rgb, yuv_nv12_to_rgb, YuvBiPlanarImageMut, YuvChromaSubsampling, YuvPlanarImage, YuvPlanarImageMut, YuvRange, YuvStandardMatrix};

#[wasm_bindgen]
extern "C" {
    fn alert(s: &str);
}

// Use `wee_alloc` as the global allocator.
#[global_allocator]
static ALLOC: wee_alloc::WeeAlloc = wee_alloc::WeeAlloc::INIT;

pub fn set_panic_hook() {
    panic::set_hook(Box::new(console_error_panic_hook::hook));
}

#[wasm_bindgen]
pub fn process(image: Uint8Array) -> Uint8Array {
    panic::set_hook(Box::new(console_error_panic_hook::hook));
    let arr = image.to_vec();
    let cursor = Cursor::new(arr);
    let img = ImageReader::new(cursor)
        .with_guessed_format()
        .unwrap()
        .decode()
        .unwrap();
    let transient = img.to_rgb8();
    let mut bytes = Vec::from(transient.as_bytes());

    let mut planar_image =
        YuvBiPlanarImageMut::alloc(img.width(), img.height(), YuvChromaSubsampling::Yuv420);
    rgb_to_yuv_nv12(
        &mut planar_image,
        &transient,
        img.width() * 3,
        YuvRange::Limited,
        YuvStandardMatrix::Bt709,
    )
    .unwrap();
    bytes.fill(0);
    let fixed_gray = planar_image.to_fixed();
    yuv_nv12_to_rgb(
        &fixed_gray,
        &mut bytes,
        img.width() * 3,
        YuvRange::Limited,
        YuvStandardMatrix::Bt709,
    )
    .unwrap();

    let img = ImageBuffer::from_raw(img.width(), img.height(), bytes)
        .map(DynamicImage::ImageRgb8)
        .expect("Failed to create image from raw data");

    let mut bytes: Vec<u8> = Vec::new();

    img.write_to(&mut Cursor::new(&mut bytes), image::ImageFormat::Jpeg)
        .expect("Successfully write");

    let fixed_slice: &[u8] = &bytes;
    Uint8Array::from(fixed_slice)
}
