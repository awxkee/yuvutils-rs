# Rust utilities for YUV format handling and conversion.

[![crates.io](https://img.shields.io/crates/v/yuvutils-rs.svg)](https://crates.io/crates/yuvutils-rs)
[![Build Status](https://github.com/awxkee/yuvutils-rs/actions/workflows/Build/badge.svg)](https://github.com/awxkee/yuvutils-rs/actions)

Fast and simple YUV approximation conversion in pure Rust. At most the same as libyuv does. Performance will be equal to libyuv or slightly higher on platforms where SIMD is implemented. Otherwise equal or slower. 

Mostly implemented AVX-512, AVX2, SSE, NEON, WASM

X86 targets with SSE and AVX uses runtime dispatch to detect available cpu features.

Supports:
- [x] YCbCr ( aka YUV )
- [x] YCgCo
- [x] YCgCo-R
- [x] YUY2
- [x] Identity ( GBR )
- [x] Sharp YUV

All the methods support RGB, BGR, BGRA and RGBA

# SIMD

rustc `avx2`, `avx512f`, `avx512bw`, `neon`, `sse4.1` features should be set when you expect than code will run on supported device.

For AVX-512 target feature `avx512bw` is required along with feature `nightly_avx512` and `nightly` rust channel compiler.

Wasm `simd128` should be enabled for implemented SIMD wasm paths support

# Rayon 

Some paths have multi-threading support, consider this feature if you're working on platform with multi-threading.

### Adding to project

```bash
cargo add yuvutils-rs
```

### RGB to YCbCr

```rust
let mut planar_image =
    YuvPlanarImageMut::<u8>::alloc(width as u32, height as u32, YuvChromaSubsampling::Yuv420);
rgb_to_yuv422(
    &mut planar_image,
    &src_bytes,
    rgba_stride as u32,
    YuvRange::Limited,
    YuvStandardMatrix::Bt601,
)
.unwrap();
```

### YCbCr to RGB

```rust
yuv420_to_rgb(
    &yuv_planar_image,
    &mut rgba,
    rgba_stride as u32,
    YuvRange::Limited,
    YuvStandardMatrix::Bt601,
)
.unwrap();
```

## Benchmarks

Tests performed on the image 1997x1331

YUV 8 bit-depth conversion

`aarch64` tested on Mac Pro M3.

AVX2 tests performed on `Digital Ocean Shared Premium Intel 2 vCPU` droplet.
AVX2 Win test performed on Windows 11 Intel Core i9-14900HX.

```bash
cargo bench --bench yuv8 --manifest-path ./app/Cargo.toml
```

AVX-512 tests performed on `AWS c5.large` with Ubuntu 24.04 instance with command

```bash
cargo +nightly bench --bench yuv8 --manifest-path ./app/Cargo.toml --features nightly_avx512
```

### Encoding 8-bit

| Conversion             | time(NEON) | Time(AVX2 Win) | Time(AVX2) | Time(AVX-512) |
|------------------------|:----------:|:--------------:|:----------:|:-------------:|
| utils RGB->YUV 4:2:0   |  412.04µs  |     5.54ms     |  1.6235ms  |   722.87µs    |
| libyuv RGB->YUV 4:2:0  |  369.59µs  |    34.30ms     |  2.4843ms  |   972.52µs    |
| utils RGBA->YUV 4:2:0  |  486.29µs  |     5.78ms     |  2.2825ms  |   803.93µs    |
| libyuv RGBA->YUV 4:2:0 |  443.51µs  |    25.29ms     |  1.6656ms  |   707.33µs    |
| utils RGBA->YUV 4:2:2  |  547.57µs  |     7.35ms     |  3.0181ms  |   980.87µs    |
| libyuv RGBA->YUV 4:2:2 |  697.86µs  |    37.65ms     |  2.4168ms  |   947.97µs    |
| utils RGBA->YUV 4:4:4  |  572.55µs  |     8.97ms     |  3.2485ms  |   1.0362ms    |

### Decoding 8-bit

| Conversion             | time(NEON) | Time(AVX2 Win) | Time(AVX2) | Time(AVX-512) | 
|------------------------|:----------:|:--------------:|:----------:|:-------------:| 
| utils YUV NV12->RGBA   |  399.37µs  |     5.15ms     |  1.9711ms  |   763.19µs    | 
| utils YUV NV12->RGB    |  470.35µs  |     6.71ms     |  1.6601ms  |   663.13µs    | 
| libyuv YUV NV12->RGB   |  621.11µs  |    50.16ms     |  1.8497ms  |   805.78µs    | 
| utils YUV 4:2:0->RGB   |  393.94µs  |     5.15ms     |  1.5455ms  |   618.66µs    | 
| libyuv YUV 4:2:0->RGB  |  747.18µs  |    48.52ms     |  2.5895ms  |   1.1899ms    | 
| utils YUV 4:2:0->RGBA  |  455.50µs  |     6.70ms     |  1.7229ms  |   766.47µs    | 
| libyuv YUV 4:2:0->RGBA |  744.49µs  |     7.20ms     |  1.8105ms  |   870.64µs    | 
| utils YUV 4:2:2->RGBA  |  512.39µs  |     7.61ms     |  2.2285ms  |   856.36µs    | 
| libyuv YUV 4:2:2->RGBA |  734.25µs  |     7.48ms     |  1.8965ms  |   888.94µs    | 
| utils YUV 4:4:4->RGBA  |  510.94µs  |     7.65ms     |  2.3391ms  |   914.94µs    | 
| libyuv YUV 4:4:4->RGBA |  596.55µs  |     7.55ms     |  1.9150ms  |   835.67µs    | 

YUV 16 bit-depth conversion

```bash
cargo bench --bench yuv16 --manifest-path ./app/Cargo.toml
```

AVX-512 tests performed on `AWS c5.large` with Ubuntu 24.04 instance with command

```bash
cargo +nightly bench --bench yuv16 --manifest-path ./app/Cargo.toml --features nightly_avx512
```

### Encoding 10-bit

10-bit encoding is not implemented in `libyuv`

|                            | time(NEON) | Time(AVX2 Win) | Time(AVX2) | Time(AVX-512) |
|----------------------------|:----------:|:--------------:|:----------:|:-------------:|
| utils RGB10->YUV10 4:2:0   |  539.82µs  |     9.13ms     |  3.1382ms  |   1.8046ms    |
| libyuv RGB10->YUV10 4:2:0  |     x      |       x        |     x      |       x       |
| utils RGBA10->YUV10 4:2:0  |  674.12µs  |    10.82ms     |  3.6665ms  |   1.8662ms    |
| libyuv RGBA10->YUV10 4:2:0 |     x      |       x        |     x      |       x       |
| utils RGBA10->YUV10 4:2:2  |  838.82µs  |    14.74ms     |  6.3362ms  |   2.5137ms    |
| libyuv RGBA10->YUV10 4:2:2 |     x      |       x        |     x      |       x       |
| utils RGBA10->YUV10 4:4:4  | 659.93 µs  |    16.49ms     |  5.2270ms  |   2.7585ms    |

### Decoding 10-bit

|                           | time(NEON) | Time(AVX2 Win) | Time(AVX2) | Time(AVX-512) |
|---------------------------|:----------:|:--------------:|:----------:|:-------------:|
| utils YUV10 4:2:0->RGB10  |  640.34µs  |    11.06ms     |  2.9178ms  |   1.6384ms    |
| libyuv YUV10 4:2:0->RGB10 |     x      |       x        |     x      |       x       |
| utils YUV10 4:2:0->RGBA10 |  814.86µs  |    14.85ms     |  3.9288ms  |   1.8072ms    |
| utils YUV10 4:2:0->RGBA8  |  812.53µs  |     8.77ms     |  2.2707ms  |   1.3771ms    |
| libyuv YUV10 4:2:0->RGBA8 |  1.7037ms  |    62.01ms     |  2.1085ms  |   986.53µs    |
| utils YUV10 4:2:2->RGBA10 |  859.39µs  |    15.92ms     |  3.7146ms  |   1.8462ms    |
| utils YUV10 4:2:2->RGBA8  |  878.54µs  |     8.76ms     |  2.2870ms  |   1.3623ms    |
| libyuv YUV10 4:2:2->RGBA8 |  1.7056ms  |    61.28ms     |  1.9400ms  |   1.0410ms    |
| utils YUV10 4:4:4->RGBA10 |  931.28µs  |    16.09ms     |  3.8239ms  |   2.3918ms    |

This project is licensed under either of

- BSD-3-Clause License (see [LICENSE](LICENSE.md))
- Apache License, Version 2.0 (see [LICENSE](LICENSE-APACHE.md))

at your option.
