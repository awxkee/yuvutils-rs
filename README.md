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
| utils RGB->YUV 4:2:0   |  412.04µs  |     5.54ms     |  16.28ms   |    8.67ms     |
| libyuv RGB->YUV 4:2:0  |  369.59µs  |    34.30ms     |  17.64ms   |    12.78ms    |
| utils RGBA->YUV 4:2:0  |  486.29µs  |     5.78ms     |  12.63ms   |    10.17ms    |
| libyuv RGBA->YUV 4:2:0 |  443.51µs  |    25.29ms     |  11.27ms   |    10.73ms    |
| utils RGBA->YUV 4:2:2  |  547.57µs  |     7.35ms     |  24.02ms   |    18.14ms    |
| libyuv RGBA->YUV 4:2:2 |  697.86µs  |    37.65ms     |  19.43ms   |    18.07ms    |
| utils RGBA->YUV 4:4:4  |  572.55µs  |     8.97ms     |  29.18ms   |    21.92ms    |

### Decoding 8-bit

| Conversion             |    time(NEON)    | Time(AVX2 Win) | Time(AVX2) | Time(AVX-512) | 
|------------------------|:----------------:|:--------------:|:----------:|:-------------:| 
| utils YUV NV12->RGBA   |     399.37µs     |     5.15ms     |  22.59ms   |    18.55ms    | 
| utils YUV NV12->RGB    |     470.35µs     |     6.71ms     |  17.56ms   |    13.64ms    | 
| libyuv YUV NV12->RGB   |     621.11µs     |    50.16ms     |  22.27ms   |    18.55ms    | 
| utils YUV 4:2:0->RGB   |     393.94µs     |     5.15ms     |  17.69ms   |    13.49ms    | 
| libyuv YUV 4:2:0->RGB  |     747.18µs     |    48.52ms     |  23.91ms   |    20.07ms    | 
| utils YUV 4:2:0->RGBA  |     455.50µs     |     6.70ms     |  20.81ms   |    18.84ms    | 
| libyuv YUV 4:2:0->RGBA |     744.49µs     |     7.20ms     |  24.32ms   |    18.50ms    | 
| utils YUV 4:2:2->RGBA  |     512.39µs     |     7.61ms     |  24.44ms   |    22.05ms    | 
| libyuv YUV 4:2:2->RGBA |     734.25µs     |     7.48ms     |  23.72ms   |    18.71ms    | 
| utils YUV 4:4:4->RGBA  |     510.94µs     |     7.65ms     |  27.58ms   |    22.85ms    | 
| libyuv YUV 4:4:4->RGBA |     596.55µs     |     7.55ms     |  34.60ms   |    21.47ms    | 

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
| utils RGB10->YUV10 4:2:0   |  539.82µs  |     9.13ms     |  33.88ms   |    29.77ms    |
| libyuv RGB10->YUV10 4:2:0  |     x      |       x        |     x      |       x       |
| utils RGBA10->YUV10 4:2:0  |  674.12µs  |    10.82ms     |  32.69ms   |    31.22ms    |
| libyuv RGBA10->YUV10 4:2:0 |     x      |       x        |     x      |       x       |
| utils RGBA10->YUV10 4:2:2  |  838.82µs  |    14.74ms     |  50.26ms   |    44.49ms    |
| libyuv RGBA10->YUV10 4:2:2 |     x      |       x        |     x      |       x       |
| utils RGBA10->YUV10 4:4:4  | 659.93 µs  |    16.49ms     |  70.11ms   |    58.08ms    |

### Decoding 10-bit

|                           | time(NEON) | Time(AVX2 Win) | Time(AVX2) | Time(AVX-512) |
|---------------------------|:----------:|:--------------:|:----------:|:-------------:|
| utils YUV10 4:2:0->RGB10  |  640.34µs  |    11.06ms     |  45.58ms   |    40.61ms    |
| libyuv YUV10 4:2:0->RGB10 |     -      |       -        |     -      |       -       |
| utils YUV10 4:2:0->RGBA10 |  814.86µs  |    14.85ms     |  65.95ms   |    60.70ms    |
| utils YUV10 4:2:0->RGBA8  |  812.53µs  |     8.77ms     |  31.15ms   |    23.50ms    |
| libyuv YUV10 4:2:0->RGBA8 |  1.7037ms  |    62.01ms     |  24.59ms   |    22.90ms    |
| utils YUV10 4:2:2->RGBA10 |  859.39µs  |    15.92ms     |  59.44ms   |    60.95ms    |
| utils YUV10 4:2:2->RGBA8  |  878.54µs  |     8.76ms     |  29.15ms   |    25.45ms    |
| libyuv YUV10 4:2:2->RGBA8 |  1.7056ms  |    61.28ms     |  29.96ms   |    23.63ms    |
| utils YUV10 4:4:4->RGBA10 |  931.28µs  |    16.09ms     |  70.84ms   |    61.91ms    |

This project is licensed under either of

- BSD-3-Clause License (see [LICENSE](LICENSE.md))
- Apache License, Version 2.0 (see [LICENSE](LICENSE-APACHE.md))

at your option.
