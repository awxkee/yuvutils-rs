# Rust utilities for YUV format handling and conversion.

[![crates.io](https://img.shields.io/crates/v/yuvutils-rs.svg)](https://crates.io/crates/yuvutils-rs)
![Build](https://github.com/awxkee/yuvutils-rs/actions/workflows/build_push.yml/badge.svg)

Fast and simple YUV approximation conversion in pure Rust. At most the same as libyuv does. Performance will be equal to libyuv or slightly higher on platforms where SIMD is implemented. Otherwise equal or slower. 

Mostly implemented AVX-512, AVX2, SSE, NEON, WASM

X86 targets with SSE and AVX uses runtime dispatch to detect available cpu features.

Supports:
- [x] YCbCr ( aka YUV )
- [x] YCgCo
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
| utils RGB->YUV 4:2:0   |  412.04µs  |    467.74µs    |  1.6235ms  |   722.87µs    |
| libyuv RGB->YUV 4:2:0  |  369.59µs  |     3.70ms     |  2.4843ms  |   972.52µs    |
| utils RGBA->YUV 4:2:0  |  486.29µs  |    447.70µs    |  2.2825ms  |   785.93µs    |
| libyuv RGBA->YUV 4:2:0 |  443.51µs  |     2.70ms     |  1.6656ms  |   707.33µs    |
| utils RGBA->YUV 4:2:2  |  547.57µs  |    586.14µs    |  3.0181ms  |   964.87µs    |
| libyuv RGBA->YUV 4:2:2 |  697.86µs  |    4.3003ms    |  2.4168ms  |   947.97µs    |
| utils RGBA->YUV 4:4:4  |  572.55µs  |    645.63µs    |  3.2485ms  |   974.29µs    |

### Decoding 8-bit

| Conversion             | time(NEON) | Time(AVX2 Win) | Time(AVX2) | Time(AVX-512) | 
|------------------------|:----------:|:--------------:|:----------:|:-------------:| 
| utils YUV NV12->RGBA   |  399.37µs  |    396.58µs    |  1.9711ms  |   736.21µs    | 
| utils YUV NV12->RGB    |  470.35µs  |    421.28µs    |  1.6601ms  |   654.37µs    | 
| libyuv YUV NV12->RGB   |  621.11µs  |    5.5533ms    |  1.8497ms  |   800.85µs    | 
| utils YUV 4:2:0->RGB   |  393.94µs  |    405.96µs    |  1.5455ms  |   614.19µs    | 
| libyuv YUV 4:2:0->RGB  |  747.18µs  |    5.8231ms    |  2.5895ms  |    1.16ms     | 
| utils YUV 4:2:0->RGBA  |  455.50µs  |    397.43µs    |  1.7229ms  |   733.88µs    | 
| libyuv YUV 4:2:0->RGBA |  744.49µs  |    557.44µs    |  1.8105ms  |   870.64µs    | 
| utils YUV 4:2:2->RGBA  |  512.39µs  |    440.47µs    |  2.2285ms  |   824.69µs    | 
| libyuv YUV 4:2:2->RGBA |  734.25µs  |    579.24µs    |  1.8965ms  |   851.24µs    | 
| utils YUV 4:4:4->RGBA  |  510.94µs  |    410.01µs    |  2.3391ms  |   819.17µs    | 
| libyuv YUV 4:4:4->RGBA |  596.55µs  |    525.90µs    |  1.9150ms  |   803.62µs    | 
| utils YUV 4:0:0->RGBA  |  220.67µs  |    260.69µs    |     -      |   600.43µs    | 
| libyuv YUV 4:0:0->RGBA |  522.71µs  |    1.8204ms    |     -      |   525.56µs    | 

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

|                            | time(NEON) | Time(AVX2 Win)  | Time(AVX2) | Time(AVX-512) |
|----------------------------|:----------:|:---------------:|:----------:|:-------------:|
| utils RGB10->YUV10 4:2:0   |  539.82µs  |    745.02µs     |   3.13ms   |    1.72ms     |
| libyuv RGB10->YUV10 4:2:0  |     x      |        x        |     x      |       x       |
| utils RGBA10->YUV10 4:2:0  |  674.12µs  |    807.89µs     |   3.66ms   |    1.64ms     |
| libyuv RGBA10->YUV10 4:2:0 |     x      |        x        |     x      |       x       |
| utils RGBA10->YUV10 4:2:2  |  838.82µs  |    1.1179ms     |   6.33ms   |    1.83ms     |
| libyuv RGBA10->YUV10 4:2:2 |     x      |        x        |     x      |       x       |
| utils RGBA10->YUV10 4:4:4  | 659.93 µs  |    1.1773ms     |   5.22ms   |    2.03ms     |

### Decoding 10-bit

|                           | time(NEON) | Time(AVX2 Win) | Time(AVX2) | Time(AVX-512) |
|---------------------------|:----------:|:--------------:|:----------:|:-------------:|
| utils YUV10 4:2:0->RGB10  |  640.34µs  |    674.92µs    |   2.91ms   |    1.23ms     |
| libyuv YUV10 4:2:0->RGB10 |     x      |       x        |     x      |       x       |
| utils YUV10 4:2:0->RGBA10 |  814.86µs  |    687.52µs    |   3.92ms   |    1.59ms     |
| utils YUV10 4:2:0->RGBA8  |  812.53µs  |    692.53µs    |   2.27ms   |   900.10µs    |
| libyuv YUV10 4:2:0->RGBA8 |  1.7037ms  |    6.8641ms    |   2.10ms   |   966.94µs    |
| utils YUV10 4:2:2->RGBA10 |  859.39µs  |    787.41µs    |   3.71ms   |    1.61ms     |
| utils YUV10 4:2:2->RGBA8  |  878.54µs  |    710.79µs    |   2.28ms   |   972.19µs    |
| libyuv YUV10 4:2:2->RGBA8 |  1.7056ms  |    6.8374ms    |   1.94ms   |   991.23µs    |
| utils YUV10 4:4:4->RGBA10 |  931.28µs  |    908.32µs    |   3.82ms   |   1.8639ms    |

This project is licensed under either of

- BSD-3-Clause License (see [LICENSE](LICENSE.md))
- Apache License, Version 2.0 (see [LICENSE](LICENSE-APACHE.md))

at your option.
