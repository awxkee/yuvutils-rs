# Rust utilities for YUV format handling and conversion.

[![crates.io](https://img.shields.io/crates/v/yuvutils-rs.svg)](https://crates.io/crates/yuvutils-rs)
![Build](https://github.com/awxkee/yuvutils-rs/actions/workflows/build_push.yml/badge.svg)

Fast and simple YUV approximation conversion in pure Rust. At most the same as libyuv does.
Performance will be equal to libyuv or slightly higher on platforms where SIMD is implemented. Otherwise, equal or slower.
Used precision is slightly higher than used in libyuv, however for decoding 4:2:0 is almost always faster than libyuv,
decoding 4:2:2, 4:4:4 almost the same, however encoding except 4:2:0 often is slower with better results.

Mostly implemented AVX-512, AVX2, SSE, NEON, WASM. Waiting for SVE to be available in `nightly`.

x86 targets with SSE and AVX uses runtime dispatch to detect available cpu features.

Supports:
- [x] YCbCr ( aka YUV )
- [x] YCgCo
- [x] YUY2
- [x] Identity ( GBR )
- [x] Sharp YUV

All the methods support RGB, BGR, BGRA and RGBA

# SIMD

Runtime dispatch is used for use if available `sse4.1`, `avx2`, `avx512bw`, `avx512vbmi`, `rdm` for NEON. 

For the `sse4.1`, `avx2`, `rdm` there is no action needed, however for AVX-512 activated feature `nightly_avx512` along `nightly` rust channel is required.

Always consider to compile with `nightly_avx512` when compiling for x86, it adds noticeable gain on supported cpus. 

Wasm `simd128` as target feature should be enabled for implemented SIMD wasm paths support,

# Rayon 

Some paths have multi-threading support, consider this feature if you're working on platform where using of multi-threading is reasonable.

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

AVX2 tests performed on `Digital Ocean Shared Premium Intel 2 vCPU` with Ubuntu 24.04 droplet.
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
| utils RGB->YUV 4:2:0   |  360.53µs  |    467.74µs    |  1.6235ms  |   722.87µs    |
| libyuv RGB->YUV 4:2:0  |  306.80µs  |     3.70ms     |  2.4843ms  |   972.52µs    |
| utils RGBA->YUV 4:2:0  |  422.36µs  |    447.70µs    |  2.2825ms  |   785.93µs    |
| libyuv RGBA->YUV 4:2:0 |  429.36µs  |     2.70ms     |  1.6656ms  |   707.33µs    |
| utils RGBA->YUV 4:2:2  |  590.36µs  |    567.99µs    |  3.0181ms  |   964.87µs    |
| libyuv RGBA->YUV 4:2:2 |  651.84µs  |    4.3003ms    |  2.4168ms  |   947.97µs    |
| utils RGBA->YUV 4:4:4  |  674.61µs  |    597.30µs    |  3.2485ms  |   974.29µs    |

### Decoding 8-bit

| Conversion             | time(NEON) | Time(AVX2 Win) | Time(AVX2) | Time(AVX-512) | 
|------------------------|:----------:|:--------------:|:----------:|:-------------:| 
| utils YUV NV12->RGBA   |  399.37µs  |    377.72µs    |   1.97ms   |   736.21µs    | 
| utils YUV NV12->RGB    |  470.35µs  |    387.70µs    |   1.66ms   |   654.37µs    | 
| libyuv YUV NV12->RGB   |  621.11µs  |    5.5533ms    |   1.84ms   |   800.85µs    | 
| utils YUV 4:2:0->RGB   |  346.13µs  |    375.34µs    |   1.54ms   |   614.19µs    | 
| libyuv YUV 4:2:0->RGB  |  747.18µs  |    5.8231ms    |   2.58ms   |    1.16ms     | 
| utils YUV 4:2:0->RGBA  |  396.58µs  |    380.13µs    |   1.72ms   |   733.88µs    | 
| libyuv YUV 4:2:0->RGBA |  710.88µs  |    557.44µs    |   1.81ms   |   870.64µs    | 
| utils YUV 4:2:2->RGBA  |  471.54µs  |    410.71µs    |   2.22ms   |   824.69µs    | 
| libyuv YUV 4:2:2->RGBA |  734.25µs  |    579.24µs    |   1.89ms   |   851.24µs    | 
| utils YUV 4:4:4->RGBA  |  458.71µs  |    366.85µs    |   2.33ms   |   819.17µs    | 
| libyuv YUV 4:4:4->RGBA |  596.55µs  |    525.90µs    |   1.91ms   |   803.62µs    | 
| utils YUV 4:0:0->RGBA  |  220.67µs  |    254.83µs    |   1.05ms   |   600.43µs    | 
| libyuv YUV 4:0:0->RGBA |  522.71µs  |    1.8204ms    |  811.03µs  |   525.56µs    | 

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
| utils RGB10->YUV10 4:2:0   |  539.82µs  |    745.02µs    |   3.13ms   |    1.21ms     |
| libyuv RGB10->YUV10 4:2:0  |     x      |       x        |     x      |       x       |
| utils RGBA10->YUV10 4:2:0  |  674.12µs  |    807.89µs    |   3.66ms   |    1.57ms     |
| libyuv RGBA10->YUV10 4:2:0 |     x      |       x        |     x      |       x       |
| utils RGBA10->YUV10 4:2:2  |  838.82µs  |     1.04ms     |   6.33ms   |    1.83ms     |
| libyuv RGBA10->YUV10 4:2:2 |     x      |       x        |     x      |       x       |
| utils RGBA10->YUV10 4:4:4  | 659.93 µs  |     1.12ms     |   5.22ms   |    2.03ms     |

### Decoding 10-bit

|                           | time(NEON) | Time(AVX2 Win) | Time(AVX2) | Time(AVX-512) |
|---------------------------|:----------:|:--------------:|:----------:|:-------------:|
| utils YUV10 4:2:0->RGB10  |  640.34µs  |    662.38µs    |   2.91ms   |    1.23ms     |
| libyuv YUV10 4:2:0->RGB10 |     x      |       x        |     x      |       x       |
| utils YUV10 4:2:0->RGBA10 |  814.86µs  |    670.15µs    |   3.92ms   |    1.59ms     |
| utils YUV10 4:2:0->RGBA8  |  812.53µs  |    492.63µs    |   2.27ms   |   900.10µs    |
| libyuv YUV10 4:2:0->RGBA8 |  1.7037ms  |    6.8641ms    |   2.10ms   |   966.94µs    |
| utils YUV10 4:2:2->RGBA10 |  859.39µs  |    787.41µs    |   3.71ms   |    1.61ms     |
| utils YUV10 4:2:2->RGBA8  |  878.54µs  |    438.14µs    |   2.28ms   |   972.19µs    |
| libyuv YUV10 4:2:2->RGBA8 |  1.7056ms  |    6.8374ms    |   1.94ms   |   991.23µs    |
| utils YUV10 4:4:4->RGBA10 |  931.28µs  |    848.02µs    |   3.82ms   |   1.8639ms    |

This project is licensed under either of

- BSD-3-Clause License (see [LICENSE](LICENSE.md))
- Apache License, Version 2.0 (see [LICENSE](LICENSE-APACHE.md))

at your option.
