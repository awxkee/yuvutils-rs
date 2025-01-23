# Rust utilities for YUV format handling and conversion.

[![crates.io](https://img.shields.io/crates/v/yuvutils-rs.svg)](https://crates.io/crates/yuvutils-rs)
![Build](https://github.com/awxkee/yuvutils-rs/actions/workflows/build_push.yml/badge.svg)

Fast and simple YUV approximation conversion in pure Rust. At most the same as libyuv does.
Performance will be equal to libyuv or slightly higher on platforms where SIMD is implemented. Otherwise, equal or slower.
Used precision is slightly higher than used in libyuv, however for decoding 4:2:0 is almost always faster than libyuv,
decoding 4:2:2, 4:4:4 almost the same, however encoding except 4:2:0 often is slower with better results.
Modern NEON CPU is almost always faster than libyuv.

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

Runtime dispatch is used for use if available `sse4.1`, `avx2`, `avx512bw`, `avx512vbmi`, `avx512vnni`, `avxvnni`, `rdm` for NEON. 

For the `sse4.1`, `avx2`, `rdm` there is no action needed, however for AVX-512 activated feature `nightly_avx512` along `nightly` rust channel is required.

Always consider to compile with `nightly_avx512` when compiling for x86, it adds noticeable gain on supported cpus. 

Wasm `simd128` as target feature should be enabled for implemented SIMD wasm paths support,

# Rayon 

Some paths have multi-threading support. However, YUV conversion usually makes more sense in single-threaded mode, thus use this feature with care.

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

AVX2 tests performed on `m4.large` with Ubuntu 24.04 droplet.
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
| utils RGB->YUV 4:2:0   |  311.78µs  |    369.37µs    |   1.05ms   |   722.87µs    |
| libyuv RGB->YUV 4:2:0  |  306.80µs  |     3.62ms     |   1.10ms   |   972.52µs    |
| utils RGBA->YUV 4:2:0  |  266.66µs  |    367.47µs    |  781.54µs  |   785.93µs    |
| libyuv RGBA->YUV 4:2:0 |  429.36µs  |     2.70ms     |  846.69µs  |   707.33µs    |
| utils RGBA->YUV 4:2:2  |  362.72µs  |    464.82µs    |   1.06ms   |   964.87µs    |
| libyuv RGBA->YUV 4:2:2 |  616.29µs  |    4.3003ms    |   1.09ms   |   947.97µs    |
| utils RGBA->YUV 4:4:4  |  378.49µs  |    534.87µs    |   1.09ms   |   974.29µs    |

### Decoding 8-bit

| Conversion             | time(NEON) | Time(AVX2 Win) | Time(AVX2) | Time(AVX-512) | 
|------------------------|:----------:|:--------------:|:----------:|:-------------:| 
| utils YUV NV12->RGBA   |  399.37µs  |    359.11µs    |  954.72µs  |   736.21µs    | 
| utils YUV NV12->RGB    |  358.64µs  |    339.53µs    |  895.20µs  |   654.37µs    | 
| libyuv YUV NV12->RGB   |  621.11µs  |     5.46ms     |  979.96µs  |   800.85µs    | 
| utils YUV 4:2:0->RGB   |  346.13µs  |    375.34µs    |  923.83µs  |   614.19µs    | 
| libyuv YUV 4:2:0->RGB  |  747.18µs  |    5.8231ms    |   1.42ms   |    1.16ms     | 
| utils YUV 4:2:0->RGBA  |  396.58µs  |    380.13µs    |  942.92µs  |   733.88µs    | 
| libyuv YUV 4:2:0->RGBA |  710.88µs  |    557.44µs    |   1.03ms   |   870.64µs    | 
| utils YUV 4:2:2->RGBA  |  449.25µs  |    410.71µs    |   1.10ms   |   824.69µs    | 
| libyuv YUV 4:2:2->RGBA |  734.25µs  |    579.24µs    |   1.04ms   |   851.24µs    | 
| utils YUV 4:4:4->RGBA  |  432.32µs  |    366.85µs    |   1.10ms   |   819.17µs    | 
| libyuv YUV 4:4:4->RGBA |  596.55µs  |    525.90µs    |  963.66µs  |   803.62µs    | 
| utils YUV 4:0:0->RGBA  |  220.67µs  |    254.83µs    |  651.88µs  |   600.43µs    | 
| libyuv YUV 4:0:0->RGBA |  522.71µs  |    1.8204ms    |  635.61µs  |   525.56µs    | 

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
| utils RGB10->YUV10 4:2:0   |  502.65µs  |    745.02µs    |   1.88ms   |    1.21ms     |
| libyuv RGB10->YUV10 4:2:0  |     x      |       x        |     x      |       x       |
| utils RGBA10->YUV10 4:2:0  |  605.45µs  |    807.89µs    |   2.03ms   |    1.57ms     |
| libyuv RGBA10->YUV10 4:2:0 |     x      |       x        |     x      |       x       |
| utils RGBA10->YUV10 4:2:2  |  769.08µs  |     1.04ms     |   2.51ms   |    1.83ms     |
| libyuv RGBA10->YUV10 4:2:2 |     x      |       x        |     x      |       x       |
| utils RGBA10->YUV10 4:4:4  |  600.56µs  |     1.12ms     |   2.82ms   |    2.03ms     |

### Decoding 10-bit

|                           | time(NEON) | Time(AVX2 Win) | Time(AVX2) | Time(AVX-512) |
|---------------------------|:----------:|:--------------:|:----------:|:-------------:|
| utils YUV10 4:2:0->RGB10  |  596.78µs  |    662.38µs    |   1.76ms   |    1.23ms     |
| libyuv YUV10 4:2:0->RGB10 |     x      |       x        |     x      |       x       |
| utils YUV10 4:2:0->RGBA10 |  727.52µs  |    670.15µs    |   1.92ms   |    1.59ms     |
| utils YUV10 4:2:0->RGBA8  |  765.23µs  |    492.63µs    |   1.36ms   |   900.10µs    |
| libyuv YUV10 4:2:0->RGBA8 |   1.54ms   |    6.8641ms    |   1.16ms   |   966.94µs    |
| utils YUV10 4:2:2->RGBA10 |  746.51µs  |    787.41µs    |   2.02ms   |    1.61ms     |
| utils YUV10 4:2:2->RGBA8  |  799.25µs  |    438.14µs    |   1.23ms   |   972.19µs    |
| libyuv YUV10 4:2:2->RGBA8 |   1.55ms   |    6.8374ms    |   1.21ms   |   991.23µs    |
| utils YUV10 4:4:4->RGBA10 |  953.73µs  |    848.02µs    |   2.26ms   |   1.8639ms    |

### Geometry and mirroring

```bash
cargo bench --bench geometry --manifest-path ./app/Cargo.toml
```

|                          | time(NEON) | Time(AVX2 Win) | Time(AVX2) |
|--------------------------|:----------:|:--------------:|:----------:|
| utils Rotate 90 RGBA8    |  560.74µs  |    655.43µs    |   2.50ms   |
| libyuv Rotate 90 RGBA8   |   1.84ms   |     1.01ms     |   2.86ms   |
| utils Rotate 90 Plane8   |  215.43µs  |    387.44µs    |   1.05ms   |
| libyuv Rotate 90 Plane8  |  291.15µs  |    416.52µs    |   1.06ms   |
| utils Rotate 180 RGBA8   |  326.44µs  |    506.35µs    |  657.93µs  |
| libyuv Rotate 180 RGBA8  |  378.70µs  |    512.83µs    |  669.12µs  |
| utils Rotate 180 Plane8  |  61.02µs   |    127.26µs    |  227.66µs  |
| libyuv Rotate 180 Plane8 |  94.11µs   |    418.83µs    |  233.66µs  |
| utils Rotate 270 RGBA8   |  564.78µs  |    652.83µs    |   2.57ms   |
| libyuv Rotate 270 RGBA8  |   1.99ms   |     1.03ms     |   3.06ms   |

This project is licensed under either of

- BSD-3-Clause License (see [LICENSE](LICENSE.md))
- Apache License, Version 2.0 (see [LICENSE](LICENSE-APACHE.md))

at your option.
