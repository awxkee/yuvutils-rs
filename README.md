# Rust utilities for YUV format handling and conversion.

[![crates.io](https://img.shields.io/crates/v/yuvutils-rs.svg)](https://crates.io/crates/yuvutils-rs)
![Build](https://github.com/awxkee/yuvutils-rs/actions/workflows/build_push.yml/badge.svg)

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

YUV 8 bit-depth conversion

```bash
cargo bench --bench yuv8 --manifest-path ./app/Cargo.toml
```

Tests performed on the image 5763x3842

### Encoding 8-bit

|                        | time(NEON) | Time(AVX) |
|------------------------|:----------:|:---------:|
| utils RGB->YUV 4:2:0   |   3.16ms   |  3.53ms   |
| libyuv RGB->YUV 4:2:0  |   3.58ms   |  33.87ms  |
| utils RGBA->YUV 4:2:0  |   4.04ms   |  5.47ms   |
| libyuv RGBA->YUV 4:2:0 |   4.87ms   |  23.48ms  |
| utils RGBA->YUV 4:2:2  |   4.34ms   |  7.08ms   |
| libyuv RGBA->YUV 4:2:2 |   5.90ms   |  35.23ms  |
| utils RGBA->YUV 4:4:4  |   4.49ms   |  7.97ms   |

### Decoding 8-bit

|                        | time(NEON) | Time(AVX) |
|------------------------|:----------:|:---------:|
| utils YUV NV12->RGB    |   3.86ms   |  6.24ms   |
| libyuv YUV NV12->RGB   |   5.20ms   |  45.28ms  |
| utils YUV 4:2:0->RGB   |   3.26ms   |  5.25ms   |
| libyuv YUV 4:2:0->RGB  |   5.70ms   |  44.95ms  |
| utils YUV 4:2:0->RGBA  |   3.77ms   |  5.98ms   |
| libyuv YUV 4:2:0->RGBA |   6.13ms   |  6.88ms   |
| utils YUV 4:2:2->RGBA  |   4.88ms   |  6.91ms   |
| libyuv YUV 4:2:2->RGBA |   5.91ms   |  6.91ms   |
| utils YUV 4:4:4->RGBA  |   4.79ms   |  7.20ms   |
| libyuv YUV 4:4:4->RGBA |   4.82ms   |  7.30ms   |

YUV 16 bit-depth conversion

```bash
cargo bench --bench yuv16 --manifest-path ./app/Cargo.toml
```

### Encoding 10-bit

10-bit encoding is not implemented in `libyuv`

|                            | time(NEON) | Time(AVX) |
|----------------------------|:----------:|:---------:|
| utils RGB10->YUV10 4:2:0   |   4.98ms   |  8.71ms   |
| libyuv RGB10->YUV10 4:2:0  |     x      |     x     |
| utils RGBA10->YUV10 4:2:0  |   6.03ms   |  9.79ms   |
| libyuv RGBA10->YUV10 4:2:0 |     x      |     x     |
| utils RGBA10->YUV10 4:2:2  |   5.99ms   |  13.54ms  |
| libyuv RGBA10->YUV10 4:2:2 |     x      |     x     |
| utils RGBA10->YUV10 4:4:4  |   4.84ms   |  16.88ms  |

### Decoding 10-bit

|                            | time(NEON) | Time(AVX) |
|----------------------------|:----------:|:---------:|
| utils YUV10 4:2:0->RGB10   |   5.64ms   |  12.27ms  |
| libyuv YUV10 4:2:0->RGB10  |     -      |     -     |
| utils YUV10 4:2:0->RGBA10  |   6.26ms   |  14.79ms  |
| utils YUV10 4:2:0->RGBA8   |   7.05ms   |  42.44ms  |
| libyuv YUV10 4:2:0->RGBA8  |  12.79ms   |  63.02ms  |
| utils YUV10 4:2:2->RGBA10  |   6.31ms   |  15.33ms  |
| utils YUV10 4:2:2->RGBA8   |   7.33ms   |  13.55ms  |
| libyuv YUV10 4:2:2->RGBA10 |  12.62ms   |  58.98ms  |
| utils YUV10 4:4:4->RGBA10  |   6.39ms   |  16.39ms  |

This project is licensed under either of

- BSD-3-Clause License (see [LICENSE](LICENSE.md))
- Apache License, Version 2.0 (see [LICENSE](LICENSE-APACHE.md))

at your option.
