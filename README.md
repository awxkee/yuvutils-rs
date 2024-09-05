# Rust utilities for YUV format handling and conversion.

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

# SIMD

rustc `avx2`, `avx512f`, `avx512bw`, `neon`, `sse4.1` features should be set when you expect than code will run on supported device.

For AVX-512 target feature `avx512bw` is required along with feature `nightly_avx512` and `nightly` rust channel compiler.

Wasm `simd128` should be enabled for implemented SIMD wasm paths support

### Adding to project

```bash
cargo add yuvutils-rs
```

### RGB to YCbCr

```rust
rgb_to_yuv422(&mut y_plane, y_stride,
              &mut u_plane, u_width,
              &mut v_plane, v_width,
              &rgb, rgb_stride,
              width, height, 
              YuvRange::Full, YuvStandardMatrix::Bt709);
```

### RGB to sharp YUV

```rust
rgb_to_sharp_yuv420(&mut y_plane, y_stride,
                    &mut u_plane, u_width,
                    &mut v_plane, v_width,
                    &rgb, rgb_stride,
                    width, height, 
                    YuvRange::Full, YuvStandardMatrix::Bt709,
                    SharpYuvGammaTransfer::Srgb);
```

### YCbCr to RGB

```rust
yuv422_to_rgb(&y_plane, y_stride, 
              &u_plane, u_stride,
              &v_plane, v_stride,
              &mut rgb, rgb_stride,
              width, height, 
              YuvRange::Full, YuvStandardMatrix::Bt709);
```

### RGB To YCgCo

```rust
rgb_to_ycgco420(&mut y_plane, y_stride,
                &mut cg_plane, cg_width,
                &mut cg_plane, cg_width,
                &rgb, rgb_stride,
                width, height, 
                YuvRange::TV);
```

### YCgCo to RGB

```rust
ycgco420_to_rgb(&y_plane, y_stride, 
                &cg_plane, cg_stride,
                &co_plane, co_stride,
                &mut rgb, rgb_stride,
                width, height, 
                YuvRange::TV);
```

This project is licensed under either of

- BSD-3-Clause License (see [LICENSE](LICENSE.md))
- Apache License, Version 2.0 (see [LICENSE](LICENSE-APACHE.md))

at your option.
