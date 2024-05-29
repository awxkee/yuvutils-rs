# Rust utilities for YUV format handling and conversion.

Fast and simple YUV approximation conversion in pure Rust. At most the same as libyuv does. Performance will be equal to libyuv or slightly higher on platforms where SIMD is implemented. Otherwise equal or slower. 

Mostly implemented AVX-512, AVX2, SSE, NEON

X86 targets with SSE and AVX uses runtime dispatch to detect available cpu features.

Supports:
- [x] YCbCr ( aka YUV )
- [x] YCgCo

Also contains AVX-512 intrinsics. Feature `nightly_avx512` and `nightly` rust channel compiler is required

rustc `avx2`, `avx512f`, `neon` features should be set when you expect than code will run on supported device. 
SSE 4.1 target feature settings is not required but preferred.

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
