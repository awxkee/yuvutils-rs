# Rust utilities for YUV format handling and conversion.

Fast and simple YUV approximation conversion in pure Rust. At most the same as libyuv does. Performance will be equal to libyuv or slightly higher on platforms where SIMD is implemented. Otherwise equal or slower. 

Mostly implemented AVX2, SSE, NEON

X86 targets with SSE and AVX uses runtime dispatch to detect available cpu features.

Also contains AVX-512 intrinsics. Feature `nightly_avx512` and `nightly` rust channel compiler is required

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