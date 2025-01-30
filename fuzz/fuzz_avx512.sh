#!/bin/bash
cargo fuzz run rgb_to_yuv --no-default-features --features nightly_avx512 -- -max_total_time=15
cargo fuzz run rgb_to_nv --no-default-features --features nightly_avx512 -- -max_total_time=15
cargo fuzz run rgb_to_y --no-default-features --features nightly_avx512 -- -max_total_time=15
cargo fuzz run rgb16_to_yuv16 --no-default-features --features nightly_avx512 -- -max_total_time=15
cargo fuzz run rgb16_to_nv16 --no-default-features --features nightly_avx512 -- -max_total_time=15
cargo fuzz run yuv_to_rgb --no-default-features --features nightly_avx512 -- -max_total_time=15
cargo fuzz run yuv_nv_to_rgb --no-default-features --features nightly_avx512 -- -max_total_time=15
cargo fuzz run y_to_rgb --no-default-features --features nightly_avx512 -- -max_total_time=15
cargo fuzz run yuv16_to_rgb16 --no-default-features --features nightly_avx512 -- -max_total_time=15
cargo fuzz run y16_to_rgb16 --no-default-features --features nightly_avx512 -- -max_total_time=15
cargo fuzz run yuv_to_yuyu2 --no-default-features --features nightly_avx512 -- -max_total_time=15
cargo fuzz run yuv_nv16_to_rgb16 --no-default-features --features nightly_avx512 -- -max_total_time=15
cargo fuzz run shuffle --no-default-features --features nightly_avx512 -- -max_total_time=15
cargo fuzz run rgb_to_f16 --no-default-features --features nightly_avx512 -- -max_total_time=15