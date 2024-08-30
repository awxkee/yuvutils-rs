/*
 * // Copyright (c) the Radzivon Bartoshyk. All rights reserved.
 * //
 * // Use of this source code is governed by a BSD-style
 * // license that can be found in the LICENSE file.
 */

#[derive(Debug, Copy, Clone, Ord, PartialOrd, Eq, PartialEq)]
pub enum YCgCoR {
    YCgCoRo = 1,
    YCgCoRe = 2,
}

impl From<usize> for YCgCoR {
    fn from(value: usize) -> Self {
        match value {
            1 => YCgCoR::YCgCoRo,
            2 => YCgCoR::YCgCoRe,
            _ => {
                panic!("Not found suitable type of YCgCoR for {}", value);
            }
        }
    }
}
