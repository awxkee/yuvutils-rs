#[derive(Debug, Copy, Clone, Ord, PartialOrd, Eq, PartialEq)]
pub enum YCgCoR {
    YCgCoRo = 1,
    YCgCoRe = 2,
}

impl From<usize> for YCgCoR {
    fn from(value: usize) -> Self {
        return match value {
            1 => YCgCoR::YCgCoRo,
            2 => YCgCoR::YCgCoRe,
            _ => {
                panic!("Not found suitable type of YCgCoR for {}", value);
            }
        };
    }
}
