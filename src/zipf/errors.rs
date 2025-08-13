use std::fmt;

/// Errors that can occur when creating or using Zipf distributions.
#[derive(Debug, Clone, PartialEq)]
pub enum ZipfError {
    /// The power parameter s must be greater than 0.
    InvalidPowerParameter(f64),
    /// The range start must be greater than 0.
    InvalidRangeStart(f64),
    /// The range end must be greater than the start.
    InvalidRangeEnd { start: f64, end: f64 },
    /// The array cannot be empty.
    EmptyArray,
}

impl fmt::Display for ZipfError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ZipfError::InvalidPowerParameter(s) => {
                write!(f, "Power parameter s must be > 0, got: {}", s)
            }
            ZipfError::InvalidRangeStart(start) => {
                write!(f, "Range start must be > 0, got: {}", start)
            }
            ZipfError::InvalidRangeEnd { start, end } => {
                write!(f, "Range end must be > start, got: {}..{}", start, end)
            }
            ZipfError::EmptyArray => {
                write!(f, "Array cannot be empty")
            }
        }
    }
}

impl std::error::Error for ZipfError {}
