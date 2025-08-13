//! Zipf distribution implementation for load testing.
//!
//! This module provides zipf distributed variates for realistic load testing scenarios.
//! See: <https://en.wikipedia.org/wiki/Zipf%27s_law>
#![doc = include_str!("README.md")]

mod errors;
mod iterator;
#[allow(clippy::module_inception)]
mod zipf;

pub use errors::ZipfError;
pub use iterator::ZipfIterator;
pub use zipf::Zipf;
