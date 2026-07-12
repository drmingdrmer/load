//! Zipf distribution implementation for load testing.
//!
//! This module provides zipf distributed variates for realistic load testing scenarios.
//! See: <https://en.wikipedia.org/wiki/Zipf%27s_law>
#![doc = include_str!("README.md")]

/// Mathematical derivation of the Zipf sampling formulas.
#[doc = include_str!("formulas.md")]
pub mod formulas {}

mod errors;
mod iterator;
#[allow(clippy::module_inception)]
mod zipf;

pub use errors::ZipfError;
pub use iterator::ZipfIterator;
pub use zipf::Zipf;
