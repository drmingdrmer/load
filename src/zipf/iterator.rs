use rand::prelude::StdRng;
use rand::Rng;
use rand::SeedableRng;

use crate::zipf::Zipf;

const DEFAULT_SEED: u64 = 666;

/// Iterator that generates zipf-distributed values with configurable random seed.
///
/// # Examples
/// ```
/// use load::zipf::Zipf;
/// use load::zipf::ZipfIterator;
/// use rand::rngs::StdRng;
/// use rand::SeedableRng;
///
/// let zipf = Zipf::new(1.0..100.0, 1.5).unwrap();
///
/// // Create iterator with default seed
/// let iter = zipf.iter();
/// let values: Vec<f64> = iter.take(3).collect();
/// let formatted: Vec<String> = values.iter().map(|v| format!("{:.4}", v)).collect();
/// assert_eq!(formatted, vec!["1.4376", "3.7734", "2.5003"]);
///
/// // Create iterator with custom seed (convenient)
/// let mut iter = ZipfIterator::with_seed(zipf.clone(), 42);
/// let values1: Vec<f64> = (&mut iter).take(3).collect();
/// let formatted1: Vec<String> = values1.iter().map(|v| format!("{:.4}", v)).collect();
/// assert_eq!(formatted1, vec!["3.6130", "3.8215", "5.4799"]);
///
/// // Create iterator with custom RNG (fluent API)
/// let rng = StdRng::seed_from_u64(123);
/// let mut iter = zipf.iter().with_rng(rng);
/// let values2: Vec<f64> = (&mut iter).take(3).collect();
/// let formatted2: Vec<String> = values2.iter().map(|v| format!("{:.4}", v)).collect();
/// assert_eq!(formatted2, vec!["1.4036", "1.3429", "37.9130"]);
/// ```
#[derive(Debug, Clone)]
pub struct ZipfIterator {
    zipf: Zipf,
    rng: StdRng,
}

impl ZipfIterator {
    /// Creates a new ZipfIterator with default seed.
    pub fn new(zipf: Zipf) -> Self {
        Self {
            zipf,
            rng: StdRng::seed_from_u64(DEFAULT_SEED),
        }
    }

    /// Creates a new ZipfIterator with the provided random number generator.
    pub fn with_rng(self, rng: StdRng) -> Self {
        Self {
            zipf: self.zipf,
            rng,
        }
    }

    /// Creates a new ZipfIterator with the specified seed.
    /// This is a convenience method that creates an StdRng internally.
    pub fn with_seed(zipf: Zipf, seed: u64) -> Self {
        Self::new(zipf).with_rng(StdRng::seed_from_u64(seed))
    }
}

impl Iterator for ZipfIterator {
    type Item = f64;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let u = self.rng.r#gen::<f64>();
        Some(self.zipf.sample(u))
    }
}

#[cfg(test)]
mod tests {

    use rand::rngs::StdRng;
    use rand::SeedableRng;

    use crate::zipf::*;

    #[test]
    fn test_zipf_iterator_rng_consistency() {
        let zipf = Zipf::new(1.0..10.0, 1.5).unwrap();

        // Test that same seed produces same sequence
        let iter1 = ZipfIterator::with_seed(zipf.clone(), 42);
        let iter2 = ZipfIterator::with_seed(zipf.clone(), 42);

        let seq1: Vec<f64> = iter1.take(10).collect();
        let seq2: Vec<f64> = iter2.take(10).collect();

        assert_eq!(seq1, seq2, "Same seed should produce identical sequences");
    }

    #[test]
    fn test_zipf_iterator_different_rngs() {
        let zipf = Zipf::new(1.0..5.0, 1.2).unwrap();
        let mut iter = ZipfIterator::with_seed(zipf.clone(), 123);

        let seq1: Vec<f64> = (&mut iter).take(5).collect();

        // Create new iterator with different seed
        iter = ZipfIterator::with_seed(zipf, 456);
        let seq2: Vec<f64> = (&mut iter).take(5).collect();

        // Sequences should be different with different seeds
        assert_ne!(
            seq1, seq2,
            "Different seeds should produce different sequences"
        );
    }

    #[test]
    fn test_zipf_iterator_rng_reproducibility() {
        let zipf = Zipf::new(2.0..8.0, 0.9).unwrap();
        let mut iter = zipf.iter().with_rng(StdRng::seed_from_u64(789));

        let seq1: Vec<f64> = (&mut iter).take(8).collect();

        // Create new iterator with same RNG setup
        iter = zipf.iter().with_rng(StdRng::seed_from_u64(789));
        let seq2: Vec<f64> = (&mut iter).take(8).collect();

        assert_eq!(seq1, seq2, "Same seed should reproduce identical sequence");
    }
}
