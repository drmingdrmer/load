use std::ops::Range;

use crate::zipf::ZipfError;
use crate::zipf::ZipfIterator;

/// Zipf distribution with unit exponent (s = 1)
#[derive(Debug, Clone, Copy)]
struct ZipfS1 {
    /// cached ln(b/a)
    ln_b_div_a: f64,
    /// cached ln(a)
    ln_a: f64,
}

impl ZipfS1 {
    fn new_unchecked(rng: Range<f64>) -> Self {
        let a = rng.start;
        let b = rng.end;
        Self {
            ln_b_div_a: (b / a).ln(),
            ln_a: a.ln(),
        }
    }

    #[inline]
    fn sample(&self, u: f64) -> f64 {
        // Optimized: pre-computed logarithms, single exp call
        (self.ln_a + u * self.ln_b_div_a).exp()
    }
}

/// Zipf distribution with arbitrary exponent (s != 1)
///
/// Let `q = 1-s`, then we cache:
/// - `q_inv = 1/q`
/// - `a_pow_q = a^q`
/// - `b_pow_q_sub_a_pow_q = b^q - a^q`
#[derive(Debug, Clone, Copy)]
struct ZipfGeneral {
    q_inv: f64,
    a_pow_q: f64,
    b_pow_q_sub_a_pow_q: f64,
}

impl ZipfGeneral {
    pub fn new_unchecked(rng: Range<f64>, s: f64) -> Self {
        let a = rng.start;
        let b = rng.end;

        let q = 1.0 - s;
        let q_inv = 1.0 / q;
        let a_pow_q = a.powf(q);
        let b_pow_q = b.powf(q);

        Self {
            q_inv,
            a_pow_q,
            b_pow_q_sub_a_pow_q: b_pow_q - a_pow_q,
        }
    }

    #[inline]
    fn sample(&self, u: f64) -> f64 {
        // Use fused multiply-add for better performance
        (u.mul_add(self.b_pow_q_sub_a_pow_q, self.a_pow_q)).powf(self.q_inv)
    }
}

/// Below this distance from `s = 1`, `ZipfGeneral` loses more accuracy to
/// cancellation in `b^q - a^q` (error ~1e-16/|1-s|) than routing to the
/// `s = 1` log form costs (true distribution difference ~|1-s|·ln²(b/a)).
const S_ONE_TOLERANCE: f64 = 1e-8;

#[derive(Debug, Clone, Copy)]
enum ZipfImpl {
    S1(ZipfS1),
    General(ZipfGeneral),
}

impl ZipfImpl {
    fn new_unchecked(rng: Range<f64>, s: f64) -> Self {
        if (s - 1.0).abs() < S_ONE_TOLERANCE {
            Self::S1(ZipfS1::new_unchecked(rng))
        } else {
            Self::General(ZipfGeneral::new_unchecked(rng, s))
        }
    }

    #[inline]
    fn sample(&self, u: f64) -> f64 {
        match self {
            Self::S1(zipf) => zipf.sample(u),
            Self::General(zipf) => zipf.sample(u),
        }
    }
}

/// Zipf generates zipf distributed variates.
///
/// The Zipf struct caches intermediate variables for efficient generation
/// of zipf distributed values.
///
/// Let `q = 1-s`, then we cache:
/// - `q_inv = 1/q`
/// - `a_pow_q = a^q`
/// - `b_pow_q_sub_a_pow_q = b^q - a^q`
#[derive(Debug, Clone, Copy)]
pub struct Zipf {
    /// Zipf parameter: The start and end of the range.
    #[allow(dead_code)]
    start: f64,
    #[allow(dead_code)]
    end: f64,
    /// The Zipf parameter: The power parameter.
    #[allow(dead_code)]
    s: f64,

    implementation: ZipfImpl,
}

impl Zipf {
    /// Creates a Zipf struct that generates values in range `a..b`, with
    /// the power `s > 0`.
    ///
    /// Usually a is greater than 1, since C x**(-s) is infinite when x gets close to 0.
    ///
    /// # Arguments
    /// * `rng` - Range of the values to generate
    /// * `s` - Power parameter, must be finite and > 0
    ///
    /// # Examples
    /// ```
    /// use load::zipf::Zipf;
    /// let zipf = Zipf::new(1.0..100.0, 1.1).unwrap();
    /// let value = zipf.sample(0.5);
    /// assert_eq!(format!("{:.4}", value), "7.6891");
    /// ```
    pub fn new(rng: Range<f64>, s: f64) -> Result<Self, ZipfError> {
        if !s.is_finite() || s <= 0.0 {
            return Err(ZipfError::InvalidPowerParameter(s));
        }
        if !rng.start.is_finite() || rng.start <= 0.0 {
            return Err(ZipfError::InvalidRangeStart(rng.start));
        }
        if !rng.end.is_finite() || rng.end <= rng.start {
            return Err(ZipfError::InvalidRangeEnd {
                start: rng.start,
                end: rng.end,
            });
        }

        let implementation = ZipfImpl::new_unchecked(rng.clone(), s);

        Ok(Self {
            start: rng.start,
            end: rng.end,
            s,
            implementation,
        })
    }

    /// Converts an evenly distributed random number `u ∈ [0, 1)`, e.g., a common
    /// random value, to a zipf distributed variate which is in range `[a, b]`.
    ///
    /// # Arguments
    /// * `u` - Uniform random value in [0, 1)
    ///
    /// # Returns
    /// A zipf distributed value in the range [a, b]
    ///
    /// # Note
    /// Because of the inaccuracy of float number, the output value may be a
    /// little bit lower than a or greater than b.
    ///
    /// # Examples
    /// ```
    /// use load::zipf::Zipf;
    /// let zipf = Zipf::new(1.0..100.0, 1.1).unwrap();
    /// let value = zipf.sample(0.5);
    /// assert_eq!(format!("{:.4}", value), "7.6891");
    /// ```
    #[inline]
    pub fn sample(&self, u: f64) -> f64 {
        self.implementation.sample(u)
    }

    /// Batch sample multiple values for better performance.
    /// This is SIMD-friendly and can be vectorized by the compiler.
    pub fn sample_batch(&self, u_values: &[f64], output: &mut [f64]) -> Result<(), ZipfError> {
        if u_values.len() != output.len() {
            return Err(ZipfError::MismatchedSliceLengths {
                input: u_values.len(),
                output: output.len(),
            });
        }

        for (u, out) in u_values.iter().zip(output.iter_mut()) {
            *out = self.implementation.sample(*u);
        }

        Ok(())
    }

    /// Creates an infinite iterator that yields zipf-distributed values with a default random number generator.
    pub fn iter(&self) -> ZipfIterator {
        ZipfIterator::new(*self)
    }

    /// Returns an iterator that yields shuffled array indices following zipf distribution.
    ///
    /// # Arguments
    /// * `rng` - Range of the array indices to generate
    /// * `s` - Power parameter, must be > 0
    pub fn indices_access(
        rng: Range<usize>,
        s: f64,
    ) -> Result<impl Iterator<Item = usize>, ZipfError> {
        let zipf = Zipf::new(rng.start as f64..rng.end as f64, s)?;
        Ok(zipf
            .iter()
            .map(move |x| Self::clamp_index(x, rng.start, rng.end)))
    }

    /// Returns an iterator that yields array elements following zipf distribution.
    ///
    /// # Arguments
    /// * `offset`: where to put the 0-th array elt. For example `offset=2` means the 0-th elt is at 2 on x-axis
    /// * `arr` - Array to generate
    /// * `s` - Power parameter, must be > 0
    ///
    /// # Examples
    /// ```
    /// use load::zipf::Zipf;
    /// let zipf = Zipf::array_access(3, vec![1, 2, 3, 4], 1.1).unwrap();
    /// // 1,2,1,4...
    /// ```
    pub fn array_access<T>(
        offset: usize,
        arr: Vec<T>,
        s: f64,
    ) -> Result<impl Iterator<Item = T>, ZipfError>
    where
        T: Copy,
    {
        if arr.is_empty() {
            return Err(ZipfError::EmptyArray);
        }
        let a = offset as f64;
        let b = a + arr.len() as f64;
        let zipf = Zipf::new(a..b, s)?;
        Ok(zipf
            .iter()
            .map(move |x| arr[Self::clamp_index(x, offset, offset + arr.len()) - offset]))
    }

    /// Clamps a sampled value to an index in `[start, end)`.
    ///
    /// `sample()` may return values slightly below `a` or at/above `b` due to
    /// float rounding; casting those to an index would escape the range.
    fn clamp_index(x: f64, start: usize, end: usize) -> usize {
        (x as usize).clamp(start, end - 1)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use crate::zipf::*;

    fn collect_histogram<T>(iter: impl Iterator<Item = T>, n: usize) -> HashMap<T, usize>
    where T: std::hash::Hash + Eq {
        iter.take(n).fold(HashMap::new(), |mut acc, x| {
            *acc.entry(x).or_insert(0) += 1;
            acc
        })
    }

    #[test]
    fn test_indices_access_iteration() {
        let iter = Zipf::indices_access(1..10, 0.7).unwrap();
        let samples: Vec<usize> = iter.take(100).collect();

        assert_eq!(samples, vec![
            1, 4, 3, 1, 8, 9, 2, 7, 1, 1, 4, 8, 7, 3, 6, 8, 2, 2, 2, 8, 7, 6, 2, 5, 9, 3, 8, 4, 4,
            4, 2, 3, 1, 1, 2, 1, 2, 6, 1, 3, 8, 1, 7, 4, 6, 1, 6, 2, 1, 1, 7, 4, 1, 2, 8, 3, 1, 8,
            1, 8, 7, 1, 5, 7, 1, 2, 8, 2, 9, 7, 1, 1, 2, 5, 2, 1, 3, 8, 4, 4, 1, 6, 4, 1, 1, 2, 2,
            8, 7, 1, 6, 1, 8, 5, 6, 7, 1, 1, 1, 5
        ]);
    }

    #[test]
    fn test_indices_access_count() {
        let iter = Zipf::indices_access(1..10, 0.8).unwrap();
        let counts = collect_histogram(iter, 1000);

        // got: {3: 112, 4: 114, 6: 73, 8: 66, 5: 88, 9: 61, 7: 81, 1: 236, 2: 169}
        assert_eq!(
            counts,
            HashMap::from_iter([
                (1, 236),
                (2, 169),
                (3, 112),
                (4, 114),
                (5, 88),
                (6, 73),
                (7, 81),
                (8, 66),
                (9, 61)
            ])
        );
    }

    #[test]
    fn test_indices_access_count_s_eq_1() {
        let iter = Zipf::indices_access(1..10, 1.0).unwrap();
        let counts = collect_histogram(iter, 1000);

        // got: {3: 123, 1: 285, 2: 171, 5: 80, 8: 57, 9: 47, 6: 69, 4: 96, 7: 72}
        assert_eq!(
            counts,
            HashMap::from_iter([
                (1, 285),
                (2, 171),
                (3, 123),
                (4, 96),
                (5, 80),
                (6, 69),
                (7, 72),
                (8, 57),
                (9, 47),
            ])
        );
    }

    #[test]
    fn test_array_access_count() {
        let iter = Zipf::array_access(3, vec!['a', 'b', 'c', 'd', 'e'], 0.8).unwrap();
        let counts = collect_histogram(iter, 1000);

        // got: {'b': 205, 'a': 261, 'e': 168, 'c': 200, 'd': 166}
        assert_eq!(
            counts,
            HashMap::from_iter([('a', 261), ('b', 205), ('c', 200), ('d', 166), ('e', 168)])
        );
    }

    #[test]
    fn test_indices_access_edge_cases() {
        // Single element range
        let iter = Zipf::indices_access(3..4, 1.0).unwrap();
        let samples: Vec<usize> = iter.take(10).collect();
        assert!(samples.iter().all(|&i| i == 3));

        // Range starting from non-zero
        let iter = Zipf::indices_access(5..8, 1.5).unwrap();
        let samples: Vec<usize> = iter.take(100).collect();
        assert!(samples.iter().all(|&i| (5..8).contains(&i)));
    }

    #[test]
    fn test_array_access_different_types() {
        let single = vec![42];
        let iter = Zipf::array_access(2, single, 2.0).unwrap();
        let samples: Vec<i32> = iter.take(10).collect();
        assert!(samples.iter().all(|&n| n == 42));
    }

    #[test]
    fn test_zipf_distribution_strength() {
        let iter = Zipf::indices_access(1..11, 2.0).unwrap();
        let samples: Vec<usize> = iter.take(10000).collect();

        let mut counts = [0; 11];
        for &idx in &samples {
            counts[idx] += 1;
        }

        // With s=2.0, the distribution should be quite skewed
        // Index 1 should have significantly more hits than index 6
        assert!(
            counts[1] > counts[6] * 2,
            "With s=2.0, index 1 should be much more frequent than index 6"
        );
    }

    #[test]
    fn test_new_rejects_invalid_params() {
        assert_eq!(
            Zipf::new(1.0..10.0, 0.0).unwrap_err(),
            ZipfError::InvalidPowerParameter(0.0)
        );
        assert_eq!(
            Zipf::new(0.0..10.0, 1.1).unwrap_err(),
            ZipfError::InvalidRangeStart(0.0)
        );
        assert_eq!(
            Zipf::new(5.0..5.0, 1.1).unwrap_err(),
            ZipfError::InvalidRangeEnd {
                start: 5.0,
                end: 5.0
            }
        );
    }

    #[test]
    fn test_new_rejects_non_finite_params() {
        assert!(matches!(
            Zipf::new(1.0..10.0, f64::NAN),
            Err(ZipfError::InvalidPowerParameter(_))
        ));
        assert_eq!(
            Zipf::new(1.0..10.0, f64::INFINITY).unwrap_err(),
            ZipfError::InvalidPowerParameter(f64::INFINITY)
        );
        assert!(matches!(
            Zipf::new(f64::NAN..10.0, 1.1),
            Err(ZipfError::InvalidRangeStart(_))
        ));
        assert!(matches!(
            Zipf::new(1.0..f64::NAN, 1.1),
            Err(ZipfError::InvalidRangeEnd { .. })
        ));
        assert_eq!(
            Zipf::new(1.0..f64::INFINITY, 1.0).unwrap_err(),
            ZipfError::InvalidRangeEnd {
                start: 1.0,
                end: f64::INFINITY
            }
        );
    }

    #[test]
    fn test_s_near_one_routes_to_log_path() {
        // One ulp above 1.0: the general path's `b^q - a^q` cancels to float
        // noise, deviating up to 40%; near-1 s must sample like s = 1.
        let near_one = Zipf::new(1.0..100.0, 1.0 + f64::EPSILON).unwrap();
        let exact_one = Zipf::new(1.0..100.0, 1.0).unwrap();
        for u in [0.0, 0.25, 0.5, 0.75, 0.9] {
            assert_eq!(near_one.sample(u), exact_one.sample(u), "u={u}");
        }
    }

    #[test]
    fn test_clamp_index_on_boundary_samples() {
        // Deterministic parameterizations where float rounding pushes
        // sample() outside [a, b); unclamped index math panicked on these.
        let max_u = f64::from_bits(0x3FEF_FFFF_FFFF_FFFF); // largest f64 < 1.0

        let zipf = Zipf::new(100.0..107.0, 1.07).unwrap();
        let low = zipf.sample(0.0);
        assert!(low < 100.0, "expected rounding below a, got {low}");
        assert_eq!(Zipf::clamp_index(low, 100, 107), 100);

        let zipf = Zipf::new(3.0..8.0, 1.1).unwrap();
        let high = zipf.sample(max_u);
        assert!(high >= 8.0, "expected rounding at/above b, got {high}");
        assert_eq!(Zipf::clamp_index(high, 3, 8), 7);

        // In-range values pass through unchanged.
        assert_eq!(Zipf::clamp_index(5.9, 3, 8), 5);
    }

    #[test]
    fn test_sample_batch_length_mismatch() {
        let zipf = Zipf::new(1.0..10.0, 1.1).unwrap();
        let u = [0.1, 0.5, 0.9];

        let mut out = [0.0; 2];
        assert_eq!(
            zipf.sample_batch(&u, &mut out).unwrap_err(),
            ZipfError::MismatchedSliceLengths {
                input: 3,
                output: 2
            }
        );

        let mut out = [0.0; 3];
        assert_eq!(zipf.sample_batch(&u, &mut out), Ok(()));
        assert_eq!(out, [zipf.sample(0.1), zipf.sample(0.5), zipf.sample(0.9)]);
    }
}
