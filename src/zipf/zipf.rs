use std::ops::Range;

use crate::zipf::ZipfError;
use crate::zipf::ZipfIterator;

/// Implement Zipf distribution when s = 1
#[derive(Debug, Clone, Copy)]
struct ZipfOne {
    /// cached ln(b/a)
    ln_b_div_a: f64,
    /// cached ln(a)
    ln_a: f64,
}

impl ZipfOne {
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

/// Implement Zipf distribution when s != 1
#[derive(Debug, Clone, Copy)]
struct ZipfNonOne {
    // Cache: `q = 1-s`; not cached.
    /// Cache: `q_inv = 1/q`
    q_inv: f64,
    /// Cache: `a_pow_q = a^q`
    a_pow_q: f64,

    // Cache: `b_pow_q = b^q`
    // b_pow_q: f64,
    /// Cache: `b^q - a^q`
    b_pow_q_sub_a_pow_q: f64,
}

impl ZipfNonOne {
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

#[derive(Debug, Clone, Copy)]
enum ZipfImpl {
    One(ZipfOne),
    NonOne(ZipfNonOne),
}

impl ZipfImpl {
    fn new_unchecked(rng: Range<f64>, s: f64) -> Self {
        if s == 1.0 {
            Self::One(ZipfOne::new_unchecked(rng))
        } else {
            Self::NonOne(ZipfNonOne::new_unchecked(rng, s))
        }
    }

    #[inline]
    fn sample(&self, u: f64) -> f64 {
        match self {
            Self::One(zipf) => zipf.sample(u),
            Self::NonOne(zipf) => zipf.sample(u),
        }
    }
}

/// Zipf generates zipf distributed variates.
///
/// The Zipf struct caches intermediate variables for efficient generation
/// of zipf distributed values.
///
/// - `q = 1-s`; not cached.
/// - `q_inv = 1/q`
/// - `a_pow_q = a^q`
/// - `c = q/(b^q - a^q)`;  not cached
/// - `q_div_c = q/c`
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
    /// * `s` - Power parameter, must be > 0
    ///
    /// # Examples
    /// ```
    /// use load::zipf::Zipf;
    /// let zipf = Zipf::new(1.0..100.0, 1.1).unwrap();
    /// let value = zipf.sample(0.5);
    /// assert_eq!(format!("{:.4}", value), "7.6891");
    /// ```
    pub fn new(rng: Range<f64>, s: f64) -> Result<Self, ZipfError> {
        if s <= 0.0 {
            return Err(ZipfError::InvalidPowerParameter(s));
        }
        if rng.start <= 0.0 {
            return Err(ZipfError::InvalidRangeStart(rng.start));
        }
        if rng.end <= rng.start {
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
    pub fn sample_batch(&self, u_values: &[f64], output: &mut [f64]) {
        assert_eq!(
            u_values.len(),
            output.len(),
            "Input and output slices must have the same length"
        );

        for (u, out) in u_values.iter().zip(output.iter_mut()) {
            *out = self.implementation.sample(*u);
        }
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
        Ok(zipf.iter().map(|x| x as usize))
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
        Ok(zipf.iter().map(move |x| arr[x as usize - offset]))
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use crate::zipf::*;

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

        let counts = iter.take(1000).fold(HashMap::new(), |mut acc, x| {
            *acc.entry(x).or_insert(0) += 1;
            acc
        });

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

        let counts = iter.take(1000).fold(HashMap::new(), |mut acc, x| {
            *acc.entry(x).or_insert(0) += 1;
            acc
        });

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

        let counts = iter.take(1000).fold(HashMap::new(), |mut acc, x| {
            *acc.entry(x).or_insert(0) += 1;
            acc
        });

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
}
