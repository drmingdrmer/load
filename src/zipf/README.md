# Zipf Distribution for Load Testing

## Overview

[Zipf distribution](https://en.wikipedia.org/wiki/Zipf%27s_law) models real-world access patterns where a small number of items account for the majority of requests. This pattern appears frequently in:

- Web cache access patterns (hot data)
- Database query frequency distributions
- File system access patterns
- Social media content popularity

Using zipf-distributed workloads in benchmarking creates realistic test scenarios that closely mirror production traffic patterns.

## Mathematical Foundation

The Zipf distribution follows the power law:

$$P(x) = C \cdot x^{-s}$$

Where:
- $C$ is a normalization constant
- $s > 0$ is the shape parameter (higher values create steeper distributions)
- $x$ represents the rank of an item

```text
 700 +--------------------------------------------------------------------+
     |      +      +      +      +      +     +      +      +      +      |
     |       :                                                    +.....+ |
 600 |-+     +                                                          +-|
     |       :                                                            |
     |       +                                                            |
 500 |-+      :                                                         +-|
     |        +                                                           |
     |         +                                                          |
 400 |-+       +                                                        +-|
     |          +                                                         |
 300 |-+         +                                                      +-|
     |           ++                                                       |
     |             +                                                      |
 200 |-+            ++                                                  +-|
     |                +++                                                 |
     |                  ++++                                              |
 100 |-+                    ++++++                                      +-|
     |                           ++++++++++++                             |
     |      +      +      +      +      +    +++++++++++++++++++++++++++++|
   0 +--------------------------------------------------------------------+
     0      10     20     30     40     50    60     70     80     90    100
```

The Zipf distribution has a key property: it's log-log linear. Taking the natural logarithm of both sides:

$$\ln(P(x)) = \ln(C) - s \cdot \ln(x)$$

This means that on a log-log plot, the distribution appears as a straight line with slope $-s$.

```text
   7 +----------------------------------------------------------------------+
     |      +      +      +      +    +  +      +      +      +      +      |
     |                                 +++                          +.....+ |
   6 |-+                                  +++                             +-|
     |                                       ++++                           |
     |                                           ++++                       |
   5 |-+                                            ++++                  +-|
     |                                                 ++++                 |
     |                                                     ++++             |
   4 |-+                                                      ++++        +-|
     |                                                           +++++      |
   3 |-+                                                             ++   +-|
     |                                                                      |
     |                                                                      |
   2 |-+                                                                  +-|
     |                                                                      |
     |                                                                      |
   1 |-+                                                                  +-|
     |                                                                      |
     |      +      +      +      +       +      +      +      +      +      |
   0 +----------------------------------------------------------------------+
     0     0.5     1     1.5     2      2.5     3     3.5     4     4.5     5
```

## Algorithm Implementation

The core algorithm uses the [inverse transform sampling](https://en.wikipedia.org/wiki/Inverse_transform_sampling) method. For the probability density function:

$$f(x) = C \cdot x^{-s}$$

We need to find the inverse of its cumulative distribution function (CDF).

**Step 1: Compute the CDF**

$$F(t) = \int_a^t C x^{-s} dx = C \frac{1}{1-s} \cdot (t^{1-s} - a^{1-s})$$

Let $q = 1-s$, then:

$$F(t) = \frac{C}{q}(t^q - a^q)$$

**Step 2: Normalize the distribution**

For the total probability to equal 1:

$$\int_a^b f(x) dx = 1$$

This gives us:

$$C = \frac{q}{b^q - a^q}$$

**Step 3: Inverse transform**

For a uniform random variable $u \in [0,1)$, solve $F(t) = u$:

$$u = \frac{C}{q}(t^q - a^q)$$

$$t = \left(\frac{q}{C} \cdot u + a^q\right)^{1/q} = ((b^q - a^q) u + a^q)^{1/q}$$

Therefore, given uniformly distributed $u$, the resulting $t$ follows a zipf distribution.

### Special Case: s = 1

When $s = 1$, the general formula breaks down because $q = 1-s = 0$, making the term $1/q$ undefined. This case requires special handling using logarithmic integration.

**Why s = 1 is special:**

The standard CDF formula $F(t) = \frac{C}{q}(t^q - a^q)$ becomes $\frac{C}{0}(t^0 - a^0) = \frac{C}{0} \cdot 0$, which is indeterminate.

**Derivation for s = 1:**

Starting with the PDF: $f(x) = C \cdot x^{-1} = \frac{C}{x}$

The CDF becomes:
$$F(t) = \int_a^t \frac{C}{x} dx = C(\ln t - \ln a) = C \ln\left(\frac{t}{a}\right)$$

**Normalization:**

For the total probability to equal 1:
$$\int_a^b \frac{C}{x} dx = C(\ln b - \ln a) = 1$$

Therefore: $C = \frac{1}{\ln b - \ln a}$

**Inverse transform:**

Setting $F(t) = u$ and solving for $t$:
$$u = \frac{\ln t - \ln a}{\ln b - \ln a} = \frac{\ln(t/a)}{\ln(b/a)}$$

$$\ln(t/a) = u \ln(b/a)$$

$$\frac{t}{a} = \left(\frac{b}{a}\right)^u$$

$$t = a \left(\frac{b}{a}\right)^u$$

This logarithmic form ensures numerical stability and is the classical Zipf distribution used in linguistic analysis and web traffic modeling.


## Usage Examples

### Basic Zipf Generation

```rust
use load::zipf::Zipf;

// Create a zipf generator for range [1, 100] with shape parameter 1.1
let zipf = Zipf::new(1.0..100.0, 1.1).unwrap();

// Generate zipf-distributed values
let value1 = zipf.sample(0.1);  // Low u → high rank (popular items)
let value2 = zipf.sample(0.9);  // High u → low rank (unpopular items)

// Assert formatted values (4 decimal places)
assert_eq!(format!("{:.4}", value1), "1.4565");
assert_eq!(format!("{:.4}", value2), "56.6416");
```

### Load Testing Scenario

```rust
use load::zipf::Zipf;

// Generate 10,000 accesses to a 1,000-item cache
// with zipf distribution (s=1.2 for realistic web traffic)
let cache_accesses: Vec<usize> = Zipf::indices_access(1..1001, 1.2).unwrap()
    .take(10)  // Number of accesses
    .collect();

// Assert exact sequence with deterministic seed
assert_eq!(cache_accesses, vec![2, 13, 6, 1, 223, 546, 4, 121, 2, 2]);
```

### Custom RNG for Reproducible Tests

```rust
use load::zipf::{Zipf, ZipfIterator};

// Create reproducible workload for benchmarking
let zipf = Zipf::new(1.0..100.0, 1.5).unwrap();
let mut iter = ZipfIterator::with_seed(zipf, 42);
let workload: Vec<usize> = iter.map(|x| x as usize).take(5).collect();

// Assert exact sequence with seed 42
assert_eq!(workload, vec![3, 3, 5, 2, 1]);
```

### Streaming Processing (Memory Efficient)

```rust
use load::zipf::Zipf;
use std::collections::HashMap;

// Process large workloads without allocating all indices upfront
let mut hit_counts = HashMap::new();

for index in Zipf::indices_access(1..11, 1.2).unwrap().take(100) {
    *hit_counts.entry(index).or_insert(0) += 1;
    
    // Process each access immediately (e.g., cache lookup, database query)
    // This approach uses constant memory regardless of workload size
}

let most_accessed = hit_counts.iter().max_by_key(|&(_, count)| count);
assert_eq!(most_accessed, Some((&1, &38)));
```

## Parameters Guide

- **Range [a, b]**: Defines the output range. For array indices, use `[1, array_length+1]`
- **Shape parameter s**: Controls distribution steepness
  - `s = 1.0`: Classical Zipf (web requests, word frequency)
  - `s = 1.2`: Typical web cache patterns
  - `s = 2.0`: Very steep distribution (80/20 rule becomes 95/5)
- **Array length**: Size of the dataset being accessed
- **Iterator control**: Use `.take(n)` to limit the number of accesses, or process indefinitely
