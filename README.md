# load

Load generator for benchmark tasks with [Zipf distribution](https://en.wikipedia.org/wiki/Zipf%27s_law).

Zipf distribution models real-world access patterns where a small number of items account for the majority of requests (hot data). This is common in web caches, database queries, and file system access patterns.

## Usage

```rust
use load::zipf::Zipf;

// Generate cache access indices with Zipf distribution
let cache_accesses: Vec<usize> = Zipf::indices_access(1..1001, 1.2).unwrap()
    .take(10000)
    .collect();

// Generate raw Zipf values
let zipf = Zipf::new(1.0..100.0, 1.1).unwrap();
let value = zipf.sample(0.1);
```

Shape parameter `s` controls distribution steepness:
- `s = 1.0`: Classical Zipf
- `s = 1.2`: Typical web cache patterns
- `s = 2.0`: Very steep (95/5 rule)
