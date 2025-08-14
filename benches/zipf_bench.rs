#![feature(test)]
extern crate test;

use load::zipf::Zipf;
use test::Bencher;

#[bench]
fn bench_zipf_sample_s_1(b: &mut Bencher) {
    let zipf = Zipf::new(1.0..1000000.0, 1.0).unwrap();
    b.iter(|| zipf.sample(0.5));
}

#[bench]
fn bench_zipf_sample_s_1_07(b: &mut Bencher) {
    let zipf = Zipf::new(1.0..1000000.0, 1.07).unwrap();
    b.iter(|| zipf.sample(0.5));
}

#[bench]
fn bench_zipf_sample_s_2(b: &mut Bencher) {
    let zipf = Zipf::new(1.0..1000000.0, 2.0).unwrap();
    b.iter(|| zipf.sample(0.5));
}

#[bench]
fn bench_zipf_iter(b: &mut Bencher) {
    let zipf = Zipf::new(1.0..1000.0, 1.07).unwrap();
    let mut iter = zipf.iter();
    b.iter(|| iter.next());
}

#[bench]
fn bench_zipf_indices_access(b: &mut Bencher) {
    let mut iter = Zipf::indices_access(1..1000, 1.07).unwrap();
    b.iter(|| iter.next());
}

#[bench]
fn bench_zipf_array_access(b: &mut Bencher) {
    let arr = (1..1000).collect::<Vec<_>>();
    let mut iter = Zipf::array_access(1, arr, 1.07).unwrap();
    b.iter(|| iter.next());
}

#[bench]
fn bench_zipf_batch_16(b: &mut Bencher) {
    let zipf = Zipf::new(1.0..1000000.0, 1.07).unwrap();
    let u_values = vec![
        0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75,
    ];
    let mut output = vec![0.0; 16];
    b.iter(|| zipf.sample_batch(&u_values, &mut output));
}

#[bench]
fn bench_zipf_batch_64(b: &mut Bencher) {
    let zipf = Zipf::new(1.0..1000000.0, 1.07).unwrap();
    let u_values: Vec<f64> = (0..64).map(|i| (i as f64) / 64.0).collect();
    let mut output = vec![0.0; 64];
    b.iter(|| zipf.sample_batch(&u_values, &mut output));
}
