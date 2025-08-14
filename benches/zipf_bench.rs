#![feature(test)]
extern crate test;

use test::Bencher;
use load::zipf::Zipf;

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