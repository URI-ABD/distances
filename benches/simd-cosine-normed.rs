use criterion::*;
use distances::Number;
use symagen::random_data;

use distances::simd;

use distances::vectors::cosine_normed as cosine_normed_generic;

// TODO this gets refactored into SyMaGen
fn normalize<T: Number>(vals: &[T]) -> Vec<T> {
    let magnitude = vals.iter().cloned().sum();
    vals.iter().map(|&v| v / magnitude).collect()
}

fn simd_f32(c: &mut Criterion) {
    let mut group = c.benchmark_group("SimdCosF32");

    let (cardinality, min_val, max_val) = (2, -10.0, 10.0);

    for d in 0..=5 {
        let dimensionality = 1_000 * 2_u32.pow(d) as usize;
        let vecs: Vec<Vec<f32>> = random_data::random_tabular_seedable(
            cardinality,
            dimensionality,
            min_val,
            max_val,
            d as u64,
        )
        .iter()
        .map(|v| normalize(v))
        .collect();

        let id = BenchmarkId::new("Cosine-normed-generic", dimensionality);
        group.bench_with_input(id, &dimensionality, |b, _| {
            b.iter(|| black_box(cosine_normed_generic::<_, f32>(&vecs[0], &vecs[1])))
        });

        let id = BenchmarkId::new("Cosine-normed-simd", dimensionality);
        group.bench_with_input(id, &dimensionality, |b, _| {
            b.iter(|| black_box(simd::cosine_normed_f32(&vecs[0], &vecs[1])))
        });
    }
    group.finish();
}

fn simd_f64(c: &mut Criterion) {
    let mut group = c.benchmark_group("SimdCosF64");

    let (cardinality, min_val, max_val) = (2, -10.0, 10.0);

    for d in 0..=5 {
        let dimensionality = 1_000 * 2_u32.pow(d) as usize;
        let vecs: Vec<Vec<f64>> = random_data::random_tabular_seedable(
            cardinality,
            dimensionality,
            min_val,
            max_val,
            d as u64,
        )
        .iter()
        .map(|v| normalize(v))
        .collect();

        let id = BenchmarkId::new("Cosine-normed-generic", dimensionality);
        group.bench_with_input(id, &dimensionality, |b, _| {
            b.iter(|| black_box(cosine_normed_generic::<_, f64>(&vecs[0], &vecs[1])))
        });

        let id = BenchmarkId::new("Cosine-normed-simd", dimensionality);
        group.bench_with_input(id, &dimensionality, |b, _| {
            b.iter(|| black_box(simd::cosine_normed_f64(&vecs[0], &vecs[1])))
        });
    }
    group.finish();
}

criterion_group!(benches, simd_f32, simd_f64);
criterion_main!(benches);
