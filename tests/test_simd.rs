use distances::Number;
use symagen::random_data;
use test_case::test_case;

use distances::{
    simd,
    vectors::{cosine, cosine_normed, euclidean, euclidean_sq},
};

// TODO this gets refactored into SyMaGen
fn normalize<T: Number>(vals: &[T]) -> Vec<T> {
    let magnitude = vals.iter().cloned().sum();
    vals.iter().map(|&v| v / magnitude).collect()
}

#[test_case(cosine_normed, simd::cosine_normed_f32, 1_f32; "cosine_normed_f32")]
fn simd_distances_normed_f32(
    naive: fn(&[f32], &[f32]) -> f32,
    simd: fn(&[f32], &[f32]) -> f32,
    limit: f32,
) {
    let (cardinality, dimensionality) = (100, 2_usize.pow(12));

    let limit = limit.abs();
    let (min_val, max_val) = (-limit, limit);

    let mut rng = rand::thread_rng();

    let data_x: Vec<Vec<f32>> =
        random_data::random_tabular(cardinality, dimensionality, min_val, max_val, &mut rng)
            .iter()
            .map(|v| normalize(v))
            .collect();
    let data_y: Vec<Vec<f32>> =
        random_data::random_tabular(cardinality, dimensionality, min_val, max_val, &mut rng)
            .iter()
            .map(|v| normalize(v))
            .collect();

    let mut failures = Vec::new();

    for (i, x) in data_x.iter().enumerate() {
        for (j, y) in data_y.iter().enumerate() {
            let expected: f32 = naive(x, y);
            let actual: f32 = simd(x, y);
            let delta = (expected - actual).abs();
            let threshold = 1e-5 * actual;
            if delta > threshold {
                failures.push((i, j, delta, threshold));
            }
        }
    }

    assert!(
        failures.is_empty(),
        "{} non-empty failures, {:?} ...",
        failures.len(),
        &failures[..5]
    );
}

#[test_case(cosine_normed, simd::cosine_normed_f64, 1_f64; "cosine_normed_f64")]
fn simd_distances_normed_f64(
    naive: fn(&[f64], &[f64]) -> f64,
    simd: fn(&[f64], &[f64]) -> f64,
    limit: f64,
) {
    let (cardinality, dimensionality) = (100, 2_usize.pow(12));

    let limit = limit.abs();
    let (min_val, max_val) = (-limit, limit);

    let mut rng = rand::thread_rng();

    let data_x: Vec<Vec<f64>> =
        random_data::random_tabular(cardinality, dimensionality, min_val, max_val, &mut rng)
            .iter()
            .map(|v| normalize(v))
            .collect();
    let data_y: Vec<Vec<f64>> =
        random_data::random_tabular(cardinality, dimensionality, min_val, max_val, &mut rng)
            .iter()
            .map(|v| normalize(v))
            .collect();
    let mut failures = Vec::new();

    for (i, x) in data_x.iter().enumerate() {
        for (j, y) in data_y.iter().enumerate() {
            let expected: f64 = naive(x, y);
            let actual: f64 = simd(x, y);
            let delta = (expected - actual).abs();
            let threshold = 1e-10 * actual;
            if delta > threshold {
                failures.push((i, j, delta, threshold));
            }
        }
    }

    assert!(
        failures.is_empty(),
        "{} non-empty failures, {:?} ...",
        failures.len(),
        &failures[..5]
    );
}

#[test_case(euclidean_sq, simd::euclidean_sq_f32, 10_f32; "euclidean_sq_f32")]
#[test_case(euclidean, simd::euclidean_f32, 10_f32; "euclidean_f32")]
#[test_case(cosine, simd::cosine_f32, 1_f32; "cosine_f32")]
fn simd_distances_f32(
    naive: fn(&[f32], &[f32]) -> f32,
    simd: fn(&[f32], &[f32]) -> f32,
    limit: f32,
) {
    let (cardinality, dimensionality) = (100, 2_usize.pow(12));

    let limit = limit.abs();
    let (min_val, max_val) = (-limit, limit);

    let mut rng = rand::thread_rng();

    let data_x =
        random_data::random_tabular(cardinality, dimensionality, min_val, max_val, &mut rng);
    let data_y =
        random_data::random_tabular(cardinality, dimensionality, min_val, max_val, &mut rng);
    let mut failures = Vec::new();

    for (i, x) in data_x.iter().enumerate() {
        for (j, y) in data_y.iter().enumerate() {
            let expected = naive(x, y);
            let actual = simd(x, y);
            let delta = (expected - actual).abs();
            let threshold = f32::EPSILON.sqrt() * actual;
            if delta > threshold {
                failures.push((i, j, delta, threshold, actual, expected));
            }
        }
    }

    assert!(
        failures.is_empty(),
        "{} non-empty failures, {:?} ...",
        failures.len(),
        &failures[..5]
    );
}

#[test_case(euclidean_sq, simd::euclidean_sq_f64, 10_f64; "euclidean_sq_f64")]
#[test_case(euclidean, simd::euclidean_f64, 10_f64; "euclidean_f64")]
#[test_case(cosine, simd::cosine_f64, 1_f64; "cosine_f64")]
fn simd_distances_f64(
    naive: fn(&[f64], &[f64]) -> f64,
    simd: fn(&[f64], &[f64]) -> f64,
    limit: f64,
) {
    let (cardinality, dimensionality) = (100, 2_usize.pow(12));

    let limit = limit.abs();
    let (min_val, max_val) = (-limit, limit);

    let mut rng = rand::thread_rng();

    let data_x =
        random_data::random_tabular(cardinality, dimensionality, min_val, max_val, &mut rng);
    let data_y =
        random_data::random_tabular(cardinality, dimensionality, min_val, max_val, &mut rng);
    let mut failures = Vec::new();

    for (i, x) in data_x.iter().enumerate() {
        for (j, y) in data_y.iter().enumerate() {
            let expected = naive(x, y);
            let actual = simd(x, y);
            let delta = (expected - actual).abs();
            let threshold = f64::EPSILON.sqrt() * actual;
            if delta > threshold {
                failures.push((i, j, delta, threshold, actual, expected));
            }
        }
    }

    assert!(
        failures.is_empty(),
        "{} non-empty failures, {:?} ...",
        failures.len(),
        &failures[..5]
    );
}
