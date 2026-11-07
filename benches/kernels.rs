use lavan::{Matrix, Vector, matrix, model::Lisrel, mvnorm::Normal, sample::SampleStats};
use serde_json::{Value, json};
use std::{
    hint::black_box,
    time::{Duration, Instant},
};

fn numbers(x: &Value) -> Vec<f64> {
    x.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect()
}
fn matrix(x: &Value) -> Matrix {
    Matrix::from_column_slice(
        x["rows"].as_u64().unwrap() as usize,
        x["cols"].as_u64().unwrap() as usize,
        &numbers(&x["data"]),
    )
}

fn measure<F>(name: &str, mut operation: F) -> Value
where
    F: FnMut() -> f64,
{
    for _ in 0..3 {
        black_box(operation());
    }
    let mut repetitions = 1_usize;
    loop {
        let start = Instant::now();
        for _ in 0..repetitions {
            black_box(operation());
        }
        if start.elapsed() >= Duration::from_millis(60) {
            break;
        }
        repetitions *= 2;
    }
    let mut times = Vec::new();
    let mut checksum = 0.0;
    for _ in 0..7 {
        let start = Instant::now();
        for _ in 0..repetitions {
            checksum = black_box(operation());
        }
        times.push(start.elapsed().as_secs_f64() * 1e6 / repetitions as f64);
    }
    let mut sorted = times.clone();
    sorted.sort_by(f64::total_cmp);
    json!({"name":name,"samples_us":times,"median_us":sorted[3],"repetitions":repetitions,"checksum":checksum})
}

fn main() {
    let path = std::env::args()
        .skip(1)
        .find(|s| s.ends_with(".json"))
        .expect("pass benchmark input JSON");
    let fixture: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let y = matrix(&fixture["y"]);
    let sigma = matrix(&fixture["sigma"]);
    let mu = Vector::from_vec(numbers(&fixture["mu"]));
    let b = matrix(&fixture["b"]);
    let normal = Normal::new(mu.clone(), sigma.clone()).unwrap();
    let mut results = Vec::new();
    results.push(measure("sample_statistics", || {
        SampleStats::from_data(black_box(&y), None)
            .unwrap()
            .covariance()[(0, 0)]
    }));
    results.push(measure("normal_log_likelihood", || {
        Normal::new(mu.clone(), sigma.clone())
            .unwrap()
            .log_likelihood(black_box(&y), None, &[])
            .unwrap()
    }));
    results.push(measure("normal_scores", || {
        Normal::new(mu.clone(), sigma.clone())
            .unwrap()
            .scores(black_box(&y), None, &[], true)
            .unwrap()[(0, 0)]
    }));
    results.push(measure("expected_information", || {
        normal.information_expected(&[], true, false).unwrap()[(0, 0)]
    }));
    results.push(measure("duplication_pre_post", || {
        matrix::duplication_pre_post(black_box(&b), false).unwrap()[(0, 0)]
    }));
    let lambda = matrix(&fixture["lambda"]);
    let psi = matrix(&fixture["psi"]);
    let theta = matrix(&fixture["theta"]);
    let beta = matrix(&fixture["beta"]);
    let mut model = Lisrel::new(lambda, theta, psi).unwrap();
    model.beta = beta;
    results.push(measure("lisrel_implied_covariance", || {
        model.implied().unwrap().covariance[(0, 0)]
    }));
    println!(
        "{}",
        json!({"implementation":"Rust lavan","results":results})
    );
}
