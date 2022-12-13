use std::f64::consts::E;

use nalgebra::{DMatrix, DVector};

pub type Vector = DVector<f64>;

pub type Matrix = DMatrix<f64>;

/// Returns index of maximum item in vector.
pub fn argmax(vec: &[f64]) -> usize {
    let mut max_index = 0;
    for i in 0..vec.len() {
        if vec[i] > vec[max_index] {
            max_index = i;
        }
    }
    max_index
}

/// Produces a [softmax] vector equivalent of the given vector.
///
/// s: ∀ e^zi / Σe^z
///
/// [softmax]: https://en.wikipedia.org/wiki/Softmax_function
pub fn softmax(vec: &Vector) -> Vector {
    let mut result = Vec::with_capacity(vec.len());
    let max_val = vec.max();
    let vec: Vec<_> = vec.iter().map(|v| v - max_val).collect();

    let div: f64 = vec.iter().map(|x| E.powf(*x)).sum();

    for n in vec {
        result.push(E.powf(n) / div);
    }

    Vector::from_vec(result)
}

pub fn small_value() -> f64 {
    E.powf(-7.0)
}
