use std::{f64::consts::E, sync::Mutex};

use nalgebra::{DMatrix, DVector};
use once_cell::sync::Lazy;
use rand::{rngs::StdRng, seq::SliceRandom, Rng, SeedableRng};
use rand_distr::StandardNormal;

pub type Vector = DVector<f64>;

pub type Matrix = DMatrix<f64>;

pub trait Arith<T> {
    #[must_use]
    fn sqrt(&self) -> T;

    #[must_use]
    fn pow2(&self, exp: i32) -> T;

    #[must_use]
    fn div2(&self, rhs: &T) -> T;
}

impl Arith<Matrix> for Matrix {
    fn sqrt(&self) -> Matrix {
        let mut r = Matrix::zeros(self.nrows(), self.ncols());
        for (i, v) in self.iter().enumerate() {
            r[i] = v.sqrt();
        }
        r
    }

    fn pow2(&self, exp: i32) -> Matrix {
        // gemm: dimensions mismatch for multiplication.
        // self.pow(exp as u32)

        let mut r = Matrix::zeros(self.nrows(), self.ncols());
        for (i, v) in self.iter().enumerate() {
            r[i] = v.powi(exp);
        }
        r
    }

    fn div2(&self, rhs: &Matrix) -> Matrix {
        let mut r = Matrix::zeros(self.nrows(), self.ncols());
        for (i, v) in self.iter().enumerate() {
            r[i] = v / rhs[i];
        }
        r
    }
}

impl Arith<Vector> for Vector {
    fn sqrt(&self) -> Vector {
        let mut r = Vector::zeros(self.nrows());
        for (i, v) in self.iter().enumerate() {
            r[i] = v.sqrt();
        }
        r
    }

    fn pow2(&self, exp: i32) -> Vector {
        let mut r = Vector::zeros(self.nrows());
        for (i, v) in self.iter().enumerate() {
            r[i] = v.powi(exp);
        }
        r
    }

    fn div2(&self, rhs: &Vector) -> Vector {
        let mut r = Vector::zeros(self.nrows());
        for (i, v) in self.iter().enumerate() {
            r[i] = v / rhs[i];
        }
        r
    }
}

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

static RNG: Lazy<Mutex<StdRng>> = Lazy::new(|| Mutex::new(StdRng::seed_from_u64(42)));

pub fn rnd_normal() -> f64 {
    // thread_rng().sample(StandardNormal)

    let mut rng = RNG.lock().unwrap();
    rng.sample(StandardNormal)
}

pub fn shuffle_vec<T>(vec: &mut Vec<T>) {
    let mut rng = RNG.lock().unwrap();
    vec.shuffle(&mut *rng)
}
