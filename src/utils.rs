use ndarray::Array1;
use ndarray_rand::{RandomExt, rand_distr::Uniform};

pub fn get_dataset(dim: usize) -> Array1<f32> {
    Array1::random(dim, Uniform::new(0.0, 1.0).unwrap())
}
