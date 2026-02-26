use anyhow::{Result, anyhow};
use ndarray::Array1;

pub trait Forward {
    type Output: ?Sized;

    fn forward(&mut self) -> Self::Output;

    fn forward_step<T>(&self, input: Array1<f32>) -> Result<T> {
        Err(anyhow!("Not implemented"))
    }
}
