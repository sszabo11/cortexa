use ndarray::{Array1, Array2};

/// Network struct
pub struct Network<T> {
    pub num_neurons: usize,
    pub neurons: Vec<Array1<T>>,
}

impl<T> Default for Network<T> {
    fn default() -> Self {
        Self {
            num_neurons: 0,
            neurons: vec![],
        }
    }
}
