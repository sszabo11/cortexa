/// Different learning methods of network
#[derive(Debug, Default)]
pub enum Learning {
    #[default]
    Hebbian,
}

#[derive(Debug, Default)]
pub enum Activation {
    /// Threshold value
    Linear(f32),
    ReLU,
    #[default]
    Tanh,
    Sigmoid,
    Softmax,
}
