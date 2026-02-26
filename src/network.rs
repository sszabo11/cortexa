use std::{fmt::Debug, process::Output, thread::current};

use anyhow::Result;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, ArrayViewMut1, Axis, Slice, s};
use ndarray_rand::{RandomExt, rand_distr::Uniform};
use num_traits::Float;
use rand::rng;

use crate::{
    forward::Forward,
    learning::{Activation, Learning},
    utils::get_dataset,
};

/// Network struct
#[derive(Default, Debug)]
pub struct Network {
    total_neurons: usize,
    total_connections: usize,
    conns_per_layer: Vec<usize>,
    num_layers: usize,

    // Neuron options
    pub neurons: Array1<f32>,
    /// Neuron values at run time

    /// History of neuron values
    pub history: Array2<f32>,

    /// Idxs of neurons connected to neuron i
    pub conns: Array2<usize>,
    layer_offsets: Array1<usize>,

    /// Permenance values of neuron connection
    pub weights: Array2<f32>,
}

impl Network {
    pub fn builder() -> NetworkBuilder {
        NetworkBuilder::default()
    }

    pub fn set_input(&mut self, input: Array1<f32>) {
        assert!(
            self.layer_offsets[0] == input.len(),
            "Input dim is not equal to first layer dim."
        );

        for (idx, &v) in input.iter().enumerate() {
            self.neurons[idx] = v;
        }
    }
}

#[derive(Default, Debug)]
pub struct NetworkBuilder {
    total_neurons: usize,
    total_connections: usize,
    conns_per_layer: Vec<usize>,
    num_layers: usize,
    pub learning: Learning,
    pub activation: Activation,

    /// History of neuron values
    pub history: Array2<f32>,
    pub neurons: Array1<f32>,
    /// Idxs of neurons connected to neuron i
    pub conns: Array2<usize>,

    /// Store num of neurons per layer, so we can get the neurons in each layer
    layer_offsets: Array1<usize>,
    /// Permenance values of neuron connection
    pub weights: Array2<f32>,
}

impl NetworkBuilder {
    pub fn new() -> NetworkBuilder {
        Self::default()
    }

    pub fn neurons(mut self, num_neurons: usize) -> NetworkBuilder {
        self.total_neurons = num_neurons;
        self
    }
    pub fn activation(mut self, activation: Activation) -> NetworkBuilder {
        self.activation = activation;
        self
    }

    pub fn learning(mut self, learning: Learning) -> NetworkBuilder {
        self.learning = learning;
        self
    }
    pub fn layers(mut self, layers: &[usize]) -> NetworkBuilder {
        self.num_layers = layers.len();
        self.total_neurons = layers.iter().sum();
        self.layer_offsets = Array1::from_vec(layers.to_vec());
        self
    }

    pub fn total_connections(mut self, num_connections: usize) -> NetworkBuilder {
        assert!(
            self.num_layers != 0,
            "Please specify number of layers before setting connections."
        );
        self.total_connections = num_connections;

        self
    }

    pub fn connections(mut self, connections: &[usize]) -> NetworkBuilder {
        assert!(
            self.num_layers != 0,
            "Please specify number of layers before setting connections."
        );
        self.total_connections = connections.iter().sum();
        self.conns_per_layer = connections.to_vec();

        self
    }

    //pub fn with_neuron_data(mut self, neurons: Array1<f32>) -> NetworkBuilder {
    //    self.neurons = neurons;
    //    self
    //}

    //pub fn with_conns(mut self, conns: Vec<Array2<usize>>) -> NetworkBuilder {
    //    self.conns = conns;
    //    self
    //}

    //pub fn with_weights(mut self, weights: Vec<Array2<f32>>) -> NetworkBuilder {
    //    self.weights = weights;
    //    self
    //}

    fn build_connections(&mut self) {
        //self.conns = Array2::random(
        //    (self.num_neurons, self.num_connections),
        //    Uniform::new(0, self.num_neurons - 1).unwrap(),
        //);

        let mut c = self.conns_per_layer.clone();
        c.sort_by(|a, b| b.partial_cmp(a).unwrap());
        println!("max: {}", c[0]);
        self.conns = Array2::random(
            (self.total_neurons, c[0]),
            Uniform::new(0, c[0] - 1).unwrap(),
        );
        self.weights = Array2::random((self.total_neurons, c[0]), Uniform::new(-1.0, 1.0).unwrap());
    }
    pub fn build(mut self) -> Network {
        self.neurons = Array1::zeros(self.total_neurons);
        self.build_connections();

        println!("{:?}", self);
        Network {
            total_connections: self.total_connections,
            num_layers: self.num_layers,
            conns_per_layer: self.conns_per_layer,
            conns: self.conns,
            history: self.history,
            layer_offsets: self.layer_offsets,
            weights: self.weights,
            total_neurons: self.total_neurons,
            neurons: self.neurons,
        }
    }
}

impl Forward for Network {
    type Output = Array1<f32>;

    fn forward(&mut self) -> Self::Output {
        println!("Layer offsets: {:?}", self.layer_offsets);
        println!("N len: {}", self.neurons.len());

        let mut offset = 0;
        for j in 1..self.num_layers {
            println!("Layer: {}\n Offset: {}", j, offset);
            let prev_start = offset;
            let prev_end = offset + self.layer_offsets[j - 1];
            let curr_start = prev_end;
            let curr_end = prev_end + self.layer_offsets[j];

            let (prev_half, mut curr_half) = self.neurons.view_mut().split_at(Axis(0), curr_start);

            let prev_layer = prev_half.slice(s![prev_start..prev_end]);

            let mut curr_layer = curr_half.slice_mut(s![0..curr_end - curr_start]);

            println!(
                "Prev start: {} | Prev end: {} | Curr start: {} | Curr end: {}",
                prev_start, prev_end, curr_start, curr_end
            );
            let weights = self.weights.slice(s![curr_start..curr_end, ..]);
            for i in 0..curr_layer.len() {
                let connections = self.conns.slice(s![curr_start + i, ..]);

                let mut sum = 0.0f32;
                for (c, &idx) in connections.iter().enumerate() {
                    sum += prev_layer[idx] * weights[[i, c]];
                }
                curr_layer[i] = sum / self.total_connections as f32;
            }
            offset += prev_end - 1
        }
        println!("D:: {}", offset);
        self.neurons
            .slice(s![offset + self.layer_offsets[self.num_layers - 1]..])
            .to_owned()
    }
}

#[test]
fn simple_network() {
    let mut network: Network = Network::builder()
        //.neurons(100)
        .layers(&[3, 4, 2])
        //.connections(20)
        .connections(&[2, 3, 1])
        .learning(Learning::Hebbian)
        .activation(Activation::Tanh)
        .build();

    for (layer, &offset) in network.layer_offsets.iter().enumerate() {
        for i in 0..offset {
            network.neurons[(layer * i) + i] = 1.0;
        }
    }
    println!("N: {}", network.neurons);
    let input = get_dataset(3);

    network.set_input(input);

    let output = network.forward();

    println!("Output: {}", output);
}
