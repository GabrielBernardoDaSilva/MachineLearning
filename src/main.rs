#![allow(unused_imports)]
#![allow(dead_code)]

use ml::NeuralNetwork;
use rand::Rng;

mod ml;

fn main() {
    // let mut nn = NeuralNetwork::new(vec![2, 6, 4, 6, 1], 0, 0.6, 1.0);

    let mut nn = NeuralNetwork::load_neural_network("./trained/nn/neural_network7315646764.json");

    let inputs = vec![
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0],
    ];

    let outputs = vec![vec![0.0], vec![1.0], vec![1.0], vec![0.0]];

    nn.train(1_000_000, inputs, outputs);
    nn.save_neural_network(None);
}
