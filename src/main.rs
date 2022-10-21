#![allow(dead_code)]

use ml::NeuralNetwork;

mod ml;

fn main() {
    // let mut nn = NeuralNetwork::new(vec![2, 6, 4, 6, 1], 0, 0.6, 1.0);

    let mut nn = NeuralNetwork::load_neural_network("./trained/nn/nn_100_000_000.json");

    // let inputs = vec![
    //     vec![0.0, 0.0],
    //     vec![0.0, 1.0],
    //     vec![1.0, 0.0],
    //     vec![1.0, 1.0],
    // ];

    // let outputs = vec![vec![0.0], vec![1.0], vec![1.0], vec![0.0]];

    println!(
        "Prediction is: {:?}",
        nn.predict(&vec![0.0, 1.0])
            .into_iter()
            .map(|item| item.round())
            .collect::<Vec<f64>>()
    )

    // nn.train(100_000_000, inputs, outputs);
    // nn.save_neural_network(Some("nn_100_000_000"));
}
