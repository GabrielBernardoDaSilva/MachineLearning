use std::path::PathBuf;

use rand::Rng;

use serde_derive::{Deserialize, Serialize};

use crate::ml::tanh;

use super::{create_filename, dtanh};

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Neruon {
    output: f64,
    weigths: Vec<f64>,
    delta: f64,
}

type Layer = Vec<Neruon>;

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct NeuralNetwork {
    input_layer: Vec<f64>,
    hidden_layer: Vec<Layer>,
    output_layer: Layer,
    topology: Vec<usize>,
    error_total: f64,
    bais: usize,
    learning_rate: f64,
    momentum: f64,
}

impl Neruon {
    pub fn new(connections: usize) -> Self {
        let mut weigths: Vec<f64> = Vec::new();
        let mut generator = rand::thread_rng();
        for _ in 0..connections {
            weigths.push(generator.gen_range(-1.0..1.0));
        }
        Self {
            output: 0.0,
            weigths,
            delta: 0.0,
        }
    }
}

impl NeuralNetwork {
    pub fn new(topology: Vec<usize>, bais: usize, lr: f64, alpha: f64) -> Self {
        let mut hidden_layers = Vec::new();
        for i in 1..topology.len() - 1 {
            let mut layer = Vec::new();
            for _ in 0..topology[i] + bais {
                layer.push(Neruon::new(topology[i - 1] + bais));
            }
            layer.last_mut().unwrap().output = 1.0;
            hidden_layers.push(layer);
        }
        let size = topology.len();
        let output_layer = vec![Neruon::new(topology[size - 2] + bais); topology[size - 1]];
        let mut input_layer = vec![0.0; topology[0]];
        for _ in 0..bais {
            input_layer.push(1.0);
        }

        Self {
            input_layer,
            hidden_layer: hidden_layers,
            output_layer,
            topology: topology,
            error_total: 0.0,
            bais,
            learning_rate: lr,
            momentum: alpha,
        }
    }

    pub fn save_neural_network(&self, name: Option<&str>) {
        let filename = PathBuf::from(format!(
            "./trained/nn/{}.json",
            if let Some(n) = name {
                n.to_string()
            } else {
                create_filename("neural_network")
            }
        ));

        std::fs::create_dir_all("./trained/nn").unwrap();
        let content = serde_json::to_string_pretty(self).unwrap();
        std::fs::write(filename, content).unwrap()
    }

    pub fn load_neural_network(name: &str) -> Self {
        let content = std::fs::read_to_string(PathBuf::from(name)).unwrap();
        let nn = serde_json::from_str::<NeuralNetwork>(&content).unwrap();
        nn
    }

    pub fn train(&mut self, epochs: usize, inputs: Vec<Vec<f64>>, outputs: Vec<Vec<f64>>) {
        for epoch in 0..epochs {
            for index in 0..inputs.len() {
                self.feed_forward(&inputs[index]);
                self.back_propagation(&outputs[index]);
            }

            self.error_total /= outputs.len() as f64;
            println!("Epoch: {}, Error rate: {}", epoch, self.error_total);
            self.error_total = 0.0;
        }
    }
    pub fn predict(&mut self, input: &Vec<f64>) -> Vec<f64> {
        self.feed_forward(input);
        self.output_layer
            .clone()
            .into_iter()
            .map(|item| item.output)
            .collect()
    }

    pub fn feed_forward(&mut self, input: &Vec<f64>) {
        assert_eq!(input.len(), self.input_layer.len() - self.bais);

        for i in 0..input.len() {
            self.input_layer[i] = input[i];
        }

        // input layer and first layer
        for neuron in self.hidden_layer.first_mut().unwrap() {
            let mut sum = 0.0;
            for i in 0..self.input_layer.len() {
                sum += self.input_layer[i] * neuron.weigths[i];
            }
            neuron.output = tanh(sum);
        }

        let first_size = self.hidden_layer.first_mut().unwrap().len();
        for i in (first_size - self.bais)..first_size {
            self.hidden_layer[0][i].output = 1.0;
        }

        // others hidden layers
        for i in 1..self.hidden_layer.len() {
            for j in 0..self.hidden_layer[i].len() - self.bais {
                let mut sum = 0.0;
                for k in 0..self.hidden_layer[i - 1].len() {
                    sum += self.hidden_layer[i - 1][k].output * self.hidden_layer[i][j].weigths[k];
                }
                self.hidden_layer[i][j].output = tanh(sum);
            }
        }

        //output layer
        for neuron in &mut self.output_layer {
            let mut sum = 0.0;
            for i in 0..self.hidden_layer.last().unwrap().len() {
                sum += self.hidden_layer.last().unwrap()[i].output * neuron.weigths[i];
            }
            neuron.output = tanh(sum);
        }
    }

    pub fn back_propagation(&mut self, outputs: &Vec<f64>) {
        for i in 0..outputs.len() {
            let error = outputs[i] - self.output_layer[i].output;
            self.error_total += error.abs();

            self.output_layer[i].delta = error * dtanh(self.output_layer[i].output);

            let last_hidden_layer = self.hidden_layer.last_mut().unwrap();
            let neuron = &self.output_layer[i];
            for j in 0..neuron.weigths.len() {
                let delta_weight = self.output_layer[i].delta * self.output_layer[i].weigths[j];
                let delta_hidden = dtanh(last_hidden_layer[j].output) * delta_weight;
                last_hidden_layer[j].delta = delta_hidden;
            }
        }

        for i in (0..self.hidden_layer.len() - 1).rev() {
            for j in 0..self.hidden_layer[i].len() {
                for k in 0..self.hidden_layer[i + i][j].weigths.len() {
                    let delta_weight =
                        self.hidden_layer[i + i][j].delta * self.hidden_layer[i + i][j].weigths[k];
                    let delta_hidden = dtanh(self.hidden_layer[i][j].output) * delta_weight;
                    self.hidden_layer[i][j].delta = delta_hidden;
                }
            }
        }

        for i in 0..self.output_layer.len() {
            for j in 0..self.output_layer[i].weigths.len() {
                let last_hidden_layer = self.hidden_layer.last().unwrap();

                let new_weight = last_hidden_layer[j].output * self.output_layer[i].delta;
                let m_weight = self.output_layer[i].weigths[j] * self.momentum;
                let l_weight = new_weight * self.learning_rate;

                self.output_layer[i].weigths[j] = (m_weight) + (l_weight);
            }
        }

        for i in (1..self.hidden_layer.len() - 1).rev() {
            for j in 0..self.hidden_layer[i].len() {
                for k in 0..self.hidden_layer[i][j].weigths.len() {
                    let new_weight =
                        self.hidden_layer[i - 1][k].output * self.hidden_layer[i][j].delta;
                    self.hidden_layer[i][j].weigths[k] = (self.hidden_layer[i][j].weigths[k]
                        * self.momentum)
                        + (new_weight * self.learning_rate);
                }
            }
        }

        for neuron in self.hidden_layer.first_mut().unwrap() {
            for i in 0..neuron.weigths.len() {
                let new_weight = self.input_layer[i] * neuron.delta;
                neuron.weigths[i] =
                    (neuron.weigths[i] * self.momentum) + (new_weight * self.learning_rate);
            }
        }
    }
}
