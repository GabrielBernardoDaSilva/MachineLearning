mod neural_network;

pub use neural_network::NeuralNetwork;
use rand::Rng;

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + f64::exp(-x))
}
fn dsigmoid(x: f64) -> f64 {
    x * (1.0 - x)
}

fn relu(x: f64) -> f64 {
    if x < 0.0 {
        0.0
    } else {
        x
    }
}

fn drelu(x: f64) -> f64 {
    if x < 0.0 {
        0.0
    } else {
        1.0
    }
}

fn tanh(x: f64) -> f64 {
    (f64::exp(x) - f64::exp(-x)) / (f64::exp(x) + f64::exp(-x))
}

fn dtanh(x: f64) -> f64 {
    1.0 - x * x
}

fn create_filename(default_name: &str) -> String {
    let mut filename = default_name.to_string();
    for _ in 0..10 {
        let mut generator = rand::thread_rng();
        let letter = generator.gen_range(48..=57) as u8;

        filename.push(letter as char)
    }

    filename
}
