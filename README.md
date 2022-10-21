# Machine Learning Lib

Project where i implement many Machine Learning Algorithms.

# Neural Network
- Set your own topology.
- Set how many bias you want.
- Set learning rate.
- Save trained network.
- Load trained network.

Exemplo:
```
Initialize:
let mut nn = NeuralNetwork::new(topology: Vec<usize>, bias: usize, lerning_rate: f64, momentum: f64)  
```

```
Train:
nn.train(epochs: usize, inputs: Vec<Vec<f64>>, outputs: Vec<Vec<f64>>);
```

```
Save:
nn.save_neural_network(name: Option<&str>);
```

```
Load:
let mut nn = NeuralNetwork::load_neural_network(filename: &str);
```


