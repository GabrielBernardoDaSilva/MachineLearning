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
let mut nn = NeuralNetwork::new(vec![2, 6, 4, 6, 1], 0, 0.6, 1.0)  
```
- Parameters
    Topology        Vec<usize>
    Bias            usize
    Learning Rate   f64
    Momentum        f64

```
Train:
nn.train(epoch, inputs, outputs);
```
- Parameters
    Epochs          usize
    Inputs          Vec<f64>
    Outputs         Vec<f64>

```
Save:
nn.save_neural_network(Option<&str>);
```

```
Load:
let mut nn = NeuralNetwork::load_neural_network(&str);
```


