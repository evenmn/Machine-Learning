# TensorNet
TensorNet is a machine learning package written in Python. The interface is inspired by Keras. The training is done in parallel.

## Installation
First download the contents:
``` bash
$ git clone https://github.com/evenmn/machine-learning.git
```
and then install tensornet:
``` bash
$ cd machine-learning
$ pip install .
```

## Key features
TensorNet currently support fully connected dense neural networks
``` python
from tensornet import Network
from tensornet.activation import ReLU, Sigmoid
from tensornet.optimizer import ADAM
from tensornet.cost import MSE
from tensornet.initialize import Normal

model = Network(input_shape=(1), cost=MSE(), optimizer=ADAM(eta=0.005)) 
model.dense(units=64, activation=ReLU())
model.dense(units=32, activation=ReLU())
model.dense(units=1, activation=Sigmoid())
```

After defining the training set and the targets, the network can be trained using
``` python
model.train(input_data, targets, max_iter=100)
```

The model is called directly to predict outputs
``` python
predicted = model(input_data)
```


