from network import Network
from activation import LeakyReLU, ReLU, Sigmoid
from optimizer import ADAM, GradientDescent
from cost import MSE
from initialize import Normal
import numpy as np

# Dense feedforward neural network
hidden_nodes = 10
inputs = 4
outputs = 1
size_data = 1000

# Set up model
model = Network((inputs), cost=MSE(), optimizer=ADAM(eta=0.1), bias=False) 
model.dense(units=hidden_nodes, activation=ReLU())
model.dense(units=outputs, activation=Sigmoid())

# Define data
def f(coordinates):
    """ Function that should be approximated
    """
    return np.exp(-(coordinates**2).sum(axis=1)/2)
    
x = np.random.normal(size=(size_data, inputs))
t = f(x)

# Train model
model.train(x, t, max_iter=1000)
