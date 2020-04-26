""" Example: Fit a dense neural network to a function. This example uses a
Gaussian function, but other functions can easily be fitted as well 
(also functions of multiple inputs)
"""

from tensornet import Network
from tensornet.activation import ReLU, Sigmoid
from tensornet.optimizer import ADAM, GradientDescent
from tensornet.cost import MSE
from tensornet.initialize import Normal

# Adjustable parameters
hidden_nodes = 25       # Number of hidden nodes
size_data = 500        # Size of training data
epochs = 200           # Number of epochs

# Set up model
model = Network(input_shape=(1), cost=MSE(), optimizer=ADAM(eta=0.005)) 
model.dense(units=hidden_nodes, activation=ReLU())
model.dense(units=1, activation=Sigmoid())

# Define training data
import numpy as np
def f(coordinates):
    """ Function that should be approximated. Here a Gaussian function.
    """
    return np.exp(-(coordinates**2).sum(axis=1)/2)
    
    
x = np.random.normal(size=(size_data, 1))
t = f(x)

# Train model
model.train(x, t, epochs = epochs)

# Plot result
import matplotlib.pyplot as plt
plt.style.use("bmh")

x_vals = np.linspace(-5, 5, 100).reshape(100,1)

Y = f(x_vals)
Y_pred = model(x_vals)

plt.plot(x_vals, Y, label="Expected")
plt.plot(x_vals, Y_pred, label="Predicted")
plt.legend(loc='best')
plt.show()
