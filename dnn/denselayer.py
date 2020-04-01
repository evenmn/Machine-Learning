import numpy as np
from layer import Layer

class DenseLayer(Layer):
    """ Add dense layer.
    
    Parameters
    ----------
    
    h_0 : int
        number of hidden units in previous layer
    h_1 : int
        number of hidden units in current layer
    eta : float
        learning rate
    init : obj
        how to initialize weights. Methods are found in initialize.py
    activation : obj
        activation function. Functions are found in activation.py
    optimizer : obj
        optimizer function. Methods are found in optimizer.py
    bias : bool
        bias on (True) / off (False)
    """
    def __init__(self, h_0, h_1, eta, init, activation, optimizer, bias):
        self.h_0 = h_0    
        if bias:
            self.h_0 += 1   # Adding bias node
        self.h_1 = h_1
        self.eta = eta
        self.activation = activation
        self.optimizer = optimizer
        self.weight = init(size=(h_0, h_1))
        
    def forward(self, input_layer):
        """Forward propagation. Multiply input_layer with weight matrix. 
        
        Parameters
        ----------
        input_layer : ndarray
            Output from previous layer.
        """
        input_layer = np.c_[input_layer, np.ones((len(input_layer),1))]       # Add bias nodes
        return input_layer.dot(self.weight)
        
    def __call__(self, input_layer):
        """Activation of output from forward propagation. 
        
        Parameters
        ----------
        input_layer : ndarray
            Output from previous layer.
        """
        self.input_layer = input_layer
        z = self.forward(input_layer)
        self.output_layer = self.activation.evaluate(z)
        return self.output_layer
        
    def backward(self, dcost):
        """ Backward propagation. 
        """
        z = self.forward(input_layer)
        df = self.activation.evaluate(z)
        delta = dcost * df
        self.gradient = self.a_0.T.dot(delta)
        
    def update_weights(self, i):
        self.calculate_gradient()
        self.weight -= self.optimizer(i+1, self.gradient)
        return self.weight
