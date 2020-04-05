import numpy as np
from layer import Layer

class DenseLayer(Layer):
    """ Add dense layer.
    
    Parameters
    ----------
    nodes_prev : int
        number of hidden units in previous layer
    nodes_curr : int
        number of hidden units in current layer
    init : obj
        how to initialize weights. Methods are found in initialize.py
    activation : obj
        activation function. Functions are found in activation.py
    optimizer : obj
        optimizer function. Methods are found in optimizer.py
    bias : bool
        bias on (True) / off (False)
    """
    def __init__(self, nodes_prev, nodes_curr, init, activation, optimizer, bias): 
        self.bias = bias
        if self.bias:
            nodes_prev += 1   # Adding bias node
        self.weight = init(size=(nodes_prev, nodes_curr))
        self.activation = activation
        self.optimizer = optimizer
        
    def forward(self, input_layer):
        """Forward propagation. Multiply input_layer with weight matrix. 
        
        Parameters
        ----------
        input_layer : ndarray
            Output from previous layer.
        """
        if self.bias:
            input_layer = np.c_[input_layer, np.ones((len(input_layer),1))]
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
        self.output_layer = self.activation(z)
        return self.output_layer
        
    def backward(self, dcost):
        """ Backward propagation. 
        """
        df = self.activation.derivate()
        self.delta = dcost * df
        print(dcost.shape)
        print(df.shape)
        print(self.weight.shape)
        #if self.bias:
        dcost_new = np.einsum('ik,jk->ij',self.delta,self.weight)
        print(dcost_new.shape)
        return dcost_new
        
    def update_weights(self, step):
        gradient = self.input_layer.T.dot(self.delta)
        self.weight -= self.optimizer(step+1, gradient)
        return self.weight
