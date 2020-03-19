import numpy as np
from ..activation import *
from ..optimizer import *
from ..layer import Layer

class DenseLayer(Layer):
    def __init__(self, h_0, h_1, eta, activation, optimizer):
        '''
        Initialization of the layer, with h_0 as the number of units in the 
        previous layer and h_1 as the number of units in the current layer.
        
        Arguments:
        ---------
        
        h_0         {int}   :   Number of units in previous layer
        h_1         {int}   :   Number of units in current layer
        activation  {str}   :   Which activation function to use
        init        {str}   :   How to initialize the weights of the layer
        '''
        self.h_0 = h_0 + 1      # Adding bias node
        self.h_1 = h_1
        self.eta = eta
        self.get_activation_function(activation)
        self.get_optimizer(optimizer)
        
        
    def initialize(self, init):
        if init.lower() == "normal":
            self.W = np.random.normal(0, 0.1, (self.h_0, self.h_1))
        elif init.lower() == "uniform":
            self.W = 2 * np.random.uniform(size=(self.h_0, self.h_1)) - 1
        else:
            raise NotImplementedError("Function {} is not implemented".format(init))
        return self.W
        
    def get_activation_function(self, activation):
        if type(activation) is not str:
            raise TypeError("Activation function needs to be given as a string")
        elif activation.lower() == "sigmoid":
            self.activation = Sigmoid()
        elif activation.lower() == "relu":
            self.activation = ReLU()
        elif activation.lower() == "leakyrelu":
            self.activation = LeakyReLU()
        elif activation.lower() == "elu":
            self.activation = ELU()
        else:
            raise NotImplementedError("Function {} is not implemented".format(activation))
        return self.activation
        
    def get_optimizer(self, optimizer):
        if type(optimizer) is not str:
            raise TypeError("Optimizer needs to be given as a string")
        elif optimizer.lower() == 'gd':
            self.opt = GradientDescent(self.eta, self.h_0, self.h_1)
        elif optimizer.lower() == 'adam':
            self.opt = ADAM(self.eta, self.h_0, self.h_1)
        else:
            raise NotImplementedError("Optimizer {} is not implemented".format(optimizer))
        return self.opt
        
    def forward(self, input_layer):
        """Forward propagation. Multiply input_layer with weight matrix. 
        
        Parameters
        ----------
        input_layer : ndarray
            Output from previous layer.
        """
        input_layer = np.c_[input_layer, np.ones((len(input_layer),1))]       # Add bias nodes
        return input_layer.dot(self.W)
        
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
        
    def deractivate(self):
        self.df = self.activation.derivate(self.z)
        return self.df
        
    def calculate_delta(self, dcost):
        self.deractivate()
        self.delta = dcost * self.df
        return self.delta
        
    def calculate_gradient(self):
        self.dC = self.a_0.T.dot(self.delta)
        return self.dC
        
    def update_weights(self, i):
        self.calculate_gradient()
        self.W -= self.opt.update_parameters(i+1, self.dC)
        return self.W
