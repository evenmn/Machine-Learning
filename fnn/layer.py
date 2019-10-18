import numpy as np
from activation import *

class Layer:
    def __init__(self, h_0, h_1, activation):
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
        self.h_0 = h_0 + 1
        self.h_1 = h_1
        self.get_activation_function(activation)
        
        
    def initialize(self, init):
        if init.lower() == "normal":
            self.W = np.random.normal(0, 0.1, (self.h_0, self.h_1))
        if init.lower() == "uniform":
            self.W = np.random.uniform((self.h_0, self.h_1))
        return self.W
        
    def get_activation_function(self, activation):
        #if callable(self.activation):
        #    pass
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
        
    def evaluate(self):
        self.a_0 = np.c_[self.a_0, np.ones((len(self.a_0),1))]       # Add bias nodes
        self.z = self.a_0.dot(self.W)
        return self.z
        
    def activate(self, a_0):
        self.a_0 = a_0
        self.evaluate()
        self.a = self.activation.evaluate(self.z)
        return self.a
        
    def deractivate(self):
        self.df = self.activation.derivate(self.z) #self.a * (1 - self.a)
        return self.df
        
    def calculate_delta(self, dcost):
        self.deractivate()
        self.delta = dcost * self.df
        return self.delta
        
    def calculate_gradient(self):
        self.dC = self.a_0.T.dot(self.delta)
        return self.dC
        
    def update_weights(self, i, learning_rate, optimizer):
        self.calculate_gradient()
        if optimizer.lower() == "gd":
            self.W -= learning_rate * self.dC
        elif optimizer.lower() == "adam":
            m = np.zeros_like(self.dC)
            v = np.zeros_like(self.dC)
            
        return self.W
