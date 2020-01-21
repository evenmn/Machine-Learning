import numpy as np
from fnn.layer import Layer
#from cnn.convolution import Convolution
#from cnn.pooling import Pooling
from fnn.cost import *

class Network:
    def __init__(self, input_units, cost='mse', activation="sigmoid", eta=0.01, init='normal', optimizer='adam'):
        '''
        Initialize network, including weights and nodes
        
        Arguments:
        ---------
        
        input_units     {int}       :   Number of input units
        cost            {string}    :   Cost function
        '''
        
        self.layers = []
        self.h = np.array([input_units])
        self.W = []
        self.a = [np.zeros(input_units)]
        if cost.lower() == 'mse':
            self.cost = MSE()
        self.activation = activation
        self.eta = eta
        self.init = init
        self.optimizer = optimizer
        
    def dense(self, units, activation=None, eta=None, init=None, optimizer=None):
        '''
        Add dense layer
        
        Arguments:
        ---------
        
        units           {int}       : Number of units in layer
        activation      {obj}       : Activation function used in layer
        init            {string}    : Initialization of layer
        optimizer       {string}    : Optimizer used in layer
        '''
        if activation is None:
            activation = self.activation
        if eta is None:
            eta = self.eta
        if init is None:
            init = self.init
        if optimizer is None:
            optimizer = self.optimizer
        self.h = np.append(self.h, units)
        self.a.append(np.zeros(units))
        self.layers.append(Layer(self.h[-2], self.h[-1], eta, activation, optimizer))
        W = self.layers[-1].initialize(init)
        self.W.append(W)
        
    def conv(self):
        return None
        
    #def pooling(self, mode='max', window=(2,2)):
    #    self.layers.append(Pooling(mode, window))
            
    def predict(self, x):
        ''' Predicting output from network, given an input data set x '''
        self.data = np.linspace(0,10,1000)
        #self.data = Pooling.pool2d(self, self.data, stride=1, padding=0)
        self.a[0] = np.array(x)
        for i, layer in enumerate(self.layers):
            self.a[i+1] = layer.activate(self.a[i])
        return self.a[-1]
        
    def mse(self, x, t):
        ''' Mean-square error, given targets t '''
        error = MSE()
        return error.evaluate(x, t)
        
    def cost(self, t):
        ''' Cost error, given targets t '''
        return self.cost.evaluate(self.a[-1], t)
        
    def backprop(self, t):
        ''' Back-propagation processing based on targets t '''
        dcost = self.cost.derivate(self.a[-1], t)
        for i, layer in reversed(list(enumerate(self.layers))):
            delta = layer.calculate_delta(dcost)
            dcost = delta.dot(self.W[i].T)[:,:-1]
            
    def update(self):
        ''' Update weights '''
        for i, layer in enumerate(self.layers):
            self.W[i] = layer.update_weights(i)
            
    def simulate(self, x, t, max_iter=1000):
        ''' Run a simulation '''
        self.predict(x)
        from tqdm import tqdm
        for i in tqdm(range(max_iter)):
            self.backprop(t)
            self.update()
            self.predict(x)
            #self.print_to_terminal(i, t)
        return self.a[-1]
        
    def print_to_terminal(self, i, t):
        print(10 * "-" + " " + str(i+1) + " " + 10 * "-")
        print(self.mse(self.a[-1], t))
        print(25 * "-")
        print(" ")
    
