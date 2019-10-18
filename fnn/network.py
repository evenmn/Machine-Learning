import numpy as np
from layer import Layer
from cost import *

class Network:
    def __init__(self, input_units, cost=MSE()):
        '''
        Initialize network, including weights and nodes
        
        Arguments:
        ---------
        
        input_units     {int}   :   Number of input units
        cost            {string}:   Cost function
        '''
        
        self.cost = cost
        self.layers = []
        self.h = np.array([input_units])
        self.W = []
        self.a = [np.zeros(input_units)]
        
    def add(self, units, activation="sigmoid", init='normal'):
        '''
        Add dense layer
        
        Arguments:
        ---------
        
        units           {int}       : Number of units in layer
        activation      {obj}       : Activation function used in layer
        init            {string}    : Initialization of layer
        '''
        
        self.h = np.append(self.h, units)
        self.a.append(np.zeros(units))
        self.layers.append(Layer(self.h[-2], self.h[-1], activation))
        W = self.layers[-1].initialize(init)
        self.W.append(W)
            
    def predict(self, x):
        ''' Predicting output from network, given a input data set x '''
        self.a[0] = np.array(x)
        for i, layer in enumerate(self.layers):
            self.a[i+1] = layer.activate(self.a[i])
        return self.a[-1]
        
    def mse(self, t):
        ''' Mean-square error, given targets t '''
        error = MSE()
        return error.evaluate(self.a[-1], t)
        
    def cost(self, t):
        ''' Cost error, given targets t '''
        return self.cost.evaluate(self.a[-1], t)
        
    def backprop(self, t):
        ''' Back-propagation processing based on targets t '''
        dcost = self.cost.derivate(self.a[-1], t)
        for i, layer in reversed(list(enumerate(self.layers))):
            delta = layer.calculate_delta(dcost)
            dcost = delta.dot(self.W[i].T)[:,:-1]
            
    def update(self, optimizer='GD', learning_rate=0.01):
        ''' Update weights '''
        for i, layer in enumerate(self.layers):
            self.W[i] = layer.update_weights(i, learning_rate, optimizer)
            
    def simulate(self, x, t, optimizer='GD', max_iter=100000, learning_rate=0.1):
        ''' Run a simulation '''
        self.predict(x)
        for i in range(max_iter):
            self.backprop(t)
            self.update(optimizer, learning_rate)
            self.predict(x)
            self.print_to_terminal(i, t)
        return self.a[-1]
        
    def print_to_terminal(self, i, t):
        print(10 * "-" + " " + str(i+1) + " " + 10 * "-")
        print(self.mse(t))
        print(25 * "-")
        print(" ")
            
            
if __name__ == "__main__":
    
    train_d = [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
               [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
               [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
               [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]]
            
    train_t = [[0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
               [1, 1, 0, 1, 1, 1, 1, 1, 1, 1],
               [1, 1, 1, 1, 1, 1, 1, 1, 0, 1],
               [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
               [1, 1, 1, 1, 1, 1, 1, 0, 1, 1],
               [1, 1, 1, 0, 1, 1, 1, 1, 1, 1],
               [1, 1, 1, 1, 1, 1, 0, 1, 1, 1],
               [1, 1, 1, 1, 1, 0, 1, 1, 1, 1]]
               
    test_d = [[0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
              [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]]
              
    test_t = [[1, 1, 1, 1, 0, 1, 1, 1, 1, 1],
              [1, 0, 1, 1, 1, 1, 1, 1, 1, 1]]

    NET = Network(input_units=len(train_d[0]))
    NET.add(128)
    NET.add(256, activation="relu")
    NET.add(len(train_t[0]))
    
    NET.simulate(train_d, train_t)
    #print(NET.predict(train_d))
    #print(NET.predict(test_d))
    
