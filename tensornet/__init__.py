""" Neural network.

@author: Even M. Nordhagen 
"""


import numpy as np

class Network:
    """ Initialize network, including weights and nodes
    
    Parameters
    ----------
    
    input_shape : ndtuple
        dimension on input. Can be 1d (works for fully connected 
        layer only), 2d (image with only 1 channel) and 3d (image
        with multiple channels)
    eta : float
        learning rate
    init : obj
        how to initialize weights. Methods are found in initialize.py
    cost : obj
        cost function. Functions are found in cost.py
    activation : obj
        activation function. Functions are found in activation.py
    optimizer : obj
        optimizer function. Methods are found in optimizer.py
    """

    from tensornet.cost import MSE
    from tensornet.activation import Sigmoid
    from tensornet.optimizer import ADAM
    from tensornet.initialize import Normal
    
    def __init__(self, input_shape, 
                       cost=MSE(), 
                       init=Normal(), 
                       activation=Sigmoid(), 
                       optimizer=ADAM(eta=0.01),
                       bias=True):
        
        self.layers = []
        self.h = np.array([input_shape])
        self.weight = []
        self.a = [np.zeros(input_shape)]
        self.activation = activation
        self.init = init
        self.optimizer = optimizer
        self.cost = cost
        self.bias = bias
        
    def append(self, layer):
        self.layers.append(layer)
        self.weight.append(layer.weight)
        
    def dense(self, units, 
                    init=None, 
                    activation=None,  
                    optimizer=None,
                    bias=None):
        """ Add dense layer.
        
        Parameters
        ----------
        
        units : int
            number of hidden units
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
        if init is None:
            init = self.init
        if activation is None:
            activation = self.activation
        if optimizer is None:
            optimizer = self.optimizer
        if bias is None:
            bias = self.bias
        self.h = np.append(self.h, units)
        self.a.append(np.zeros(units))
        
        from tensornet.layer import DenseLayer
        layer = DenseLayer(self.h[-2], self.h[-1], init, activation, optimizer, bias)
        self.append(layer)
        
    def conv(self, kernel=(3,32,32), 
                   pad_size=(15,15), 
                   stride=(1,1), 
                   init=None,
                   activation=None, 
                   optimizer=None,
                   bias=None):
        """ Add convolutional layer.
        
        Parameters
        ----------
        
        kernel : 3dtuple of ints
            kernel size in vertical and horizontal direction
        pad_size : 2dtuple of ints
            zero padding in horizontal and vertical direction
        stride : 2dtuple of ints
            stride in horizontal and vertical direction 
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
        if init is None:
            init = self.init
        if activation is None:
            activation = self.activation
        if optimizer is None:
            optimizer = self.optimizer
        if bias is None:
            bias = self.bias
            
        from tensornet.layer import ConvLayer
        layer = ConvLayer(kernel, pad_size, stride, init, activation, optimizer, bias)
        self.append(layer)
        
    def pooling(self, kernel=(2,2), pad_size=(0,0), stride=(1,1), mode='max'):
        """ Add pooling layer.
    
        Parameters
        ----------
        kernel : 2dtuple of ints
            kernel size in vertical and horizontal direction. (2,2) by default
        pad_size : 2dtuple of ints
            pad size in vertical and horizontal direction. No padding by default.
        stride : 2dtuple of ints
            stride of pooling (height,width). By default the 
            size of kernel (no overlap)
        mode : str
            mode of pooling. Max pooling ('max'), min pooling
            ('min') and mean pooling ('mean'/'avg') implemented
        """
            
        from tensornet.layer import Pooling
        layer = Pooling(kernel, pad_size, stride, mode)
        self.layers.append(layer)
        self.weight.append(0)
            
    def __call__(self, input_data):
        """ Predicting output from network, given a input data set.
        
        Parameters
        ----------
        input_data : ndarray
            input data needs to match the input shape of model
        """
        a = np.array(input_data)
        for i, layer in enumerate(self.layers):
            a = layer(a)
        self.predicted = a
        return a
        
    @staticmethod
    def mse(predicted, targets):
        """ Mean-square error, given inputs and targets.
        
        Parameters
        ----------
        predicted : ndarray
            input data needs to match the input shape of model
        targets : ndarray
            number of targets need to match number of inputs
            size of target needs to match last layer of model
        """
        mse_cost = MSE()
        return mse_cost(predicted, targets)
        
    @staticmethod
    def score(predicted, targets):
        """ Returns the R2-score.
        """
        u = ((targets - predicted) ** 2).sum()
        v = ((targets - targets.mean()) ** 2).sum()
        return 1 - u/v
        
    def backprop(self):
        """ Back-propagation processing based on some targets
        """
        dcost = self.cost.derivate()
        for i, layer in reversed(list(enumerate(self.layers))):
            dcost = layer.backward(dcost)
            
    def update(self):
        """ Update weights.
        """
        for i, layer in enumerate(self.layers):
            self.weight[i] = layer.update_weights(i)
            
    def train(self, input_data, targets, max_iter=1000):
        """ Train the model. 
        
        Parameters
        ----------
        input_data : ndarray
            input data needs to match the input shape of model
        targets : ndarray
            number of targets need to match number of inputs
            size of target needs to match last layer of model
        max_iter : int
            max number of training interations
        """
        
        for step in range(max_iter):
            predicted = self(input_data)
            cost = self.cost(predicted, targets)
            self.backprop()
            self.update()
            self.print_to_terminal(step, targets)
        return predicted
        
    def print_to_terminal(self, step, targets):
        """ Print information to terminal.
        
        Parameters
        ----------
        step : int
            step number
        targets : ndarray
            number of targets need to match number of inputs
            size of target needs to match last layer of model
        """
        print(10 * "-" + "\t" + str(step+1) + "\t" + 10 * "-")
        print("Cost    : ", self.cost(self.predicted, targets).sum())
        #print("R2-score: ", self.score(self.predicted, targets))
        print(35 * "-")
        print(" ")
        
        
if __name__ == "__main__":
    from tensornet.activation import LeakyReLU, ReLU, Sigmoid
    from tensornet.optimizer import ADAM, GradientDescent
    from tensornet.cost import MSE
    from tensornet.initialize import Normal
    
    # XOR GATE
    data = [[0, 0], [0, 1], [1, 0], [1, 1]]
    targets = [[0], [1], [1], [0]]
    
    model = Network((2), cost=MSE(), activation=LeakyReLU(a=0.2), optimizer=ADAM(eta=0.1), bias=False) 
    model.dense(units=5, optimizer=GradientDescent(eta=0.1), activation=ReLU())
    model.dense(units=1)
    model.train(data, targets, max_iter=1000)
    
    
