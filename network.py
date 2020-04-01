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

    from cost import MSE
    from activation import Sigmoid
    from optimizer import ADAM
    from initialize import Normal
    
    def __init__(self, input_shape, 
                       eta=0.01, 
                       init=Normal(), 
                       cost=MSE(), 
                       activation=Sigmoid(), 
                       optimizer=ADAM(),
                       bias=True):
        
        self.layers = []
        self.h = np.array([input_shape])
        self.weight = []
        self.a = [np.zeros(input_shape)]
        self.eta = eta
        self.activation = activation
        self.init = init
        self.optimizer = optimizer
        self.cost = cost
        self.bias = bias
        
    def dense(self, units, 
                    eta=None,
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
        if eta is None:
            eta = self.eta
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
        
        from dnn.denselayer import DenseLayer
        layer = DenseLayer(self.h[-2], self.h[-1], eta, init, activation, optimizer, bias)
        self.layers.append(layer)
        self.weight.append(layer.weight)
        
    def conv(self, kernel=(3,32,32), 
                   pad_size=(15,15), 
                   stride=(1,1), 
                   eta=None, 
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
        if eta is None:
            eta = self.eta
        if init is None:
            init = self.init
        if activation is None:
            activation = self.activation
        if optimizer is None:
            optimizer = self.optimizer
        if bias is None:
            bias = self.bias
            
        from cnn.convlayer import ConvLayer
        layer = ConvLayer(kernel, pad_size, stride, eta, init, activation, optimizer, bias)
        self.layers.append(layer)
        self.weight.append(layer.weight)
        
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
            
        from cnn.pooling import Pooling
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
        self.a[0] = np.array(input_data)
        for i, layer in enumerate(self.layers):
            self.a[i+1] = layer(self.a[i])
        return self.a[-1]
        
    def mse(self, input_data, targets):
        """ Mean-square error, given inputs and targets.
        
        Parameters
        ----------
        input_data : ndarray
            input data needs to match the input shape of model
        targets : ndarray
            number of targets need to match number of inputs
            size of target needs to match last layer of model
        """
        error = MSE()
        return error(input_data, targets)
        
    def cost(self, targets):
        """ Cost error, given targets 
        
        Parameters
        ----------
        targets : ndarray
            number of targets need to match number of inputs
            size of target needs to match last layer of model
        """
        return self.cost(self.a[-1], targets)
        
    def backprop(self, targets):
        """ Back-propagation processing based on some targets
        
        Parameters
        ----------
        targets : ndarray
            number of targets need to match number of inputs
            size of target needs to match last layer of model
        """
        dcost = self.cost.derivate(self.a[-1], t)
        for i, layer in reversed(list(enumerate(self.layers))):
            delta = layer.calculate_delta(dcost)
            dcost = delta.dot(self.W[i].T)[:,:-1]
            
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
        for i in range(max_iter):
            self.__call__(input_data)
            self.backprop(targets)
            self.update()
            self.print_to_terminal(i, t)
        return self.a[-1]
        
    def print_to_terminal(self, i, targets):
        """ Print information to terminal.
        
        Parameters
        ----------
        i : int
            step number
        targets : ndarray
            number of targets need to match number of inputs
            size of target needs to match last layer of model
        """
        print(10 * "-" + " " + str(i+1) + " " + 10 * "-")
        print(self.mse(self.a[-1], targets))
        print(25 * "-")
        print(" ")
        
        
if __name__ == "__main__":
    from activation import LeakyReLU, ReLU
    from optimizer import ADAM, GradientDescent
    from cost import MSE
    
    data = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    
    model = Network((3), cost=MSE(), activation=LeakyReLU(a=0.2), optimizer=ADAM(epsilon=1e-7))
    model.dense(units=5, Optimizer=GradientDescent(y=0.1), activation=ReLU())
    model.dense(units=10)
    #model(data)
    
    
