import numpy as np
from tensornet.tools import pad, stride

class Layer:
    """Layer shell for all the various layers in a neural network.
    """
    
    def __init__(self, init, activation, optimizer, bias):
        self.init = init
        self.activation = activation
        self.optimizer = optimizer
        self.bias = bias
        
    def __call__(self, input_layer):
        raise NotImplementedError("Class {} has no instance '__call__'."
                                  .format(self.__class__.__name__))
        
    def get_gradients(self):
        raise NotImplementedError("Class {} has no instance 'get_gradients'."
                                  .format(self.__class__.__name__))
        
    def update_weights(self, i):
        raise NotImplementedError("Class {} has no instance 'update_weights'."
                                  .format(self.__class__.__name__))
        
class Flatten(Layer):
    """Layer to be used between a convolutional or pooling layer
    and a dense layer. 
    """
    def __init__(self):
        pass
        
    def __call__(self, input_layer):
        return input_layer.flatten
        
    def get_gradients(self):
        pass
        
    def update_weights(self, i):
        pass
        
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
        bias on previous layer on (True) / off (False)
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
        return input_layer.dot(self.weight)
        
    def __call__(self, input_layer):
        """Activation of output from forward propagation. 
        
        Parameters
        ----------
        input_layer : ndarray
            Output from previous layer.
        """
        if self.bias:
            input_layer = np.c_[input_layer, np.ones((len(input_layer),1))]
        self.input_layer = input_layer
        z = self.forward(input_layer)
        self.output_layer = self.activation(z)
        return self.output_layer
        
    def backward(self, dcost, start, stop):
        """ Backward propagation.
        
        Parameters
        ----------
        dcost : ndarray
            derivative of cost function with respect to the activation array 
        start : int
            start index of batch
        stop : int
            final index of batch
        """
        self.input_layer = self.input_layer[start:stop]
        df = self.activation.derivate(start, stop)
        self.delta = dcost * df
        dcost_new = np.einsum('ik,jk->ij',self.delta,self.weight)
        if self.bias:
            return dcost_new[:,:-1]
        else:
            return dcost_new
        
    def update_weights(self, step):
        """ Update the weight matrix 
        """
        gradient = np.einsum('ij,ik->jk', self.input_layer, self.delta)
        self.weight -= self.optimizer(step+1, gradient)
        return self.weight

class ConvLayer(Layer):
    """Convolutional layer
        
    Parameters
    ----------
    
    kernel : 3dtuple of ints
        kernel size in vertical and horizontal direction
    pad_size : 2dtuple of ints
        zero padding in horizontal and vertical direction
    stride : 2dtuple of ints
        stride in horizontal and vertical direction 
    init : obj
        how to initialize weights. Methods are found in initialize.py
    activation : obj
        activation function. Functions are found in activation.py
    optimizer : obj
        optimizer function. Methods are found in optimizer.py
    bias : bool
        bias on (True) / off (False)
    """
    def __init__(self, kernel, pad_size, stride, init, activation, optimizer, bias):
        self.kernel = kernel
        self.pad_size = pad_size
        self.stride = stride
        self.activation = activation
        self.optimizer = optimizer
        self.weight = init(size=(1) + self.kernel)
        if self.bias:
            self.bias_weight = init(self.kernel[0])
        
    def forward(self, input_layer):
        """
        A naive implementation of the forward pass for a convolutional layer.

        The input consists of M data points, each with C channels, height H and
        width W. We convolve each input with C_o different filters, where each filter
        spans all C_i channels and has height H_w and width W_w.

        Args:
            input_layer: The input layer with shape (batch_size, channels_x, height_x, width_x)
            weight: Filter kernels with shape (num_filters, channels_x, height_w, width_w)
            bias: Biases of shape (num_filters)

        Returns:
            output_layer: The output layer with shape (batch_size, num_filters, height_y, width_y)
            
        >>> input_layer = np.arange(16).reshape(1,1,4,4)
        >>> weight = np.arange(9).reshape(1,1,3,3)
        >>> bias = np.array([2])

        >>> Covolution.forward(input_layer, weight, bias)
            [[[[ 93. 157. 202. 139.]
               [225. 348. 402. 264.]
               [381. 564. 618. 396.]
               [223. 319. 346. 213.]]]]
        """
        
        # TODO: Make multiple filters possible
        self.input_layer = input_layer
        batch_size, channels_x, height_x, width_x = input_layer.shape
        num_filters, channels_w, height_w, width_w = self.weight.shape
        
        assert channels_w == channels_x, (
            "The number of filter channels be the same as the number of input layer channels")

            
        # Add padding to input layer
        padded = pad(input_layer, pad_size = (0,0) + pad_size)
        
        # Shape after stride
        strided = stride(padded, kernel = (1) + kernel, stride = (1,1) + stride)
        
        z = np.einsum('ijpqklrs,klrs->ijpq',strided,weight)
        if self.bias:
            z += np.einsum('ijpqklrs,k->ijpq',strided,bias_weight)
        
        return z
        
    def __call__(self, input_layer):
        """Activation of output from forward propagation. 
        
        Parameters
        ----------
        input_layer : ndarray
            Output from previous layer.
        """
        z = self.forward(input_layer)
        self.output_layer = self.activation(z)
        return self.output_layer


    def backward(self, output_layer_gradient):
        """
        A naive implementation of the backward pass for a convolutional layer.

        Args:
            output_layer_gradient: Gradient of the loss L wrt the next layer y, with shape
                (batch_size, num_filters, height_y, width_y)
            input_layer: Input layer x with shape (batch_size, channels_x, height_x, width_x)
            weight: Filter kernels with shape (num_filters, channels_x, height_w, width_w)
            bias: Biases of shape (num_filters)

        Returns:
            input_layer_gradient: Gradient of the loss L with respect to the input layer x
            weight_gradient: Gradient of the loss L with respect to the filters w
            bias_gradient: Gradient of the loss L with respect to the biases b
        """

        batch_size, channels_y, height_y, width_y = output_layer_gradient.shape
        batch_size, channels_x, height_x, width_x = self.input_layer.shape
        num_filters, channels_w, height_w, width_w = self.weight.shape

        assert num_filters == channels_y, (
            "The number of filters must be the same as the number of output layer channels")
        assert channels_w == channels_x, (
            "The number of filter channels be the same as the number of input layer channels")
        
        # Add padding to input layer
        x = np.zeros((batch_size, channels_x, height_x + 2*self.pad_size[0], 
                                              width_x  + 2*self.pad_size[1]))
        x[:,:,self.pad_size[0]:height_x + self.pad_size[0],
              self.pad_size[1]:width_x  + self.pad_size[1]] = self.input_layer
        y = np.zeros((batch_size, channels_y, height_y + 2*self.pad_size[0], 
                                              width_y  + 2*self.pad_size[1]))
        y[:,:,self.pad_size[0]:height_y + self.pad_size[0],
              self.pad_size[1]:width_y  + self.pad_size[1]] = output_layer_gradient
        
        bias_gradient = np.zeros(self.bias_weight.shape)
        weight_gradient = np.zeros(self.weight.shape)
        input_layer_gradient = np.zeros(self.input_layer.shape)
        '''
        output_layer = np.empty((batch_size, num_filters, height_y, width_y))
        for p in range(0, height_x, stride):
            p_ = int((p-1)/stride + 1)      # Mapping x coordinate to y coordinate
            for q in range(0, width_x, stride):
                q_ = int((q-1)/stride + 1)  # Mapping x coordinate to y coordinate
                conv_sec = x[:,:,p:p+height_w,q:q+width_w]
                output_layer[:,:,p_,q_] = bias + np.einsum('ikrs,jkrs->ij',conv_sec,weight)
        return output_layer
        '''
        
        # Stride the input layer
        from numpy.lib.stride_tricks import as_strided
        x_strided = as_strided(x, shape = output_shape + weight.shape, 
                               strides =(x.strides[0],
                                         x.strides[1],
                                         stride[0]*x.strides[2],
                                         stride[1]*x.strides[3]) + x.strides)
        y_strided = as_strided(y, shape = output_shape + weight.shape, 
                               strides =(y.strides[0],
                                         y.strides[1],
                                         stride[0]*y.strides[2],
                                         stride[1]*y.strides[3]) + y.strides)
                                         
        
        
        
        for p in range(0, height_x, stride):
            p_ = int((p-1)/stride + 1)      # Mapping x coordinate to y coordinate
            for q in range(0, width_x, stride):
                q_ = int((q-1)/stride + 1)  # Mapping x coordinate to y coordinate
                bias_gradient += np.sum(output_layer_gradient[:,:,p_,q_], axis=0)
                for r in range(height_w):
                    r_ = height_w-r-1       # Mapping y coordinate to w coordinate
                    for s in range(width_w):
                        s_ = width_w-s-1    # Mapping y coordinate to w coordinate
                        output_sec = output_layer_gradient[:,:,p_,q_]
                        input_sec = x[:,:,p+r,q+s]
                        weight_gradient[:,:,r,s] += np.einsum('ij,ik->jk',output_sec,input_sec)
                        input_layer_gradient[:,:,p,q] += np.einsum('ij,jk->ik',y[:,:,p_+r,q_+s],weight[:,:,r_,s_])

        return input_layer_gradient, weight_gradient, bias_gradient
        
class Pooling(Layer):
    """Perform pooling on some image. 
    
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
    def __init__(self, kernel, pad_size, stride, mode):
        self.kernel = kernel
        self.pad_size = pad_size
        self.stride = stride
        self.mode = mode
        
    @staticmethod
    def get_mode(strided_image, mode):
        """ Given a strided image, return the pooled image.
        
        Parameters
        ----------
        strided_image : ndarray
            
        """
        if mode == 'max':
            return A_w.max(axis=(1,2)).reshape(output_shape)
        elif mode == 'min':
            return A_w.min(axis=(1,2)).reshape(output_shape)
        elif mode == 'avg' or mode == 'mean':
            return A_w.mean(axis=(1,2)).reshape(output_shape)
        else:
            raise NotImplementedError("Mode {} is not implemented".format(mode))

    def pool2d(self):
        if stride is None:
            stride = kernel
        
        # Padding
        A = np.zeros((data.shape[0] + 2*pad_size[0], 
                      data.shape[1] + 2*pad_size[1]))
        A[pad_size[0]:data.shape[0] + pad_size[0],
          pad_size[1]:data.shape[1] + pad_size[1]] = data

        # Window view of data
        from numpy.lib.stride_tricks import as_strided
        output_shape = ((data.shape[0] - kernel[0])//stride[0] + 1,
                        (data.shape[1] - kernel[1])//stride[1] + 1)
        A_w = as_strided(data, shape = output_shape + kernel, 
                            strides = (stride[0]*A.strides[0],
                                       stride[1]*A.strides[1]) + A.strides)
                                       
        A_w = A_w.reshape(-1, *kernel)

        # Return the result of pooling
