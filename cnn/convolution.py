import numpy as np
from numpy.lib.stride_tricks import as_strided

class Convolution:
    def __init__(self):
        pass
    
    def evaluate(self):
        pass
        
    def initialize_filters(self, filter_size, channels=1):
        '''
        filter size is supposed to be given as a 2D tuple
        ex. (5, 5)
        '''
        self.filter_size = filter_size
        self.filters = np.random.randn(channels, filter_size[0], filter_size[1])/(filter_size[0] * filter_size[1])
        
    def iterate_regions(self, image):
        '''
        Generates all possible 3x3 image regions using valid padding.
        - image is a 2d numpy array
        '''
        h, w = image.shape
        filter_h = self.filter_size[0]
        filter_w = self.filter_size[1]

        for i in range(h - filter_h + 1):
          for j in range(w - filter_w + 1):
            im_region = image[i:(i + filter_h), j:(j + filter_w)]
            yield im_region, i, j
            
    def forward(self, input):
        '''
        Performs a forward pass of the conv layer using the given input.
        Returns a 3d numpy array with dimensions (h, w, num_filters).
        - input is a 2d numpy array
        '''
        h, w = input.shape
        output = np.zeros((h - 2, w - 2, self.num_filters))

        for im_region, i, j in self.iterate_regions(input):
          output[i, j] = np.sum(im_region * self.filters, axis=(1, 2))
        return output
        
    @staticmethod
    def padding(data, pad_size=None, value=0):
        """Returns the (zero) padded data array.
        
        Parameters
        ----------
        data : ndarray
            Array that should be padded
        pad_size : tuple of ints
            Tuple with pad_size along respective axes. 
            Number of axes must correspond to data.shape.
        value : float
            Type of padding. Zero padding by default.
        """
        num_axes = len(data.shape)
        if pad_size == None:
            pad_size = np.zero(num_axes)
            pad_size[-1] = pad_size[-2] = 1
            
        output_shape = []
        for i in range(num_axes):
            output_shape.append(data.shape[i], 2 * pad_size[i])
        padded = np.zeros(output_shape)
        
        padded[:,:,pad_size[0]:height_x + pad_size[0],
              pad_size[1]:width_x  + pad_size[1]] = input_layer
        
    def conv_layer_forward(self, input_layer, weight, bias, pad_size=(1,1), stride=(1,1)):
        """
        A naive implementation of the forward pass for a convolutional layer.

        The input consists of M data points, each with C channels, height H and
        width W. We convolve each input with C_o different filters, where each filter
        spans all C_i channels and has height H_w and width W_w.

        Args:
            input_alyer: The input layer with shape (batch_size, channels_x, height_x, width_x)
            weight: Filter kernels with shape (num_filters, channels_x, height_w, width_w)
            bias: Biases of shape (num_filters)

        Returns:
            output_layer: The output layer with shape (batch_size, num_filters, height_y, width_y)
            
        >>> input_layer = np.arange(16).reshape(1,1,4,4)
        >>> weight = np.arange(9).reshape(1,1,3,3)
        >>> bias = np.array([2])

        >>> Covolution.conv_layer_forward(input_layer, weight, bias)
            [[[[ 93. 157. 202. 139.]
               [225. 348. 402. 264.]
               [381. 564. 618. 396.]
               [223. 319. 346. 213.]]]]
        """
        
        # TODO: Make multiple filters possible

        batch_size, channels_x, height_x, width_x = input_layer.shape
        num_filters, channels_w, height_w, width_w = weight.shape
        height_y = 1 + (height_x - height_w + 2 * pad_size[0]) // stride[0]
        width_y = 1 + (width_x - width_w + 2 * pad_size[1]) // stride[1]
        
        assert channels_w == channels_x, (
            "The number of filter channels be the same as the number of input layer channels")

            
        # Add padding to input layer
        x = np.zeros((batch_size, channels_x, height_x + 2*pad_size[0], 
                                              width_x  + 2*pad_size[1]))
        x[:,:,pad_size[0]:height_x + pad_size[0],
              pad_size[1]:width_x  + pad_size[1]] = input_layer
        
        # Shape after stride
        output_shape = (batch_size, num_filters, height_y, width_y)
        
        
        # Stride the input layer
        x_strided = as_strided(x, shape = output_shape + weight.shape, 
                               strides =(x.strides[0],
                                         x.strides[1],
                                         stride[0]*x.strides[2],
                                         stride[1]*x.strides[3]) + x.strides)
        
        output_layer = np.einsum('ijpqklrs,klrs->ijpq',x_strided,weight)
        output_layer += np.einsum('ijpqklrs,k->ijpq',x_strided,bias)
        
        return output_layer


    def conv_layer_backward(self, output_layer_gradient, input_layer, weight, bias, pad_size=(1,1), stride=(1,1)):
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
        batch_size, channels_x, height_x, width_x = input_layer.shape
        num_filters, channels_w, height_w, width_w = weight.shape

        assert num_filters == channels_y, (
            "The number of filters must be the same as the number of output layer channels")
        assert channels_w == channels_x, (
            "The number of filter channels be the same as the number of input layer channels")
        
        # Add padding to input layer
        x = np.zeros((batch_size, channels_x, height_x + 2*pad_size[0], 
                                              width_x  + 2*pad_size[1]))
        x[:,:,pad_size[0]:height_x + pad_size[0],
              pad_size[1]:width_x  + pad_size[1]] = input_layer
        y = np.zeros((batch_size, channels_y, height_y + 2*pad_size[0], 
                                              width_y  + 2*pad_size[1]))
        y[:,:,pad_size[0]:height_y + pad_size[0],
              pad_size[1]:width_y  + pad_size[1]] = output_layer_gradient
        
        bias_gradient = np.zeros(bias.shape)
        weight_gradient = np.zeros(weight.shape)
        input_layer_gradient = np.zeros(input_layer.shape)
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
        
if __name__ == "__main__":
    input_layer = np.arange(16).reshape(1,1,4,4)
    weight = np.arange(9).reshape(1,1,3,3)
    bias = np.array([2])

    conv = Convolution()
    out = conv.conv_layer_forward(input_layer, weight, bias)
    print(out)
