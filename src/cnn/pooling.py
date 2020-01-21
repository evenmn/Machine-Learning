import numpy as np
from numpy.lib.stride_tricks import as_strided

class Pooling(object):
    def __init__(self, mode, window):
        self.mode = mode
        self.window = window

    def pool2d(self, data, stride, padding):
        # Padding
        A = 1
        A = np.pad(A, padding, mode='constant')

        # Window view of data
        output_shape = ((data.shape[0] - self.window[0])//stride + 1,
                        (data.shape[1] - self.window[1])//stride + 1)
        A_w = as_strided(data, shape = output_shape + window, 
                            strides = (stride*A.strides[0],
                                       stride*A.strides[1]) + A.strides)
        A_w = A_w.reshape(-1, *kernel_size)

        # Return the result of pooling
        if self.mode == 'max':
            return A_w.max(axis=(1,2)).reshape(output_shape)
        elif self.mode == 'min':
            return A_w.min(axis=(1,2)).reshape(output_shape)
        elif self.mode == 'avg':
            return A_w.mean(axis=(1,2)).reshape(output_shape)
        
