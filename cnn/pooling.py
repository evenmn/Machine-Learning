import numpy as np
from layer import Layer

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
        
        
if __name__ == "__main__":
    pool = Pooling()
    data = np.arange(16).reshape(4,4)
    print(data)
    print(pool.pool2d(data, kernel=(2,2), mode='max'))
