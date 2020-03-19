import numpy as np
    
def pad(data, pad_size=None, value=0):
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
        
    Returns
    -------
    ndarray
        padded data
        
    >>> data = np.arange(8).reshape(2,2,2)
    >>> pad(data)
        [[[0. 0. 0. 0.]
          [0. 0. 1. 0.]
          [0. 2. 3. 0.]
          [0. 0. 0. 0.]]

         [[0. 0. 0. 0.]
          [0. 4. 5. 0.]
          [0. 6. 7. 0.]
          [0. 0. 0. 0.]]]

    """
    # Pad size is 1 along the two last axes by default
    num_axes = len(data.shape)
    if pad_size == None:
        pad_size = np.zeros(num_axes)
        pad_size[-1] = pad_size[-2] = 1
        
    # Create padded array with correct shape
    output_shape = [int(2 * pad_size[dim] + data.shape[dim]) for dim in range(num_axes)]
    padded = value * np.ones(output_shape)
    
    # Insert data into padded array
    insertHere = [slice(int(pad_size[dim]), int(pad_size[dim] + data.shape[dim])) for dim in range(num_axes)]
    padded[tuple(insertHere)] = data
    return padded
    
def stride(input_data, kernel, stride):
        """Returns the strided input data.
        
        Parameters
        ----------
        input_data : ndarray
            The array to be strided
        kernel : ndtuple
            shape of each new array
        stride : ndtuple
            stride along each axis
            
        Returns
        -------
        ndarray
            strided data
            
        >>> data = np.arange(8).reshape(2,2,2)
        >>> padded_data = pad(data)
        >>> stride(padded_data, kernel=(2,3,3), stride=(1,1,1))
        [[[[[[0. 0. 0.]
             [0. 0. 1.]
             [0. 2. 3.]]

            [[0. 0. 0.]
             [0. 4. 5.]
             [0. 6. 7.]]]


           [[[0. 0. 0.]
             [0. 1. 0.]
             [2. 3. 0.]]

            [[0. 0. 0.]
             [4. 5. 0.]
             [6. 7. 0.]]]]



          [[[[0. 0. 1.]
             [0. 2. 3.]
             [0. 0. 0.]]

            [[0. 4. 5.]
             [0. 6. 7.]
             [0. 0. 0.]]]


           [[[0. 1. 0.]
             [2. 3. 0.]
             [0. 0. 0.]]

            [[4. 5. 0.]
             [6. 7. 0.]
             [0. 0. 0.]]]]]]

        """
        num_axes = len(input_data.shape)
        assert len(kernel) == num_axes, ("Mismatch between kernel size and data axes.")
        assert len(stride) == num_axes, ("Mismatch between stride size and data axes.")
        
        output_shape = []
        stride_shape = []
        for dim in range(num_axes):
            assert (input_data.shape[dim] - kernel[dim]) % stride[dim] == 0, \
                   ("Kernel does not match data size.")
            output_shape.append(1 + (input_data.shape[dim] - kernel[dim]) // stride[dim])
            stride_shape.append(input_data.strides[dim] * stride[dim])
        
        from numpy.lib.stride_tricks import as_strided
        strided = as_strided(input_data, shape = tuple(output_shape) + tuple(kernel), 
                                strides = tuple(stride_shape) + tuple(input_data.strides))
        return strided
    
if __name__ == "__main__":
    data = np.arange(8).reshape(2,2,2)
    padded_data = pad(data)
    strided_data = stride(padded_data, kernel=(2,3,3), stride=(1,1,1))
    print(strided_data)
