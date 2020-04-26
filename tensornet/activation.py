import numpy as np

class Activation:
    def __init__(self):
        pass
        
    def __call__(self, Z):
        raise NotImplementedError("Class {} has no instance '__call__'."
                                  .format(self.__class__.__name__))
        
    def derivate(self):
        raise NotImplementedError("Class {} has no instance 'derivate'."
                                  .format(self.__class__.__name__))
        
class PureLinear(Activation):
    def __init__(self):
        pass
        
    def __call__(self, Z):
        self.Z = Z
        return Z
        
    def derivate(self):
        return np.ones(self.Z.shape)
        
class Sigmoid(Activation):
    def __init__(self):
        pass
        
    def __call__(self, Z):
        self.a = 1/(1 + np.exp(-Z))
        return self.a
        
    def derivate(self):
        return self.a * (1 - self.a)
        
class ReLU(Activation):
    def __init__(self):
        pass
        
    def __call__(self, Z):
        self.Z = Z
        return np.where(Z>0, Z, 0)
        
    def derivate(self):
        return np.where(self.Z>0, 1, 0)
            
class LeakyReLU(Activation):
    def __init__(self, a=0.1):
        self.a = a
        
    def __call__(self, Z):
        self.Z = Z
        return np.where(Z>0, Z, self.a * abs(Z))
        
    def derivate(self):
        return np.where(self.Z>0, 1, self.a)
            
class ELU(Activation):
    def __init__(self):
        pass
        
    def __call__(self, Z):
        self.Z = Z
        return np.where(Z>0, Z, np.exp(Z) - 1)
            
    def derivate(self):
        return np.where(self.Z>0, 1, np.exp(self.Z))
        
class Softmax(Activation):
    def __init__(self, shift=False):
        self.shift = shift
        
    def __call__(self, Z):
        """Compute and return the softmax of the input.

        To improve numerical stability, we do the following

        1: Subtract Z from max(Z) in the exponentials
        2: Take the logarithm of the whole softmax, and then take the exponential of that in the end

        Args:
            Z: numpy array of floats with shape [n, m]
            shift: Boolean indicating whether or not the Z should be shifted to negative numbers.
        Returns:
            numpy array of floats with shape [n, m]
        """
        if self.shift: 
            Z -= np.max(Z)
        log_softmax = Z - np.log(np.sum(np.exp(Z), axis=0))
        return np.exp(log_softmax)
