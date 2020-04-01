import numpy as np

class Optimizer:
    def __init__(self):
        pass
        
    def update_parameters(self, step, gradient):
        raise NotImplementedError("Class {} has no instance 'derivate'.".format(self.__class__.__name__))
        
class GradientDescent(Optimizer):
    def __init__(self, y=0.0, lamb=0.0):
        self.y = y
        self.lamb = lamb
        
    def initialize(self, eta, h_0, h_1):
        self.eta = eta
        self.m = np.zeros((h_0, h_1))
        
    def __call__(self, i, gradient):
        """ Update weights.
        
        Parameters
        ----------
        
        i : int
            current step
        gradient : ndarray
            ??
        """
        self.m = self.y * self.m + self.eta * gradient
        return self.m / i**self.lamb
        
class ADAM(Optimizer):
    def __init__(self, y1=0.1, y2=0.001, epsilon=1e-8):
        self.y1 = y1
        self.y2 = y2
        self.epsilon = epsilon

    def initialize(self, eta, h_0, h_1):
        self.eta = eta
        self.m = np.zeros((h_0, h_1))
        self.v = np.zeros((h_0, h_1))
        
    def __call__(self, i, gradient):
        """ Update weights.
        
        Parameters
        ----------
        
        i : int
            current step
        gradient : ndarray
            ??
        """
        self.m = y1 * self.m + (1 - self.y1) * gradient
        self.v = y2 * self.v + (1 - self.y2) * np.power(gradient, 2)
        m_hat = self.m/(1 - self.y1**i)
        v_hat = self.v/(1 - self.y2**i)
        return self.eta * np.nan_to_num(m_hat/(np.sqrt(v_hat) + self.epsilon))
