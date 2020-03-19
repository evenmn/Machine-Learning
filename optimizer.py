import numpy as np

class Optimizer:
    def __init__(self, eta, h_0, h_1):
        self.eta = eta
        self.h_0 = h_0
        self.h_1 = h_1
        
    def update_parameters(self, step, gradient):
        raise NotImplementedError("Class {} has no instance 'derivate'.".format(self.__class__.__name__))
        
class GradientDescent(Optimizer):
    def __init__(self, eta, h_0, h_1):
        self.eta = eta
        self.m = np.zeros((h_0, h_1))
        
    def update_parameters(self, i, gradient, y=0.0, lamb=0.0):
        self.m = y * self.m + self.eta * gradient
        return self.m / i**lamb
        
class ADAM(Optimizer):
    def __init__(self, eta, h_0, h_1):
        self.eta = eta
        self.m = np.zeros((h_0, h_1))
        self.v = np.zeros((h_0, h_1))
        
    def update_parameters(self, i, gradient, y1=0.1, y2=0.001, epsilon=1e-8):
        self.m = y1 * self.m + (1 - y1) * gradient
        self.v = y2 * self.v + (1 - y2) * np.power(gradient, 2)
        m_hat = self.m/(1 - y1**i)
        v_hat = self.v/(1 - y2**i)
        return self.eta * np.nan_to_num(m_hat/(np.sqrt(v_hat) + epsilon))
