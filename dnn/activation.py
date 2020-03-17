import numpy as np

class Activation:
    def __init__(self):
        self.k = 5
        
    def evaluate(self, x):
        raise NotImplementedError("Class {} has no instance 'evaluate'.".format(self.__class__.__name__))
        
    def derivate(self, x):
        raise NotImplementedError("Class {} has no instance 'derivate'.".format(self.__class__.__name__))
        
class Sigmoid(Activation):
    def evaluate(self, x):
        self.a = 1/(1 + np.exp(-x))
        return self.a
        
    def derivate(self, x):
        return self.a * (1 - self.a)
        
class ReLU(Activation):
    def evaluate(self, x):
        return np.where(x>0, x, 0)
        
    def derivate(self, x):
        return np.where(x>0, 1, 0)
            
class LeakyReLU(Activation):
    def evaluate(self, x, a=0.1):
        return np.where(x>0, x, a * abs(x))
            
    def derivate(self, x, a=0.1):
        return np.where(x>0, 1, a)
            
class ELU(Activation):
    def evaluate(self, x):
        return np.where(x>0, x, np.exp(x) - 1)
            
    def derivate(self, x):
        return np.where(x>0, 1, np.exp(x))
        
    
