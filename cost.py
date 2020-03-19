import numpy as np

class Cost:
    def __init__(self):
        self.k = 5
        
    def evaluate(self, x, t):
        raise NotImplementedError("Class {} has no instance 'evaluate'.".format(self.__class__.__name__))
        
    def derivate(self, x, t):
        raise NotImplementedError("Class {} has no instance 'derivate'.".format(self.__class__.__name__))
        
class MSE(Cost):
    def evaluate(self, x, t):
        diff = x - t
        return 0.5 * np.diag(diff.dot(diff.T))
        
    def derivate(self, x, t):
        return x - t
        

