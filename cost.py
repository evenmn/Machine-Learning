import numpy as np

class Cost:
    def __init__(self):
        pass
        
    def __call__(self, outputs, targets):
        raise NotImplementedError("Class {} has no instance '__call__'."
                                  .format(self.__class__.__name__))
        
    def derivate(self):
        raise NotImplementedError("Class {} has no instance 'derivate'."
                                  .format(self.__class__.__name__))
        
class MSE(Cost):
    def __call__(self, outputs, targets):
        self.diff = outputs - targets
        return 0.5 * np.diag(self.diff.dot(self.diff.T))
        
    def derivate(self):
        return self.diff
        
class CrossEntropy(Cost):
    def __call__(self, outputs, targets):
        self.diff = outputs - targets
        m = len(Y_proposed[0])    # Number of samples
        Y_proposed = np.log(Y_proposed)
        cost_value = -np.einsum('ij,ij',Y_batch,Y_proposed)
        encoded_proposed = np.argmax(Y_proposed, axis=0)
        encoded_batch = np.argmax(Y_batch, axis=0)
        num_correct = np.sum(encoded_proposed==encoded_batch)
        return cost_value/m, num_correct
        
    def derivative(self):
        return self.diff
        
        
