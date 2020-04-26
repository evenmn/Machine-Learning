import numpy as np

class Cost:
    """ Cost function class.
    """
    def __init__(self):
        pass
        
    def __call__(self, predicted, targets):
        raise NotImplementedError("Class {} has no instance '__call__'."
                                  .format(self.__class__.__name__))
        
    def derivate(self):
        raise NotImplementedError("Class {} has no instance 'derivate'."
                                  .format(self.__class__.__name__))
        
class MSE(Cost):
    """ Mean square error cost. To be used in networks with 
    continuous outputs.
    """
    def __init__(self):
        pass
        
    def __call__(self, predicted, targets):
        """ Compute the MSE cost.
        
        Parameters
        ----------
        predicted : ndarray
            predicted outputs of the neural network
        targets : ndarray
            expected outputs of the neural network
        """
        targets = targets.reshape(predicted.shape)
        self.diff = predicted - targets
        return 0.5 * np.diag(self.diff.dot(self.diff.T)) / len(targets)
        
    def derivate(self):
        """ The derivative of the cost
        """
        return self.diff
        
class CrossEntropy(Cost):
    """ Mean square error cost. To be used in classification problems.
    """
    def __init__(self):
        pass
        
    def __call__(self, predicted, targets):
        """ Compute the cross-entropy cost.
        
        Parameters
        ----------
        predicted : ndarray
            predicted outputs of the neural network
        targets : ndarray
            expected outputs of the neural network
        """
        targets = targets.reshape(predicted.shape)
        self.diff = predicted - targets
        m = len(predicted[0])    # Number of samples
        Y_proposed = np.log(predicted)
        cost_value = -np.einsum('ij,ij',targets,predicted)
        encoded_predicted = np.argmax(predicted, axis=0)
        encoded_targets = np.argmax(targets, axis=0)
        num_correct = np.sum(encoded_predicted==encoded_targets)
        return cost_value/m, num_correct
        
    def derivative(self):
        """ The derivative of the cost
        """
        return self.diff
        
        
