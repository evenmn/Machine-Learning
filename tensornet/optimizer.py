import numpy as np

class Optimizer:
    def __init__(self):
        raise NotImplementedError("Class {} has no instance '__init__'."
                                  .format(self.__class__.__name__))
        
    def __call__(self, step, gradient):
        raise NotImplementedError("Class {} has no instance '__call__'."
                                  .format(self.__class__.__name__))
        
class SGD(Optimizer):
    """ Stochastic gradient descent optimizer.
    
    Parameters
    ----------
    lr : float
        learning rate
    y : float
        momentum parameter
    lamb : float
        decay rate
    """

    def __init__(self, lr=0.01, y=0.0, lamb=0.0):
        self.lr = lr
        self.y = y
        self.lamb = lamb
        
    def __call__(self, step, gradient):
        """ Update weights.
        
        Parameters
        ----------
        
        step : int
            current step
        gradient : ndarray
            gradient of cost function with respect to parameters
        """
        m = np.zeros(gradient.shape)        # momentum
        m = self.y * m + self.lr * gradient
        return m / step**self.lamb
        
class ADAM(Optimizer):
    """ ADAM optimizer.
    
    Parameters
    ----------
    lr : float
        learning rate
    y1 : float
        parameter of first momentum
    y2 : float
        parameter of second momentum
    epsilon : float
        factor to avoid zero division. Usually ~1e-8
    """

    def __init__(self, lr=0.01, y1=0.01, y2=0.001, epsilon=1e-8):
        self.lr = lr
        self.y1 = y1
        self.y2 = y2
        self.epsilon = epsilon
        
    def __call__(self, step, gradient):
        """ Update weights.
        
        Parameters
        ----------
        
        step : int
            current step
        gradient : ndarray
            gradient of cost function with respect to parameters
        """
        m = np.zeros(gradient.shape)        # first momentum
        v = np.zeros(gradient.shape)        # second momentum
        m = self.y1 * m + (1 - self.y1) * gradient
        v = self.y2 * v + (1 - self.y2) * gradient**2
        m_hat = m/(1 - self.y1**step)
        v_hat = v/(1 - self.y2**step)
        return self.lr * np.nan_to_num(m_hat/(np.sqrt(v_hat) + self.epsilon))
