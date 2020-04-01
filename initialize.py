class Initialize:
    def __init__(self):
        pass
        
    def __call__(self, size):
        raise NotImplementedError("Class {} has no instance '__call__'.".format(self.__class__.__name__))
        
class Uniform(Initialize):
    def __init__(self, minval=-1, maxval=1):
        self.minval = minval
        self.maxval = maxval
    def __call__(self, size):
        """ Initialize an array uniformly
        
        Parameters
        ----------
        size : ndtuple
            size of array that should be initialized
        """
        import numpy as np
        valrange = self.maxval - self.minval
        return (valrange * np.random.uniform(size=size) - self.minval)/sum(size)
        
class Normal(Initialize):
    def __init__(self, mean=0, var=0.1):
        self.mean = mean
        self.var = var
    def __call__(self, size):
        """ Initialize an array according to a normal distribution
        
        Parameters
        ----------
        size : ndtuple
            size of array that should be initialized
        """
        import numpy as np
        return np.random.normal(self.mean, self.var, size=size)/sum(size)
