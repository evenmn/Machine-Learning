class Layer:
    """Layer shell for all the various layers in a neural network.
    """
    
    def __init__(self, h_0, h_1, eta, activation, optimizer):
        pass
        
    def __call__(self, input_layer):
        pass
        
    def get_gradients(self):
        pass
        
    def update_weights(self, i):
        pass
        
class Flatten(Layer):
    """Layer to be used between a convolutional or pooling layer
    and a dense layer. 
    """
    def __init__(self):
        pass
        
    def __call__(self, input_layer):
        return input_layer.flatten
        
    def get_gradients(self):
        pass
        
    def update_weights(self, i):
        pass
