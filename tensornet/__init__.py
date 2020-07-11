import numpy as np

class Network:
    """Initialize network, including weights and nodes

    :type input_shape: ndtuple
    :param input_shape: dimension on input. Can be 1d (works for fully connected layer only), 2d (image with only 1 channel) and 3d (image with multiple channels)
    :type cost: obj
    :param cost: cost function. Functions are found in cost.py
    """

    from .cost import MSE

    def __init__(self, input_shape, cost=MSE(), **kwargs):
        from .layer import Layer
        Layer(**kwargs)
        self.cost = cost
        self.layers = []
        self.h = np.array([input_shape])
        self.weight = []
        self.a = [np.zeros(input_shape)]

    def append(self, layer):
        self.layers.append(layer)
        self.weight.append(layer.weight)

    def dense(self, units, **kwargs):
        """Add dense layer.

        :param units: number of hidden units
        :type units: int
        """
        self.h = np.append(self.h, units)
        self.a.append(np.zeros(units))

        from .layer import DenseLayer
        layer = DenseLayer(self.h[-2], self.h[-1], **kwargs)
        self.append(layer)

    def conv(self, kernel=(3,32,32), pad_size=(15,15), stride=(1,1), **kwargs):
        """Add convolutional layer.

        :type kernel: 3d-tuple of ints
        :param kernel: kernel size in vertical and horizontal direction
        :type pad_size: 2d-tuple of ints
        :param pad_size: zero padding in horizontal and vertical direction
        :type stride: 2d-tuple of ints
        :param stride: stride in horizontal and vertical direction
        """

        from .layer import ConvLayer
        layer = ConvLayer(kernel, pad_size, stride, **kwargs)
        self.append(layer)

    def pooling(self, kernel=(2,2), pad_size=(0,0), stride=(1,1), mode='max'):
        """ Add pooling layer.

        Parameters
        ----------
        kernel : 2dtuple of ints
            kernel size in vertical and horizontal direction. (2,2) by default
        pad_size : 2dtuple of ints
            pad size in vertical and horizontal direction. No padding by default.
        stride : 2dtuple of ints
            stride of pooling (height,width). By default the
            size of kernel (no overlap)
        mode : str
            mode of pooling. Max pooling ('max'), min pooling
            ('min') and mean pooling ('mean'/'avg') implemented
        """

        from .layer import Pooling
        layer = Pooling(kernel, pad_size, stride, mode)
        self.layers.append(layer)
        self.weight.append(0)

    def __call__(self, input_data):
        """Predicting output from network, given a input data set.

        :type input_data: ndarray
        :param input_data: input data needs to match the input shape of model
        """
        a = np.array(input_data)
        for layer in self.layers:
            a = layer(a)
        self.predicted = a
        return a

    @staticmethod
    def mse(predicted, targets):
        """Mean-square error, given inputs and targets.

        Parameters
        ----------
        predicted : ndarray
            input data needs to match the input shape of model
        targets : ndarray
            number of targets need to match number of inputs
            size of target needs to match last layer of model
        """
        mse_cost = MSE()
        return mse_cost(predicted, targets)

    @staticmethod
    def score(predicted, targets):
        """ Returns the R2-score.
        """
        u = ((targets - predicted) ** 2).sum()
        v = ((targets - targets.mean()) ** 2).sum()
        return 1 - u/v

    def backprop(self, start, stop):
        """ Back-propagation processing based on some targets
        """
        dcost = self.cost.derivate()[start:stop]
        for layer in reversed(list(self.layers)):
            dcost = layer.backward(dcost, start, stop)

    def update(self, step):
        """ Update weights.

        Parameters
        ----------
        step : int
            current step
        """
        for i, layer in enumerate(self.layers):
            self.weight[i] = layer.update_weights(step)

    def train(self, input_data, targets, epochs=1000, mini_batches=10):
        """ Train the model.

        Parameters
        ----------
        input_data : ndarray
            input data needs to match the input shape of model
        targets : ndarray
            number of targets need to match number of inputs
            size of target needs to match last layer of model
        max_iter : int
            max number of training interations
        mini_batches : int
            number of mini batches
        """

        samples = len(input_data)
        samples_per_batch = int(samples/mini_batches)

        from tqdm import trange
        with trange(epochs, unit=' epochs') as epoch:
            for step in epoch:
                start = 0
                for batch in range(mini_batches):
                    # Forward
                    predicted = self(input_data)
                    loss = self.cost(predicted, targets).sum()

                    # Backward
                    stop = start + samples_per_batch
                    self.backprop(start, stop)
                    self.update(batch*(1+step))
                    epoch.set_description('Training ' + '.' * (step%4) + \
                                                        ' ' * (4-step%4))
                    epoch.set_postfix(loss=loss)
                    start = stop
        return loss


if __name__ == "__main__":
    from tensornet.activation import LeakyReLU, ReLU, Sigmoid
    from tensornet.optimizer import ADAM, GradientDescent
    from tensornet.cost import MSE
    from tensornet.initialize import Normal

    # XOR GATE
    data = [[0, 0], [0, 1], [1, 0], [1, 1]]
    targets = [[0], [1], [1], [0]]

    model = Network((2), cost=MSE(), activation=LeakyReLU(a=0.2), optimizer=ADAM(lr=0.1), bias=False)
    model.dense(units=5, optimizer=GradientDescent(lr=0.1), activation=ReLU())
    model.dense(units=1)
    model.train(data, targets, max_iter=1000)
