from network import Network
import numpy as np
import matplotlib.pyplot as plt

x_max = 5
N = 1000

def f(x):
    ''' Function to approximate. '''
    return np.exp(-0.5 * np.square(x))

x_train = np.linspace(-x_max, x_max, N)
y_train = f(x_train)

# Define neural network
NET = Network(input_units=1, cost='mse')
NET.dense(2, activation='leakyrelu', eta=0.0001, optimizer='adam', init='normal')
NET.dense(1, activation=None, eta=0.001, optimizer='adam', init='normal')

# Run simulation
NET.simulate(x_train[:,np.newaxis], y_train[:,np.newaxis], max_iter=20000)


# Test
x_test = np.linspace(-4, 4, 100)
y_test = f(x_test)

# Compare
plt.plot(x_test, y_test, label="Exact")
plt.plot(x_test, NET.predict(x_test), label="Approx")
plt.legend(loc='best')
plt.show()

# Dump weights to file
np.savetxt('test.dat', NET.W, fmt='%s')
