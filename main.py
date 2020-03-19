from network import Network
import numpy as np

'''
=======
# Load data
>>>>>>> 98dd7b2932d76142da3bf4372ada17e82c4396f3:main.py
train_d = [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
           [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
           [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
           [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]]
        
train_t = [[0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
           [1, 1, 0, 1, 1, 1, 1, 1, 1, 1],
           [1, 1, 1, 1, 1, 1, 1, 1, 0, 1],
           [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
           [1, 1, 1, 1, 1, 1, 1, 0, 1, 1],
           [1, 1, 1, 0, 1, 1, 1, 1, 1, 1],
           [1, 1, 1, 1, 1, 1, 0, 1, 1, 1],
           [1, 1, 1, 1, 1, 0, 1, 1, 1, 1]]
           
test_d = [[0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
          [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]]
          
test_t = [[1, 1, 1, 1, 0, 1, 1, 1, 1, 1],
          [1, 0, 1, 1, 1, 1, 1, 1, 1, 1]]
'''

train_d = np.random.uniform(-10, 10, size=(1000,1))
train_t = np.exp(train_d)

test_d = np.random.uniform(-10, 10, size=(100,1))
test_t = np.exp(test_d)

# Define neural network
NET = Network(input_units=len(train_d[0]), cost='mse')
NET.dense(256, activation='relu', eta=0.2, optimizer='adam', init='normal')
NET.dense(512, activation='leakyrelu', eta=0.2, optimizer='adam', init='normal')
NET.dense(len(train_t[0]), activation='sigmoid', eta=0.1, optimizer='gd', init='uniform')

# Run simulation
NET.simulate(train_d, train_t, max_iter=3000)
print(NET.predict(train_d))
x = NET.predict(test_d)
print(NET.mse(x, test_d))

import matplotlib.pyplot as plt
plt.plot(test_d, test_t)
plt.show()
