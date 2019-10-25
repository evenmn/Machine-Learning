from network import Network
import numpy as np

'''
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

NET = Network(input_units=len(train_d[0]))
#NET.add(128, activation='leakyrelu')
NET.add(256, activation='sigmoid')
NET.add(len(train_t[0]))

NET.simulate(train_d, train_t, learning_rate=0.2, max_iter=1000)
print(NET.predict(train_d))
#print(NET.predict(train_d))
#print(NET.predict(test_d))
