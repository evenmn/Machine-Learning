from network import Network

# Load data
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

# Define neural network
NET = Network(input_units=len(train_d[0]), cost='mse')
NET.dense(512, activation='leakyrelu', eta=0.2, optimizer='adam', init='normal')
NET.dense(len(train_t[0]), activation='sigmoid', eta=0.1, optimizer='gd', init='uniform')

# Run simulation
NET.simulate(train_d, train_t, max_iter=3000)
print(NET.predict(train_d))
x = NET.predict(test_d)
print(NET.mse(x, test_d))
