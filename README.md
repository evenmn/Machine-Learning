# NEURAL NETWORKS



# Perceptrons (Feedforward Neural Networks)

In the first place this repository contains neural networks, where the most interesting ones might be the general networks based on a given number of hidden layers where each contains a given number of nodes. All the networks are implemented in both Python and C++, where the majority of Python code is based on the Numpy package while the C++ programs are based on armadillo. A performance test to compare the CPU time for each of those languages might be added. 


# How to run the code

The general networks are named multilayer.py and multilayer.cpp in Python and C++ respectively, and to run them one needs to know a few parameters:

1. The input arrays need to be stored in a matrix X
2. The exact outputs corresponding to the inputs need to be stored in a matrix t
3. The number of iterations N needs to be large such that the network gets enough training
4. An array h where the length corresponds to the number of hidden layers and each element is hidden nodes

To see some simple example implementations, you could check out main.py / main.cpp. The networks should be able to run much more complex problems than presented here.


# Documentation

If you look at the code, you might wonder where all the operations come from and how it works. All the documentation is given in a document found in the folder Documents, and it should be readable even though you have never seen a neural network before. 


# Boltzmann Machines

Using Boltzmann Machines to find the optimal wavefunction for a multiparticle system. RBM is already implemented and DBM will be investigated. I might switch from armadillo to eigen.

