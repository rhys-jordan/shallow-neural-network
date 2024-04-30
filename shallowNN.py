# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 13:51:12 2024

@author: rhysj

https://www.linkedin.com/pulse/shallow-neural-network-from-scratch-deeplearningai-assignment-kim/
"""

import numpy as np
import matplotlib.pyplot as plt

def plotResults(X, Y, results):
    plt.axhline(0, color='grey', ls =":")
    plt.plot(X, results, color="black", linestyle='solid')
    plt.plot(X, Y, color="blue", linestyle='solid')
    plt.axis([0, 2, -1, 1])
    plt.show()


def reLU(x):
    return x * (x > 0)

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def layer_sizes(X,Y):
    n_x = X.shape[0]
    n_h = 3
    n_y = Y.shape[0]
    return (n_x, n_h, n_y)


def initalize_parameters(n_x, n_h, n_y):
    np.random.seed(2)
    W1 = np.random.randn(n_h,n_x)*0.01
    b1 = np.zeros((n_h,1))
    W2 = np.random.randn(n_y, n_h)*0.01
    b2 = np.zeros((n_y,1))
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    return parameters

def forward_propagation(X, parameters):
    X = np.reshape(X, (21,1))
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    Z1 = np.dot(W1,X)+b1
    A1 = sigmoid(Z1)
    Z2 = np.dot(W2,A1)+b2
    #A2 = reLU(Z2)
    
    #assert(Z2.shape == (1, X.shape[1]))
    assert(Z2.shape == (X.shape[0],1))
    
    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": Z2}
    
    return Z2, cache
    
#Cost function    
def mean_squared_error(act, pred):
   diff = pred - act
   differences_squared = diff ** 2
   mean_diff = differences_squared.mean()
   return mean_diff  

def backward_propagation(parameters, cache, X, Y):
    m = X.shape[1]
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    A1 = cache["A1"]
    A2 = cache["A2"]
    
    dZ2 = A2-Y
    dW2 = 1./m*(np.dot(dZ2,A1.T))
    db2 = (1./m)*np.sum(dZ2,axis=1,keepdims=True)
    dZ1 = np.dot(W2.T,dZ2) * (1-np.power(A1,2))
    dW1 = (1./m)*np.dot(dZ1,X.T)
    db1 = (1./m)*np.sum(dZ1,axis=1,keepdims=True)
    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}
    
    return grads    

def update_parameters(parameters, grads, learning_rate = .09):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    
    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]
    W1 = W1 - learning_rate*dW1
    b1 = b1 - learning_rate*db1
    W2 = W2 - learning_rate*dW2
    b2 = b2 - learning_rate*db2
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters


def nn_model(X, Y, n_h, num_iterations = 10000, print_cost=False):
    np.random.seed(3)
    n_x = layer_sizes(X, Y)[0]
    n_y = layer_sizes(X, Y)[2]
    parameters = initalize_parameters(n_x, n_h, n_y)
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    for i in range(0, num_iterations):
        A2, cache = forward_propagation(X, parameters)
        #plotResults(X, Y, A2)
        cost = mean_squared_error(A2, Y)
        grads = backward_propagation(parameters, cache, X, Y)
        parameters = update_parameters(parameters, grads)
        plotResults(X, Y, A2)
        print ("Cost after iteration %i: %f" %(i, cost))
        '''
        if print_cost and i % 10 == 0:
            plotResults(X, Y, A2)
            print ("Cost after iteration %i: %f" %(i, cost))
        '''
    return parameters
    
X = np.arange(0.0, 2.1, .1).reshape((21,1))
'''
Y = np.array([0.7, 0.65, 0.57, 0.5, 0.48,
                0.4, 0.3, 0.15, 0.05, -0.2, 
                -0.3, -0.27, -0.25, -0.22, -0.2,
                -0.1, -0.05, 0, 0.1, 0.2, 0.2]).reshape((21,1))
'''

Y = np.array([0.7, -0.6, 0.5, -0.2, 0.01,
                0.2, -0.3, 0.5, -0.05, -0.3, 
                -0.5, -0.7, -0.3, -0.7, -0.2,
                0.1, -0.05, .5, 0.1, 0.2, 0.2]).reshape((21,1))
print("Input Shape: " + str(X.shape))
print("Output Shape: " + str(Y.shape))

plt.axhline(0, color='grey', ls =":")
plt.plot(X, Y, color="blue", linestyle='solid')
plt.axis([0, 2, -1, 1])
plt.show()

nn_model(X,Y, 3, 10,True)


'''
plt.axhline(0, color='grey', ls =":")
#plt.plot(x, results, color="black", linestyle='solid')
plt.plot(X, Y, color="blue", linestyle='solid')
plt.axis([0, 2, -1, 1])
plt.show()
n_x = layer_sizes(X, Y)[0]
n_y = layer_sizes(X, Y)[2]


parameters = initalizeParameters(n_x, 3, n_y)
W1 = parameters["W1"]
b1 = parameters["b1"]
W2 = parameters["W2"]
b2 = parameters["b2"]
#forward_propagation

Z2, cache = forward_propagation(X, parameters)

mse = mean_squared_error(Y, Z2)
#print(mse)

grads= backward_propagation(parameters,cache, X, Y)

update_parameters(parameters, grads)

#print(grads)

#print(Z2)
#plotResults(X, Y, Z2)
#plt.plot(X, Z2, color="blue", linestyle='solid')
#plt.show()
'''