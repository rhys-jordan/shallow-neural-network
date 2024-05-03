# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 10:24:00 2024

@author: rhysj
"""

import matplotlib.pyplot as plt
import numpy as np

def reLU(out):
    if (out < 0):
        out = 0 
    return out

'''
Tried to run this with sigmoid function but the results were not very good
Would have to continue to mess with the weights to see what sigmoid can do
'''
def sigmoid(out):
    out1 = 1/(1+(np.e)**(-out))
    return out1

#Adds all the hidden layers  
def predictedOutput(layers, x, bias):
    final = []
    for i in range(len(x)):
        y = b_out + layers[0][i] + layers[1][i] + layers[2][i]
        final.append(y)
    return final


def plotLine(line, x, pltNum):
    plt.subplot(3,3,pltNum)
    plt.axhline(0, color='grey', ls =":")
    plt.plot(x, line, color = "blue", linestyle='solid')
    plt.axis([0, 2, -1, 1])
    
   
'''
This Function does all the logic of creating linear functions with 
the prechosen parameters. Then applying the activation function and 
weighting the output. Finally it adds all the hidden layers and adds
the bias.  
'''
def visualizeHiddenLayers(weights, xAxis):
    weightedLayers = []
    weightIndex = 4;
    for node in range(3):
        line = []
        hiddenLayer = []
        weightedHiddenLayer = []
        for i in range(len(xAxis)):
            line.append(weights[weightIndex+1] * xAxis[i] + weights[weightIndex])
            hiddenLayer.append(reLU(line[i]))
            weightedHiddenLayer.append(weights[node+1]*hiddenLayer[i])
        weightIndex += 2
        weightedLayers.append(weightedHiddenLayer)
        plotLine(line, x, node+1)
        plotLine(hiddenLayer, x, node+4)
        plotLine(weightedHiddenLayer, x, node+7)
    plt.show()
    return predictedOutput(weightedLayers, x, weights[0])
    
  
'''
Parameters that I messed with to ultimatly find what looked like the image in 
the text book. Each line I messed with differnt slopes and intercepts. Then figured out
how to weight that output after the activation function. Finally I figured out the bias
on the whole function. 
'''   
b_10 = -0.25
w_11 = 0.7

b_20 = -.9
w_21 = .8

b_30 = 1.1
w_31 = -.8

w_1 = -1
w_2 = 1.7
w_3 = .6

b_out = .1

#Found it was easier to put weights in a list 
#Eventally learned that a matrix is was eaiser then the list 
weights = [b_out, w_1, w_2, w_3,
           b_10, w_11, 
           b_20, w_21,
           b_30, w_31]

x = np.arange(0.0, 2.1, .1)

output = []

results = visualizeHiddenLayers(weights, x)

#Plots the output line
plt.axhline(0, color='grey', ls =":")
plt.plot(x, results, color="black", linestyle='solid')
plt.axis([0, 2, -1, 1])
plt.show()
