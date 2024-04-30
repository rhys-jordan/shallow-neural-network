# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 10:24:00 2024

@author: rhysj
"""

import matplotlib.pyplot as plt
import numpy as np
import random


def reLU(out):
    if (out < 0):
        out = 0 
    return out


def sigmoid(out):
    out1 = 1/(1+(np.e)**(-out))
    return out1


def visualizeResult(layers, x, bias):
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
    return visualizeResult(weightedLayers, x, weights[0])
    



'''
def loss_function(output, expected):
    loss = 
'''   
    

random.seed(100)
#b_10 = random.random()

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
#x = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]



weights = [b_out, w_1, w_2, w_3,
           b_10, w_11, 
           b_20, w_21,
           b_30, w_31]


x = np.arange(0.0, 2.1, .1)
#x = [0,3,4,6,9]
expected_out = [0.7, 0.65, 0.57, 0.5, 0.48,
                0.4, 0.3, 0.15, 0.05, -0.2, 
                -0.3, -0.27, -0.25, -0.22, -0.2,
                -0.1, -0.05, 0, 0.1, 0.2, 0.2]
output = []

results = visualizeHiddenLayers(weights, x)

print(x)
print(results)

plt.axhline(0, color='grey', ls =":")
plt.plot(x, results, color="black", linestyle='solid')
#plt.plot(x, expected_out, color="blue", linestyle='solid')
plt.axis([0, 2, -1, 1])
plt.show()


'''

h_1 = [sigmoid(w_11 * i + b_10) for i in x]
h_1_2 = [w_11 * i + b_10 for i in x] 
h_2 = [sigmoid(w_21 * i + b_20) for i in x]
h_2_2 = [w_21 * i + b_20 for i in x]
h_3 = [sigmoid(w_31 * i + b_30) for i in x]
h_3_2 = [w_31 * i + b_30 for i in x]

w_h_1 = [w_1*i for i in h_1]
w_h_2 = [w_2*i for i in h_2]
w_h_3 = [w_3*i for i in h_3]

final = []

for i in range(len(x)):
    y = b_out + w_h_1[i] + w_h_2[i] + w_h_3[i]
    final.append(y)

#h_1_p = reLU()

plt.subplot(3,3,1)
plt.axhline(0, color='grey', ls =":")
plt.plot(x, h_1_2, color='orange', linestyle='solid')
plt.axis([0, 2, -1, 1])

plt.subplot(3,3,2)
plt.axhline(0, color='grey', ls =":")
#plt.plot(x, h_2, linestyle='solid')
plt.plot(x, h_2_2, linestyle='solid')
plt.axis([0, 2, -1, 1])

plt.subplot(3,3,3)
plt.axhline(0, color='grey', ls =":")
#plt.plot(x, h_3, linestyle='solid')
plt.plot(x, h_3_2,color='green', linestyle='solid')
plt.axis([0, 2, -1, 1])

plt.subplot(3,3,4)
plt.axhline(0, color='grey', ls =":")
plt.plot(x, h_1, color='orange', linestyle='solid')
plt.axis([0, 2, -1, 1])

plt.subplot(3,3,5)
plt.axhline(0, color='grey', ls =":")
plt.plot(x, h_2, linestyle='solid')
plt.axis([0, 2, -1, 1])

plt.subplot(3,3,6)
plt.axhline(0, color='grey', ls =":")
plt.plot(x, h_3, color='green',linestyle='solid')
plt.axis([0, 2, -1, 1])

plt.subplot(3,3,7)
plt.axhline(0, color='grey', ls =":")
plt.plot(x, w_h_1,  color='orange', linestyle='solid')
plt.axis([0, 2, -1, 1])

plt.subplot(3,3,8)
plt.axhline(0, color='grey', ls =":")
plt.plot(x, w_h_2, linestyle='solid')
plt.axis([0, 2, -1, 1])

plt.subplot(3,3,9)
plt.axhline(0, color='grey', ls =":")
plt.plot(x, w_h_3,color='green', linestyle='solid')
plt.axis([0, 2, -1, 1])

plt.show()

plt.axhline(0, color='grey', ls =":")
plt.plot(x, final, color="black", linestyle='solid')
plt.axis([0, 2, -1, 1])
plt.show()

#plt.plot(x, h_2, linestyle='solid')
#plt.plot(x, h_3, linestyle='solid')

'''

'''
for q in x:
    y = b_out + w_1*(reLU(b_10 + w_11*(q))) + w_2*(reLU(b_20 + w_21*(q))) + w_3*(reLU(b_30 + w_31*(q)))
    
    output.append(y)
'''
#print(output)
'''
plt.plot(x, h_1, linestyle='solid')
#plt.plot(x, h_2, linestyle='solid')
#plt.plot(x, h_3, linestyle='solid')
plt.show()
'''
'''
plt.plot(x, expected_out)
plt.plot(x, output)
plt.show()
'''

