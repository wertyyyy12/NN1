import math
import random
import numpy

#[[input]    [hidden]  [hidden]  [output]]
#neuron: [net, out]
#3 input neurons, 4 hidden layer 1, 3 hidden layer 2, 3 output

neurons = [numpy.zeros((1, 3)), numpy.zeros((1, 4)), numpy.zeros((1, 3)), numpy.zeros((1, 3))]
# neurons[0] = input
# neurons[1] = hidden1
# neurons[2] = hidden2
# neurons[3] = output

#intialize weights randomly
weights = [numpy.random.uniform(low=-0.316, high=0.316, size=(3, 4)), numpy.random.uniform(low=-0.316, high=0.316, size=(4, 3)), numpy.random.uniform(low=-0.316, high=0.316, size=(3, 3))]
biases = [0, 0, 0]


random.seed()

#sigmoid function
def sigmoid(x):
    return 1 / (1 + numpy.exp(-x))
      
def feedforward():
    for i in range(len(neurons)):
        if i != 0:
            #W = sigmoid(weighted sum + bias)
            neurons[i] = sigmoid(neurons[i-1].dot(weights[i-1]) + biases[i-1])

        print(neurons[i])


def train():
    #feed a random input 

        #initialize random r g b values
    r = random.randrange(0, 255, 1)
    b = random.randrange(0, 255, 1)
    g = random.randrange(0, 255, 1)

        #convert the 0-255 values to 0-1 values
    rN = numpy.interp(r, [0, 255], [0, 1])
    bN = numpy.interp(b, [0, 255], [0, 1])
    gN = numpy.interp(g, [0, 255], [0, 1])

        #set input neurons to the 0-1 values of the random colors
    neurons[0] = numpy.array([[rN, bN, gN]])
    feedforward()

    # #output-h2 backprop
    # for i in range(len(weights)):
    #     if (i == len(weights) - 1): #last weight
    #         weightDer = 
    #         weights[len(weights) - i]




train()