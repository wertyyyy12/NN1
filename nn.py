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


#sigmoid function
def sigmoid(x):
    return 1 / (1 + numpy.exp(-x))
      
def feedforward():
    print("Neurons: ")
    for i in range(len(neurons)):
        if i != 0:
            #W = sigmoid(weighted sum + bias)
            neurons[i] = sigmoid(neurons[i-1].dot(weights[i-1]) + biases[i-1])

        print(neurons[i])

    print(" ")


def train(learningRate):
    #feed a random input 
    neurons[0] = numpy.interp(numpy.random.uniform(high=255, size=(1, 3)), [0, 255], [0, 1])
    
    feedforward()
    #find target values for given input (complimentary color, for simplicity)
        #takes neurons 0-1 values
        #converts to 0-255 values (using interp)
        #subtracts these from the white color: (255, 255, 255) - (output1, output2, output3)
        #converts this new difference into a 0-1 value again (interp)
        #strips an array layer caused by using [[]] arrays the whole time (with [0] at the end)
    complementaryColor = numpy.interp(numpy.array([[255, 255, 255]]) - numpy.interp(neurons[0], [0, 1], [0, 255]), [0, 255], [0, 1])[0]

    #output-h2 backprop
    output = neurons[-1][0]
    # the * operator for 
    scalarDerivatives = (output * (1 - output)) * (output - complementaryColor)
    #scaledWeights contains the derivative for every weight in a matrix
    scaledWeights = []
    for i in range(len(scalarDerivatives)):
        lastLayerWeights = weights[-1]
        print("Weights: ")
        print(lastLayerWeights)
        splitWeights = numpy.split(numpy.transpose(lastLayerWeights), 3)
        #dy/dx = w * Oi * (1 - Oi) * (Oi - Yi)
        #          | --------- Scalars -------|
        scaledWeights.append(splitWeights[i].dot(scalarDerivatives[i])[0])
    scaledWeights = numpy.transpose(numpy.array(scaledWeights))
    print(scaledWeights)
    print(scaledWeights.shape)

    #set weights to new values using gradiant descent but on a matrix scale
    weights[-1] = weights[-1] - (learningRate * scaledWeights)
    print(weights[-1])


    # print("")
    # print(numpy.array(scaledWeights))
    # print(numpy.array(scaledWeights).shape)
    # print(numpy.array(scaledWeights).reshape(3, 3))
        # for i in range(len(scalarDerivatives[0])):




train(0.5)