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


def train():
    #feed a random input 
    neurons[0] = numpy.interp(numpy.random.uniform(high=255, size=(1, 3)), [0, 255], [0, 1])
    
    feedforward()
    #find target values for given input (complimentary color, for simplicity)
        #takes neurons 0-1 values
        #converts to 0-255 values (using interp)
        #subtracts these from the white color: (255, 255, 255) - (output1, output2, output3)
        #converts this new difference into a 0-1 value again (interp)
    complementaryColor = numpy.interp(numpy.array([[255, 255, 255]]) - numpy.interp(neurons[0], [0, 1], [0, 255]), [0, 255], [0, 1])

    #output-h2 backprop
    output = neurons[-1]
    scalarDerivatives = ((output * (1 - output)) * (output - complementaryColor))[0]
    print(neurons)
    print(scalarDerivatives)
    scaledWeights = []
    for i in range(len(scalarDerivatives)):
        lastLayerWeights = weights[-1]
        splitWeights = numpy.split(numpy.transpose(lastLayerWeights), 3)
        print("Weights: ")
        print(lastLayerWeights)
        print("")
        #dy/dx = w * Oi * (1 - Oi) * (Oi - Yi)
        #          | --------- Scalars -------|
        print("Scalars: ")
        print(scalarDerivatives[0])
        print("")
        print(splitWeights[i])
        print(scalarDerivatives[i])

        print(splitWeights[i].dot(scalarDerivatives[i])[0])
        scaledWeights.append(splitWeights[i].dot(scalarDerivatives[i])[0])
    print("S0: ")
    print(scaledWeights)
    print(numpy.array(scaledWeights))
    print(numpy.transpose(numpy.array(scaledWeights)))
    print(numpy.array(scaledWeights).shape)
    # print("")
    # print(numpy.array(scaledWeights))
    # print(numpy.array(scaledWeights).shape)
    # print(numpy.array(scaledWeights).reshape(3, 3))
        # for i in range(len(scalarDerivatives[0])):




train()