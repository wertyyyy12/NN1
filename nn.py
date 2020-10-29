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
weightGradients = [[], [], []]
biasGradients = [[], [], []]
biases = [numpy.zeros((1, 4)), numpy.zeros((1, 3)), numpy.zeros((1, 3))]


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

#gradientCollection = collection of matrices with weight gradients
def gradientAvg(gradientCollection):
    sum = 0
    gradientCollectionLength = len(gradientCollection)
    for i in range(gradientCollectionLength):
        #cumulative sum of matrices for the gradient collection
        sum = sum + gradientCollection[i]
    finalAvg = sum / gradientCollectionLength
    return finalAvg

def train(learningRate, epochs, batchSize):
    for _ in range(epochs):
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
        # the * operator for matrices is a 1 to 1 multiplication, only matching elements are multiplied
        # same with subtraction 
        scalarDerivatives = (output * (1 - output)) * (output - complementaryColor)

        #scaledWeights contains the derivative for every weight, in a matrix
        scaledWeights = []
        for i in range(len(scalarDerivatives)):
            lastLayerWeights = weights[-1]
            splitWeights = numpy.split(numpy.transpose(lastLayerWeights), 3)
            #dy/dx = w * Oi * (1 - Oi) * (Oi - Yi)
            #          | --------- Scalars -------|
            scaledWeights.append(splitWeights[i].dot(scalarDerivatives[i])[0])
        scaledWeights = numpy.transpose(numpy.array(scaledWeights))

        #set weights to new values using gradient descent but on a matrix scale
        weightGradients[-1].append(scaledWeights)
        # print(batchSize)
        # print(len(gradients[0]))
        if (batchSize == len(weightGradients[-1])):
            #avg of gradients calculated so far
            outputH2gradientAvg = gradientAvg(weightGradients[-1])
            weights[-1] = weights[-1] - (learningRate * outputH2gradientAvg)

            #clear the gradient "queue"
            weightGradients[-1] = []
        
        #bias gradient is same as weight gradient but without multiplying by any weight (just 1)
        #dy/dx = 1 * Oi * (1 - Oi) * (Oi - Yi)
        #          | --------- Scalars -------|
        #                   (from before!)

        biasGradients[-1].append(numpy.array(scalarDerivatives))

        if (batchSize == len(biasGradients[-1])):
            #avg of gradients calculated so far
            outputH2bias_gAvg = gradientAvg(biasGradients[-1])
            biases[-1] = biases[-1] - (learningRate * outputH2bias_gAvg)

            #clear the gradient "queue"
            biasGradients[-1] = []
            
        



train(0.5, 10, 2)