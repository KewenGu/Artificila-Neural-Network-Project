
# Kewen Gu & Zhaochen Ding
# CS4341 Artificial Intelligence, Project 2
# Artificial Neural Networks

import random
import numpy as np


class NeuralNet:
    def __init__(self, nInput, nHidden, nOutput):
        # Number of nodes in input, hidden, and output layers
        self.nInput = nInput
        self.nHidden = nHidden
        self.nOutput = nOutput

        # Activations for input, hidden, and output nodes
        self.aInput = [1.0] * self.nInput
        self.aHidden = [1.0] * self.nHidden
        self.aOutput = [1.0] * self.nOutput

        # Weights for input and output nodes
        self.wInput = makeMatrix(self.nInput, self.nHidden)
        self.wOutput = makeMatrix(self.nHidden, self.nOutput)

        # Make the weight matrices for both input and output layers
        # Randomly assign value to each weight
        for i in range(self.nInput):
            for j in range(self.nHidden):
                self.wInput[i][j] = rand(-0.1, 0.1)
        for j in range(self.nHidden):
            for k in range(self.nOutput):
                self.wOutput[j][k] = rand(-1.0, 1.0)


    def ANN(self, inputs):
        if len(inputs) != self.nInput:
            raise ValueError('Incorrect number of inputs.')

        # The activation of the input layer is the inputs themselves
        for i in range(self.nInput):
            # self.aInput[i] = sigmoid(inputs[i])
            self.aInput[i] = inputs[i]

        # Calculate activations for the hidden layer
        for j in range(self.nHidden):
            sum = 0.0
            for i in range(self.nInput):
                sum += self.aInput[i] * self.wInput[i][j]
            self.aHidden[j] = sigmoid(sum)

        # Calculate activations for the output layer
        for k in range(self.nOutput):
            sum = 0.0
            for j in range(self.nHidden):
                sum += self.aHidden[j] * self.wOutput[j][k]
            self.aOutput[k] = sigmoid(sum)

        return self.aOutput[:]


    # Backpropagation Method for training the training set
    # rate is the learning rate, which is set to 0.5
    def backPropagate(self, targets, rate):
        if len(targets) != self.nOutput:
            raise ValueError("Wrong number of target values!")

        # Calculate deltas for the output layer
        output_deltas = [0.0] * self.nOutput
        for k in range(self.nOutput):
            error = targets[k] - self.aOutput[k]
            output_deltas[k] = dsigmoid(self.aOutput[k]) * error

        # Calculate deltas for the hidden layer
        hidden_deltas = [0.0] * self.nHidden
        for j in range(self.nHidden):
            error = 0.0
            for k in range(self.nOutput):
                error += output_deltas[k] * self.wOutput[j][k]
            hidden_deltas[j] = dsigmoid(self.aHidden[j]) * error

        # Update weights for the output layer
        for j in range(self.nHidden):
            for k in range(self.nOutput):
                change = output_deltas[k] * self.aHidden[j]
                self.wOutput[j][k] += rate * change

        # Update weights for the input layer
        for i in range(self.nInput):
            for j in range(self.nHidden):
                change = hidden_deltas[j] * self.aInput[i]
                self.wInput[i][j] += rate * change

        # Calculate cumulative error
        error = 0.0
        for k in range(len(targets)):
            error += 0.5 * (targets[k] - self.aOutput[k]) ** 2
        return error


    # Method for testing the training result
    def validate(self, patterns):
        outputs = []
        i = 0
        errcnt = 0
        for p in patterns:
            i += 1
            output = round(self.ANN(p[0])[0], 0)
            outputs.append([p[0], [output]])
            if output != p[1][0]:
                err = True
                errcnt += 1
            else:
                err = False
            # print i, '. \tInputs:', p[0], '\t\t---> \tOutput: ', [output], '\tTarget: ', p[1], '\tError: ', err
        print("\nTesting Count: \t%d" % i)
        print("Error Count: \t%d" % errcnt)
        print("Error Rate: \t%.3f" % (float(errcnt)/float(i)))
        return outputs


    # Function used to train neural network
    def train(self, patterns, epochs=1000, rate=0.15):
        # N: learning rate
        # M: momentum factor
        errors = []
        for i in range(epochs):
            error = 0.0
            for p in patterns:
                inputs = p[0]
                target = p[1]
                self.ANN(inputs)
                error += self.backPropagate(target, rate)

            if i % 100 == 0:
                # print('epoch = %d' % i + '\t\terror = %0.5f' % error)
                errors.append([i, error])
        return errors


# calculate a random number where:  a <= rand < b
def rand(a, b):
    return (b - a) * random.random() + a


# make an empty matrix with dimension I*J
def makeMatrix(i, j):
    m = []
    for i in range(i):
        m.append([0.0] * j)
    return m


# sigmoid threshold function 1/(1+e^-x)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# derivative of the _sigmoid function
def dsigmoid(y):
    return y * (1 - y)
