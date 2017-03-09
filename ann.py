
# Kewen Gu & Zhaochen Ding
# CS4341 Artificial Intelligence, Project 2
# Artificial Neural Networks

import random
import sys
import pylab as pl
import neural_net as nn


def main():

    # Default number of hidden layers and holdout percentage
    hidden = 5
    holdout = 0.2

    if len(sys.argv) > 6:
        print("Usage: python ann.py h [number of hidden nodes] p [holdout percentage]")
        sys.exit(1)

    for i in range(len(sys.argv)):
        if sys.argv[i] == 'h':
            hidden = int(sys.argv[i + 1])
        elif sys.argv[i] == 'p':
            holdout = float(sys.argv[i + 1])

    print("Number of Hidden Layers: \t%d" % hidden)
    print("Holdout Percentage: \t\t%.1f" % holdout)

    # Input checking
    if hidden < 0:
        raise ValueError("Number of hidden layers should be positive!")
    if holdout < 0 or holdout > 1:
        raise ValueError("Holdout percentage should be between 0 and 1!")

    # Read from file
    fp = open(sys.argv[1], "r")
    lines = fp.readlines()
    # print("Reading from file successful.")
    # print("**************************************")

    # Construct data patterns
    patterns = []

    for line in lines:
        array = line.split(" ")
        strX, strY, strData = array
        x = float(strX)
        y = float(strY)
        data = float(strData)
        input = [x, y]
        patterns.append([input, [data]])

    # Diving data set into training set and testing set
    length = len(patterns)
    random.seed()
    startIndex = random.randint(1, length - int(length * holdout)) - 1
    endIndex = startIndex + int(length * holdout)

    training = patterns[: startIndex] + patterns[endIndex :]
    testing = patterns[startIndex : endIndex]

    '''
    training = patterns[:160]
    testing = patterns[160:]
    '''

    # Training the training set using back propagation and validating the result using the testing set
    ann = nn.NeuralNet(2, hidden, 1)
    errors = ann.train(training)
    # print("**************************************")
    outputs = ann.validate(testing)

    # plot the results
    x = []
    y = []
    for e in errors:
        x.append(e[0])
        y.append(e[1])

    pl.plot(x, y)
    pl.show()

    x1 = []
    y1 = []
    x2 = []
    y2 = []
    for o in outputs:
        if o[1][0] == 0:
            x1.append(o[0][0])
            y1.append(o[0][1])
        elif o[1][0] == 1:
            x2.append(o[0][0])
            y2.append(o[0][1])

    pl.plot(x1, y1, 'bo')
    pl.plot(x2, y2, 'ro')
    # pl.show()

    x3 = []
    y3 = []
    x4 = []
    y4 = []
    for p in patterns:
        if p[1][0] == 0:
            x3.append(p[0][0])
            y3.append(p[0][1])
        elif p[1][0] == 1:
            x4.append(p[0][0])
            y4.append(p[0][1])

    pl.plot(x3, y3, 'bo')
    pl.plot(x4, y4, 'ro')
    # pl.show()


if __name__ == "__main__":
    main()
