"""Training and classification using a MultiLayer Perceptron (MLP) neural network.
"""


import numpy as np
import neurolab as nl


def newff(minmax, size, transf=None):
    net_ci = len(minmax)
    net_co = size[-1]

    if transf is None:
        transf = [nl.trans.TanSig()] * len(size)
    assert len(transf) == len(size)

    layers = []
    for i, nn in enumerate(size):
        layer_ci = size[i - 1] if i > 0 else net_ci
        l = nl.layer.Perceptron(layer_ci, nn, transf[i])
        layers.append(l)
    connect = [[i - 1] for i in range(len(layers) + 1)]

    net = nl.core.Net(minmax, net_co, layers, connect, nl.train.train_gdx, nl.error.MSE())
    return net


if __name__ == '__main__':
    from bases import gendata, readdata
    import matplotlib.pyplot as plt

    # classe 1 = 1, classe 2 = -1
    (data1_1, data1_2, data2) = readdata(False, False)
    training_data = np.concatenate((data1_1[ : 75], data1_2[ : 75], data2[ : 75]))
    training_classes = np.concatenate((np.ones(150, np.int), np.zeros(75, np.int)))
    test_data = np.concatenate((data1_1[75 : ], data1_2[75 : ], data2[75 : ]))
    test_classes = np.concatenate((np.ones(50, np.int), np.zeros(25, np.int)))

    indices = np.array(range(len(training_classes)))
    np.random.shuffle(indices)
    training_data = training_data[indices]
    training_classes = training_classes[indices]
    print(training_classes)

    # defining limits
    lim_inp1 = [np.amin(training_data[:, 0]), np.amax(training_data[:, 0])]
    lim_inp2 = [np.amin(training_data[:, 1]), np.amax(training_data[:, 1])]
    training_classes = training_classes.reshape(len(training_classes), 1)

    layers = [5, 2, 3, 1] # cada valor determina o numero de neuronios por layer
    #net = nl.net.newff([lim_inp1, lim_inp2], layers)
    net = newff([lim_inp1, lim_inp2], layers)
    error = net.train(training_data, training_classes, epochs=1000)#, epochs=500, show=100, goal=1)
    out = net.sim(test_data)

    #print(error)
    print(out)