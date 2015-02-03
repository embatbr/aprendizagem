import numpy as np


def kNN(x, k, data):
    """Returns the indices for the k nearest neighbours.
    """
    neighbours = list()
    num_neighbours = 0

    distances = np.linalg.norm(x - data, axis=1)    # euclidian distance
    indices = distances.argsort()[1 : k + 1]

    return indices

def classify(train_data, test_data, train_classes, test_classes):
    """Classifies a dataset given the training data and classes and the expected
    classes.
    """
    P1 = np.zeros(len(test_classes))
    P2 = np.zeros(len(test_classes))
    classified = np.zeros(len(test_classes))
    i = 0

    for x in test_data:
        indices = kNN(x, k, train_data)
        neighbours = train_data[indices]
        neighbours_classes = train_classes[indices]

        k1 = len(neighbours_classes[neighbours_classes == 1])
        k2 = len(neighbours_classes[neighbours_classes == 2])
        P1[i] = k1 / k
        P2[i] = k2 / k
        classified[i] = 2 if (k2 > k1) else 1
        i = i + 1

    equal = (classified == test_classes)
    numequal = len(equal[equal == True])
    hits = numequal / len(test_classes)

    return (hits, P1, P2, classified)


if __name__ == '__main__':
    from bases import gendata, readdata
    import matplotlib.pyplot as plt

    (data1_1, data1_2, data2) = readdata(False, False)
    train_data = np.concatenate((data1_1[ : 75], data1_2[ : 75], data2[ : 75]))
    train_classes = np.concatenate((np.ones(150, np.int8), 2*np.ones(75, np.int8)))
    test_data = np.concatenate((data1_1[75 : ], data1_2[75 : ], data2[75 : ]))
    test_classes = np.concatenate((np.ones(50, np.int8), 2*np.ones(25, np.int8)))

    num_ks = 9
    hits = np.zeros(num_ks)
    P1 = np.zeros((num_ks, len(test_classes)))
    P2 = np.zeros((num_ks, len(test_classes)))
    classified = np.zeros((num_ks, len(test_classes)))

    for k in range(1, num_ks + 1):
        (hits[k - 1], P1[k - 1], P2[k - 1], classified[k - 1]) = classify(train_data, test_data,
                                                       train_classes, test_classes)

        fig = plt.figure()
        fig.suptitle('P(w1|x)\nk-NN, k = %d' % k)
        plt.grid(True)
        plt.fill_between(np.linspace(1, 75, 75), P1[k - 1], color='blue')
        plt.plot(np.linspace(1, 75, 75), classified[k - 1], 'g.')
        plt.xlim(1, 75)
        plt.ylim(-0.1,2.1)
        plt.savefig('outputs/kNN-k%d-P1.png' % k)

        fig = plt.figure()
        fig.suptitle('P(w2|x)\nk-NN, k = %d' % k)
        plt.grid(True)
        plt.fill_between(np.linspace(1, 75, 75), P2[k - 1], color='red')
        plt.plot(np.linspace(1, 75, 75), classified[k - 1], 'g.')
        plt.xlim(1, 75)
        plt.ylim(-0.1,2.1)
        plt.savefig('outputs/kNN-k%d-P2.png' % k)

    # plot hits and misses
    fig = plt.figure()
    fig.suptitle('Hits (blue) and Misses (red)')
    plt.grid(True)
    plt.xlabel('K')
    plt.plot(np.linspace(1, 9, 9), hits, 'b')
    plt.plot(np.linspace(1, 9, 9), 1 - hits, 'r')
    plt.ylim(-0.1,1.1)
    plt.savefig('outputs/kNN-hits-and-misses.png')
    print('hits:', hits)

    plt.show()