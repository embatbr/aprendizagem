from bases import gendata, readdata
import numpy as np


def kNN(x, k, data):
    """Return the indices for the k nearest neighbours.
    """
    neighbours = list()
    num_neighbours = 0

    distances = np.linalg.norm(x - data, axis=1)    # euclidian distance
    indices = distances.argsort()[1 : k + 1]

    return indices

def prob_nKK(k, data, classes):
    P_w1_given_x = list()
    P_w2_given_x = list()

    for x in data:
        indices = kNN(x, k, data)
        neighbours = data[indices]
        neighbourclasses = classes[indices]

        k1 = len(neighbourclasses[neighbourclasses == 1])
        P_w1_given_xk = k1/k
        k2 = len(neighbourclasses[neighbourclasses == 2])
        P_w2_given_xk = k2/k

        P_w1_given_x.append(P_w1_given_xk)
        P_w2_given_x.append(P_w2_given_xk)

    P_w1_given_x = np.array(P_w1_given_x)
    P_w2_given_x = np.array(P_w2_given_x)

    return (P_w1_given_x, P_w2_given_x)

def classify(data, classes, P1, P2):
    """Classify a data set given the data, the classes and the conditional
    probabilities (as a matrix with each class probability in a row).
    """
    c = P_w1_given_x - P_w2_given_x
    c[c >= 0] = 1
    c[c < 0] = 2
    equal = (c == classes)
    numequal = len(equal[equal == True])
    hits = numequal / len(data)

    return (c, hits)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    data = readdata(shuffle=False)
    classes = np.ones(200, np.int8)
    classes = np.concatenate((classes, 2*np.ones(100, np.int8)))
    hits = np.zeros(9)

    for k in range(1, 10):
        (P_w1_given_x ,P_w2_given_x) = prob_nKK(k, data, classes)
        (_, hits[k - 1]) = classify(data, classes, P_w1_given_x, P_w2_given_x)

        fig = plt.figure()
        fig.suptitle('P(w1|x)\nk-NN, k = %d' % k)
        plt.grid(True)
        plt.fill_between(np.linspace(1, 300, 300), P_w1_given_x, color='blue')
        plt.ylim(-0.1,1.1)
        plt.savefig('outputs/kNN-k%d-P_w1_given_x.png' % k)

        fig = plt.figure()
        fig.suptitle('P(w2|x)\nk-NN, k = %d' % k)
        plt.grid(True)
        plt.fill_between(np.linspace(1, 300, 300), P_w2_given_x, color='red')
        plt.ylim(-0.1,1.1)
        plt.savefig('outputs/kNN-k%d-P_w2_given_x.png' % k)

    fig = plt.figure()
    fig.suptitle('Hits (blue) and Misses (red)')
    plt.grid(True)
    plt.xlabel('K')
    plt.plot(np.linspace(1, 9, 9), hits, 'b')
    plt.plot(np.linspace(1, 9, 9), 1 - hits, 'r')
    plt.ylim(-0.1,1.1)
    plt.savefig('outputs/kNN-hits-and-misses.png')
    print('hits:', hits)