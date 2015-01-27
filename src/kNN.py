from bases import gendata, readdata
import numpy as np


def kNN(x, k, data):
    neighbours = list()
    num_neighbours = 0

    distances = np.linalg.norm(x - data, axis=1)    # euclidian distance
    indices = distances.argsort()[1 : k + 1]

    return indices


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    data = readdata(shuffle=False)
    dataclasses = np.ones(200, np.int8)
    dataclasses = np.concatenate((dataclasses, 2*np.ones(100, np.int8)))

    for k in range(1, 10):
        P_w1_given_x = list()
        P_w2_given_x = list()
        for x in data:
            indices = kNN(x, k, data)
            neighbours = data[indices]
            classes = dataclasses[indices]

            k1 = len(classes[classes == 1])
            P_w1_given_xk = k1/k
            k2 = len(classes[classes == 2])
            P_w2_given_xk = k2/k

            P_w1_given_x.append(P_w1_given_xk)
            P_w2_given_x.append(P_w2_given_xk)

        fig = plt.gcf()
        fig.suptitle('P(w1|x)\nk-NN, k = %d' % k)
        plt.grid(True)
        plt.plot(P_w1_given_x)
        plt.ylim(-0.1,1.1)
        plt.savefig('outputs/kNN-k%d-P_w1_given_x.png' % k)

        fig = plt.figure()
        fig.suptitle('P(w2|x)\nk-NN, k = %d' % k)
        plt.grid(True)
        plt.plot(P_w2_given_x, 'r')
        plt.ylim(-0.1,1.1)
        plt.savefig('outputs/kNN-k%d-P_w2_given_x.png' % k)

        plt.show()