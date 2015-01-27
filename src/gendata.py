import numpy as np
from numpy.random import multivariate_normal as gaussian


# Classe 1, subclasse 1
means1_1 = np.array([60, 30])
cov1_1 = np.diag(np.array([9, 144]))

# Classe 1, subclasse 2
means1_2 = np.array([52, 30])
cov1_2 = np.diag(np.array([9, 9]))

# Classe 2
means2 = np.array([45, 22])
cov2 = np.diag(np.array([100, 9]))


def gendata(order=False):
    """Gera os dados, salvado-os no data.txt.

    Caso order seja true, Order.txt salva a posição original dos dados,
    antes de embaralhar.
    """
    data1_1 = gaussian(means1_1, cov1_1, 100)
    data1_2 = gaussian(means1_2, cov1_2, 100)
    data2 = gaussian(means2, cov2, 100)
    data = np.concatenate((data1_1, data1_2, data2))
    if not order:
        np.random.shuffle(data)
        return data
    order = np.array(range(300))
    np.random.shuffle(order)
    shuffled_data = data[order]
    np.savetxt('order.txt', order)
    np.savetxt('data.txt', shuffled_data)
    return shuffled_data, order


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    data = gendata()

    plt.grid(True)
    plt.plot(data.T[0], data.T[1], linestyle=':', linewidth=2)
    plt.show()
