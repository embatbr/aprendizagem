from functools import reduce
from operator import mul

from matplotlib import pyplot as plt
import numpy as np

from bases import gendata


def prod(iterable):
    return reduce(mul, iterable, 1)


def mmul_3(a, b):  # aT * b * a
    return np.dot(np.dot(a.T, b), a)


def c2(n):  # n - dois a dois
    return n * (n - 1) / 2


def algorithm(data, K):
    m = 2
    n = data.shape[0]  # 300
    p = data.shape[1]  # 2
    T = 150
    e = 0.0000000001
    t = 0
    prototypes = data[np.random.choice(n, K, replace=False)]
    lambdas = np.ones((K, p))

    u = np.zeros(n * K).reshape((n, K))
    for i, x in enumerate(data):
        for k in range(K):
            numerador = sum(lambdas[k] * (x - prototypes[k])**2)
            soma = 0
            for h in range(K):
                denominador = sum(lambdas[h] * (x - prototypes[h])**2)
                soma += (numerador / denominador)  # **(1./m-1) ... m = 2
            soma = soma**(-1)
            u[i][k] = soma if not np.isnan(soma) else 1

    J = sum(u[i][k]**m * mmul_3(data[k] - prototypes[k], np.diag(lambdas[k]))
            for i in range(n) for k in range(K))

    while t < T:
        t = t + 1
        for k in range(K):
            for j in range(p):
                prototypes[k][j] = sum(
                    u[i][k]**m * data[i][j]
                    for i in range(0, n)) / sum(u[i][k]**m for i in range(0, n))
        for k in range(K):
            for j in range(p):
                numerador = prod(sum(
                    (u[i][k]**m * (data[i][h] - prototypes[k][h])**2 for i in range(n))) for h in range(p))**(1 / p)
                denominador = sum(
                    u[i][k]**m * (data[i][j] - prototypes[k][j])**2 for i in range(n))
                lambdas[k][j] = numerador / denominador

        for i, x in enumerate(data):
            for k in range(K):
                numerador = sum(lambdas[k] * (x - prototypes[k])**2)
                soma = 0
                for h in range(K):
                    denominador = sum(
                        lambdas[h] * (x - prototypes[h])**2)
                    # **(1./m-1) ... m = 2
                    soma += (numerador / denominador)
                soma = soma**(-1)
                u[i][k] = soma if not np.isnan(soma) else 1

        new_J = sum(u[i][k]**m * mmul_3(data[k] - prototypes[k], np.diag(lambdas[k]))
                    for i in range(n) for k in range(K))
        if abs(new_J - J) < e:
            print("convergência.")
            break
        J = new_J
    return J, u, prototypes, lambdas


def rand_index3x3(u, order, K, n=300):
    table = np.zeros(K * K).reshape((K, K))
    for i in range(300):
        priori_class = order[i] // 100
        cluster = max(zip(u[i], range(K)))[-1]
        table[priori_class][cluster] += 1

    sum_n = sum(c2(x) for x in table.reshape(K**2))
    sum_a = sum(c2(sum(table[:, k])) for k in range(K))
    sum_b = sum(c2(sum(table[k, :])) for k in range(K))
    assert(sum_b == 3 * c2(100))
    numerador = sum_n - sum_a * sum_b / c2(n)
    denominador = (sum_a + sum_b) / 2. - sum_a * sum_b / c2(n)
    return numerador / denominador


def rand_index2x3(u, order, K, n=300):
    table = np.zeros(2 * K).reshape((2, K))
    for i in range(300):
        priori_class = 0 if order[i] < 200 else 1
        cluster = max(zip(u[i], range(K)))[-1]
        table[priori_class][cluster] += 1

    sum_n = sum(c2(x) for x in table.reshape(K * 2))
    sum_a = sum(c2(sum(table[:, k])) for k in range(K))
    sum_b = sum(c2(sum(table[k, :])) for k in range(2))
    assert(sum_b == c2(200) + c2(100))
    numerador = sum_n - sum_a * sum_b / c2(n)
    denominador = (sum_a + sum_b) / 2. - sum_a * sum_b / c2(n)
    return numerador / denominador


def hard_partitions(u, data, K):
    cluster_data = []
    plt.clf()
    for k in range(K):
        cluster_data.append(
            np.array([data[i] for i in range(300)
                      if max(zip(u[i], range(K)))[-1] == k]))
    plt.plot(cluster_data[0][:, 0], cluster_data[0][:, 1], 'r.')
    plt.plot(cluster_data[1][:, 0], cluster_data[1][:, 1], 'g.')
    plt.plot(cluster_data[2][:, 0], cluster_data[2][:, 1], 'b.')
    plt.show()

    return cluster_data


def main(N=100, K=3, new_data=False):
    if new_data:
        data, order = gendata(save_order=True)
    else:
        try:
            data = np.loadtxt('data.txt')
            order = np.loadtxt('order.txt', np.intc)
        except FileNotFoundError:
            data, order = gendata(save_order=True)
    min_J = np.inf
    for i in range(N):
        print("Iteração {}:".format(i))
        j, u, p, lambdas = algorithm(data, K=K)
        if j < min_J:
            min_J = j
            best_u = u
            best_p = p
            best_lambdas = lambdas
    print("Prototipos: {}".format(best_p))
    rand = rand_index2x3(u, order, K)
    print("Indice de Rand(2x3): {}".format(rand))
    rand = rand_index3x3(u, order, K)
    print("Indice de Rand(3x3): {}".format(rand))
    return data, order, best_u, best_p, rand, best_lambdas, hard_partitions(best_u, data, K)