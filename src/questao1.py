from functools import reduce
from operator import mul

import numpy as np

from generate_data import gen_data


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


def rand_index(u, K, n=300):
    table = np.zeros(K * K).reshape((K, K))
    it = iter(u)
    for k in range(K):  # K = 3
        for i in range(100):
            x = next(it)
            x_is_in = max(zip(x, range(K)))[-1]
            table[k][x_is_in] += 1

    sum_n = sum(c2(x) for x in table.reshape(K**2))
    sum_a = sum(c2(sum(table[:, k])) for k in range(K))
    sum_b = sum(3 * [c2(100)])
    numerador = sum_n - sum_a * sum_b / c2(n)
    denominador = (sum_a + sum_b) / 2. - sum_a * sum_b / c2(n)
    return numerador / denominador


def hard_partitions(u, data, K):
    pass


def main(N=100, K=3, new_data=False):
    if new_data:
        data = gen_data()
    else:
        try:
            data = np.loadtxt('all_data')
        except FileNotFoundError:
            data = gen_data()
    min_J = np.inf
    for i in range(N):
        print("Iteração {}:", i)
        j, u, p, lambdas = algorithm(data, K=K)
        if j < min_J:
            min_J = j
            best_u = u
            best_p = p
            best_lambdas = lambdas
    return data, best_u, best_p, rand_index(u, K), best_lambdas