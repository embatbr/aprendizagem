from gendata import gendata
import numpy as np


def partition(data, c):
    """Partitions the shuffled data.
    """
    step = int(len(data) / c)
    clusters = list()
    means = list()
    for i in range(c):
        cluster = data[i*step : (i + 1)*step]
        clusters.append(cluster)

        mean = (np.mean(cluster.T[0]), np.mean(cluster.T[1]))
        means.append(mean)

    means = np.array(means)
    return (means, clusters)

def kmeans(means, clusters, k):
    data = clusters[0]
    for cluster in clusters[1:]:
        data = np.concatenate((data, cluster))

    while(True):
        clusters = [list() for _ in range(k)]
        for x in data:
            distances = np.linalg.norm(x - means, axis=1)**2
            index = np.argmin(distances)
            clusters[index].append(x)

        clusters = [np.array(cluster) for cluster in clusters]

        oldmeans = means
        means = list()
        for cluster in clusters:
            mean = (np.mean(cluster.T[0]), np.mean(cluster.T[1]))
            means.append(mean)

        means = np.array(means)
        distances = np.linalg.norm(means - oldmeans, axis=1)**2

        if len(distances[np.where(distances > 0.01)]) == 0:
            break

    return (means, clusters)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    data = gendata()
    c = 3
    (means, clusters) = partition(data, c)
    for i in range(c):
        print('cluster %d' % i)
        print('mean =', means[i])
        print('num eltos:', len(clusters[i]))

    plt.grid(True)
    plt.plot(clusters[0].T[0], clusters[0].T[1], linestyle=':', linewidth=2)
    plt.plot(clusters[1].T[0], clusters[1].T[1], linestyle=':', linewidth=2)
    plt.plot(clusters[2].T[0], clusters[2].T[1], linestyle=':', linewidth=2)

    (means, clusters) = kmeans(means, clusters, c)
    print('\nkmeans')
    for i in range(c):
        print('cluster %d' % i)
        print('mean =', means[i])
        print('num eltos:', len(clusters[i]))

    plt.figure()
    plt.grid(True)
    plt.plot(clusters[0].T[0], clusters[0].T[1], linestyle=':', linewidth=2)
    plt.plot(clusters[1].T[0], clusters[1].T[1], linestyle=':', linewidth=2)
    plt.plot(clusters[2].T[0], clusters[2].T[1], linestyle=':', linewidth=2)

    plt.show()