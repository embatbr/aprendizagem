from bases import gendata, readdata
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
    """Divides the elements into k clusters.
    """
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

    (r11, r12, r2) = readdata(False)
    mean11 = (np.mean(r11.T[0]), np.mean(r11.T[1]))
    mean12 = (np.mean(r12.T[0]), np.mean(r12.T[1]))
    mean2 = (np.mean(r2.T[0]), np.mean(r2.T[1]))

    fig = plt.gcf()
    plt.plot(r11.T[0], r11.T[1], 'r. ')
    plt.plot(r12.T[0], r12.T[1], 'g. ')
    plt.plot(r2.T[0], r2.T[1], 'b. ')
    plt.plot(mean11[0], mean11[1], 'ro ')
    plt.plot(mean12[0], mean12[1], 'go ')
    plt.plot(mean2[0], mean2[1], 'bo ')
    plt.savefig('outputs/kmeans/original_data.png')

    data = readdata()
    c = 3
    (means, clusters) = partition(data, c)
    for i in range(c):
        print('cluster %d' % i)
        print('mean =', means[i])
        print('num eltos:', len(clusters[i]))

    fig = plt.figure()
    plt.plot(clusters[0].T[0], clusters[0].T[1], 'r. ')
    plt.plot(clusters[1].T[0], clusters[1].T[1], 'g. ')
    plt.plot(clusters[2].T[0], clusters[2].T[1], 'b. ')
    plt.plot(means[0][0], means[0][1], 'ro ')
    plt.plot(means[1][0], means[1][1], 'go ')
    plt.plot(means[2][0], means[2][1], 'bo ')
    plt.savefig('outputs/kmeans/partitioned_data.png') # by Random Partitions

    (means, clusters) = kmeans(means, clusters, c)
    print('\nkmeans')
    for i in range(c):
        print('cluster %d' % i)
        print('mean =', means[i])
        print('num eltos:', len(clusters[i]))

    fig = plt.figure()
    plt.plot(clusters[0].T[0], clusters[0].T[1], 'r. ')
    plt.plot(clusters[1].T[0], clusters[1].T[1], 'g. ')
    plt.plot(clusters[2].T[0], clusters[2].T[1], 'b. ')
    plt.plot(means[0][0], means[0][1], 'ro ')
    plt.plot(means[1][0], means[1][1], 'go ')
    plt.plot(means[2][0], means[2][1], 'bo ')
    plt.savefig('outputs/kmeans/classified_data.png')

    plt.show()