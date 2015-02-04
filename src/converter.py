import numpy as np


tipo = ['train', 'test']
index = 0
lista = list()

arquivo = open('files/sets.txt')

k = 0
for line in arquivo:
    if line == '\n':
        output = open('files/cross-validation/sets/%s%d.txt' % (tipo[index], k), 'w')
        for l in lista[: -1]:
            output.write('%s\n' % l)
        output.write('%s' % lista[-1])

        lista = list()
        index = (index + 1) % 2
        if index == 0:
            k = k + 1
    else:
        line = line.strip()
        line = line.split()
        line = '%s %s %s' % (line[0], line[1], line[2])
        lista.append(line)