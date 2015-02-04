import numpy as np
import kNN


NUM_SETS = 10


# cross-validation do parzen
parzen_valid_err = np.loadtxt('files/cross-validation/parzen.txt')
parzen_valid_meanerr = np.mean(parzen_valid_err)
print('h = 1\nparzen_valid_meanerr = 0.5111\n')
print('h = 0.5\nparzen_valid_meanerr: %f\n' % parzen_valid_meanerr)
print('h = 0.3\nparzen_valid_meanerr = 0.2111\n')

# cross-validation do kNN
kNN_valid_err = np.array(range(NUM_SETS))
for conj in range(NUM_SETS):
    training_data = np.loadtxt('files/cross-validation/sets/train%d.txt' % conj)
    training_indices = np.array(training_data[:, 2], dtype=np.int)
    training_data = training_data[:, : 2]
    training_classes = np.ones(len(training_data), dtype=np.int)
    print(training_classes.shape)
    print(training_classes)

    test_data = np.loadtxt('files/cross-validation/sets/test%d.txt' % conj)
    test_indices = np.array(test_data[:, 2], dtype=np.int)

    (kNN_valid_err[conj], _, _, _) = kNN.classify(training_data, test_data,
                                                  training_classes, test_classes)

    break