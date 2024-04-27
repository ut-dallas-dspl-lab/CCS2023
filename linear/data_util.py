import random
import numpy as np


class MNISTMultiLabelSample():
    def __init__(self, problem_description):
        self.problem_description = problem_description   

    # taking original data, make new problems based on the problem_description
    def make_mnist_samples(self, X_train, y_train, X_test, y_test):
        n_features = X_train.shape[1]*X_train.shape[2]

        if self.problem_description == 'even_odd':
            # convert to binary classification
            c = [0,2,4,6,8]
        elif self.problem_description == 'greater':
            c = [0,1,2,3,4]
        elif self.problem_description == 'zero':
            c = [0]
        else:
            print('no problem given, returning the original')
            return X_train, y_train, X_test, y_test

        X_train_c = np.reshape(X_train, (-1, n_features))
        y_train_c = np.isin(y_train, c)
        y_train_c = np.where(y_train_c == True, 1, -1)
        y_train_c = np.reshape(y_train_c, (X_train_c.shape[0],))

        X_test_c = np.reshape(X_test, (-1, n_features))
        y_test_c = np.isin(y_test, c)
        y_test_c = np.where(y_test_c == True, 1, -1)
        y_test_c = np.reshape(y_test_c, (X_test_c.shape[0],))

        return X_train_c, y_train_c, X_test_c, y_test_c


