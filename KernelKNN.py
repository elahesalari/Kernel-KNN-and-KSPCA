import numpy as np


class KernelKNN:

    def __init__(self):
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None

    def linear(self, test):
        x_train = self.x_train
        x_test = test

        k_xtrain = (np.linalg.norm(x_train, axis=1) ** 2) + 1
        k_xtest = (np.linalg.norm(x_test) ** 2) + 1
        k_train_test = x_train.dot(x_test) + 1
        distance = k_xtrain + k_xtest - (2 * k_train_test)

        return distance

    def RBF(self, test, sigma):
        train = self.x_train

        k_train = np.exp((- np.linalg.norm(train - train) ** 2) / (2 * (sigma ** 2)))
        k_test = np.exp((- np.linalg.norm(test - test) ** 2) / (2 * (sigma ** 2)))
        k_train_test = np.exp(- np.linalg.norm(train - test, axis=1) ** 2 / (2 * (sigma ** 2)))
        distance = k_train + k_test - (2 * k_train_test)

        return distance

    def polynomial(self, test, d):
        alpha = 0.3
        train = self.x_train
        k_train = (alpha * (np.linalg.norm(train, axis=1) ** 2) + 1) ** d
        k_test = (alpha * (np.linalg.norm(test) ** 2) + 1) ** d
        k_train_test = (alpha * train.dot(test) + 1) ** d
        distance = k_train + k_test - (2 * k_train_test)

        return distance

    def kernel_knn(self, kernel, x_train, x_test, y_train, y_test):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

        sigma = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

        accuracy = 0
        if kernel == 'RBF':
            for s in sigma:
                pred_rbf = []
                for test in self.x_test:
                    dis = self.RBF(test, s)
                    min_dis = np.argmin(dis)
                    pred_rbf.append(self.y_train[min_dis])
                acc_rbf = np.mean(pred_rbf == self.y_test) * 100
                if acc_rbf > accuracy:
                    accuracy = acc_rbf
                    best_sigma = s
            # print(f'Kernel: {kernel} ,Best sigma:{best_sigma} and Accuracy:{accuracy:0.2f} ')
        else:
            predict = []
            for test in self.x_test:
                if kernel == '1NN':
                    dis = np.linalg.norm(self.x_train - test, axis=1)    # eucledian distance

                elif kernel == 'linear':
                    dis = self.linear(test)

                elif kernel == 'polynomial1':
                    dis = self.polynomial(test, d=1)

                elif kernel == 'polynomial2':
                    dis = self.polynomial(test, d=2)

                elif kernel == 'polynomial3':
                    dis = self.polynomial(test, d=3)

                min_dis = np.argmin(dis)
                predict.append(self.y_train[min_dis])
            accuracy = np.mean(predict == self.y_test) * 100

        return accuracy

    def table(self, dict, datasets, key):
        print('╔' + '═' * 20 + '╦' + '═' * 127 + '╗')
        if key == 'acc':
            print('║' + ' ' * 20 + f'║{"Accuracy of Algorithms ":^127}' + '║')
        elif key == 'time':
            print('║' + ' ' * 20 + f'║{"Running Time of Algorithms (seconds) ":^127}' + '║')

        print('║' + f'{" Datasets ":^20}' + '╟ ' + '-' * 125 + ' ╢')
        print(
            '║' + ' ' * 20 + '║   1NN ' + '|'.center(3) + '1NN+LinearKernel' + '|'.center(
                3) + '1NN+RBFKernel' + '|'.center(
                3) +
            '1NN+PolynomialKernel(d=1)' + '|'.center(3) + '1NN+PolynomialKernel(d=2)' + '|'.center(
                3) + '1NN+PolynomialKernel(d=3) ' + '║')

        print('╠' + '═' * 20 + '╬' + '═' * 127 + '╣')

        size_sp = [5, 10, 15, 20, 20, 8]

        kernels = ['1NN', 'linear', 'RBF', 'polynomial1', 'polynomial2', 'polynomial3']
        for dt in datasets:
            # print(f'{dt}', end='')
            print(f'║{dt[:-4]:^20}║', end='')
            for i in range(len(kernels)):
                accur = dict[dt][kernels[i]][key]
                if key == 'acc':
                    lp = str(accur + '%')
                elif key == 'time':
                    lp = str(accur)
                # print(f'{"  " + lp + int(size_sp[i] / 2) * " " + "|" + round(size_sp[i] / 2) * " "}', end='')
                print(f'{"  " + lp + size_sp[i] * " "}', end='')
            print(f'{" "}║')
            if dt != 'Wine.txt':
                print('╟' + '─' * 20 + '╫' + '─' * 127 + '╢')

        print('╚' + '═' * 20 + '╩' + '═' * 127 + '╝')
