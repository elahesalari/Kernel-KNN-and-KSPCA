import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import time
from KernelKNN import KernelKNN
from KernelSupervisedPCA import KernelPCA
from matplotlib import pyplot as plt


def read_data(dtset: str):
    data = pd.read_csv(f'Datasets\\{dtset}', header=None, sep=',|,\s+|\t', engine='python')
    data.iloc[:, -1] = data.iloc[:, -1].astype('category').cat.codes
    labels = data.iloc[:, -1]

    normalizedData = (data - data.mean(axis=0)) / data.std(axis=0)
    data = normalizedData
    data[np.isnan(data)] = 0

    return np.array(data), np.array(labels)


def split(x: np.ndarray, y: np.ndarray):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    x_train = np.array(x_train)[:, :-1]
    x_test = np.array(x_test)[:, :-1]
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    return x_train, x_test, y_train, y_test


def plot(X: np.ndarray, X_s: list, Xt: np.ndarray, Xt_s: list, y: np.ndarray, yt: np.ndarray, fn: str, s: list):
    fig = plt.figure(figsize=(6, 7))
    row = len(X_s) // 3 + 1
    row += 1 if (len(X_s) + 1) % 3 == 0 else 0
    ax_main = fig.add_subplot(4, 3, 1)
    ax_projs = [fig.add_subplot(4, 3, i) for i in range(2, len(X_s) + 2)]  # , projection='3d'
    fig.suptitle(fn)
    ax_main.set_title('Main Data')
    colors = [('midnightblue', 'deepskyblue'), ('darkred', 'red'),
              ('green', 'lime'), ('orange', 'gold')]
    C = np.unique(y).astype(int)
    legends = []
    for c in C:
        ax_main.scatter(X[y == c, 0], X[y == c, 1], ec='k', label=f'C{c + 1}', c=colors[c][0])
        ax_main.scatter(Xt[yt == c, 0], Xt[yt == c, 1], ec='k', label=f'C{c + 1} Test', c=colors[c][1])
        legends.append(f'C{c + 1}')
        legends.append(f'C{c + 1} Test')

    for i in range(len(X_s)):
        ax_projs[i].set_title(f'\nKernel PCA (s = {s[i]:<3.1})', fontsize=8)
        for c in C:
            ax_projs[i].scatter(X_s[i][y == c, 0], X_s[i][y == c, 1], ec='k', label=f'C{c + 1}', c=colors[c][0],
                                alpha=0.7)
            ax_projs[i].scatter(Xt_s[i][yt == c, 0], Xt_s[i][yt == c, 1], ec='k', label=f'C{c + 1} Test',
                                c=colors[c][1], alpha=0.7)

    fig.legend(legends, fontsize=10, loc=4, markerscale=1)
    plt.tight_layout()
    plt.show()

    if not os.path.exists('Plots/'):
        os.mkdir('Plots')
    fig.savefig(f'Plots/{fn[:-4]}.jpg')


if __name__ == '__main__':

    # ----------------------------kernel KNN------------------------------------

    knn = KernelKNN()
    datasets = ['BreastTissue.txt', 'Diabetes.txt', 'Glass.txt', 'Ionosphere.txt', 'Sonar.txt', 'Wine.txt']
    kernels = ['1NN', 'linear', 'RBF', 'polynomial1', 'polynomial2', 'polynomial3']
    d = [1, 2, 3]
    dict = {}
    for dt in datasets:
        dict[dt] = {}
        # print(f'dataset:{dt}')
        data, labels = read_data('Kernel-KNN\\' + dt)

        for k in kernels:
            dict[dt][k] = {}
            st_time = time.time()
            total_accuracy = []

            for i in range(10):
                x_train, x_test, y_train, y_test = split(data, labels)

                acc = knn.kernel_knn(k, x_train, x_test, y_train, y_test)
                total_accuracy.append(acc)

            mean_accuracy = np.mean(total_accuracy)

            end_time = time.time()
            total = end_time - st_time

            dict[dt][k]['acc'] = f'{mean_accuracy:5.1f}'
            dict[dt][k]['time'] = f'{total:0.4f}'

    knn.table(dict, datasets, key='acc')
    knn.table(dict, datasets, key='time')

    # ---------------------------- KS-PCA ------------------------------------

    print(' Results '.center(45, '=') + '\n')

    dataset_list = ['Binary_XOR.txt', 'Concentric_rectangles.txt',
                    'Concentric_rings.txt', 'Twomoons.txt']

    for fn in dataset_list:
        X, y = read_data(f'KS-PCA\\{fn}')
        train_X, test_X, train_y, test_y = split(X, y)
        kpca = KernelPCA()
        X_s, Xt_s, s_ = [], [], []
        for s in np.arange(0.1, 1.0, 0.1):
            try:
                X_, Xt_ = kpca.kpca(train_X, train_y, test_X, s)
                X_s.append(X_)
                Xt_s.append(Xt_)
                s_.append(s)
            except Exception as e:
                print(f"Sigma {s:<5.2} didn't work on {fn}")
        plot(train_X, X_s, test_X, Xt_s, train_y, test_y, fn, s_)
