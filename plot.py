# -*- coding: utf-8 -*-
#-------------------------------------------------------------------------------------#
#  Name:           plot.py                                                            #
#  Description:    plot graphs for demonstration                                      #
#-------------------------------------------------------------------------------------#
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

cm1 = ListedColormap(['#666666', '#FF0000'])
cm2 = ListedColormap(['#0000FF', '#FF0000'])
np.random.seed(3)
grid_step = 0.02
quartersize = 50
halfsize = quartersize * 2
size = halfsize * 2


X = np.random.randn(quartersize * 4, 2)
X[halfsize:] += 5
x1_min, x1_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
x2_min, x2_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

y = np.concatenate((np.ones(quartersize), np.zeros(quartersize * 3)), axis=0)
y_true = np.concatenate((np.ones(halfsize), np.zeros(halfsize)), axis=0)


def set_lim_ticks(ax):
    ax.set_xlim(x1_min, x1_max)
    ax.set_ylim(x2_min, x2_max)
    ax.set_xticks(())
    ax.set_yticks(())


def plot_general():
    fig = plt.figure()

    ax1 = fig.add_subplot(121, aspect='equal')
    ax1.scatter(X[:, 0], X[:, 1], c=y, cmap=cm1, edgecolors='k')
    set_lim_ticks(ax1)

    ax2 = fig.add_subplot(122, aspect='equal')
    ax2.plot([x1_min, x1_max], [x2_max, x2_min], 'g--', linewidth=6)
    ax2.scatter(X[:, 0], X[:, 1], c=y_true, cmap=cm2, edgecolors='k')
    set_lim_ticks(ax2)

    fig.tight_layout()


def plot_kfold():
    X[:halfsize] = np.array(sorted(X[:halfsize], key=lambda x: sum(x)))
    X[quartersize:halfsize] += 0.3

    fig = plt.figure()

    ax1 = fig.add_subplot(131, aspect='equal')
    ax1.scatter(X[:, 0], X[:, 1], c=y, cmap=cm1, edgecolors='k')
    set_lim_ticks(ax1)

    ax2 = fig.add_subplot(132, aspect='equal')
    ax2.plot([x1_min, 3], [3, x2_min], 'g--', linewidth=6)
    ax2.scatter(X[:, 0], X[:, 1], c=y, cmap=cm2, edgecolors='k')
    set_lim_ticks(ax2)

    ax3 = fig.add_subplot(133, aspect='equal')
    ax3.plot([x1_min, x1_max], [x2_max, x2_min], 'g--', linewidth=6)
    ax3.scatter(X[:, 0], X[:, 1], c=y_true, cmap=cm2, edgecolors='k')
    set_lim_ticks(ax3)

    fig.tight_layout()


def plot_spy2():
    y_tmp = y.copy()
    maxid = np.argmax(np.sum(X, axis=1))
    y_tmp[maxid] = 1

    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, 0.02), np.arange(x2_min, x2_max, 0.02))
    zz = xx1 + xx2 - np.mean(xx1 + xx2)

    fig  = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')
    ax.contourf(xx1, xx2, zz, cmap=plt.cm.RdBu, alpha=0.9)

    ax.scatter(X[quartersize // 3:, 0], X[quartersize // 3:, 1],
                c=y_tmp[quartersize // 3:], cmap=cm1, edgecolors='k')
    ax.scatter(X[:quartersize // 3, 0],
                X[:quartersize // 3, 1], c='y', edgecolors='k')
    ax.scatter(X[maxid, 0], X[maxid, 1], c='y', edgecolors='k')

    set_lim_ticks(ax)


def plot_spy():

    y_tmp = y.copy()
    maxid = np.argmax(np.sum(X, axis=1))
    y_tmp[maxid] = 1
    fig = plt.figure()

    ax1 = fig.add_subplot(141, aspect='equal')
    ax1.scatter(X[quartersize // 3:, 0], X[quartersize // 3:, 1],
                c=y_tmp[quartersize // 3:], cmap=cm1, edgecolors='k')
    ax1.scatter(X[:quartersize // 3, 0],
                X[:quartersize // 3, 1], c='y', edgecolors='k')
    ax1.scatter(X[maxid, 0], X[maxid, 1], c='y', edgecolors='k')
    set_lim_ticks(ax1)

    ax2 = fig.add_subplot(142, aspect='equal')
    ax2.plot([x1_min, 17.5], [17.5, x2_min], 'g--', linewidth=6)
    ax2.scatter(X[:, 0], X[:, 1], c=y,
                cmap=cm1, edgecolors='k')
    set_lim_ticks(ax2)

    ax3 = fig.add_subplot(143, aspect='equal')
    ax3.plot([x1_min, x2_max], [x1_max, x2_min], 'g--', linewidth=6)
    ax3.scatter(X[:quartersize, 0], X[:quartersize, 1], c='r', edgecolors='k')
    ax3.scatter(X[quartersize:, 0], X[quartersize:, 1],
                c='#666666', cmap=cm1, edgecolors='k')
    ax3.scatter(X[halfsize:, 0], X[halfsize:, 1],
                c='b', cmap=cm1, edgecolors='k')
    set_lim_ticks(ax3)

    ax3 = fig.add_subplot(144, aspect='equal')
    ax3.plot([x1_min, x1_max], [x2_max, x2_min], 'g--', linewidth=6)
    ax3.scatter(X[:, 0], X[:, 1], c=y_true, cmap=cm2, edgecolors='k')
    set_lim_ticks(ax3)

    fig.tight_layout()


def main():
    plot_general()
    plot_kfold()
    plot_spy2()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
