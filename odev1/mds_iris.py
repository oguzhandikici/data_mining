from sklearn.manifold import MDS
import matplotlib.pyplot as plt
from sklearn import datasets


def plot_MDS():
    '''
    graph after MDS
    :param data: train_data, train_value
    :return: None
    '''

    iris = datasets.load_iris()

    X = iris.data
    y = iris.target
    target_names = iris.target_names

    mds = MDS(n_components=2)
    X_r = mds.fit_transform(X)

    # Graph
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    colors = ((1, 0, 0), (0, 1, 0), (0, 0, 1), (0.5, 0.5, 0), (0, 0.5, 0.5), (0.5, 0, 0.5),
              (0.4, 0.6, 0), (0.6, 0.4, 0), (0, 0.6, 0.4), (0.5, 0.3, 0.2),)
    for color, i, target_name in zip(colors, [0, 1, 2], target_names):
        position = y == i
        plt.scatter(X_r[position, 0], X_r[position, 1], color=color, alpha=.8, label=target_name)

    ax.set_xlabel("X[0]")
    ax.set_ylabel("X[1]")
    ax.legend(loc="best")
    ax.set_title("MDS")
    plt.show()


plot_MDS()
