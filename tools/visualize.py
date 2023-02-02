import matplotlib.pyplot as plt
import numpy as np


def save_figure(filepath, features, labels):
    if type(features) == list:
        features = np.array(features)
    plt.figure(figsize=(10, 10), dpi=100)
    print('features:', features.shape)
    if features.shape[1] == 2:
        scatter = plt.scatter(features[:, 0],
                              features[:, 1],
                              c=labels,
                              cmap='coolwarm')
    else:
        ax = plt.axes(projection='3d')
        scatter = ax.scatter3D(features[:, 0],
                               features[:, 1],
                               features[:, 2],
                               c=labels)
    plt.legend(*scatter.legend_elements(), loc="lower right", title="Classes")
    plt.savefig(filepath)


def save_hist(filepath, datas):
    plt.figure()
    n, bins, patches = plt.hist(datas)
    plt.savefig(filepath)


def save_plots(filepath,
               datas,
               format_strs='ro',
               labels='Original data',
               xlabel='Images',
               ylabel='Euc'):
    plt.figure()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if type(datas) == list:
        for data, format_str, label in zip(datas, format_strs, labels):
            plt.plot(data, format_str, label=label)
        plt.legend(labels)
    else:
        plt.plot(datas, format_strs, label=labels)
        plt.legend([labels])
    plt.savefig(filepath)