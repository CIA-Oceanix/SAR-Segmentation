import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from bct import modularity_und

LIMIT = 1000

MEAN_CLUSTER_SIZE = 3
GAMMAS = [x / 10 for x in range(20, 0, -1)]

GAMMAS = [0.7]


def compute_confusion_matrix(model, generator, limit=LIMIT, canals=None):
    i = 0
    if canals is None:
        _, batch_output = next(generator)
        canals = batch_output.shape[-1]

    confusion_matrix = np.zeros((canals, canals))

    while i < limit:
        batch_input, batch_output = next(generator)
        predictions = model.predict(batch_input)
        for groundtruth, prediction in zip(batch_output, predictions):
            groundtruth_arg = np.argmax(groundtruth)
            prediction_arg = np.argmax(prediction)
            confusion_matrix[groundtruth_arg, prediction_arg] += 1

            i += 1

    for i in range(confusion_matrix.shape[0]):
        line_sum = np.sum(confusion_matrix[i])
        if line_sum:
            confusion_matrix[i] = confusion_matrix[i] / line_sum
    return confusion_matrix


def plot_confusion_matrix(confusion_matrix, labels=None, clustering=False, figsize=(18, 9), fmt='.2f', ax=None,
                          title=None, cmap='magma', vmax=None,
                          rotation=30):
    if len(labels) == 2:
        clustering = False
    if clustering:
        confusion_matrix, labels, lines = create_clusters(confusion_matrix, labels)

    if ax is None:
        plt.figure(figsize=figsize)
        ax = plt.subplot()

    sns.heatmap(confusion_matrix[::-1], annot=True, ax=ax, vmin=0, vmax=1 if vmax is None else vmax, fmt=fmt, cmap=cmap)

    # labels, title and ticks
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title(title if title is not None else 'Confusion Matrix')
    if labels is not None:
        ax.xaxis.set_ticklabels(labels, rotation=rotation, horizontalalignment='right')
        ax.yaxis.set_ticklabels(labels[::-1], rotation=0)

    # https://stackoverflow.com/questions/56942670/matplotlib-seaborn-first-and-last-row-cut-in-half-of-heatmap-plot
    ax.set_ylim(confusion_matrix.shape[0], 0)
    ax.set_xlim(confusion_matrix.shape[0], 0)

    if clustering:
        for line in lines:
            ax.axhline(line, color=(.5, .5, .5))
            ax.axvline(confusion_matrix.shape[0] - line, color=(.5, .5, .5))

    plt.tight_layout()


def create_clusters(confusion_matrix, labels, mean_cluster_size=MEAN_CLUSTER_SIZE, gammas=GAMMAS):
    def get_lines(indexs):
        lines = [0]
        current_index = 0
        for index in indexs:
            if current_index == index:
                lines[-1] += 1
            else:
                lines.append(lines[-1])
                current_index = index
        return lines

    def select_best_gamma(confusion_matrix, gammas, mean_cluster_size):
        for gamma in gammas:
            clusters = [(element, i) for i, element in enumerate(modularity_und(confusion_matrix, gamma=gamma)[0])]
            clusters.sort()
            if confusion_matrix.shape[0] / clusters[-1][0] > mean_cluster_size:
                break

        lines = get_lines([element for (element, i) in clusters])
        clusters = [i for (element, i) in clusters]
        return clusters, lines

    clusters, lines = select_best_gamma(confusion_matrix, gammas, mean_cluster_size)
    confusion_matrix = confusion_matrix[clusters][:, clusters]
    if labels:
        labels = np.array(labels)[clusters]

    return confusion_matrix, labels, lines
