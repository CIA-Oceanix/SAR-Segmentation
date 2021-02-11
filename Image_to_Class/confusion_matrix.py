import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

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


def plot_confusion_matrix(confusion_matrix, labels=None, clustering=False, figsize=(18, 9), fmt='.2f', ax=None, title=None, cmap='magma', vmax=None,
                          rotation=30):
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


    plt.tight_layout()

