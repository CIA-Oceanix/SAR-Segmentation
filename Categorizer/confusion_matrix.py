import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

LIMIT = 1000


def compute_confusion_matrix(model, generator, limit=LIMIT, canals=None):
    i = 0
    if canals is None:
        canals = next(generator)[1].shape[-1]

    confusion_matrix = np.zeros((canals, canals))

    while i < limit:
        inputs, groundtruths = next(generator)
        predictions = model.predict(inputs)
        for groundtruth, prediction in zip(groundtruths, predictions):
            groundtruth_arg = np.argmax(groundtruth)
            prediction_arg = np.argmax(prediction)
            confusion_matrix[groundtruth_arg, prediction_arg] += 1

            i += 1

    for i in range(confusion_matrix.shape[0]):
        confusion_matrix[i] /= np.sum(confusion_matrix[i])
    return confusion_matrix


def plot_confusion_matrix(confusion_matrix, labels=None):
    plt.figure(figsize=(18, 9))
    ax = plt.subplot()

    sns.heatmap(confusion_matrix[::-1], annot=True, ax=ax, vmin=0, vmax=1, fmt='.2f')

    # labels, title and ticks
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    if labels is not None:
        ax.xaxis.set_ticklabels(labels[::-1])
        ax.yaxis.set_ticklabels(labels, rotation=0)

    # https://stackoverflow.com/questions/56942670/matplotlib-seaborn-first-and-last-row-cut-in-half-of-heatmap-plot
    ax.set_ylim(confusion_matrix.shape[0], 0)
    ax.set_xlim(confusion_matrix.shape[0], 0)
