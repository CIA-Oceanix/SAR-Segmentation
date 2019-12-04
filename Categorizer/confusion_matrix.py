import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from bct import modularity_und

LIMIT = 1000

MEAN_CLUSTER_SIZE = 3
GAMMAS = [x / 10 for x in range(20, 0, -1)]
GAMMAS = [0.8]


def compute_confusion_matrix(model, generator, limit=LIMIT, canals=None):
    i = 0
    if canals is None:
        canals = next(generator)[1].shape[-1]

    confusion_matrix = np.zeros((canals, canals))

    while i < limit:
        inputs, groundtruths = next(generator)
        predictions = model.predict(inputs)
        if len(predictions) == 2:  # added to support the bimode
            predictions = predictions[0]
            groundtruths = groundtruths[0]
        for groundtruth, prediction in zip(groundtruths, predictions):
            if groundtruth.shape[0] == 2:
                groundtruth = groundtruth[0]
                prediction = prediction[0]
            groundtruth_arg = np.argmax(groundtruth)
            prediction_arg = np.argmax(prediction)
            confusion_matrix[groundtruth_arg, prediction_arg] += 1

            i += 1

    for i in range(confusion_matrix.shape[0]):
        confusion_matrix[i] /= np.sum(confusion_matrix[i])
    return confusion_matrix


def plot_confusion_matrix(confusion_matrix, labels=None, clustering=True):
    if clustering:
        confusion_matrix, labels = create_clusters(confusion_matrix, labels)
        print(confusion_matrix.shape, labels.shape)

    plt.figure(figsize=(18, 9))
    ax = plt.subplot()

    sns.heatmap(confusion_matrix[::-1], annot=True, ax=ax, vmin=0, vmax=1, fmt='.2f')

    # labels, title and ticks
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    if labels is not None:
        ax.xaxis.set_ticklabels(labels, rotation=30)
        ax.yaxis.set_ticklabels(labels[::-1], rotation=0)

    # https://stackoverflow.com/questions/56942670/matplotlib-seaborn-first-and-last-row-cut-in-half-of-heatmap-plot
    ax.set_ylim(confusion_matrix.shape[0], 0)
    ax.set_xlim(confusion_matrix.shape[0], 0)
    plt.tight_layout()


def create_clusters(confusion_matrix, labels, mean_cluster_size=MEAN_CLUSTER_SIZE, gammas=GAMMAS):
    def select_best_gamma(confusion_matrix, gammas, mean_cluster_size):
        for gamma in gammas:
            clusters = [(element, i) for i, element in enumerate(modularity_und(confusion_matrix, gamma=gamma)[0])]
            clusters.sort()
            if confusion_matrix.shape[0] / clusters[-1][0] > mean_cluster_size:
                clusters = [i for (element, i) in clusters]
                return clusters
        return clusters

    clusters = select_best_gamma(confusion_matrix, gammas, mean_cluster_size)
    confusion_matrix = confusion_matrix[clusters][:, clusters]
    if labels:
        labels = np.array(labels)[clusters]

    return confusion_matrix, labels
