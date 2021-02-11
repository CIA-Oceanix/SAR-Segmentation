import numpy as np
import matplotlib.pyplot as plt

from Rignak_Misc.plt import imshow


def plot_example(examples, truths, predictions, labels):
    plt.figure(figsize=(4 * 4, 3 * 4))
    threshold = 1/len(labels)

    for i, example in zip(range(8), examples):
        if i < len(example):
            plt.subplot(3, 4, 1 + i)
            imshow(example, vmax=1)

            predicted_tags = '¤'.join([f'{label}:({e:.2f})' for e, label in zip(predictions[i], labels) if e > threshold])
            while '¤' in predicted_tags:
                predicted_tags = predicted_tags.replace('¤', '\n')
                predicted_tags = predicted_tags.replace('¤', ', ')
            plt.title(predicted_tags, fontdict={'horizontalalignment': 'right'})

    recall = []
    precision = []
    f1 = []
    accuracy = []
    for i in range(len(labels)):
        true_positive = np.mean([pred > threshold and truth > threshold for pred, truth in zip(predictions[:, i], truths[:, i])])
        true_negative = np.mean([pred < threshold and truth < threshold for pred, truth in zip(predictions[:, i], truths[:, i])])

        false_negative = np.mean([pred < threshold < truth for pred, truth in zip(predictions[:, i], truths[:, i])])
        false_positive = np.mean([pred > threshold > truth for pred, truth in zip(predictions[:, i], truths[:, i])])

        recall.append(true_positive / (true_positive + false_negative))
        precision.append(true_positive / (true_positive + false_positive))
        f1.append(2 * recall[-1] * precision[-1] / (recall[-1] + precision[-1]))
        accuracy.append((true_positive + true_negative) / (true_positive + true_negative + false_negative + false_positive))

    plt.subplot(3, 4, 9)
    plt.barh(labels[::-1], recall[::-1], tick_label=labels[::-1])
    plt.xlim(0, 1)
    plt.title('Recall')

    plt.subplot(3, 4, 10)
    plt.barh(labels[::-1], precision[::-1], tick_label=labels[::-1])
    plt.xlim(0, 1)
    plt.title('Precision')

    plt.subplot(3, 4, 11)
    plt.barh(labels[::-1], f1[::-1], tick_label=labels[::-1])
    plt.xlim(0, 1)
    plt.title('F1-score')

    plt.subplot(3, 4, 12)
    plt.barh(labels[::-1], accuracy[::-1], tick_label=labels[::-1])
    plt.xlim(0, 1)
    plt.title('Accuracy')
    plt.tight_layout()
