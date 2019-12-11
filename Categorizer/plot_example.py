import matplotlib.pyplot as plt

from Rignak_Misc.plt import imshow

LIMIT = 8

def plot_example(input_, prediction, labels, limit=LIMIT):
    input_ = input_[:limit]
    n = input_.shape[0]
    plt.figure(figsize=(18, 8))
    plt.tight_layout()
    for i, (im, classes) in enumerate(zip(input_, prediction)):
        if i != 0:
            tick_label = [' ' for label in labels]
        else:
            tick_label = labels
        plt.subplot(2, n, i + 1)
        imshow(im)

        plt.subplot(2, n, i + 1 + n)
        plt.barh(labels, classes, tick_label=tick_label)
        plt.xlim(0, 1)
    plt.tight_layout()
