import matplotlib.pyplot as plt


def plot_example(input_, prediction, labels):
    n = input_.shape[0]
    plt.figure(figsize=(18, 8))
    plt.tight_layout()
    for i, (im, classes) in enumerate(zip(input_, prediction)):
        if i != 0:
            tick_label = [' ' for label in labels]
        else:
            tick_label = labels
        plt.subplot(2, n, i + 1)
        plt.imshow(im[:, :, ::-1])

        plt.subplot(2, n, i + 1 + n)
        plt.barh(labels, classes, tick_label=tick_label)
        plt.xlim(0, 1)
    plt.tight_layout()
