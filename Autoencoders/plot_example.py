import numpy as np
import matplotlib.pyplot as plt


def plot_example(input_, prediction, groundtruth=None, min_value=0, max_value=1, max_thumbs=8):
    input_ = input_[:, :, :, ::-1][:max_thumbs]
    prediction = prediction[:, :, :, ::-1]
    if groundtruth is None:
        n = 2
    else:
        n = 3

    plt.figure(figsize=(20, 8))
    for i, im in enumerate(input_):
        plt.subplot(n, len(input_), 1 + i)
        plt.imshow(im)
        plt.subplot(n, len(input_), i + 1 + len(input_))
        if prediction.shape[-1] == 1:
            plt.imshow(prediction[i][:, :, 0], vmin=min_value, vmax=max_value, cmap='hot')
        else:
            plt.imshow(prediction[i], vmin=min_value, vmax=max_value)
        if n == 3:
            plt.subplot(n, len(input_), i + 1 + len(input_) * 2)
            if groundtruth.shape[-1] == 1:
                plt.imshow(groundtruth[i, :, :][:, :, 0], cmap='hot', vmin=min_value, vmax=max_value)
            else:
                plt.imshow(groundtruth[i, :, :, ::-1], vmin=min_value, vmax=max_value)

    plt.tight_layout()
