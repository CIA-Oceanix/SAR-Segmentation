import scipy
from glob import glob
import numpy as np
import os
import cv2

from Rignak_Misc.path import get_local_file

DATASET_ROOT = get_local_file(__file__, '.')


def augment_data(batch_input, zoom=0.0, rotation=0):
    def uniform_noise(x, f=0.15):
        xmax = np.max(x)
        xmin = np.min(x)
        noise = f * x.std() * (2 * np.random.random(x.shape) - 1)
        x = x.astype('float64')
        x += noise
        x[x > xmax] = xmax
        x[x < xmin] = xmin
        return x

    center = (batch_input.shape[1] // 2, batch_input.shape[2] // 2)

    angles = (np.random.random(batch_input.shape[0]) - 0.5) * rotation
    zooms = np.random.random(batch_input.shape[0]) * zoom + 1
    h_flips = np.random.randint(0, 2, batch_input.shape[0])

    for i, (img_input, angle, zoom, h_flip) in enumerate(zip(batch_input, angles, zooms, h_flips)):
        if h_flip:
            img_input = img_input[:, :-1]
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, zoom)
        batch_input[i] = cv2.warpAffine(img_input, rotation_matrix, batch_input.shape[1:3])

    batch_input = uniform_noise(batch_input)

    return batch_input


def normalize(imgs):
    return imgs / 127.5 - 1


class DataLoader():
    def __init__(self, dataset_name, img_res=(128, 128)):
        self.dataset_name = dataset_name
        self.img_res = img_res

    def load_data(self, domain, batch_size=1, is_testing=False, index=None):
        data_type = f"train{domain}" if not is_testing else f"val{domain}"
        path = glob(os.path.join(DATASET_ROOT, self.dataset_name, f'{data_type}', '*.png'))

        if index is None:
            batch_images = np.random.choice(path, size=batch_size)
        else:
            batch_images = np.array(path)[index % len(path)]

        imgs = []
        for img_path in batch_images:
            img = self.imread(img_path)
            imgs.append(img)

        imgs = augment_data(np.array(imgs))
        imgs = normalize(imgs)

        return imgs

    def load_batch(self, batch_size=1, is_testing=False):
        data_type = "train" if not is_testing else "val"

        path_A = glob(os.path.join(DATASET_ROOT, self.dataset_name, f'{data_type}A', '*.png'))
        path_B = glob(os.path.join(DATASET_ROOT, self.dataset_name, f'{data_type}B', '*.png'))

        self.n_batches = int(min(len(path_A), len(path_B)) / batch_size)
        total_samples = self.n_batches * batch_size

        # Sample n_batches * batch_size from each path list so that model sees all
        # samples from both domains

        path_A = np.random.choice(path_A, total_samples)
        path_B = np.random.choice(path_B, total_samples)

        for i in range(self.n_batches - 1):
            batch_A = path_A[i * batch_size:(i + 1) * batch_size]
            batch_B = path_B[i * batch_size:(i + 1) * batch_size]
            imgs_A, imgs_B = [], []
            for img_A, img_B in zip(batch_A, batch_B):
                img_A = self.imread(img_A)
                img_B = self.imread(img_B)

                imgs_A.append(img_A)
                imgs_B.append(img_B)

            imgs_A = augment_data(np.array(imgs_A))
            imgs_B = augment_data(np.array(imgs_B))

            imgs_A = normalize(imgs_A)
            imgs_B = normalize(imgs_B)

            yield imgs_A, imgs_B

    def load_img(self, path):
        img = self.imread(path)
        img = normalize(img)
        return img[np.newaxis, :, :, :]

    def imread(self, path):
        return scipy.misc.imread(path, mode='RGB').astype(np.float)

