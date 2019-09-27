import os
import cv2
from PIL import Image
import numpy as np

from Rignak_Misc.path import get_local_file
from Rignak_DeepLearning.Autoencoders.unet import import_model
from Rignak_DeepLearning.OCR.train_text_segmentation import MODEL_FILENAME, NEURON_BASIS, SEGMENTATION_DATASET_ROOT
from Rignak_DeepLearning.normalization import intensity_normalization

INPUT_SHAPE = (512, 512, 3)
INPUT_FOLDER = get_local_file(__file__, os.path.join(SEGMENTATION_DATASET_ROOT, 'test', 'x'))
OUTPUT_FOLDER = get_local_file(__file__, os.path.join(SEGMENTATION_DATASET_ROOT, 'test', 'y'))


def segment_image(model, image, normalize, denormalize, input_shape=INPUT_SHAPE):
    padded_image = np.zeros(((image.shape[0] // input_shape[0] + 1) * input_shape[0],
                             (image.shape[1] // input_shape[1] + 1) * input_shape[0],
                             3))
    segmentation = np.zeros((padded_image.shape[0], padded_image.shape[1], 1))

    for i in range(image.shape[0] // input_shape[0]):
        i_slice = slice(i * input_shape[0], (i + 1) * input_shape[0])
        for j in range(image.shape[1] // input_shape[1]):
            j_slice = slice(j * input_shape[1], (j + 1) * input_shape[1])
            sample = np.expand_dims(padded_image[i_slice, j_slice], axis=0)
            segmentation[i_slice, j_slice] = model.predict(normalize(sample))[0]

    segmentation = denormalize(segmentation)

    return segmentation, padded_image


def get_components(image, segmentation):
    components = cv2.connectedComponentsWithStats
    num_labels, labels, stats, centroids = components(segmentation, 4, cv2.CV_8U)
    thumbnails = []
    for left, top, width, height, area in stats[1:]:
        thumbnails.append(image[top:top + height, left:left + width])
    return thumbnails


def main(input_shape=INPUT_SHAPE, input_folder=INPUT_FOLDER, output_folder=OUTPUT_FOLDER,
         neuron_basis=NEURON_BASIS):
    model = import_model(input_shape=input_shape, neuron_basis=neuron_basis)
    model.load_weights(MODEL_FILENAME)

    normalize, denormalize = intensity_normalization()

    for filename in os.listdir(input_folder):
        full_filename = os.path.join(input_folder, filename)
        with Image.open(full_filename) as image:
            segmentation, padded_image = segment_image(model, image, normalize, denormalize, input_shape=input_shape)
            for i, thumbnail in enumerate(get_components(segmentation, padded_image)):
                thumbnail = Image.fromarray(thumbnail)
                thumbnail.save(os.path.join(output_folder, f'{filename}_{i}.png'))

            segmentation = Image.fromarray(segmentation)
            segmentation.save(os.path.join(output_folder, f'{filename}.png'))


if __name__ == '__main__':
    main()
