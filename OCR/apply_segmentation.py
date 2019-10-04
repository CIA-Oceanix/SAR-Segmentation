import os
import cv2
from PIL import Image
import numpy as np
import fire

from keras.models import load_model

from Rignak_Misc.path import get_local_file
from Rignak_DeepLearning.normalization import intensity_normalization
from Rignak_DeepLearning.config import get_config

config = get_config('style_transfer')

INPUT_SHAPE = (512, 512, 3)

INPUT_FOLDER = get_local_file(__file__, os.path.join('res', 'input'))
OUTPUT_FOLDER = get_local_file(__file__, os.path.join('res', 'output'))


def segment_image(model, image, normalize, denormalize, input_shape=INPUT_SHAPE):
    padded_image = np.zeros(((image.shape[0] // input_shape[0] + 1) * input_shape[0],
                             (image.shape[1] // input_shape[1] + 1) * input_shape[0],
                             3))
    segmentation = np.zeros((padded_image.shape[0], padded_image.shape[1]))
    padded_image[:image.shape[0], :image.shape[1]] = image

    for i in range(image.shape[0] // input_shape[0]):
        i_slice = slice(i * input_shape[0], (i + 1) * input_shape[0])
        for j in range(image.shape[1] // input_shape[1]):
            j_slice = slice(j * input_shape[1], (j + 1) * input_shape[1])
            sample = np.expand_dims(padded_image[i_slice, j_slice], axis=0)
            segmentation[i_slice, j_slice] = np.mean(model.predict(normalize(sample))[0], axis=-1)

    segmentation = denormalize(segmentation)
    segmentation[segmentation<128] = 0
    segmentation[segmentation>128] = 255
    segmentation = 255 - segmentation

    return segmentation.astype('uint8'), padded_image.astype('uint8')


def get_components(image, segmentation):
    components = cv2.connectedComponentsWithStats

    num_labels, labels, stats, centroids = components(segmentation, 4, cv2.CV_8U)
    thumbnails = []
    for left, top, width, height, area in stats[1:]:
        thumbnails.append(image[top:top + height, left:left + width])
    return thumbnails


def main(model_filename, input_folder=INPUT_FOLDER, output_folder=OUTPUT_FOLDER):
    """
    Use a trained segmentation model to create the thumbnail of each segmented element.

    :param model_filename: name of the file containing a model trained for the segmentation
    :param input_folder: folder containing the images to segment
    :param output_folder: folder which will contain the thumbnails
    :return:
    """
    model = load_model(model_filename)

    normalize, denormalize = intensity_normalization()

    for filename in os.listdir(input_folder):
        full_filename = os.path.join(input_folder, filename)
        with Image.open(full_filename) as image:
            image = np.array(image)
            segmentation, padded_image = segment_image(model, image, normalize, denormalize,
                                                       input_shape=model.layers[0].input_shape[-3:])
            for i, thumbnail in enumerate(get_components(padded_image, segmentation)):
                thumbnail = Image.fromarray(thumbnail.astype('uint8'))
                thumbnail.save(os.path.join(output_folder, f'{os.path.splitext(filename)[0]}_{i}.png'))

            segmentation = Image.fromarray(segmentation)
            segmentation.save(os.path.join(output_folder, f'{os.path.splitext(filename)[0]}.png'))


if __name__ == '__main__':
    fire.Fire(main)
