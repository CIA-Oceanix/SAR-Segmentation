import numpy as np
import skimage
import PIL.Image
from keras.utils import Sequence

import dnnlib.tflib as tflib
from Rignak_DeepLearning.StyleGan.run import get_generative
from Rignak_DeepLearning.StyleGan.load import load_model
from Rignak_Misc.path import get_local_file

TRUNCATION_PSI = 1.0
BATCH_SIZE = 8
INPUT_SHAPE = (256, 256, 3)
OUTPUT_SHAPE = (512,)


class GanGenerator(Sequence):
    def __init__(self, truncation_psi=TRUNCATION_PSI, batch_size=BATCH_SIZE,
                 input_shape=INPUT_SHAPE, output_shape=OUTPUT_SHAPE):
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.truncation_psi = truncation_psi

        self.Gs = load_model()
        self.generative = get_generative(self.Gs, truncation_psi=truncation_psi)
        self.layers = {name: tensor for name, tensor, _ in self.Gs.list_layers()}

    def __getitem__(self, item):
        input_batch = np.zeros((self.batch_size, self.input_shape[0], self.input_shape[1], self.input_shape[2]))
        output_batch = np.zeros((self.batch_size, self.output_shape[0]))

        print('erhehrherh', item)
        for i in range(self.batch_size):
            latents = np.random.randn(1, self.Gs.input_shape[1])
            print('here', i)
            image = self.generative(latents)[0]

            output_batch[i] = tflib.run(self.layers['Truncation'], feed_dict={self.layers['latents_in']: latents})[
                              :, 1]

            if image.shape[-2] != self.input_shape[-2] or image.shape[-3] != self.input_shape[-3]:
                if image.shape[-1] == 1:
                    image = skimage.transform.resize(image[:, :, 0], (self.input_shape[0], self.input_shape[1]))
                else:
                    image = PIL.Image.fromarray(image, 'RGB')
                    image = np.array(image.resize((self.input_shape[0], self.input_shape[1]), PIL.Image.BICUBIC))
            input_batch[i] = image
            return input_batch / 255, output_batch

    def __len__(self):
        return 1000
