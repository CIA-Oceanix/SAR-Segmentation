import tensorflow as tf
import numpy as np
import dnnlib.tflib as tflib
from functools import partial

LAYER_NUMBER = 18


def create_stub(name, batch_size):
    return tf.constant(0, dtype='float32', shape=(batch_size, 0))


def create_variable_for_generator(name, batch_size, layer_number=LAYER_NUMBER):
    return tf.get_variable('learnable_dlatents',
                           shape=(batch_size, layer_number, 512),
                           dtype='float32',
                           initializer=tf.initializers.random_normal())


class Generator:
    def __init__(self, model, batch_size, randomize_noise=False, use_noise=False, layer_number=LAYER_NUMBER, truncation_psi=0.1):
        self.batch_size = batch_size
        self.layer_number = layer_number

        self.initial_dlatents = np.zeros((self.batch_size, layer_number, 512))
        model.components.synthesis.run(self.initial_dlatents,
                                       truncation_psi=truncation_psi,
                                       use_noise=use_noise,
                                       randomize_noise=randomize_noise,
                                       minibatch_size=self.batch_size,
                                       custom_inputs=[partial(create_variable_for_generator, batch_size=batch_size,
                                                              layer_number=layer_number),
                                                      partial(create_stub, batch_size=batch_size)],
                                       structure='fixed')

        self.sess = tf.get_default_session()
        self.graph = tf.get_default_graph()

        self.dlatent_variable = next(v for v in tf.global_variables() if 'learnable_dlatents' in v.name)
        self.dlatents_placeholder = None
        self.set_dlatents(self.initial_dlatents)

        self.generator_output = self.graph.get_tensor_by_name('G_synthesis_1/_Run/concat:0')
        self.generated_image = tflib.convert_images_to_uint8(self.generator_output, nchw_to_nhwc=True, uint8_cast=False)
        self.generated_image_uint8 = tf.saturate_cast(self.generated_image, tf.uint8)

    def reset_dlatents(self):
        self.set_dlatents(self.initial_dlatents)

    def set_dlatents(self, dlatents):
        assert (dlatents.shape == (self.batch_size, self.layer_number, 512))
        if self.dlatents_placeholder is None:
            self.dlatents_placeholder = tf.placeholder(np.float32, shape=dlatents.shape)
            self.dlatents_update_operation = self.dlatent_variable.assign(self.dlatents_placeholder)

        self.sess.run(self.dlatents_update_operation,
                      {self.dlatents_placeholder: dlatents})

    def get_dlatents(self):
        return self.sess.run(self.dlatent_variable)

    def generate_images(self, dlatents=None):
        if dlatents:
            self.set_dlatents(dlatents)
        return self.sess.run(self.generated_image_uint8)

    def generate_images(self, dlatents=None, truncation_psi=None):
        # self.graph.finalize()
        res = []
        if dlatents is not None:
            dlatents = np.array(dlatents)
            if len(dlatents.shape) == 2:
                dlatents = np.expand_dims(dlatents, 0)
            for batch_index in range(dlatents.shape[0]):
                self.set_dlatents(dlatents[[batch_index]])
                res.append(self.sess.run(self.generated_image_uint8)[0])
        else:
            res = self.sess.run(self.generated_image_uint8)

        return res
