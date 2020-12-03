import os
import numpy as np
import tqdm

from keras.models import Model
from keras.layers import Input, Lambda, concatenate, GlobalAveragePooling2D, Dense, Conv2D
from keras.applications.inception_v3 import InceptionV3
from keras.losses import mean_squared_error
from keras_radam.training import RAdamOptimizer
import keras.backend as K

from Rignak_DeepLearning.Image_to_Image.unet import import_model as import_unet_model
from Rignak_Misc.path import get_local_file
from Rignak_DeepLearning.config import get_config
from Rignak_DeepLearning.loss import weighted_binary_crossentropy
from Rignak_DeepLearning.models import write_summary

WEIGHT_ROOT = get_local_file(__file__, os.path.join('..', '_outputs', 'models'))
SUMMARY_ROOT = get_local_file(__file__, os.path.join('..', '_outputs', 'summary'))
LOAD = False
LEARNING_RATE = 10 ** -4

CONFIG_KEY = 'segmenter'
CONFIG = get_config()[CONFIG_KEY]
DEFAULT_NAME = CONFIG.get('NAME', 'DEFAULT_MODEL_NAME')


def import_inception(config, input_layer, learning_rate=LEARNING_RATE):
    input_shape = config.get('INPUT_SHAPE')
    if input_shape[0] < 75 or input_shape[1] < 75:  # minimal input for InceptionV3
        input_layer_resize = Lambda(lambda x: K.tf.image.resize_bilinear(x, (75, 75)))(input_layer)
    else:
        input_layer_resize = input_layer

    if input_shape[-1] == 1:
        input_layer_tricanals = concatenate([input_layer_resize, input_layer_resize, input_layer_resize])
    elif input_shape[-1] == 3:
        input_layer_tricanals = input_layer_resize
    else:
        input_layer_tricanals = Conv2D(3, (1, 1))(input_layer_resize)

    inception_base_model = InceptionV3(input_tensor=input_layer_tricanals, classes=1, include_top=False,
                                       activation=config['ACTIVATION'], weights=None)
    x = inception_base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1, activation='sigmoid')(x)
    inception = Model(input_layer, outputs=[x], name="inception")
    inception.compile(loss='mse', optimizer=RAdamOptimizer(learning_rate))
    return inception


class Yes_UGAN():
    def __init__(self, weight_root=WEIGHT_ROOT, summary_root=SUMMARY_ROOT, load=LOAD, learning_rate=LEARNING_RATE,
                 config=CONFIG, name=DEFAULT_NAME, skip=True, metrics=None, labels=None):
        self.discriminator_loss = mean_squared_error
        self.generator_loss = config.get('LOSS', 'mse')
        if self.generator_loss == 'mse':
            self.generator_loss = mean_squared_error
        elif self.generator_loss == 'WBCE':
            self.generator_loss = weighted_binary_crossentropy

        self.generator = import_unet_model(weight_root=weight_root, summary_root=summary_root, load=load,
                                           learning_rate=learning_rate, config=config, name="unet", skip=skip,
                                           metrics=metrics, labels=labels)

        input_layer = Input(self.generator.input_shape[1:])
        generator_prediction = self.generator(input_layer)

        self.discriminator = import_inception(config, input_layer, learning_rate=learning_rate)
        self.discriminator.trainable = False

        discriminator_prediction = self.discriminator(generator_prediction)

        self.adversarial_autoencoder = Model(input_layer, [generator_prediction, discriminator_prediction],
                                             name=name)
        self.loss_weights = [1., 1.]
        self.adversarial_autoencoder.compile(loss=[self.generator_loss, self.discriminator_loss],
                                             loss_weights=self.loss_weights, optimizer=RAdamOptimizer(learning_rate))
        self.name = name
        self.weight_filename = os.path.join(weight_root, f"{self.name}.h5")
        self.summary_filename = os.path.join(summary_root, f"{self.name}.txt")
        self.generator.weight_filename = self.weight_filename
        self.generator.summary_filename = self.summary_filename
        self.generator.name = self.name
        self.adversarial_autoencoder.summary_filename = self.summary_filename
        write_summary(self.adversarial_autoencoder)

    def fit_generator(self, generator, validation_data, verbose, steps_per_epoch, validation_steps,
                      epochs, callbacks, initial_epoch):
        def process_batch(generator):
            generator_input, generator_truth = next(generator)
            generator_input = generator_input.astype('float32')
            generator_truth = generator_truth.astype('float32')
            generator_prediction = self.generator.predict(generator_input)

            # Train the discriminator
            discriminator_input = np.concatenate([generator_prediction, generator_truth], axis=0)
            discriminator_truth = np.concatenate([np.ones(generator_prediction.shape[0]),
                                                  np.zeros(generator_prediction.shape[0])], axis=0)
            discriminator_truth += np.random.normal(size=discriminator_truth.shape, scale=0.05)
            self.discriminator.train_on_batch([discriminator_input], [discriminator_truth])

            # Train the generator
            generator_loss = self.adversarial_autoencoder.train_on_batch([generator_input],
                                                                         [generator_truth,
                                                                          np.zeros(generator_prediction.shape[0])])
            return generator_loss

        for callback in callbacks:
            callback.model = self.generator
            callback.on_train_begin(logs=None)

        for epoch in range(epochs):
            pbar = tqdm.trange(steps_per_epoch, bar_format='')

            generator_losses = []
            discriminator_losses = []
            total_losses = []
            for batch_i in range(steps_per_epoch):
                total_loss, generator_loss, discriminator_loss = process_batch(generator)
                total_losses.append(total_loss)
                generator_losses.append(generator_loss)
                discriminator_losses.append(discriminator_loss)
                pbar.set_description(
                    f"Epoch {epoch}/{epochs} - Batch {batch_i + 1}/{steps_per_epoch} - "
                    f"[D_loss = {np.mean(discriminator_losses):.4f} "
                    f" E_loss = {np.mean(generator_losses):.4f}"
                )
                pbar.update(1)

            validation_generator_losses = []
            validation_discriminator_losses = []
            validation_discriminator_accuracies = []
            for batch_i in range(validation_steps):
                generator_input, generator_truth = next(validation_data)
                generator_prediction = self.generator.predict(generator_input)

                discriminator_input = np.concatenate([generator_prediction, generator_truth], axis=0)
                discriminator_truth = np.concatenate([np.ones(generator_prediction.shape[0]),
                                                      np.zeros(generator_prediction.shape[0])], axis=0)
                discriminator_prediction = self.discriminator.predict(discriminator_input)

                validation_generator_loss = np.mean((generator_truth - generator_prediction) ** 2)
                validation_generator_losses.append(validation_generator_loss)

                validation_discriminator_loss = np.mean((discriminator_truth - discriminator_prediction) ** 2)
                validation_discriminator_losses.append(validation_discriminator_loss)

                validation_discriminator_accuracy = np.mean(abs(discriminator_prediction - discriminator_truth) > 0.5)
                validation_discriminator_accuracies.append(validation_discriminator_accuracy)

            logs = {
                "loss": float(np.mean(generator_losses) + np.mean(discriminator_losses)),
                "val_loss": float(np.mean(validation_generator_losses) * self.loss_weights[0] +
                                  np.mean(validation_discriminator_losses) * self.loss_weights[1]),
                # "acc": float(np.mean(discriminator_accuracies)),
                # "val_acc": float(np.mean(validation_discriminator_accuracies)),
                "generator_loss": float(np.mean(generator_losses)),
                "val_generator_loss": float(np.mean(validation_generator_losses) * self.loss_weights[0]),
                "discriminator_loss": float(np.mean(discriminator_losses)),
                "val_discriminator_loss": float(np.mean(validation_discriminator_losses) * self.loss_weights[1])
            }

            for callback in callbacks:
                callback.on_epoch_end(epoch=epoch, logs=logs)
