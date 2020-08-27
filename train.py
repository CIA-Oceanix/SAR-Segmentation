import os, sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
for arg in sys.argv:
    if arg.startswith('--run_on_gpu='):
        os.environ['CUDA_VISIBLE_DEVICES'] = arg.split('=')[-1]

import fire

import deprecation_warnings

deprecation_warnings.filter_warnings()
import tensorflow as tf
import keras

gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
keras.backend.tensorflow_backend.set_session(sess)

from keras.callbacks import ModelCheckpoint

from Rignak_DeepLearning.data import get_dataset_roots
from Rignak_DeepLearning.normalization import NORMALIZATION_FUNCTIONS
from Rignak_DeepLearning.noise import get_composition
from Rignak_DeepLearning.callbacks import \
    HistoryCallback, \
    AutoencoderExampleCallback, \
    ConfusionCallback, \
    ClassificationExampleCallback, \
    SaveAttributes
from Rignak_DeepLearning.Autoencoders.flat import import_model as import_flat_model
from Rignak_DeepLearning.Autoencoders.unet import import_model as import_unet_model
from Rignak_DeepLearning.Categorizer.flat import import_model as import_categorizer
from Rignak_DeepLearning.Categorizer.mosaic_categorizer import import_model as import_mosaic_categorizer
from Rignak_DeepLearning.Categorizer.multiscale_mosaic_categorizer import \
    import_model as import_multiscale_mosaic_categorizer
from Rignak_DeepLearning.Categorizer.Inception import import_model_v3 as InceptionV3
from Rignak_DeepLearning.Categorizer.masked_categorizer import import_model as import_masked_categorizer
from Rignak_DeepLearning.Categorizer.unet_categorizer import import_model as import_unet_categorizer

from Rignak_DeepLearning.BiOutput.flat import import_model as import_bimode
from Rignak_DeepLearning.BiOutput.multiscale_autoencoder import import_model as import_multiscale_bimode
from Rignak_DeepLearning.BiOutput.generator import generator as bimode_generator, \
    normalize_generator as bimode_normalize, augment_generator as bimode_augment
from Rignak_DeepLearning.BiOutput.callbacks import \
    ExampleCallback as BimodeExampleCallback, \
    HistoryCallback as BimodeHistoryCallback

from Rignak_DeepLearning.Categorization2Segmentation.heatmap_model import import_model as import_heatmap_model
from Rignak_DeepLearning.Categorization2Segmentation.generator import get_heatmap_generator_with_dummy_data
from Rignak_DeepLearning.generator import \
    autoencoder_generator, \
    categorizer_generator, \
    saliency_generator, \
    thumbnail_generator as thumb_generator, \
    normalize_generator, \
    augment_generator, \
    regressor_generator, \
    rotsym_augmentor, \
    occlusion_generator
from Rignak_DeepLearning.StyleGan.callbacks import GanRegressorExampleCallback
from Rignak_DeepLearning.config import get_config

from Rignak_Misc.path import list_dir

BATCH_SIZE = 16
TRAINING_STEPS = 2000
VALIDATION_STEPS = 500
EPOCHS = 1000
INITIAL_EPOCH = 0

DEFAULT_INPUT_SHAPE = (256, 256, 3)
DEFAULT_SCALING = 1


def get_generators(config, task, dataset, batch_size, default_input_shape=DEFAULT_INPUT_SHAPE,
                   default_scaling=DEFAULT_SCALING):
    def get_saliency_generators():
        train_generator = saliency_generator(train_folder, input_shape=input_shape, batch_size=batch_size,
                                             downsampling=len(config[task]['CONV_LAYERS']))
        val_generator = saliency_generator(val_folder, input_shape=input_shape, batch_size=batch_size,
                                           downsampling=len(config[task]['CONV_LAYERS']))
        callback_generator = saliency_generator(val_folder, input_shape=input_shape, batch_size=batch_size,
                                                downsampling=len(config[task]['CONV_LAYERS']))
        next(train_generator), next(val_generator), next(callback_generator)
        return train_generator, val_generator, callback_generator, train_folder

    def get_autoencoder_generators():
        train_generator = autoencoder_generator(train_folder, input_shape=input_shape, batch_size=batch_size)
        val_generator = autoencoder_generator(val_folder, input_shape=input_shape, batch_size=batch_size)
        callback_generator = autoencoder_generator(val_folder, input_shape=input_shape, batch_size=batch_size)
        next(train_generator), next(val_generator), next(callback_generator)
        return train_generator, val_generator, callback_generator, train_folder

    def get_categorizer_generators():
        train_generator = categorizer_generator(train_folder, input_shape=input_shape, batch_size=batch_size)
        val_generator = categorizer_generator(val_folder, input_shape=input_shape, batch_size=batch_size)
        callback_generator = categorizer_generator(val_folder, input_shape=input_shape, batch_size=batch_size)
        next(train_generator), next(val_generator), next(callback_generator)
        return train_generator, val_generator, callback_generator, train_folder

    def get_style_transfer_generators():
        train_generator = thumb_generator(train_folder, input_shape=input_shape, batch_size=batch_size, scaling=scaling)
        val_generator = thumb_generator(val_folder, input_shape=input_shape, batch_size=batch_size, scaling=scaling)
        callback_generator = thumb_generator(val_folder, input_shape=input_shape, batch_size=batch_size,
                                             scaling=scaling)
        next(train_generator), next(val_generator), next(callback_generator)
        return train_generator, val_generator, callback_generator, train_folder

    def get_bimode_generators():
        train_generator = bimode_generator(train_folder, input_shape=input_shape, batch_size=batch_size)
        val_generator = bimode_generator(val_folder, input_shape=input_shape, batch_size=batch_size)
        callback_generator = bimode_generator(val_folder, input_shape=input_shape, batch_size=batch_size)
        next(train_generator), next(val_generator), next(callback_generator)
        return train_generator, val_generator, callback_generator, train_folder

    def get_regressor_generators():
        train_generator = regressor_generator(train_folder, input_shape=input_shape, batch_size=batch_size)
        val_generator = regressor_generator(val_folder, input_shape=input_shape, batch_size=batch_size)
        callback_generator = regressor_generator(val_folder, input_shape=input_shape, batch_size=batch_size)
        next(train_generator), next(val_generator), next(callback_generator)
        return train_generator, val_generator, callback_generator, train_folder

    def get_heatmap_generators():
        output_shape = config[task].get('OUTPUT_SHAPE', input_shape)
        input_label = config[task].get('INPUT_LABEL', 'input')
        output_label = config[task].get('OUTPUT_LABEL', 'output')

        dummy_probability_increase = config[task].get('DUMMY_PROBABILITY_INCREASE')
        dummy_initial_batch = config[task].get('DUMMY_INITIAL_BATCH')
        dummy_maximum_probability = config[task].get('DUMMY_MAXIMUM_PROBABILITY')

        heatmap_generator = get_heatmap_generator_with_dummy_data(dummy_probability_increase=dummy_probability_increase,
                                                                  dummy_initial_batch=dummy_initial_batch,
                                                                  dummy_maximum_probability=dummy_maximum_probability)

        train_generator = heatmap_generator(train_folder, input_shape=input_shape, output_shape=output_shape,
                                            batch_size=batch_size, input_label=input_label, output_label=output_label)
        val_generator = heatmap_generator(val_folder, input_shape=input_shape, output_shape=output_shape,
                                          batch_size=batch_size, input_label=input_label, output_label=output_label)
        callback_generator = heatmap_generator(val_folder, input_shape=input_shape, output_shape=output_shape,
                                               batch_size=batch_size, input_label=input_label,
                                               output_label=output_label)

        next(train_generator), next(val_generator), next(callback_generator)
        return train_generator, val_generator, callback_generator, train_folder

    input_shape = config[task].get('INPUT_SHAPE', default_input_shape)
    scaling = config[task].get('SCALING', default_scaling)
    train_folder, val_folder = get_dataset_roots(task, dataset=dataset)

    functions = {"saliency": get_saliency_generators,
                 "autoencoder": get_autoencoder_generators,
                 "heatmap": get_heatmap_generators,
                 "flat_autoencoder": get_autoencoder_generators,
                 "categorizer": get_categorizer_generators,
                 "inceptionV3": get_categorizer_generators,
                 "mosaic_categorizer": get_categorizer_generators,
                 "multiscale_mosaic_categorizer": get_categorizer_generators,
                 "masked_categorizer": get_categorizer_generators,
                 "unet_categorizer": get_categorizer_generators,
                 "style_transfer": get_style_transfer_generators,
                 "bimode": get_bimode_generators,
                 "multiscale_bimode": get_bimode_generators,
                 "regressor": get_regressor_generators,
                 }
    return functions[task]()


def get_data_augmentation(config, task, train_generator, val_generator, callback_generator):
    def get_im2im_data_augmentation():
        new_train_generator = normalize_generator(
            augment_generator(train_generator, noise_function=noise_function, apply_on_output=True),
            normalizer, apply_on_output=True)
        new_val_generator = normalize_generator(
            augment_generator(val_generator, noise_function=noise_function, apply_on_output=True),
            normalizer, apply_on_output=True)
        new_callback_generator = normalize_generator(
            augment_generator(callback_generator, noise_function=noise_function, apply_on_output=True),
            normalizer, apply_on_output=True)

        # new_train_generator = rotsym_augmentor(new_train_generator)
        # new_val_generator = rotsym_augmentor(new_val_generator)
        # new_callback_generator = rotsym_augmentor(new_callback_generator)

        return new_train_generator, new_val_generator, new_callback_generator

    def get_bimode_augmentation():
        new_train_generator = bimode_normalize(train_generator, normalizer, apply_on_output=True)
        new_val_generator = bimode_normalize(val_generator, normalizer, apply_on_output=True)
        new_callback_generator = bimode_normalize(callback_generator, normalizer, apply_on_output=True)

        return new_train_generator, new_val_generator, new_callback_generator

    def get_categorizer_augmentation():
        new_train_generator = normalize_generator(
            augment_generator(train_generator, noise_function=noise_function, apply_on_output=False),
            normalizer, apply_on_output=False)

        new_val_generator = normalize_generator(
            augment_generator(val_generator, noise_function=noise_function, apply_on_output=False),
            normalizer, apply_on_output=False)

        new_callback_generator = normalize_generator(
            augment_generator(callback_generator, noise_function=noise_function, apply_on_output=False),
            normalizer, apply_on_output=False)

        return new_train_generator, new_val_generator, new_callback_generator

    def get_regressor_augmentation():
        return train_generator, val_generator, callback_generator

    normalizer = NORMALIZATION_FUNCTIONS[config[task].get('NORMALIZATION', 'none')]()[0]
    noise_function = get_composition(config[task].get('NOISE', [None]),
                                     config[task].get('NOISE_PARAMETERS', (())))

    functions = {"style_transfer": get_im2im_data_augmentation,
                 "saliency": get_im2im_data_augmentation,
                 "heatmap": get_im2im_data_augmentation,
                 "autoencoder": get_im2im_data_augmentation,
                 "flat_autoencoder": get_im2im_data_augmentation,
                 "categorizer": get_categorizer_augmentation,
                 "inceptionV3": get_categorizer_augmentation,
                 "mosaic_categorizer": get_categorizer_augmentation,
                 "multiscale_mosaic_categorizer": get_categorizer_augmentation,
                 "masked_categorizer": get_categorizer_augmentation,
                 "unet_categorizer": get_categorizer_augmentation,
                 "bimode": get_bimode_augmentation,
                 "multiscale_bimode": get_bimode_augmentation,
                 "regressor": get_regressor_augmentation,
                 }
    assert task in functions, f'You asked for {task}, but functions.keys is {list(functions.keys())}'
    return functions[task]()


def get_models(config, task, name, train_folder, default_input_shape=DEFAULT_INPUT_SHAPE, load=False):
    def get_saliency_model():
        if len(labels) == 2:
            config[task]['OUTPUT_CANALS'] = 1
        else:
            config[task]['OUTPUT_CANALS'] = len(labels)
        model = import_flat_model(name=name, config=config[task], load=load)
        model.callback_titles = ['Input', 'Prediction', 'Truth'] + labels
        model.class_weight = None
        return model

    def get_autoencoder_model():
        if task == 'flat_autoencoder':
            model = import_flat_model(name=name, config=config[task], load=load)
        elif task == "unet_categorizer":
            model = import_unet_categorizer(name=name, config=config[task], load=load)
            model.labels = labels
        else:
            model = import_unet_model(name=name, config=config[task], load=load)
        model.callback_titles = ['Input', 'Prediction', 'Truth'] + labels
        model.class_weight = None
        return model

    def get_categorizer_model():
        if task == 'inceptionV3':
            model = InceptionV3(input_shape, len(labels), name, load=load, imagenet=config[task].get('IMAGENET', False),
                                class_weight=class_weight,
                                last_activation=config[task].get('LAST_ACTIVATION', 'softmax'))
        else:
            model = import_categorizer(len(labels), config=config[task], name=name, load=load,
                                       class_weight=class_weight)
        model.labels = labels
        return model

    def get_mosaic_categorizer_model():
        model = import_mosaic_categorizer(input_shape, len(labels), name, load=load,
                                          class_weight=class_weight,
                                          last_activation=config[task].get('LAST_ACTIVATION', 'softmax'),
                                          modality=config[task].get('MODALITY', 'mean')
                                          )
        model.labels = labels
        return model

    def get_multiscale_mosaic_categorizer_model():
        model = import_multiscale_mosaic_categorizer(input_shape, len(labels), name, load=load,
                                                     class_weight=class_weight,
                                                     last_activation=config[task].get('LAST_ACTIVATION', 'softmax'),
                                                     modality=config[task].get('MODALITY', 'mean')
                                                     )
        model.labels = labels
        return model

    def get_masked_categorizer_model():
        model = import_masked_categorizer(input_shape, len(labels), name, config[task], load=load,
                                          class_weight=class_weight)
        model.labels = labels
        return model

    def get_bimode_model():
        model = import_bimode(output_canals, labels, config=config[task], name=name, load=load)
        model.labels = labels
        model.class_weight = None
        return model

    def get_multiscale_bimode_model():
        model = import_multiscale_bimode(labels, config=config[task], name=name, load=load)
        model.labels = labels
        model.class_weight = None
        return model

    def get_regressor_model():
        model = InceptionV3(input_shape, output_canals, name, load=load, imagenet=config[task].get('IMAGENET', False),
                            last_activation='linear', loss='mse', metrics=[])
        model.class_weight = None
        return model

    def get_heatmap_model():
        model = import_heatmap_model(name=name, config=config[task], load=load)
        model.callback_titles = ['Input', 'Prediction', 'Truth']
        model.class_weight = None
        return model

    path_labels = list_dir(train_folder)
    labels = [os.path.split(label)[-1] for label in path_labels]
    class_weight = [len(os.listdir(folder)) for folder in path_labels]
    input_shape = config[task].get('INPUT_SHAPE', default_input_shape)
    output_canals = config[task].get('OUTPUT_CANALS')
    functions = {"saliency": get_saliency_model,
                 "autoencoder": get_autoencoder_model,
                 "heatmap": get_heatmap_model,
                 "flat_autoencoder": get_autoencoder_model,
                 "style_transfer": get_autoencoder_model,
                 "unet_categorizer": get_autoencoder_model,
                 "categorizer": get_categorizer_model,
                 "mosaic_categorizer": get_mosaic_categorizer_model,
                 "multiscale_mosaic_categorizer": get_multiscale_mosaic_categorizer_model,
                 "masked_categorizer": get_masked_categorizer_model,
                 "inceptionV3": get_categorizer_model,
                 "bimode": get_bimode_model,
                 "multiscale_bimode": get_multiscale_bimode_model,
                 "regressor": get_regressor_model,
                 }
    assert task in functions, f'You asked for {task}, but functions.keys is {list(functions.keys())}'
    return functions[task]()


def get_callbacks(config, task, model, callback_generator):
    def get_im2im_callbacks():
        callbacks = [SaveAttributes(callback_generator, config[task]),
                     ModelCheckpoint(model.weight_filename, save_best_only=True),
                     HistoryCallback(batch_size, training_steps),
                     AutoencoderExampleCallback(callback_generator, denormalizer=denormalizer)]
        return callbacks

    def get_bimode_callbacks():
        callbacks = [SaveAttributes(callback_generator, config[task], labels=model.labels),
                     ModelCheckpoint(model.weight_filename, save_best_only=True),
                     BimodeHistoryCallback(batch_size, training_steps),
                     BimodeExampleCallback(callback_generator, denormalizer=denormalizer),
                     ConfusionCallback(callback_generator, model.labels)]
        return callbacks

    def get_categorizer_callbacks():
        callbacks = [SaveAttributes(callback_generator, config[task], labels=model.labels),
                     ModelCheckpoint(model.weight_filename, save_best_only=True),
                     # ModelCheckpoint(model.weight_filename + ".{epoch:02d}.h5", save_best_only=True),
                     HistoryCallback(batch_size, training_steps),
                     ConfusionCallback(callback_generator, model.labels),
                     ClassificationExampleCallback(callback_generator, denormalizer=denormalizer)]
        return callbacks

    def get_regressor_callback():
        callbacks = [SaveAttributes(callback_generator, config[task]),
                     ModelCheckpoint(model.weight_filename, save_best_only=False),
                     HistoryCallback(batch_size, training_steps),
                     GanRegressorExampleCallback(callback_generator,
                                                 gan_filename=config[task]["GAN_FILENAME"],
                                                 layer_number=config[task].get('LAYER_NUMBER', 1),
                                                 truncation_psi=config[task].get("TRUNCATION_PSI", 1),
                                                 denormalizer=denormalizer
                                                 )]
        return callbacks

    denormalizer = NORMALIZATION_FUNCTIONS[config[task].get('NORMALIZATION', 'intensity')]()[1]
    batch_size = config[task].get('BATCH_SIZE')
    training_steps = config[task].get('TRAINING_STEPS')

    functions = {"saliency": get_im2im_callbacks,
                 "autoencoder": get_im2im_callbacks,
                 "heatmap": get_im2im_callbacks,
                 "flat_autoencoder": get_im2im_callbacks,
                 "style_transfer": get_im2im_callbacks,
                 "categorizer": get_categorizer_callbacks,
                 "mosaic_categorizer": get_categorizer_callbacks,
                 "multiscale_mosaic_categorizer": get_categorizer_callbacks,
                 "masked_categorizer": get_categorizer_callbacks,
                 "inceptionV3": get_categorizer_callbacks,
                 "unet_categorizer": get_categorizer_callbacks,
                 "bimode": get_bimode_callbacks,
                 "multiscale_bimode": get_bimode_callbacks,
                 "regressor": get_regressor_callback,
                 }
    assert task in functions, f'You asked for {task}, but functions.keys is {list(functions.keys())}'
    return functions[task]()


def main(task, dataset, batch_size=BATCH_SIZE, epochs=EPOCHS,
         training_steps=TRAINING_STEPS, validation_steps=VALIDATION_STEPS, initial_epoch=INITIAL_EPOCH,
         **kwargs):
    config = get_config()
    if task not in config:
        config[task] = {'TASK': task}
    for key, value in kwargs.items():
        config[task][key] = value
    config[task]['DATASET'] = dataset
    config[task]['TRAINING_STEPS'] = training_steps
    config[task]['VALIDATION_STEPS'] = validation_steps
    config[task]['EPOCHS'] = epochs
    config[task]['BATCH_SIZE'] = batch_size
    task = config[task]['TASK']

    train_folder, val_folder = get_dataset_roots(task, dataset=dataset)
    train_generator, val_generator, callback_generator, train_dir = get_generators(config, task, dataset, batch_size)
    train_generator, val_generator, callback_generator = get_data_augmentation(config, task, train_generator,
                                                                               val_generator, callback_generator)
    name = config[task].get('NAME', f'{dataset}_{task}')
    print('train.py, l326: NAME is', name)
    model = get_models(config, task, name, train_folder, load=initial_epoch != 0)
    callbacks = get_callbacks(config, task, model, callback_generator)
    print('FIRIN MAH LASER')
    train(model, train_generator, val_generator, callbacks,
          epochs=epochs, training_steps=training_steps, validation_steps=validation_steps, initial_epoch=initial_epoch)


def train(model, train_generator, val_generator, callbacks, training_steps=TRAINING_STEPS,
          validation_steps=VALIDATION_STEPS, epochs=EPOCHS, initial_epoch=INITIAL_EPOCH):
    model.fit_generator(generator=train_generator,
                        validation_data=val_generator,
                        verbose=1,
                        steps_per_epoch=training_steps,
                        validation_steps=validation_steps,
                        epochs=epochs,
                        callbacks=callbacks,
                        initial_epoch=initial_epoch,
                        # class_weight=model.class_weight
                        )


if __name__ == '__main__':
    fire.Fire(main)
