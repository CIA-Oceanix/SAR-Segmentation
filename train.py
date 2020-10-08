import os, sys, fire

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
for arg in sys.argv:
    if arg.startswith('--run_on_gpu='):
        os.environ['CUDA_VISIBLE_DEVICES'] = arg.split('=')[-1]
task = sys.argv[1]

import Rignak_DeepLearning.deprecation_warnings

Rignak_DeepLearning.deprecation_warnings.filter_warnings()

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
    RegressorCallback, \
    SaveAttributes

from Rignak_DeepLearning.generator import normalize_generator, augment_generator, rotsym_augmentor
from Rignak_DeepLearning.config import get_config

if task in ("saliency", "autoencoder", "segmenter", "flat_autoencoder"):
    from Rignak_DeepLearning.Image_to_Image.unet import import_model as import_unet_model
    from Rignak_DeepLearning.Image_to_Image.generator import \
        segmenter_base_generator as segmenter_generator, \
        saliency_base_generator as saliency_generator, \
        autoencoder_base_generator as autoencoder_generator, \
        thumbnail_base_generator as thumbnail_generator

elif task in ("categorizer", "inceptionV3", "mosaic_categorizer", "multiscale_mosaic_categorizer",
              "unet_categorizer", "regressor"):
    from Rignak_DeepLearning.Image_to_Class.flat import import_model as import_categorizer
    from Rignak_DeepLearning.Image_to_Class.Inception import import_model_v3 as InceptionV3
    from Rignak_DeepLearning.Image_to_Class.unet_categorizer import import_model as import_unet_categorizer
    from Rignak_DeepLearning.Image_to_Class.generator import \
        categorizer_base_generator as categorizer_generator, \
        regressor_base_generator as regressor_generator

from Rignak_Misc.path import list_dir

BATCH_SIZE = 16
TRAINING_STEPS = 2000
VALIDATION_STEPS = 500
EPOCHS = 1000
INITIAL_EPOCH = 0

DEFAULT_INPUT_SHAPE = (256, 256, 3)
ROOT = 'E:\\\\datasets'


def get_generators(config, task, dataset, batch_size, train_folder, val_folder,
                   default_input_shape=DEFAULT_INPUT_SHAPE):
    def get_saliency_generators():
        kwargs = {"input_shape": input_shape, "batch_size": batch_size, "output_shape": output_shape,
                  "folders": config[task].get('LABELS')}
        train_generator = saliency_generator(train_folder, **kwargs)
        val_generator = saliency_generator(val_folder, **kwargs)
        callback_generator = saliency_generator(val_folder, **kwargs)
        next(train_generator), next(val_generator), next(callback_generator)
        return train_generator, val_generator, callback_generator, train_folder

    def get_autoencoder_generators():
        train_generator = autoencoder_generator(train_folder, input_shape=input_shape, batch_size=batch_size)
        val_generator = autoencoder_generator(val_folder, input_shape=input_shape, batch_size=batch_size)
        callback_generator = autoencoder_generator(val_folder, input_shape=input_shape, batch_size=batch_size)
        next(train_generator), next(val_generator), next(callback_generator)
        return train_generator, val_generator, callback_generator, train_folder

    def get_categorizer_generators():
        kwargs = {"input_shape": input_shape, "batch_size": batch_size, "folders": config[task].get('LABELS')}
        train_generator = categorizer_generator(train_folder, **kwargs)
        val_generator = categorizer_generator(val_folder, **kwargs)
        callback_generator = categorizer_generator(val_folder, **kwargs)
        next(train_generator), next(val_generator), next(callback_generator)
        return train_generator, val_generator, callback_generator, train_folder

    def get_regressor_generators():
        attributes = config[task].get('ATTRIBUTES')
        assert attributes, 'No --ATTRIBUTES were passed'
        kwargs = {"input_shape": input_shape, "batch_size": batch_size}
        train_generator = regressor_generator(train_folder, attributes, **kwargs)
        val_generator = regressor_generator(val_folder, attributes, **kwargs)
        callback_generator = regressor_generator(val_folder, attributes, **kwargs)
        next(train_generator), next(val_generator), next(callback_generator)
        return train_generator, val_generator, callback_generator, train_folder

    def get_segmenter_generators():
        input_label = config[task].get('INPUT_LABEL', 'input')
        output_label = config[task].get('OUTPUT_LABEL', 'output')

        kwargs = {"input_shape": input_shape, "output_shape": output_shape, "batch_size": batch_size,
                  "input_label": input_label, "output_label": output_label}

        train_generator = segmenter_generator(train_folder, **kwargs)
        val_generator = segmenter_generator(val_folder, **kwargs)
        callback_generator = segmenter_generator(val_folder, **kwargs)

        next(train_generator), next(val_generator), next(callback_generator)
        return train_generator, val_generator, callback_generator, train_folder

    input_shape = config[task].get('INPUT_SHAPE', default_input_shape)
    output_shape = config[task].get('OUTPUT_SHAPE', input_shape)

    functions = {"saliency": get_saliency_generators,
                 "autoencoder": get_autoencoder_generators,
                 "segmenter": get_segmenter_generators,
                 "flat_autoencoder": get_autoencoder_generators,
                 "categorizer": get_categorizer_generators,
                 "inceptionV3": get_categorizer_generators,
                 "mosaic_categorizer": get_categorizer_generators,
                 "multiscale_mosaic_categorizer": get_categorizer_generators,
                 "unet_categorizer": get_categorizer_generators,
                 "regressor": get_regressor_generators,
                 }
    return functions[task]()


def get_data_augmentation(config, task, train_generator, val_generator, callback_generator):
    def get_im2im_data_augmentation():
        kwargs = {"noise_function": noise_function, "apply_on_output": True, "zoom_factor": zoom_factor,
                  "rotation": rotation}

        new_train_generator = augment_generator(train_generator, **kwargs)
        new_train_generator = normalize_generator(new_train_generator, normalizer, apply_on_output=True)

        new_val_generator = augment_generator(val_generator, **kwargs)
        new_val_generator = normalize_generator(new_val_generator, normalizer, apply_on_output=True)

        new_callback_generator = augment_generator(callback_generator, **kwargs)
        new_callback_generator = normalize_generator(new_callback_generator, normalizer, apply_on_output=True)

        new_train_generator = rotsym_augmentor(new_train_generator)
        new_val_generator = rotsym_augmentor(new_val_generator)
        new_callback_generator = rotsym_augmentor(new_callback_generator)

        return new_train_generator, new_val_generator, new_callback_generator

    def get_categorizer_augmentation():
        kwargs = {"noise_function": noise_function, "apply_on_output": False, "zoom_factor": zoom_factor,
                  "rotation": rotation}

        new_train_generator = augment_generator(train_generator, **kwargs)
        new_train_generator = normalize_generator(new_train_generator, normalizer, apply_on_output=False)

        new_val_generator = augment_generator(val_generator, **kwargs)
        new_val_generator = normalize_generator(new_val_generator, normalizer, apply_on_output=False)

        new_callback_generator = augment_generator(callback_generator, **kwargs)
        new_callback_generator = normalize_generator(new_callback_generator, normalizer, apply_on_output=False)

        return new_train_generator, new_val_generator, new_callback_generator

    def get_regressor_augmentation():
        return train_generator, val_generator, callback_generator

    normalizer = NORMALIZATION_FUNCTIONS[config[task].get('NORMALIZATION', 'none')]()[0]
    noise_function = get_composition(config[task].get('NOISE', [None]),
                                     config[task].get('NOISE_PARAMETERS', (())))
    zoom_factor = config[task].get('ZOOM', 0)
    rotation = config[task].get('ROTATION', 0)

    functions = {"style_transfer": get_im2im_data_augmentation,
                 "saliency": get_im2im_data_augmentation,
                 "segmenter": get_im2im_data_augmentation,
                 "autoencoder": get_im2im_data_augmentation,
                 "categorizer": get_categorizer_augmentation,
                 "inceptionV3": get_categorizer_augmentation,
                 "unet_categorizer": get_categorizer_augmentation,
                 "regressor": get_regressor_augmentation,
                 }
    assert task in functions, f'You asked for {task}, but functions.keys is {list(functions.keys())}'
    return functions[task]()


def get_models(config, task, name, train_folder, default_input_shape=DEFAULT_INPUT_SHAPE, load=False):
    def get_autoencoder_model():
        if task == "unet_categorizer":
            model = import_unet_categorizer(name=name, config=config[task], load=load)
            model.labels = labels
        elif task == "autoencoder":
            model = import_unet_model(name=name, config=config[task], load=load, skip=True)
        else:
            model = import_unet_model(name=name, config=config[task], load=load)
        model.callback_titles = ['Input', 'Prediction', 'Truth'] + labels
        return model

    def get_categorizer_model():
        if task == 'inceptionV3':
            model = InceptionV3(input_shape, len(labels), name, load=load,
                                imagenet=config[task].get('IMAGENET', False),
                                last_activation=config[task].get('LAST_ACTIVATION', 'softmax'))
        else:
            model = import_categorizer(len(labels), config=config[task], name=name, load=load)
        model.labels = labels
        return model

    def get_regressor_model():
        attributes = config[task].get('ATTRIBUTES')
        if isinstance(attributes, str):
            while ' ' in attributes:
                attributes = attributes.replace(' ', '')
            if attributes.startswith('(') and attributes.endswith(')'):
                attributes = attributes[1:-1]
            attributes = attributes.split(',')

        model = InceptionV3(input_shape, len(attributes), name, load=load, imagenet=config[task].get('IMAGENET', False),
                            last_activation='tanh', loss='mse', metrics=[], last_dense=True)
        model.labels = attributes
        return model

    folders = config[task].get('LABELS')
    path_labels = list_dir(train_folder) if folders is None else [os.path.join(train_folder, folder) for folder in
                                                                  folders[1:-1].split(', ')]
    labels = [os.path.split(label)[-1] for label in path_labels]
    input_shape = config[task].get('INPUT_SHAPE', default_input_shape)
    functions = {"saliency": get_autoencoder_model,
                 "autoencoder": get_autoencoder_model,
                 "segmenter": get_autoencoder_model,
                 "unet_categorizer": get_autoencoder_model,
                 "categorizer": get_categorizer_model,
                 "inceptionV3": get_categorizer_model,
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
                     RegressorCallback(callback_generator, validation_steps)
                     ]
        return callbacks

    denormalizer = NORMALIZATION_FUNCTIONS[config[task].get('NORMALIZATION', 'intensity')]()[1]
    batch_size = config[task].get('BATCH_SIZE')
    training_steps = config[task].get('TRAINING_STEPS')
    validation_steps = config[task].get('VALIDATION_STEPS')

    functions = {"saliency": get_im2im_callbacks,
                 "autoencoder": get_im2im_callbacks,
                 "segmenter": get_im2im_callbacks,
                 "style_transfer": get_im2im_callbacks,
                 "categorizer": get_categorizer_callbacks,
                 "inceptionV3": get_categorizer_callbacks,
                 "unet_categorizer": get_categorizer_callbacks,
                 "regressor": get_regressor_callback,
                 }
    assert task in functions, f'You asked for {task}, but functions.keys is {list(functions.keys())}'
    return functions[task]()


def main(task, dataset, batch_size=BATCH_SIZE, epochs=EPOCHS,
         training_steps=TRAINING_STEPS, validation_steps=VALIDATION_STEPS, initial_epoch=INITIAL_EPOCH,
         root=ROOT,
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
    config[task]['sys.argv'] = ' '.join(sys.argv)
    task = config[task]['TASK']

    train_folder, val_folder = get_dataset_roots(dataset=dataset, root=root)
    train_generator, val_generator, callback_generator, train_dir = get_generators(config, task, dataset, batch_size,
                                                                                   train_folder, val_folder)
    train_generator, val_generator, callback_generator = get_data_augmentation(config, task, train_generator,
                                                                               val_generator, callback_generator)
    name = config[task].get('NAME', f'{dataset}_{task}')
    model = get_models(config, task, name, train_folder, load=initial_epoch != 0)
    callbacks = get_callbacks(config, task, model, callback_generator)
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
                        # class_weight=class_weight
                        )


if __name__ == '__main__':
    fire.Fire(main)
