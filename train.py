import os, sys, fire

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
for arg in sys.argv:
    if arg.startswith('--run_on_gpu='):
        os.environ['CUDA_VISIBLE_DEVICES'] = arg.split('=')[-1]
        
    seed_value = 42
    if arg.startswith('--seed='):
        seed_value = int(arg.split('=')[-1])
     
task = sys.argv[1]

#import Rignak_DeepLearning.deprecation_warnings

#Rignak_DeepLearning.deprecation_warnings.filter_warnings()

import tensorflow as tf
#tf.compat.v1.disable_eager_execution()
import tensorflow.keras as keras

physical_devices = tf.config.list_physical_devices('GPU')
for physical_device in physical_devices:
    tf.config.experimental.set_memory_growth(physical_device, True)

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from Rignak_DeepLearning.data import get_dataset_roots
from Rignak_DeepLearning.noise import get_composition
from Rignak_DeepLearning.callbacks import \
    HistoryCallback, \
    AutoencoderExampleCallback, \
    ConfusionCallback, \
    ClassificationExampleCallback, \
    RegressorCallback, \
    TaggerCallback, \
    SaveAttributes

from Rignak_DeepLearning.generator import augment_generator, rotsym_augmentor
from Rignak_DeepLearning.config import get_config

if task in ("saliency", "autoencoder", "segmenter", "flat_autoencoder", "gan_segmenter", "experimental_unet", ):
    from Rignak_DeepLearning.Image_to_Image.unet import import_model as import_unet_model
    from Rignak_DeepLearning.Image_to_Image.generator import \
        segmenter_base_generator as segmenter_generator, \
        saliency_base_generator as saliency_generator, \
        autoencoder_base_generator as autoencoder_generator
       
    if task in ("experimental_unet", ):
        from Rignak_DeepLearning.Image_to_Image.experimental_unet import import_model

elif task in ("categorizer", "inceptionV3"):
    from Rignak_DeepLearning.Image_to_Class.Inception import import_model_v3 as InceptionV3
    from Rignak_DeepLearning.Image_to_Class.generator import categorizer_base_generator as categorizer_generator

elif task in ('regressor',):
    from Rignak_DeepLearning.Image_to_Class.Inception import import_model_v3 as InceptionV3
    from Rignak_DeepLearning.Image_to_float.generator import regressor_base_generator as regressor_generator

elif task in ('tagger',):
    from Rignak_DeepLearning.Image_to_Class.Inception import import_model_v3 as InceptionV3
    from Rignak_DeepLearning.Image_to_Tag.generator import tagger_base_generator as tagger_generator

elif task in ("siamese",):
    from Rignak_DeepLearning.Image_to_Distance.model import import_model as import_siamese_model
    from Rignak_DeepLearning.Image_to_Distance.generator import image_distance_base_generator, siamese_augment_generator
    from Rignak_DeepLearning.Image_to_Distance.callbacks import DistanceCallback
    
if task in ('gan_autoencoder', 'gan_segmenter'):
    from Rignak_DeepLearning.GAN.Yes_UGAN import Yes_UGAN as import_gan_model
    from Rignak_DeepLearning.Image_to_Image.generator import autoencoder_base_generator as autoencoder_generator
    from Rignak_DeepLearning.GAN.generator import gan_im2im_generator

from Rignak_Misc.path import list_dir

BATCH_SIZE = 16
TRAINING_STEPS = 2000
VALIDATION_STEPS = 500
EPOCHS = 1000
INITIAL_EPOCH = 0

DEFAULT_INPUT_SHAPE = (256, 256, 3)

ROOT = 'C:\\Users/Rignak/Documents/datasets'
if not os.path.exists(ROOT): ROOT = 'E:\\datasets'


def get_generators(config, task, batch_size, train_folder, val_folder,
                   default_input_shape=DEFAULT_INPUT_SHAPE):
    def get_saliency_generators():
        kwargs = {"input_shape": input_shape, "batch_size": batch_size, "output_shape": output_shape,
                  "folders": folders, "attributes": attributes}
        train_generator = saliency_generator(train_folder, **kwargs)
        val_generator = saliency_generator(val_folder, validation=True, **kwargs)
        callback_generator = saliency_generator(val_folder, **kwargs)
        next(train_generator), next(val_generator), next(callback_generator)
        return train_generator, val_generator, callback_generator, train_folder

    def get_autoencoder_generators():
        train_generator = autoencoder_generator(train_folder, input_shape=input_shape, batch_size=batch_size)
        val_generator = autoencoder_generator(val_folder, validation=True, input_shape=input_shape, batch_size=batch_size)
        callback_generator = autoencoder_generator(val_folder, input_shape=input_shape, batch_size=batch_size)
        next(train_generator), next(val_generator), next(callback_generator)
        return train_generator, val_generator, callback_generator, train_folder

    def get_categorizer_generators():
        kwargs = {"input_shape": input_shape, "batch_size": batch_size, "folders": folders, "attributes": attributes}
        train_generator = categorizer_generator(train_folder, **kwargs)
        val_generator = categorizer_generator(val_folder, validation=True, **kwargs)
        callback_generator = categorizer_generator(val_folder, **kwargs)
        next(train_generator), next(val_generator), next(callback_generator)
        return train_generator, val_generator, callback_generator, train_folder

    def get_regressor_generators():
        assert attributes, 'No --ATTRIBUTES were passed'
        kwargs = {"input_shape": input_shape, "batch_size": batch_size, 
                  "additional_inputs": config[task].get('ADDITIONAL_INPUTS', [])}
        train_generator = regressor_generator(train_folder, attributes, **kwargs)
        val_generator = regressor_generator(val_folder, attributes, validation=True, **kwargs)
        callback_generator = regressor_generator(val_folder, attributes, **kwargs)

        config[task]['MEANS'], config[task]['STDS'] = next(train_generator)
        config[task]['VAL_MEANS'], config[task]['VAL_STDS'] = next(val_generator)
        next(callback_generator)
        return train_generator, val_generator, callback_generator, train_folder

    def get_tagger_generators():
        kwargs = {"input_shape": input_shape, "batch_size": batch_size}
        train_generator = tagger_generator(train_folder, **kwargs)
        val_generator = tagger_generator(val_folder, validation=True, **kwargs)
        callback_generator = tagger_generator(val_folder, **kwargs)

        config[task]['LABELS'] = next(train_generator)[0]
        next(val_generator), next(callback_generator)
        return train_generator, val_generator, callback_generator, train_folder

    def get_segmenter_generators():
        input_label = config[task].get('INPUT_LABEL', 'input')
        output_label = config[task].get('OUTPUT_LABEL', 'output')

        kwargs = {"input_shape": input_shape, "output_shape": output_shape, "batch_size": batch_size,
                  "input_label": input_label, "output_label": output_label, "attributes": attributes}

        train_generator = segmenter_generator(train_folder, **kwargs)
        val_generator = segmenter_generator(val_folder, validation=True, **kwargs)
        callback_generator = segmenter_generator(train_folder, **kwargs)

        next(train_generator), next(val_generator), next(callback_generator)
        return train_generator, val_generator, callback_generator, train_folder

    def get_siamese_generators():
        kwargs = {"batch_size": batch_size, "input_shape": input_shape}

        train_generator = image_distance_base_generator(train_folder, **kwargs)
        val_generator = image_distance_base_generator(val_folder, **kwargs)
        callback_generator = image_distance_base_generator(val_folder, **kwargs)

        next(train_generator), next(val_generator), next(callback_generator)
        return train_generator, val_generator, callback_generator, train_folder

    input_shape = config[task].get('INPUT_SHAPE', default_input_shape)
    output_shape = config[task].get('OUTPUT_SHAPE', input_shape)
    attributes = config[task].get('ATTRIBUTES')
    folders = config[task].get('LABELS')

    functions = {
        "saliency": get_saliency_generators,
        "autoencoder": get_autoencoder_generators,
        "segmenter": get_segmenter_generators,
        "experimental_unet": get_segmenter_generators,
        "inceptionV3": get_categorizer_generators,
        "regressor": get_regressor_generators,
        "tagger": get_tagger_generators,
        "gan_autoencoder": get_autoencoder_generators,
        "gan_segmenter": get_segmenter_generators,
        "siamese": get_siamese_generators,
    }
    return functions[task](), config


def get_data_augmentation(config, task, root, train_generator, val_generator, callback_generator):
    def get_im2im_data_augmentation():
        kwargs = {"noise_function": noise_function, "apply_on_output": True, "zoom_factor": zoom_factor,
                  "rotation": rotation}
        new_train_generator = augment_generator(train_generator, **kwargs)
        new_callback_generator = augment_generator(callback_generator, **kwargs)

        #new_train_generator = rotsym_augmentor(new_train_generator)
        #new_callback_generator = rotsym_augmentor(new_callback_generator)

        return new_train_generator, val_generator, new_callback_generator
        
    def get_im2im_gan_augmentation():
        new_train_generator, new_val_generator, new_callback_generator = get_im2im_data_augmentation()
        new_train_generator = gan_im2im_generator(new_train_generator)
        new_val_generator = gan_im2im_generator(new_val_generator)
        new_callback_generator = gan_im2im_generator(new_callback_generator)
        return new_train_generator, new_val_generator, new_callback_generator

    def get_categorizer_augmentation():
        kwargs = {"noise_function": noise_function, "apply_on_output": False, "zoom_factor": zoom_factor,
                  "rotation": rotation}

        new_train_generator = augment_generator(train_generator, **kwargs)
        new_callback_generator = augment_generator(callback_generator, **kwargs)

        return new_train_generator, val_generator, new_callback_generator

    def get_siamese_augmentation():
        kwargs = {"zoom_factor": zoom_factor, "rotation": rotation}

        new_train_generator = siamese_augment_generator(train_generator, **kwargs)
        new_val_generator = siamese_augment_generator(val_generator, **kwargs)
        new_callback_generator = siamese_augment_generator(callback_generator, **kwargs)
        return new_train_generator, new_val_generator, new_callback_generator
        
    noise_function = get_composition(config[task].get('NOISE', [None]), config[task].get('NOISE_PARAMETERS', (())))
    zoom_factor = config[task].get('ZOOM', 0)
    rotation = config[task].get('ROTATION', 0)

    functions = {
        "saliency": get_im2im_data_augmentation,
        "segmenter": get_im2im_data_augmentation,
        "experimental_unet": get_im2im_data_augmentation,
        "autoencoder": get_im2im_data_augmentation,
        "inceptionV3": get_categorizer_augmentation,
        "regressor": get_categorizer_augmentation,
        "tagger": get_categorizer_augmentation,
        "gan_autoencoder": get_im2im_gan_augmentation,
        "gan_segmenter": get_im2im_gan_augmentation,
        "siamese": get_siamese_augmentation
    }
    assert task in functions, f'You asked for {task}, but functions.keys is {list(functions.keys())}'
    generators = functions[task]()

    return generators


def get_models(config, task, name, train_folder):
    def get_autoencoder_model():
        print()
        print(task)
        print()
        if task == "autoencoder":
            model = import_unet_model(name=name, config=config[task], skip=False)
        elif task == "experimental_unet":
            print('load transunet')
            model = import_model(config[task]['ARCHITECTURE'], name=name, config=config[task], metrics=config[task].get('METRICS'), labels=labels)
            model.summary()
          
        else:
            model = import_unet_model(name=name, config=config[task], metrics=config[task].get('METRICS'),
                                      labels=labels, additional_input_number=len(attributes))
        model.callback_titles = ['Input', 'Prediction', 'Truth'] + labels
        return model

    def get_categorizer_model():
        if 'LABELS' not in config[task]:
            config[task]['LABELS'] = labels
        model = InceptionV3(config=config[task], name=name, resnet=config[task].get('RESNET'),
                            additional_input_number=len(attributes))
        return model

    def get_regressor_model():
        config[task]['LABELS'] = attributes
        model = InceptionV3(config=config[task], name=name,
                            additional_input_number=len(config[task].get('ADDITIONAL_INPUTS', [])))
        return model

    def get_gan_model():
        model = import_gan_model(name=name, config=config[task], skip='segmenter' in task)
        model.generator.callback_titles = ['Input', 'Prediction', 'Truth'] + labels
        return model

    def get_siamese_model():
        model = import_siamese_model(name=name, config=config[task])
        model.labels = labels
        return model

    attributes = config[task].get('ATTRIBUTES', [])
    if isinstance(attributes, str):
        while ' ' in attributes:
            attributes = attributes.replace(' ', '')
        if attributes.startswith('(') and attributes.endswith(')'):
            attributes = attributes[1:-1]
        attributes = attributes.split(',')

    labels = config[task].get('LABELS')
    if labels is None:
        labels = [os.path.split(folder)[-1] for folder in list_dir(train_folder)]
    elif isinstance(labels, str):
        labels = [os.path.split(label)[-1] for label in labels[1:-1].split(', ')]
    functions = {
        "saliency": get_autoencoder_model,
        "autoencoder": get_autoencoder_model,
        "segmenter": get_autoencoder_model,
        "experimental_unet": get_autoencoder_model,
        "inceptionV3": get_categorizer_model,
        "regressor": get_regressor_model,
        "tagger": get_categorizer_model,
        "gan_autoencoder": get_gan_model,
        "gan_segmenter": get_gan_model,
        "siamese": get_siamese_model
    }
    assert task in functions, f'You asked for {task}, but functions.keys is {list(functions.keys())}'
    return functions[task]()


def get_callbacks(config, task, model, callback_generator):
    def get_im2im_callbacks():
        callbacks = [
            SaveAttributes(callback_generator, config[task]),
            ModelCheckpoint(model.weight_filename, save_best_only=True, save_weights_only=False),
            HistoryCallback(batch_size, training_steps),
            AutoencoderExampleCallback(callback_generator),
            EarlyStopping(patience=100)
        ]
        return callbacks

    def get_categorizer_callbacks():
        callbacks = [
            SaveAttributes(callback_generator, config[task], labels=model.labels),
            ModelCheckpoint(model.weight_filename, save_best_only=True, save_weights_only=False),
            # ModelCheckpoint(model.weight_filename + ".{epoch:02d}.h5", save_best_only=True),
            HistoryCallback(batch_size, training_steps),
            ConfusionCallback(callback_generator, model.labels),
            ClassificationExampleCallback(callback_generator),
            EarlyStopping(patience=10)
        ]
        return callbacks

    def get_regressor_callbacks():
        callbacks = [
            SaveAttributes(callback_generator, config[task]),
            ModelCheckpoint(model.weight_filename, save_best_only=True, save_weights_only=False),
            HistoryCallback(batch_size, training_steps),
            RegressorCallback(callback_generator, validation_steps, config[task]['ATTRIBUTES'],
                              config[task]['VAL_MEANS'], config[task]['VAL_STDS']),
            EarlyStopping(patience=50)
        ]
        return callbacks

    def get_tagger_callbacks():
        callbacks = [
            SaveAttributes(callback_generator, config[task]),
            HistoryCallback(batch_size, training_steps),
            TaggerCallback(callback_generator, validation_steps, config[task]['LABELS']),
            EarlyStopping(patience=20),
            ModelCheckpoint(model.weight_filename, save_best_only=True, save_weights_only=False)
        ]
        return callbacks

    def get_gan_callbacks():
        callbacks = [
            SaveAttributes(callback_generator, config[task]),
            #ModelCheckpoint(model.weight_filename, save_best_only=True),
            HistoryCallback(batch_size, training_steps),
            AutoencoderExampleCallback(callback_generator),
            EarlyStopping(patience=10)
        ]
        return callbacks

    def get_siamese_callbacks():
        callbacks = [
            SaveAttributes(callback_generator, config[task]),
            ModelCheckpoint(model.weight_filename, save_best_only=True),
            HistoryCallback(batch_size, training_steps),
            DistanceCallback(callback_generator, validation_steps, model.labels),
            EarlyStopping(patience=10)
        ]
        return callbacks

    batch_size = config[task].get('BATCH_SIZE')
    training_steps = config[task].get('TRAINING_STEPS')
    validation_steps = config[task].get('VALIDATION_STEPS')

    functions = {
        "saliency": get_im2im_callbacks,
        "autoencoder": get_im2im_callbacks,
        "segmenter": get_im2im_callbacks,
        "experimental_unet": get_im2im_callbacks,
        "inceptionV3": get_categorizer_callbacks,
        "regressor": get_regressor_callbacks,
        "tagger": get_tagger_callbacks,
        "gan_autoencoder": get_gan_callbacks,
        "gan_segmenter": get_gan_callbacks,
        "siamese": get_siamese_callbacks
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
    (train_generator, val_generator, callback_generator, train_dir), config = get_generators(config, task, batch_size,
                                                                                             train_folder, val_folder)
    train_generator, val_generator, callback_generator = get_data_augmentation(config, task, root, train_generator,
                                                                               val_generator, callback_generator)
    name = config[task].get('NAME', f'{dataset}_{task}')
    model = get_models(config, task, name, train_folder)
    callbacks = get_callbacks(config, task, model, callback_generator)
    train(model, train_generator, val_generator, callbacks,
          epochs=epochs, training_steps=training_steps, validation_steps=validation_steps, initial_epoch=initial_epoch)


def train(model, train_generator, val_generator, callbacks, training_steps=TRAINING_STEPS,
          validation_steps=VALIDATION_STEPS, epochs=EPOCHS, initial_epoch=INITIAL_EPOCH):
    model.fit(x=train_generator,
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
