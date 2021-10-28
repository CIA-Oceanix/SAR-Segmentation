import os

from keras_unet_collection.models import u2net_2d, transunet_2d, swin_unet_2d
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam

from Rignak_Misc.path import get_local_file
from Rignak_DeepLearning.models import write_summary, load_weights
from Rignak_DeepLearning.config import get_config
from Rignak_DeepLearning.loss import LOSS_TRANSLATION, get_metrics


ROOT = get_local_file(__file__, os.path.join('..', '_outputs'))
LEARNING_RATE = 10 ** -5

CONFIG_KEY = 'segmenter'
CONFIG = get_config()[CONFIG_KEY]
DEFAULT_NAME = CONFIG.get('NAME', 'DEFAULT_MODEL_NAME')

def import_model(architecture, root=ROOT, learning_rate=LEARNING_RATE,
                 config=CONFIG, name=DEFAULT_NAME, metrics=None, labels=range(3)):
    load = config.get('LOAD', False)
    conv_layers = list(config['CONV_LAYERS'])
    conv_layers += [config.get('CENTRAL_SHAPE', conv_layers[-1] * 2)]
    learning_rate = config.get('LEARNING_RATE', learning_rate)
    input_shape = config.get('INPUT_SHAPE', (512, 512, 3))
    batch_normalization = config.get('BATCH_NORMALIZATION', False)
    freeze = config.get('FREEZE', False)
    
    output_shape = config.get('OUTPUT_SHAPE', input_shape)
    
    activation = config.get('ACTIVATION', 'ReLU')
    if activation == 'sin':
        activation = K.sin
    last_activation = config.get('LAST_ACTIVATION', 'Sigmoid')
    if last_activation == 'sin':
        last_activation = K.sin
    loss = config.get('LOSS', 'mse')
    loss = LOSS_TRANSLATION.get(loss, loss)

    optimizer = Adam(learning_rate=learning_rate)
    
    if architecture == "u2net":
        model = u2net_2d(
            input_shape, output_shape[-1], conv_layers, name=name, 
            activation=activation, output_activation=last_activation,
            batch_norm=batch_normalization
        )
    elif architecture == "transunet":
        model = transunet_2d(
            input_shape, conv_layers, output_shape[-1], name=name, 
            activation=activation, output_activation=last_activation,
            batch_norm=batch_normalization, freeze_backbone=freeze
        )
    elif architecture == "swin_unet":
        model = swin_unet_2d(
            input_shape, 64, output_shape[-1], 4, 
            3, 3, (2,2), [4, 8, 16, 16],[4, 2, 2, 2], 128,
            output_activation=last_activation, name=name
        )
        
    model.compile(optimizer=optimizer, loss=loss, run_eagerly=True)

    print(root, name)
    model.weight_filename = os.path.join(root, name, "model.h5")
    model.summary_filename = os.path.join(root, name, "model.txt")
        
    load_weights(model, model.weight_filename, load, freeze)
    write_summary(model)
    return model

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = "-1"
    config = {
        "INPUT_SHAPE": (256,256,1),
        "CONV_LAYERS": (32,64, 128),
        "LABELS": (0, 1, 2)
    }
    model = import_model("transunet", config=config)
    model.summary()
