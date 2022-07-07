import sys
import pygame

from Rignak_DeepLearning.ReinforcementLearning.ReinforcementLearning import train

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, MaxPooling2D, concatenate, Dense, Input,Flatten
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

"""
>>> python Launcher.py Asteroid
"""

CONV_LAYERS = (8, 16, 32)
DENSE_LAYERS = (32,)
LAST_ACTIVATION = 'sigmoid'
ACTIVATION = 'relu'
INPUT_SHAPE = (128, 128, 3)
KERNEL_SIZE = 3

def import_model(n_outputs, input_shape=INPUT_SHAPE, 
                 conv_layers=CONV_LAYERS, dense_layers=DENSE_LAYERS,
                 last_activation=LAST_ACTIVATION, activation=ACTIVATION, 
                 kernel_size=KERNEL_SIZE):

    get_layer = lambda layer: Conv2D(layer[0], kernel_size, 
                                     activation=activation,  padding='same',  
                                     kernel_regularizer=regularizers.L2(l2=1e-3), 
                                     bias_regularizer=regularizers.L2(l2=1e-4),
                                     )(layer[1])
    input_layer = Input(input_shape)
    for i, conv_layer in enumerate(conv_layers):
       layer = get_layer((conv_layer, layer)) if i else get_layer((conv_layer, input_layer))
       layer = get_layer((conv_layer, layer))
       layer = MaxPooling2D(pool_size=(2,2))(layer)
    layer = Flatten()(layer)
    for dense_layer in dense_layers:
        layer = Dense(dense_layer, activation=activation,
                                     kernel_regularizer=regularizers.L2(l2=1e-3), 
                                     bias_regularizer=regularizers.L2(l2=1e-3),)(layer)
    layer = Dense(n_outputs,activation='linear',
                                     kernel_regularizer=regularizers.L2(l2=1e-3), 
                                     bias_regularizer=regularizers.L2(l2=1e-3),)(layer)
    
    model = Model(inputs=[input_layer], outputs=[layer])

    optimizer = Adam(learning_rate=10**-3)
    model.compile(optimizer=optimizer, loss='mse', run_eagerly=True)
    return model


def parse_input(argvs):
    if True or argvs[1] == 'Asteroid':
        import Rignak_Games.Asteroid as library
        from Rignak_Games.Asteroid.Inputs import COMMAND_NUMBER
        from Rignak_Games.Asteroid.GameState import GameState
        model = import_model(COMMAND_NUMBER)
        model.summary()
    return library, model, INPUT_SHAPE, COMMAND_NUMBER, GameState


if __name__ == '__main__':
    pygame.init()
    library, model, input_shape, command_number, GameState = parse_input(sys.argv)
    train(library, model, input_shape, command_number, GameState)
