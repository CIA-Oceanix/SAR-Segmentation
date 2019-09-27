import sys
import pygame

from Rignak_DeepLearning.Categorizer.flat import import_model
from Rignak_DeepLearning.ReinforcementLearning.ReinforcementLearning import train

"""
>>> python Launcher.py Asteroid
"""

CONV_LAYERS = (4, 8, 16)
DENSE_LAYERS = (8, 8)
LAST_ACTIVATION = 'linear'



def parse_input(argvs):
    if argvs[1] == 'Asteroid':
        import Rignak_Games.Asteroid as library
        from Rignak_Games.Asteroid.Inputs import COMMAND_NUMBER
        from Rignak_Games.Asteroid.GameState import GameState
        INPUT_SHAPE = (128, 128, 4)
        model = import_model(library.Inputs.COMMAND_NUMBER,
                             input_shape=INPUT_SHAPE,
                             conv_layers=CONV_LAYERS,
                             dense_layers=DENSE_LAYERS,
                             last_activation=LAST_ACTIVATION,
                             learning_rate=0)
        model.summary()
    return library, model, INPUT_SHAPE, COMMAND_NUMBER, GameState


if __name__ == '__main__':
    import tensorflow as tf
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    from keras import backend as K
    K.set_session(sess)

    pygame.init()
    library, model, input_shape, command_number, GameState = parse_input(sys.argv)
    train(library, model, input_shape, command_number, GameState)
