from keras.models import Model
from keras.layers import Input, Conv2D, Dropout, MaxPooling2D, Flatten, Dense
from keras.optimizers import SGD, Adam


def get_model(commands, input_shape, learning_rate=10 ** -3):
    input_layer = Input(input_shape)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(input_layer)
    x = MaxPooling2D()(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D()(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D()(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D()(x)
    x = Flatten()(x)
    x = Dense(16, activation='relu')(x)
    x = Dense(commands, activation='sigmoid')(x)

    model = Model(inputs=input_layer, outputs=x)
    model.compile(optimizer=Adam(lr=learning_rate), loss='binary_crossentropy')

    print("We finish building the model")
    return model
