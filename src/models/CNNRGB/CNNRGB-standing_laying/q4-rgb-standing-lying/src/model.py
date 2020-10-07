from tensorflow.keras import models, layers
from tensorflow import keras
from config import CONFIG

base_model = keras.applications.Xception(
    weights="imagenet",  # Load weights pre-trained on ImageNet.
    input_shape=(CONFIG.IMAGE_TARGET_HEIGHT, CONFIG.IMAGE_TARGET_WIDTH, 3),
    include_top=False,
)


def create_cnn(input_shape, dropout):

    base_model.trainable = False

    model = models.Sequential()
    model.add(base_model)
    model.add(layers.Flatten())
    if dropout is True:
        model.add(layers.Dropout(0.02))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    return model


def fine_tuning(layer_name):
    set_trainable = False
    for layer in base_model.layers:
        if layer.name == layer_name:
            set_trainable = True
        if set_trainable:
            layer.trainable = True
        else:
            layer.trainable = False
