import tensorflow as tf
from tensorflow import keras


def down_flow(filters, size, stride=1, apply_pooling=False, apply_batchnorm=True):
    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2D(filters, size, stride, padding='same'))

    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())
    result.add(tf.keras.layers.ReLU())

    if apply_pooling:
        result.add(tf.keras.layers.MaxPool2D())

    return result


def up_flow(filters, size, stride=1, apply_transpose=False):
    result = tf.keras.Sequential()
    if apply_transpose:
        result.add(
            tf.keras.layers.Conv2DTranspose(filters, size, strides=2))
    else:
        result.add(
            tf.keras.layers.Conv2D(filters, size, strides = stride, padding='same'))

    result.add(tf.keras.layers.BatchNormalization())
    result.add(tf.keras.layers.ReLU())

    return result

