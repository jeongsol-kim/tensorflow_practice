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

def make_down_stack(IFS):
    return [
                down_flow(IFS, 3), # 28
                down_flow(IFS, 3, apply_pooling=True), # 14
                down_flow(IFS * 2, 3), # 14
                #down_flow(IFS * 2, 3, apply_pooling=True), # 7
                down_flow(IFS * 4, 3), # 7
                #down_flow(IFS * 4, 3, apply_pooling=True), # 3
                down_flow(IFS * 8, 3), # 7
                down_flow(IFS * 16, 3), # 7
                down_flow(IFS * 16, 3) # 7
            ]

def make_up_stack(IFS):
    return [
                #up_flow(IFS * 8, 2, apply_transpose=True), # 14
                up_flow(IFS * 8, 2), # 14
                #up_flow(IFS * 4, 2, apply_transpose=True), # 16
                up_flow(IFS * 4, 2), # 14
                #up_flow(IFS * 2, 2, apply_transpose=True), # 32
                up_flow(IFS * 4, 2), #
                up_flow(IFS * 2, 2, apply_transpose=True), # 64
                up_flow(IFS * 1, 2) # 64
            ]