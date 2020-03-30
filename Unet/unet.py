import tensorflow as tf
import numpy as pn
from Unet.network_architecture_helper import *

class Unet():
    def __init__(self):
        self.model = self.create()
        self.l2_loss = tf.keras.losses.MeanSquaredError()
        self.optimizer = tf.keras.optimizers.Adam(1e-4)

    def __call__(self, input): # for inference.
        return self.model(input, training=False)

    def create(self):
        self.model = tf.keras.Sequential()
        input = tf.keras.layers.Input(shape=(28, 28, 1), dtype=tf.float32)
        x = input
        x, skip_layers = self.addNpass_down_path(x)
        x = self.add_up_path(x, skip_layers)

        # residual learning
        # x = x + input

        return tf.keras.Model(inputs=input, outputs=x)

    def Unet_loss(self, output, label):
        return self.l2_loss(output, label)

    def addNpass_down_path(self, x):
        skip_layer = []

        x = addNpass_conv_block(x, 64)
        assert tuple(x.shape) == (None, 28, 28, 64)
        skip_layer.append(x)

        x = addNpass_maxpooling(x)
        assert tuple(x.shape) == (None, 14, 14, 64)

        x = addNpass_conv_block(x, 128)
        assert tuple(x.shape) == (None, 14, 14, 128)
        skip_layer.append(x)

        x = addNpass_maxpooling(x)
        assert tuple(x.shape) == (None, 7, 7, 128)

        x = addNpass_conv_block(x, 256)
        assert tuple(x.shape) == (None, 7, 7, 256)
        x = addNpass_conv_block(x, 512)
        assert tuple(x.shape) == (None, 7, 7, 512)
        x = addNpass_conv_block(x, 256)
        assert tuple(x.shape) == (None, 7, 7, 256)

        return x, skip_layer

    def add_up_path(self, x, skip_layers):
        x = addNpass_transpose_conv2d(x, 128, 2)
        assert tuple(x.shape) == (None, 14, 14, 128)

        x = tf.keras.layers.Concatenate()([x, skip_layers[-1]])
        assert tuple(x.shape) == (None, 14, 14, 256)

        x = addNpass_conv_block(x, 128)
        assert tuple(x.shape) == (None, 14, 14, 128)

        x = addNpass_transpose_conv2d(x, 64, 2)
        assert tuple(x.shape) == (None, 28, 28, 64)

        x = tf.keras.layers.Concatenate()([x, skip_layers[-2]])
        assert tuple(x.shape) == (None, 28, 28, 128)

        x = addNpass_conv_block(x, 64)
        assert tuple(x.shape) == (None, 28, 28, 64)
        x = addNpass_conv_block(x, 1, BN=False)
        assert tuple(x.shape) == (None, 28, 28, 1)

        return x




