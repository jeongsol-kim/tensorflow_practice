import tensorflow as tf
import numpy as np

class DCGAN:
    def __init__(self):
        self.g_model = None
        self.d_model = None
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.generator_optimizer = tf.keras.optimizers.Adam(lr=1e-4)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(lr=1e-4)

    def make_generator_model(self):
        self.g_model = tf.keras.Sequential()
        self.g_model.add(tf.keras.layers.Dense(7*7*256, use_bias=False, input_shape=(100, )))
        self.g_model.add(tf.keras.layers.BatchNormalization())
        self.g_model.add(tf.keras.layers.ReLU())

        self.g_model.add(tf.keras.layers.Reshape((7, 7, 256)))
        assert self.g_model.output_shape == (None, 7, 7, 256)  # 주목: 배치사이즈로 None이 주어집니다.

        self.g_model.add(tf.keras.layers.Conv2DTranspose(128, (5, 5), (1, 1), padding='same', use_bias=False))
        assert self.g_model.output_shape == (None, 7, 7, 128)
        self.g_model.add(tf.keras.layers.BatchNormalization())
        self.g_model.add(tf.keras.layers.ReLU())

        self.g_model.add(tf.keras.layers.Conv2DTranspose(64, (5, 5), (2, 2), padding='same', use_bias=False))
        assert self.g_model.output_shape == (None, 14, 14, 64)
        self.g_model.add(tf.keras.layers.BatchNormalization())
        self.g_model.add(tf.keras.layers.ReLU())

        self.g_model.add(tf.keras.layers.Conv2DTranspose(1, (5, 5), (2, 2), padding='same', use_bias=False))
        assert self.g_model.output_shape == (None, 28, 28, 1)

        return self.g_model

    def make_discriminator_model(self):
        self.d_model = tf.keras.Sequential()
        self.d_model.add(tf.keras.layers.Conv2D(64, (5, 5), (2, 2), padding='same', input_shape=[28, 28, 1]))
        self.d_model.add(tf.keras.layers.ReLU())
        self.d_model.add(tf.keras.layers.Dropout(0.3))

        self.d_model.add(tf.keras.layers.Conv2D(128, (5, 5), (2, 2), padding='same'))
        self.d_model.add(tf.keras.layers.ReLU())
        self.d_model.add(tf.keras.layers.Dropout(0.3))

        self.d_model.add(tf.keras.layers.Flatten())
        self.d_model.add(tf.keras.layers.Dense(1))

        return self.d_model

    def generator_loss(self, fake_output):
        return self.cross_entropy(tf.ones_like(fake_output), fake_output)

    def discriminator_loss(self, real_output, fake_output):
        real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss


