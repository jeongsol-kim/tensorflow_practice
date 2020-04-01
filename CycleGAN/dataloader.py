import tensorflow as tf
import tensorflow_datasets as tfd
import CycleGAN.hyperparams as hp
import numpy as np
import matplotlib.pyplot as plt
from Unet.noise_helper import NoiseHelper

class DataLoader():
    def __init__(self, patch=False):
        self.monet_train = tfd.load(name='cycle_gan/monet2photo', split='trainA')
        self.pic_train = tfd.load(name='cycle_gan/monet2photo', split='trainB')
        self.monet_test = tfd.load(name='cycle_gan/monet2photo', split='testA')
        self.pic_test = tfd.load(name='cycle_gan/monet2photo', split='testB')

        self.PATCH_SIZE = hp.PATCH_SIZE
        self.IsPatchBased = patch
        self.BATCH_SIZE = hp.BATCH_SIZE

    def batch_preparing(self, limit = 0):
        # Extract images
        self.monet_train = tf.convert_to_tensor([x['image'] for x in self.monet_train], dtype=tf.float32)/255.0
        self.pic_train = tf.convert_to_tensor([x['image'] for x in self.pic_train], dtype=tf.float32)/255.0
        self.monet_test = tf.convert_to_tensor([x['image'] for x in self.monet_test], dtype=tf.float32)/255.0
        self.pic_test = tf.convert_to_tensor([x['image'] for x in self.pic_test], dtype=tf.float32)/255.0

        if limit != 0:
            self.monet_train = self.monet_train[:limit, :, :, :]
            self.pic_train = self.monet_train[:limit, :, :, :]
            self.monet_test = self.monet_test[:limit, :, :, :]
            self.pic_test = self.pic_test[:limit, :, :, :]

        # If the training is patch-based, separate dataset into patches.
        if self.IsPatchBased:
            self.monet_train = self.patch_separating(self.monet_train)
            self.pic_train = self.patch_separating(self.pic_train)
            self.monet_test = self.patch_separating(self.monet_test)
            self.pic_test = self.patch_separating(self.pic_test)

        # Convert data tensor into Dataset class and shuffle & make batch.
        self.monet_train = tf.data.Dataset.from_tensor_slices(self.monet_train).\
            shuffle(self.monet_train.get_shape()[0], reshuffle_each_iteration=True)

        self.pic_train = tf.data.Dataset.from_tensor_slices(self.pic_train).\
            shuffle(self.pic_train.get_shape()[0], reshuffle_each_iteration=True)

        self.monet_test = tf.data.Dataset.from_tensor_slices(self.monet_test).\
            shuffle(self.monet_test.get_shape()[0], reshuffle_each_iteration=False)

        self.pic_test = tf.data.Dataset.from_tensor_slices(self.pic_test).\
            shuffle(self.pic_test.get_shape()[0], reshuffle_each_iteration=False)

        self.train_zipped = tf.data.Dataset.zip((self.monet_train, self.pic_train)).batch(self.BATCH_SIZE)
        self.test_zipped = tf.data.Dataset.zip((self.monet_test, self.pic_test)).batch(self.BATCH_SIZE)


    def patch_separating(self, image):
        patch_set = tf.image.extract_patches(image, [1, self.PATCH_SIZE, self.PATCH_SIZE, 1],
                                             [1, self.PATCH_SIZE, self.PATCH_SIZE, 1], [1, 1, 1, 1], padding='VALID')

        patch_set = tf.reshape(patch_set,
                               [patch_set.get_shape()[1] * patch_set.get_shape()[2] * image.get_shape()[0],
                                self.PATCH_SIZE, self.PATCH_SIZE, 3])

        return patch_set

