import tensorflow as tf
import tensorflow_datasets as tdf
import numpy as np
import matplotlib.pyplot as plt
from Unet.noise_helper import NoiseHelper

class DataLoader():
    def __init__(self):
        (self.data, _), _ = tf.keras.datasets.mnist.load_data()
        self.BUFFER_SIZE = np.shape(self.data)[0]
        self.BATCH_SIZE = 256
        self.noise_helper = NoiseHelper()

    def batch_preparing(self, train_num = 0, test_num = 5000):
        self.label_data = self.data.reshape(self.data.shape[0], 28, 28, 1).astype('float32')
        self.label_data = self.label_data / 255.0
        if train_num != 0:
            self.train_label_data = self.label_data[:train_num, :, :, :]
            self.train_noise_data = self.noise_helper.make_noise(self.train_label_data, features=[0, 0.05]).astype('float32')
        else:
            self.train_label_data = self.label_data[:-test_num, :, :, :]
            self.train_noise_data = self.noise_helper.make_noise(self.train_label_data, features=[0, 0.05]).astype('float32')


        # Add different noise with training dataset.
        self.test_label_data = self.label_data[train_num:train_num+test_num, :, :, :]
        self.test_noise_data = self.noise_helper.make_noise(self.test_label_data, features=[0, 0.05]).astype('float32')

        self.train_label_data = tf.data.Dataset.from_tensor_slices(self.train_label_data).batch(self.BATCH_SIZE)
        self.train_noise_data = tf.data.Dataset.from_tensor_slices(self.train_noise_data).batch(self.BATCH_SIZE)
        self.test_label_data = tf.data.Dataset.from_tensor_slices(self.test_label_data).batch(self.BATCH_SIZE)
        self.test_noise_data = tf.data.Dataset.from_tensor_slices(self.test_noise_data).batch(self.BATCH_SIZE)

        self.train_zipped = tf.data.Dataset.zip((self.train_label_data, self.train_noise_data))
        self.test_zipped = tf.data.Dataset.zip((self.test_label_data, self.test_noise_data))

        self.train_zipped = self.train_zipped.shuffle(self.BUFFER_SIZE, reshuffle_each_iteration=True)
        self.test_zipped = self.test_zipped.shuffle(self.BUFFER_SIZE, reshuffle_each_iteration=False)

        '''
        for i, j in self.zipped:
            print(np.shape(i)==np.shape(j))
  
        plt.figure()
        plt.subplot(1,2,1)
        plt.imshow(i[0,:,:,0], cmap='gray')
        plt.subplot(1,2,2)
        plt.imshow(j[0,:,:,0],cmap='gray')
        plt.show()
        '''


