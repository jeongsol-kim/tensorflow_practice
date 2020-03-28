import tensorflow as tf
import numpy as np


class DataLoader():
    def __init__(self):
        (self.data,_), _ = tf.keras.datasets.mnist.load_data()
        self.BUFFER_SIZE = np.shape(self.data)[0]
        self.BATCH_SIZE = 256

    def batch_preparing(self):
        self.fashion_data = self.data.reshape(self.data.shape[0], 28, 28, 1).astype('float32')
        self.fashion_data = (self.fashion_data - 127.5) / 127.5
        self.fashion_data = tf.data.Dataset.from_tensor_slices(self.fashion_data).shuffle(self.BUFFER_SIZE).batch(self.BATCH_SIZE)

