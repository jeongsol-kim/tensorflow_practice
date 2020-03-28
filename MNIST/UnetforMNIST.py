import tensorflow as tf, numpy as np, datetime
from MNIST.unet_block import *
from MNIST.train_utils import *
from MNIST.utils import *

class Unet:
    def __init__(self):
        self.IMG_SIZE = 28
        self.BATCH_SIZE = 100
        self.EPOCH_NUM = 20
        self.IFS = 64
        self.LEARNING_RATE = 0.0002
        self.LOG_DIR = 'log/fit/'

        self.create()


    def data_prepare(self):
        # dataset loading
        mnist = tf.keras.datasets.mnist
        (x_train, _), (x_test, _) = mnist.load_data()
        self.x_train, self.x_test = x_train / 255.0, x_test / 255.0
        self.x_train, self.x_valid = self.x_train[:55000,:,:], self.x_train[55000:,:,:]

        self.noise_set_generate()

    def noise_set_generate(self):
        self.x_train_noise = add_salt_pepper_noise(self.x_train)
        self.x_valid_noise = add_salt_pepper_noise(self.x_valid)
        self.x_test_noise = add_salt_pepper_noise(self.x_test)

    def create(self):
        input = tf.keras.layers.Input(shape=[self.IMG_SIZE, self.IMG_SIZE, 1])

        down_stack = make_down_stack(self.IFS)
        up_stack = make_up_stack(self.IFS)

        last = tf.keras.layers.Conv2D(1, (3, 3), (1, 1), padding='same')

        x = input
        skips = []
        for down in down_stack:
            x = down(x)
            skips.append(x)
        skips = list(reversed(skips[:-2]))

        # I want concatenate the layer which follows convolution layer rather than following max pooling layer.
        skips.pop(0)

        for up, skip in zip(up_stack, skips):
            x = up(x)
            if [layer.name for layer in up.layers if layer.name.startswith('conv2d_transpose')]:
                x = tf.keras.layers.Concatenate()([x, skip])

        x = last(x)
        tf.summary.image('Output patch', x)

        self.model = tf.keras.Model(inputs=input, outputs=x)
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.LEARNING_RATE),
                           loss=tf.keras.losses.MeanSquaredError(),
                           metrics=[PSNR()])

    def train(self):
        log_dir = self.LOG_DIR + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        self.model.fit(np.expand_dims(self.x_train_noise, -1), np.expand_dims(self.x_train, -1),
                       batch_size=self.BATCH_SIZE, epochs=self.EPOCH_NUM, callbacks=[tensorboard_callback],
                       steps_per_epoch=10)