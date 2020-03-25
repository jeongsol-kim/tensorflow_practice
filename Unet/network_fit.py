# from tensorflow import keras
import tensorflow as tf, datetime
from Unet.data_preparation import DataPreparation
from Unet import unet_block
from Unet.data_utils import *
import Unet.train_utils as train_utils


class Network:
    def __init__(self, args, arch ='Unet'):
        self.architecture = arch
        self.model = None

        self.IMG_SIZE = args.IMG_SIZE
        self.PATCH_SIZE = args.PATCH_SIZE
        self.PATCH_STRIDE = args.PATCH_STRIDE
        self.BATCH_SIZE = args.BATCH_SIZE
        self.IFS = 64 # initial filters

        self.TRAIN_INPUT_DIR = args.TRAIN_INPUT_DIR
        self.TRAIN_LABEL_DIR = args.TRAIN_LABEL_DIR
        self.LOG_DIR = args.LOG_DIR

        self.LEARNING_RATE = args.LEARNING_RATE
        self.EPOCH_NUM = args.EPOCH_NUM

        self.DP = DataPreparation()
        self.data_load()
        self.create()
        self.current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    def create(self):
        input = tf.keras.layers.Input(shape=[self.PATCH_SIZE, self.PATCH_SIZE, 1])

        if self.architecture == 'Unet':
            down_stack = unet_block.make_down_stack(self.IFS)
            up_stack = unet_block.make_up_stack(self.IFS)

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
                               metrics=[train_utils.PSNR()])

    def data_load(self):
        print('Data set are loading...')
        # ------ DATA LOADING & PRE-PROCESS------ #
        self.DP.read_data(self.TRAIN_INPUT_DIR)
        self.DP.scale_normalization(self.DP.raw_data)
        train_data = self.DP.scaled_data
        self.DP.read_data(self.TRAIN_LABEL_DIR)
        self.DP.scale_normalization(self.DP.raw_data)
        train_label = self.DP.scaled_data

        (self.train_dataset, self.train_label), (self.test_dataset, self.test_label) = self.DP.data_separation(train_data, train_label, 450, 50)
        print('Data set loading is Done!!\n')

    def train(self):
        train_patch_set, patch_x, patch_y = self.DP.separate_into_patch_set(self.train_dataset, self.PATCH_SIZE,
                                                                            self.PATCH_STRIDE, self.BATCH_SIZE)
        train_label_patch_set, _, _ = self.DP.separate_into_patch_set(self.train_label, self.PATCH_SIZE,
                                                                      self.PATCH_STRIDE, self.BATCH_SIZE)

        log_dir = self.LOG_DIR + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        self.model.fit(train_patch_set, train_label_patch_set,
                       batch_size=self.BATCH_SIZE, epochs=self.EPOCH_NUM, callbacks=[tensorboard_callback])

    def test(self):
        test_patch_set,patch_x,patch_y = self.DP.separate_into_patch_set(self.test_dataset, self.PATCH_SIZE, self.PATCH_STRIDE, self.BATCH_SIZE)
        test_label_patch_set,_,_ = self.DP.separate_into_patch_set(self.test_label, self.PATCH_SIZE, self.PATCH_STRIDE, self.BATCH_SIZE)

        test_loss, test_acc = self.model.evaluate(test_patch_set, test_label_patch_set, verbose=2)
        print('Test loss: ', test_loss, 'Test accuracy: ', test_acc)