# from tensorflow import keras
import tensorflow as tf, datetime
from Unet.data_preparation import DataPreparation
from Unet import unet_block
from Unet.data_utils import *
import Unet.train_utils as train_utils


class Network:
    def __init__(self, args, arch = 'Unet'):
        self.architecture = arch
        self.model = None

        self.IMG_SIZE = args.IMG_SIZE
        self.PATCH_SIZE = args.PATCH_SIZE
        self.PATCH_STRIDE = args.PATCH_STRIDE
        self.BATCH_SIZE = args.BATCH_SIZE
        self.IFS = 64  # initial filters

        self.TRAIN_INPUT_DIR = args.TRAIN_INPUT_DIR
        self.TRAIN_LABEL_DIR = args.TRAIN_LABEL_DIR
        self.LOG_DIR = args.LOG_DIR

        self.LEARNING_RATE = args.LEARNING_RATE
        self.EPOCH_NUM = args.EPOCH_NUM

        self.DP = DataPreparation()
        self.data_load()
        self.create()
        self.metric()
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

    def data_load(self):
        print('Data set are loading...')
        # ------ DATA LOADING & PRE-PROCESS------ #
        self.DP.read_data(self.TRAIN_INPUT_DIR)
        self.DP.scale_normalization(self.DP.raw_data)
        train_data = self.DP.scaled_data
        self.DP.read_data(self.TRAIN_LABEL_DIR)
        self.DP.scale_normalization(self.DP.raw_data)
        train_label = self.DP.scaled_data

        (self.train_dataset, self.train_label), (self.test_dataset, self.test_label) = self.DP.data_separation(
            train_data, train_label, 450, 50)
        print('Data set loading is Done!!\n')

    def metric(self):
        self.loss_object = tf.keras.metrics.MeanSquaredError()
        self.optimizer = tf.keras.optimizers.Adam()
        self.train_loss = tf.keras.metrics.Mean('train_loss', tf.float32)
        self.train_psnr = train_utils.PSNR(name='train_psnr')
        self.test_loss = tf.keras.metrics.Mean('test_loss', tf.float32)
        self.test_psnr = train_utils.PSNR(name='test_psnr')

    def train_step(self, optimizer, train_input, train_label):
        with tf.GradientTape() as tape:
            predictions = self.model(train_input, training=True)
            loss = self.loss_object(train_label, predictions)
        grads = tape.gradient(loss, self.model.trainable_variables)
        optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        self.train_loss(loss)
        self.train_psnr.update_state(train_label, predictions)

    def test_step(self, test_input, test_label):
        predictions = self.model(test_input)
        loss = self.loss_object(test_label, predictions)

        self.test_loss(loss)
        self.test_psnr.update_state(test_label, predictions)

    def train(self):
        train_patch_set, patch_x, patch_y = self.DP.separate_into_patch_set(self.train_dataset, self.PATCH_SIZE,
                                                                            self.PATCH_STRIDE, self.BATCH_SIZE)
        train_label_patch_set, _, _ = self.DP.separate_into_patch_set(self.train_label, self.PATCH_SIZE,
                                                                      self.PATCH_STRIDE, self.BATCH_SIZE)

        train_log_dir = 'logs/gradient_tape/' + self.current_time + '/train'
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)

        for epoch in range(self.EPOCH_NUM):
            for img_num in range(450):
                self.train_step(self.optimizer, train_patch_set[img_num*self.BATCH_SIZE:(1+img_num)*self.BATCH_SIZE,:,:,:],
                                train_label_patch_set[img_num*self.BATCH_SIZE:(1+img_num)*self.BATCH_SIZE,:,:,:])

            with train_summary_writer.as_default():
                tf.summary.scalar('loss', self.train_loss.result(), step=epoch)
                tf.summary.scalar('accuracy', self.train_psnr.result(), step=epoch)

            self.valid(epoch)

            template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
            print(template.format(epoch + 1,
                                  self.train_loss.result(),
                                  self.train_psnr.result(),
                                  self.test_loss.result(),
                                  self.test_psnr.result()))
            # Reset metrics every epoch
            self.train_loss.reset_states()
            self.test_loss.reset_states()
            self.train_psnr.reset_states()
            self.test_psnr.reset_states()

    def valid(self, epoch_num):
        test_patch_set, patch_x, patch_y = self.DP.separate_into_patch_set(self.test_dataset, self.PATCH_SIZE,
                                                                           self.PATCH_STRIDE, self.BATCH_SIZE)
        test_label_patch_set, _, _ = self.DP.separate_into_patch_set(self.test_label, self.PATCH_SIZE,
                                                                     self.PATCH_STRIDE, self.BATCH_SIZE)

        test_log_dir = 'logs/gradient_tape/' + self.current_time + '/test'
        test_summary_writer = tf.summary.create_file_writer(test_log_dir)

        for img_num in range(50):
            self.test_step(test_patch_set, test_label_patch_set)

        with test_summary_writer.as_default():
            tf.summary.scalar('loss', self.test_loss.result(), step=epoch_num)
            tf.summary.scalar('accuracy', self.test_psnr.result(), step=epoch_num)
