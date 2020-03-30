import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import imageio
import glob

class Trainer():
    def __init__(self, network):
        self.BATCH_SIZE = 256
        self.network = network

        self.checkpoint_dir = './training_checkpoints'
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")
        self.checkpoint = tf.train.Checkpoint(optimizer=self.network.optimizer, Unet=self.network.model)

        self.summary_log_dir = './training_log/train4'
        self.train_loss_writer = tf.summary.create_file_writer(self.summary_log_dir + '/train/loss')
        self.test_loss_writer = tf.summary.create_file_writer(self.summary_log_dir + '/test/loss')
        self.train_psnr_writer = tf.summary.create_file_writer(self.summary_log_dir + '/train/psnr')
        self.tets_psnr_writer = tf.summary.create_file_writer(self.summary_log_dir + '/test/psnr')


    def PSNR(self, output, label):
        return tf.reduce_mean(tf.image.psnr(output, label, 1.0))

    @tf.function
    def train_step(self, batch_image, label_image):
        with tf.GradientTape() as grad_tape:
            output_image = self.network.model(batch_image, training=True)

            #
            loss = self.network.l2_loss(output_image, label_image)
            psnr = self.PSNR(output_image, label_image)

        gradients = grad_tape.gradient(loss, self.network.model.trainable_variables)
        self.network.optimizer.apply_gradients(zip(gradients, self.network.model.trainable_variables))

        return loss, psnr

    def train(self, train_zipped, test_zipped, epochs):
        # prepare arbitrary dataset for saving image.
        label_for_history, data_for_history = [x for x in test_zipped][0]
        data_for_history = data_for_history[0:4, :, :, :]
        label_for_history = label_for_history[0:4, :, :, :]

        for epoch in range(epochs):
            start_time = time.time()
            avg_loss = []
            avg_psnr = []
            for image_batch, label_batch in train_zipped:
                loss, psnr = self.train_step(image_batch, label_batch)
                avg_loss.append(loss)
                avg_psnr.append(psnr)

            # print training information.
            print('Time for epoch {} is {} min'.format(epoch + 1, (time.time() - start_time)/60.0))
            print('Loss : {} / PSNR : {}'.format(sum(avg_loss)/len(avg_loss), sum(avg_psnr)/len(avg_psnr)))

            # write on tensorboard.
            self.make_summaries(sum(avg_loss) / len(avg_loss), sum(avg_psnr) / len(avg_psnr), epoch)

            # save the model as checkpoint.
            if (epoch + 1) % 15 == 0:
                self.checkpoint.save(file_prefix=self.checkpoint_prefix)

            # Testing model
            self.test(test_zipped, epoch)

            self.generate_and_save_images(self.network.model, epoch, data_for_history, label_for_history)

        self.make_summaries(sum(avg_loss) / len(avg_loss), sum(avg_psnr) / len(avg_psnr), epochs)
        self.generate_and_save_images(self.network.model, epoch, data_for_history, label_for_history)

    def test(self, test_dataset, epoch):
        start_time = time.time()
        avg_loss = []
        avg_psnr = []
        for test_input, test_label in test_dataset:
            output = self.network(test_input) # possible because of __call__ function.
            avg_loss.append(self.network.l2_loss(output, test_label))
            avg_psnr.append(self.PSNR(output, test_label))

        print('Time for test {} is {} sec'.format(epoch + 1, time.time() - start_time))
        print('Loss : {} / PSNR : {}'.format(sum(avg_loss) / len(avg_loss), sum(avg_psnr) / len(avg_psnr)))
        print('')

        self.make_summaries(sum(avg_loss) / len(avg_loss), sum(avg_psnr) / len(avg_psnr), epoch, training=False)

    def make_summaries(self, loss, psnr, epoch, training=True):
        if training:
            with self.train_loss_writer.as_default():
                tf.summary.scalar('MSE Loss', loss, step=epoch)
            with self.train_psnr_writer.as_default():
                tf.summary.scalar('PSNR (dB)', psnr, step=epoch)
        else:
            with self.test_loss_writer.as_default():
                tf.summary.scalar('MSE Loss', loss, step=epoch)
            with self.tets_psnr_writer.as_default():
                tf.summary.scalar('PSNR (dB)', psnr, step=epoch)

    def generate_and_save_images(self, model, epoch, test_input, test_label):
        predictions = model(test_input, training=False)

        plt.figure()
        plt.subplot(4,4,1)
        plt.title('Noisy input')
        plt.subplot(4,4,2)
        plt.title('Network output')
        plt.subplot(4,4,3)
        plt.title('Filtered noise')
        plt.subplot(4,4,4)
        plt.title('Clear label')

        for i in range(predictions.shape[0]):
            plt.subplot(4, 4, i*4 + 1)
            plt.imshow(test_input[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
            plt.axis('off')
            plt.subplot(4, 4, i*4 + 2)
            plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
            plt.axis('off')
            plt.subplot(4, 4, i*4 + 3)
            plt.imshow(test_input[i, :, :, 0] * 127.5 + 127.5 - predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
            plt.axis('off')
            plt.subplot(4, 4, i*4 + 4)
            plt.imshow(test_label[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
            plt.axis('off')

        plt.tight_layout()
        plt.savefig('./training_history/image_at_epoch_{:04d}.png'.format(epoch))
        plt.close()

    def make_train_history_gif(self):
        anim_file = './training_history/Unet_denoising.gif'

        with imageio.get_writer(anim_file, mode='I') as writer:
            filenames = glob.glob('./training_history/image*.png')
            filenames = sorted(filenames)

            last = -1
            for i, filename in enumerate(filenames):
                frame = 2 * (i ** 0.5)
                if round(frame) > round(last):
                    last = frame
                else:
                    continue
                image = imageio.imread(filename)
                writer.append_data(image)
            image = imageio.imread(filename)
            writer.append_data(image)