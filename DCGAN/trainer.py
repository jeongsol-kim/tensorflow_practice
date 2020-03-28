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
        self.NOISE_DIM = 100
        self.NUM_EX_TO_GEN = 16
        self.network = network
        self.seed = tf.random.normal([self.NUM_EX_TO_GEN, self.NOISE_DIM])

        self.checkpoint_dir = './training_checkpoints'
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")
        self.checkpoint = tf.train.Checkpoint(generator_optimizer=self.network.generator_optimizer,
                                         discriminator_optimizer=self.network.discriminator_optimizer,
                                         generator=self.network.g_model,
                                         discriminator=self.network.d_model)

        self.summary_log_dir = './training_log/train1'
        self.train_gloss_writer = tf.summary.create_file_writer(self.summary_log_dir+'/g_loss')
        self.train_dloss_writer = tf.summary.create_file_writer(self.summary_log_dir+'/d_loss')
        self.train_fscore_writer = tf.summary.create_file_writer(self.summary_log_dir+'/fake_score')
        self.train_rscore_writer = tf.summary.create_file_writer(self.summary_log_dir + '/real_score')


    @tf.function
    def train_step(self, images):
        noise = tf.random.normal([self.BATCH_SIZE, self.NOISE_DIM])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as dis_tape:
            generated_image = self.network.g_model(noise, training=True)

            real_output = self.network.d_model(images)
            fake_output = self.network.d_model(generated_image)

            gen_loss = self.network.generator_loss(fake_output)
            dis_loss = self.network.discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, self.network.g_model.trainable_variables)
        gradients_of_discriminator = dis_tape.gradient(dis_loss, self.network.d_model.trainable_variables)

        self.network.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.network.g_model.trainable_variables))
        self.network.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.network.d_model.trainable_variables))

        return gen_loss, dis_loss

    def train(self, dataset, epochs):
        for epoch in range(epochs):
            start_time = time.time()

            avg_gen_loss = []
            avg_dis_loss = []

            image_for_scoring = None
            for image_batch in dataset:
                gen_loss, dis_loss = self.train_step(image_batch)
                avg_gen_loss.append(gen_loss)
                avg_dis_loss.append(dis_loss)
                image_for_scoring = image_batch

            real_score, fake_score = self.report_discriminator_score(image_for_scoring, self.network.g_model(self.seed, training=False))
            self.generate_and_save_images(self.network.g_model, epoch, self.seed)

            # print training information.
            print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start_time))
            print('Generator Loss: {} / Discriminator Loss: {}'.format(sum(avg_gen_loss)/len(avg_gen_loss),
                                                                       sum(avg_dis_loss)/len(avg_dis_loss)))
            print('Discriminator score for Real: {} / Fake: {}'.format(np.mean(real_score), np.mean(fake_score)))

            # write on tensorboard.
            self.make_summaries(sum(avg_gen_loss)/len(avg_gen_loss), sum(avg_dis_loss)/len(avg_dis_loss),
                                np.mean(real_score), np.mean(fake_score), epoch)

            # save the model as checkpoint.
            if (epoch + 1) % 15 == 0:
                self.checkpoint.save(file_prefix=self.checkpoint_prefix)

        self.generate_and_save_images(self.network.g_model, epochs, self.seed)
        self.make_summaries(sum(avg_gen_loss) / len(avg_gen_loss), sum(avg_dis_loss) / len(avg_dis_loss),
                            np.mean(real_score), np.mean(fake_score), epochs)

    def make_summaries(self, gloss, dloss, rscore, fscore, epoch):
        with self.train_gloss_writer.as_default():
            tf.summary.scalar('Gen/Dis Loss', gloss, step=epoch)
        with self.train_dloss_writer.as_default():
            tf.summary.scalar('Gen/Dis Loss', dloss, step=epoch)
        with self.train_rscore_writer.as_default():
            tf.summary.scalar('Discriminator Score', rscore, step=epoch)
        with self.train_fscore_writer.as_default():
            tf.summary.scalar('Discriminator Score', fscore, step=epoch)

    def report_discriminator_score(self, real_image, fake_image):
        dis_real_score = self.network.d_model(real_image, training=False)
        dis_fake_score = self.network.d_model(fake_image, training=False)
        return dis_real_score, dis_fake_score

    def generate_and_save_images(self, model, epoch, test_input):
        predictions = model(test_input, training=False)

        fig = plt.figure(figsize=(4, 4))

        for i in range(predictions.shape[0]):
            plt.subplot(4, 4, i + 1)
            plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
            plt.axis('off')

        plt.savefig('./training_history/image_at_epoch_{:04d}.png'.format(epoch))
        plt.close()

    def make_train_history_gif(self):
        anim_file = 'dcgan.gif'

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

