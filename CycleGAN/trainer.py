import tensorflow as tf
import CycleGAN.hyperparams as hp
import matplotlib.pyplot as plt
import time
import os
import imageio
import glob

class Trainer():
    def __init__(self, network):
        self.BATCH_SIZE = hp.BATCH_SIZE
        self.Wc = hp.weight_cycle_consistency # weight for cycle consistency loss
        self.Wi = hp.weight_identity          # weight for identity loss
        self.network = network

        self.checkpoint_dir = './training_checkpoints'
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")
        self.checkpoint = tf.train.Checkpoint(FGoptimizer=self.network.FG_optimizer,
                                              BGoptimizer=self.network.BG_optimizer,
                                              DXoptimizer=self.network.DX_optimizer,
                                              DYoptimizer=self.network.DY_optimizer,
                                              ForwardGenerator=self.network.Forward_Generator,
                                              BackwardGenerator=self.network.Backward_Generator,
                                              XDiscriminator=self.network.X_Discriminator,
                                              YDiscriminator=self.network.Y_Discriminator)

        self.summary_log_dir = './training_log/train1'
        self.train_FGloss_writer = tf.summary.create_file_writer(self.summary_log_dir + '/train/ForwardGen')
        self.train_BGloss_writer = tf.summary.create_file_writer(self.summary_log_dir + '/train/BackwardGen')
        self.train_DXloss_writer = tf.summary.create_file_writer(self.summary_log_dir + '/train/BackwardDis')
        self.train_DYloss_writer = tf.summary.create_file_writer(self.summary_log_dir + '/train/ForwardDis')
        self.test_FGloss_writer = tf.summary.create_file_writer(self.summary_log_dir + '/test/ForwardGen')
        self.test_BGloss_writer = tf.summary.create_file_writer(self.summary_log_dir + '/test/BackwardGen')
        self.test_DXloss_writer = tf.summary.create_file_writer(self.summary_log_dir + '/test/BackwardDis')
        self.test_DYloss_writer = tf.summary.create_file_writer(self.summary_log_dir + '/test/ForwardDis')

    def PSNR(self, output, label):
        return tf.reduce_mean(tf.image.psnr(output, label, 1.0))

    def list_avg(self, L):
        return sum(L)/len(L)

    @tf.function
    def train_step(self, img_X, img_Y):
        with tf.GradientTape(persistent = True) as grad_tape:
            output_XY = self.network.Forward_Generator(img_X, training=True)
            output_YX = self.network.Backward_Generator(img_Y, training=True)
            output_XYX = self.network.Backward_Generator(output_XY, training=True)
            output_YXY = self.network.Forward_Generator(output_YX, training=True)

            dis_score_X = self.network.X_Discriminator(img_X, training=True)
            dis_score_Y = self.network.Y_Discriminator(img_Y, training=True)
            dis_score_XY = self.network.Y_Discriminator(output_XY, training=True)
            dis_score_YX = self.network.X_Discriminator(output_YX, training=True)

            # Calculate the loss value.
            # 1. cycle consistency loss
            cycle_con_loss = self.network.Cycle_loss(img_X, output_XYX) + self.network.Cycle_loss(img_Y, output_YXY)

            # Add. Identity loss
            FG_identity_loss = self.network.Identity_loss(img_Y, output_XY)
            BG_identity_loss = self.network.Identity_loss(img_X, output_YX)

            # 2. Generator loss ( adversarial_loss + consistency_loss )
            FG_loss = self.network.generator_loss(dis_score_XY) + self.Wc * cycle_con_loss + self.Wi * FG_identity_loss
            BG_loss = self.network.generator_loss(dis_score_YX) + self.Wc * cycle_con_loss + self.Wi * BG_identity_loss

            # 3. Discriminator loss
            DX_loss = self.network.discriminator_loss(dis_score_X, dis_score_YX)
            DY_loss = self.network.discriminator_loss(dis_score_Y, dis_score_XY)

        FG_gradient = grad_tape.gradient(FG_loss, self.network.Forward_Generator.trainable_variables)
        BG_gradient = grad_tape.gradient(BG_loss, self.network.Backward_Generator.trainable_variables)
        DX_gradient = grad_tape.gradient(DX_loss, self.network.X_Discriminator.trainable_variables)
        DY_gradient = grad_tape.gradient(DY_loss, self.network.Y_Discriminator.trainable_variables)

        self.network.FG_optimizer.apply_gradients(zip(FG_gradient, self.network.Forward_Generator.trainable_variables))
        self.network.BG_optimizer.apply_gradients(zip(BG_gradient, self.network.Backward_Generator.trainable_variables))
        self.network.DX_optimizer.apply_gradients(zip(DX_gradient, self.network.X_Discriminator.trainable_variables))
        self.network.DY_optimizer.apply_gradients(zip(DY_gradient, self.network.Y_Discriminator.trainable_variables))

        return FG_loss, BG_loss, DX_loss, DY_loss

    def train(self, train_zipped, test_zipped, epochs):
        # prepare arbitrary dataset for saving image.
        imgA_for_history, imgB_for_history = [x for x in test_zipped][0]
        imgA_for_history = imgA_for_history[0:4, :, :, :]
        imgB_for_history = imgB_for_history[0:4, :, :, :]

        for epoch in range(epochs):
            start_time = time.time()
            avg_FG_loss = []
            avg_BG_loss = []
            avg_DX_loss = []
            avg_DY_loss = []
            for image_X, image_Y in train_zipped:
                FG_loss, BG_loss, DX_loss, DY_loss = self.train_step(image_X, image_Y)
                avg_FG_loss.append(FG_loss)
                avg_BG_loss.append(BG_loss)
                avg_DX_loss.append(DX_loss)
                avg_DY_loss.append(DY_loss)

            # print training information.
            print('Time for epoch {} is {} min'.format(epoch + 1, (time.time() - start_time)/60.0))
            frame = 'Forward Generator Loss: {} / Backward Generator Loss: {}\nDiscriminator X Loss: {} / Discriminator Y Loss: {}'
            print(frame.format(self.list_avg(avg_FG_loss), self.list_avg(avg_BG_loss),
                               self.list_avg(avg_DX_loss), self.list_avg(avg_DY_loss)))

            # write on tensorboard.
            self.make_summaries(self.list_avg(avg_FG_loss), self.list_avg(avg_BG_loss),
                               self.list_avg(avg_DX_loss), self.list_avg(avg_DY_loss), epoch)

            # save the model as checkpoint.
            if (epoch + 1) % 15 == 0:
                self.checkpoint.save(file_prefix=self.checkpoint_prefix)

            # Testing model
            self.test(test_zipped, epoch)

            self.generate_and_save_images(self.network.Forward_Generator, self.network.Backward_Generator,
                                          epoch, imgA_for_history, imgB_for_history)

        self.make_summaries(self.list_avg(avg_FG_loss),self.list_avg(avg_BG_loss),
                               self.list_avg(avg_DX_loss),self.list_avg(avg_DY_loss), epochs)
        self.generate_and_save_images(self.network.Forward_Generator, self.network.Backward_Generator,
                                      epoch, imgA_for_history, imgB_for_history)

    def test(self, test_dataset, epoch):
        start_time = time.time()
        avg_FG_loss = []
        avg_BG_loss = []
        avg_DX_loss = []
        avg_DY_loss = []
        for test_X, test_Y in test_dataset:
            FG_loss, BG_loss, DX_loss, DY_loss = self.network(test_X, test_Y) # possible because of __call__ function.
            avg_FG_loss.append(FG_loss)
            avg_BG_loss.append(BG_loss)
            avg_DX_loss.append(DX_loss)
            avg_DY_loss.append(DY_loss)

        print('Time for test {} is {} sec'.format(epoch + 1, time.time() - start_time))
        frame = 'Forward Generator Loss: {} / Backward Generator Loss: {}\nDiscriminator X Loss: {} / Discriminator Y Loss: {}'
        print(frame.format(self.list_avg(avg_FG_loss), self.list_avg(avg_BG_loss),
                           self.list_avg(avg_DX_loss), self.list_avg(avg_DY_loss)))
        print('')

        self.make_summaries(self.list_avg(avg_FG_loss), self.list_avg(avg_BG_loss),
                            self.list_avg(avg_DX_loss), self.list_avg(avg_DY_loss), epoch, training=False)

    def make_summaries(self, FG_loss, BG_loss, DX_loss, DY_loss, epoch, training=True):
        if training:
            with self.train_FGloss_writer.as_default():
                tf.summary.scalar('Forward_losses', FG_loss, step=epoch)
            with self.train_BGloss_writer.as_default():
                tf.summary.scalar('Backward_losses', BG_loss, step=epoch)
            with self.train_DXloss_writer.as_default():
                tf.summary.scalar('Backward_losses', DX_loss, step=epoch)
            with self.train_DYloss_writer.as_default():
                tf.summary.scalar('Forward_losses', DY_loss, step=epoch)
        else:
            with self.test_FGloss_writer.as_default():
                tf.summary.scalar('Forward_losses', FG_loss, step=epoch)
            with self.test_BGloss_writer.as_default():
                tf.summary.scalar('Backward_losses', BG_loss, step=epoch)
            with self.test_DXloss_writer.as_default():
                tf.summary.scalar('Backward_losses', DX_loss, step=epoch)
            with self.test_DYloss_writer.as_default():
                tf.summary.scalar('Forward_losses', DY_loss, step=epoch)

    def generate_and_save_images(self, FG, BG, epoch, test_X, test_Y):
        fake_Y = FG(test_X, training=False)
        fake_X = BG(test_Y, training=False)

        plt.figure()
        plt.subplot(4,4,1)
        plt.title('Real Monet (X)')
        plt.subplot(4,4,2)
        plt.title('Fake Picture (X->Y)')
        plt.subplot(4,4,3)
        plt.title('Real Picture (Y)')
        plt.subplot(4,4,4)
        plt.title('Fake Monet (Y->X)')

        for i in range(4):
            plt.subplot(4, 4, i*4 + 1)
            plt.imshow(test_X[i, :, :, :])
            plt.axis('off')
            plt.subplot(4, 4, i*4 + 2)
            plt.imshow(fake_Y[i, :, :, :])
            plt.axis('off')
            plt.subplot(4, 4, i*4 + 3)
            plt.imshow(test_Y[i, :, :, :])
            plt.axis('off')
            plt.subplot(4, 4, i*4 + 4)
            plt.imshow(fake_X[i, :, :, :])
            plt.axis('off')

        plt.tight_layout()
        plt.savefig('./training_history/image_at_epoch_{:04d}.png'.format(epoch))
        plt.close()

    def make_train_history_gif(self):
        anim_file = './training_history/CycleGAN_transfer.gif'

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