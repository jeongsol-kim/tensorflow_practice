import tensorflow as tf
import CycleGAN.hyperparams as hp
from CycleGAN.generator_structure import GeneratorStructure
from CycleGAN.discriminator_structure import DiscriminatorStructure

class CycleGAN():
    def __init__(self):
        self.l1_loss = tf.keras.losses.MeanAbsoluteError()
        self.l2_loss = tf.keras.losses.MeanSquaredError()
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.Wc = hp.weight_cycle_consistency  # weight for cycle consistency loss
        self.Wi = hp.weight_identity  # weight for identity loss
        self.FG_optimizer = tf.keras.optimizers.Adam(hp.lr_forward_gen)
        self.BG_optimizer = tf.keras.optimizers.Adam(hp.lr_backward_gen)
        self.DX_optimizer = tf.keras.optimizers.Adam(hp.lr_x_dis)
        self.DY_optimizer = tf.keras.optimizers.Adam(hp.lr_y_dis)

        self.PATCH_SIZE = hp.PATCH_SIZE

        self.Forward_Generator = self.create_Generator()   # domain X -> Y
        self.Backward_Generator = self.create_Generator()  # domain Y -> X
        self.X_Discriminator = self.create_Discriminator()  # discriminator for domain X
        self.Y_Discriminator = self.create_Discriminator()  # discriminator for domain Y

    def __call__(self, input_X, input_Y): # for inference.
        forward_output = self.Forward_Generator(input_X, training=False)
        forward_cycle = self.Backward_Generator(forward_output, training=False)

        backward_output = self.Backward_Generator(input_Y, training=False)
        backward_cycle = self.Forward_Generator(backward_output, training=False)

        forward_fake_score = self.Y_Discriminator(forward_output, training=False)
        backward_fake_score = self.X_Discriminator(backward_output, training=False)
        x_real_score = self.X_Discriminator(input_X, training=False)
        y_real_score = self.Y_Discriminator(input_Y, training=False)
        DX_loss = self.discriminator_loss(x_real_score, backward_fake_score)
        DY_loss = self.discriminator_loss(y_real_score, forward_fake_score)


        cycle_loss = self.Cycle_loss(input_X, forward_cycle) + self.Cycle_loss(input_Y, backward_cycle)
        FG_identity_loss = self.Identity_loss(input_Y, forward_output)
        BG_identity_loss = self.Identity_loss(input_X, backward_output)
        FG_loss = self.generator_loss(forward_fake_score) + self.Wc*cycle_loss + self.Wi*FG_identity_loss
        BG_loss = self.generator_loss(backward_fake_score) + self.Wc*cycle_loss + self.Wi*BG_identity_loss


        return FG_loss, BG_loss, DX_loss, DY_loss

    def create_Generator(self):
        GStructure = GeneratorStructure()

        input = tf.keras.layers.Input(shape=(self.PATCH_SIZE, self.PATCH_SIZE, 3), dtype=tf.float32)
        x = input
        x, skip_layers = GStructure.addNpass_down_path(x)
        x = GStructure.addNpass_up_path(x, skip_layers)
        return tf.keras.Model(inputs=input, outputs=x)

    def create_Discriminator(self):
        DStructure = DiscriminatorStructure()

        input = tf.keras.layers.Input(shape=(self.PATCH_SIZE, self.PATCH_SIZE, 3), dtype=tf.float32)
        x = input
        x = DStructure.makeNpass_network(x)
        return tf.keras.Model(inputs=input, outputs=x)

    def Cycle_loss(self, x, xyx): # cycle consistency loss
        return self.l1_loss(xyx, x)

    def Identity_loss(self, y, xy):
        return tf.reduce_mean(tf.abs(y-xy))

    def generator_loss(self, fake_output_score):
        return self.cross_entropy(tf.ones_like(fake_output_score), fake_output_score)

    def discriminator_loss(self, real_output_score, fake_output_score):
        real_loss = self.cross_entropy(tf.ones_like(real_output_score), real_output_score)
        fake_loss = self.cross_entropy(tf.zeros_like(fake_output_score), fake_output_score)
        total_loss = real_loss + fake_loss
        return total_loss









