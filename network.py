#from tensorflow import keras
import tensorflow as tf, numpy as np, unet_block

class Network:
    def __init__(self, args, arch ='Unet'):
        self.architecture = arch
        self.model = None

        self.IMG_SIZE = args.IMG_SIZE
        self.PATCH_SIZE = args.PATCH_SIZE
        self.BATCH_SIZE = args.BATCH_SIZE
        self.LEARNING_RATE = args.LEARNING_RATE
        self.IFS = 64 # initial filters

        self.create()

    def create(self):
        input = tf.keras.layers.Input(shape=[self.PATCH_SIZE, self.PATCH_SIZE, 1])
        if self.architecture == 'Unet':
            down_stack = [
                unet_block.down_flow(self.IFS, 3), # 64
                unet_block.down_flow(self.IFS, 3, apply_pooling=True), # 32
                unet_block.down_flow(self.IFS*2, 3), # 32
                unet_block.down_flow(self.IFS*2, 3, apply_pooling=True), # 16
                unet_block.down_flow(self.IFS*4, 3), # 16
                unet_block.down_flow(self.IFS*4, 3, apply_pooling=True), # 8
                unet_block.down_flow(self.IFS*8, 3), # 8
                unet_block.down_flow(self.IFS*8, 3, apply_pooling=True), # 4
                unet_block.down_flow(self.IFS*16, 3), # 4
                unet_block.down_flow(self.IFS*16, 3) # 4
            ]

            up_stack = [
                unet_block.up_flow(self.IFS * 8, 2, apply_transpose=True), # 8
                unet_block.up_flow(self.IFS * 8, 2), # 8
                unet_block.up_flow(self.IFS * 4, 2, apply_transpose=True), # 16
                unet_block.up_flow(self.IFS * 4, 2), # 16
                unet_block.up_flow(self.IFS * 2, 2, apply_transpose=True), # 32
                unet_block.up_flow(self.IFS * 2, 2), # 32
                unet_block.up_flow(self.IFS * 1, 2, apply_transpose=True), # 64
                unet_block.up_flow(self.IFS * 1, 2) # 64
            ]

            last = tf.keras.layers.Conv2D(1, (3, 3), (1, 1), padding='same') # 64

            x = input
            skips = []
            for down in down_stack:
                x = down(x)
                skips.append(x)
            skips = list(reversed(skips[:-2])) # [4,4,4,'8',8,'16',16,'32',32,'64']

            # I want concatenate the layer which follows convolution layer rather than following max pooling layer.
            skips.pop(0) # ['8',8,'16',16,'32',32,'64']

            for up, skip in zip(up_stack, skips): #['8',8,'16',16,'32',32,'64',64]
                x = up(x)
                if 'conv2d_transpose' in [layer.name for layer in up.layers]:
                    x = tf.keras.layers.Concatenate()([x, skip])

            x = last(x)
            self.model = tf.keras.Model(inputs=input, outputs=x)