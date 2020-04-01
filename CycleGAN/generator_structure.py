from CycleGAN.generator_layers import *

class GeneratorStructure:
    def __init__(self):
        self.IFS = 30  # initial filter size

    def addNpass_down_path(self, x):
        skip_layer = []
        w, h, c = x.shape[1:]

        x = addNpass_conv_block(x, self.IFS)
        x = addNpass_conv_block(x, self.IFS)
        skip_layer.append(x)

        x = addNpass_maxpooling(x)
        assert tuple(x.shape) == (None, int(w / 2), int(h / 2), self.IFS)

        x = addNpass_conv_block(x, self.IFS * 2)
        x = addNpass_conv_block(x, self.IFS * 2)
        skip_layer.append(x)

        x = addNpass_maxpooling(x)
        assert tuple(x.shape) == (None, int(w / 4), int(h / 4), self.IFS * 2)

        x = addNpass_conv_block(x, self.IFS * 4)
        x = addNpass_conv_block(x, self.IFS * 4)
        skip_layer.append(x)

        x = addNpass_maxpooling(x)
        assert tuple(x.shape) == (None, int(w / 8), int(h / 8), self.IFS * 4)

        x = addNpass_conv_block(x, self.IFS * 8)
        x = addNpass_conv_block(x, self.IFS * 8)
        skip_layer.append(x)

        x = addNpass_maxpooling(x)
        assert tuple(x.shape) == (None, int(w / 16), int(h / 16), self.IFS * 8)

        x = addNpass_conv_block(x, self.IFS * 16)
        x = addNpass_conv_block(x, self.IFS * 16)

        # if we input 128x128 size data, output size will be 8x8.
        # if we set self.IFS as 64, output channel will be 1024.
        assert tuple(x.shape) == (None, int(w / 16), int(h / 16), self.IFS * 16)

        return x, skip_layer

    def addNpass_up_path(self, x, skip_layers):
        w, h, c = x.shape[1:]

        x = addNpass_transpose_conv2d(x, self.IFS * 8, 2)
        assert tuple(x.shape) == (None, int(w * 2), int(h * 2), self.IFS * 8)

        x = tf.keras.layers.Concatenate()([x, skip_layers[-1]])
        assert tuple(x.shape) == (None, int(w * 2), int(h * 2), self.IFS * 16)

        x = addNpass_conv_block(x, self.IFS * 8)
        x = addNpass_conv_block(x, self.IFS * 8)
        assert tuple(x.shape) == (None, int(w * 2), int(h * 2), self.IFS * 8)

        x = addNpass_transpose_conv2d(x, self.IFS * 4, 2)
        assert tuple(x.shape) == (None, int(w * 4), int(h * 4), self.IFS * 4)

        x = tf.keras.layers.Concatenate()([x, skip_layers[-2]])
        assert tuple(x.shape) == (None, int(w * 4), int(h * 4), self.IFS * 8)

        x = addNpass_conv_block(x, self.IFS * 4)
        x = addNpass_conv_block(x, self.IFS * 4)

        x = addNpass_transpose_conv2d(x, self.IFS * 2, 2)
        assert tuple(x.shape) == (None, int(w * 8), int(h * 8), self.IFS * 2)

        x = tf.keras.layers.Concatenate()([x, skip_layers[-3]])
        assert tuple(x.shape) == (None, int(w * 8), int(h * 8), self.IFS * 4)

        x = addNpass_conv_block(x, self.IFS * 2)
        x = addNpass_conv_block(x, self.IFS * 2)

        x = addNpass_transpose_conv2d(x, self.IFS, 2)
        assert tuple(x.shape) == (None, int(w * 16), int(h * 16), self.IFS)

        x = tf.keras.layers.Concatenate()([x, skip_layers[-4]])
        assert tuple(x.shape) == (None, int(w * 16), int(h * 16), self.IFS * 2)

        x = addNpass_conv_block(x, self.IFS)
        x = addNpass_conv_block(x, 3, BN=False)

        assert tuple(x.shape) == (None, int(w * 16), int(h * 16), 3)

        return x