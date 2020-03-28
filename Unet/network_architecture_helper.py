import tensorflow as tf

def addNpass_conv_block(x, output_ch, BN=True, Relu=True):
    block = tf.keras.Sequential()
    block.add(tf.keras.layers.Conv2D(output_ch, (5, 5), (1, 1), padding='same', use_bias=False))
    if BN: block.add(tf.keras.layers.BatchNormalization())
    if Relu: block.add(tf.keras.layers.ReLU())
    return block(x)

def addNpass_maxpooling(x, ratio=2):
    block = tf.keras.Sequential()
    block.add(tf.keras.layers.MaxPool2D((ratio, ratio)))
    return block(x)

def addNpass_transpose_conv2d(x, output_ch, ratio=2):
    block = tf.keras.Sequential()
    block.add(tf.keras.layers.Conv2DTranspose(output_ch, (ratio, ratio), (ratio, ratio)))
    return block(x)
