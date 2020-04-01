import tensorflow as tf

class DiscriminatorStructure:
    def __init__(self):
        pass

    def makeNpass_network(self, x):
        w, h, c = x.shape[1:]
        network = tf.keras.Sequential()
        network.add(tf.keras.layers.Conv2D(64, (5, 5), (2, 2), padding='same', input_shape=[w, h, c]))
        network.add(tf.keras.layers.ReLU())
        network.add(tf.keras.layers.Dropout(0.3))

        network.add(tf.keras.layers.Conv2D(128, (3, 3), (1, 1), padding='same'))
        network.add(tf.keras.layers.ReLU())
        network.add(tf.keras.layers.Dropout(0.3))

        network.add(tf.keras.layers.Conv2D(256, (3, 3), (1, 1), padding='same'))
        network.add(tf.keras.layers.ReLU())
        network.add(tf.keras.layers.Dropout(0.3))

        network.add(tf.keras.layers.Conv2D(512, (3, 3), (1, 1), padding='same'))
        network.add(tf.keras.layers.ReLU())
        network.add(tf.keras.layers.Dropout(0.3))

        network.add(tf.keras.layers.Flatten())
        network.add(tf.keras.layers.Dense(1))

        return network(x)