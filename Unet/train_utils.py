import tensorflow as tf
from tensorflow import keras


class PSNR(keras.metrics.Metric):
    def __init__(self, name = 'psnr', **kwargs):
        super(PSNR, self).__init__(name=name, **kwargs)
        self.psnr = self.add_weight(name='psnr', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        value = tf.image.psnr(y_true, y_pred, 1.0)
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, 'float32')
            value = tf.multiply(value, sample_weight)
        self.psnr.assign_add(tf.reduce_sum(value))

    def result(self):
        return self.psnr

    def reset_states(self):
        # The state of the metric will be reset at the start of each epoch.
        self.psnr.assign(0.)
