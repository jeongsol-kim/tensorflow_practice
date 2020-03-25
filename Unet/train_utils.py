import tensorflow as tf
from tensorflow import keras
from Unet.hyperparameters import *


class PSNR(keras.metrics.Metric):
    def __init__(self, name = 'PSNR', **kwargs):
        super(PSNR, self).__init__(name=name, **kwargs)
        self.psnr = self.add_weight(name='psnr', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        # here, change patch set into images.
        #y_true_one = data_utils.patch2img(np.array(y_true), PATCH_STRIDE)
        #y_pred_one = data_utils.patch2img(np.array(y_pred), PATCH_STRIDE)

        value = tf.image.psnr(y_true, y_pred, 1.0)
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, 'float32')
            value = tf.multiply(value, sample_weight)
        self.psnr.assign(tf.reduce_sum(value)/BATCH_SIZE)

    def result(self):
        return self.psnr

    def reset_states(self):
        # The state of the metric will be reset at the start of each epoch.
        self.psnr.assign(0.)

'''
 # change patch to image here?
            for recon_num in range(self.BATCH_SIZE):
                if recon_num == 0:
                    output = 

                    output = tf.expand_dims(output, 0)
                else:
                    temp_img = patch2img(x[recon_num*patch_set.get_shape()[1]:(recon_num+1)*patch_set.get_shape()[1],
                                         :, :, :], self.PATCH_STRIDE)

                    output = tf.concat([output, temp_img],0)
'''