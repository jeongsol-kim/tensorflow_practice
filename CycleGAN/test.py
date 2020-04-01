import tensorflow as tf
import tensorflow_datasets as tfd
import matplotlib.pyplot as plt
import numpy as np

x = tf.keras.Input(shape=(28,28,1))
w,h,c = x.shape[1:]
print(w,h,c)