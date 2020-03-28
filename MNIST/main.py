import tensorflow as tf, numpy as np, matplotlib.pyplot as plt
from tensorflow import keras
from MNIST.UnetforMNIST import Unet
from MNIST.utils import *


network = Unet()
network.model.summary()
network.data_prepare()


network.train()