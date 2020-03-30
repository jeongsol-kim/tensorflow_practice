import tensorflow as tf
import matplotlib.pyplot as plt

from CycleGAN.dataloader import DataLoader
from CycleGAN.cyclegan import CycleGAN
from Unet.trainer import Trainer

# Data loading
loader = DataLoader()
loader.batch_preparing(50, 10)


'''
# Network creation
network = CycleGAN()

# Trainer create
trainer = Trainer(network)

# Training
trainer.train(loader.train_zipped, loader.test_zipped, 1)

# make gif file.
trainer.make_train_history_gif()
'''