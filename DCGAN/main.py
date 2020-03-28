import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import PIL
from DCGAN.dataloader import DataLoader
from DCGAN.dcgan import DCGAN
from DCGAN.trainer import Trainer

# Data loading
loader = DataLoader()
loader.batch_preparing()

# Model create
network = DCGAN()
network.make_generator_model()
network.make_discriminator_model()

# Trainer create
trainer = Trainer(network)

# Training
trainer.train(loader.fashion_data, 50) # training for 50 epochs.

# Make history gif file
# trainer.make_train_history_gif()
