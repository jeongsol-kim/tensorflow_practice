import tensorflow as tf
import matplotlib.pyplot as plt

from Unet.dataloader import DataLoader
from Unet.unet import Unet
from Unet.trainer import Trainer

# Data loading
loader = DataLoader()
loader.batch_preparing(50, 10)

# Network creation
network = Unet()

# Trainer create
trainer = Trainer(network)

# Training
trainer.train(loader.train_zipped, loader.test_zipped, 1)

# make gif file.
trainer.make_train_history_gif()
