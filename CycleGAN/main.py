from CycleGAN.dataloader import DataLoader
from CycleGAN.cyclegan import CycleGAN
from CycleGAN.trainer import Trainer

# Data loading
loader = DataLoader(patch=True)
loader.batch_preparing()

# Network creation
network = CycleGAN()

# Trainer create
trainer = Trainer(network)

# Training
trainer.train(loader.train_zipped, loader.test_zipped, 20)

# make gif file.
trainer.make_train_history_gif()
