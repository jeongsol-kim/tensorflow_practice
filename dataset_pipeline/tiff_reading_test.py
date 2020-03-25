import tensorflow as tf, numpy as np, matplotlib.pyplot as plt
from PIL import Image
from dataset_pipeline.utils import *
from dataset_pipeline.imagedataloader import ImageDataLoader

# data load and pre-processing
data_dir = '/home/bispl/바탕화면/super-res-cycleGAN/data/2DHD/train_input/2DHD_Train.tif'
loader = ImageDataLoader('test_loader', 20)
loader.load_multi_page_tif_image(data_dir)

train_img = next(loader.data_iter)
print(np.shape(train_img))
