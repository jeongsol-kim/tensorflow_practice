import tensorflow as tf, numpy as np, matplotlib.pyplot as plt
from PIL import Image
from dataset_pipeline.utils import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class ImageDataLoader():
    def __init__(self, name = 'ImageLoader', batch_size = 1):
        self.data_gen = ImageDataGenerator()
        self.data_iter = None
        self.img_dir = ''
        self.name = name
        self.batch_size = batch_size

    def load_multi_page_tif_image(self, img_dir):
        raw_img = read_tif_image(img_dir)
        if not raw_img.any():
            print(self.name, ' - Something is wrong when loading tiff images.')
            return False
        self.img_dir = img_dir
        self.data_iter = self.data_gen.flow(raw_img, batch_size=self.batch_size)

    # only use when train, valid, test sets are separably included in subdirectories.
    # Here, img_dir should be a list of subdirectories of train, valid and test set.
    def load_image_automatically(self, img_dir, imsize_x, imsize_y, rescale_factor=1, color_mode='rgb'):
        if len(img_dir) != 3:
            print('Please give three directories - data set for training, validation and test respectively.')
            return False
        self.data_gen = [ImageDataGenerator(rescale=rescale_factor),
                         ImageDataGenerator(rescale=rescale_factor),
                         ImageDataGenerator(rescale=rescale_factor)]

        self.data_iter = []
        for i in range(3):
            dir_iter = self.data_gen[i].flow_from_directory(img_dir[i], [imsize_x, imsize_y],
                                                              color_mode=color_mode, batch_size=self.batch_size,
                                                            class_mode='input')
            self.data_iter.append(dir_iter)