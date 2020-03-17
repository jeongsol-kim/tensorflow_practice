from tensorflow import keras
from network import Network
from data_preparation import DataPreparation
import tensorflow as tf, numpy as np, matplotlib.pyplot as plt, argparse

# ------ HYPER PARAMETERS ------ #
data_dir = '/home/bispl/바탕화면/super-res-cycleGAN/data/2DHD/'
parser = argparse.ArgumentParser(description='')
parser.add_argument('--TRAIN_INPUT_DIR', dest='TRAIN_INPUT_DIR', default=data_dir + '2DHD_Train.tif')
parser.add_argument('--TRAIN_LABEL_DIR', dest='TRAIN_LABEL_DIR', default=data_dir + '2DHD_Train_Label.tif')
parser.add_argument('--TEST_INPUT_DIR', dest='TEST_INPUT_DIR', default=data_dir + '2DHD_Train_Label.tif')
parser.add_argument('--CKP_DIR', dest='CKP_DIR', default='../model/')
parser.add_argument('--LOG_DIR', dest='LOG_DIR', default='../log/fit/')
parser.add_argument('--IMG_SIZE', dest='IMG_SIZE', type=int, default=512)
parser.add_argument('--PATCH_SIZE', dest='PATCH_SIZE', type=int, default=64)
parser.add_argument('--BATCH_SIZE', dest='BATCH_SIZE', type=int, default=50)
parser.add_argument('--LEARNING_RATE', dest='LEARNING_RATE', type=float, default=0.0004)

args, unknown = parser.parse_known_args()

# ------ DATA LOADING & PRE-PROCESS------ #
DP = DataPreparation()
DP.read_data(args.TRAIN_INPUT_DIR)
DP.scale_normalization(DP.raw_data)

(train_dataset, train_label),(test_dataset, test_label) = DP.data_separation(DP.scaled_data, 450, 50)

Unet = Network(args)
result = Unet.model(np.expand_dims(train_dataset, 3), training=False)
print(np.shape(result))