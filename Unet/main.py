from Unet.network_fit import Network as Network_Fit
from Unet.network_gradienttape import Network as Network_Grad
from Unet.hyperparameters import *
import numpy as np, argparse, tensorflow as tf, Unet.data_utils as data_utils


# ------ HYPER PARAMETERS ------ #
data_dir = '/home/bispl/바탕화면/super-res-cycleGAN/data/2DHD/'
parser = argparse.ArgumentParser(description='')
parser.add_argument('--TRAIN_INPUT_DIR', dest='TRAIN_INPUT_DIR', default=data_dir + '2DHD_Train.tif')
parser.add_argument('--TRAIN_LABEL_DIR', dest='TRAIN_LABEL_DIR', default=data_dir + '2DHD_Train_Label.tif')
# parser.add_argument('--TEST_INPUT_DIR', dest='TEST_INPUT_DIR', default=data_dir + '2DHD_Train_Label.tif')
parser.add_argument('--CKP_DIR', dest='CKP_DIR', default='./Unet/model/')
parser.add_argument('--LOG_DIR', dest='LOG_DIR', default='./Unet/log/fit/')

parser.add_argument('--IMG_SIZE', dest='IMG_SIZE', type=int, default=IMG_SIZE)
parser.add_argument('--PATCH_SIZE', dest='PATCH_SIZE', type=int, default=PATCH_SIZE)
parser.add_argument('--PATCH_STRIDE', dest='PATCH_STRIDE',type=int, default=PATCH_STRIDE)
parser.add_argument('--BATCH_SIZE', dest='BATCH_SIZE', type=int, default=BATCH_SIZE)

parser.add_argument('--EPOCH_NUM',dest='EPOCH_NUM', type=int, default=5)
parser.add_argument('--LEARNING_RATE', dest='LEARNING_RATE', type=float, default=0.002)

args, unknown = parser.parse_known_args()

Unet = Network_Fit(args)
Unet.model.summary()

Unet.train()
Unet.test()

#result = Unet.model(np.expand_dims(train_dataset, 3), training=False)
#print(np.shape(result))