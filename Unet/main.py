from Unet.network import Network
from Unet.data_preparation import DataPreparation
import numpy as np, argparse, Unet.train_utils as train_utils, tensorflow as tf


# ------ HYPER PARAMETERS ------ #
data_dir = '/home/bispl/바탕화면/super-res-cycleGAN/data/2DHD/'
parser = argparse.ArgumentParser(description='')
parser.add_argument('--TRAIN_INPUT_DIR', dest='TRAIN_INPUT_DIR', default=data_dir + '2DHD_Train.tif')
parser.add_argument('--TRAIN_LABEL_DIR', dest='TRAIN_LABEL_DIR', default=data_dir + '2DHD_Train_Label.tif')
# parser.add_argument('--TEST_INPUT_DIR', dest='TEST_INPUT_DIR', default=data_dir + '2DHD_Train_Label.tif')
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
train_data = DP.scaled_data
DP.read_data(args.TRAIN_LABEL_DIR)
DP.scale_normalization(DP.raw_data)
train_label = DP.scaled_data

(train_dataset, train_label),(test_dataset, test_label) = DP.data_separation(train_data, train_label, 450, 50)

Unet = Network(args)
Unet.model.summary()

Unet.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.LEARNING_RATE),
                   loss=tf.keras.losses.MeanSquaredError(),
                   metrics=[train_utils.PSNR()])

Unet.model.fit(np.expand_dims(train_dataset, -1), np.expand_dims(train_label, -1), batch_size=args.BATCH_SIZE, epochs=1)
test_loss, test_acc = Unet.model.evaluate(np.expand_dims(test_dataset, -1), np.expand_dims(test_label, -1), verbose=2)


#result = Unet.model(np.expand_dims(train_dataset, 3), training=False)
#print(np.shape(result))