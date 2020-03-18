from Unet.data_utils import *
import random


class DataPreparation:
    def __init__(self):
        self.raw_data = None
        self.scaled_data = None

    def read_data(self, img_dir):
        self.raw_data = read_image(img_dir)

    def scale_normalization(self, image, x=0.0, y=1.0):
        temp = image
        for num in range(np.shape(image)[0]):
            if len(np.shape(temp)) == 3:
                temp[num, :, :] = scale_normalization(temp[num, :, :])
                #if not temp[num, :, :]: break
            elif len(np.shape(temp)) == 4:
                temp[num, :, :, :] = scale_normalization(temp[num, :, :, :])
                #if not temp[num, :, :]: break

        self.scaled_data = temp

    # This should be changed to become more reasonable.
    def data_separation(self, input_image, label_image, train_num, test_num, shuffle=True):
        total_num = np.shape(input_image)[0]
        if total_num < train_num + test_num:
            print('Image is less than required number (train number + test number).')
            print('Total image: '+str(total_num)+'. Required image: '+str(train_num+test_num)+'.')
            return False

        ref_list = [x for x in range(total_num)]
        if shuffle:
            random.shuffle(ref_list)

        if len(np.shape(input_image)) == 3:
            train_set = input_image[ref_list[0:train_num], :, :]
            test_set = input_image[ref_list[train_num:train_num+test_num], :, :]
            train_label_set = label_image[ref_list[0:train_num], :, :]
            test_label_set = label_image[ref_list[train_num:train_num+test_num], :, :]

        elif len(np.shape(input_image)) == 4:
            train_set = input_image[ref_list[0:train_num], :, :, :]
            test_set = input_image[ref_list[train_num:train_num+test_num], :, :, :]
            train_label_set = label_image[ref_list[0:train_num], :, :, :]
            test_label_set = label_image[ref_list[train_num:train_num + test_num], :, :, :]

        else:
            print('Cannot separate image set. Please check the dimension. It should be 3 or 4.')
            return False

        return (train_set, train_label_set),(test_set, test_label_set)



