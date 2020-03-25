import tensorflow as tf, numpy as np, matplotlib.pyplot as plt
from dataset_pipeline.imagedataloader import ImageDataLoader

data_dir = '/home/bispl/바탕화면/super-res-cycleGAN/data/natural/'
dir_list = [data_dir+'train set/train', data_dir+'train set/valid', data_dir+'test set']
loader = ImageDataLoader('test', 5)
loader.load_image_automatically(dir_list, 128, 128, 1./255)

train_img, train_label = next(loader.data_iter[0])
print(np.shape(train_img), np.shape(train_label))


plt.figure(0)
for i in range(5):
    train_img, train_label = next(loader.data_iter[0])
    #valid_img, valid_label = next(valid_generator)
    for j in range(5):
        plt.subplot(5, 5, j + 1 + 5 * i)
        plt.imshow(train_label[j])
        plt.colorbar()
plt.show()


'''
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator()
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator()

train_generator = train_datagen.flow_from_directory(data_dir+'train set/train', target_size=(128, 128), batch_size=5)
valid_generator = valid_datagen.flow_from_directory(data_dir+'train set/valid', target_size=(128, 128), batch_size=5)
test_generator = test_datagen.flow_from_directory(data_dir+'test set', target_size=(128, 128), batch_size=5)

plt.figure(0)
for i in range(5):
    train_img, train_label = next(train_generator)
    #valid_img, valid_label = next(valid_generator)
    for j in range(5):
        plt.subplot(5, 5, j + 1 + 5 * i)
        plt.imshow(train_img[j])
        plt.colorbar()
plt.show()
'''