import tensorflow as tf, numpy as np, matplotlib.pyplot as plt
from Unet.data_utils import *


data_dir = '/home/bispl/바탕화면/super-res-cycleGAN/data/2DHD/'
img = read_image(data_dir + '2DHD_Train.tif', 10)
img = np.expand_dims(img,-1)
print(np.shape(img))

patch_set = tf.image.extract_patches(img, [1,128,128,1], [1,128,128,1], [1,1,1,1], 'VALID')
print(np.shape(patch_set))
recon_patch = tf.reshape(patch_set, [10*4*4,128,128,1])


recon_img = patch2img(recon_patch[:16,:,:,:], 128)
print(np.shape(recon_img))

plt.figure(1)
plt.subplot(221)
plt.imshow(recon_patch[0,:,:,0])
plt.subplot(222)
plt.imshow(recon_img[:,:,0])
plt.subplot(223)
plt.imshow(img[0,:,:,0])
plt.subplot(224)
plt.imshow(recon_img[:,:,0]-img[0,:,:,0])
plt.show()

print(np.sum(recon_img[:,:,0]-img[0,:,:,0]))
