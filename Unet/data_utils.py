from PIL import Image
import numpy as np, math, copy, tensorflow as tf


def read_image(img_dir, limit = 0, RGB = False): # default type = float32.
    if img_dir.endswith('.tif') or img_dir.endswith('.tiff'):
        img = Image.open(img_dir)
        img_array = np.zeros(np.shape(np.array(img)), dtype=np.float32)
        img_array = np.expand_dims(img_array, axis=0)
        count = 0
        while True:
            try:
                img.seek(count)
                next_img = np.expand_dims(np.array(img, dtype=np.float32), axis=0)
                if count == 0:
                    img_array = img_array + next_img
                else:
                    img_array = np.concatenate((img_array, next_img), axis=0)
                count += 1

                if limit != 0:
                    if count == limit:
                        return img_array
            except EOFError:
                break

        if len(np.shape(img_array)) == 3:
            img_array = np.expand_dims(img_array, -1)

        return img_array # output shape = [samples, w, h, c]

    elif img_dir.endswith('.png'): # only one image. how to collect images?
        img = Image.open(img_dir)
        if not RGB:
            img = img.convert('L')

        png = np.array(img)
        if len(np.shape(png))==3:
            png = np.expand_dims(png, -1)
        return png # output shape = [samples, w, h, c]

    else:
        print('.tiff or .png file is available only.')
        return False


def scale_normalization(image, x = 0.0, y = 1.0):  # datatype = numpy
    image = image + 1e-10  # If there are zeros in raw image, divide by zero can occur.
    dim = np.shape(image)
    if len(dim) == 3: # input shape: [w,h,c]
        if dim[-1] == 1:
            max_value = np.amax(image)
            min_value = np.amin(image)

            # scale normalization
            image = (image - min_value) / (max_value - min_value)
            image = image * (y - x) + x
            return image # output shape: [w,h,c]

        else:
            print('I did not implement this function for RGB image.')
            return False

    else:
        print("2D images can be normalized. If you use (a set of) 3D image(s), please separate them.")
        return False

# merge patches into one image
# input shape = [pn,ps,ps,ch]
# how to apply for tensor!!!!!!
def patch2img(patch_set, strides, patch_per_line_X = 0, patch_per_line_Y = 0):
    patch_set = np.array(patch_set)

    patch_num = np.shape(patch_set)[0]
    patch_size = np.shape(patch_set)[1]
    ch = np.shape(patch_set)[-1]

    if patch_per_line_X == 0 and patch_per_line_Y == 0:
        patch_per_line_X = int(math.sqrt(patch_num))
        patch_per_line_Y = int(math.sqrt(patch_num))

    image_size_X = patch_size * patch_per_line_X - (patch_size - strides) * (patch_per_line_X - 1)
    image_size_Y = patch_size * patch_per_line_Y - (patch_size - strides) * (patch_per_line_Y - 1)
    image = np.zeros([image_size_X, image_size_Y, ch], dtype=np.float32)
    count_mat = np.zeros([image_size_X, image_size_Y, ch], dtype=np.float32)
    patch_count = 0

    for i in range(int(patch_per_line_Y)):
        for j in range(int(patch_per_line_X)):
            patch = patch_set[patch_count, :, :, :]  # [1,ps,ps,ch]

            np.squeeze(patch)  # [ps,ps,ch]
            one_patch = np.copy(patch)
            one_patch[one_patch != 0] = 1

            image[i * strides:i * strides + patch_size, j * strides:j * strides + patch_size, :] += patch
            count_mat[i * strides:i * strides + patch_size, j * strides:j * strides + patch_size, :] += one_patch
            patch_count += 1

    count_mat[count_mat == 0] = -1
    count_mat = np.reciprocal(count_mat)
    avg_image = np.multiply(image, count_mat)

    return tf.convert_to_tensor(avg_image, dtype=tf.float32)