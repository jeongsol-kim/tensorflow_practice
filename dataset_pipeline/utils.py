import tensorflow as tf, numpy as np, matplotlib.pyplot as plt
from PIL import Image

def read_tif_image(img_dir, limit = 0, RGB = False): # default type = float32.
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
