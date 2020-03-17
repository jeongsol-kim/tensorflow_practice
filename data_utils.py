from PIL import Image
import numpy as np
import copy


def read_image(img_dir, RGB = False): # default type = float32.
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
            except EOFError:
                break
        return img_array # output shape = [samples, w, h, c]

    elif img_dir.endswith('.png'): # only one image. how to collect images?
        img = Image.open(img_dir)
        if not RGB:
            img = img.convert('L')
        return np.array(img)

    else:
        print('.tiff or .png file is available only.')
        return False


def scale_normalization(image, x = 0.0, y = 1.0):  # datatype = numpy
    image = image + 1e-10  # If there are zeros in raw image, divide by zero can occur.
    dim = np.shape(image)
    if len(dim) == 2:  # input shape: [w,h]
        max_value = np.amax(image)
        min_value = np.amin(image)

    else:
        print("2D images can be normalized. If you use (a set of) 3D image(s), please separate them.")
        return False

    # scale normalization
    image = (image - min_value) / (max_value - min_value)
    image = image * (y - x) + x
    return image
