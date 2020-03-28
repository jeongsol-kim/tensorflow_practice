import copy
import numpy as np

def add_salt_pepper_noise(X_imgs, ratio = 0.5, amount=0.2):
    # Need to produce a copy as to not modify the original image
    X_imgs_copy = copy.deepcopy(X_imgs)
    _, row, col = X_imgs_copy.shape
    salt_vs_pepper = ratio
    num_salt = np.ceil(amount * X_imgs_copy.size * salt_vs_pepper)
    num_pepper = np.ceil(amount * X_imgs_copy.size * (1.0 - salt_vs_pepper))


    # Add Salt noise
    coords = [np.random.randint(0, i, int(num_salt)) for i in X_imgs_copy.shape]
    X_imgs_copy[coords[0], coords[1], coords[2]] = 1

    # Add Pepper noise
    coords = [np.random.randint(0, i, int(num_pepper)) for i in X_imgs_copy.shape]
    X_imgs_copy[coords[0], coords[1], coords[2]] = 0
    return X_imgs_copy