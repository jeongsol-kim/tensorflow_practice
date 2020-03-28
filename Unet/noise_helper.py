import numpy as np

class NoiseHelper():
    def __init__(self):
        pass

    def make_noise(self, image, noise_type='gauss', features=[0, 0.1]):

        if noise_type == "gauss":
            return self.gaussian_noise(image, features[0], features[1])
        elif noise_type == 'salt_and_pepper':
            return self.salt_and_pepper_noise(image, features[0], features[1])
        elif noise_type == "poisson":
            return self.possion_noise(image)
        elif noise_type == "speckle":
            return self.speckle_noise(image)
        else:
            print('NoiseHelper did not learn that noise. Please check the possible noises.')
            print('Possible noises: Gaussian, Salt and Pepper, Poisson, Speckle')
            return None

    def gaussian_noise(self, image, mean=0, var=0.1):
        bn, row, col, ch = image.shape
        sigma = var ** 0.5
        gauss = np.random.normal(mean, sigma, (bn, row, col, ch))
        gauss = gauss.reshape(bn, row, col, ch)
        return image + gauss

    def salt_and_pepper_noise(self, image, ratio=0.5, amount=0.004):
        bn, row, col, ch = image.shape
        out = np.copy(image)
        for batch in range(bn):
            temp = np.copy(image[batch,:,:,0])

            # Salt mode
            num_salt = np.ceil(amount * temp.size * ratio)
            coords = [np.random.randint(0, i - 1, int(num_salt))
                      for i in temp.shape]
            temp[coords] = 1

            # Pepper mode
            num_pepper = np.ceil(amount * temp.size * (1. - ratio))
            coords = [np.random.randint(0, i - 1, int(num_pepper))
                      for i in temp.shape]
            temp[coords] = 0
            out[batch,:,:,:] = np.expand_dims(temp, -1)
        return out

    def possion_noise(self, image):
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        return np.random.poisson(image * vals) / float(vals)

    def speckle_noise(self, image):
        bn, row, col, ch = image.shape
        gauss = np.random.randn(bn, row, col, ch)
        gauss = gauss.reshape(bn, row, col, ch)
        return image + image * gauss