import random

import numpy as np

GAUSS_MEAN = 0


class RandomNoise():

    def __init__(self, seed=None, p=1.0, intensity=0.5):
        """
        Initialises an instance.

        Parameters
        ----------
        seed      : int
                    Random seed for random number generator
        p         : float
                    Probability of augmenting a single example, must be in a
                    range of [0, 1]
        intensity : float
                    Augmentation intensity, must be in a [0, 1] range
        """

        self.p = p
        self.intensity = intensity
        self.random = random.Random()
        if seed is not None:
            self.seed = seed
            self.random.seed(seed)

    def add_noise(self, image, noise_typ="gauss"):

        if self.random.uniform(0, 1) < self.p:
            return self._add_noise(image, noise_typ)
        else:
            return image

    def _add_noise(self, image, noise_typ):
        """
        (Code snippet taken from https://stackoverflow.com/questions/22937589/
        how-to-add-noise-gaussian-salt-and-pepper-etc-to-image-in-python-with-opencv)
        """

        if noise_typ == "gauss":
            row, col, ch = image.shape
            sigma = self.intensity
            gauss = np.random.normal(GAUSS_MEAN, sigma, (row, col, ch))
            gauss = gauss.reshape(row, col, ch)
            noisy = image + gauss
            return noisy
        elif noise_typ == "s&p":
            row, col, ch = image.shape
            s_vs_p = 0.5
            amount = self.intensity
            out = np.copy(image)
            # Salt mode
            num_salt = np.ceil(amount * image.size * s_vs_p)
            coords = [np.random.randint(0, i - 1, int(num_salt))
                      for i in image.shape]
            out[coords] = 1

            # Pepper mode
            num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
            coords = [np.random.randint(0, i - 1, int(num_pepper))
                      for i in image.shape]
            out[coords] = 0
            return out
        elif noise_typ == "poisson":
            vals = len(np.unique(image))
            vals = 2 ** np.ceil(np.log2(vals))
            noisy = np.random.poisson(image * vals) / float(vals)
            return noisy
        elif noise_typ == "speckle":
            row, col, ch = image.shape
            gauss = np.random.randn(row, col, ch)
            gauss = gauss.reshape(row, col, ch)
            noisy = image + image * gauss
            return noisy
        else:
            raise ValueError("Invalid noise type.")
