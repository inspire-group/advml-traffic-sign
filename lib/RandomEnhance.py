import random

import numpy as np
from PIL import Image, ImageEnhance


class RandomEnhance():
    """
    Class to randomly enhance image by adjusting color, contrast, sharpness and
    brightness
    """

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
        self.last_factors = None

    def get_last_factors(self):
        return self.last_factors

    def enhance(self, image):
        """
        Randomly enhance input image with probability p
        """

        if self.random.uniform(0, 1) < self.p:
            return self.intensity_enhance(image)
        else:
            return image

    def intensity_enhance(self, im):
        """
        Perform random enhancement with specified intensity [0,1]. The range of
        random factors are chosen to be in an appropriate range.
        """

        color_factor = self.intensity * self.random.uniform(-0.4, 0.4) + 1
        contrast_factor = self.intensity * self.random.uniform(-0.5, 0.5) + 1
        sharpess_factor = self.intensity * self.random.uniform(-0.8, 0.8) + 1
        bright_factor = self.intensity * self.random.uniform(-0.5, 0.5) + 1
        self.last_factors = [color_factor, contrast_factor, sharpess_factor, bright_factor]

        image = Image.fromarray(np.uint8(im * 255))
        enhancer = ImageEnhance.Color(image)
        image = enhancer.enhance(color_factor)
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(contrast_factor)
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(sharpess_factor)
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(bright_factor)

        return np.asarray(image) / 255.

    def enhance_factors(self, im, factors):

        image = Image.fromarray(np.uint8(im * 255))
        enhancer = ImageEnhance.Color(image)
        image = enhancer.enhance(factors[0])
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(factors[1])
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(factors[2])
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(factors[3])

        return np.asarray(image) / 255.
