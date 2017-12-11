import random

import numpy as np
from skimage.transform import ProjectiveTransform, rotate, warp


class RandomTransform():
    """
    Class to random transform images
    http://navoshta.com/traffic-signs-classification/
    """

    def __init__(self, seed=None, p=1.0, intensity=0.5):
        """
        Initialises an instance.

        Parameters
        ----------
        seed      :
                    Random seed.
        p         :
                    Probability of augmenting a single example, should be in a
                    range of [0, 1]. Defines data augmentation coverage.
        intensity :
                    Augmentation intensity, should be in a [0, 1] range.
        """

        self.p = p
        self.intensity = intensity
        self.random = random.Random()
        if seed is not None:
            self.seed = seed
            self.random.seed(seed)
        self.last_tran = None

    def get_last_transform(self):
        return self.last_tran

    def transform(self, image, order=1):
        """
        Transform image by random rotation and random projection transform.
        Assume image has shape (height, width, channels).

        order :
                0: Nearest-neighbor
                1: Bi-linear (default)
                2: Bi-quadratic
                3: Bi-cubic
                4: Bi-quartic
                5: Bi-quintic
        """

        #image_rotate = self.rotate(image)
        image_rotate = image
        image_trans = self.apply_projection_transform(image_rotate, order)

        return image_trans

    def rotate(self, image):
        """
        Applies random rotation in a defined degrees range to an image.
        """

        if self.random.random() < self.p:
            delta = 30. * self.intensity  # scale by self.intensity
            image_rotate = rotate(image, self.random.uniform(-delta, delta),
                                  mode='edge')
            return image_rotate
        else:
            return image

    def apply_projection_transform(self, image, order=1):
        """
        Applies projection transform to an image. Projection margins are
        randomised in a range depending on the size of the image.
        """

        image_size = image.shape[0]
        d = image_size * 0.3 * self.intensity

        if self.random.random() < self.p:

            tl_top = self.random.uniform(-d, d)     # Top left corner, top
            tl_left = self.random.uniform(-d, d)    # Top left corner, left
            bl_bottom = self.random.uniform(-d, d)  # Bot left corner, bot
            bl_left = self.random.uniform(-d, d)    # Bot left corner, left
            tr_top = self.random.uniform(-d, d)     # Top right corner, top
            tr_right = self.random.uniform(-d, d)   # Top right corner, right
            br_bottom = self.random.uniform(-d, d)  # Bot right corner, bot
            br_right = self.random.uniform(-d, d)   # Bot right corner, right

            transform = ProjectiveTransform()
            transform.estimate(np.array((
                (tl_left, tl_top),
                (bl_left, image_size - bl_bottom),
                (image_size - br_right, image_size - br_bottom),
                (image_size - tr_right, tr_top)
            )), np.array((
                (0, 0),
                (0, image_size),
                (image_size, image_size),
                (image_size, 0)
            )))
            self.last_tran = transform
            image_trans = warp(image, transform,
                               output_shape=(image_size, image_size),
                               order=order, mode='edge')
            return image_trans
        else:
            return image

    def apply_transform(self, image, transform, order=1):

        image_size = image.shape[0]
        image_trans = warp(image, transform,
                           output_shape=(image_size, image_size),
                           order=order, mode='edge')
        return image_trans
