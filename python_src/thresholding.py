# Written by: Michael Imhof
# Date: 01/31/2020
# Part of Udacity Self-Driving Car Nanodegree
# Advanced Lane Finding Project

# Imports
import numpy as np
import cv2
from dataclasses import dataclass


@dataclass
class Thresholds:
    x_thresh: tuple = (10, 160)
    y_thresh: tuple = (10, 160)
    mag_thresh: tuple = (30, 100)
    dir_thresh: tuple = (0.7, 1.3)


class Thresholding(object):
    """Class to apply transforms and thresholds to the road images."""

    def __init__(self, image, thresholds, sobel_kernel=3):
        self._image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)[:, :, 2]
        self._thresholds = thresholds
        self._sobel_kernel = sobel_kernel
        self._sobel_x, self._sobel_y = self._calculate_sobel_gradients(
            sobel_kernel=self._sobel_kernel)

    def _calculate_sobel_gradients(self, sobel_kernel=3):
        """
        Apply the Sobel kernel to compute the image gradients.

        :param sobel_kernel: The size of the Sobel kernel (default = 3).
        :return: The Sobel image gradients x and y.
        """
        sobel_x = cv2.Sobel(self._image, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobel_y = cv2.Sobel(self._image, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        return sobel_x, sobel_y

    def _threshold_sobel_xy(self):
        """
        Calculate the Sobel gradient in x and y and apply thresholding.

        :return: The binary thresholded x & y Sobel gradient images.
        """
        scaled_sobel_x = np.uint8(255.*np.absolute(self._sobel_x) /
                                  np.max(np.absolute(self._sobel_x)))
        scaled_sobel_y = np.uint8(255. * np.absolute(self._sobel_y) /
                                  np.max(np.absolute(self._sobel_y)))
        binary_x = Thresholding.threshold(scaled_sobel_x,
                                          self._thresholds.x_thresh)
        binary_y = Thresholding.threshold(scaled_sobel_y,
                                          self._thresholds.y_thresh)
        return binary_x, binary_y

    def _threshold_sobel_magnitude(self):
        """
        Calculate the magnitude Sobel gradient and apply a threshold.

        :return: The binary thresholded magnitude Sobel gradient images.
        """
        sobel_mag = np.sqrt(self._sobel_x**2 + self._sobel_y**2)
        scale_factor = np.max(sobel_mag) / 255
        sobel_mag = (sobel_mag / scale_factor).astype(np.uint8)
        return Thresholding.threshold(sobel_mag, self._thresholds.mag_thresh)

    def _threshold_sobel_direction(self):
        """
        Calculate the direction Sobel gradient and apply a threshold.

        :return: The binary thresholded direction Sobel gradient images.
        """
        sobel_dir = np.arctan2(np.absolute(self._sobel_y),
                               np.absolute(self._sobel_x))
        return Thresholding.threshold(sobel_dir, self._thresholds.dir_thresh)

    def combine_gradients(self):
        """
        Combine the thresholded Sobel gradients into a composite output.

        :return: The combined Sobel thresholded gradient output.
        """
        sobel_x, sobel_y = self._threshold_sobel_xy()
        sobel_mag = self._threshold_sobel_magnitude()
        sobel_dir = self._threshold_sobel_direction()
        combined = np.zeros_like(self._image)
        combined[((sobel_x == 1) & (sobel_y == 1) | (sobel_mag == 1) & (sobel_dir == 1))] = 1
        return combined

    @staticmethod
    def threshold(image, thresh):
        """
        Apply a minimum and maximum threshold to produce a binary image.

        :param image: The image to threshold.
        :param thresh: A threshold tuple as (min, max)
        :return: A binary image produced through thresholding.
        """
        binary_image = np.zeros_like(image)
        binary_image[(image >= thresh[0]) & (image <= thresh[1])] = 1
        return binary_image


if __name__ == '__main__':
    from python_src.camera_calibration import CameraCalibration
    from matplotlib import pyplot as plt

    camera_cal = CameraCalibration()
    img = plt.imread("test_images/straight_lines1.jpg")
    undist = camera_cal.undistort_image(img)
    thresholds = Thresholds()
    thresh = Thresholding(undist, thresholds, sobel_kernel=5)
    binary = thresh.combine_gradients()
    warped = camera_cal.warp_image(binary)

    fig = plt.figure(figsize=(20, 10))
    ax0 = fig.add_subplot(121)
    ax0.imshow(binary, cmap='gray')
    ax1 = fig.add_subplot(122)
    ax1.imshow(warped, cmap='gray')

    plt.show()
