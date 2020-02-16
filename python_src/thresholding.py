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

    def __init__(self, thresholds, sobel_kernel=3):
        """
        Initialize attributes and Sobel x and y gradients.

        :param thresholds: A Thresholds object for x, y, mag, and dir.
        :param sobel_kernel: Size of the Sobel kernel to use.
        """
        self._thresholds = thresholds
        self._sobel_kernel = sobel_kernel

    def _calculate_sobel_gradients(self, image):
        """
        Apply the Sobel kernel to compute the image gradients.

        :param image: An image object.
        :return: The Sobel image gradients x and y.
        """
        sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=self._sobel_kernel)
        sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=self._sobel_kernel)
        return sobel_x, sobel_y

    def _threshold_sobel_xy(self, sobel_x, sobel_y):
        """
        Calculate the Sobel gradient in x and y and apply thresholding.

        :param sobel_x: Sobel gradient over x-orientation.
        :param sobel_y: Sobel gradient over y-orientation.
        :return: The binary thresholded x & y Sobel gradient images.
        """
        scaled_sobel_x = np.uint8(255.*np.absolute(sobel_x) /
                                  np.max(np.absolute(sobel_x)))
        scaled_sobel_y = np.uint8(255. * np.absolute(sobel_y) /
                                  np.max(np.absolute(sobel_y)))
        binary_x = Thresholding.threshold(scaled_sobel_x,
                                          self._thresholds.x_thresh)
        binary_y = Thresholding.threshold(scaled_sobel_y,
                                          self._thresholds.y_thresh)
        return binary_x, binary_y

    def _threshold_sobel_magnitude(self, sobel_x, sobel_y):
        """
        Calculate the magnitude Sobel gradient and apply a threshold.

        :param sobel_x: Sobel gradient over x-orientation.
        :param sobel_y: Sobel gradient over y-orientation.
        :return: The binary thresholded magnitude Sobel gradient images.
        """
        sobel_mag = np.sqrt(sobel_x**2 + sobel_y**2)
        scale_factor = np.max(sobel_mag) / 255
        sobel_mag = (sobel_mag / scale_factor).astype(np.uint8)
        return Thresholding.threshold(sobel_mag, self._thresholds.mag_thresh)

    def _threshold_sobel_direction(self, sobel_x, sobel_y):
        """
        Calculate the direction Sobel gradient and apply a threshold.

        :param sobel_x: Sobel gradient over x-orientation.
        :param sobel_y: Sobel gradient over y-orientation.
        :return: The binary thresholded direction Sobel gradient images.
        """
        sobel_dir = np.arctan2(np.absolute(sobel_y),
                               np.absolute(sobel_x))
        return Thresholding.threshold(sobel_dir, self._thresholds.dir_thresh)

    def combine_gradients(self, image):
        """
        Combine the thresholded Sobel gradients into a composite output.

        :param image: An image object.
        :return: The combined Sobel thresholded gradient output.
        """
        sobel_x, sobel_y = self._calculate_sobel_gradients(image)
        sobel_x, sobel_y = self._threshold_sobel_xy(sobel_x, sobel_y)
        sobel_mag = self._threshold_sobel_magnitude(sobel_x, sobel_y)
        sobel_dir = self._threshold_sobel_direction(sobel_x, sobel_y)
        combined = np.zeros_like(image)
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
    img = plt.imread("test_images/test1.jpg")
    undist = camera_cal.undistort_image(img)
    thresholds = Thresholds()
    hls_image = cv2.cvtColor(undist, cv2.COLOR_RGB2HLS)
    s_chan = hls_image[:, :, 2]
    l_chan = hls_image[:, :, 1]
    gray_image = cv2.cvtColor(undist, cv2.COLOR_RGB2GRAY)
    thresh = Thresholding(thresholds, sobel_kernel=5)
    s_binary = thresh.combine_gradients(s_chan)
    l_binary = thresh.combine_gradients(l_chan)
    gray_binary = thresh.combine_gradients(gray_image)
    s_warped = camera_cal.warp_image(s_binary)

    plt.imsave("writeup_media/s_channel_binary.jpg", s_binary)
    plt.imsave("writeup_media/l_channel_binary.jpg", l_binary)
    plt.imsave("writeup_media/gray_binary.jpg", gray_binary)
    plt.imsave("writeup_media/s_channel_warped.jpg", s_warped)
