# Written by: Michael Imhof
# Date: 02/01/2020
# Part of Udacity Self-Driving Car Nanodegree
# Advanced Lane Finding Project

# Imports
import numpy as np
import cv2
from matplotlib import pyplot as plt


class LineFinder(object):
    """Class contains methods to find lane lines."""

    def __init__(self, margin=100):
        """

        """
        self._left_fit = None
        self._right_fit = None
        self._leftx = None
        self._lefty = None
        self._rightx = None
        self._righty = None
        self._margin = margin

    @property
    def polynomials(self):
        return self._left_fit, self._right_fit

    def _find_lane_pixels(self, image, nwindows=9, minpix=50):
        imshape = image.shape
        # Take a histogram of the bottom half of the image
        histogram = np.sum(image[imshape[0] // 2:, :], axis=0)

        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0] // 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Set height of windows - based on nwindows above and image shape
        window_height = np.int(imshape[0] // nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = image.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated later for each window in nwindows
        leftx_current = leftx_base
        rightx_current = rightx_base

        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = imshape[0] - (window + 1) * window_height
            win_y_high = imshape[0] - window * window_height
            win_xleft_low = leftx_current - self._margin
            win_xleft_high = leftx_current + self._margin
            win_xright_low = rightx_current - self._margin
            win_xright_high = rightx_current + self._margin

            good_left_inds = ((nonzeroy >= win_y_low) &
                              (nonzeroy < win_y_high) &
                              (nonzerox >= win_xleft_low) &
                              (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) &
                               (nonzeroy < win_y_high) &
                               (nonzerox >= win_xright_low) &
                               (nonzerox < win_xright_high)).nonzero()[0]

            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            # If you found > minpix pixels, recenter next window ###
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        # (previously was a list of lists of pixels)
        try:
            left_lane_inds = np.concatenate(left_lane_inds)
            right_lane_inds = np.concatenate(right_lane_inds)
        except ValueError:
            # Avoids an error if the above is not implemented fully
            pass

        # Extract left and right line pixel positions
        self._leftx = nonzerox[left_lane_inds]
        self._lefty = nonzeroy[left_lane_inds]
        self._rightx = nonzerox[right_lane_inds]
        self._righty = nonzeroy[right_lane_inds]

    def _fit_polynomial(self, order=2):
        # Fit a Nth-order polynomial to each using `np.polyfit`
        self._left_fit = np.polyfit(self._lefty, self._leftx, order)
        self._right_fit = np.polyfit(self._righty, self._rightx, order)

    def _fit_poly(self, order=2):
        # Fit a second order polynomial to each with np.polyfit
        self._left_fit = np.polyfit(self._lefty, self._leftx, order)
        self._right_fit = np.polyfit(self._righty, self._rightx, order)

        # Calc both polynomials using ploty, left_fit and right_fit
        left_fitx = np.polyval(self._left_fit, self._ploty)
        right_fitx = np.polyval(self._right_fit, self._ploty)

        return left_fitx, right_fitx

    def _search_around_poly(self, image):
        nonzero = image.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # Set the area of search based on activated x-values
        # within the +/- margin of our polynomial function
        left_lane_inds = (
            (nonzerox > (np.polyval(self._left_fit, nonzeroy) - self._margin)) &
            (nonzerox < (np.polyval(self._left_fit, nonzeroy) + self._margin)))
        right_lane_inds = (
            (nonzerox > (np.polyval(self._right_fit, nonzeroy) - self._margin)) &
            (nonzerox < (np.polyval(self._right_fit, nonzeroy) + self._margin)))

        # Again, extract left and right line pixel positions
        self._leftx = nonzerox[left_lane_inds]
        self._lefty = nonzeroy[left_lane_inds]
        self._rightx = nonzerox[right_lane_inds]
        self._righty = nonzeroy[right_lane_inds]

    def find_lane_lines(self, image, reset=False):
        if reset or self._left_fit is None or self._right_fit is None:
            self._find_lane_pixels(image)
        else:
            self._search_around_poly(image)
        self._fit_polynomial(order=2)

    def evaluate_polynomials(self, y):
        try:
            left_fitx = np.polyval(self._left_fit, y)
            right_fitx = np.polyval(self._right_fit, y)
        except TypeError:
            # Avoids an error if `left` and `right_fit` are still
            # none or incorrect
            print('The function failed to fit a line!')
            left_fitx = 1 * y ** 2 + 1 * y
            right_fitx = 1 * y ** 2 + 1 * y

        return left_fitx, right_fitx


if __name__ == '__main__':
    from python_src.camera_calibration import CameraCalibration
    from python_src.thresholding import Thresholds, Thresholding

    camera_cal = CameraCalibration()
    img = plt.imread("test_images/test1.jpg")
    undist = camera_cal.undistort_image(img)
    thresholds = Thresholds()
    s_chan = cv2.cvtColor(undist, cv2.COLOR_RGB2HLS)[:, :, 2]
    thresh = Thresholding(thresholds, sobel_kernel=5)
    binary = thresh.combine_gradients(s_chan)
    warped = camera_cal.warp_image(binary)

    line_finder = LineFinder(margin=200)
    line_finder.find_lane_lines(warped)
    # line_finder.find_lane_lines()
    imshape = warped.shape
    ploty = np.linspace(0, imshape[0] - 1, imshape[0])
    left_x, right_x = line_finder.evaluate_polynomials(ploty)

    empty_image = np.zeros_like(img)
    cv2.polylines(
        empty_image,
        [np.array([[a, b] for (a, b) in zip(left_x, ploty)], np.int0)],
        False, [255, 0, 0], 50)
    cv2.polylines(
        empty_image,
        [np.array([[a, b] for (a, b) in zip(right_x, ploty)], np.int0)],
        False, [255, 0, 0], 50)
    total_pts = np.concatenate([
        np.array([[a, b] for (a, b) in zip(left_x, ploty)], np.int0),
        np.flipud(np.array([[a, b] for (a, b) in zip(right_x, ploty)], np.int0))])
    cv2.fillPoly(empty_image, [total_pts], (0, 255, 0))
    empty_unwarped = camera_cal.unwarp_image(empty_image)
    result = cv2.addWeighted(undist, 1, empty_unwarped, 0.3, 0)

    fig = plt.figure(figsize=(20, 10))
    ax0 = fig.add_subplot(121)
    ax0.imshow(result)
    ax1 = fig.add_subplot(122)
    ax1.imshow(warped, cmap='gray')
    ax1.plot(left_x, ploty, '-y', linewidth=3)
    ax1.plot(right_x, ploty, '-y', linewidth=3)
    plt.show()
