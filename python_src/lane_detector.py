# Written by: Michael Imhof
# Date: 02/01/2020
# Part of Udacity Self-Driving Car Nanodegree
# Advanced Lane Finding Project

# Imports
import numpy as np
from collections import deque
import cv2
from matplotlib import pyplot as plt
from python_src.camera_calibration import CameraCalibration
from python_src.thresholding import Thresholding, Thresholds
from python_src.line_finder import LineFinder


class Line(object):
    """Class holds the current attributes describing the current line."""

    def __init__(self, ploty, history_len=30):
        self._ploty = ploty
        self._history_len = history_len
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self._recent_xfitted = deque(maxlen=self._history_len)
        # average x values of the fitted line over the last n iterations
        self.bestx = None
        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        # polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        # radius of curvature of the line in some units
        self.radius_of_curvature = None
        # distance in meters of vehicle center from the line
        self.line_base_pos = None
        # Define conversions in x and y from pixels space to meters
        self._ym_per_pix = 30 / 720  # meters per pixel in y dimension
        self._xm_per_pix = 3.7 / 847  # meters per pixel in x dimension

    def append(self, poly, xvals):
        # Set current_fit to poly
        self.current_fit = poly

        # Append xvals to the deque
        self._recent_xfitted.append(xvals)

        # Update bestx
        self.bestx = np.mean(self._recent_xfitted, axis=0)

        # Calculate a new polynomial based on bestx
        self.best_fit = np.polyfit(self._ploty, self.bestx, 2)

        # Calculate curvature
        self._find_curvature()

        # Determine line_base_pos
        self._find_line_base_pos()

    def _find_line_base_pos(self):
        # Determine distance of line from vehicle center
        self.line_base_pos = (self.bestx[0] - 640) * self._xm_per_pix

    def _find_curvature(self):
        y_eval = np.max(self._ploty)
        polyfit_cr = np.polyfit(self._ploty * self._ym_per_pix,
                                self.bestx * self._xm_per_pix, 2)
        self.radius_of_curvature = Line.calculate_curvature(
            polyfit_cr, y_eval * self._ym_per_pix)

    @staticmethod
    def calculate_curvature(poly, y):
        return ((1 + (2 * poly[0] * y + poly[1]) ** 2) ** 1.5) / abs(2 * poly[0])

    def reset(self):
        last_val = self._recent_xfitted.pop()
        self._recent_xfitted.clear()
        self._recent_xfitted.append(last_val)
        self.best_fit = self.current_fit
        self._find_curvature()
        self._find_line_base_pos()


def sanity_check(left_line, right_line):
    return True


class LaneDetector(object):
    """Top-level class for detecting lanes."""

    def __init__(self):
        self._imshape = (720, 1280)
        self._ploty = np.linspace(0, self._imshape[0] - 1, self._imshape[0])
        self._left_line = Line(self._ploty, history_len=10)
        self._right_line = Line(self._ploty, history_len=10)
        self._camera_cal = CameraCalibration()
        self._thresholds = Thresholds()
        self._thresh = Thresholding(self._thresholds, sobel_kernel=5)
        self._line_finder = LineFinder(margin=200)

    def find_lane(self, image, reset=False):
        # Correct camera distortion
        undist = self._camera_cal.undistort_image(image)

        # Convert to the HLS color space
        hls_image = cv2.cvtColor(undist, cv2.COLOR_RGB2HLS)
        s_chan = hls_image[:, :, 2]
        l_chan = hls_image[:, :, 1]

        # Threshold image based on Thresholds
        binary = self._thresh.combine_gradients(s_chan)

        # Warp the image to bird's-eye view
        warped = self._camera_cal.warp_image(binary)

        # Find the lane lines
        self._line_finder.find_lane_lines(warped, reset=reset)
        left_x, right_x = self._line_finder.evaluate_polynomials(self._ploty)

        # Determine if lines make sense and use history to smooth
        # frame to frame variation.
        polynomials = self._line_finder.polynomials
        self._left_line.append(polynomials[0], left_x)
        self._right_line.append(polynomials[1], right_x)

        sane_lines = sanity_check(self._left_line, self._right_line)
        self._left_line.detected = sane_lines
        self._right_line.detected = sane_lines

        if sane_lines is False:
            # Reset the line finder
            self._line_finder.find_lane_lines(warped, reset=True)
            self._left_line.reset()
            self._right_line.reset()
        else:
            left_x = self._left_line.bestx
            right_x = self._right_line.bestx

        # Draw lane lines on an empty image
        drawn_image = np.zeros_like(image)
        cv2.polylines(
            drawn_image,
            [np.array([[a, b] for (a, b) in zip(left_x, self._ploty)], np.int0)],
            False, [255, 0, 0], 50)
        cv2.polylines(
            drawn_image,
            [np.array([[a, b] for (a, b) in zip(right_x, self._ploty)], np.int0)],
            False, [0, 0, 255], 50)

        # Draw filled polynomial lane line on image
        total_pts = np.concatenate([
            np.array([[a, b] for (a, b) in zip(left_x, self._ploty)], np.int0),
            np.flipud(np.array([[a, b] for (a, b) in zip(right_x, self._ploty)], np.int0))])
        cv2.fillPoly(drawn_image, [total_pts], (0, 255, 0))

        # Un-warp the drawn image
        empty_unwarped = self._camera_cal.unwarp_image(drawn_image)

        # Combine the undistorted image with the drawn image
        result = cv2.addWeighted(undist, 1, empty_unwarped, 0.5, 0)

        # Curvature and center info
        left_curvature = self._left_line.radius_of_curvature
        right_curvature = self._right_line.radius_of_curvature
        curvature = np.minimum(left_curvature, right_curvature)
        vehicle_pos = abs(self._right_line.line_base_pos +
                          self._left_line.line_base_pos) / 2
        if self._right_line.line_base_pos > abs(self._left_line.line_base_pos):
            message = 'Vehicle is {0:.2f}m right of center'.format(vehicle_pos)
        elif abs(self._left_line.line_base_pos) > self._right_line.line_base_pos:
            message = 'Vehicle is {0:.2f}m left of center'.format(vehicle_pos)
        else:
            message = 'Vehicle is 0.00m right of center'

        # Draw info
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_color = (255, 255, 255)
        cv2.putText(result,
                    'Radius of Curvature: {:.0f} (m)'.format(curvature),
                    (30, 60), font, 2, font_color, 2)
        cv2.putText(result,
                    message,
                    (30, 120), font, 2, font_color, 2)

        return result


if __name__ == '__main__':
    from moviepy.editor import VideoFileClip
    lane_detector = LaneDetector()
    file_clip = VideoFileClip('project_video.mp4').subclip(0, 5)
    # img = plt.imread('test_images/test1.jpg')
    clip = file_clip.fl_image(lane_detector.find_lane)
    # plt.imshow(lane_detector.find_lane(img, reset=False))
    clip.write_videofile('project_video_out.mp4', audio=False)
    plt.show()
