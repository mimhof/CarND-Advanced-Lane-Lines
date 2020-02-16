# Written by: Michael Imhof
# Date: 02/01/2020
# Part of Udacity Self-Driving Car Nanodegree
# Advanced Lane Finding Project

# Imports
import numpy as np
from collections import deque
import cv2
from matplotlib import pyplot as plt
import os
from python_src.camera_calibration import CameraCalibration
from python_src.thresholding import Thresholding, Thresholds
from python_src.line_finder import LineFinder


class Line(object):
    """Class holds the current attributes describing the current line."""

    def __init__(self, ploty, history_len=30, use_equal_weights=True):
        self._ploty = ploty
        self._history_len = history_len
        self._use_equal_weights = use_equal_weights
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

        self.bad_frames = 0

    def append(self, poly, xvals):

        curverad = self.compute_curvature(xvals, self._ploty)
        self.detected = self.sanity_check_lane(curverad)

        if self.detected:
            # Append xvals to the deque
            self._recent_xfitted.append(xvals)
            # Set current_fit to poly
            self.current_fit = poly

            # Update bestx
            if len(self._recent_xfitted) == 1:
                self.bestx = np.mean(self._recent_xfitted, axis=0)
            else:
                weights = np.zeros(len(self._recent_xfitted))
                if self._use_equal_weights:
                    weights[:] = 1.0 / len(weights)
                else:
                    weights[0] = 0.5
                    weights[1:] = (1.0 - weights[0]) / (len(weights) - 1)
                self.bestx = np.sum([x*y for x, y in
                                     zip(weights, self._recent_xfitted)], axis=0)

            # Calculate a new polynomial based on bestx
            self.best_fit = np.polyfit(self._ploty, self.bestx, 2)

            curverad = self.compute_curvature(self.bestx, self._ploty)
            self.radius_of_curvature = curverad

            # Determine line_base_pos
            self._find_line_base_pos()

            self.bad_frames = 0

        else:
            self.bad_frames += 1

    def _find_line_base_pos(self):
        # Determine distance of line from vehicle center
        self.line_base_pos = (self.bestx[0] - 640) * self._xm_per_pix

    def compute_curvature(self, xvals, yvals):
        fit = np.polyfit(
            yvals * self._ym_per_pix, xvals * self._xm_per_pix, 2)
        y_eval = np.max(yvals)
        curveature = ((1 + (2 * fit[0] * y_eval + fit[1]) ** 2) ** 1.5) /\
            np.absolute(2 * fit[0])
        return curveature

    def reset(self):
        last_val = self._recent_xfitted.pop()
        self._recent_xfitted.clear()
        self._recent_xfitted.append(last_val)
        self.best_fit = self.current_fit
        self.radius_of_curvature = self.compute_curvature(last_val, self._ploty)
        self._find_line_base_pos()

    def sanity_check_lane(self, new_curvature):
        if self.radius_of_curvature is None:
            return True
        return (abs(new_curvature - self.radius_of_curvature) /
                self.radius_of_curvature) <= 0.5


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
        gray_image = cv2.cvtColor(undist, cv2.COLOR_RGB2GRAY)

        # Threshold image based on Thresholds
        s_binary = self._thresh.combine_gradients(s_chan)
        l_binary = self._thresh.combine_gradients(l_chan)
        gray_binary = self._thresh.combine_gradients(gray_image)
        combined = np.zeros_like(gray_binary)
        combined[(s_binary == 1) | (l_binary == 1) | (gray_binary == 1)] = 1

        # Warp the image to bird's-eye view
        warped = self._camera_cal.warp_image(combined)

        # Find the lane lines
        self._line_finder.find_lane_lines(warped, reset=reset)
        left_x, right_x = self._line_finder.evaluate_polynomials(self._ploty)

        # Determine if lines make sense and use history to smooth
        # frame to frame variation.
        polynomials = self._line_finder.polynomials
        self._left_line.append(polynomials[0], left_x)
        self._right_line.append(polynomials[1], right_x)

        detected_left = self._left_line.bad_frames < 10
        detected_right = self._right_line.bad_frames < 10

        if detected_left and detected_right and not reset:
            left_x = self._left_line.bestx
            right_x = self._right_line.bestx
            detected = True
        else:
            # Reset the line finder
            self._line_finder.find_lane_lines(warped, reset=True)
            self._left_line.reset()
            self._right_line.reset()
            detected = False

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
        # cv2.putText(result,
        #             'Tracking Locked' if detected else 'Tracking Lost Lock',
        #             (30, 180), font, 2,
        #             [0, 255, 0] if detected else [255, 0, 0], 2)

        return result


if __name__ == '__main__':
    from moviepy.editor import VideoFileClip

    # Process the test images
    # for imfile in os.listdir('test_images/'):
    #     lane_detector = LaneDetector()
    #     img = plt.imread('test_images/' + imfile)
    #     result = lane_detector.find_lane(img, reset=True)
    #     plt.imsave('output_images/' + imfile, result, format='jpg')

    # Process the project videos
    # lane_detector = LaneDetector()
    # file_clip = VideoFileClip('project_video.mp4')
    # clip = file_clip.fl_image(lane_detector.find_lane)
    # clip.write_videofile('project_video_out.mp4', audio=False)
    # lane_detector = LaneDetector()
    # file_clip = VideoFileClip('challenge_video.mp4')
    # clip = file_clip.fl_image(lane_detector.find_lane)
    # clip.write_videofile('challenge_video_out.mp4', audio=False)
    lane_detector = LaneDetector()
    file_clip = VideoFileClip('harder_challenge_video.mp4')
    clip = file_clip.fl_image(lane_detector.find_lane)
    clip.write_videofile('harder_challenge_video_out.mp4', audio=False)
