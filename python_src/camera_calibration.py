# Written by: Michael Imhof
# Date: 01/31/2020
# Part of Udacity Self-Driving Car Nanodegree
# Advanced Lane Finding Project

# Imports
import numpy as np
import cv2
import pickle


class CameraCalibration(object):
    """Class to calibrate the camera and warp perspective."""

    def __init__(self):
        # Initialize internal attributes
        pickle_dict = pickle.load(open("distortion_cal.p", "rb"))
        self._mtx = pickle_dict["mtx"]
        self._dist = pickle_dict["dist"]

        # Source points from test_images/straight_lines2.jpg
        self._imshape = (1280, 720)
        offset = 200
        # src = np.float32([[275, 679], [588, 455], [698, 455], [1042, 679]])
        src = np.float32([[218, 720], [588, 455], [698, 455], [1119, 720]])
        dst = np.float32([[offset, self._imshape[1]],
                          [offset, 0],
                          [self._imshape[0]-offset, 0],
                          [self._imshape[0]-offset, self._imshape[1]]])

        self._warp_matrix = cv2.getPerspectiveTransform(src, dst)
        self._unwarp_matrix = cv2.getPerspectiveTransform(dst, src)

    def calibrate_distortion(self, images, num_corners=(9, 6)):
        """
        Calculate the distortion of the camera based on provided chessboard
        images.

        :param images: Image filenames for use in calibration.
        :param num_corners: Number of internal corners on the chessboard image.
        """
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(9,6,0)
        objp = np.zeros((6*9, 3), np.float32)
        objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

        # Initialize arrays to hold object points and image points.
        objpoints = []
        imgpoints = []

        # Step through the list and search for chessboard corners
        image = None
        for fname in images:
            image = cv2.imread(fname)

            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, num_corners, None)

            # If found, add object points, image points
            if ret:
                objpoints.append(objp)
                imgpoints.append(corners)

        # Calculate distortion
        _, self._mtx, self._dist, _, _ = cv2.calibrateCamera(
            objpoints,
            imgpoints,
            image.shape[1::-1],
            None,
            None
        )

    def undistort_image(self, image):
        return cv2.undistort(image, self._mtx, self._dist, None, self._mtx)

    def warp_image(self, image):
        return cv2.warpPerspective(image,
                                   self._warp_matrix,
                                   self._imshape)

    def unwarp_image(self, image):
        return cv2.warpPerspective(image,
                                   self._unwarp_matrix,
                                   self._imshape)


if __name__ == '__main__':
    # Make a list of calibration images
    # import glob
    # image_filenames = glob.glob('camera_cal/calibration*.jpg')
    # camera_cal = CameraCalibration()
    # camera_cal.calibrate_distortion(image_filenames, num_corners=(9, 6))
    # pickle_dict = {"mtx": camera_cal._mtx, "dist": camera_cal._dist}
    # pickle.dump(pickle_dict, open("distortion_cal.p", "wb"))

    from matplotlib import pyplot as plt
    test_img = plt.imread('test_images/test1.jpg')
    # test_img = plt.imread('camera_cal/calibration1.jpg')
    camera_cal = CameraCalibration()
    undist = camera_cal.undistort_image(test_img)
    warped = camera_cal.warp_image(undist)
    # plt.imsave('writeup_media/calibration_image_distorted.jpg', test_img)
    # plt.imsave('writeup_media/calibration_image_undistorted.jpg', undist)
    plt.imsave('writeup_media/test_image_distorted.jpg', test_img)
    plt.imsave('writeup_media/test_image_undistorted.jpg', undist)
    plt.imsave('writeup_media/test_image_warped.jpg', warped)
    plt.figure(figsize=(16, 8))
    plt.imshow(warped)
    plt.show()
