## Advanced Lane-Finding Project (P2)

### Michael Imhof
### Udacity Self-Driving Car Nanodegree
### Submitted: 02/18/20

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[cal_distorted]: ./writeup_media/calibration_image_distorted.jpg "Calibration Image Distorted"
[cal_undistorted]: ./writeup_media/calibration_image_undistorted.jpg "Calibration Image Undistorted"
[detected_lines]: ./writeup_media/detected_lines.png "Detected Lane Lines"
[gray_binary]: ./writeup_media/gray_binary.jpg "Gray-scale Binary"
[l_binary]: ./writeup_media/l_channel_binary.jpg "L-Channel Binary"
[s_binary]: ./writeup_media/s_channel_binary.jpg "S-Channel Binary"
[s_warped]: ./writeup_media/s_channel_warped.jpg "Warped S-Channel"
[test_distorted]: ./writeup_media/test_image_distorted.jpg "Test Image Distorted"
[test_undistorted]: ./writeup_media/test_image_undistorted.jpg "Test Image Undistorted"
[test_warped]: ./writeup_media/test_image_warped.jpg "Warped Test Image"
[test_output]: ./output_images/test1.jpg "Output Test Image"
[output_video]: ./project_video_out.mp4 "Project Video Output"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

This is the Writeup/README and its existence satisfies this criteria. 

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for all things related to the camera (distortion calibration, warping, unwarping) is contained in `python_src/camera_calibration.py`.

To compute the distortion coefficients I first prepare "object points" which are (x, y, z) tuples that represent the chessboard corners in the world.
I assume that the chessboard is fixed in 3D space such that z=0 for all points on each calibration image.
In `CameraCalibration.calibrate_distortion(...)`, `objp` contains these coordinates and is appended onto
`objpoints` each time all the (9, 6) chessboard corners are detected. In addition, `imgpoints` is appended
with the (x,y) pixel position of each corner in the calibration image (if successful detection).

`objpoints` and `imgpoints` are then used with `cv2.calibrateCamera(...)` to compute the distortion
coefficients `mtx` and `dist`. Images are then un-distorted using `cv2.undistort(...)` with the 
distortion coefficients computed in the previous step. An example of (first) a distorted image followed by
an undistorted version of the same image (second) can be seen below:

![A Distorted Calibration Image][cal_distorted]

![An Un-Distorted Calibration Image][cal_undistorted] 

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, an example test image (`test_images/test1.jpg`) can be seen below:
![A Distorted Test Image][test_distorted]

After distortion correction, the following image is produced:
![An Un-Distorted Test Image][test_undistorted]

The change in the image can be seen mostly in the boundaries of the image where the lens distorts the most.

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used x, y, magnitude, and direction thresholds on the Sobel gradients. The code for this step is in 
`python_src/thresholding.py` where the `Thresholding` class implements this code. The thresholds are in
a `Thresholds` dataclass in this same file and seen below:

```python
@dataclass
class Thresholds:
    x_thresh: tuple = (10, 160)
    y_thresh: tuple = (10, 160)
    mag_thresh: tuple = (30, 100)
    dir_thresh: tuple = (0.7, 1.3)
```

This gets passed in to `Thresholding` upon instantiation and the thresholds can be seen in the snippet above.
Given a single-channel input image, Sobel gradients are then calculated and these thresholds are applied to generate
the binary image. The code to combine the 4-different gradients is:

```python
combined[((sobel_x == 1) & (sobel_y == 1) | (sobel_mag == 1) & (sobel_dir == 1))] = 1
```

Applying this technique to different channels revealed that using HLS colorspace (by converting the image using
`cv2.cvt_color(img, cv2.COLOR_RGB2HLS)`) resulted in the best image to use in detection of the lane lines, in particular
the Saturation (S) Channel was much clearer than any other channel in this regard. Below are three binary images that
represent different channels run through this gradient thresholding. First is Gray-Scale, second is the L channel,
and last is the S channel.

![Gray-Scale Binary][gray_binary]

![L-Channel Binary][l_binary]

![S-Channel Binary][s_binary]

It is clear from these images that not only are the lane lines clear in the S binary image, but it removes much more
"noise" from the image than the other two.

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for doing the perspective transform is in the `CameraCalibration` class (as mentioned in #1). In this class,
the methods `warp_image(...)` and `unwarp_image(...)` provide the functionality to convert to/from 
a bird's-eye view perspective. Both methods use the `cv2.warpPerspective(...)` method with the src/dst
values dictated below:

```python
offset = 200
src = np.float32([[218, 720], [588, 455], [698, 455], [1119, 720]])
dst = np.float32([[offset, self._imshape[1]],
                  [offset, 0],
                  [self._imshape[0]-offset, 0],
                  [self._imshape[0]-offset, self._imshape[1]]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 218, 720      | 200, 720      | 
| 588, 455      | 200, 0        |
| 698, 455      | 1080, 0       |
| 1119, 720     | 1080, 720     |

To demonstrate the output of this perspective transform, the following two images show a warped
version of the test image shown earlier (first) and a warped version of the S-channel binary
of the same image (second).

![Warped test image][test_warped]

![Warped S-channel image][s_warped]

Clearly the images show that we are now looking at the lanes from a bird's-eye view
indicating the perspective transform was successful.

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

The code for identifying lane-line pixels and determining polynomial fit is in 
`python_src/line_finder.py` through the `LineFinder` class. This class offers
both the initial sliding-window approach to finding the line pixels but also
the ability to search around a previous polynomial. This dual method strategy allows
us to get an initial polynomial and then help to reduce the probability of 
identifying non-lane line pixels. By searching around the previous polynomial
we can reduce the window size and concentrate on areas we expect the lines to
be present. The `_find_lane_pixels(...)` method does the first method while
the `_search_around_poly(...)` method does the latter. A common entry-point is
the `find_lane_lines(...)` method which switches between the two.

Once the pixel positions have been found, they are fit using the method below
which fits a 2nd-order polynomial of the form x = p[0]y**2 + p[1]y + p[2]
where p represents the polynomial generated. We fit the line vertically as
it varies much more than the lanes do in the horizontal direction.

```python
poly = np.polyfit(y, x, 2)
```

The result of doing this polynomial fit can be seen in the plot below on the 
right. The image on the left is the lane that results from this line finder
warped back to the original perspective.

![Detected lane lines][detected_lines]

It can clearly be seen that the lane has successfully beeen detected and the
polynomial fit is performing well.

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The code to compute the radius of curvature of the lane and the position of the vehicle with respect to center
can be located in `python_src/lane_detector.py` which is the top-level for this project.
The `Line` class here (among other things) computes the radius of curvature using x and y values
produced from a polynomial fit. It calculates its own polynomial fit using x and y values that
are scaled by `xm_per_pix` and `ym_per_pix` respectively. This is to map the lines into real-world
distances. The values I used for these parameters are seen below in units of meters.

```python
xm_per_pix = 30 / 720
ym_per_pix = 3.7 / 847
```

The complete method can is below:

```python
def compute_curvature(self, xvals, yvals):
    fit = np.polyfit(
        yvals * self._ym_per_pix, xvals * self._xm_per_pix, 2)
    y_eval = np.max(yvals)
    curveature = ((1 + (2 * fit[0] * y_eval + fit[1]) ** 2) ** 1.5) /\
        np.absolute(2 * fit[0])
    return curveature
```

This method uses the calculation indicated in the lecture videos as:

```python
r_curve = ((1 + (2*A*y + B)**2)**1.5)/abs(2*A)
```

where `A` and `B` are polynomial coefficients p[0] and p[1] respectively.

For determining the position of the vehicle I first calculate the distance from
image center to the line on each side (at the bottom of the image) (see
`Line._find_line_base_pos(...)`). I then take the average of the two values
(left is negative while right is positive). If the vehicle were centered, the
sum of the two would be zero. If this quantity is negative, the vehicle is left
of center and if positive, it is right of center.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

This code is contained in the `find_lane` method of the `LaneDetector` class.
Once the polynomials have been determined, the constituent x,y values are then
concatenated and passed to `cv2.fillPoly(...)` to draw the polygon. This can be
seen in the code below.

```python
# Draw filled polynomial lane line on image
total_pts = np.concatenate([
    np.array([[a, b] for (a, b) in zip(left_x, self._ploty)], np.int0),
    np.flipud(np.array([[a, b] for (a, b) in zip(right_x, self._ploty)], np.int0))])
cv2.fillPoly(drawn_image, [total_pts], (0, 255, 0))

# Un-warp the drawn image
empty_unwarped = self._camera_cal.unwarp_image(drawn_image)

# Combine the undistorted image with the drawn image
result = cv2.addWeighted(undist, 1, empty_unwarped, 0.5, 0)
```
After the polygon has been drawn, the image is warped back to the original
perspective and overlaid on top of the undistorted test image as in the image
below:

![Output test image][test_output]

This image shows a correct determination of the lane, and also the radius of
curvature and position of the vehicle (although these last two may be a bit
off - getting precise real-world values is tough!).

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result][output_video].

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

One of the main issues I faced in this project was jitter in-between frames of the
video or the lines getting off-track. To solve this issue I introduced some amount
of low-pass filtering to smooth the high-frequency variations. Toward that end,
I first created a deque in Python to hold previously generated x-values with a history
length of 10 frames. At first I would take a mean of the x-values across the pixels
to generate the resultant x-values. However, I changed it to a weighted average
where the most recent frame is weighted higher than the other previous frames.
This provides a good trade-off between low-pass filtering and responding to lane
line changes.

Additionally, the S-channel alone would sometimes not perform well on the white
lane lines and as a result I incorporated the gray-scale channel as well. When
I did this, the result became much smoother. 

There is also a sanity check mechanism that looks at changes in radius of curvature
to say whether the line looks at all similar to the previous lines. This method is hard
to get right and it fails when the lines are very straight (as the radius can
change a lot in these sections since there isn't really a curve). There is perhaps
a better method of checking sanity on these lines like comparing the distance 
between the lines in bird's-eye view which may present a better solution.

My pipeline really struggles on the challenge videos. Likely I need to do a better
job in the selection of gradient thresholds and color values in order to pull the lane
lines out from the road surface. When there is not a high-contrast between
the asphalt/cement and lane line, the pipeline struggles a bit.