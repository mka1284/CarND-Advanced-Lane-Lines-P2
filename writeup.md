##Advanced Lane Finding Project

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

[image1]: file://camera_cal_undistorted/calibration1.jpg "Undistorted Chessboard"
[image2]: file://test_images/straight_lines1.jpg "Road Distorted"
[image3]: file://test_images_undistorted/straight_lines1.jpg "Road Undistorted"
[image4]: file://output_images/straight_lines1_binary2.png "Straight Lines Binary"
[image5]: file://output_images/straight_lines1_transformed2.png "Perspective Transform"
[image6]: file://output_images/straight_lines1_polynoms2.png "Fitted Polynom"
[image7]: file://output_images/straight_lines1.jpg "Detected Lanes" 
[video1]: file://project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points


### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

First, the chessboard corners are found using the opencv function *findChessboardCorners()*. Then,
the camera is calibrated by calling *calibrateCamera()* calculating the distortion coefficients and 
camera matrix.


The code for this step is contained in file advll_helpers.py, functions *chessboad_calibration()* and
*compute_calib_from_chessboards()*.  


I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![Undistorted Chessboard][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![Road Distorted][image2]

The result of the undistortion process can be found here:

![Road Undistorted][image3]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
All functions mentioned in the following are implemented in advll_helpers.py.

First, I mask the image, and I filter out all parts that are neither white nor yellow, applying thresholds in HSV color space, which is done by the function white_yellow_mask. Then, I transform the image to grayscale, cut out the area defining a triangle, implemented in function grayscale. Finally, the image is converted into a binary image.


![Straight Lines Binary][image4]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.


The code for my perspective transform includes a function called `perspective_transform()`, which appears in lines 285 through 304 in the file `advll_helpers.py`. For determination of the perspective transform matrix, I implemented the function determine_transform_matrix() with hard-coded source and destination points.


I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image:

![Perspective Transform][image5]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial.

The identification of the lane-line pixels is done in find_lane_pixels(), lines 310 to 400.
As an input, the function gets the binary image. Then, for each lane, a histogram is created. The height of the image is then split up into windows with 1/9 of the total height. The peak of the initial histogram is then taken as the y-position of the first window. For each window, if more then 50 white pixels are found, these pixels are added to the ones counting as the respective lane, and the average y value of these pixels is the y-position of the following window.

For each lane, the positions of the identified pixels is passed to the function polyfit(), which fits a polynomial of order 2 to the pixels. An example result can be seen here:

![Fitted Polynom][image6]

To become more robust, detection data is averaged, and only data which is within certain boundaries is accepted.
The filtering, i.e. the rejection of outliers, is done by the following filter criteria:

1. The distance between both lines close to the vehicle and at the furthest point taken into account needs to be between 900 +/- 200 pixels.
2. The difference between the x value of the lines (left and right, closest and furthest to the vehicle) and the averaged values need to be below 80 pixels
3. The quotient of the current curvature of the left curve and the the averaged value over the last 20 values must be between 1/4 and 4


The polynoms are calculated based on the average of the last 20 coefficient values that have passed the test.


If there are less than 20 coefficient values available, filtering is only done by (1). All other measurement values are added to the history.


#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The radius is calculated in function measure_curvature_pixels(), at the position closest to the vehicle,
based on the coefficients of the polygon. 

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines 536 through 568 in the function `draw_lane_and_warp_back_to_original()`.  Here is an example of my result on a test image:

![Detected Lanes][image7]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
