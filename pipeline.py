import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import pickle
import os

import glob

################# * 0. Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.

def compute_calib_from_chessboards(nx, ny, filename_pattern):
    """
    This function calculates the objectpoints and imagepoints given calibration images containing a chessboard.
    Copied/adapted from: https://github.com/udacity/CarND-Camera-Calibration/blob/master/camera_calibration.ipynb
    :param nx: chessboard dimension in x
    :param ny: chessboard dimension in y
    :param filename_pattern: calibration images to take into account
    :return: camera matrix and distortion coefficients
    """

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((ny * nx, 3), np.float32)
    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane.

    # Make a list of calibration images
    images = glob.glob(filename_pattern)

    print("get_objpoints_imgpoints:filename:" + filename_pattern)

    # Step through the list and search for chessboard corners
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

            # Draw and display the corners
            #cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
            # write_name = 'corners_found'+str(idx)+'.jpg'
            # cv2.imwrite(write_name, img)

            #cv2.imshow('img', img)
            #cv2.show()
            #cv2.waitKey(500)
            #plt.imshow(img)
            #plt.show()
            #plt.waitforbuttonpress()

            #plt.close('all')
        else:
            print("warning: chessboard corners not found in file " + fname)
    #cv2.destroyAllWindows()

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    return mtx, dist, rvecs, tvecs


def chessboard_calibration():

    #correction coefficients
    nx = 9
    ny = 6
    filename_pattern = 'camera_cal/calibration*.jpg'
    mtx, dist, rvecs, tvecs = compute_calib_from_chessboards(nx, ny, filename_pattern)
    return mtx, dist, rvecs, tvecs

def correct_imgs_in_folder(mtx, dist, rvecs, tvec, folder):
    """
    This functions iterates through a folder and undistorts all images into <folder>_undistorted
    :param mtx:
    :param dist:
    :param rvecs:
    :param tvec:
    :param folder: the folder where the images to be undistorted are in
    :return:
    """
    # iterate through all files in the folder and apply the pipeline functions
    for filename in os.listdir(folder + "/"):
        #image = mpimg.imread('camera_cal/' + filename)
        image = cv2.imread(folder + "/" + filename)
        undistorted = cv2.undistort(image, mtx, dist, None, mtx)

        #plt.figure()
        #plt.imshow(final_img)
        #plt.title(filename)
        cv2.imwrite(folder + '_undistorted/' + filename, undistorted)

    return

################### End calibration 0.


###################* 2. Use color transforms, gradients, etc., to create a thresholded binary image.

def white_yellow_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # For HSV, Hue range is [0,179], Saturation range is [0,255] and Value range is [0,255].

    white_mask = cv2.inRange(hsv, np.array([0, 0, 150]), np.array([179, 25, 255]))
    white_image = cv2.bitwise_and(img, img, mask=white_mask)

    yellow_mask = cv2.inRange(hsv, np.array([90, 120, 0]), np.array([120, 255, 255]))
    yellow_image = cv2.bitwise_and(img, img, mask=yellow_mask)

    #combined_mask = cv2.bitwise_or(yellow_mask, white_mask);

    final_image = cv2.add(white_image, yellow_image)

    return final_image

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.
    """

    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)

    return masked_image


def cut_area(img):
    """
    Makes black everything laying outside of the desired area
    """
    # pts – Array of polygons where each polygon is represented as an array of points.
    vertices = np.array([[(100, 700), (650, 400), (1200, 700)]], dtype=np.int32)
    masked_image = region_of_interest(img, vertices)

    return masked_image


def create_binary_image(initial_image):
    #plt.imshow(initial_image)
    #plt.title('original image')

    #white_masked = initial_image
    white_yellow_masked = white_yellow_mask(initial_image)
    #plt.figure()
    #plt.imshow(white_masked)
    #plt.title('white mask')

    gray_image = grayscale(white_yellow_masked)
    #plt.figure()
    #plt.imshow(gray_image, cmap='gray')
    #plt.title('grayscale')

    #blurred_image = gaussian_blur(gray_image, 5)
    #plt.figure()
    #plt.imshow(blurred_image, cmap='gray')
    #plt.title('gaussian_blur')

    #canny_image = canny(blurred_image, 50, 150)
    # plt.figure()
    # plt.imshow(canny_image, cmap='gray')
    # plt.title('canny')

    cut_image = cut_area(gray_image)
    # plt.figure()
    # plt.imshow(cut_image, cmap='gray')
    # plt.title('cut image')

    #hough_image, lines = hough_trans(canny_image)
    # plt.figure()
    # plt.imshow(hough_image)
    # plt.title('hough image')

    s_thresh_min = 100
    s_thresh_max = 255
    s_binary = np.zeros_like(cut_image)
    s_binary[(cut_image >= s_thresh_min) & (cut_image <= s_thresh_max)] = 255

    return s_binary

################### End thresholded binary image


################### 3. Apply a perspective transform to rectify binary image ("birds-eye view").


def determine_perspective_transform_matrix():

    #img = mpimg.imread("test_images_undistorted/straight_lines1.jpg")
    #plt.imshow(img)
    #plt.show()

    #img_size = (img.shape[1], img.shape[0])

    # For source points I'm grabbing the outer four detected corners

    #src = np.float32([corners[0], corners[nx - 1], corners[-1], corners[-nx]])

    #src = np.float32([[203,719],[537,491], [1091, 717], [749, 492]])
    #dst = np.float32([[203,719],[203,191], [1091, 717], [1091, 192]])

    src = np.float32([[203, 719], [537, 491], [1091, 717], [749, 492]])
    dst = np.float32([[203, 719], [203, 191], [1091, 717], [1091, 192]])

    # For destination points, I'm arbitrarily choosing some points to be
    # a nice fit for displaying our warped result
    # again, not exact, but close enough for our purposes

    #dst = np.float32([[offset, offset], [img_size[0] - offset, offset], [img_size[0] - offset, img_size[1] - offset], [offset, img_size[1] - offset]])

    # Given src and dst points, calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    # Warp the image using OpenCV warpPerspective()
    #warped = cv2.warpPerspective(img, M, img_size)

    #plt.imshow(warped)
    #plt.show()

    return M

def perspective_transform(img):

    if not os.path.isfile('M_pickle.p'):
        M = determine_perspective_transform_matrix()
        pickle.dump(M, open( "M_pickle.p", "wb" ))
    else:
        M = pickle.load(open("M_pickle.p", "rb"))

    img_size = (img.shape[1], img.shape[0])
    warped = cv2.warpPerspective(img, M, img_size)

    return warped, M

################## End perspective transform

#################* 4. Detect lane pixels and fit to find the lane boundary.

def find_lane_pixels(binary_warped):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] // 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0] // nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
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
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low, win_y_low),
                      (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low),
                      (win_xright_high, win_y_high), (0, 255, 0), 2)

        # Identify the nonzero pixels in x and y within the window #
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, out_img


def fit_polynomial(leftx, lefty, rightx, righty, out_img):
    # Find our lane pixels first


    # Fit a second order polynomial to each using `np.polyfit`
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, out_img.shape[0] - 1, out_img.shape[0])


    try:
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1 * ploty ** 2 + 1 * ploty
        right_fitx = 1 * ploty ** 2 + 1 * ploty

    ## Visualization ##
    # Colors in the left and right lane regions
    out_img[lefty, leftx] = [255, 255, 255]
    out_img[righty, rightx] = [255, 255, 255]
    ploty_int = ploty.astype(int)
    left_fitx_int = left_fitx.astype(int)

    #if all(0 <= left_fitx < out_img.shape[1] for item in left_fitx):

    if not any(0 <= left_fitx) and not any( left_fitx > out_img.shape[1]):
        out_img[ploty.astype(int), left_fitx.astype(int)-1] = [255,0,0]
        out_img[ploty.astype(int), left_fitx.astype(int)] = [255, 0, 0]
        out_img[ploty.astype(int), left_fitx.astype(int)+1] = [255, 0, 0]

    #if all(0 <= right_fitx < out_img.shape[1] for item in right_fitx):

    if not any(0 <= right_fitx) and not any( right_fitx > out_img.shape[1]):
        out_img[ploty.astype(int), right_fitx.astype(int)-1] = [255,0,0]
        out_img[ploty.astype(int), right_fitx.astype(int)] = [255,0,0]
        out_img[ploty.astype(int), right_fitx.astype(int)+1] = [255,0,0]

    # Plots the left and right polynomials on the lane lines
    #plt.plot(left_fitx, ploty, color='yellow')
    #plt.plot(right_fitx, ploty, color='yellow')

    return out_img, ploty, left_fit, right_fit, left_fitx, right_fitx


################ End detect lane pixels

################* 5. Determine the curvature of the lane and vehicle position with respect to center.

def measure_curvature_pixels(ploty, left_fit, right_fit, xm_per_pix, ym_per_pix):
    '''
    Calculates the curvature of polynomial functions in pixels.
    '''
    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)

    # Calculation of R_curve (radius of curvature)
    left_curverad = ((1 + (2 * left_fit[0] * y_eval + left_fit[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit[0])
    right_curverad = ((1 + (2 * right_fit[0] * y_eval + right_fit[1]) ** 2) ** 1.5) / np.absolute(2 * right_fit[0])

    return left_curverad, right_curverad


def measure_curvature_real(leftx, lefty, rightx, righty):
    '''
    Calculates the curvature of polynomial functions in meters.
    '''
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

    left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)

    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(lefty)

    # Calculation of R_curve (radius of curvature)
    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * right_fit_cr[0])

    return left_curverad, right_curverad

################ End Determination of curvature

################ 6. Warp the detected lane boundaries back onto the original image.

def warp_back_to_original(warped, left_fitx, right_fitx, ploty, original_img, Minv):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (original_img.shape[1], original_img.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(original_img, 1, newwarp, 0.3, 0)
    #plt.imshow(result)

    return result

############### End Warp detected lane boundaries back

# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        #average x values of the fitted line over the last n iterations
        self.bestx = None
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        #radius of curvature of the line in some units
        self.radius_of_curvature = None
        #distance in meters of vehicle center from the line
        self.line_base_pos = None
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float')
        #x values for detected line pixels
        self.allx = None
        #y values for detected line pixels
        self.ally = None

class LaneDetectionPipeline():
    def __init__(self, on_video, show_imgs):
        self.on_video = on_video
        self.show_imgs = show_imgs

    def pipeline(self, img):

        #* 0. Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
        if not os.path.isfile('undistort_pickle.p'):
            mtx, dist, rvecs, tvec = chessboard_calibration()
            pickle.dump([mtx, dist, rvecs, tvec], open( "undistort_pickle.p", "wb" ))
        else:
            mtx, dist, rvecs, tvec = pickle.load(open("undistort_pickle.p", "rb"))
            correct_imgs_in_folder(mtx, dist, rvecs, tvec, 'camera_cal')

        #* 1. Apply a distortion correction to raw image
        undistorted_img = cv2.undistort(img, mtx, dist, None, mtx)

        #* 2. Use color transforms, gradients, etc., to create a thresholded binary image.
        binary_img = create_binary_image(undistorted_img)

        #* 3. Apply a perspective transform to rectify binary image ("birds-eye view").
        perspective_trans_img, M = perspective_transform(binary_img)

        #* 4. Detect lane pixels and fit to find the lane boundary.
        leftx, lefty, rightx, righty, marked_img = find_lane_pixels(perspective_trans_img)

        #throw exception if not found
        warped_img_with_lanes, ploty, left_fit, right_fit, left_fitx, right_fitx = fit_polynomial(leftx, lefty, rightx, righty, marked_img)

        #* 5. Determine the curvature of the lane and vehicle position with respect to center.
        #left_curverad, right_curverad = measure_curvature_pixels(ploty, left_fit, right_fit)
        left_curverad, right_curverad = measure_curvature_real(leftx, lefty, rightx, righty)

        Minv = np.linalg.inv(M)

        #* 6. Warp the detected lane boundaries back onto the original image.
        final_img = warp_back_to_original(perspective_trans_img, left_fitx, right_fitx, ploty, undistorted_img, Minv)

        #* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.
        print("Left curverad: {}, right curverad: {}".format(left_curverad, right_curverad))

        if self.show_imgs:

            f = plt.figure(figsize=(18, 7))
            plt.tight_layout()

            p1 = plt.subplot(2, 3, 1)
            p1.imshow(img)
            p1.set_title(('Original Image'))

            p2 = plt.subplot(2, 3, 2)
            p2.imshow(undistorted_img)
            p2.set_title(('Undistorted Image'))

            p2 = plt.subplot(2, 3, 3)
            p2.imshow(binary_img, cmap='gray')
            p2.set_title(('Binary Image'))

            p2 = plt.subplot(2, 3, 4)
            p2.imshow(perspective_trans_img, cmap='gray')
            p2.set_title(('Perspective Transform'))

            p2 = plt.subplot(2, 3, 5)
            p2.imshow(warped_img_with_lanes)
            p2.set_title(('Detected Lane Pixels'))

            p2 = plt.subplot(2, 3, 6)
            p2.imshow(final_img)
            p2.set_title(('Final Image'))

            plt.subplots_adjust(left=0, right=1, top=0.9, bottom=0)
            plt.show()

        return final_img

def pipeline_on_images():

    p = LaneDetectionPipeline(False, True)

    for filename in os.listdir("test_images/"):

        #filename = "straight_lines1.jpg"
        print(filename)
        #image = mpimg.imread('camera_cal/' + filename)
        image = mpimg.imread("test_images/" + filename)
        final_image = p.pipeline(image)
        cv2.imwrite('output_images/' + filename, final_image)

    return


def pipeline_on_video():
    # Import everything needed to edit/save/watch video clips
    from moviepy.editor import VideoFileClip
    #from IPython.display import HTML

    p = LaneDetectionPipeline(True, False)

    def process_image(image):
        # NOTE: The output you return should be a color image (3 channel) for processing video below
        # TODO: put your pipeline here,
        # you should return the final output (image where lines are drawn on lanes)
        result = p.pipeline(image)

        return result

    white_output = 'project_video_output.mp4'
    ## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
    ## To do so add .subclip(start_second,end_second) to the end of the line below
    ## Where start_second and end_second are integer values representing the start and end of the subclip
    ## You may also uncomment the following line for a subclip of the first 5 seconds
    ##clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
    clip1 = VideoFileClip("project_video.mp4")
    white_clip = clip1.fl_image(process_image)  # NOTE: this function expects color images!!
    #% time white_clip.write_videofile(white_output, audio=False)
    white_clip.write_videofile(white_output, audio=False)

    return

pipeline_on_video()
#pipeline_on_images()
#determine_perspective_transform_matrix()