import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import pickle
import os

import glob


########## Helper functions

###############  Calibration / Undistortion

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

        else:
            print("warning: chessboard corners not found in file " + fname)

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

        cv2.imwrite(folder + '_undistorted/' + filename, undistorted)

    return



###############  Creation of Binary Image

def white_yellow_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # For HSV, Hue range is [0,179], Saturation range is [0,255] and Value range is [0,255].

    white_mask = cv2.inRange(hsv, np.array([0, 0, 150]), np.array([179, 25, 255]))
    white_image = cv2.bitwise_and(img, img, mask=white_mask)

    #yellow_mask = cv2.inRange(hsv, np.array([90, 120, 0]), np.array([120, 255, 255]))

    yellow_mask = cv2.inRange(hsv, np.array([90, 100, 0]), np.array([120, 255, 255]))
    yellow_image = cv2.bitwise_and(img, img, mask=yellow_mask)

    #combined_mask = cv2.bitwise_or(yellow_mask, white_mask);

    final_image = cv2.add(white_image, yellow_image)

    return final_image

def grayscale(img):
    """
    Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')
    """

    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()

def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

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
    Makes black everything outside of the area of interest.

    :param img: the image that will be cut
    :return: the cut image
    """


    vertices = np.array([[(100, 700), (650, 400), (1200, 700)]], dtype=np.int32)
    masked_image = region_of_interest(img, vertices)

    return masked_image


def create_binary_image(initial_image):
    """
    Applies various filters on the given image
    :param initial_image: the image to apply the filters on
    :return: the filtered images
    """

    white_yellow_masked = white_yellow_mask(initial_image)
    gray_image = grayscale(white_yellow_masked)
    cut_image = cut_area(gray_image)


    s_thresh_min = 100
    s_thresh_max = 255
    binary_img = np.zeros_like(cut_image)
    binary_img[(cut_image >= s_thresh_min) & (cut_image <= s_thresh_max)] = 255

    plt.figure()
    plt.imshow(binary_img, cmap='gray')
    plt.title('binary_img')

    return white_yellow_masked, gray_image, gray_image, gray_image, cut_image, binary_img



################### Perspective Transform

def determine_perspective_transform_matrix():
    """
    Determines the perspective transform matrix.

    Figures with the original and transformed image are opened in order
    to easily change the transformation parameters.

    :return: The perspective transformation matrix
    """

    img = mpimg.imread("test_images_undistorted/straight_lines1.jpg")
    plt.imshow(img)
    plt.show()

    img_size = (img.shape[1], img.shape[0])

    src = np.float32([[203, 719], [580, 460], [1091, 717], [702, 460]])
    dst = np.float32([[203, 719], [203, 100], [1091, 717], [1091, 100]])

    # Given src and dst points, calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, img_size)

    #cv2.imwrite('output_images/straight_lines1_transformed2.png', warped)

    plt.imshow(warped)
    plt.show()

    return M

def perspective_transform(img):
    """
    Applies the perspective transformation to the image. If the pickle file
    does not exist, the transformation is determined first and saved in the
    pickle file.

    :param img: The image to be transformed
    :return: The warped/transformed image and the transformation Matrix
    """

    if not os.path.isfile('M_pickle.p'):
        M = determine_perspective_transform_matrix()
        pickle.dump(M, open( "M_pickle.p", "wb" ))
    else:
        M = pickle.load(open("M_pickle.p", "rb"))

    img_size = (img.shape[1], img.shape[0])
    warped = cv2.warpPerspective(img, M, img_size)

    return warped, M



################# Detection of Lane Pixels And Polygon Generation

def find_lane_pixels(binary_warped):
    """
    Find the pixels that are part of the image.

    :param binary_warped: the binary warped image
    :return: The x and y coordinates of the lane pixels and a visualization image
    """

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


def generate_polygon_lines(left_fit_coeffs, right_fit_coeffs, perspective_trans_img):
    """
    Generates the polygon lines based on the polynom coefficients

    :param left_fit_coeffs:
    :param right_fit_coeffs:
    :param perspective_trans_img:
    :return: polygon line coefficients
    """

    poly_y = np.linspace(0, perspective_trans_img.shape[0] - 1, perspective_trans_img.shape[0])

    try:
        poly_left_x = left_fit_coeffs[0] * poly_y ** 2 + left_fit_coeffs[1] * poly_y + left_fit_coeffs[2]
        poly_right_x = right_fit_coeffs[0] * poly_y ** 2 + right_fit_coeffs[1] * poly_y + right_fit_coeffs[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        poly_left_x = 1 * poly_y ** 2 + 1 * poly_y
        poly_right_x = 1 * poly_y ** 2 + 1 * poly_y


    return poly_y, poly_left_x, poly_right_x


def plot_polygon_lines(poly_y, poly_left_x, poly_right_x, out_img, color=[255,0,0]):
    """
    Plots the polygon lines on the image.

    :param poly_y:
    :param poly_left_x:
    :param poly_right_x:
    :param out_img:
    :param color:
    :return: The image with the lines drawn on it
    """

    if not any(0 > poly_left_x) and not any( poly_left_x > out_img.shape[1]-1):
        out_img[poly_y.astype(int), poly_left_x.astype(int)-1] = color
        out_img[poly_y.astype(int), poly_left_x.astype(int)] = color
        out_img[poly_y.astype(int), poly_left_x.astype(int)+1] = color

    if not any(0 > poly_right_x) and not any( poly_right_x > out_img.shape[1]-1):
        out_img[poly_y.astype(int), poly_right_x.astype(int)-1] = color
        out_img[poly_y.astype(int), poly_right_x.astype(int)] = color
        out_img[poly_y.astype(int), poly_right_x.astype(int)+1] = color

    return out_img


################ Curvature And Vehicle Position Determination

def measure_curvature_pixels(ploty, left_fit, right_fit):
    """
    Calculate the curvature of lanes in pixels

    :param ploty: the y values of the polynom
    :param left_fit: the polynom coefficients of the left polynom
    :param right_fit: the polynom coefficients of the right polynom
    :param xm_per_pix: the density of pixels in x direction
    :param ym_per_pix: the density of pixels in the y direction
    :return: the curvature of both lanes
    """

    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)

    # Calculation of R_curve (radius of curvature)
    left_curverad = ((1 + (2 * left_fit[0] * y_eval + left_fit[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit[0])
    right_curverad = ((1 + (2 * right_fit[0] * y_eval + right_fit[1]) ** 2) ** 1.5) / np.absolute(2 * right_fit[0])

    return left_curverad, right_curverad

def measure_curvature_real(leftx, lefty, rightx, righty):
    """
    Calculate the curvature of lanes in meters

    :param leftx: the x coordinates of the left lane
    :param lefty: the y coordinates of the left lane
    :param rightx: the x coordinates of the right lane
    :param righty: the y coordinates of teh right lane
    :return: the radians of the lanes in meters
    """

    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    #xm_per_pix = 3.7 / 700  # meters per pixel in x dimension
    xm_per_pix = 1.33 / 700  # meters per pixel in x dimension

    try:
        left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
        right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
    except TypeError:
        return -1,-1

    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(lefty)

    # Calculation of R_curve (radius of curvature)
    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * right_fit_cr[0])

    return left_curverad, right_curverad


def determine_vehicle_pos(left_line_start, right_line_start):
    """
    Calculates the deviation of the vehicle center in meters.

    :param left_line_start: The current position of the left lane
    :param right_line_start: The current position of the right lane
    :return: The vehicle position relative to the lines
    """

    LEFT_LINE_ZERO = 203
    RIGHT_LINE_ZERO = 1091
    PIXELS2METERS = 1.33 / 700

    ideal_pos = LEFT_LINE_ZERO + (RIGHT_LINE_ZERO - LEFT_LINE_ZERO)/2
    true_pos = left_line_start + (right_line_start - left_line_start)/2

    diff_pix = ideal_pos - true_pos
    diff_meters = diff_pix * PIXELS2METERS

    return diff_meters


################ Unwarping

def draw_lane_and_warp_back_to_original(warped, left_fitx, right_fitx, ploty, original_img, Minv):
    """
    Draw the drivable area on the image

    :param warped: the warped image
    :param left_fitx: the x coordinates of the left lane pixels
    :param right_fitx: the x coordinates of the right lane pixels
    :param ploty: the y coordinates of the lane pixels
    :param original_img: the original image
    :param Minv: the inverse matrix for warping
    :return: the combined, unwarped image
    """

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

def print_info_on_img(logfile, img, left_curve, right_curve, vehicle_pos):
    """
    Print the info on the image and also writes it to a logfile

    :param logfile:
    :param img:
    :param left_curve:
    :param right_curve:
    :param vehicle_pos:
    :return:
    """

    if (left_curve < 10000 or right_curve < 10000):
        avg_curve = (left_curve + right_curve) / 2
        strToPlot = "Curve: {0:.2f}, x-Pos: {1:.3f}".format(avg_curve, vehicle_pos)
    else:
        strToPlot = "Curve: infinite, x-Pos: {0:.3f}".format(vehicle_pos)

    #### write curvature on image

    logfile.write(strToPlot + "\n")
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (50, 50)
    fontScale = 1
    fontColor = (255, 255, 255)
    lineType = 2

    cv2.putText(img, strToPlot,
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                lineType)

    return img

def show_imgs(original_img, undistorted_img, white_yellow_img, canny_img, binary_img, debug_img, final_img):
    """
    Show the given images in a figure.

    :param original_img:
    :param undistorted_img:
    :param white_yellow_img:
    :param canny_img:
    :param binary_img:
    :param debug_img:
    :param final_img:
    :return:
    """

    f = plt.figure(figsize=(19, 8))
    plt.tight_layout()

    p1 = plt.subplot(2, 4, 1)
    p1.imshow(original_img)
    p1.set_title(('Original Image'))

    p2 = plt.subplot(2, 4, 2)
    p2.imshow(undistorted_img)
    p2.set_title(('Undistorted Image'))

    p2 = plt.subplot(2, 4, 3)
    p2.imshow(white_yellow_img)
    p2.set_title(('White-Yellow Image'))

    p2 = plt.subplot(2, 4, 4)
    p2.imshow(canny_img)
    p2.set_title(('Canny-Image'))

    p2 = plt.subplot(2, 4, 5)
    p2.imshow(binary_img, cmap='gray')
    p2.set_title(('Binary Image'))

    p2 = plt.subplot(2, 4, 6)
    p2.imshow(debug_img)
    p2.set_title(('Detected Lane Pixels'))

    cv2.imwrite('output_images/straight_lines1_polynoms2.png', debug_img)

    p2 = plt.subplot(2, 4, 7)
    p2.imshow(final_img)
    p2.set_title(('Final Image'))

    plt.subplots_adjust(left=0, right=1, top=0.9, bottom=0)
    plt.show()

