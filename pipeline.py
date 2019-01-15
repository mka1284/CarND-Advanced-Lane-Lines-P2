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

    img = mpimg.imread("test_images/straight_lines1.jpg")
    plt.imshow(img)
    plt.show()

    img_size = (img.shape[1], img.shape[0])

    # For source points I'm grabbing the outer four detected corners

    #src = np.float32([corners[0], corners[nx - 1], corners[-1], corners[-nx]])

    src = np.float32([[203,719],[537,491], [1091, 717], [749, 492]])
    dst = np.float32([[203,719],[203,491], [1091, 717], [1091, 492]])

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

    return warped

################## End perspective transform





def pipeline(img, filename):

    #* 0. Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
    if not os.path.isfile('undistort_pickle.p'):
        mtx, dist, rvecs, tvec = chessboard_calibration()
        pickle.dump([mtx, dist, rvecs, tvec], open( "undistort_pickle.p", "wb" ))
    else:
        mtx, dist, rvecs, tvec = pickle.load(open("undistort_pickle.p", "rb"))
        correct_imgs_in_folder(mtx, dist, rvecs, tvec, 'camera_cal')

    #* 1. Apply a distortion correction to raw image
    undistorted = cv2.undistort(img, mtx, dist, None, mtx)

    #* 2. Use color transforms, gradients, etc., to create a thresholded binary image.
    binary_img = create_binary_image(undistorted)

    #* 3. Apply a perspective transform to rectify binary image ("birds-eye view").
    output_img = perspective_transform(binary_img)

    #* 4. Detect lane pixels and fit to find the lane boundary.


    #* Determine the curvature of the lane and vehicle position with respect to center.
    #* Warp the detected lane boundaries back onto the original image.
    #* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

    f = plt.figure(figsize=(18, 7))
    plt.tight_layout()

    p1 = plt.subplot(2, 3, 1)
    p1.imshow(img)
    p1.set_title(('Original Image'))

    p2 = plt.subplot(2, 3, 2)
    p2.imshow(undistorted)
    p2.set_title(('Undistorted Image'))

    p2 = plt.subplot(2, 3, 3)
    p2.imshow(binary_img, cmap='gray')
    p2.set_title(('Binary Image'))

    p2 = plt.subplot(2, 3, 4)
    p2.imshow(output_img, cmap='gray')
    p2.set_title(('Output Image'))

    plt.subplots_adjust(left=0, right=1, top=0.9, bottom=0)
    plt.show()

    return output_img


def pipeline_on_images():

    for filename in os.listdir("test_images/"):

        #filename = "straight_lines1.jpg"

        #image = mpimg.imread('camera_cal/' + filename)
        image = mpimg.imread("test_images/" + filename)
        final_image = pipeline(image, filename)
        cv2.imwrite('output_images/' + filename, final_image)

    return


pipeline_on_images()