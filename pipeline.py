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



def pipeline():

    #* 0. Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
    if not os.path.isfile('undistort_pickle.p'):
        mtx, dist, rvecs, tvec = chessboard_calibration()
        pickle.dump([mtx, dist, rvecs, tvec], open( "undistort_pickle.p", "wb" ))
    else:
        [mtx, dist, rvecs, tvec] = pickle.load(open("undistort_pickle.p", "r"))

    correct_imgs_in_folder(mtx, dist, rvecs, tvec, 'camera_cal')

    #* 1. Apply a distortion correction to raw images.
    correct_imgs_in_folder(mtx, dist, rvecs, tvec, 'test_images')


    #* 2. Use color transforms, gradients, etc., to create a thresholded binary image.
    #' 2_CreateBinaryImage

    #* 3. Apply a perspective transform to rectify binary image ("birds-eye view").
    #' 3_PerspectiveTransform
    #'
    #* Detect lane pixels and fit to find the lane boundary.
    #* Determine the curvature of the lane and vehicle position with respect to center.
    #* Warp the detected lane boundaries back onto the original image.
    #* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.


    return


pipeline()