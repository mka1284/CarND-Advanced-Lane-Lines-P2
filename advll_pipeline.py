import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import pickle
import os

import advll_helpers

#A class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        self.x = [np.array([False])] #The x-values of the pixels of the lane
        self.coeffs = None #The polynomial coefficients of the lane
        self.curverad = None #The radius of lane in meters
        #self.logfile = None

class LaneDetectionPipeline():
    """
    This class represents the pipeline with averaging/filtering capabilities.
    """

    def __init__(self, on_video, show_intermediate_imgs, attach_polynom_img):
        self.on_video = on_video #Whether the pipeline is running on a video
        self.show_imgs = show_intermediate_imgs #Whether debug images afer each step are shown
        self.attach_polynom_img = attach_polynom_img #Whether the polynom debug image should be attached to the output image
        self.left_line = Line()
        self.right_line = Line()
        self.MAX_DEV_X_STEP = 150 #The maximum deviation of the x position of the lane from the average
        #self.MAX_DEV_CURVE_STEP = 100000
        self.MAX_DEV_CURVE_QUOT = 3 #The maximum deviation of the quotient of the current curve from the average
        self.MAX_CURVERAD = 10000 #The maximum curve radius

        self.MAX_HIST_LEN = 10 #The length of the detection history
        self.hist_len = 0 #The current length of the history

        self.EXPECTED_LINE_DIST = 900 #The expected distance of the lanes
        self.LINE_DIST_TOL = 200 #The tolerance of the lane distance

        self.MAX_SUBSEQ_NOT_DETECT = 10 #The number of times a detection may fail
        self.subseq_not_detect = 0 #The number of times a detection has failed

        self.logfile = open("logs.txt", "w")

    def log(self, text):
        print(text)
        self.logfile.write(text + "\n")


    def add_to_list_and_avg(self, line, fit_coeffs, poly_x, curverad):
        """
        Add the data to the saved average.

        :param line:
        :param fit_coeffs:
        :param poly_x:
        :param curverad:
        :return:
        """

        if self.hist_len == 0:
            line.x = poly_x
            line.coeffs = fit_coeffs
            line.curverad = curverad

        elif self.hist_len < self.MAX_HIST_LEN:
            # add current detection to line class
            line.x = (line.x * self.hist_len + poly_x) / (self.hist_len + 1)
            line.coeffs = (line.coeffs * self.hist_len + fit_coeffs) / (self.hist_len + 1)
            line.curverad = (line.curverad * self.hist_len + curverad) / (self.hist_len + 1)

        else:
            line.coeffs = (line.coeffs * (self.hist_len - 1) + fit_coeffs) / self.hist_len
            line.x = (line.x * (self.hist_len - 1) + poly_x) / self.hist_len
            line.curverad = (line.curverad * (self.hist_len-1) + curverad) / self.hist_len


    def check_data(self, line_dist_abs_begin, line_dist_abs_end, line_pos_begin_left_delta, line_pos_end_left_delta,
                   line_pos_begin_right_delta, line_pos_end_right_delta, curverad_delta_left_quot, curverad_delta_right_quot):
        """
        Check whether the submitted frame is usable.

        :param line_dist_abs_begin: The distance between both detected lines in pixels right at the vehicle
        :param line_dist_abs_end:  The distance between both detected lines in pixels at most distant point from the vehicle
        :param line_pos_begin_left_delta: The difference between the x value of the left line closest to the vehicle and the averaged value
        :param line_pos_end_left_delta: The difference between the x value of the left line closest to the vehicle and the averaged value
        :param line_pos_begin_right_delta: The difference between the x value of the right line closest to the vehicle and the averaged value
        :param line_pos_end_right_delta: The difference between the x value of the right line closest to the vehicle and the averaged value
        :param curverad_delta_left_quot: The quotient of the current curvature of the left curve and the averaged value
        :param curverad_delta_right_quot: The quotient of the current curvature of the right curve and the averaged value
        :return:
        """

        if (self.EXPECTED_LINE_DIST - self.LINE_DIST_TOL) < line_dist_abs_begin < (
                self.EXPECTED_LINE_DIST + self.LINE_DIST_TOL) \
                and (self.EXPECTED_LINE_DIST - self.LINE_DIST_TOL) < line_dist_abs_end < (
                self.EXPECTED_LINE_DIST + self.LINE_DIST_TOL) \
                and line_pos_begin_left_delta < self.MAX_DEV_X_STEP and line_pos_end_left_delta < self.MAX_DEV_X_STEP \
                and line_pos_begin_right_delta < self.MAX_DEV_X_STEP and line_pos_end_right_delta < self.MAX_DEV_X_STEP \
                and 1 / self.MAX_DEV_CURVE_QUOT < curverad_delta_left_quot < self.MAX_DEV_CURVE_QUOT \
                and 1 / self.MAX_DEV_CURVE_QUOT < curverad_delta_right_quot < self.MAX_DEV_CURVE_QUOT:
            # and curverad_delta_left < self.MAX_DEV_CURVE \
            # and curverad_delta_right < self.MAX_DEV_CURVE:

            self.subseq_not_detect = 0

            return True

        elif (not (self.EXPECTED_LINE_DIST - self.LINE_DIST_TOL) < line_dist_abs_begin < (
                self.EXPECTED_LINE_DIST + self.LINE_DIST_TOL)):
            self.log("line_dist_abs_begin({}) out of range: must be between {} and {}".format(line_dist_abs_begin,
                                                                                              self.EXPECTED_LINE_DIST - self.LINE_DIST_TOL,
                                                                                              self.EXPECTED_LINE_DIST + self.LINE_DIST_TOL))

        elif (not (self.EXPECTED_LINE_DIST - self.LINE_DIST_TOL) < line_dist_abs_end < (
                self.EXPECTED_LINE_DIST + self.LINE_DIST_TOL)):
            self.log("line_dist_abs_end({}) out of range: must be between {} and {}".format(line_dist_abs_end,
                                                                                            self.EXPECTED_LINE_DIST - self.LINE_DIST_TOL,
                                                                                            self.EXPECTED_LINE_DIST + self.LINE_DIST_TOL))

        elif (not (line_pos_begin_left_delta < self.MAX_DEV_X_STEP and line_pos_end_left_delta < self.MAX_DEV_X_STEP)):
            self.log("line_pos_begin_left_delta({}) or end({}) out of range: must be below {}".format(
                line_pos_begin_left_delta, line_pos_end_left_delta, self.MAX_DEV_X_STEP))

        elif (not (line_pos_begin_right_delta < self.MAX_DEV_X_STEP and line_pos_end_right_delta < self.MAX_DEV_X_STEP)):
            self.log("line_pos_begin_right_delta({}) or end({}) out of range: must be below {}".format(
                line_pos_begin_right_delta, line_pos_end_right_delta, self.MAX_DEV_X_STEP))

        elif (not (1 / self.MAX_DEV_CURVE_QUOT < curverad_delta_left_quot < self.MAX_DEV_CURVE_QUOT)):
            self.log(
                "curverad_delta_left_quot({}) out of range: must be between {} and {}".format(curverad_delta_left_quot,
                                                                                              1 / self.MAX_DEV_CURVE_QUOT,
                                                                                              self.MAX_DEV_CURVE_QUOT))

        elif (not (1 / self.MAX_DEV_CURVE_QUOT < curverad_delta_right_quot < self.MAX_DEV_CURVE_QUOT)):
            self.log("curverad_delta_right_quot({}) out of range: must be between {} and {}".format(
                curverad_delta_right_quot, 1 / self.MAX_DEV_CURVE_QUOT, self.MAX_DEV_CURVE_QUOT))

        else:
            self.log("Checkdata failed for some other reason")

        return False

    def process_new_line_data(self, left_fit_coeffs, right_fit_coeffs, poly_left_x, poly_right_x, left_pix_x, left_pix_y, right_pix_x, right_pix_y):
        """
        Process the new frame.

        :param left_fit_coeffs: the coefficients of the left polynom
        :param right_fit_coeffs: the coefficients of the right polynom
        :param poly_left_x: the x coordinates of the left polynom
        :param poly_right_x: the x coordinates of the right polynom
        :param left_pix_x: the x coordinates of the pixels of the left lane
        :param left_pix_y: the y coordinates of the pixels of the left lane
        :param right_pix_x: the x coordinates of the pixels of the right lane
        :param right_pix_y: the y coordinates of the pixels of the right lane
        :return: radius and x-coordinates of the lanes
        """

        left_curverad, right_curverad = advll_helpers.measure_curvature_real(left_pix_x, left_pix_y, right_pix_x, right_pix_y)

        if(left_curverad > self.MAX_CURVERAD):
            left_curverad = self.MAX_CURVERAD

        if(right_curverad > self.MAX_CURVERAD):
            right_curverad = self.MAX_CURVERAD

        line_dist_abs_begin = abs(poly_left_x[0] - poly_right_x[0])
        line_dist_abs_end = abs(poly_left_x[len(poly_right_x) - 1] - poly_right_x[len(poly_right_x) - 1])

        if(not self.on_video):
            print(
                "Line dist begin:{}, end:{}".format(line_dist_abs_begin, line_dist_abs_end))
            return left_curverad, right_curverad, poly_left_x, poly_right_x

        ###if it is among the first n measurements:
        if (self.hist_len < self.MAX_HIST_LEN):

            self.log("Filling detection history:{}/{}".format(self.hist_len, self.MAX_HIST_LEN))

            # if the lines are close enough to each other
            if (self.EXPECTED_LINE_DIST - self.LINE_DIST_TOL) < line_dist_abs_begin < (self.EXPECTED_LINE_DIST + self.LINE_DIST_TOL)\
                    and (self.EXPECTED_LINE_DIST - self.LINE_DIST_TOL) < line_dist_abs_end < (self.EXPECTED_LINE_DIST + self.LINE_DIST_TOL):
                # add it to the list, and return the new average
                self.add_to_list_and_avg(self.left_line, left_fit_coeffs, poly_left_x, left_curverad)
                self.add_to_list_and_avg(self.right_line, right_fit_coeffs, poly_right_x, right_curverad)
                self.hist_len = self.hist_len + 1

            elif (not (self.EXPECTED_LINE_DIST - self.LINE_DIST_TOL) < line_dist_abs_begin < (self.EXPECTED_LINE_DIST + self.LINE_DIST_TOL)):
                self.log("Line dist begin out of range:{} (must be {} +- {})".format(line_dist_abs_begin, self.EXPECTED_LINE_DIST, self.LINE_DIST_TOL))

            else:
                self.log("Line dist end out of range:{} (must be {} +- {})".format(line_dist_abs_end, self.EXPECTED_LINE_DIST, self.LINE_DIST_TOL))

            return self.left_line.curverad, self.right_line.curverad, self.left_line.x, self.right_line.x

        else:
            line_pos_begin_left_delta = abs(poly_left_x[0] - self.left_line.x[0])
            line_pos_end_left_delta = abs(poly_left_x[len(poly_left_x) - 1] - self.left_line.x[len(poly_left_x) - 1])

            line_pos_begin_right_delta = abs(poly_right_x[0] - self.right_line.x[0])
            line_pos_end_right_delta = abs(poly_right_x[len(poly_right_x) - 1] - self.right_line.x[len(poly_right_x) - 1])

            curverad_delta_left_quot = abs(left_curverad/self.left_line.curverad)
            curverad_delta_right_quot = abs(right_curverad/self.right_line.curverad)

            str = " curverad_left:{}, curverad_right:{}" + \
                  " (AVG_LEFT: {}, AVG_RIGHT:{})"

            strFormatted = str.format(left_curverad, right_curverad, self.left_line.curverad, self.right_line.curverad)

            self.logfile.write(strFormatted + "\n")


            if(not self.check_data(line_dist_abs_begin, line_dist_abs_end, line_pos_begin_left_delta, line_pos_end_left_delta, line_pos_begin_right_delta, line_pos_end_right_delta, curverad_delta_left_quot, curverad_delta_right_quot)):

                self.subseq_not_detect = self.subseq_not_detect + 1
                errstr = "At least one value out of tolerance:\n" + strFormatted
                print(errstr)
                self.logfile.write("\n")

                if self.subseq_not_detect >= self.MAX_SUBSEQ_NOT_DETECT:
                    self.log("Resetting history because of {} bad detections".format(self.MAX_SUBSEQ_NOT_DETECT))
                    self.subseq_not_detect = 0
                    self.hist_len = 0
            else:
                self.add_to_list_and_avg(self.left_line, left_fit_coeffs, poly_left_x, left_curverad)
                self.add_to_list_and_avg(self.right_line, right_fit_coeffs, poly_right_x, right_curverad)

            return self.left_line.curverad, self.right_line.curverad, self.left_line.x, self.right_line.x


    def pipeline(self, original_img):

        #* 0. Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
        if not os.path.isfile('undistort_pickle.p'):
            mtx, dist, rvecs, tvec = advll_helpers.chessboard_calibration()
            pickle.dump([mtx, dist, rvecs, tvec], open( "undistort_pickle.p", "wb" ))
        else:
            mtx, dist, rvecs, tvec = pickle.load(open("undistort_pickle.p", "rb"))
            #advll_helpers.correct_imgs_in_folder(mtx, dist, rvecs, tvec, 'camera_cal')

        #* 1. Apply a distortion correction to raw image
        undistorted_img = cv2.undistort(original_img, mtx, dist, None, mtx)

        #* 2. Use color transforms, gradients, etc., to create a thresholded binary image.
        white_yellow_img, gray_img, blurred_img, canny_img, cut_img, binary_img = advll_helpers.create_binary_image(undistorted_img)

        #* 3. Apply a perspective transform to rectify binary image ("birds-eye view").
        perspective_trans_img, transformMatrix = advll_helpers.perspective_transform(binary_img)

        #* 4. Detect lane pixels and fit to find the lane boundary.
        left_pix_x, left_pix_y, right_pix_x, right_pix_y, debug_img = advll_helpers.find_lane_pixels(perspective_trans_img)

        try:
            left_fit_coeffs = np.polyfit(left_pix_y, left_pix_x, 2)
            right_fit_coeffs = np.polyfit(right_pix_y, right_pix_x, 2)
        except TypeError:
            left_fit_coeffs = [1, 1, 1]
            right_fit_coeffs = [1, 1, 1]

        #* 5. Fit polynoms.
        poly_y, poly_left_x, poly_right_x =  advll_helpers.generate_polygon_lines(left_fit_coeffs, right_fit_coeffs, perspective_trans_img)
        debug_img = advll_helpers.plot_polygon_lines(poly_y, poly_left_x, poly_right_x, debug_img, [255, 0, 0])

        # 6. Process the line data and discard crappy data
        opt_left_curverad, opt_right_curverad, opt_poly_left_x, opt_poly_right_x = \
            self.process_new_line_data(left_fit_coeffs, right_fit_coeffs, poly_left_x, poly_right_x, left_pix_x, left_pix_y, right_pix_x, right_pix_y)

        debug_img = advll_helpers.plot_polygon_lines(poly_y, opt_poly_left_x, opt_poly_right_x, debug_img, [0, 0, 255])

        Minv = np.linalg.inv(transformMatrix)

        #* 7. Warp the detected lane boundaries back onto the original image.
        final_img = advll_helpers.draw_lane_and_warp_back_to_original(perspective_trans_img, opt_poly_left_x, opt_poly_right_x, poly_y, undistorted_img, Minv)

        ## 8. Calculate vehicle position
        vehicle_pos = advll_helpers.determine_vehicle_pos(opt_poly_left_x[len(opt_poly_left_x) - 1], opt_poly_right_x[len(opt_poly_right_x) - 1])
        final_img = advll_helpers.print_info_on_img(self.logfile, final_img, opt_left_curverad, opt_right_curverad, vehicle_pos)

        if self.attach_polynom_img:
            final_img = np.concatenate((final_img, debug_img), axis=1)

        if self.show_imgs:
            advll_helpers.show_imgs(original_img, undistorted_img, white_yellow_img, canny_img, binary_img, debug_img, final_img)

        return final_img

def pipeline_on_images():

    p = LaneDetectionPipeline(False, True, False)

    for filename in os.listdir("test_images/"):

        #filename = "straight_lines1.jpg"
        print(filename)
        #image = mpimg.imread('camera_cal/' + filename)
        image = mpimg.imread("test_images/" + filename)
        final_image = p.pipeline(image)
        cv2.imwrite('output_images/' + filename, final_image)

    return

def pipeline_on_single_image(filename):

    p = LaneDetectionPipeline(False, True, False)
    image = mpimg.imread("test_images/" + filename)
    final_image = p.pipeline(image)
    cv2.imwrite('output_images/' + filename, final_image)


def pipeline_on_video():
    # Import everything needed to edit/save/watch video clips
    from moviepy.editor import VideoFileClip
    #from IPython.display import HTML

    p = LaneDetectionPipeline(True, False, False)

    def process_image(image):
        result = p.pipeline(image)
        return result


    white_output = 'project_video_output.mp4'
    clip1 = VideoFileClip("project_video.mp4")

    white_clip = clip1.fl_image(process_image)
    white_clip.write_videofile(white_output, audio=False)

    return
