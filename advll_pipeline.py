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
        self.x = [np.array([False])]
        self.coeffs = None
        self.curverad = None
        self.logfile = None


class LaneDetectionPipeline():
    def __init__(self, on_video, show_imgs):
        self.on_video = on_video
        self.show_imgs = show_imgs
        self.left_line = Line()
        self.right_line = Line()
        self.MAX_DEV_X_STEP = 80
        #self.MAX_DEV_CURVE_STEP = 100000
        self.MAX_DEV_CURVE_QUOT = 4
        self.MAX_CURVERAD = 10000
        self.MAX_HIST_LEN = 20

        self.EXPECTED_LINE_DIST = 900
        self.LINE_DIST_TOL = 200
        self.subseq_not_detect = 0
        self.MAX_SUBSEQ_NOT_DETECT = 10
        self.hist_len = 0

        self.logfile = open("logs.txt", "w")

    def log(self, text):
        print(text)
        self.logfile.write(text + "\n")

    def write_info(self, img, left_curve, right_curve, vehicle_pos):

        avg_curve = (left_curve + right_curve) / 2
        #### write curvature on image
        strToPlot = "Curve: {0:.2f} (L:{1:.2f}, R:{2:.2f}), x-Pos: {3:.3f}".format(avg_curve, left_curve, right_curve,
                                                                                   vehicle_pos)
        self.logfile.write(strToPlot + "\n")
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


    def add_to_list_and_avg(self, line, fit_coeffs, poly_x, curverad):
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

    def check_lines(self, left_line, right_line, left_fit_coeffs, right_fit_coeffs, poly_left_x, poly_right_x, left_pix_x, left_pix_y, right_pix_x, right_pix_y):

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
                self.add_to_list_and_avg(left_line, left_fit_coeffs, poly_left_x, left_curverad)
                self.add_to_list_and_avg(right_line, right_fit_coeffs, poly_right_x, right_curverad)
                self.hist_len = self.hist_len + 1

            elif (not (self.EXPECTED_LINE_DIST - self.LINE_DIST_TOL) < line_dist_abs_begin < (self.EXPECTED_LINE_DIST + self.LINE_DIST_TOL)):
                self.log("Line dist begin out of range:{} (must be {} +- {})".format(line_dist_abs_begin, self.EXPECTED_LINE_DIST, self.LINE_DIST_TOL))

            else:
                self.log("Line dist end out of range:{} (must be {} +- {})".format(line_dist_abs_end, self.EXPECTED_LINE_DIST, self.LINE_DIST_TOL))
            return left_line.curverad, right_line.curverad, left_line.x, right_line.x

        else:
            line_pos_begin_left_delta = abs(poly_left_x[0] - left_line.x[0])
            line_pos_end_left_delta = abs(poly_left_x[len(poly_left_x) - 1] - left_line.x[len(poly_left_x) - 1])

            line_pos_begin_right_delta = abs(poly_right_x[0] - right_line.x[0])
            line_pos_end_right_delta = abs(poly_right_x[len(poly_right_x) - 1] - right_line.x[len(poly_right_x) - 1])

            #curverad_delta_left = abs(left_curverad - left_line.curverad)
            #curverad_delta_right = abs(right_curverad - right_line.curverad)

            curverad_delta_left_quot = abs(left_curverad/left_line.curverad)
            curverad_delta_right_quot = abs(right_curverad/right_line.curverad)

            #str = "line_dist_abs_begin:{} " + \
            #      " \n line_pos_begin_left_delta:{} \n line_pos_end_left_delta:{}" + \
            #      " \n line_pos_begin_right_delta:{} \n line_pos_end_right_delta:{}" + \
            #      " \n curverad_delta_right:{}, curverad_delta_left:{}" + \
            #      " \n curverad_right:{}, curverad_left:{}" + \
            #      " \n (EXPECTED_LINE_DIST: {}, LINE_DIST_TOL:{}, MAX_DEV_X:{}, MAX_DEV_CURVE:{})"

            #str = "line_dist_abs_begin:{} " + \
            #      " line_pos_begin_left_delta:{} line_pos_end_left_delta:{}" + \
            #      " line_pos_begin_right_delta:{} line_pos_end_right_delta:{}" + \
            #      " curverad_delta_right_quot:{}, curverad_delta_left_quot:{}" + \
            #      " curverad_left:{}, curverad_right:{}" + \
            #      " (EXPECTED_LINE_DIST: {}, LINE_DIST_TOL:{}, MAX_DEV_X:{}, MAX_DEV_CURVE_QUOT:{})"

            #strFormatted = str.format(\
            #              line_dist_abs_begin, line_pos_begin_left_delta, \
            #              line_pos_end_left_delta, line_pos_begin_right_delta, \
            #              line_pos_end_right_delta, curverad_delta_right_quot, curverad_delta_left_quot, \
            #             left_curverad, right_curverad, \
            #                self.EXPECTED_LINE_DIST, self.LINE_DIST_TOL, self.MAX_DEV_X_STEP, self.MAX_DEV_CURVE_QUOT)

            #left_line.curverad

            str = " curverad_left:{}, curverad_right:{}" + \
                  " (AVG_LEFT: {}, AVG_RIGHT:{})"

            strFormatted = str.format(left_curverad, right_curverad, left_line.curverad, right_line.curverad)

            self.logfile.write(strFormatted + "\n")

            # if the lines are close enough to each other and it is not too different from the previous detection
            success = False

            if (self.EXPECTED_LINE_DIST - self.LINE_DIST_TOL) < line_dist_abs_begin < (self.EXPECTED_LINE_DIST + self.LINE_DIST_TOL) \
                    and (self.EXPECTED_LINE_DIST - self.LINE_DIST_TOL) < line_dist_abs_end < (self.EXPECTED_LINE_DIST + self.LINE_DIST_TOL) \
                    and line_pos_begin_left_delta < self.MAX_DEV_X_STEP and line_pos_end_left_delta < self.MAX_DEV_X_STEP \
                    and line_pos_begin_right_delta < self.MAX_DEV_X_STEP and line_pos_end_right_delta < self.MAX_DEV_X_STEP \
                    and 1/self.MAX_DEV_CURVE_QUOT < curverad_delta_left_quot < self.MAX_DEV_CURVE_QUOT \
                    and 1/self.MAX_DEV_CURVE_QUOT < curverad_delta_right_quot < self.MAX_DEV_CURVE_QUOT:
                    #and curverad_delta_left < self.MAX_DEV_CURVE \
                    #and curverad_delta_right < self.MAX_DEV_CURVE:

                self.subseq_not_detect = 0

                # add it to the list, return the new average
                self.add_to_list_and_avg(left_line, left_fit_coeffs, poly_left_x, left_curverad)
                self.add_to_list_and_avg(right_line, right_fit_coeffs, poly_right_x, right_curverad)
                print(strFormatted + "\n")

                success = True

            elif(not(self.EXPECTED_LINE_DIST - self.LINE_DIST_TOL) < line_dist_abs_begin < (self.EXPECTED_LINE_DIST + self.LINE_DIST_TOL)):
                self.log("line_dist_abs_begin({}) out of range: must be between {} and {}".format(line_dist_abs_begin, self.EXPECTED_LINE_DIST - self.LINE_DIST_TOL, self.EXPECTED_LINE_DIST + self.LINE_DIST_TOL))

            elif (not (self.EXPECTED_LINE_DIST - self.LINE_DIST_TOL) < line_dist_abs_end < (self.EXPECTED_LINE_DIST + self.LINE_DIST_TOL)):
                self.log("line_dist_abs_end({}) out of range: must be between {} and {}".format(line_dist_abs_end, self.EXPECTED_LINE_DIST - self.LINE_DIST_TOL, self.EXPECTED_LINE_DIST + self.LINE_DIST_TOL))

            elif(not (line_pos_begin_left_delta < self.MAX_DEV_X_STEP and line_pos_end_left_delta < self.MAX_DEV_X_STEP)):
                self.log("line_pos_begin_left_delta({}) or end({}) out of range: must be below {}".format(line_pos_begin_left_delta, line_pos_end_left_delta, self.MAX_DEV_X_STEP))

            elif(not (line_pos_begin_right_delta < self.MAX_DEV_X_STEP and line_pos_end_right_delta < self.MAX_DEV_X_STEP)):
                self.log("line_pos_begin_right_delta({}) or end({}) out of range: must be below {}".format(line_pos_begin_right_delta, line_pos_end_right_delta, self.MAX_DEV_X_STEP))

            elif(not (1 / self.MAX_DEV_CURVE_QUOT < curverad_delta_left_quot < self.MAX_DEV_CURVE_QUOT)):
                self.log("curverad_delta_left_quot({}) out of range: must be between {} and {}".format(curverad_delta_left_quot, 1 / self.MAX_DEV_CURVE_QUOT, self.MAX_DEV_CURVE_QUOT))

            elif(not (1 / self.MAX_DEV_CURVE_QUOT < curverad_delta_right_quot < self.MAX_DEV_CURVE_QUOT)):
                self.log("curverad_delta_right_quot({}) out of range: must be between {} and {}".format(curverad_delta_right_quot, 1 / self.MAX_DEV_CURVE_QUOT, self.MAX_DEV_CURVE_QUOT))


            if(not success):
                self.subseq_not_detect = self.subseq_not_detect + 1
                errstr = "At least one value out of tolerance:\n" + strFormatted
                print(errstr)
                self.logfile.write("\n")

                if self.subseq_not_detect >= self.MAX_SUBSEQ_NOT_DETECT:
                    self.log("Resetting history because of {} bad detections".format(self.MAX_SUBSEQ_NOT_DETECT))
                    self.subseq_not_detect = 0
                    self.hist_len = 0

            return left_line.curverad, right_line.curverad, left_line.x, right_line.x



    def pipeline(self, img):

        #* 0. Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
        if not os.path.isfile('undistort_pickle.p'):
            mtx, dist, rvecs, tvec = advll_helpers.chessboard_calibration()
            pickle.dump([mtx, dist, rvecs, tvec], open( "undistort_pickle.p", "wb" ))
        else:
            mtx, dist, rvecs, tvec = pickle.load(open("undistort_pickle.p", "rb"))
            #advll_helpers.correct_imgs_in_folder(mtx, dist, rvecs, tvec, 'camera_cal')

        #* 1. Apply a distortion correction to raw image
        undistorted_img = cv2.undistort(img, mtx, dist, None, mtx)

        #* 2. Use color transforms, gradients, etc., to create a thresholded binary image.
        white_yellow_image, gray_image, blurred_image, canny_image, cut_image, binary_img = advll_helpers.create_binary_image(undistorted_img)

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

        poly_y, poly_left_x, poly_right_x =  advll_helpers.generate_polygon_lines(left_fit_coeffs, right_fit_coeffs, perspective_trans_img)
        debug_img = advll_helpers.plot_polygon_lines(poly_y, poly_left_x, poly_right_x, debug_img, [255, 0, 0])

        opt_left_curverad, opt_right_curverad, opt_poly_left_x, opt_poly_right_x = \
            self.check_lines(self.left_line, self.right_line, left_fit_coeffs, right_fit_coeffs, poly_left_x, poly_right_x, left_pix_x, left_pix_y, right_pix_x, right_pix_y)

        #poly_y, opt_poly_left_x, opt_poly_right_x =  generate_polygon_lines(opt_left_fit_coeffs, opt_right_fit_coeffs, perspective_trans_img)
        debug_img = advll_helpers.plot_polygon_lines(poly_y, opt_poly_left_x, opt_poly_right_x, debug_img, [0, 0, 255])

        #* 5. Determine the curvature of the lane and vehicle position with respect to center.
        #left_curverad, right_curverad = measure_curvature_pixels(ploty, left_fit, right_fit)
        #left_curverad, right_curverad = measure_curvature_real(left_pix_x, left_pix_y, right_pix_x, right_pix_y)

        Minv = np.linalg.inv(transformMatrix)

        #* 6. Warp the detected lane boundaries back onto the original image.
        #final_img = advll_helpers.draw_lane_and_warp_back_to_original(perspective_trans_img, opt_poly_left_x, opt_poly_right_x, poly_y, undistorted_img, Minv)
        final_img = debug_img;

        ## calculate vehicle position
        vehicle_pos = advll_helpers.determine_vehicle_pos(opt_poly_left_x[len(opt_poly_left_x) - 1], opt_poly_right_x[len(opt_poly_right_x) - 1])
        final_img = self.write_info(final_img, opt_left_curverad, opt_right_curverad, vehicle_pos)


#white_yellow_image, gray_image, blurred_image, canny_image, cut_image, binary_img

        if self.show_imgs:

            #plt.imshow(debug_img)
            #plt.show()

            f = plt.figure(figsize=(19, 8))
            plt.tight_layout()

            p1 = plt.subplot(2, 4, 1)
            p1.imshow(img)
            p1.set_title(('Original Image'))

            p2 = plt.subplot(2, 4, 2)
            p2.imshow(undistorted_img)
            p2.set_title(('Undistorted Image'))

            p2 = plt.subplot(2, 4, 3)
            p2.imshow(white_yellow_image)
            p2.set_title(('White-Yellow Image'))

            p2 = plt.subplot(2, 4, 4)
            p2.imshow(canny_image)
            p2.set_title(('Canny-Image'))

            p2 = plt.subplot(2, 4, 5)
            p2.imshow(binary_img, cmap='gray')
            p2.set_title(('Binary Image'))

            #p2 = plt.subplot(2, 3, 4)
            #p2.imshow(perspective_trans_img, cmap='gray')
            #p2.set_title(('Perspective Transform'))

            p2 = plt.subplot(2, 4, 6)
            p2.imshow(debug_img)
            p2.set_title(('Detected Lane Pixels'))

            p2 = plt.subplot(2, 4, 7)
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

def pipeline_on_single_image(filename):

    p = LaneDetectionPipeline(False, True)
    image = mpimg.imread("test_images/" + filename)
    final_image = p.pipeline(image)
    cv2.imwrite('output_images/' + filename, final_image)


def pipeline_on_video():
    # Import everything needed to edit/save/watch video clips
    from moviepy.editor import VideoFileClip
    #from IPython.display import HTML

    p = LaneDetectionPipeline(True, False)

    def process_image(image):
        result = p.pipeline(image)
        return result

    white_output = 'project_video_output.mp4'

    #clip1 = VideoFileClip("project_video.mp4").subclip(0,20)
    clip1 = VideoFileClip("project_video.mp4")

    white_clip = clip1.fl_image(process_image)
    white_clip.write_videofile(white_output, audio=False)

    return
