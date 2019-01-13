#### unused
def chessboard_find_corners(image, nx, ny):

    # prepare object points
    nx = 9#TODO: enter the number of inside corners in x
    ny = 6#TODO: enter the number of inside corners in y

    # Make a list of calibration images
    fname = 'camera_cal/calibration2.jpg'
    img = cv2.imread(fname)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

    # If found, draw corners
    if ret == True:
        # Draw and display the corners
        cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
        plt.imshow(img)
        plt.show()

    #pickle.dump( corners, open( "wide_dist_pickle.p", "wb" ) )

    #dist_pickle = pickle.load(open("wide_dist_pickle.p", "rb"))
    #mtx = dist_pickle["mtx"]
    #dist = dist_pickle["dist"]

    return corners


    # Read in the saved objpoints and imgpoints
    #dist_pickle = pickle.load(open("wide_dist_pickle.p", "rb"))
    #objpoints = dist_pickle["objpoints"]
    #imgpoints = dist_pickle["imgpoints"]


    #f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    #f.tight_layout()
    #ax1.imshow(img)
    #ax1.set_title('Original Image', fontsize=50)
    #ax2.imshow(undistorted)
    #ax2.set_title('Undistorted Image', fontsize=50)
    #plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    #plt.show()