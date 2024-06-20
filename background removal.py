# import cv2 to capture videofeed
import cv2

import numpy as np

# attach camera indexed as 0
camera = cv2.VideoCapture(0)

# setting framewidth and frameheight as 640 X 480
camera.set(3, 640)
camera.set(4, 480)

# loading the mountain image
mountain = cv2.imread('mount everest.jpg')
mountain = cv2.resize(mountain, (640, 480))

while True:
    # read a frame from the attached camera
    status, frame = camera.read()

    # if we got the frame successfully
    if status:
        # flip it
        frame = cv2.flip(frame, 1)

        # converting the image to RGB for easy processing
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # thresholding image
        lower_bound = np.array([0, 48, 80])  # lower bound for skin tone detection
        upper_bound = np.array([20, 255, 255])  # upper bound for skin tone detection
        mask = cv2.inRange(frame_rgb, lower_bound, upper_bound)

        # inverting the mask
        mask_inv = cv2.bitwise_not(mask)

        # bitwise AND operation to extract foreground / person
        result = cv2.bitwise_and(frame, frame, mask=mask_inv)

        # show it
        cv2.imshow('frame', result)

        # wait for 1ms before displaying another frame
        code = cv2.waitKey(1)
        if code == 32:
            break

# release the camera and close all opened windows
camera.release()
cv2.destroyAllWindows()