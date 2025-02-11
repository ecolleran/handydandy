# Computer Vision Course (CSE 40535/60535)
# University of Notre Dame, Fall 2024
# ________________________________________________________________
# Adam Czajka, Andrey Kuehlkamp, September 2017 - 2024

# Here are your tasks:
#
# Task 2a:
# - Select one object that you want to track and set the RGB
#   channels to the selected ranges (found by colorSelection.py).
# - Check if HSV color space works better. Can you ignore one or two
#   channels when working in HSV color space? Why?
# - Try to track candies of different colors (blue, yellow, green).
# ANSWER:
# The HSV color space is easier to wrk with because the peaks are more exteme, so it is
# easier to identify the range of colors that you want to track. Within the HSV space, 
# the S and V channels can be ignored since H is the most prominent. Value is much more
# focused on lighting. 
# 
# Task 2b:
# - Adapt your code to track multiple objects of *the same* color simultaneously, 
#   and show them as separate objects in the camera stream.
#
# Task 2c:
# - Adapt your code to track multiple objects of *different* colors simultaneously,
#   and show them as separate objects in the camera stream. Make your code elegant 
#   and requiring minimum changes when the number of different objects to be detected increases.
# ANSWER:
# I made a JSON of color ran ges and color boxes to account for more colors here. I also adapted
# the if loop into a for loop and rmeoved the max perameter to account for multiple objects of one color.
#
# Task for students attending 60000-level course:
# - Choose another color space (e.g., LAB or YCrCb), modify colorSelection.py, select color ranges 
#   and after some experimentation say which color space was best (RGB, HSV or the additional one you selected).
#   Try to explain the reasons why the selected color space performed best. 

import cv2
import numpy as np

cam = cv2.VideoCapture(0)
#Color ranges for detecting selected color (NOTE: OpenCV uses BGR instead of RGB)
color_ranges = {
    "blue": ([140, 70, 0], [210, 120, 50]),
    "green": ([40, 120, 0], [90, 190, 50]),
    "yellow": ([20, 170, 180], [70, 220, 230]),
    "orange": ([0, 40, 180], [60, 100, 220])
}

#boxes for outlining objects in the right colors
box_colors = {
    "blue": (255, 0, 0),
    "green": (0, 255, 0),
    "yellow": (0, 255, 255),
    "orange": (0, 165, 255)
}

while (True):
    retval, img = cam.read()

    res_scale = 0.5 # rescale the input image if it's too large
    img = cv2.resize(img, (0,0), fx = res_scale, fy = res_scale)

    for color_name, (lower, upper) in color_ranges.items():
        lower = np.array(lower, dtype=np.uint8)
        upper = np.array(upper, dtype=np.uint8)
        objmask = cv2.inRange(img, lower, upper)


        # You may use this for debugging
        cv2.imshow("Binary image", objmask)

        # Resulting binary image may have large number of small objects.
        # You may check different morphological operations to remove these unnecessary
        # elements. You may need to check your ROI defined in step 1 to
        # determine how many pixels your object may have.
        kernel = np.ones((5,5), np.uint8)
        objmask = cv2.morphologyEx(objmask, cv2.MORPH_CLOSE, kernel=kernel)
        objmask = cv2.morphologyEx(objmask, cv2.MORPH_DILATE, kernel=kernel)
        
        cv2.imshow("Image after morphological operations", objmask)

        # find connected components
        cc = cv2.connectedComponents(objmask)
        ccimg = cc[1].astype(np.uint8)

        # Find contours of these objects
        contours, hierarchy = cv2.findContours(ccimg,
                                                cv2.RETR_TREE,
                                                cv2.CHAIN_APPROX_SIMPLE)[-2:]

        # We are using [-2:] to select the last two return values from the above function to make the code work with
        # both opencv3 and opencv4. This is because opencv3 provides 3 return values but opencv4 discards the first.

        # You may display the countour points if you want:
        # cv2.drawContours(img, contours, -1, (0,255,0), 3)

        # Ignore bounding boxes smaller than "minObjectSize"
        minObjectSize = 20;
    

        #######################################################
        # TIP: think if the "if" statement
        # can be replaced with a "for" loop
        for contor in contours:
        #######################################################
        
            #want to have multiple objects tracked so max function is removed
            x, y, w, h = cv2.boundingRect(contor)

            #######################################################
            # TIP: you want to get bounding boxes
            # of ALL contours (not only the first one)
            #######################################################

            # Do not show very small objects
            if w > minObjectSize or h > minObjectSize:
                cv2.rectangle(img, (x, y), (x+w, y+h), box_colors[color_name], 3)
                cv2.putText(img,            # image
                f"Here's my {color_name} object!", # text
                (x, y-10),                  # start position
                cv2.FONT_HERSHEY_SIMPLEX,   # font
                0.7,                        # size
                box_colors[color_name],     # BGR color
                1,                          # thickness
                cv2.LINE_AA)                # type of line

    cv2.imshow("Live WebCam", img)

    action = cv2.waitKey(1)
    if action==27:
        break
    
cam.release()
cv2.destroyAllWindows()