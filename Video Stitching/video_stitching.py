# -*- coding: utf-8 -*-
"""
@author: R. Erdem Uysal
@company: Turk AI
Resources: www.pyimagesearch.com/2016/01/25/real-time-panorama-and-image-stitching-with-opencv
Usage: python .py
"""

# Import necessary libraries
import cv2
import numpy as py
import sys
import time
import datetime


# Initialize the video stream
print("[INFO] starting cameras...")
leftStream = cv2.VideoCapture("left.mp4")
rightStream = cv2.VideoCapture("right.mp4")

# Set the width and unseccusfuly set the exposure time
#leftStream.set(3, 400)
#leftStream.set(15,0.5) # Warmup the first camera
#rightStream.set(3, 400)
#rightStream.set(15, 0.5) # Warmup the second camera

# Warmup the cameras
#time.sleep(2.0)s

# Find OpenCV version
(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
# Initialize the frame stitcher from Stitcher class
if int(major_ver) == 3:
    stitcher = cv2.createStitcher()
else:
    stitcher = cv2.Stitcher_create()

frames = []

while True:
    retval, frameLeft = leftStream.read()
    retval_, frameRight = rightStream.read()
    frameLeft = cv2.resize(frameLeft, (480, 300))
    frameRight = cv2.resize(frameRight, (480, 300))

    (status, stitched) = stitcher.stitch([frameLeft, frameRight])

    if status == 0:
        starttime = time.time()
        # Get frame
        timestamp = time.time() - starttime
        print(timestamp)
        #cv2.putText(frameLeft, timestamp, (10,500), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2)
        #cv2.putText(frameRight, timestamp, (10,500), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2)
    
        cv2.imshow("Result", stitched)
        cv2.imshow("Left Frame", frameLeft)
        cv2.imshow("Right Frame", frameRight )

        key = cv2.waitKey(1) & 0xFF

        # If the 'q' key was pressed, break from the loop
        if key == ord('q'):
            break
    else:
        print("[INFO] image stitching failed ({})".format(status))

# Do a bit of cleanup
print("[INFO] cleaning up...")
leftStream.release()
rightStream.release()
cv2.destroyAllWindows()