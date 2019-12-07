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

# Custom packages
from src.panorama import Stitcher

# Initialize the video stream
print("[INFO] starting cameras...")
leftStream = cv2.VideoCapture(1)
rightStream = cv2.VideoCapture(2)

# Set the width and unseccusfuly set the exposure time
leftStream.set(3, 360)
leftStream.set(15,0.5) # Warmup the first camera
rightStream.set(3, 360)
rightStream.set(15, 0.5) # Warmup the second camera

# Warmup the cameras
#time.sleep(2.0)s

# Initialize the frame stitcher from Stitcher class
stitcher = Stitcher()
total_frame = 0

while True:
    left = leftStream.read()
    right = rightStream.read()

    result = stitcher.stitch([left, right])

    if result is None:
        print("[INFO] homography could not be computed")
        break

    # Convert the panorama to grayscale and blur it slightly
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    cv2.imshow("Resul", result)
    cv2.imshow("Left Frame", left)
    cv2.imshow("Right Frame", right)

    key = cv2.waitKey(1) & 0xFF

    # If the 'q' key was pressed, break from the loop
    if key == ord('q'):
        break

# Do a bit of cleanup
print("[INFO] cleaning up...")
leftStream.release()
rightStream.release()
cv2.DestroyAllWindows()
