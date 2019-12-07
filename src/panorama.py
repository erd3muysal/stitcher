#! -*- coding: utf-8 -*-
"""
@author: R. Erdem Uysal
@company: Turk AI

Resources:
Usage:
"""
# Import the necessary libraries
import cv2
import numpy as np

class Stitcher(object):
    def __init__(self):
	# Initialize cached homography matrix
	self.cachedH = None

    def stitch(self, frames, ratio=0.75, reprojThresh=4.0):
	# Unpack the images
	(frameLeft, frameRight) = frames

	# If the cached homography matrix is 'None', then we need to
	# apply keypoint matching to construct it
	if self.cachedH is None:
        # Detect keypoints and extract
	    (kpsRight, featuresRight) = self.detectAndDescribe(frameRight)
            (kpsLeft, featuresLeft) = self.detectAndDescribe(frameLeft)

	    # Match features between the two images
	    M = self.matchKeypoints(kpsRight, kpsLeft,
				featuresRight, featuresLeft, ratio, reprojThresh)

	    # If the match is None, then there aren't enough matched
	    # keypoints to create a panorama
	    if M is None:
	        return None

	    # Cache the homography matrix
	    self.cachedH = M[1]

	    # Apply a perspective transform to stitch the images together
	    # Using the cached homography matrix
	    result = cv2.warpPerspective(frameRight, self.cachedH,
	                                (frameRight.shape[1] + frameLeft.shape[1], frameRight.shape[0]))
	    result[0:frameLeft.shape[0], 0:frameLeft.shape[1]] = frameLeft

        # Return the stitched image
        return result

    def detectAndDescribe(self, frame):
        # Convert the image to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Check to see if we are using OpenCV 3.X
        if self.isv3:
            # Detect and extract features from the image
            descriptor = cv2.xfeatures2d.SIFT_create()
            (kps, features) = descriptor.detectAndCompute(frame, None)

        # Otherwise, we are using OpenCV 2.4.X
        else:
            # Detect keypoints in the image
            detector = cv2.FeatureDetector_create("SIFT")
            kps = detector.detect(gray)

            # Extract features from the image
            extractor = cv2.DescriptorExtractor_create("SIFT")
            (kps, features) = extractor.compute(gray, kps)

        # Convert the keypoints from KeyPoint objects to NumPy arrays
        kps = np.float32([kp.pt for kp in kps])

        # Return a tuple of keypoints and features
	return (kps, features)

    def matchKeypoints(self, kpsRight, kpsLeft, featuresRight, featuresLeft, ratio, reprojThresh):
	# Compute the raw matches and initialize the list of actual matches
        matcher = cv2.DescriptorMatcher_create("BruteForce")
        rawMatches = matcher.knnMatch(featuresRight, featuresLeft, 2)
        matches = []

	# Loop over the raw matches
        for m in rawMatches:
            # Ensure the distance is within a certain ratio of each
            # other (i.e. Lowe's ratio test)
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                matches.append((m[0].trainIdx, m[0].queryIdx))

        # Computing a homography requires at least 4 matches
        if len(matches) > 4:
            # Construct the two sets of points
            ptsRight = np.float32([kpsRight[i] for (_, i) in matches])
            ptsLeft = np.float32([kpsLeft[i] for (i, _) in matches])

            # Compute the homography between the two sets of points
            (H, status) = cv2.findHomography(ptsRight, ptsLeft, cv2.RANSAC,
                                             reprojThresh)

            # Return the matches along with the homograpy matrix
            # and status of each matched point
            return (matches, H, status)

        # Otherwise, no homograpy could be computed
	return None
