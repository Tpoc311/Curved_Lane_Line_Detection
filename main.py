"""
DESCRIPTION
    This module - is a main module containing a high-level methods and functions
    to implement Advanced Lane Line Detection.
"""

import sys

import cv2
import numpy as np

from utils import undistort, perspective_warp, sliding_window, draw_lanes


def abs_sobel_thresh(img, orient='x', thresh_min=25, thresh_max=255):
    """
    Preprocesses image with Sobel filter
    :param img: image to filter
    :param orient: axis to process which
    :param thresh_min: low threshold
    :param thresh_max: high threshold
    :return: filtered image
    """
    # Convert to grayscale
    # gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    l_channel = hls[:, :, 1]
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(l_channel, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(l_channel, cv2.CV_64F, 0, 1))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    # Return the result
    return binary_output


def color_threshold(image: np.ndarray, sthresh=(0, 255), vthresh=(0, 255)) -> np.ndarray:
    """
    Thresholds image with HLS and HSV color models.=
    :param image: image to threshold
    :param sthresh: saturation threshold
    :param vthresh: value threshold
    :return: thresholded image
    """
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    s_channel = hls[:, :, 2]
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel > sthresh[0]) & (s_channel <= sthresh[1])] = 1

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    v_channel = hsv[:, :, 2]
    v_binary = np.zeros_like(v_channel)
    v_binary[(v_channel > vthresh[0]) & (v_channel <= vthresh[1])] = 1

    output = np.zeros_like(s_channel)
    output[(s_binary == 1) & (v_binary) == 1] = 1

    # Return the combined s_channel & v_channel binary image
    return output


def preprocess_image(img: np.ndarray) -> np.ndarray:
    """
    Preprocesses image to get only required lane lines.
    :param img: image to preprocess
    :return: preprocessed image
    """
    # Apply Sobel operator in X-direction to experiment with gradient thresholds
    gradx = abs_sobel_thresh(img, orient='x', thresh_min=20, thresh_max=100)

    # Apply Sobel operator in Y-direction to experiment with gradient thresholds
    grady = abs_sobel_thresh(img, orient='y', thresh_min=20, thresh_max=100)

    # Experiment with HLS & HSV color spaces along with thresholds
    c_binary = color_threshold(img, sthresh=(100, 255), vthresh=(50, 255))

    preprocessed = np.zeros_like(img[:, :, 0])
    preprocessed[((gradx == 1) & (grady == 1) | (c_binary == 1))] = 255

    return preprocessed


if __name__ == '__main__':
    VIDEO_PATH = r'videos/challenge_1.mp4'

    cap = cv2.VideoCapture(VIDEO_PATH, 0)

    while cap.isOpened():
        ret, original_image = cap.read()
        if not ret:
            sys.exit()
        # original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

        # 1. Distortion correction
        undistorted_image = undistort(original_image)

        # 2. Preprocess image
        filtered_image = preprocess_image(undistorted_image)

        # 3. Perspective warp
        warped_image = perspective_warp(img=filtered_image, dst_size=(1280, 720))

        # 4. Sliding window search
        out_img, curves, lanes, ploty = sliding_window(warped_image)

        # 5. Plot lines
        img_ = draw_lanes(original_image, curves[0], curves[1])

        cv2.imshow('Sliding window search', out_img)
        cv2.imshow('result', img_)

        if cv2.waitKey(1) & 0xff == ord('q'):
            sys.exit()
    print('Success')
