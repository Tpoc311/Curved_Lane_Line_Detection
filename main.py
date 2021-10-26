import cv2
import numpy as np

from utils import undistort, perspective_warp, sliding_window, draw_lanes


def filter_sobel(img):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    sobelx = cv2.Sobel(hls[:, :, 1], cv2.CV_64F, 1, 1)  # Take the derivative in x
    abs_sobelx = np.absolute(sobelx)  # Absolute x derivative to accentuate lines away from horizontal
    max = np.max(abs_sobelx)
    scaled_sobel = np.uint8(255 * abs_sobelx / max)

    s_thresh = (100, 255)
    sx_thresh = (15, 255)
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

    s_channel = hls[:, :, 2]
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

    filtered_image = np.zeros_like(sxbinary)
    filtered_image[(s_binary == 1) | (sxbinary == 1)] = 1

    # filtered_image = cv2.bitwise_and(filtered_image, filtered_image, mask=mask)

    return filtered_image


if __name__ == '__main__':
    VIDEO_PATH = r'videos/baseline.mp4'

    cap = cv2.VideoCapture(VIDEO_PATH, 0)
    # mask = cv2.bitwise_not(np.zeros((720, 1280), dtype="uint8")) - 254
    # cv2.rectangle(mask, (400, 200), (880, 1280), 0, -1)

    while cap.isOpened():

        # 1. Distortion correction
        ret, original_image = cap.read()

        if not ret:
            exit()

        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        undistorted_image = undistort(original_image)

        # 2. Perspective warp
        warped_image = perspective_warp(img=undistorted_image, dst_size=(1280, 720))

        # 3. Sobel filtering
        filtered_image = filter_sobel(warped_image)

        # 4. Sliding window search
        out_img, curves, lanes, ploty = sliding_window(filtered_image)

        # 5. Plot lines
        img_ = draw_lanes(original_image, curves[0], curves[1])

        cv2.imshow('Sliding window search', out_img)
        cv2.imshow('result', img_)

        if cv2.waitKey(1) & 0xff == ord('q'):
            exit()
    print('Success')
