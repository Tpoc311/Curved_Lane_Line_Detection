from typing import List

import cv2
from numpy import ndarray


def cut_video(video_path: str) -> List[ndarray]:
    cap = cv2.VideoCapture(video_path)
    success, image = cap.read()

    images = []

    if cap.isOpened():
        while success:
            images.append(image)
            success, image = cap.read()
        return images
    else:
        print('Video opening error!')
        return -1
