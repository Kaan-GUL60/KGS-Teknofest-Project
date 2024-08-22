import cv2
from ultralytics import YOLO
import supervision as sv
import numpy as np
import argparse


def main():

    cap = cv2.VideoCapture(1)

    model = YOLO("best.pt")

    while True:
        ret, frame = cap.read()

        result = model(frame, show=True)


        #cv2.imshow("yolov8", frame)

        if (cv2.waitKey(30) == 27):
            break


if __name__ == "__main__":
    main()