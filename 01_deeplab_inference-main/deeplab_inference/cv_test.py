import numpy as np
import cv2 as cv

# ---- import libraries
from PIL import Image
from vidstab import VidStab
import matplotlib.pyplot as plt

# ---- import DeepLab
from model import DeepLabModel, DeepLab

model = DeepLabModel('model/frozen_inference_graph_cars.pb')

cap =cv.VideoCapture('input/video/DJI_0008.mp4')

while True:
    ret, frame = cap.read()
    if frame is None:
        break

    mask = model.run(Image.fromarray(frame)) 

    cv.imshow('Frame',mask)

    keyboard = cv.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break

cap.release()
cv.destroyAllWindows()

