import cv2
import numpy as np
import glob
from PIL import Image
import time

import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

from model import DeepLabModel, DeepLab


def create__label_colormap():
     return np.asarray([
         [0, 0, 0],
         [0, 150, 0],
         [255, 150, 150],
     ])
    

def label_to_color_image(label):
    """Adds color defined by the dataset colormap to the label.

    Args:
    label: A 2D array with integer type, storing the segmentation label.

    Returns:
    result: A 2D array with floating type. The element of the array
      is the color indexed by the corresponding element in the input label
      to the PASCAL color map.

    Raises:
    ValueError: If label is not of rank 2 or its value is larger than color
      map maximum entry.
    """
    if label.ndim != 2:
        raise ValueError('Expect 2-D input label')

    colormap = create__label_colormap()

    if np.max(label) >= len(colormap):
        raise ValueError('label value too large.')

    return colormap[label]



model = DeepLabModel('model/frozen_inference_graph_cars.pb')



flagProcessVideo = False
txtname = "Nadir-6-DeepLab"

if flagProcessVideo:
    video_path = str('../../') # Path to Input-Video, '0' for Webcam, #Dimension 3840 x 2160
    video_name = "Nadir-90m-6-001.MOV"
    # video_name = "PETS09-S2L1-raw.webm"
    video = cv2.VideoCapture(video_path + video_name)
else:
    path = str('../../../frames-Nadir-90m-6/')


cv2.namedWindow("Main_Frame", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Main_Frame", 1280,720)

cv2.namedWindow("Mask", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Mask", 1280,720)

imgCounter = 0
if not flagProcessVideo:
    imgCounter = len(glob.glob1(path,"*.jpg"))

frameId = 1
while True:

    if flagProcessVideo:
        return_value, frame = video.read()
        if not return_value:
            break
    else:
        if frameId>imgCounter:
            break
        filename = "frame" + str(frameId) + ".jpg"
        frame = cv2.imread(path + filename)
    
    print('Frame #: ', frameId)
    start_time = time.time()

    mask = model.run(Image.fromarray(frame))       
    

    fps = 1 / (time.time() - start_time) #1.0
    print("FPS: %.2f" % fps)

    cv2.imshow("Main_Frame", frame)
    cv2.imshow("Mask", label_to_color_image(mask).astype(np.uint8))

    frameId = frameId + 1
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break