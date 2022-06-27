import os
# comment out below line to enable tensorflow logging outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from core.config import cfg
from PIL import Image
import cv2
import numpy as np


import myutils

def get_anchors(anchors_path, tiny=False):
    anchors = np.array(anchors_path)
    return anchors.reshape(3, 3, 2)

def read_class_names(class_file_name):
    names = {}
    with open(class_file_name, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    return names

def format_boxes(bboxes, image_height, image_width):
    for box in bboxes:
        ymin = int(box[0] * image_height)
        xmin = int(box[1] * image_width)
        ymax = int(box[2] * image_height)
        xmax = int(box[3] * image_width)
        width = xmax - xmin
        height = ymax - ymin
        box[0], box[1], box[2], box[3] = xmin, ymin, width, height
    return bboxes


from _collections import deque
pts = [deque(maxlen=3000) for _ in range(50000)]


video_path = str('../../Nadir-90m-6-001.MOV') # Path to Input-Video, '0' for Webcam, #Dimension 3840 x 2160

thresh_iou = float(0.45) # IOU-Threshold, e.g. 0.45
thresh_score = float(0.50) # Score-Threshold, e.g. 0.50

# Set Parameters for Tracking
# Definition of the parameters
max_cosine_distance = 0.9 # e.g. 0.4
nn_budget = None #e.g. None
nms_max_overlap = 1.0 # e.g. 1.0

weights_yolo = './checkpoints/yolov4-416'
yolo_width, yolo_height = 416, 416

video = cv2.VideoCapture(video_path)

# get dimension of video input
width_input  = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))   # width`
height_input = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)) 



# Load Object-Detetion Model
saved_model_loaded = tf.saved_model.load(weights_yolo, tags=[tag_constants.SERVING])
infer = saved_model_loaded.signatures['serving_default']


bbox_output = str('./data/video/Output/Object-detector-bbox_output.txt') # Path to BBox-Output
bbbox_output_file = open(bbox_output, "w") # Open File to store BBox-Coordinates

cv2.namedWindow("Main_Frame", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Main_Frame", 1280,720)

windowSize, stepSize = 1000, 500
frame_num = 0
while True:
    # Capture frame-by-frame
    return_value, frame = video.read()
    main_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


    
    frame_num +=1
    print('Frame #: ', frame_num)
    start_time = time.time()

    bboxes, scores = [], []
    for y1 in range(0, main_frame.shape[0], stepSize):
        for x1 in range(0, main_frame.shape[1], stepSize):
                            
            y2 = y1 + windowSize
            x2 = x1 + windowSize

            patch = frame[y1:y2, x1:x2]
            bboxes_patch, scores_patch = myutils.runYoloV4(patch, infer, cfg, thresh_iou, thresh_score, x1, y1,)

            if 0 == len(bboxes):
                bboxes = bboxes_patch
                scores = scores_patch
            else:
                bboxes = bboxes + bboxes_patch
                scores = scores_patch + scores

            cv2.rectangle(main_frame, (x1,y1), (x2,y2), (0,255,0), 2)
            
            # cv2.imshow("patch", patch)
            # cv2.imshow("Main_Frame", main_frame)
            # cv2.waitKey(0)


   
        
    
    color = (255,0,0)
    for box in bboxes:
        # print(box)
        cv2.rectangle(main_frame, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), color, 2)
    
    
    # calculate frames per second of running detections
    fps = 1 / (time.time() - start_time) #1.0
    print("FPS: %.2f" % fps)

    

    main_frame = cv2.cvtColor(main_frame, cv2.COLOR_RGB2BGR)
    cv2.namedWindow("Main_Frame", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Main_Frame", 1280,720)
    cv2.imshow("Main_Frame", main_frame)
    
    
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        running = False
        break
cv2.destroyAllWindows()
bbbox_output_file.close() # Close BBox-Text-File
