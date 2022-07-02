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

from deep_sort import preprocessing
import myutils


applySlidingWindow = True


path = str('../../frames-Nadir-90m-6/') # Path to Input-Video, '0' for Webcam, #Dimension 3840 x 2160
txtname = "Nadir-6-YOLO"

thresh_iou = float(0.45) # IOU-Threshold, e.g. 0.45
thresh_score = float(0.50) # Score-Threshold, e.g. 0.50

weights_yolo = './checkpoints/yolov4-416'
yolo_width, yolo_height = 416, 416


# Load Object-Detetion Model
saved_model_loaded = tf.saved_model.load(weights_yolo, tags=[tag_constants.SERVING])
infer = saved_model_loaded.signatures['serving_default']


bbox_output = str('./data/video/Output/Object-detector-' + txtname + '.txt') # Path to BBox-Output
bbbox_output_file = open(bbox_output, "w") # Open File to store BBox-Coordinates

cv2.namedWindow("Main_Frame", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Main_Frame", 1280,720)

windowSize, stepSize = 1000, 800

for frameId in range(1,177):

    # Capture frame-by-frame
    filename = "frame" + str(frameId) + ".jpg"
    frame = cv2.imread(path + filename)
    main_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    main_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    print('Frame #: ', frameId)
    start_time = time.time()

    bboxes, classes = [], []
    for y1 in range(0, main_frame.shape[0], stepSize):
        for x1 in range(0, main_frame.shape[1], stepSize):

            if False == applySlidingWindow:    
                 bboxes, classes = myutils.runYoloV4(frame, infer, cfg, thresh_iou, thresh_score, 0, 0)
                 break


            y2 = y1 + windowSize
            x2 = x1 + windowSize

            patch = frame[y1:y2, x1:x2]

            # x1,x2,y1,y2 = 1750, 2750, 500, 1500
            # patch = frame[y1:y2, x1:x2]

            h,w,_ = patch.shape
            if h < 300 or w < 300:
                continue


            bboxes_patch, classes_patch = myutils.runYoloV4(patch, infer, cfg, thresh_iou, thresh_score, x1, y1)
            if 0 == len(bboxes):
                bboxes = bboxes_patch
                classes = classes_patch
            else:
                bboxes = bboxes + bboxes_patch
                classes = classes_patch + classes


            # cv2.rectangle(main_frame, (x1,y1), (x2,y2), (0,255,0), 2)
            # break

            # for i,box in enumerate(bboxes_patch):
            #     x,y,w,h = box[0], box[1], box[2], box[3]
            #     cv2.rectangle(patch, (x,y), (x+w, y+h), (0,0,255), 2)
            #     cv2.putText(patch, classes_patch[i], (x,y-5), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2, lineType=cv2.LINE_AA)

            
            # cv2.imshow("patch", patch)
            # cv2.imshow("Main_Frame", main_frame)
            # cv2.waitKey(0)


   
        
    bboxes = np.array(bboxes)
    classes = np.array(classes)
    indices = preprocessing.non_max_suppression(bboxes, classes, 0.5)
    bboxes = [bboxes[i] for i in indices]  
    classes = [classes[i] for i in indices]  


    color = (255,0,0)
    for i,box in enumerate(bboxes):
        # print(box)
        x,y,w,h = box[0], box[1], box[2], box[3]

        if w > 400 or h > 400:
            continue
        
        cv2.rectangle(main_frame, (x,y), (x+w, y+h), color, 2)
        bbbox_output_file.write("Frame: "+ str(frameId)+", Class: {}, Coor: {},{},{},{}\n".format(classes[i], x,y,x+w,y+h))

    
    # calculate frames per second of running detections
    fps = 1 / (time.time() - start_time) #1.0
    print("FPS: %.2f" % fps)

    

    main_frame = cv2.cvtColor(main_frame, cv2.COLOR_RGB2BGR)
    cv2.imshow("Main_Frame", main_frame)
    
    
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        running = False
        break
cv2.destroyAllWindows()
bbbox_output_file.close() # Close BBox-Text-File
