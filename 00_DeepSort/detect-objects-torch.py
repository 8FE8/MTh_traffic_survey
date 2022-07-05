import os
from PIL import Image
import cv2
import glob
import numpy as np
import time

from yoloV5_wrapper import YOLOWrapper
from fasterRCNN_wrapper import FasterRCNNWrapper
from deep_sort import preprocessing



applySlidingWindow = True
flagProcessVideo = False
txtname = "Nadir-6-YOLO"
txtname = "Nadir-6-fasterRCNN"


if flagProcessVideo:
    video_path = str('../../') # Path to Input-Video, '0' for Webcam, #Dimension 3840 x 2160
    video_name = "Nadir-90m-6-001.MOV"
    # video_name = "PETS09-S2L1-raw.webm"
    video = cv2.VideoCapture(video_path + video_name)
else:
    path = str('../../frames-Nadir-90m-6/')

# video = cv2.VideoCapture(video_path + video_name)
object_detector = FasterRCNNWrapper()
# object_detector = YOLOWrapper()



bbox_output = str('./data/video/Output/Object-detector-' + txtname + '.txt') # Path to BBox-Output
bbbox_output_file = open(bbox_output, "w") # Open File to store BBox-Coordinates

cv2.namedWindow("Main_Frame", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Main_Frame", 1280,720)

windowSize, stepSize = 960, 800
frame_num = 0

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

    main_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    print('Frame #: ', frameId)
    start_time = time.time()

    
    bboxes, classes = [], []

    for y1 in range(0, main_frame.shape[0], stepSize):
        for x1 in range(0, main_frame.shape[1], stepSize):
                            
            if False == applySlidingWindow:    
                bboxes, classes = object_detector.detect(frame, 0, 0)
                break

            
            y2 = y1 + windowSize
            x2 = x1 + windowSize

            patch = frame[y1:y2, x1:x2]

            h,w,_ = patch.shape
            if h < 500 or w < 500:
                continue

            bboxes_patch, classes_patch = object_detector.detect(patch, x1, y1)

            # for label, score, top_left, bottom_right in predictions:
            #     cv2.rectangle(patch, tuple(top_left), tuple(bottom_right), (0,0,255), 2)
            #     cv2.putText(patch, label + " " + str(np.round(score,2)), (top_left[0], top_left[1]-3), 0, 0.5, [0, 255, 255], thickness=1, lineType=cv2.LINE_AA)

            if 0 == len(bboxes):
                bboxes = bboxes_patch
                classes = classes_patch
            else:
                bboxes = bboxes + bboxes_patch
                classes = classes + classes_patch

            # cv2.rectangle(main_frame, (x1,y1), (x2,y2), (0,255,0), 2)
            
            # cv2.imshow("patch", patch)
            # cv2.imshow("Main_Frame", main_frame)
            # cv2.waitKey(0)


   
    bboxes = np.array(bboxes)
    classes = np.array(classes)
    indices = preprocessing.non_max_suppression(bboxes, classes, 0.5)
    bboxes = [bboxes[i] for i in indices]  
    classes = [classes[i] for i in indices]  

    color = (255,0,0)
    for i, (x,y,w,h) in enumerate(bboxes):
        cv2.rectangle(main_frame, (x, y), (x+w, y+h), color, 2)
        bbbox_output_file.write("Frame: "+ str(frame_id)+", Class: {}, Coor: {},{},{},{}\n".format(classes[i], x,y,x+w,y+h))


    
    # calculate frames per second of running detections
    fps = 1 / (time.time() - start_time) #1.0
    print("FPS: %.2f" % fps)

    

    main_frame = cv2.cvtColor(main_frame, cv2.COLOR_RGB2BGR)
    cv2.imshow("Main_Frame", main_frame)
    
    frameId = frameId + 1
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        running = False
        break
cv2.destroyAllWindows()
bbbox_output_file.close() # Close BBox-Text-File
