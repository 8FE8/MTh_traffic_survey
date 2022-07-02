import os
# comment out below line to enable tensorflow logging outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from core.config import cfg
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

# deep sort imports
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet

import myutils
def read_content(xml_file: str):

    tree = ET.parse(xml_file)
    root = tree.getroot()
        
    boxes, classes = [], []
    for obje in root.findall('object'):
        objeName = obje[0].text
        objeId = None
        if 'ped' == objeName:
            objeName = 'pedestrian'
        if 'tru' == objeName:
            objeName = 'truck'
        if 'per' == objeName:
            objeName = 'person'
        if 'mot' == objeName:
            objeName = 'motor'
        if 'bic' == objeName:
            objeName = 'bicycle'
        if 'awn' == objeName:
            objeName = 'tricycle'
        if 'tri' == objeName:
            objeName = 'tricycle'

        xmin = int(obje[4][0].text)
        ymin = int(obje[4][1].text)
        xmax = int(obje[4][2].text)
        ymax = int(obje[4][3].text)
        boxes.append([xmin,ymin,xmax-xmin,ymax-ymin])
        classes.append(objeName)

    return boxes, classes

from _collections import deque
pts = [deque(maxlen=3000) for _ in range(50000)]



path = str('../../frames-Nadir-90m-6/')


# Set Parameters for Tracking
# Definition of the parameters
max_cosine_distance = 0.9 # e.g. 0.4
nn_budget = None #e.g. None
nms_max_overlap = 1.0 # e.g. 1.0


# initialize deep sort
deepsort_modelname = 'model_data/mars-small128.pb' # TF-Modell for DeepSort
encoder = gdet.create_box_encoder(deepsort_modelname, batch_size=2)
# calculate cosine distance metric
metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
# initialize tracker
tracker = Tracker(metric)


bbox_output = str('./data/video/Output/Object-tracking-Nadir-6-GT.txt') # Path to BBox-Output
bbbox_output_file = open(bbox_output, "w") # Open File to store BBox-Coordinates


cv2.namedWindow("Main_Frame", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Main_Frame", 1280,720)

for frameId in range(1,177):

    # Capture frame-by-frame
    filename = "frame" + str(frameId) + ".jpg"
    frame = cv2.imread(path + filename)
    main_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


    main_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    
    print('Frame #: ', frameId)
    start_time = time.time()

    
    #print("BBoxes : " + str(bboxes))
    bboxes, labels = read_content(path + filename.replace(".jpg", ".xml"))

    for box in bboxes:
        cv2.rectangle(main_frame, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), (255,0,0), 2)
    
    

    # encode yolo detections and feed to tracker
    features = encoder(frame, bboxes)
    detections = [Detection(bbox, 1.0, class_name, feature) for bbox, class_name, feature in zip(bboxes, labels,features)]
    
    #initialize color map
    cmap = plt.get_cmap('tab20b')
    colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]
    
    # run non-maxima supression
    boxs = np.array([d.tlwh for d in detections])
    scores = np.array([d.confidence for d in detections])
    classes = np.array([d.class_name for d in detections])
    indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
    detections = [detections[i] for i in indices]       
    
    # Call the tracker
    tracker.predict()
    tracker.update(detections)
    
    # update tracks
    for track in tracker.tracks:
        if not track.is_confirmed() or track.time_since_update > 1:
            continue 
        bbox = track.to_tlbr()
        class_name = track.get_class()
     
        color = colors[int(track.track_id) % len(colors)]
        color = [i * 255 for i in color]
                
    #Trajectories
        center = (int(((bbox[0]) + (bbox[2]))/2), int(((bbox[1])+(bbox[3]))/2))
        pts[track.track_id].append(center)
        for j in range(1, len(pts[track.track_id])):
            if pts[track.track_id][j-1] is None or pts[track.track_id][j] is None:
                continue
            thickness = 2
            #thickness = int(np.sqrt(64/float(j+1))*2)
            cv2.line(main_frame, (pts[track.track_id][j-1]), (pts[track.track_id][j]), color, thickness)
    
        bbbox_output_file.write("Frame: "+ str(frameId)+", ID: {} Class: {}, Coor: {},{},{},{}\n".format(
                                                                                    str(track.track_id),
                                                                                    class_name, 
                                                                                    int(bbox[0]),int(bbox[1]), 
                                                                                    int(bbox[2]),int(bbox[3])))

    
    # calculate frames per second of running detections
    fps = 1 / (time.time() - start_time) #1.0
    print("FPS: %.2f" % fps)

    main_frame = cv2.cvtColor(main_frame, cv2.COLOR_RGB2BGR) 
    cv2.imshow("Main_Frame", main_frame)
    
    
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break
        

cv2.destroyAllWindows()
bbbox_output_file.close() # Close BBox-Text-File
