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
from tensorflow.python.saved_model import tag_constants
from core.config import cfg
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt

# deep sort imports
from deep_sort import preprocessing, nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from vidstab import VidStab

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



video_path = str('../../') # Path to Input-Video, '0' for Webcam, #Dimension 3840 x 2160
video_name = "Nadir-90m-6-001.MOV"


detection_txt_file = './data/video/Output/Object-detector-' + video_name[:-4] + "-fasterRCNN.txt"
bboxes_dict, labels_dict = myutils.read_detection(detection_txt_file)

# Set Parameters for Tracking
# Definition of the parameters
max_cosine_distance = 0.9 # e.g. 0.4
nn_budget = None #e.g. None
nms_max_overlap = 1.0 # e.g. 1.0

video = cv2.VideoCapture(video_path + video_name)



# initialize deep sort
deepsort_modelname = 'model_data/mars-small128.pb' # TF-Modell for DeepSort
encoder = gdet.create_box_encoder(deepsort_modelname, batch_size=2)
# calculate cosine distance metric
metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
# initialize tracker
tracker = Tracker(metric)


bbox_output = str('./data/video/Output/Object-tracking-' + video_name[:-4] + ".txt") # Path to BBox-Output
bbbox_output_file = open(bbox_output, "w") # Open File to store BBox-Coordinates


cv2.namedWindow("Main_Frame", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Main_Frame", 1280,720)

frame_num = 0
while True:
    # Capture frame-by-frame
    return_value, frame = video.read()
    if not return_value:
        break

    main_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    
    frame_num +=1
    print('Frame #: ', frame_num)
    start_time = time.time()

    
    #print("BBoxes : " + str(bboxes))
    bboxes, labels = bboxes_dict[frame_num], labels_dict[frame_num]

    # for box in bboxes:
    #     cv2.rectangle(main_frame, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), (255,0,0), 2)
    
    

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
    
        bbbox_output_file.write("Frame: "+ str(frame_num)+", ID: {} Class: {}, Coor: {},{},{},{}\n".format(
                                                                                    str(track.track_id),
                                                                                    class_name, 
                                                                                    int(bbox[0]),int(bbox[1]), 
                                                                                    int(bbox[0])+int(bbox[2]),
                                                                                    int(bbox[1]) + int(bbox[3])))

    
    # calculate frames per second of running detections
    fps = 1 / (time.time() - start_time) #1.0
    print("FPS: %.2f" % fps)

    main_frame = cv2.cvtColor(main_frame, cv2.COLOR_RGB2BGR) 
    cv2.imshow("Main_Frame", main_frame)
    
    
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break
        

cv2.destroyAllWindows()
bbbox_output_file.close() # Close BBox-Text-File
