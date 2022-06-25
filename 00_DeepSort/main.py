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
import matplotlib.pyplot as plt


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

subset_x_start = int(1750) # Only needed for single tile processing, Left top corner (x = Value horizontally)
subset_y_start = int(500) # Only needed for single tile processing, Left top corner (y = Value vertically)
size_of_subset = int(1000) #Dimension of Subsampling Frame, e.g. 416

size_output = (size_of_subset,size_of_subset)
iou = float(0.45) # IOU-Threshold, e.g. 0.45
score = float(0.50) # Score-Threshold, e.g. 0.50

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


# initialize deep sort
deepsort_modelname = 'model_data/mars-small128.pb' # TF-Modell for DeepSort
encoder = gdet.create_box_encoder(deepsort_modelname, batch_size=2)
# calculate cosine distance metric
metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
# initialize tracker
tracker = Tracker(metric)

# Load Object-Detetion Model
saved_model_loaded = tf.saved_model.load(weights_yolo, tags=[tag_constants.SERVING])
infer = saved_model_loaded.signatures['serving_default']


bbox_output = str('./data/video/Output/D1-DJI_0001--Detektion-bbox_output.txt') # Path to BBox-Output
bbbox_output_file = open(bbox_output, "w") # Open File to store BBox-Coordinates

frame_num = 0
while True:
    # Capture frame-by-frame
    return_value, main_frame = video.read()
    main_frame = cv2.cvtColor(main_frame, cv2.COLOR_BGR2RGB)
    main_frame_edit = main_frame
    sub_frame = main_frame[subset_y_start:subset_y_start + size_of_subset,
                           subset_x_start:subset_x_start + size_of_subset]

    

    
    frame_num +=1
    print('Frame #: ', frame_num)
    image_data = cv2.resize(sub_frame, (yolo_width, yolo_height))
    image_data = image_data / 255.
    image_data = image_data[np.newaxis, ...].astype(np.float32)
    start_time = time.time()

    batch_data = tf.constant(image_data)
    pred_bbox = infer(batch_data)
    for key, value in pred_bbox.items():
        boxes = value[:, :, 0:4]
        pred_conf = value[:, :, 4:]
    
    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
    boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
    scores=tf.reshape(
        pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
    max_output_size_per_class=500, #50
    max_total_size=500, #50
    iou_threshold=iou,
    score_threshold=score)
                        

    # convert data to numpy arrays and slice out unused elements
    num_objects = valid_detections.numpy()[0]
    bboxes = boxes.numpy()[0]
    bboxes = bboxes[0:int(num_objects)]
    scores = scores.numpy()[0]
    scores = scores[0:int(num_objects)]
    classes = classes.numpy()[0]
    classes = classes[0:int(num_objects)]
    

    # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, width, height
    original_h, original_w, _ = sub_frame.shape
    bboxes = utils.format_boxes(bboxes, original_h, original_w)
    
    # store all predictions in one parameter for simplicity when calling functions
    pred_bbox = [bboxes, scores, classes, num_objects]
    
    # read in all class names from config
    class_names = utils.read_class_names(cfg.YOLO.CLASSES)
    
    # by default allow all classes in .names file
    allowed_classes = list(class_names.values())
            
    # custom allowed classes (uncomment line below to customize tracker for only people)
    #allowed_classes = ['person', 'car', 'truck', 'bus', 'motorbike', 'bicycle']

    # loop through objects and use class index to get class name, allow only classes in allowed_classes list
    names = []
    deleted_indx = []
    for i in range(num_objects):
        class_indx = int(classes[i])
        class_name = class_names[class_indx]
        if class_name not in allowed_classes:
            deleted_indx.append(i)
        else:
            names.append(class_name)
    names = np.array(names)
    count = len(names)
                
    cv2.putText(sub_frame, "Objects being tracked: {}".format(count), (5, 35), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0), 2)
    print("Objects being tracked: {}".format(count))
    # delete detections that are not in allowed_classes
    bboxes = np.delete(bboxes, deleted_indx, axis=0) # [175. 619. 123.  77.] --> xmin, ymin, width, height
    scores = np.delete(scores, deleted_indx, axis=0) # [0.98072225 0.7607064]
    
    #print("BBoxes : " + str(bboxes))
    
    # encode yolo detections and feed to tracker
    features = encoder(sub_frame, bboxes)
    detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, names, features)]
    
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
            cv2.line(sub_frame, (pts[track.track_id][j-1]), (pts[track.track_id][j]), color, thickness)
    
    # draw bbox on screen
        bbox_topleft = (subset_x_start + int(bbox[0]), subset_y_start + int(bbox[1]))
        bbox_bottomright = (subset_x_start +  int(bbox[2]), subset_y_start  + int(bbox[3]))
        bbox_topleft_fill = (subset_x_start + int(bbox[0]), subset_y_start + int(bbox[1]-30))
        bbox_bottomrigh_fill = (subset_x_start + int(bbox[0])+(len(class_name)+len(str(track.track_id)))*17, subset_y_start + int(bbox[1]))
        bbox_text_position = (subset_x_start + int(bbox[0]), subset_y_start + int(bbox[1]-10))
        cv2.rectangle(main_frame_edit, bbox_topleft, 
                      bbox_bottomright, color, 2)
        cv2.rectangle(main_frame_edit, bbox_topleft_fill, 
                      bbox_bottomrigh_fill, color, -1)
        cv2.putText(main_frame_edit, class_name + "-" + str(track.track_id),bbox_text_position,0, 0.75, (255,255,255),2)
    
    # Print and Store Details of BBox in Console and File

        print("Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(str(track.track_id), 
                                                                                            class_name, 
                                                                                            (subset_x_start + int(bbox[0]), 
                                                                                             subset_y_start + int(bbox[1]), 
                                                                                             subset_x_start + int(bbox[2]), 
                                                                                             subset_y_start + int(bbox[3]))))
        
        bbbox_output_file.write("Frame-Number: "+ str(frame_num)+", Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {} \n".format(str(track.track_id), 
                                                                                            class_name, 
                                                                                            (subset_x_start + int(bbox[0]), 
                                                                                             subset_y_start + int(bbox[1]), 
                                                                                             subset_x_start + int(bbox[2]), 
                                                                                             subset_y_start + int(bbox[3]))))
    
    # calculate frames per second of running detections
    fps = 1 / (time.time() - start_time) #1.0
    print("FPS: %.2f" % fps)
    result = np.asarray(sub_frame)
    result = cv2.cvtColor(sub_frame, cv2.COLOR_RGB2BGR)

    

    cv2.imshow("Output Video", result)

    #Show single subtile of main frame
    main_frame_edit = cv2.rectangle(main_frame_edit, (subset_x_start,subset_y_start),
                                    (subset_x_start + size_of_subset,subset_y_start + size_of_subset),
                                    (255,0,0), 5) 
    main_frame_tile = np.asarray(main_frame_edit)
    main_frame_tile = cv2.cvtColor(main_frame_tile, cv2.COLOR_RGB2BGR)
    cv2.namedWindow("Main_Frame", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Main_Frame", 1920,1080)
    cv2.imshow("Main_Frame", main_frame_tile)
    
    
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        running = False
        break
cv2.destroyAllWindows()
bbbox_output_file.close() # Close BBox-Text-File
