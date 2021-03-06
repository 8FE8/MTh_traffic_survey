import cv2
import numpy as np
import core.utils as utils

import tensorflow as tf

def runYoloV4(frame, infer, cfg, thresh_iou, thresh_score, x_offset, y_offset, yolo_width = 416, yolo_height = 416):

    image_data = cv2.resize(frame, (yolo_width, yolo_height))
    image_data = image_data / 255.
    image_data = image_data[np.newaxis, ...].astype(np.float32)

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
    iou_threshold=thresh_iou,
    score_threshold=thresh_score)
                        

    # convert data to numpy arrays and slice out unused elements
    num_objects = valid_detections.numpy()[0]
    bboxes = boxes.numpy()[0]
    bboxes = bboxes[0:int(num_objects)]
    scores = scores.numpy()[0]
    scores = scores[0:int(num_objects)]
    classes = classes.numpy()[0]
    classes = classes[0:int(num_objects)]
    

    # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, width, height
    original_h, original_w, _ = frame.shape
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
    # names = np.array(names)
    # count = len(names)
                
    # delete detections that are not in allowed_classes
    bboxes = np.delete(bboxes, deleted_indx, axis=0) # [175. 619. 123.  77.] --> xmin, ymin, width, height
    scores = np.delete(scores, deleted_indx, axis=0) # [0.98072225 0.7607064]

    bboxes.tolist()
    boxesFinal = []
    for box in bboxes:
        box[0] = box[0] + x_offset
        box[1] = box[1] + y_offset
        boxesFinal.append([int(box[0]), int(box[1]), int(box[2]), int(box[3])])

    #print("BBoxes : " + str(bboxes))
    return boxesFinal, names


def read_detection(detection_file):

    bboxes = {}
    labels = {}
    with open(detection_file) as f:
        lines = f.readlines()
        for line in lines:
            idx = line.find(",")
            frame_id = int(line[7:idx])
            idx = line.find("Class:")+7
            idx_end = line[idx:].find(",") + idx
            label = line[idx:idx_end]
            idx = line.find("Coor:")+6
            x1,y1,x2,y2 = line[idx:].split(',')
            x1,y1,x2,y2 = int(x1), int(y1), int(x2), int(y2)
            if frame_id in bboxes:
                bboxes[frame_id].append([x1,y1,x2-x1,y2-y1])
                labels[frame_id].append(label)
            else:
                bboxes[frame_id] = [[x1,y1,x2-x1,y2-y1]]
                labels[frame_id] = [label]
            
            # print(frame_id, " ", x1,y1,x2,y2)
            # break
    return bboxes, labels




def checkBorders(startX, startY, endX, endY, width, height):
        if (startX < 0):
            startX = 0
        
        if (startY < 0): 
            startY = 0
        
        if (endX >= width):
            endX = width - 1
        
        if (endY >= height):
            endY = height - 1

        return startX, startY, endX, endY

def enlarge(startX, startY, endX, endY, offset = 5):
    return startX-offset, startY-offset, endX+offset*2, endY+offset*2


def maskToBoxes(mask):
    height, width = mask.shape
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    motion = np.zeros(mask.shape, np.uint8)
    for cnt in contours:
        if cv2.contourArea(cnt) < 40:
            continue

        (x, y, w, h) = cv2.boundingRect(cnt)
        x,y,w,h = enlarge(x,y,w,h)
        x,y,w,h = checkBorders(x,y,w,h,width, height)
        motion[y:y+h, x:x+w] = 255
    
    contours, _ = cv2.findContours(motion, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        boxes.append((x, y, x+w, y+h))
    return boxes



