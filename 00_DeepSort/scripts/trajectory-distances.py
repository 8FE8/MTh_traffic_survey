from turtle import distance
import cv2
import os
import math
import numpy as np
import argparse

def getNumbers(just):
    x = ''.join(filter(str.isdigit, str(just)))
    return int(x)


def read_tracker_results(detection_file):

    bboxes = {}
    labels = {}
    with open(detection_file) as f:
        lines = f.readlines()
        for line in lines:
            idx = line.find(",")
            frame_id = int(line[7:idx])
            idx_start = line.find("ID:")+4
            idx_end = line.find(" Class:")
            tracker_id  = int(line[idx_start:idx_end])
            idx = line.find("Class:")+7
            idx_end = line[idx:].find(",") + idx
            label = line[idx:idx_end]
            # if 'person' != label and 'pedestrian' != label:
            #     continue
            idx = line.find("Coor:")+6
            x1,y1,x2,y2 = line[idx:].split(',')
            x1,y1,x2,y2 = int(x1), int(y1), int(x2), int(y2)
            if frame_id in bboxes:
                bboxes[frame_id].append([tracker_id, x1,y1,x2-x1,y2-y1])
                labels[frame_id].append(label)
            else:
                bboxes[frame_id] = [[tracker_id, x1,y1, x2-x1, y2-y1]]
                labels[frame_id] = [label]
            
            # print(frame_id, " ", x1,y1,x2,y2)
            # break
    return bboxes, labels



txtname = "Object-tracking-Nadir-6-YOLO"
# txtname = "Motion-tracking-Nadir-6-MOG2"
folder_tracking = "Nadir-6-GT"

text_path = "../data/video/Output/" + txtname + ".txt"
bboxes_dict, labels_dict = read_tracker_results(text_path)

text_path = "../data/video/Output/Object-tracking-" + folder_tracking + ".txt"
bboxes_dict_gt, labels_dict_gt = read_tracker_results(text_path)

folderGT = folder_tracking
path = str('../../../frames-Nadir-90m-6/')

cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
cv2.resizeWindow("frame", 1280,720)

if not os.path.exists(folder_tracking):
    os.mkdir(folder_tracking)


totalGT, totalFound = 0, 0
center_distances = []
for frameId in range(1,177):

    # Capture frame-by-frame
    filename = "frame" + str(frameId) + ".jpg"
    frame = cv2.imread(path + filename)

    if frameId in bboxes_dict_gt:
        boxes_gt = bboxes_dict_gt[frameId]
        totalGT = totalGT + len(boxes_gt)

    if frameId in bboxes_dict:
        boxes = bboxes_dict[frameId]
        boxes_gt = bboxes_dict_gt[frameId]

        totalFound = totalFound + len(boxes)

        for idx,x,y,w,h in boxes:
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255), thickness=2, lineType=cv2.LINE_AA)
            cv2.putText(frame, str(idx), (x,y-5), 0, 2, (0,0,255),2 )

            xd, yd = x+w//2, y+h//2
            distance = 125
            for idx,x,y,w,h in boxes_gt:
                xg, yg = x+w//2, y+h//2
                dist = math.sqrt((xg-xd)*(xg-xd) + (yg-yd)*(yg-yd))
                if dist < distance:
                    distance = dist
            
            if distance < 125:
                center_distances.append(distance)
        

    # Display the resulting frame
    cv2.imshow('frame', frame)
    frameId = frameId + 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
cv2.destroyAllWindows()

print("total GT: %d  Found: %d  center_distances count %d mean: %.2f" %(totalGT, totalFound,len(center_distances), np.mean(center_distances)))