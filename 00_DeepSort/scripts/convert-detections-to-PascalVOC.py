import cv2
import os
import argparse
from pascal_voc_writer import Writer


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


parser = argparse.ArgumentParser()
parser.add_argument('-v', '--video', required=True, help='Video Path')
parser.add_argument('-t', '--text', required=True, help='text file tracking result')
parser.add_argument('-f', '--folder',required=True, help='folder name to save outputs')
args = vars(parser.parse_args())

folder_save = args['folder']

bboxes_dict, labels_dict = read_detection(args['text'])

cap = cv2.VideoCapture(args['video'])

width_input  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height_input = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) 

cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
cv2.resizeWindow("frame", 1280,720)

if not os.path.exists(folder_save):
    os.mkdir(folder_save)

frameId = 1
while(True):
    
    # Capture the video frame
    # by frame
    ret, frame = cap.read()
    if not ret:
        break

    if frameId in bboxes_dict:
        boxes = bboxes_dict[frameId]
        labels = labels_dict[frameId]
        writer = Writer(folder_save + "/frame-" + str(frameId) + ".jog",width_input, height_input)

        for i, (x,y,w,h) in enumerate(boxes):
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255), thickness=2, lineType=cv2.LINE_AA)
            writer.addObject(labels[i], x, y, x+w, y+h)
        
        writer.save(folder_save + "/frame-" + str(frameId) + ".xml")

    # Display the resulting frame
    cv2.imshow('frame', frame)
    frameId = frameId + 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
cap.release()
cv2.destroyAllWindows()