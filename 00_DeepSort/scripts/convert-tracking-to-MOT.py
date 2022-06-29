import cv2
import os
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
            print(label)
            if 'person' != label and 'pedestrian' != label:
                continue
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


parser = argparse.ArgumentParser()
parser.add_argument('-v', '--video', required=True, help='Video Path')
parser.add_argument('-t', '--text', required=True, help='text file tracking result')
parser.add_argument('-f', '--folder',required=True, help='folder name')
args = vars(parser.parse_args())

folder_tracking = args['folder']

bboxes_dict, labels_dict = read_tracker_results(args['text'])


cap = cv2.VideoCapture(args['video'])

cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
cv2.resizeWindow("frame", 1280,720)

if not os.path.exists(folder_tracking):
    os.mkdir(folder_tracking)

frameId = 1
while(True):
    
    # Capture the video frame
    # by frame
    ret, frame = cap.read()
    if not ret:
        break

    frameFolder = "seq" + str(frameId).zfill(4)
    
    

    if frameId in bboxes_dict:
        boxes = bboxes_dict[frameId]

        fileLabel = open(folder_tracking + "/" + frameFolder + ".txt","w+") 
        for idx,x,y,w,h in boxes:
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255), thickness=2, lineType=cv2.LINE_AA)
            cv2.putText(frame, str(idx), (x,y-5), 1, 1, (255,0,0))
            txt1 = "{} {} {} {} {} {} 1 1 1".format(frameId, idx, x, y, w, h)
            fileLabel.write( txt1 + "\n")
        
        fileLabel.close()

    # Display the resulting frame
    cv2.imshow('frame', frame)
    frameId = frameId + 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
cap.release()
cv2.destroyAllWindows()