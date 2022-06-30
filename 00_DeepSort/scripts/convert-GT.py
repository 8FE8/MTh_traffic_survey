import cv2
import motmetrics
import motmetrics as mm
import numpy as np
import os
import xml.etree.ElementTree as ET
import argparse


def read_content(xml_file: str):

    tree = ET.parse(xml_file)
    root = tree.getroot()

    boxes = {}
    for frame in root.iter('frame'):
        
        frameId = int(frame.attrib['number'])
        boxes[frameId] = []
        for objectlist in frame.findall('objectlist'):
            for obje in objectlist.findall('object'):
                objeId = int(obje.attrib['id'])
                box = obje.find('box')
                xc, yc, w, h = float(box.attrib['xc']), float(box.attrib['yc']), float(box.attrib['w']), float(box.attrib['h'])
                x,y,w,h = int(xc-w/2), int(yc-h/2), int(w), int(h)
                boxes[frameId].append([objeId, x,y,w,h])

    return boxes


parser = argparse.ArgumentParser()
parser.add_argument('-x', '--xml', required=True, help='xml file')
parser.add_argument('-v', '--video', required=True, help='vide file')
parser.add_argument('-f', '--folder',required=True, help='folder name to create GT')
args = vars(parser.parse_args())


folderGT = args['folder']


cap = cv2.VideoCapture(args['video'])

bboxes = read_content(args['xml'])

if not os.path.exists(folderGT):
    os.mkdir(folderGT)

frameId = 0
while(True):
    
    # Capture the video frame
    # by frame
    ret, frame = cap.read()
    if not ret:
        break

    frameFolder = "seq" + str(frameId).zfill(4)

    os.mkdir(folderGT + "/" + frameFolder)
    os.mkdir(folderGT + "/" + frameFolder + "/gt/")

    fileLabel = open(folderGT + "/" + frameFolder + "/gt/gt.txt","w+") 

    for idx, x,y,w,h in bboxes[frameId]:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255), thickness=2, lineType=cv2.LINE_AA)
        txt1 = "{} {} {} {} {} {} 1 1 1".format(frameId, idx, x, y, w, h)
        fileLabel.write( txt1 + "\n")
    
    fileLabel.close()

    # Display the resulting frame
    cv2.imshow('frame', frame)
    frameId = frameId + 1
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
  
cap.release()
cv2.destroyAllWindows()