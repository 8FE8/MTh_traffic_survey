from __future__ import print_function
import cv2 as cv
import argparse
import numpy as np
import fiona
import pandas as pd
from collections import defaultdict

import pprint
#from VP1_CentroidTracker import CentroidTracker


#ct = CentroidTracker()

#Create Shapefile
#structure of fiona object

#fiona.open( fp, mode='r', driver=None, schema=None, crs=None, encoding=None, layer=None, vfs=None, enabled_drivers=None, crs_wkt=None, **kwargs, )
# define schema
schema = {
    'geometry':'LineString',
    'properties':[('Name','str')]
}
#open a fiona object
lineShp = fiona.open('prova_shape.shp', mode='w', driver='ESRI Shapefile',
          schema = schema, crs = "EPSG:4326")

#get list of points
dates_dict = defaultdict(list)
xyList = []
rowName = ''

#video path
video_path = '/Users/scuola/PycharmProjects/VP1/exclude/DJI_0008.MP4'

parser = argparse.ArgumentParser(description='This program shows how to use background subtraction methods provided by \
                                              OpenCV. You can process both videos and images.')
parser.add_argument('--input', type=str, help='Path to a video or a sequence of image.', default=video_path)
parser.add_argument('--algo', type=str, help='Background subtraction method (KNN, MOG2).', default='MOG2')
args = parser.parse_args()

## [create]
#create Background Subtractor objects
if args.algo == 'MOG2':
    backSub = cv.createBackgroundSubtractorMOG2()
else:
    backSub = cv.createBackgroundSubtractorKNN()
## [create]

## [capture]
capture = cv.VideoCapture(cv.samples.findFileOrKeep(args.input))
if not capture.isOpened():
    print('Unable to open: ' + args.input)
    exit(0)
## [capture]

# Parameters for Shi-Tomasi corner detection
feature_params = dict(maxCorners = 300, qualityLevel = 0.2, minDistance = 7, blockSize = 7)
# Parameters for Lucas-Kanade optical flow
lk_params = dict(winSize = (15,15), maxLevel = 2, criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

# Variable for color to draw optical flow track
# Create some random colors
color = np.random.randint(0,255,(300,3))
# ret = a boolean return value from getting the frame, first_frame = the first frame in the entire video sequence
ret, first_frame = capture.read()
# Converts frame to grayscale because we only need the luminance channel for detecting edges - less computationally expensive
prev_gray = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)
# Finds the strongest corners in the first frame by Shi-Tomasi method - we will track the optical flow for these corners
# https://docs.opencv.org/3.0-beta/modules/imgproc/doc/feature_detection.html#goodfeaturestotrack
prev = cv.goodFeaturesToTrack(prev_gray, mask = None, **feature_params)
# Creates an image filled with zero intensities with the same dimensions as the frame - for later drawing purposes
mask = np.zeros_like(first_frame)
frame_counter= 0

path =[]
id = 1

while True:
    ret, frame = capture.read()
    frame_counter+=1
    print("frame:", frame_counter)


    if frame is None:
        print("finished")
        loop=0
        for i in path:
            rowDict = {
               'geometry': {'type': 'LineString',
                             'coordinates': path[loop][-1]},
                'properties': {'Name': path[loop][0]}}
           # print(rowDict);
            lineShp.write(rowDict)
            loop=loop +1
        break
    ## [apply]
    #update the background model
    fgMask = backSub.apply(frame)
    ## [apply]

    # Converts each frame to grayscale - we previously only converted the first frame to grayscale

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # In each iteration, calculate absolute difference between current frame and reference frame
    #difference = cv.absdiff(gray, fgMask)
    kernel_dil = np.ones((20, 20), np.uint8)
    kernel_er = np.ones((4, 4), np.uint8)
    # Apply thresholding to eliminate noise
    fgMask = cv.GaussianBlur(fgMask, (11,11), cv.BORDER_ISOLATED)
    thresh = cv.threshold(fgMask, 70, 255, cv.THRESH_BINARY)[1]
    thresh = cv.erode(thresh, kernel_er, iterations=3)
    thresh = cv.dilate(thresh, kernel_dil, iterations=3)

    # Find contours, obtain bounding box, extract and save ROI
    ROI_number = 0
    cnts = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    rects = []
    for c in cnts:
        x, y, w, h = cv.boundingRect(c)
        cv.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
        ROI = thresh[y:y + h, x:x + w]
        cv.putText(thresh, str(ROI_number), (x, y),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
        rects.append([x,y,x+w,y+h])
        ROI_number += 1
        #cv.imwrite('ROI_{}.png'.format(ROI_number), ROI)

    ## [display_frame_number]
    #get the frame number and write it on the current frame
    cv.rectangle(frame, (10, 2), (100,20), (255,255,255), -1)
    cv.putText(frame, str(capture.get(cv.CAP_PROP_POS_FRAMES)), (15, 15),
               cv.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
    ## [display_frame_number]

    keyboard = cv.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break


    # Calculates sparse optical flow by Lucas-Kanade method
    # https://docs.opencv.org/3.0-beta/modules/video/doc/motion_analysis_and_object_tracking.html#calcopticalflowpyrlk
    prev = cv.goodFeaturesToTrack(prev_gray, mask = None, **feature_params)
    next, status, error = cv.calcOpticalFlowPyrLK(prev_gray, gray, prev, None, **lk_params)
    # Selects good feature points for previous position
    good_old = prev[status == 1].astype(int)
    # Selects good feature points for next position
    good_new = next[status == 1].astype(int)
    # Draws the optical flow tracks

    for i, (new, old) in enumerate(zip(good_new, good_old)):
        if len(path)==0:
            #print("lunghezza 000000000")
            path.append([id,frame_counter,[tuple(new)]])
        loop = 0
        check = False
        for z in path:
            #print("im a for")
            #print(z)
            #print(tuple(old))
            #print(tuple(path[loop][-1][-1]))
            test=tuple(i-j for i, j in zip(tuple(old),tuple(path[loop][-1][-1])))
            frame_diff=frame_counter-path[loop][-2]-1
            print(frame_diff)
            if frame_diff==0:
                check = True
                break
            elif ((-50<test[0]<50) and (-50<test[1]<50)) and (frame_diff<5):
                #print(test)
                path[loop][-1].append(tuple(new))
                path[loop][-2]=frame_counter
                #print("im an if")
                #print(path)
                check = True
                #print("break")
                break

            loop = loop + 1

        #print("finito")
        if not check:
            id =id+1
            path.append([id,frame_counter,[tuple(new)]])
            check = False
            #print("im an else")
            #print(path)

        #print(id, loop)
        #cv.putText(frame, str(id), tuple(new),
                   #cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

        # Returns a contiguous flattened array as (x, y) coordinates for new point
        a, b = new.ravel()

        # Returns a contiguous flattened array as (x, y) coordinates for old point
        c, d = old.ravel()
        # Draws line between new and old position with green color and 2 thickness
        #dates_dict[i].append((a, b))
        mask = cv.line(mask, (a, b), (c, d), color[i].tolist(), 2)
        # Draws filled circle (thickness of -1) at new position with green color and radius of 3
        #frame = cv.circle(frame, (a, b), 3, color, -1)
    # Overlays the optical flow tracks on the original frame
    output = cv.add(frame, mask)
    # Updates previous frame
    prev_gray = gray.copy()
    # Updates previous good feature points
    prev = good_new.reshape(-1, 1, 2)

    # update our centroid tracker using the computed set of bounding
    # box rectangles
    #print(path)
    #objects = ct.update(rects)
    #if objects != None:
        #for (objectID, centroid) in objects.items():
            # draw both the ID of the object and the centroid of the
            # object on the output frame
            #text = "ID {}".format(objectID)
            #cv.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                       # cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            #cv.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)


    # Opens a new window and displays the output frame
    #cv.imshow('Frame', frame)
    #cv.imshow('FG Mask', fgMask)
    #cv.imshow('tresh', thresh)
    cv.imshow("sparse optical flow", output)
    # Frames are read by intervals of 10 milliseconds. The programs breaks out of the while loop when the user presses the 'q' key
    if cv.waitKey(10) & 0xFF == ord('q'):
        break