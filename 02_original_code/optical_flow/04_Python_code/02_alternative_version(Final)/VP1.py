from __future__ import print_function
import cv2
import argparse
import time
import numpy as np
import fiona
from VP1_CentroidTracker import CentroidTracker, progressBar, showWindows, georeference
import VP1_param
from collections import defaultdict
from math import sqrt
import os

# Set inizializaion Parameters to run the script
parser = argparse.ArgumentParser(description='This program detect path of veichles from a stabilized Drone video')
parser.add_argument('--input', type=str, help='Path to a video or a sequence of image.', default=VP1_param.video_path)
parser.add_argument('--algo_BS', type=str, help='Background subtraction method (KNN, MOG2).', default='MOG2')
parser.add_argument('--georeference', type=str, help='If the image is referenced select "yes"', default='yes')
args = parser.parse_args()

# Set basic variables
ct = CentroidTracker()
current_frame = 0
id = 1
path = defaultdict(list)
# schema for shp file
schema = {
    'geometry': 'LineString',
    'properties': [('Name', 'str')]
}

# ------  Initialization -----
# open a fiona object for shape file
lineShp = fiona.open('output.shp', mode='w', driver='ESRI Shapefile', schema=schema, crs="EPSG:2056")

# Background Subtraction (BS): Select choosen algorithm
if args.algo_BS == 'MOG2':
    backSub = cv2.createBackgroundSubtractorMOG2()
else:
    backSub = cv2.createBackgroundSubtractorKNN()

# Initialize video
capture = cv2.VideoCapture(args.input)
if not capture.isOpened():
    print('Unable to open: ' + args.input)
    exit(0)

# Optical Flow (OF) Inizialize Optical flow
# ret = a boolean return value from getting the frame, first_frame = the first frame in the entire video sequence
ret, first_frame = capture.read()
# Converts frame to grayscale because we only need the luminance channel for detecting edges
prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
# Finds the strongest corners in the first frame by Shi-Tomasi method - we will track the optical flow for these corners
# https://docs.opencv2.org/3.0-beta/modules/imgproc/doc/feature_detection.html#goodfeaturestotrack
prev = cv2.goodFeaturesToTrack(prev_gray, mask=None, **VP1_param.feature_params)
mask = np.zeros_like(first_frame)
mask2 = np.zeros_like(first_frame)

# Georeferencing
if args.georeference == 'yes':
    coord_pixel0_x, coord_pixel0_y, scale = georeference()
    #contour = findcontour(prev_gray)
else:
    coord_pixel0_x = 0
    coord_pixel0_y = 0
    scale = 1

# Start time
start = time.time()

# ------ While loop for elaboration of every frame ------
while True:
    current_frame += 1
    ret, frame = capture.read()

# ------ Execute at the end of the video ------
    if frame is None:
        for key in path:
            if len(path[key]) > VP1_param.min_lenght:
                distance = sqrt((path[key][0][0] - path[key][-1][0]) ** 2 + (path[key][0][1] - path[key][-1][1]) ** 2)
                if distance > VP1_param.min_distance:  # in meters
                    rowDict = {
                        'geometry': {'type': 'LineString', 'coordinates': path[key]}, 'properties': {'Name': key}}
                    lineShp.write(rowDict)

        print(f"{os.linesep} finished")
        break

    # ----    BACKGROUND SUBTRACTION (BS)   -----
    # update the background model
    fgMask = backSub.apply(frame)
    # optimization of elements
    thresh = cv2.threshold(fgMask, VP1_param.threshold, 255, cv2.THRESH_BINARY)[1]
    se0 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, se0)
    kernel_dil = np.ones(VP1_param.kernel_dilatation, np.uint8)
    thresh = cv2.dilate(thresh, kernel_dil, iterations=1)

    se1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    se2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, se1)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, se2)

    # detect the contours on the binary image using cv2.CHAIN_APPROX_NONE
    _, contours, hierarchy = cv2.findContours(image=closed, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
    mask = np.zeros_like(first_frame)
    # draw contours on the original image
    contours = [x for x in contours if len(x) >= 30]
    cv2.drawContours(image=mask, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2,
                     lineType=cv2.LINE_AA)
    # ------ Draw bounding boxes ----
    rects = []
    ROI_number = 0
    for cntr in contours:
        x, y, w, h = cv2.boundingRect(cntr)
        if VP1_param.min_width < w < VP1_param.max_width and VP1_param.min_height < h < VP1_param.max_height:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            ROI = mask[y:y + h, x:x + w]
            cv2.putText(mask, str(ROI_number), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
            rects.append([x, y, x + w, y + h])
            ROI_number += 1

    # update centroid tracker using the computed set of bounding box rectangles
    objects, life = ct.update(rects)

    # build path in swiss coordinate system
    if objects is not None:
        for (objectID, centroid) in objects.items():
            if life[objectID] > VP1_param.min_life:
                # draw both the ID of the object and the centroid of the object on the output frame
                text = "ID {}".format(objectID)
                x = centroid[0]
                y = centroid[1]
                cv2.putText(frame, text, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)
                if objectID in path.keys():
                    path[objectID].append((coord_pixel0_x+(x*scale), coord_pixel0_y-(y*scale)))
                else:
                    path[objectID] = [(coord_pixel0_x+(x*scale), coord_pixel0_y-(y*scale))]

    # End time
    end = time.time()
    # Calculate frames per second
    fps = round(current_frame / (end-start), 2)

    # get the frame number and fps rate and write it on the current frame
    cv2.rectangle(frame, (10, 2), (600, 60), (255, 255, 255), -1)
    cv2.putText(frame, str("fps : {0}, current frame: {1}".format(fps, current_frame)), (15, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0))

    length = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)-1)
    progressBar(current_frame, length, fps)

    # windows to show, uncomment what you want to see
    # showWindows(frame, str("final frame"))
    # showWindows(mask,str("mask"))

    # Frames are read by intervals of 2 milliseconds. The programs breaks out of the while loop when
    # the user presses the 'q' key
    if cv2.waitKey(2) & 0xFF == ord('q'):
        print(f"{os.linesep}you stopped the programm")
        break
