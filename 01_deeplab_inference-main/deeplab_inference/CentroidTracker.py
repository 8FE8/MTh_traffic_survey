from collections import OrderedDict
import numpy as np
from scipy.spatial import distance as dist
import cv2
import params
import csv


class CentroidTracker():
    def __init__(self, maxDisappeared=params.max_disappeared):
        # initialize the next unique object ID along with two ordered
        # dictionaries used to keep track of mapping a given object
        # ID to its centroid and number of consecutive frames it has
        # been marked as "disappeared", respectively
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.life = OrderedDict()
        # store the number of maximum consecutive frames a given
        # object is allowed to be marked as "disappeared" until we
        # need to deregister the object from tracking
        self.maxDisappeared = maxDisappeared

    def register(self, centroid):
        # when registering an object we use the next available object
        # ID to store the centroid
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.life[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        # to deregister an object ID we delete the object ID from
        # both of our respective dictionaries
        del self.objects[objectID]
        del self.disappeared[objectID]
        del self.life[objectID]

    def update(self, rects):
        # check to see if the list of input bounding box rectangles
        # is empty
        if len(rects) == 0:
            # loop over any existing tracked objects and mark them
            # as disappeared
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                self.life[objectID] += 1
                # if we have reached a maximum number of consecutive
                # frames where a given object has been marked as
                # missing, deregister it
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            # return early as there are no centroids or tracking info
            # to update
            return self.objects, self.life
    # UNIQUE ID
        # initialize an array of input centroids for the current frame
        inputCentroids = np.zeros((len(rects), 2), dtype="int")
        # loop over the bounding box rectangles
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            # use the bounding box coordinates to derive the centroid
            cX = int((int(startX) + int(endX)) / 2.0)
            cY = int((int(startY) + int(endY)) / 2.0)
            inputCentroids[i] = (cX, cY)
        # if we are currently not tracking any objects take the input
        # centroids and register each of them
        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i])
        # otherwise, are are currently tracking objects so we need to
        # try to match the input centroids to existing object
        # centroids
        else:
            # grab the set of object IDs and corresponding centroids
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())
            # compute the distance between each pair of object
            # centroids and input centroids, respectively -- our
            # goal will be to match an input centroid to an existing
            # object centroid
            D = dist.cdist(np.array(objectCentroids), inputCentroids)
            # in order to perform this matching we must (1) find the
            # smallest value in each row and then (2) sort the row
            # indexes based on their minimum values so that the row
            # with the smallest value is at the *front* of the index
            # list
            rows = D.min(axis=1).argsort()
            # next, we perform a similar process on the columns by
            # finding the smallest value in each column and then
            # sorting using the previously computed row index list
            cols = D.argmin(axis=1)[rows]
            # in order to determine if we need to update, register,
            # or deregister an object we need to keep track of which
            # of the rows and column indexes we have already examined
            usedRows = set()
            usedCols = set()
            # loop over the combination of the (row, column) index
            # tuples
            for (row, col) in zip(rows, cols):
                # if we have already examined either the row or
                # column value before, ignore it
                # val
                if row in usedRows or col in usedCols:
                    continue

                # otherwise, grab the object ID for the current row,
                # set its new centroid, and reset the disappeared
                # counter
                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.disappeared[objectID] = 0
                self.life[objectID] += 1
                # indicate that we have examined each of the row and
                # column indexes, respectively
                usedRows.add(row)
                usedCols.add(col)

                # compute both the row and column index we have NOT yet
                # examined
            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)
                # in the event that the number of object centroids is
                # equal or greater than the number of input centroids
                # we need to check and see if some of these objects have
                # potentially disappeared

            if D.shape[0] >= D.shape[1]:
                # loop over the unused row indexes
                for row in unusedRows:
                        # grab the object ID for the corresponding row
                        # index and increment the disappeared counter
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1
                    self.life[objectID] += 1
                        # check to see if the number of consecutive
                        # frames the object has been marked "disappeared"
                        # for warrants deregistering the object
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)
                # otherwise, if the number of input centroids is greater
                # than the number of existing object centroids we need to
                # register each new input centroid as a trackable object
            else:
                for col in unusedCols:
                    self.register(inputCentroids[col])
        # return the set of trackable objects


        return self.objects, self.life

def progressBar(current, total, fps, barLength=20):
        percent = float(current) * 100 / total
        arrow = '-' * int(percent / 100 * barLength - 1) + '>'
        spaces = ' ' * (barLength - len(arrow))
        print('Actual fps: %f, Progress: [%s%s] %d %%' % (fps,arrow, spaces, percent), end='\r')

def showWindows(frame, frame_name):
        frame= cv2.resize(frame, (params.frame_width, params.frame_height))
        cv2.imshow(frame_name, frame)

def georeference():
    file = open('exclude/georef_pxcoord_gcps.csv')
    type(file)
    csvreader = csv.reader(file)
    header = []
    header = next(csvreader)
    header
    rows = []
    for row in csvreader:
        rows.append(row)
    rows
    file.close()

    LV95_1_x = float(rows[0][1])
    LV95_1_y = float(rows[0][2])
    LV95_2_x = float(rows[1][1])
    LV95_2_y = float(rows[1][2])
    pixel_1_x = float(rows[0][3])
    pixel_1_y = float(rows[0][4])
    pixel_2_x = float(rows[1][3])
    pixel_2_y = float(rows[1][4])

    scale1 = round(abs((LV95_1_x - LV95_2_x) / (pixel_1_x - pixel_2_x)), 4)
    scale2 = round(abs((LV95_1_y - LV95_2_y) / (pixel_1_y - pixel_2_y)), 4)

    if scale1 == scale2:
        coord_pixel0_x = LV95_1_x - (pixel_1_x * scale1)
        coord_pixel0_y = LV95_1_y + (pixel_1_y * scale1)
    else:
        print("The images are not well georeferenced, the programm cannot work with different x and y scale")
        coord_pixel0_x = 0
        coord_pixel0_y = 0
        print("the output will not be georeferenced, will be in a local system")

    return coord_pixel0_x, coord_pixel0_y, scale1


