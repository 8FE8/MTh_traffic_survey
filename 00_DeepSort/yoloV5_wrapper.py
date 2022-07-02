import os
import torch

from models.experimental import *
from utils.datasets import *
from utils.general import check_img_size, check_requirements, check_imshow, colorstr, non_max_suppression, \
    apply_classifier, scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
from utils.torch_utils import select_device, load_classifier, time_sync

import gflags
FLAGS = gflags.FLAGS

class YOLOWrapper():

    def __init__(self):

        self.conf_thres = 0.5 
        self.iou_thres = 0.5
        self.augment = False
        self.weights = "../../train-model/best.pt"
        self.imgszWidth, self.imgszHeight = 640, 480 #FLAGS.width
        # Initialize
        self.deviceId = '0'
        self.device = select_device(self.deviceId)

        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        self.model = attempt_load(self.weights, map_location=self.device)  # load FP32 model
        self.imgszWidth = check_img_size(self.imgszWidth, s=self.model.stride.max())  # check img_size
        self.imgszHeight = check_img_size(self.imgszHeight, s=self.model.stride.max())  # check img_size

        if self.half:
            self.model.half()  # to FP16

        # Get names and colors
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names

        img = torch.zeros((1, 3, self.imgszHeight, self.imgszWidth), device=self.device)  # init img
        _ = self.model(img.half() if self.half else img) if self.device.type != 'cpu' else None  # run once
        print("\n*** Created YOLOWrapper ***\n")


    def detect(self, image, x_offset, y_offset):
        
        img = letterbox(image, new_shape=(self.imgszHeight, self.imgszWidth))[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        pred = self.model(img, augment=self.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres)


        bboxes, classes = [], []
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            
            gn = torch.tensor(image.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], image.shape).round()

                # Write results
                for *xyxy, conf, cls in det:
                    label = self.names[int(cls)]
                    x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                    bboxes.append([x1 + x_offset, y1 + y_offset, x2-x1, y2-y1])
                    classes.append(label)

        return bboxes, classes