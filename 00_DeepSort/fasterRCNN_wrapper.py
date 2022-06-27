
from maskrcnn_benchmark.config import cfg
from torch import classes
from predictor import COCODemo


class FasterRCNNWrapper():

    def __init__(self):

        config_file = "e2e_faster_rcnn_X_101_32x8d_FPN_1x_visdrone.yaml"

        # update the config options with the config file
        cfg.merge_from_file(config_file)
        # manual override some options
        #cfg.merge_from_list(["MODEL.DEVICE", "cpu"])
        cfg.merge_from_list(["MODEL.WEIGHT", "/home/ibrahim/MyProjects/visdrone_model_0360000.pth"])

        self.coco_demo = COCODemo(
            cfg,
            min_image_size=640,
            confidence_threshold=0.8,
        )

    def detect(self, img, x_offset, y_offset):
        predictions = []
        self.coco_demo.run_on_opencv_image(img, predictions)

        bboxes, classes = [], []
        for label, score, top_left, bottom_right in predictions:
            # if "pedestrian" == label:
            #     continue
            x1, y1 = top_left[0] + x_offset, top_left[1] + y_offset
            width, height = bottom_right[0] - top_left[0], bottom_right[1] - top_left[1]
            bboxes.append([x1, y1, width, height])
            classes.append(label)
        return bboxes, classes