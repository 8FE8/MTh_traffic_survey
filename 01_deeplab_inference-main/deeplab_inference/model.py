import os
import tensorflow as tf
import numpy as np
from PIL import Image

from utils import resize_label


class DeepLab(object):
    INPUT_SIZE = 1025
    INPUT_TENSOR = 'ImageTensor:0'
    OUTPUT_TENSOR = 'SemanticPredictions:0'


class DeepLabModel(object):

    def __init__(self, path):
        """Initializes the class.
        
        Args:
            path: The full path to the deeplab frozen graph file.
        """
        self.graph = tf.Graph()

        graph_def = tf.compat.v1.GraphDef()
        with open(path, 'rb') as g:
            loaded = graph_def.ParseFromString(g.read())

        deeplab_func = self._wrap_frozen_graph(
            graph_def, inputs=DeepLab.INPUT_TENSOR,
            outputs=DeepLab.OUTPUT_TENSOR)

        with self.graph.as_default():
            tf.import_graph_def(graph_def, name='')

        self.sess = tf.compat.v1.Session(graph=self.graph)
    
    
    def run(self, image):
        """Runs inference on a single image.

        Args:
          image: A PIL.Image object, raw input image.

        Returns:
          seg_map: Resized segmentation map of 'image'.
        """
        width, height = image.size
        resize_ratio = 1.0 * DeepLab.INPUT_SIZE / max(width, height)
        target_size = (int(resize_ratio * width), int(resize_ratio * height))
        resized_image = image.convert('RGB').resize(target_size,
                                                    Image.ANTIALIAS)
        batch_seg_map = self.sess.run(
            DeepLab.OUTPUT_TENSOR,
            feed_dict={DeepLab.INPUT_TENSOR: [np.asarray(resized_image)]})
        seg_map = batch_seg_map[0]

        return seg_map
    
    
    def _wrap_frozen_graph(self, graph_def, inputs, outputs):
        def _imports_graph_def():
            tf.compat.v1.import_graph_def(graph_def, name='')
        wrapped_import = tf.compat.v1.wrap_function(_imports_graph_def, [])
        import_graph = wrapped_import.graph
        return wrapped_import.prune(
            tf.nest.map_structure(import_graph.as_graph_element, inputs),
            tf.nest.map_structure(import_graph.as_graph_element, outputs))
