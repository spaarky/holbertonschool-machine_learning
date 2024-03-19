#!/usr/bin/env python3

"""YOLO Object Detection"""
import tensorflow.Keras as K


class Yolo():
    """YOLO Object Detection"""

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """Constructor

        Args:
            model_path (string): path to where a Darknet Keras model is stored
            classes_path (string): path to where the list of class names used
                for the Darknet model, listed in order of index, can be found
            class_t (float): represent the box score threshold for the initial
                filtering step
            nms_t (float): represent the IOU threshold for non-max suppression
            anchors (numpy.ndarray): contains all of the anchor boxes
        """
        self.model = K.models.load_model(model_path)
        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors
