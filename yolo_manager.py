import torch
import tensorflow as tf
from ultralytics import YOLO
from ultralytics.engine.results import Results
from ultralytics.utils import ops  # for postprocess
from pathlib import Path
import cv2
import numpy as np

try:
    from pycoral.utils.edgetpu import make_interpreter
    from pycoral.adapters import common
except ModuleNotFoundError as m_err:
    pass


# .pt files contains names in there but exported onnx/tflite don't have them.
yolo_default_label_names = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train',
                            7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign',
                            12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse',
                            18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe',
                            24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee',
                            30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat',
                            35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle',
                            40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana',
                            47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog',
                            53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant',
                            59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse',
                            65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster',
                            71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors',
                            77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}


class YoloDetector:
    def __init__(self, model_path, task='detect'):
        self.model = YOLO(model_path, task=task)

        self.imgsz = 640  # assume 640 at the moment since it is the default one
        if model_path.suffix == '.onnx':
            # once exported to onnx, auto resizing doesn't seem to work as expected
            # probably there is a better way but I'll just read it from onnx file
            # and set the dimension when predict
            # note, square images only atm
            import onnx
            dummy_model = onnx.load(str(model_path))
            self.imgsz = dummy_model.graph.input[0].type.tensor_type.shape.dim[-1].dim_value
            del dummy_model

    def predict(self, frame, conf):
        return self.model.predict(source=frame, save=False, conf=conf, save_txt=False, show=False, verbose=False,
                                  imgsz=self.imgsz)

    def get_label_names(self):
        if self.model.names is None or len(self.model.names) == 0:
            return yolo_default_label_names
        return self.model.names


class YoloDetectorTFLite:
    def __init__(self, model_path, use_coral_tpu=False):
        self.name = model_path.name
        
        self.use_coral_tpu = use_coral_tpu
        if use_coral_tpu:
            # only use coral tpu interpreter if specified
            self.interpreter = make_interpreter(str(model_path))
        else:
            # use normal tf.lite
            self.interpreter = tf.lite.Interpreter(model_path=str(model_path))

        self.interpreter.allocate_tensors()

    def predict(self, frame, conf):
        orig_imgs = [frame]

        # Get input and output tensors.
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()

        # Test the model on random input data.
        input_shape = input_details[0]['shape']

        # TODO check shape of input_shape and frame.shape
        # input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
        _, w, h, _ = input_shape

        # check width and height
        if frame.shape[0] != h or frame.shape[1] != w:
            input_img = cv2.resize(frame, (w, h))
        else:
            input_img = frame
        input_img = input_img[np.newaxis, ...]  # add batch dim

        if self.use_coral_tpu:
            params = common.input_details(self.interpreter, 'quantization_parameters')
            scale = params['scales']
            zero_point = params['zero_points']
            input_mean = 128.
            input_std = 128.

            normalized_input = (input_img - input_mean) / (input_std * scale) + zero_point
            np.clip(normalized_input, 0, 255, out=normalized_input)
            common.set_input(self.interpreter, normalized_input.astype(np.uint8))
        else:
            input_img = input_img.astype(np.float32) / 255.  # change to float img
            self.interpreter.set_tensor(input_details[0]['index'], input_img)

        self.interpreter.invoke()

        preds = self.interpreter.get_tensor(output_details[0]['index'])

        if self.use_coral_tpu:
            output_details = self.interpreter.get_output_details()[0]
           
            if np.issubdtype(preds.dtype, np.integer):
                scale, zero_point = output_details['quantization']
                # Always convert to np.int64 to avoid overflow on subtraction.
                preds = scale * (preds.astype(np.int64) - zero_point)
                preds = preds.astype(np.float32)
            
        
        ######################################################################
        # borrowed from ultralytics\models\yolo\detect\predict.py #postprocess

        # convert to torch to use ops.non_max_suppression
        # ultralytics is working on none-deeplearning based non_max_suppression
        # https://github.com/ultralytics/ultralytics/issues/1777
        # maybe someday, but for now, just workaround
        preds = torch.from_numpy(preds)
        preds = ops.non_max_suppression(preds,
                                        conf,
                                        0.7,  # todo, make into arg
                                        agnostic=False,
                                        max_det=300,
                                        classes=None)  # hack. just copied values from execution of yolov8n.pt

        results = []
        for i, pred in enumerate(preds):
            orig_img = orig_imgs[i]

            # tflite result are in [0, 1]
            # scale them by width (w == h)
            pred[:, :4] *= w

            pred[:, :4] = ops.scale_boxes(input_img.shape[1:], pred[:, :4], orig_img.shape)
            img_path = ""
            results.append(Results(orig_img, path=img_path, names=yolo_default_label_names, boxes=pred))

        return results

    def get_label_names(self):
        return yolo_default_label_names


class YoloDetectorWrapper:
    def __init__(self, model_path, use_coral_tpu=False):
        model_path = Path(model_path)

        if use_coral_tpu or model_path.suffix == '.tflite':
            self.detector = YoloDetectorTFLite(model_path, use_coral_tpu)
        else:
            self.detector = YoloDetector(model_path)

    def predict(self, frame, conf=0.5):
        return self.detector.predict(frame, conf=conf)

    def get_label_names(self):
        return self.detector.get_label_names()
