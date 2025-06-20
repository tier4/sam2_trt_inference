# Copyright 2025 TIER IV, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http:#www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import cv2
import numpy as np
import ctypes
from ctypes import POINTER, Structure, c_int
import json
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Rect(Structure):
        _fields_ = [("x", c_int), ("y", c_int), ("width", c_int), ("height", c_int)]
        
class SAM2Image:
    """
    A wrapper for interacting with the SAM2Image inference library using ctypes.
    """
    def __init__(self, encoder_path: str, decoder_path: str, model_precision: str, decoder_batch_limit: int):
        self.lib = ctypes.CDLL('libtrt_sam2_infer.so')
        self.lib.create_sam2image.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
        self.lib.create_sam2image.restype = ctypes.c_void_p

        self.lib.InitOpenCVThreads()
        # Create SAM2Image instance
        self.instance = self.lib.create_sam2image(
            encoder_path.encode('utf-8'),
            decoder_path.encode('utf-8'),
            model_precision.encode('utf-8'),
            decoder_batch_limit
        )
        if not self.instance:
            raise RuntimeError("Failed to create SAM2Image instance")

        # Define argument and return types for the shared library functions
        self.lib.sam2image_set_image.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ubyte), ctypes.c_int, ctypes.c_int]
        self.lib.sam2image_set_image.restype = None

        self.lib.sam2image_set_box.argtypes = [ctypes.c_void_p, POINTER(Rect), c_int]
        self.lib.sam2image_set_box.restype = None

        self.lib.sam2image_get_masks.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ubyte), ctypes.c_int, ctypes.c_int]
        self.lib.sam2image_get_masks.restype = None

        self.lib.sam2image_get_polygon_str.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_char_p),
                ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_int), ctypes.c_char_p, ctypes.c_float
        ]
        self.lib.sam2image_get_polygon_str.restype = ctypes.c_char_p

        self.lib.sam2image_get_max_entropy.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_ubyte), ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_float)
        ]
        self.lib.sam2image_get_max_entropy.restype = None

    def set_image(self, image: np.ndarray):
        """Set the input image for processing."""
        if image.dtype != np.uint8:
            raise ValueError("Image must be of type np.uint8")
        height, width, _ = image.shape
        img_data = image.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))
        self.lib.sam2image_set_image(self.instance, img_data, width, height)

    def set_box(self, rects):
        """Set the bounding box for processing."""
        rect_array = (Rect * len(rects))(*rects)
        self.lib.sam2image_set_box(self.instance, rect_array, len(rects))

    def get_masks(self, width: int, height: int) -> np.ndarray:
        """Retrieve segmentation masks from the model."""
        buffer = np.zeros((height, width, 3), dtype=np.uint8)
        self.lib.sam2image_get_masks(self.instance, buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)), width, height)
        return buffer

    def get_polygon_str(self, width: int, height: int, bboxes: list, names: list, filename: str, uncertainty: float):
        """Retrieve polygonal segmentation data in JSON format."""
        name_list = []
        prob = np.array([], dtype=np.float32)
        prob = np.array([bbox["prob"] for bbox in bboxes], dtype=np.float32)
        ids = np.array([bbox["id"] for bbox in bboxes], dtype=np.int32)        
        for bbox in bboxes:
            if "sub_name" in bbox:
                name_list.append(bbox["sub_name"])
            else:
                name_list.append(names[bbox["label"]])

        c_string_array = (ctypes.c_char_p * len(name_list))(*[s.encode('utf-8') for s in name_list])
        c_string = filename.encode('utf-8')

        c_prob = prob.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        c_id = ids.ctypes.data_as(ctypes.POINTER(ctypes.c_int))        
        result_ptr = self.lib.sam2image_get_polygon_str(self.instance, width, height, c_string_array, len(c_string_array), c_prob, c_id, c_string, uncertainty)

        return json.loads(result_ptr.decode("utf-8"))

    def sam2_from_bboxes(self, image: np.ndarray, bboxes: list, names: list, filename: str, decoder_batch_size: int):
        """Process bounding boxes and return polygonal annotations."""
        height, width, _ = image.shape
        rects = []
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox["box"]
            rects.append(Rect(int(x1), int(y1), int(x2 - x1), int(y2 - y1)))
        self.set_box(rects)
        ent_mat, uncertainty = self.get_max_entropy(256, 256)
        return self.get_polygon_str(width, height, bboxes, names, filename, uncertainty), ent_mat

    def get_max_entropy(self, width: int, height: int):
        """Retrieve entropy map and associated uncertainty score."""
        output_image = np.zeros((height, width, 3), dtype=np.uint8)
        entropy_score = ctypes.c_float()
        self.lib.sam2image_get_max_entropy(
            self.instance, output_image.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)), width, height, ctypes.byref(entropy_score)
        )
        return output_image, entropy_score.value 