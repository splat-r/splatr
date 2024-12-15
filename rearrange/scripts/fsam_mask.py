import torch
from fastsam import FastSAM, FastSAMPrompt
import cv2
import numpy as np
import sys
from PIL import Image
sys.path.append("FastSAM/")
from utils.tools import convert_box_xywh_to_xyxy
class FSAMPredictor():
    def __init__(self, config):
        self.model_path = config.fastsam_model_path
        self.imgsz = 224
        self.iou = 0.9  # iou threshold for filtering annotations
        self.text_prompt = None
        self.conf = 0.4  # Object confidence threshold
        self.random_color = True  # mask with random color
        self.better_quality = False
        self.device = config.fastsam_device
        self.model = FastSAM(self.model_path)

    def masks_to_bool(self, masks):
        if type(masks) == np.ndarray:
            return masks.astype(bool)
        return masks.cpu().numpy().astype(bool)

    def get_mask(self, img, bbox):
        input_img = Image.fromarray(img)
        everything_results = self.model(input_img,
                                    device=self.device,
                                    retina_masks=True,
                                    imgsz=self.imgsz,
                                    conf=self.conf,
                                    iou=self.iou
                                )
        prompt_process = FastSAMPrompt(input_img, everything_results, device=self.device)
        masks = prompt_process.box_prompt(bbox=bbox)
        masks = self.masks_to_bool(masks)
        return masks