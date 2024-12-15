import argparse
import os
import copy
import sys
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.ops import box_convert

sys.path.append("Grounded_Segment_Anything/GroundingDINO")

import Grounded_Segment_Anything.GroundingDINO.groundingdino.datasets.transforms as T
from Grounded_Segment_Anything.GroundingDINO.groundingdino.models import build_model
from Grounded_Segment_Anything.GroundingDINO.groundingdino.util import box_ops
from Grounded_Segment_Anything.GroundingDINO.groundingdino.util.slconfig import SLConfig
from Grounded_Segment_Anything.GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from Grounded_Segment_Anything.GroundingDINO.groundingdino.util.inference import annotate, predict, annotate_new

import supervision as sv

from segment_anything import build_sam, SamPredictor
import cv2
import matplotlib.pyplot as plt

from rearrange.scripts.sam_mask import SAMPredictor
from rearrange.scripts.config import GaussianConfig

from navigation.constants import PICKUPABLE_OBJECTS, OPENABLE_OBJECTS


class GSAM:
    def __init__(self, config, device):
        self.device = device

        config_file = "Grounded_Segment_Anything/GroundingDINO_SwinT_OGC.py"
        grounded_checkpoint = "Grounded_Segment_Anything/groundingdino_swint_ogc.pth"
        self.sam_version = "sam"
        self.sam_checkpoint = "ckpts/sam_vit_h_4b8939.pth"
        self.device = "cuda:0"
        self.box_threshold = 0.3
        self.text_threshold = 0.25

        self.GDino_model = self.load_model(config_file, grounded_checkpoint, device=self.device)

        # SAM
        self.sam = SAMPredictor(config)

        # objects
        self.all_objects = OPENABLE_OBJECTS + PICKUPABLE_OBJECTS
        self.text_prompt = ""
        for obj in self.all_objects:
            if len(self.text_prompt) == 0:
                self.text_prompt += obj
            else:
                self.text_prompt += " . "
                self.text_prompt += obj

        print("TEXT PROMPT : ", self.text_prompt)



    def load_model(self, model_config_path, model_checkpoint_path, device):
        args = SLConfig.fromfile(model_config_path)
        args.device = device
        model = build_model(args)
        checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
        load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        _ = model.eval()
        return model

    def load_image(self, image_pil):
        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image = np.asarray(image_pil)
        image_transformed, _ = transform(image_pil, None)
        return image, image_transformed

    def get_dino_output(self, image_pil, image_full, center):
        TEXT_PROMPT = self.text_prompt
        BOX_TRESHOLD = 0.3
        TEXT_TRESHOLD = 0.25

        image_source, image = self.load_image(image_pil)
        image_full_source, _ = self.load_image(image_full)

        boxes, logits, phrases = predict(
            model=self.GDino_model,
            image=image,
            caption=TEXT_PROMPT,
            box_threshold=BOX_TRESHOLD,
            text_threshold=TEXT_TRESHOLD,
            device=self.device)

        annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
        annotated_frame = annotated_frame[..., ::-1]  # BGR to RGB

        if center is not None:
            if boxes.shape[0] != 0:
                annotated_full = annotate_new(image_full=image_full_source, image_source=image_source, boxes=boxes, logits=logits, phrases=phrases, center=center)
        else:
            annotated_full = annotated_frame

        return boxes, logits, phrases, annotated_frame, image_source, annotated_full

    def get_sam_output(self, image_source, boxes):
        if boxes.shape[0] != 0:
            H, W, _ = image_source.shape
            boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])
            masks = self.sam.get_sam_output_gsam(image_source, boxes_xyxy)
            return masks
        else:
            return None

    def show_mask(self, mask, image, random_color=True):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.8])], axis=0)
        else:
            color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)

        annotated_frame_pil = Image.fromarray(image).convert("RGBA")
        mask_image_pil = Image.fromarray((mask_image * 255).astype(np.uint8)).convert("RGBA")

        return np.array(Image.alpha_composite(annotated_frame_pil, mask_image_pil))

    def get_gsam_output(self, image_cv2, image_full, center):
        image_pil = Image.fromarray(image_cv2)
        image_full_pil = Image.fromarray(image_full)
        boxes, logits, phrases, annotated_frame, image_source, annotated_full = self.get_dino_output(image_pil, image_full_pil, center)
        # masks = self.get_sam_output(image_source, boxes)
        masks = None
        if masks is not None:
            masks = masks.detach().cpu().numpy()
            image_vis = self.show_mask(masks[0][0], annotated_frame)
        else:
            image_vis = annotated_frame
        return boxes, logits, phrases, annotated_frame, masks, image_vis, annotated_full