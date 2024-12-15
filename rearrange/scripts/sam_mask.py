import numpy as np
import torch
import torchvision
import cv2
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

class SAMPredictor():
    def __init__(self, config):
        self.sam_checkpoint = config.sam_checkpoint
        self.model_type = config.sam_model_type
        self.device = config.sam_device
        self.sam = sam_model_registry[self.model_type](checkpoint=self.sam_checkpoint)
        self.sam.to(device=self.device)

    def get_mask(self, img, bbox):
        """
        img : CV2 RGB image
        """
        predictor = SamPredictor(self.sam)
        predictor.set_image(img)
        masks, _, _ = predictor.predict(box=np.array(bbox),
                                             point_coords=None,
                                             point_labels=None,
                                             multimask_output=False)
        del predictor
        torch.cuda.empty_cache()
        return masks[0]

    def init_predictor(self, img):
        self.predictor = SamPredictor(self.sam)
        self.predictor.set_image(img)

    def get_mask_bbox(self, img, bbox):
        masks, _, _ = self.predictor.predict(box=np.array(bbox),
                                             point_coords=None,
                                             point_labels=None,
                                             multimask_output=False)
        torch.cuda.empty_cache()
        return masks[0]

    def del_predictor(self):
        torch.cuda.empty_cache()
        del self.predictor
        torch.cuda.empty_cache()

    def get_mask_all(self, img, bbox=None):
        predictor = SamPredictor(self.sam)
        predictor.set_image(img)
        if bbox is not None:
            masks, _, _ = predictor.predict(box=np.array(bbox),
                                            point_coords=None,
                                            point_labels=None,
                                            multimask_output=True)
        else:
            masks, _, _ = predictor.predict(box=None,
                                            point_coords=None,
                                            point_labels=None,
                                            multimask_output=True)
        del predictor
        torch.cuda.empty_cache()
        return masks

    def get_sam_output_gsam(self, img, boxes_xyxy):
        predictor = SamPredictor(self.sam)
        predictor.set_image(img)

        boxes_xyxy = boxes_xyxy.to(self.device)

        transformed_boxes = predictor.transform.apply_boxes_torch(boxes_xyxy, img.shape[:2]).to(self.device)
        masks, _, _ = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )
        del predictor
        torch.cuda.empty_cache()
        return masks
