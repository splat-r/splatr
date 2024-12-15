import os
import cv2
import copy
import torch
import numpy as np
import torchvision
import time


class Runner():
    def __init__(self,
                 matcher,
                 config):

        self.matcher = matcher
        self.config = config

    def match_frames(self, sim_image, rendered_image, timestep):
        rendered_image = np.clip(rendered_image, 0, 1)
        rendered_image = (rendered_image * 255).astype(np.uint8)

        sim_mask = sim_image[..., 0] > -1
        rendered_mask = rendered_image[..., 0] > -1

        diff_image = np.sum(np.abs(sim_image - rendered_image), axis=-1)
        diff_image = (diff_image - diff_image.min()) / (diff_image.max() - diff_image.min())


        image_tensor1, grid_size1, resize_scale1 = self.matcher.prepare_image(sim_image)
        features1 = self.matcher.extract_features(image_tensor1)

        image_tensor2, grid_size2, resize_scale2 = self.matcher.prepare_image(rendered_image)
        features2 = self.matcher.extract_features(image_tensor2)

        resized_mask1 = self.matcher.prepare_mask(sim_mask, grid_size1, resize_scale1)
        resized_mask2 = self.matcher.prepare_mask(rendered_mask, grid_size2, resize_scale2)

        vis_image3, vis_image4, tk1, tk2 = self.matcher.get_combined_embedding_visualization(
                                                            features1, features2, grid_size1,
                                                            grid_size2, resized_mask1,
                                                            resized_mask2)

        cosine_similarity = torch.nn.functional.cosine_similarity(torch.tensor(tk1), torch.tensor(tk2))
        cosine_similarity_ = cosine_similarity.reshape(*grid_size1)
        cosine_similarity_ = cosine_similarity_.cpu().numpy()
        cs_diff_image = np.kron(cosine_similarity_, np.ones((7, 7)))
        cs_mask_1 = cs_diff_image < 0.35
        cs_mask_2_1 = cs_diff_image > 0.35
        cs_mask_2_2 = cs_diff_image < 0.5
        cs_mask_2 = np.logical_and(cs_mask_2_2, cs_mask_2_1)
        cs_diff_image_new = np.zeros_like(cs_diff_image)
        cs_diff_image_new[cs_mask_1] = 1
        cs_diff_image_new[cs_mask_2] = 0.5
        cs_diff_image[np.logical_or(~cs_mask_1, ~cs_mask_2)] = 0

        mask_cs = cosine_similarity < 0.35

        mask_cs = mask_cs.reshape(*grid_size1)
        mask_img = cv2.resize(np.array(mask_cs * 1), (sim_image.shape[1], sim_image.shape[0]),
                              interpolation=cv2.INTER_NEAREST)

        # This mask corresponds to the region where there is a change in the objects
        mask_img = mask_img.astype(np.bool_)

        if self.config.viz_dense_features:
            sim_image_mask = sim_image[mask_img]
            sim_image_masked = copy.deepcopy(sim_image)
            sim_image_masked[mask_img] = sim_image_mask * 0.6 + np.array([255, 0, 0]) * 0.4

            rendered_image_mask = rendered_image[mask_img]
            rendered_image_masked = copy.deepcopy(rendered_image)
            rendered_image_masked[mask_img] = rendered_image_mask * 0.6 + np.array([255, 0, 0]) * 0.4

            vis_image3 = cv2.resize(vis_image3, (sim_image.shape[1], sim_image.shape[0]), interpolation=cv2.INTER_NEAREST)
            vis_image4 = cv2.resize(vis_image4, (sim_image.shape[1], sim_image.shape[0]), interpolation=cv2.INTER_NEAREST)
            vis_image3 = (vis_image3 * 255).astype(np.uint8)
            vis_image4 = (vis_image4 * 255).astype(np.uint8)
            save_image = np.zeros((sim_image.shape[0] * 2 + 100, sim_image.shape[0] * 2 + 100, 3), dtype=np.uint8)
            save_image[0:sim_image.shape[0], 0:sim_image.shape[1], :] = cv2.cvtColor(sim_image_masked, cv2.COLOR_RGB2BGR)
            save_image[sim_image.shape[0] + 100:sim_image.shape[0] * 2 + 100, 0:sim_image.shape[1], :] = cv2.cvtColor(rendered_image_masked,
                                                                                                             cv2.COLOR_RGB2BGR)
            save_image[0:sim_image.shape[0], sim_image.shape[1] + 100:sim_image.shape[1] * 2 + 100, :] = vis_image3
            save_image[sim_image.shape[0] + 100:sim_image.shape[0] * 2 + 100, sim_image.shape[1] + 100:sim_image.shape[1] * 2 + 100,
            :] = vis_image4

            save_path_img = os.path.join(self.config.save_dino_frames, "dino_{}.png".format(timestep))
            print("path : ", save_path_img)
            cv2.imwrite(save_path_img, save_image)

        if self.config.dilate_object_mask:
            mask_img = mask_img.astype(np.uint8)
            kernel = np.ones((5, 5), np.uint8)
            dilated_mask = cv2.dilate(mask_img, kernel, iterations=self.config.dilate_iterations)
            dilated_mask = dilated_mask.astype(np.bool_)
            mask_img = dilated_mask

        return diff_image, mask_img

    def delineate_masks(self, mask_all):
        mask_all = (mask_all * 255).astype(np.uint8)
        num_labels, labeled_mask = cv2.connectedComponents(mask_all, connectivity=4)
        individual_masks = []

        # Start from 1 to ignore the background
        for label in range(1, num_labels):
            individual_mask = np.uint8(labeled_mask == label) * 255
            individual_masks.append(individual_mask.astype(np.bool_))

        return individual_masks
