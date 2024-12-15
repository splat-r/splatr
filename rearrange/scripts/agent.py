import os.path

import cv2
import copy
import torch
import torchvision
import faiss
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation

# 2d mapping
from navigation.navigation import Navigation
from navigation.constants import PICKUPABLE_OBJECTS, OPENABLE_OBJECTS

from rearrange.scripts.clip_feature_extractor import CLIPFeatureExtractor
from rearrange.scripts.interfaces import ObjectScene

from scipy.optimize import linear_sum_assignment
from sklearn.metrics.pairwise import cosine_similarity


class Agent():
    def __init__(self, config, run_path):
        self.config = config
        self.run_path = run_path

        self.nav_action_to_rearrange_action = {
            'MoveAhead': 'move_ahead', 'MoveLeft': 'move_left', 'MoveRight': 'move_right', 'MoveBack': 'move_back',
            'RotateRight': 'rotate_right', 'RotateLeft': 'rotate_left', 'LookUp': 'look_up', 'LookDown': 'look_down',
            'Pass': 'done',
        }

        self.rearrange_action_to_nav_action = {'move_ahead': 'MoveAhead', 'move_left': 'MoveLeft', 'move_right': 'MoveRight',
                                               'move_back': 'MoveBack', 'rotate_right': 'RotateRight', 'rotate_left': 'RotateLeft',
                                               'look_up': 'LookUp', 'look_down': 'LookDown'}

        self.actions = [
            'done', 'move_ahead', 'move_left', 'move_right', 'move_back', 'rotate_right', 'rotate_left',
            'stand', 'crouch', 'look_up', 'look_down']

        self.action_to_ind = {}
        self.ind_to_action = {}
        for a_i in range(len(self.actions)):
            a = self.actions[a_i]
            self.action_to_ind[a] = a_i
            self.ind_to_action[a_i] = a

        self.navigation = Navigation(run_path=run_path)

        self.clip_ft_extractor = CLIPFeatureExtractor(self.config.clip_device)

        self.all_objects = OPENABLE_OBJECTS + PICKUPABLE_OBJECTS + ["wall", "plain surface", "mirror", "window"]

        # self.all_objects = ["An object that I can pick up", "An object that I can open"]
        self.all_object_fts = self.clip_ft_extractor.tokenize_text(self.all_objects).detach()

        self.objects_sim = {}
        self.objects_rend = {}
        self.sim_obj_id = 0
        self.rend_obj_id = 0
        self.sim_centers = []
        self.rend_centers = []
        self.padding = 3

    def reinit_clip_cpu(self):
        del self.clip_ft_extractor
        torch.cuda.empty_cache()
        self.clip_ft_extractor = CLIPFeatureExtractor("cpu")

    def adjust_pose(self, rot_matrix, translation):
        flip_y = np.array([
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        adjusted_rotation = flip_y[:3, :3] @ rot_matrix @ flip_y[:3, :3]
        adjusted_translation = flip_y[:3, :3] @ translation

        adjusted_pose = np.eye(4)
        adjusted_pose[:3, :3] = adjusted_rotation
        adjusted_pose[:3, 3] = adjusted_translation

        R = Rotation.from_euler('x', 180, degrees=True).as_matrix()
        R_homogeneous = np.eye(4)
        R_homogeneous[:3, :3] = R
        T_open3d_rotated = R_homogeneous @ adjusted_pose
        adjusted_pose = T_open3d_rotated

        return adjusted_pose

    def get_pcd(self, evt):
        fov = evt.metadata['fov']
        rgb_image = evt.frame
        height, width, _ = evt.frame.shape
        fx = (width / 2) / (np.tan(np.deg2rad(fov / 2)))
        fy = (height / 2) / (np.tan(np.deg2rad(fov / 2)))
        cx = (width - 1) / 2
        cy = (height - 1) / 2

        position = copy.deepcopy(evt.metadata['cameraPosition'])
        pos_trans = [position['x'], position['y'], position['z']]

        rotation = copy.deepcopy(evt.metadata['agent']['rotation'])
        rot_matrix = Rotation.from_euler("zxy", [rotation['z'], rotation['x'], rotation['y']], degrees=True).as_matrix()

        pose = np.eye(4)
        pose[:3, :3] = rot_matrix
        pose[:3, 3] = pos_trans
        depth = evt.depth_frame
        frame_size = depth.shape[:2]
        x = np.arange(0, frame_size[1])
        y = np.arange(frame_size[0], 0, -1)
        xx, yy = np.meshgrid(x, y)

        xx = xx.flatten()
        yy = yy.flatten()
        zz = depth.flatten()
        x_camera = (xx - cx) * zz / fx
        y_camera = (yy - cy) * zz / fy
        z_camera = zz
        points_camera = np.stack([x_camera, y_camera, z_camera], axis=1)
        homogenized_points = np.hstack([points_camera, np.ones((points_camera.shape[0], 1))])
        points_world = pose @ homogenized_points.T
        points_world = points_world.T[..., :3]
        points_color = rgb_image.reshape(-1, 3)

        return points_world.reshape(rgb_image.shape[0], rgb_image.shape[1], 3), points_camera.reshape(rgb_image.shape[0], rgb_image.shape[1], 3)

    def init_success_checker(self, rgb, controller):
        self.navigation.init_success_checker(rgb, controller)

    def update_navigation_obs(self, rgb, depth, action_successful, update_success_checker=True):
        self.navigation.obs.image_list = [rgb]
        self.navigation.obs.depth_map_list = [depth]
        self.navigation.obs.return_status = "SUCCESSFUL" if action_successful else "OBSTRUCTED"

        if update_success_checker:
            self.navigation.success_checker.update_image(rgb)

    def step(self, action, controller):
        if 'move' in action:
            action_thor = self.rearrange_action_to_nav_action[action]
            event = controller.step(action=action_thor)
        elif 'rotate' in action:
            action_thor = self.rearrange_action_to_nav_action[action]
            event = controller.step(action=action_thor, degrees=self.config.map_args.DT)
        else:
            action_thor = self.rearrange_action_to_nav_action[action]
            event = controller.step(action=action_thor, degrees=self.config.map_args.HORIZON_DT)
        return event

    def crop_image_along_mask(self, image, mask):
        image_new = np.zeros_like(image)
        image_new[mask] = image[mask]
        image_new = image_new.astype(np.uint8)
        return image_new

    def crop_image_mask(self, image, mask):
        xpadding = self.padding
        ypadding = self.padding
        h, w, _ = image.shape
        y, x = np.where(mask*1 != 0)
        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()

        x_min = x_min - xpadding
        x_max = x_max + xpadding
        y_min = y_min - ypadding
        y_max = y_max + ypadding

        x_min = max(x_min, 0)
        x_max = min(x_max, w)
        y_min = max(y_min, 0)
        y_max = min(y_max, h)

        cropped_image = image[y_min:y_max, x_min:x_max]
        return cropped_image

    def crop_image_mask_plainbg(self, image, mask):
        xpadding = self.padding
        ypadding = self.padding
        h, w, _ = image.shape
        y, x = np.where(mask*1 != 0)
        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()

        x_min = x_min - xpadding
        x_max = x_max + xpadding
        y_min = y_min - ypadding
        y_max = y_max + ypadding

        x_min = max(x_min, 0)
        x_max = min(x_max, w)
        y_min = max(y_min, 0)
        y_max = min(y_max, h)

        new_image = np.zeros_like(image)
        # new_image = np.ones_like(image)*255
        new_image[mask] = image[mask]
        new_image = new_image.astype(np.uint8)
        cropped_image = new_image[y_min:y_max, x_min:x_max]
        return cropped_image

    def get_center_from_mask(self, mask):
        y, x = np.where(mask * 1 != 0)
        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()
        xc = int((x_min + x_max) / 2)
        yc = int((y_min + y_max) / 2)
        return (xc, yc)

    def get_bottom_center_from_mask(self, mask):
        y, x = np.where(mask * 1 != 0)
        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()
        xc = int((x_min + x_max) / 2)
        yc = int(y_max)
        return (xc, yc)

    # computing iou and overlap was inspired from https://github.com/concept-graphs/concept-graphs
    def expand_3d_box(self, bbox: torch.Tensor, eps=0.02) -> torch.Tensor:
        '''
        Expand the side of 3D boxes such that each side has at least eps length.
        Assumes the bbox cornder order in open3d convention.

        bbox: (N, 8, D)

        returns: (N, 8, D)
        '''
        center = bbox.mean(dim=1)  # shape: (N, D)

        va = bbox[:, 1, :] - bbox[:, 0, :]  # shape: (N, D)
        vb = bbox[:, 2, :] - bbox[:, 0, :]  # shape: (N, D)
        vc = bbox[:, 3, :] - bbox[:, 0, :]  # shape: (N, D)

        a = torch.linalg.vector_norm(va, ord=2, dim=1, keepdim=True)  # shape: (N, 1)
        b = torch.linalg.vector_norm(vb, ord=2, dim=1, keepdim=True)  # shape: (N, 1)
        c = torch.linalg.vector_norm(vc, ord=2, dim=1, keepdim=True)  # shape: (N, 1)

        va = torch.where(a < eps, va / a * eps, va)  # shape: (N, D)
        vb = torch.where(b < eps, vb / b * eps, vb)  # shape: (N, D)
        vc = torch.where(c < eps, vc / c * eps, vc)  # shape: (N, D)

        new_bbox = torch.stack([
            center - va / 2.0 - vb / 2.0 - vc / 2.0,
            center + va / 2.0 - vb / 2.0 - vc / 2.0,
            center - va / 2.0 + vb / 2.0 - vc / 2.0,
            center - va / 2.0 - vb / 2.0 + vc / 2.0,
            center + va / 2.0 + vb / 2.0 + vc / 2.0,
            center - va / 2.0 + vb / 2.0 + vc / 2.0,
            center + va / 2.0 - vb / 2.0 + vc / 2.0,
            center + va / 2.0 + vb / 2.0 - vc / 2.0,
        ], dim=1)  # shape: (N, 8, D)

        new_bbox = new_bbox.to(bbox.device)
        new_bbox = new_bbox.type(bbox.dtype)

        return new_bbox

    def compute_3d_iou_accuracte_batch(self, bbox1, bbox2):
        '''
        Compute IoU between two sets of oriented (or axis-aligned) 3D bounding boxes.

        bbox1: (M, 8, D), e.g. (M, 8, 3)
        bbox2: (N, 8, D), e.g. (N, 8, 3)

        returns: (M, N)
        '''
        # Must expend the box beforehand, otherwise it may results overestimated results
        bbox1 = self.expand_3d_box(bbox1, 0.02)
        bbox2 = self.expand_3d_box(bbox2, 0.02)
        import pytorch3d.ops as ops
        bbox1 = bbox1[:, [0, 2, 5, 3, 1, 7, 4, 6]]
        bbox2 = bbox2[:, [0, 2, 5, 3, 1, 7, 4, 6]]
        inter_vol, iou = ops.box3d_overlap(bbox1.float(), bbox2.float())

        return iou

    def compute_iou_batch(self, bbox1: torch.Tensor, bbox2: torch.Tensor) -> torch.Tensor:
        '''
        Compute IoU between two sets of axis-aligned 3D bounding boxes.

        bbox1: (M, V, D), e.g. (M, 8, 3)
        bbox2: (N, V, D), e.g. (N, 8, 3)

        returns: (M, N)
        '''
        # Compute min and max for each box
        bbox1_min, _ = bbox1.min(dim=1)  # Shape: (M, 3)
        bbox1_max, _ = bbox1.max(dim=1)  # Shape: (M, 3)
        bbox2_min, _ = bbox2.min(dim=1)  # Shape: (N, 3)
        bbox2_max, _ = bbox2.max(dim=1)  # Shape: (N, 3)

        # Expand dimensions for broadcasting
        bbox1_min = bbox1_min.unsqueeze(1)  # Shape: (M, 1, 3)
        bbox1_max = bbox1_max.unsqueeze(1)  # Shape: (M, 1, 3)
        bbox2_min = bbox2_min.unsqueeze(0)  # Shape: (1, N, 3)
        bbox2_max = bbox2_max.unsqueeze(0)  # Shape: (1, N, 3)

        # Compute max of min values and min of max values
        # to obtain the coordinates of intersection box.
        inter_min = torch.max(bbox1_min, bbox2_min)  # Shape: (M, N, 3)
        inter_max = torch.min(bbox1_max, bbox2_max)  # Shape: (M, N, 3)

        # Compute volume of intersection box
        inter_vol = torch.prod(torch.clamp(inter_max - inter_min, min=0), dim=2)  # Shape: (M, N)

        # Compute volumes of the two sets of boxes
        bbox1_vol = torch.prod(bbox1_max - bbox1_min, dim=2)  # Shape: (M, 1)
        bbox2_vol = torch.prod(bbox2_max - bbox2_min, dim=2)  # Shape: (1, N)

        # Compute IoU, handling the special case where there is no intersection
        # by setting the intersection volume to 0.
        iou = inter_vol / (bbox1_vol + bbox2_vol - inter_vol + 1e-10)

        return iou

    def compute_overlap_matrix_2set(self, objects_map, object_new) -> np.ndarray:
        '''
        compute pairwise overlapping between two set of objects in terms of point nearest neighbor.
        objects_map is the existing objects in the map, objects_new is the new object to be added to the map
        objects_map -> List of object instances stored in the map
        object_new -> new object instance
        '''
        m = len(objects_map)
        n = 1
        overlap_matrix = np.zeros((m, n))

        # Convert the point clouds into numpy arrays and then into FAISS indices for efficient search
        points_map = [objects_map[idx].points for idx in objects_map]  # m arrays
        indices = [faiss.IndexFlatL2(arr.shape[1]) for arr in points_map]  # m indices

        # Add the points from the numpy arrays to the corresponding FAISS indices
        for index, arr in zip(indices, points_map):
            index.add(arr)

        points_new = [object_new.points]  # n arrays

        bbox_map = torch.stack([objects_map[idx].bbox for idx in objects_map])
        bbox_new = object_new.bbox.unsqueeze(0)

        try:
            iou = self.compute_3d_iou_accuracte_batch(bbox_map, bbox_new)  # (m, n)
        except ValueError:
            print("Met `Plane vertices are not coplanar` error, use axis aligned bounding box instead")
            bbox_map = []
            bbox_new = []
            for idx in objects_map:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(objects_map[idx].points)
                bbox_map.append(np.asarray(
                    pcd.get_axis_aligned_bounding_box().get_box_points()))
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(object_new.points)
            bbox_new.append(np.asarray(
                pcd.get_axis_aligned_bounding_box().get_box_points()))
            bbox_map = torch.from_numpy(np.stack(bbox_map))
            bbox_new = torch.from_numpy(np.stack(bbox_new))
            iou = self.compute_iou_batch(bbox_map, bbox_new)  # (m, n)

        # Compute the pairwise overlaps
        for i in range(m):
            for j in range(n):
                if iou[i, j] < 1e-6:
                    continue
                D, I = indices[i].search(points_new[j], 1)  # search new object j in map object i
                overlap = (D < 0.0006).sum()  # D is the squared distance
                # Calculate the ratio of points within the threshold
                overlap_matrix[i, j] = overlap / len(points_new[j])

        return overlap_matrix

    def check_object_similarity_clip(self, clip_ft, clip_fts):
        clip_fts = torch.stack(clip_fts).squeeze(1)
        cs = torch.nn.functional.cosine_similarity(clip_fts, clip_ft)
        return cs
        # if torch.max(cs) > 0.87:
        #     # same object
        #     idx = torch.argmax(cs)
        #     idx = int(idx.item())
        # else:
        #     idx = -1
        # return idx

    def check_object_similarity_spatial(self, center, centers):
        threshold = 0.1
        # checking with centers of sim objects
        centers_np = np.array(centers)
        center_np = np.array(center)
        dist = np.linalg.norm(centers_np - center_np, axis=-1)
        closest_dist = min(dist)
        if closest_dist < threshold:
            obj_id = np.argmin(dist)
            return obj_id
        else:
            return -1

    def check_spatial_similarity(self, objects_map, object_new):
        overlap_matrix = self.compute_overlap_matrix_2set(objects_map, object_new)
        overlap_matrix = overlap_matrix.reshape(-1)
        return overlap_matrix

    def check_similarity(self, objects_map, object_new, clip_ft, clip_fts, delta=1):
        # 0.8 -> 0.85
        # 0 -> 0.8
        sp_sim = torch.from_numpy(self.check_spatial_similarity(objects_map, object_new))
        sem_sim = self.check_object_similarity_clip(clip_ft, clip_fts)
        sim = sem_sim*delta + sp_sim*(1-delta)
        if torch.max(sim) > 0.85:
            # same object
            idx = torch.argmax(sim)
            idx = int(idx.item())
        else:
            idx = -1
        return idx, float(torch.max(sim).item()), float(torch.max(sem_sim).item()), float(torch.max(sp_sim).item())

    def merge_detections(self, scene_type, new_object, obj_id):
        if scene_type == "sim":
            if new_object.object_name[0] not in self.objects_sim[obj_id].object_name:
                self.objects_sim[obj_id].n_diff_det += 1
            self.objects_sim[obj_id].object_name.append(new_object.object_name[0])
            self.objects_sim[obj_id].object_pos_world.append(new_object.object_pos_world[0])
            self.objects_sim[obj_id].object_pos_map.append(new_object.object_pos_map[0])
            self.objects_sim[obj_id].task.append(new_object.task[0])
            self.objects_sim[obj_id].conf.append(new_object.conf[0])
            # if new_object.avg_dist >= self.objects_sim[obj_id].avg_dist:
            #     self.objects_sim[obj_id].clip_ft[0] = new_object.clip_ft[0]
            self.objects_sim[obj_id].clip_ft[0] = (((len(self.objects_sim[obj_id].object_name)-1)*self.objects_sim[obj_id].clip_ft[0]) + new_object.clip_ft[0])/(len(self.objects_sim[obj_id].object_name))
            # self.objects_sim[obj_id].clip_ft.append(new_object.clip_ft[0])

            # if new_object.object_name[0] is not "oh":
            #     if self.objects_sim[obj_id].nav_pos_map is None:
            if self.objects_sim[obj_id].max_mask_size < new_object.max_mask_size < self.config.patch_size * 3:
                self.objects_sim[obj_id].nav_pos_map = new_object.nav_pos_map
                self.objects_sim[obj_id].max_mask_size = new_object.max_mask_size
                self.objects_sim[obj_id].image = new_object.image
                self.objects_sim[obj_id].mask = new_object.mask
                self.objects_sim[obj_id].depth_image = new_object.depth_image
                self.objects_sim[obj_id].pcd_frame = new_object.pcd_frame
                self.objects_sim[obj_id].center_current_frame = new_object.center_current_frame
                self.objects_sim[obj_id].bbox_area = new_object.bbox_area
                # else:
                #     if self.objects_sim[obj_id].max_mask_size < new_object.max_mask_size < self.config.patch_size*3:
                #         self.objects_sim[obj_id].nav_pos_map = new_object.nav_pos_map
                #         self.objects_sim[obj_id].max_mask_size = new_object.max_mask_size
                #         self.objects_sim[obj_id].image = new_object.image
                #         self.objects_sim[obj_id].mask = new_object.mask
                #         self.objects_sim[obj_id].depth_image = new_object.depth_image
                #         self.objects_sim[obj_id].pcd_frame = new_object.pcd_frame
                #         self.objects_sim[obj_id].center_current_frame = new_object.center_current_frame
                #         self.objects_sim[obj_id].bbox_area = new_object.bbox_area

        elif scene_type == "rend":
            if new_object.object_name[0] not in self.objects_rend[obj_id].object_name:
                self.objects_rend[obj_id].n_diff_det += 1
            self.objects_rend[obj_id].object_name.append(new_object.object_name[0])
            self.objects_rend[obj_id].object_pos_world.append(new_object.object_pos_world[0])
            self.objects_rend[obj_id].object_pos_map.append(new_object.object_pos_map[0])
            self.objects_rend[obj_id].task.append(new_object.task[0])
            self.objects_rend[obj_id].conf.append(new_object.conf[0])
            self.objects_rend[obj_id].clip_ft[0] = ((len(self.objects_rend[obj_id].object_name)-1)*self.objects_rend[obj_id].clip_ft[0] + new_object.clip_ft[0]) / (len(self.objects_rend[obj_id].object_name))
            # self.objects_rend[obj_id].clip_ft.append(new_object.clip_ft[0])
            # if new_object.avg_dist <= self.objects_rend[obj_id].avg_dist:
            #     self.objects_rend[obj_id].clip_ft[0] = new_object.clip_ft[0]
            # if new_object.object_name[0] is not "oh":
            #     if self.objects_rend[obj_id].nav_pos_map is None:
            if self.objects_rend[obj_id].max_mask_size < new_object.max_mask_size < self.config.patch_size * 3:
                self.objects_rend[obj_id].nav_pos_map = new_object.nav_pos_map
                self.objects_rend[obj_id].max_mask_size = new_object.max_mask_size
                self.objects_rend[obj_id].image = new_object.image
                self.objects_rend[obj_id].mask = new_object.mask
                self.objects_rend[obj_id].depth_image = new_object.depth_image
                self.objects_rend[obj_id].pcd_frame = new_object.pcd_frame
                self.objects_rend[obj_id].center_current_frame = new_object.center_current_frame
                self.objects_rend[obj_id].bbox_area += new_object.bbox_area
                # else:
                #     if self.objects_rend[obj_id].max_mask_size < new_object.max_mask_size < self.config.patch_size*3:
                #         self.objects_rend[obj_id].nav_pos_map = new_object.nav_pos_map
                #         self.objects_rend[obj_id].max_mask_size = new_object.max_mask_size
                #         self.objects_rend[obj_id].image = new_object.image
                #         self.objects_rend[obj_id].mask = new_object.mask
                #         self.objects_rend[obj_id].depth_image = new_object.depth_image
                #         self.objects_rend[obj_id].pcd_frame = new_object.pcd_frame
                #         self.objects_rend[obj_id].center_current_frame = new_object.center_current_frame
                #         self.objects_rend[obj_id].bbox_area += new_object.bbox_area
        else:
            raise Exception("Invalid scene type")

    def reason_about_change_gsam(self, gsam, evt, sim_image, rendered_image, rendered_lang, masks, step):
        rendered_image = rendered_image.detach().clone().cpu().numpy()
        rendered_copy = copy.deepcopy(rendered_image)
        rendered_copy = np.clip(rendered_copy, 0, 1)
        rendered_copy = (rendered_copy * 255).astype(np.uint8)
        # rendered_image = (rendered_image*255).astype(np.uint8)

        save_image_sim = copy.deepcopy(sim_image)
        save_image_rend = copy.deepcopy(rendered_image)

        pcd_frame, pcd_cam = self.get_pcd(evt)
        pcd_cam_z = pcd_cam[:, :, 2]
        pcd_cam_z = pcd_cam_z.reshape(-1)
        mean_z = pcd_cam_z.mean()
        var_z = pcd_cam_z.var()

        if mean_z > 0.7 and var_z > 0.2:
            for i, mask in enumerate(masks):
                rows, cols = np.where(mask)
                y_min, y_max = rows.min(), rows.max()
                x_min, x_max = cols.min(), cols.max()
                center = (int((x_max + x_min)/2), int((y_max + y_min)/2))
                save_image_sim[mask] = (sim_image[mask] * 0.65 + np.array([0, 255, 0]) * 0.35).astype(np.uint8)
                # Calculate the bounding box
                rows, cols = np.where(mask)
                y_min, y_max = rows.min(), rows.max()
                x_min, x_max = cols.min(), cols.max()

                sim_object = ObjectScene()
                rend_object = ObjectScene()

                # Extracting clip features for image crops
                cropped_image = self.crop_image_mask(sim_image, mask)
                crop_fts = self.clip_ft_extractor.tokenize_image(cropped_image)

                cropped_image_rend = self.crop_image_mask(rendered_copy, mask)
                crop_fts_rend = self.clip_ft_extractor.tokenize_image(cropped_image_rend)

            # GSAM
            boxes, logits, phrases, annotated_frame, masks, sim_image_vis, annotated_full = gsam.get_gsam_output(sim_image, save_image_sim, center=None)

            boxes, logits, phrases, annotated_frame, masks, rend_image_vis, annotated_full = gsam.get_gsam_output(rendered_copy, save_image_sim, center=None)

            cv2.imwrite(os.path.join(self.run_path, "reason/", "sim_{:04d}_save.png".format(step)), sim_image_vis)
            cv2.imwrite(os.path.join(self.run_path, "reason/", "rend_{:04d}_save.png".format(step)), rend_image_vis)
            # cv2.imwrite(os.path.join(self.run_path, "reason/", "main_{:04d}_save.png".format(step)), save_image_sim)


    def reason_about_change(self, evt, sim_image, rendered_image, rendered_lang, masks, step):

        # rendered_lang = rendered_lang.permute(1, 2, 0)
        rendered_image = rendered_image.detach().clone().cpu().numpy()
        rendered_copy = copy.deepcopy(rendered_image)
        rendered_copy = np.clip(rendered_copy, 0, 1)
        rendered_copy = (rendered_copy*255).astype(np.uint8)
        #rendered_image = (rendered_image*255).astype(np.uint8)

        cv2.imwrite("test_new/{}.png".format(step), rendered_copy)

        save_image_sim = copy.deepcopy(sim_image)
        save_image_rend = copy.deepcopy(rendered_image)

        pcd_frame, pcd_cam = self.get_pcd(evt)
        pcd_cam_z = copy.deepcopy(pcd_cam[:, :, 2])
        pcd_cam_z = pcd_cam_z.reshape(-1)
        mean_z = pcd_cam_z.mean()
        var_z = pcd_cam_z.var()
        min_z = pcd_cam_z.min()
        max_z = pcd_cam_z.max()
        std_z = np.std(pcd_cam_z)

        # gaussian splatting has an inherent problem with flat surfaces
        # with very fewer features like walls. To avoid false detections,
        # we don't consider detections if the robot is very close to a wall

        # if mean_z > 0.7 and var_z > 0.2:
        if (mean_z > 0.3 and var_z > 0.02) or mean_z > 0.5:
            walls = True
            for mask in masks:
                # Calculate the bounding box
                rows, cols = np.where(mask)
                y_min, y_max = rows.min(), rows.max()
                x_min, x_max = cols.min(), cols.max()

                # If the mask is more than 1 patch in size then continue with the process
                # There can be a lot of false detection, the masks of which are 1 patch size
                # This post-processing step removes them
                len_max = max(y_max-y_min, x_max-x_min)
                bbox_area = (y_max-y_min) * (x_max-x_min)
                if len_max > (self.config.patch_size-2) * (1/2):
                    sim_object = ObjectScene()
                    rend_object = ObjectScene()

                    # Extracting clip features for image crops
                    cropped_image = self.crop_image_mask(sim_image, mask)
                    crop_fts = self.clip_ft_extractor.tokenize_image(cropped_image)
                    # masked_lang = rendered_lang[mask]
                    # masked_lang = masked_lang.mean()

                    cropped_image_rend = self.crop_image_mask(rendered_copy, mask)
                    crop_fts_rend = self.clip_ft_extractor.tokenize_image(cropped_image_rend)

                    # Finding the cosine similarity with the objects
                    cs_sim = torch.nn.functional.cosine_similarity(self.all_object_fts, crop_fts)
                    cs_rend = torch.nn.functional.cosine_similarity(self.all_object_fts, crop_fts_rend)

                    # # Normalizing the cosine similarity
                    # cs_sim = (cs_sim - cs_sim.min())/(cs_sim.max() - cs_sim.min())
                    # cs_rend = (cs_rend - cs_rend.min())/(cs_rend.max() - cs_rend.min())

                    save_image_sim[mask] = (sim_image[mask]*0.65 + np.array([0, 255, 0])*0.35).astype(np.uint8)
                    save_image_rend[mask] = (save_image_rend[mask]*0.65 + np.array([0, 0.5, 0.5])*0.35)#.astype(np.uint8)

                    center = self.get_center_from_mask(mask)
                    bottom_center = self.get_bottom_center_from_mask(mask)
                    pos_map = self.navigation.explorer.mapper.get_goal_position_on_map(
                        np.array(pcd_frame[center[1], center[0], 0], pcd_frame[center[1], center[0], 0]))

                    # if cs_sim.max() > 0:
                    sim_object.object_name.append(self.all_objects[cs_sim.argmax()])
                    sim_object.object_pos_world.append(pcd_frame[center[1], center[0], :])
                    sim_object.object_pos_map.append(pos_map)
                    sim_object.conf.append(cs_sim.max())
                    sim_object.clip_ft.append(crop_fts)
                    sim_object.img_crops.append(cropped_image)
                    sim_object.points = pcd_frame[mask].reshape(-1, 3)
                    sim_object.bbox = sim_object.compute_bbox3d()
                    z_pcd = pcd_cam[:, :, 2]
                    avg_z = np.mean(z_pcd[mask])
                    sim_object.avg_dist = avg_z
                    if self.all_objects[cs_sim.argmax()] in OPENABLE_OBJECTS:
                        sim_object.task.append("open")
                    else:
                        sim_object.task.append("pick")
                    sim_object.nav_pos_map = [0, 0, 0]
                    sim_object.max_mask_size = len_max
                    sim_object.image = evt.frame
                    sim_object.mask = mask
                    sim_object.depth_image = evt.depth_frame
                    sim_object.pcd_frame = pcd_frame
                    sim_object.center_current_frame = pcd_cam[center[1], center[0], :]
                    sim_object.bbox_area = bbox_area
                    # else:
                    #     sim_object.object_name.append("oh")
                    #     sim_object.object_pos_world.append(pcd_frame[center[1], center[0], :])
                    #     sim_object.object_pos_map.append(pos_map)
                    #     sim_object.task.append("nah")
                    #     sim_object.conf.append(0)
                    #     sim_object.clip_ft.append(crop_fts)

                    # if cs_rend.max() > 0:
                    rend_object.object_name.append(self.all_objects[cs_rend.argmax()])
                    rend_object.object_pos_world.append(pcd_frame[bottom_center[1], bottom_center[0], :])
                    rend_object.object_pos_map.append(pos_map)
                    rend_object.conf.append(cs_rend.max())
                    rend_object.clip_ft.append(crop_fts_rend)
                    rend_object.img_crops.append(cropped_image_rend)
                    rend_object.points = pcd_frame[mask].reshape(-1, 3)
                    rend_object.bbox = rend_object.compute_bbox3d()
                    z_pcd = pcd_cam[:, :, 2]
                    avg_z = np.mean(z_pcd[mask])
                    rend_object.avg_dist = avg_z
                    if self.all_objects[cs_rend.argmax()] in OPENABLE_OBJECTS:
                        rend_object.task.append("open")
                    else:
                        rend_object.task.append("pick")
                    # rend_object.nav_pos_map = self.navigation.explorer.mapper.get_position_on_map()
                    rend_object.nav_pos_map = [0, 0, 0]
                    rend_object.max_mask_size = len_max
                    rend_object.image = rendered_copy
                    rend_object.mask = mask
                    rend_object.depth_image = evt.depth_frame
                    rend_object.pcd_frame = pcd_frame
                    rend_object.center_current_frame = pcd_cam[bottom_center[1], bottom_center[0], :]
                    rend_object.bbox_area = bbox_area
                    # else:
                    #     rend_object.object_name.append("oh")
                    #     rend_object.object_pos_world.append(pcd_frame[bottom_center[1], bottom_center[0], :])
                    #     rend_object.object_pos_map.append(pos_map)
                    #     rend_object.task.append("nah")
                    #     rend_object.conf.append(0)
                    #     rend_object.clip_ft.append(crop_fts_rend)

                    if sim_object.object_name[0] not in ["plain surface", "wall", "mirror", "window"]:
                        # TODO: check for similar objects in memory
                        if self.sim_obj_id == 0:
                            # if cs_sim.max() > 0:
                            self.objects_sim[self.sim_obj_id] = sim_object
                            self.sim_centers.append(sim_object.clip_ft[0])
                            self.sim_obj_id += 1
                            obj_id_save = 0
                        else:
                            # if cs_sim.max() > 0:
                            obj_id, ssim, ssem_sim, ssp_sim = self.check_similarity(self.objects_sim, sim_object, sim_object.clip_ft[0], self.sim_centers)

                            if obj_id == -1:
                                obj_id_save = copy.deepcopy(self.sim_obj_id)
                                self.objects_sim[self.sim_obj_id] = sim_object
                                self.sim_centers.append(sim_object.clip_ft[0])
                                self.sim_obj_id += 1
                            else:
                                self.merge_detections("sim", sim_object, obj_id)
                                obj_id_save = copy.deepcopy(obj_id)
                        # else:
                        #     obj_id = self.check_similarity(self.objects_sim, sim_object, sim_object.clip_ft[0], self.sim_centers)
                        #
                        #     # if it is a new detection, and it is a false detection, then discard it
                        #     if obj_id != -1:
                        #         self.objects_sim[obj_id].n_false_det += 1
                        #         obj_id_save = copy.deepcopy(obj_id)
                        #     else:
                        #         obj_id_save = "#"

                            # string_txt = str(obj_id) + "/" + str(self.sim_obj_id) + "/" + str(ssim)
                            # save_image_sim = cv2.putText(save_image_sim, string_txt, center,
                            #                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                    if rend_object.object_name[0] not in ["plain surface", "wall", "mirror", "window"]:
                        if self.rend_obj_id == 0:
                            # if cs_rend.max() > 0:
                            self.objects_rend[self.rend_obj_id] = rend_object
                            self.rend_centers.append(rend_object.clip_ft[0])
                            self.rend_obj_id += 1
                            obj_id_save = 0
                        else:
                            # if cs_rend.max() > 0:
                            obj_id, rsim, rsem_sim, rsp_sim = self.check_similarity(self.objects_rend, rend_object, rend_object.clip_ft[0],
                                                           self.rend_centers)

                            if obj_id == -1:
                                obj_id_save = copy.deepcopy(self.rend_obj_id)
                                self.objects_rend[self.rend_obj_id] = rend_object
                                self.rend_centers.append(rend_object.clip_ft[0])
                                self.rend_obj_id += 1
                            else:
                                self.merge_detections("rend", rend_object, obj_id)
                                obj_id_save = copy.deepcopy(obj_id)
                            # else:
                            #     obj_id = self.check_similarity(self.objects_rend, rend_object, rend_object.clip_ft[0],
                            #                                    self.rend_centers)
                            #
                            #     # if it is a new detection, and it is a false detection, then discard it
                            #     if obj_id != -1:
                            #         self.objects_rend[obj_id].n_false_det += 1
                            #         obj_id_save = copy.deepcopy(obj_id)
                            #     else:
                            #         obj_id_save = "#"
                            x, y = center
                            string_txt = str(obj_id) + "/" + str(self.rend_obj_id) + "/" + str(rsim)
                            # save_image_sim = cv2.putText(save_image_sim, string_txt,
                            #                              (x, y + 15),
                            #                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    del sim_object
                    del rend_object
            save_image_sim = torch.tensor(save_image_sim / 255)
            save_image_rend = torch.tensor(save_image_rend)
            full_image = torch.zeros(save_image_sim.shape[0], save_image_sim.shape[1]*2, 3)
            full_image[:, :save_image_sim.shape[1], :] = save_image_sim
            full_image[:, save_image_sim.shape[1]:, :] = save_image_rend

            if isinstance(step, str):
                torchvision.utils.save_image(full_image.permute(2, 0, 1),
                                             os.path.join(self.run_path, "reason/", "{}_save.png".format(step)))
            else:
                torchvision.utils.save_image(full_image.permute(2, 0, 1),
                                             os.path.join(self.run_path, "reason/", "{:04d}_save.png".format(step)))


            # print("sim objects at step {}".format(step), len(self.objects_sim))
            # print("rend objects at step {}".format(step), len(self.objects_rend))

        else:
            walls = False
            # print("too close to wall")
            return walls

    def calculate_cosine_similarity_matrix(self):
        # features1 = np.array([(self.objects_sim[idx_sim].clip_ft[0].cpu().numpy()).tolist() for idx_sim in self.objects_sim])
        # features2 = np.array([(self.objects_rend[idx_rend].clip_ft[0].cpu().numpy()).tolist() for idx_rend in self.objects_rend])
        features1 = np.array([(self.objects_sim[idx_sim].final_clip_ft.cpu().numpy()).tolist() for idx_sim in self.objects_sim])
        features2 = np.array([(self.objects_rend[idx_rend].final_clip_ft.cpu().numpy()).tolist() for idx_rend in self.objects_rend])
        features1 = features1.reshape(features1.shape[0], features1.shape[2])
        features2 = features2.reshape(features2.shape[0], features2.shape[2])
        similarity_matrix = cosine_similarity(features1, features2)
        return similarity_matrix

    def bipartite_matching(self):
        print("Started matching objects")
        len_sim = len(self.objects_sim)
        len_rend = len(self.objects_rend)
        sim_matrix = self.calculate_cosine_similarity_matrix()
        # Padding for bipartite matching with Hungarian algorithm
        if len_sim != len_rend:
            if len_sim > len_rend:
                padded_similarity_matrix = np.hstack((sim_matrix, np.zeros((len_sim, len_sim - len_rend))))
            else:
                padded_similarity_matrix = np.vstack((sim_matrix, np.zeros((len_rend - len_sim, len_rend))))
        else:
            padded_similarity_matrix = sim_matrix

        cost_matrix = -padded_similarity_matrix
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        rearrange_dict = {}
        sim = {}
        for r, c in zip(row_ind, col_ind):
            if r < len_sim and c < len_rend:
                rearrange_dict[c] = r
                sim[c] = padded_similarity_matrix[r, c]
        print("Finished matching objects")
        return rearrange_dict, sim

    def postprocess_detections(self, sam_predictor):
        rearrange_dict = {}
        sim_checker = {}
        other_data = {}
        open_list = []
        open_list_rend = []

        # finding accurate centers
        print("\nFinding accurate centers for detected segments in rendered splat")
        for rend_idx in self.objects_rend:
            rows, cols = np.where(self.objects_rend[rend_idx].mask)
            y_min, y_max = rows.min(), rows.max()
            x_min, x_max = cols.min(), cols.max()
            h, w, _ = self.objects_rend[rend_idx].image.shape

            padding = 2
            x_min = x_min - padding
            x_max = x_max + padding
            y_min = y_min - padding
            y_max = y_max + padding

            x_min = max(x_min, 0)
            x_max = min(x_max, w)
            y_min = max(y_min, 0)
            y_max = min(y_max, h)
            bbox = [x_min, y_min, x_max, y_max]

            # Getting a better mask
            mask_sam = sam_predictor.get_mask(self.objects_rend[rend_idx].image, bbox)

            # Getting image crop
            crop_sam = self.crop_image_mask_plainbg(self.objects_rend[rend_idx].image, mask_sam)
            self.objects_rend[rend_idx].accurate_image_crop = crop_sam

            # Getting better feature vector
            final_clip_vector = self.clip_ft_extractor.tokenize_image(crop_sam)
            self.objects_rend[rend_idx].final_clip_ft = final_clip_vector

            # sampling an interior pixel
            mask_sam_copy = copy.deepcopy(mask_sam)
            mask_sam = (mask_sam * 255).astype(np.uint8)
            kernel = np.ones((3, 3), np.uint8)
            eroded_mask = cv2.erode(mask_sam, kernel, iterations=1)

            rows, cols = np.where(eroded_mask != 0)
            if len(rows) != 0 and len(cols) != 0:
                y_sel = rows[int(len(rows) / 2)]
                x_sel = cols[int(len(cols) / 2)]
            else:
                y_sel = int((bbox[1] + bbox[3]) / 2)
                x_sel = int((bbox[0] + bbox[2]) / 2)

            self.objects_rend[rend_idx].center_accurate = self.objects_rend[rend_idx].pcd_frame[y_sel, x_sel, :]

            rgb_copy = copy.deepcopy(self.objects_rend[rend_idx].image)
            mask_copy = copy.deepcopy(mask_sam_copy)

            rgb_copy[mask_copy] = (rgb_copy[mask_copy] * 0.65 + np.array([0, 255, 0]) * 0.35).astype(np.uint8)
            bgr_img = cv2.cvtColor(rgb_copy, cv2.COLOR_RGB2BGR)
            bgr_img = cv2.putText(bgr_img, "-P", org=(int(x_sel), int(y_sel)),
                                  fontFace=cv2.FONT_HERSHEY_PLAIN, color=(255, 0, 0), fontScale=1)
            bgr_img = cv2.rectangle(bgr_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=(0, 0, 255), thickness=1)
            cv2.imwrite(os.path.join(self.config.run_path, 'pick/' + "post_rend" + str(rend_idx) + '.jpg'), bgr_img)

        print("\nFinding accurate centers for detected segments in simulation")
        for sim_idx in self.objects_sim:
            rows, cols = np.where(self.objects_sim[sim_idx].mask)
            y_min, y_max = rows.min(), rows.max()
            x_min, x_max = cols.min(), cols.max()
            h, w, _ = self.objects_sim[sim_idx].image.shape

            padding = 2
            x_min = x_min - padding
            x_max = x_max + padding
            y_min = y_min - padding
            y_max = y_max + padding

            x_min = max(x_min, 0)
            x_max = min(x_max, w)
            y_min = max(y_min, 0)
            y_max = min(y_max, h)
            bbox = [x_min, y_min, x_max, y_max]

            # Getting a better mask
            mask_sam = sam_predictor.get_mask(self.objects_sim[sim_idx].image, bbox)

            # Getting image crop
            crop_sam = self.crop_image_mask_plainbg(self.objects_sim[sim_idx].image, mask_sam)
            self.objects_sim[sim_idx].accurate_image_crop = crop_sam

            # Getting better feature vector
            final_clip_vector = self.clip_ft_extractor.tokenize_image(crop_sam)
            self.objects_sim[sim_idx].final_clip_ft = final_clip_vector

            # sampling an interior pixel
            mask_sam_copy = copy.deepcopy(mask_sam)
            mask_sam = (mask_sam * 255).astype(np.uint8)
            kernel = np.ones((3, 3), np.uint8)
            eroded_mask = cv2.erode(mask_sam, kernel, iterations=1)

            rows, cols = np.where(eroded_mask != 0)
            if len(rows) != 0 and len(cols) != 0:
                y_sel = rows[int(len(rows) / 2)]
                x_sel = cols[int(len(cols) / 2)]
            else:
                y_sel = int((bbox[1] + bbox[3]) / 2)
                x_sel = int((bbox[0] + bbox[2]) / 2)

            self.objects_sim[sim_idx].center_accurate = self.objects_sim[sim_idx].pcd_frame[y_sel, x_sel, :]

            rgb_copy = copy.deepcopy(self.objects_sim[sim_idx].image)
            mask_copy = copy.deepcopy(mask_sam_copy)

            rgb_copy[mask_copy] = (rgb_copy[mask_copy] * 0.65 + np.array([0, 255, 0]) * 0.35).astype(np.uint8)
            bgr_img = cv2.cvtColor(rgb_copy, cv2.COLOR_RGB2BGR)
            bgr_img = cv2.putText(bgr_img, "-P", org=(int(x_sel), int(y_sel)),
                                  fontFace=cv2.FONT_HERSHEY_PLAIN, color=(255, 0, 0), fontScale=1)
            bgr_img = cv2.rectangle(bgr_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=(0, 0, 255), thickness=1)
            cv2.imwrite(os.path.join(self.config.run_path, 'pick/' + "post_rend" + str(sim_idx) + '.jpg'), bgr_img)


        for i in self.objects_sim:
            tasks = self.objects_sim[i].task
            task_pick = 0
            for task in tasks:
                if task == 'pick':
                    task_pick += 1
            task_pick_percent = task_pick / len(tasks)
            self.objects_sim[i].pick_percent = task_pick_percent
            if task_pick_percent >= 0.5:
                self.objects_sim[i].task_final = 'pick'
            else:
                self.objects_sim[i].task_final = 'open'
            save_path_sim = os.path.join(self.config.run_path, 'rearr/sim')
            if not os.path.exists(save_path_sim):
                os.makedirs(save_path_sim)
            cv2.imwrite(os.path.join(save_path_sim, "before_{}.png".format(i)),
                        cv2.cvtColor(self.objects_sim[i].accurate_image_crop, cv2.COLOR_BGR2RGB))

        for i in self.objects_rend:
            tasks = self.objects_rend[i].task
            task_pick = 0
            for task in tasks:
                if task == 'pick':
                    task_pick += 1
            task_pick_percent = task_pick / len(tasks)
            self.objects_rend[i].pick_percent = task_pick_percent
            if task_pick_percent >= 0.5:
                self.objects_rend[i].task_final = 'pick'
            else:
                self.objects_rend[i].task_final = 'open'
            save_path_rend = os.path.join(self.config.run_path, 'rearr/rend')
            if not os.path.exists(save_path_rend):
                os.makedirs(save_path_rend)
            cv2.imwrite(os.path.join(save_path_rend, "before_{}.png".format(i)),
                        cv2.cvtColor(self.objects_rend[i].accurate_image_crop, cv2.COLOR_BGR2RGB))



        # Pruning out similar detections across simulated scene
        new_objects_sim_idx = []
        new_clip_fts = []
        for i in self.objects_sim:
            if len(new_objects_sim_idx) == 0:
                new_objects_sim_idx.append(i)
                new_clip_fts.append(self.objects_sim[i].final_clip_ft)
            else:
                clip_fts_ = torch.stack(new_clip_fts).squeeze(1)
                clip_ft_curr = self.objects_sim[i].final_clip_ft
                cs = torch.nn.functional.cosine_similarity(clip_ft_curr, clip_fts_)
                if torch.max(cs) > 0.8:
                    # same object
                    idx = torch.argmax(cs)
                    idx = int(idx.item())
                    if self.objects_sim[i].bbox_area > self.objects_sim[new_objects_sim_idx[idx]].bbox_area:
                        new_objects_sim_idx[idx] = i
                        new_clip_fts[idx] = clip_ft_curr
                else:
                    # new object
                    new_objects_sim_idx.append(i)
                    new_clip_fts.append(clip_ft_curr)

        new_objects_sim = {}
        for i in range(len(new_objects_sim_idx)):
            new_objects_sim[i] = self.objects_sim[new_objects_sim_idx[i]]
        del self.objects_sim
        self.objects_sim = new_objects_sim

        # Pruning out similar detections across rendered scene
        new_objects_rend_idx = []
        new_clip_fts = []
        for i in self.objects_rend:
            if len(new_objects_rend_idx) == 0:
                new_objects_rend_idx.append(i)
                new_clip_fts.append(self.objects_rend[i].final_clip_ft)
            else:
                clip_fts_ = torch.stack(new_clip_fts).squeeze(1)
                clip_ft_curr = self.objects_rend[i].final_clip_ft
                cs = torch.nn.functional.cosine_similarity(clip_ft_curr, clip_fts_)
                if torch.max(cs) > 0.8:
                    # same object
                    idx = torch.argmax(cs)
                    idx = int(idx.item())
                    if self.objects_rend[i].bbox_area > self.objects_rend[new_objects_rend_idx[idx]].bbox_area:
                        new_objects_rend_idx[idx] = i
                        new_clip_fts[idx] = clip_ft_curr
                else:
                    # new object
                    new_objects_rend_idx.append(i)
                    new_clip_fts.append(clip_ft_curr)

        new_objects_rend = {}
        for i in range(len(new_objects_rend_idx)):
            new_objects_rend[i] = self.objects_rend[new_objects_rend_idx[i]]
        del self.objects_rend
        self.objects_rend = new_objects_rend


        for i in self.objects_sim:
            tasks = self.objects_sim[i].task
            task_pick = 0
            for task in tasks:
                if task == 'pick':
                    task_pick += 1
            task_pick_percent = task_pick / len(tasks)
            self.objects_sim[i].pick_percent = task_pick_percent
            if task_pick_percent >= 0.5:
                self.objects_sim[i].task_final = 'pick'
            else:
                self.objects_sim[i].task_final = 'open'
            save_path_sim = os.path.join(self.config.run_path, 'rearr/sim')
            if not os.path.exists(save_path_sim):
                os.makedirs(save_path_sim)
            cv2.imwrite(os.path.join(save_path_sim, "{}.png".format(i)),
                        cv2.cvtColor(self.objects_sim[i].accurate_image_crop, cv2.COLOR_BGR2RGB))

        for i in self.objects_rend:
            tasks = self.objects_rend[i].task
            task_pick = 0
            for task in tasks:
                if task == 'pick':
                    task_pick += 1
            task_pick_percent = task_pick / len(tasks)
            self.objects_rend[i].pick_percent = task_pick_percent
            if task_pick_percent >= 0.5:
                self.objects_rend[i].task_final = 'pick'
            else:
                self.objects_rend[i].task_final = 'open'
            save_path_rend = os.path.join(self.config.run_path, 'rearr/rend')
            if not os.path.exists(save_path_rend):
                os.makedirs(save_path_rend)
            cv2.imwrite(os.path.join(save_path_rend, "{}.png".format(i)),
                        cv2.cvtColor(self.objects_rend[i].accurate_image_crop, cv2.COLOR_BGR2RGB))


        # GREEDY ASSIGNMENT STRATEGY
        if self.config.greedy_strat:
            for i in self.objects_sim:
                # Prune out detections if the object is observed from a far away distance
                if abs(self.objects_sim[i].center_current_frame[2]) < 1.5:
                    if self.objects_sim[i].n_false_det < 40: #and self.objects_sim[i].n_diff_det < 3:
                        #print(self.objects_sim[i].object_name, self.objects_sim[i].conf, self.objects_sim[i].n_false_det)
                        cosine_sim_max = 0
                        idx_max = 0
                        img_sim = self.objects_sim[i].img_crops[0]
                        for j in self.objects_rend:
                            if abs(self.objects_rend[j].center_current_frame[2]) < 1.5:
                                if self.objects_rend[j].n_false_det < 40: #and self.objects_rend[j].n_diff_det < 3:
                                    cosine_sim = torch.nn.functional.cosine_similarity(self.objects_sim[i].clip_ft[0], self.objects_rend[j].clip_ft[0])
                                    if cosine_sim > cosine_sim_max:
                                        cosine_sim_max = cosine_sim
                                        idx_max = j
                        if idx_max in rearrange_dict.keys():
                            if sim_checker[idx_max] < cosine_sim_max < 0.9:
                                sim_checker[idx_max] = cosine_sim_max
                                rearrange_dict[idx_max] = i
                        else:
                            if 0.75 < cosine_sim_max < 0.9: # 0.75, 0.9
                                rearrange_dict[idx_max] = i
                                sim_checker[idx_max] = cosine_sim_max
                if self.objects_sim[i].pick_percent < 0.7:
                    # object to open
                    if self.objects_sim[i].n_false_det < 5:
                        if abs(self.objects_sim[i].center_current_frame[2]) < 8:
                            # TODO: work out postprocessing steps for openable objects
                            if self.objects_sim[i].max_mask_size > self.config.patch_size:
                                open_list.append(i)
        else:
            # HUNGARIAN ALGORITHM
            rearrange_dict, sim_checker = self.bipartite_matching()

        for rend_idx in rearrange_dict:
            sim_idx = rearrange_dict[rend_idx]
            img_sim = self.objects_sim[sim_idx].accurate_image_crop
            img_rend = self.objects_rend[rend_idx].accurate_image_crop
            image = torch.zeros(img_sim.shape[0] + img_rend.shape[0],
                                max(img_sim.shape[1], img_rend.shape[1]), 3)
            image[:img_sim.shape[0], :img_sim.shape[1], :] = torch.tensor(img_sim/255)
            image[img_sim.shape[0]:, :img_rend.shape[1], :] = torch.tensor(img_rend/255)
            torchvision.utils.save_image(image.permute(2, 0, 1),
                                         os.path.join(self.run_path, "rearr/before_pruning_{:03d}_{:03d}_save.png".format(sim_idx, rend_idx)))

        for rend_idx in rearrange_dict:
            sim_idx = rearrange_dict[rend_idx]
            img_sim = self.objects_sim[sim_idx].img_crops[0]
            img_rend = self.objects_rend[rend_idx].img_crops[0]
            image = torch.zeros(img_sim.shape[0] + img_rend.shape[0],
                                max(img_sim.shape[1], img_rend.shape[1]), 3)
            image[:img_sim.shape[0], :img_sim.shape[1], :] = torch.tensor(img_sim/255)
            image[img_sim.shape[0]:, :img_rend.shape[1], :] = torch.tensor(img_rend/255)
            torchvision.utils.save_image(image.permute(2, 0, 1),
                                         os.path.join(self.run_path, "rearr/before_pruning_{:03d}_{:03d}_save.png".format(sim_idx, rend_idx)))

        # for i in self.objects_sim:
        #     if self.objects_sim[i].task_final == 'open':
        #         open_list.append(i)
        #
        # for i in self.objects_rend:
        #     if self.objects_rend[i].task_final == 'open':
        #         open_list_rend.append(i)

        open_list = []
        open_list_rend = []

        pruned_open_list = []
        open_object_clip_fts = []
        for open_idx in open_list:
            if len(pruned_open_list) == 0:
                pruned_open_list.append(open_idx)
                open_object_clip_fts.append(self.objects_sim[open_idx].clip_ft[0])
            else:
                clip_fts = torch.stack(open_object_clip_fts).squeeze(1)
                cs = torch.nn.functional.cosine_similarity(self.objects_sim[open_idx].clip_ft[0], clip_fts)
                if torch.max(cs) > 0.8:
                    # same object
                    idx = torch.argmax(cs)
                    idx = int(idx.item())
                    if self.objects_sim[open_idx].bbox_area > self.objects_sim[pruned_open_list[idx]].bbox_area:
                        pruned_open_list[idx] = open_idx
                        open_object_clip_fts[idx] = (self.objects_sim[open_idx].clip_ft[0])
                else:
                    # new object
                    pruned_open_list.append(open_idx)
                    open_object_clip_fts.append(self.objects_sim[open_idx].clip_ft[0])
        open_list = pruned_open_list

        pruned_open_list_rend = []
        open_object_clip_fts = []
        for open_idx in open_list_rend:
            if len(pruned_open_list_rend) == 0:
                pruned_open_list_rend.append(open_idx)
                open_object_clip_fts.append(self.objects_rend[open_idx].clip_ft[0])
            else:
                clip_fts = torch.stack(open_object_clip_fts).squeeze(1)
                cs = torch.nn.functional.cosine_similarity(self.objects_rend[open_idx].clip_ft[0], clip_fts)
                if torch.max(cs) > 0.8:
                    # same object
                    idx = torch.argmax(cs)
                    idx = int(idx.item())
                    if self.objects_rend[open_idx].bbox_area > self.objects_rend[pruned_open_list_rend[idx]].bbox_area:
                        pruned_open_list_rend[idx] = open_idx
                        open_object_clip_fts[idx] = (self.objects_rend[open_idx].clip_ft[0])
                else:
                    # new object
                    pruned_open_list_rend.append(open_idx)
                    open_object_clip_fts.append(self.objects_rend[open_idx].clip_ft[0])
        open_list_rend = pruned_open_list_rend

        # finding accurate centers
        for open_idx in open_list:
            rows, cols = np.where(self.objects_sim[open_idx].mask)
            y_min, y_max = rows.min(), rows.max()
            x_min, x_max = cols.min(), cols.max()
            h, w, _ = self.objects_sim[open_idx].image.shape
            padding = 3
            x_min = x_min - padding
            x_max = x_max + padding
            y_min = y_min - padding
            y_max = y_max + padding

            x_min = max(x_min, 0)
            x_max = min(x_max, w)
            y_min = max(y_min, 0)
            y_max = min(y_max, h)
            bbox = [x_min, y_min, x_max, y_max]
            y_sel = int((bbox[1] + bbox[3]) / 2)
            x_sel = int((bbox[0] + bbox[2]) / 2)
            self.objects_sim[open_idx].center_accurate = self.objects_sim[open_idx].pcd_frame[y_sel, x_sel, :]

            rgb_copy = copy.deepcopy(self.objects_sim[open_idx].image)
            mask_copy = copy.deepcopy(self.objects_sim[open_idx].mask)
            rgb_copy[mask_copy] = (rgb_copy[mask_copy] * 0.65 + np.array([0, 255, 0]) * 0.35).astype(np.uint8)
            bgr_img = cv2.cvtColor(rgb_copy, cv2.COLOR_RGB2BGR)
            bgr_img = cv2.putText(bgr_img, "-P", org=(int(x_sel), int(y_sel)),
                                  fontFace=cv2.FONT_HERSHEY_PLAIN, color=(255, 0, 0), fontScale=1)
            bgr_img = cv2.rectangle(bgr_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=(0, 0, 255), thickness=1)
            cv2.imwrite(os.path.join(self.config.run_path, 'pick/' + "open_post_" + str(open_idx) + '.jpg'), bgr_img)


        # for i in self.objects_rend:
        #     if self.objects_rend[i].pick_percent < 0.7:
        #         # pick object
        #         clip_ft_rend = self.objects_rend[i].clip_ft[0]
        #         max_clip_sim = 0
        #         sim_idx_max = 0
        #         for open_idx_ in open_list:
        #             clip_ft_sim = self.objects_sim[open_idx_].clip_ft[0]
        #             cs = torch.nn.functional.cosine_similarity(clip_ft_rend, clip_ft_sim)
        #             if cs > max_clip_sim:
        #                 max_clip_sim = cs
        #                 sim_idx_max = open_idx_
        #         if max_clip_sim > 0.8:
        #             # same object, save only one detection
        #             pass
        #         else:
        #             open_list_rend.append(i)

        pruned_rearrange_dict = {}
        for rend_idx in rearrange_dict:
            sim_idx = rearrange_dict[rend_idx]
            if sim_checker[rend_idx] > 0.6:
                pruned_rearrange_dict[rend_idx] = sim_idx
        rearrange_dict = pruned_rearrange_dict

        # pruned_rearrange_dict = {}
        # for rend_idx in rearrange_dict:
        #     sim_idx = rearrange_dict[rend_idx]
        #     if len(pruned_rearrange_dict) == 0:
        #         pruned_rearrange_dict[rend_idx] = rearrange_dict[rend_idx]
        #     else:
        #         max_cs = 0
        #         max_id = 0
        #         for id_ in pruned_rearrange_dict:
        #             cs = torch.nn.functional.cosine_similarity(self.objects_sim[sim_idx].clip_ft[0],
        #                                                        self.objects_sim[pruned_rearrange_dict[id_]].clip_ft[0])
        #             if cs > max_cs:
        #                 max_cs = cs
        #                 max_id = id_
        #         if max_cs > 0.85:
        #             pass
        #             # same object
        #             # TODO: check the threshold
        #             # if sim_checker[max_id] > sim_checker[rend_idx]:
        #             #     # replace detection
        #             #     del pruned_rearrange_dict[max_id]
        #             #     pruned_rearrange_dict[rend_idx] = rearrange_dict[rend_idx]
        #         else:
        #             pruned_rearrange_dict[rend_idx] = rearrange_dict[rend_idx]
        # rearrange_dict = pruned_rearrange_dict

        pruned_sim_checker = {}
        for rend_idx in rearrange_dict:
            pruned_sim_checker[rend_idx] = sim_checker[rend_idx]

        sorted_sim_checker = {k: v for k, v in sorted(rearrange_dict.items(), key=lambda item: item[1], reverse=True)}

        pruned_rearrange_dict = {}
        for rend_idx in sorted_sim_checker:
            pruned_rearrange_dict[rend_idx] = rearrange_dict[rend_idx]
        rearrange_dict = pruned_rearrange_dict

        # # finding accurate centers
        # for rend_idx in rearrange_dict:
        #     sim_idx = rearrange_dict[rend_idx]
        #     rows, cols = np.where(self.objects_sim[sim_idx].mask)
        #     y_min, y_max = rows.min(), rows.max()
        #     x_min, x_max = cols.min(), cols.max()
        #     h, w, _ = self.objects_sim[sim_idx].image.shape
        #     padding = 2
        #     x_min = x_min - padding
        #     x_max = x_max + padding
        #     y_min = y_min - padding
        #     y_max = y_max + padding
        #
        #     x_min = max(x_min, 0)
        #     x_max = min(x_max, w)
        #     y_min = max(y_min, 0)
        #     y_max = min(y_max, h)
        #     bbox = [x_min, y_min, x_max, y_max]
        #
        #     # Getting a better mask
        #     mask_sam = sam_predictor.get_mask(self.objects_sim[sim_idx].image, bbox)
        #
        #     # sampling an interior pixel
        #     mask_sam_copy = copy.deepcopy(mask_sam)
        #     mask_sam = (mask_sam*255).astype(np.uint8)
        #     kernel = np.ones((3, 3), np.uint8)
        #     eroded_mask = cv2.erode(mask_sam, kernel, iterations=1)
        #
        #     rows, cols = np.where(eroded_mask != 0)
        #     if len(rows) != 0 and len(cols) != 0:
        #         y_sel = rows[int(len(rows)/2)]
        #         x_sel = cols[int(len(cols)/2)]
        #     else:
        #         y_sel = int((bbox[1]+bbox[3])/2)
        #         x_sel = int((bbox[0]+bbox[2])/2)
        #
        #     self.objects_sim[sim_idx].center_accurate = self.objects_sim[sim_idx].pcd_frame[y_sel, x_sel, :]
        #
        #     rgb_copy = copy.deepcopy(self.objects_sim[sim_idx].image)
        #     mask_copy = copy.deepcopy(mask_sam_copy)
        #
        #     rgb_copy[mask_copy] = (rgb_copy[mask_copy] * 0.65 + np.array([0, 255, 0]) * 0.35).astype(np.uint8)
        #     bgr_img = cv2.cvtColor(rgb_copy, cv2.COLOR_RGB2BGR)
        #     bgr_img = cv2.putText(bgr_img, "-P", org=(int(x_sel), int(y_sel)),
        #                           fontFace=cv2.FONT_HERSHEY_PLAIN, color=(255, 0, 0), fontScale=1)
        #     bgr_img = cv2.rectangle(bgr_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=(0, 0, 255), thickness=1)
        #     cv2.imwrite(os.path.join(self.config.run_path, 'pick/' + "post_" + str(sim_idx) + '.jpg'), bgr_img)

        # saving the detected pairs
        for rend_idx in rearrange_dict:
            sim_idx = rearrange_dict[rend_idx]
            img_sim = self.objects_sim[sim_idx].accurate_image_crop
            img_rend = self.objects_rend[rend_idx].accurate_image_crop
            image = torch.zeros(img_sim.shape[0] + img_rend.shape[0],
                                max(img_sim.shape[1], img_rend.shape[1]), 3)
            image[:img_sim.shape[0], :img_sim.shape[1], :] = torch.tensor(img_sim/255)
            image[img_sim.shape[0]:, :img_rend.shape[1], :] = torch.tensor(img_rend/255)
            if 0.5 < sim_checker[rend_idx] < 0.95:
                save_path_ = os.path.join(self.run_path, "rearr/high")
                if not os.path.exists(save_path_):
                    os.makedirs(save_path_)
                torchvision.utils.save_image(image.permute(2, 0, 1),
                                             os.path.join(save_path_, "{}_{:03d}_{:03d}_save.png".format(sim_checker[rend_idx], sim_idx, rend_idx)))
            else:
                save_path_ = os.path.join(self.run_path, "rearr/low")
                if not os.path.exists(save_path_):
                    os.makedirs(save_path_)
                torchvision.utils.save_image(image.permute(2, 0, 1),
                                             os.path.join(save_path_,
                                                          "{}_{:03d}_{:03d}_save.png".format(sim_checker[rend_idx], sim_idx, rend_idx)))

        for open_idx in open_list:
            rows, cols = np.where(self.objects_sim[open_idx].mask)
            y_min, y_max = rows.min(), rows.max()
            x_min, x_max = cols.min(), cols.max()
            h, w, _ = self.objects_sim[open_idx].image.shape
            padding = 2
            x_min = x_min - padding
            x_max = x_max + padding
            y_min = y_min - padding
            y_max = y_max + padding

            x_min = max(x_min, 0)
            x_max = min(x_max, w)
            y_min = max(y_min, 0)
            y_max = min(y_max, h)
            bbox = [x_min, y_min, x_max, y_max]

            # Getting a better mask
            mask_sam = sam_predictor.get_mask(self.objects_sim[open_idx].image, bbox)

            # sampling an interior pixel
            mask_sam_copy = copy.deepcopy(mask_sam)
            mask_sam = (mask_sam * 255).astype(np.uint8)
            kernel = np.ones((3, 3), np.uint8)
            eroded_mask = cv2.erode(mask_sam, kernel, iterations=1)

            rows, cols = np.where(eroded_mask != 0)
            if len(rows) != 0 and len(cols) != 0:
                y_sel = rows[int(len(rows) / 2)]
                x_sel = cols[int(len(cols) / 2)]
            else:
                y_sel = int((bbox[1] + bbox[3]) / 2)
                x_sel = int((bbox[0] + bbox[2]) / 2)

            self.objects_sim[open_idx].center_accurate = self.objects_sim[open_idx].pcd_frame[y_sel, x_sel, :]


            sim_image = copy.deepcopy(self.objects_sim[open_idx].image)
            mask = copy.deepcopy(self.objects_sim[open_idx].mask)
            sim_image[mask] = (sim_image[mask] * 0.65 + np.array([0, 255, 0]) * 0.35).astype(np.uint8)
            cv2.imwrite(os.path.join(self.run_path, "rearr/open_{}_{}.png".format(open_idx, str(self.objects_sim[open_idx].object_name))), sim_image)

        for open_idx in open_list_rend:
            rows, cols = np.where(self.objects_rend[open_idx].mask)
            y_min, y_max = rows.min(), rows.max()
            x_min, x_max = cols.min(), cols.max()
            h, w, _ = self.objects_rend[open_idx].image.shape
            padding = 2
            x_min = x_min - padding
            x_max = x_max + padding
            y_min = y_min - padding
            y_max = y_max + padding

            x_min = max(x_min, 0)
            x_max = min(x_max, w)
            y_min = max(y_min, 0)
            y_max = min(y_max, h)
            bbox = [x_min, y_min, x_max, y_max]

            # Getting a better mask
            mask_sam = sam_predictor.get_mask(self.objects_rend[open_idx].image, bbox)

            # sampling an interior pixel
            mask_sam_copy = copy.deepcopy(mask_sam)
            mask_sam = (mask_sam * 255).astype(np.uint8)
            kernel = np.ones((3, 3), np.uint8)
            eroded_mask = cv2.erode(mask_sam, kernel, iterations=1)

            rows, cols = np.where(eroded_mask != 0)
            if len(rows) != 0 and len(cols) != 0:
                y_sel = rows[int(len(rows) / 2)]
                x_sel = cols[int(len(cols) / 2)]
            else:
                y_sel = int((bbox[1] + bbox[3]) / 2)
                x_sel = int((bbox[0] + bbox[2]) / 2)

            self.objects_rend[open_idx].center_accurate = self.objects_rend[open_idx].pcd_frame[y_sel, x_sel, :]

            rend_image = copy.deepcopy(self.objects_rend[open_idx].image)
            mask = copy.deepcopy(self.objects_rend[open_idx].mask)
            rend_image[mask] = (rend_image[mask] * 0.65 + np.array([0, 255, 0]) * 0.35).astype(np.uint8)
            cv2.imwrite(os.path.join(self.run_path, "rearr/rend_open_{}_{}.png".format(open_idx, str(self.objects_rend[open_idx].object_name))), rend_image)

        # ordering the rearrange dict in terms of clip similarity
        rend_idxs = list(rearrange_dict.keys())
        sim_val = []
        for rend_idx in rend_idxs:
            sim_val.append(sim_checker[rend_idx])
        sorted_pairs = sorted(zip(sim_val, rend_idxs), reverse=True)
        sorted_sim_scores, sorted_rend_idx = zip(*sorted_pairs)
        sorted_rearrange_dict = {}
        for rend_idx in sorted_rend_idx:
            if sim_checker[rend_idx] < 0.95 and sim_checker[rend_idx] > 0.65:
                sorted_rearrange_dict[rend_idx] = rearrange_dict[rend_idx]
        #rearrange_dict = copy.deepcopy(sorted_rearrange_dict)

        # ordering the rearrange dict in terms of mask_size
        # if len(rearrange_dict) != 0:
        #     rend_idxs = list(rearrange_dict.keys())
        #     bbox_area = []
        #     for rend_idx in rend_idxs:
        #         sim_idx = rearrange_dict[rend_idx]
        #         bbox_area.append(self.objects_sim[sim_idx].bbox_area)
        #     sorted_pairs = sorted(zip(bbox_area, rend_idxs), reverse=True)
        #     sorted_sim_scores, sorted_rend_idx = zip(*sorted_pairs)
        #     sorted_rearrange_dict = {}
        #     for rend_idx in sorted_rend_idx:
        #         sorted_rearrange_dict[rend_idx] = rearrange_dict[rend_idx]
        # else:
        #     sorted_rearrange_dict = rearrange_dict


        return sorted_rearrange_dict, open_list, open_list_rend
