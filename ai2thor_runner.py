import time
from ai2thor.controller import Controller
from scipy.spatial.transform import Rotation

import torch
import torchvision
from rich.console import Console
from tqdm import trange
import numpy as np
import copy
import cv2
import os
import open_clip
import open3d as o3d
import faiss
from pytorch3d.transforms import quaternion_apply, quaternion_invert

from sugar_scene.gs_model import GaussianSplattingWrapper
from sugar_scene.sugar_model import SuGaR
from sugar_utils.spherical_harmonics import SH2RGB, RGB2SH
from autoencoder.model import Autoencoder
from sugar_utils.general_utils import inverse_sigmoid
from sugar_scene.sugar_optimizer import OptimizationParams, SuGaROptimizer
from sugar_utils.loss_utils import ssim, l1_loss
from rearrange.utils.read_write_model import CameraModel, Camera, Point3D
from rearrange.utils.read_write_model import Image as ImageModel
from gaussian_splatting.arguments import ModelParams, PipelineParams
from gaussian_splatting.arguments import OptimizationParams as optiparams
from argparse import ArgumentParser, Namespace
from gaussian_splatting.train import training as gaussiantrainer
from rearrange.utils.read_write_model import write_images_text, write_cameras_text, write_points3D_text

# save data
# from scan_scene import data

# dinov2
import sys
sys.path.append("dinov2")
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA

# 2d mapping
from navigation.navigation import Navigation
from navigation.utils import geom, aithor
from navigation.utils.arguments import args
from navigation.constants import PICKUPABLE_OBJECTS, OPENABLE_OBJECTS


CONSOLE = Console(width=120)


class GaussianConfig():
    def __init__(self, width, height):
        self.scene_path = "/home/nune/gaussian_splatting/lgsplat-mesh/dataset/FloorPlan303_physics/"
        self.gs_checkpoint_path = "output/"
        self.iteration_to_load = 7000
        self.use_eval_split = False
        self.n_skip_images_for_eval_split = 8
        self.use_white_background = False
        self.coarse_model_path = "output/coarse/FloorPlan303_physics/sugarcoarse_3Dgs7000_sdfestim02_sdfnorm02/15000.pt"
        # self.coarse_model_path = "/home/nune/gaussian_splatting/lgsplat-mesh/output/WorldModel/splat_update/coarse/3000.pt"
        self.include_feature = False
        self.lang_ft_dim = 3
        self.width = width # set in accordance with AI2THOR
        self.height = height  # set in accordance with AI2THOR

        # decoder
        self.decoder_device = "cpu"
        self.autoenc_ckpt = "autoencoder/ckpt/FloorPlan303_physics/best_ckpt.pth"
        self.encoder_dims = [256, 128, 64, 32, 3]
        self.decoder_dims = [16, 32, 64, 128, 256, 256, 512]

        # clip
        self.clip_device = "cuda:0"

        # gaussian splat optimization parameters
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.language_feature_lr = 0.0025
        self.dssim_factor = 0.2
        self.debug = True
        self.num_iterations = 500
        self.start_sdf_at = self.num_iterations//2

        # DINOV2 and CLIP
        self.viz_dense_features = True
        self.save_dino_frames = "output/WorldModel/dense_fts"
        self.clip_device = "cuda:0"
        self.dilate_object_mask = False
        self.dilate_iterations = 2

        # scene configs
        self.height = 500
        self.width = 500
        self.fov = 90
        self.cam_height = 0.675

        # map configs
        self.du_scale = 4
        self.max_depth = 3.5
        self.nav_verbose = False
        self.map_args = args
        self.map_debug = True

class Data():
    def __init__(self, path):
        self.points_ = []
        self.point_colors = []

        self.images = {}
        self.image_id = 0

        self.pcds = {}
        self.point_id = 0

        self.viz_pcd = True
        self.base_path = path
        self.sparse_path = os.path.join(self.base_path, "sparse/0/")
        self.img_path = os.path.join(self.base_path, "images/")

        self.camera_model = {}
        self.trans = None
        self.rot = None
        self.camera_pose = []

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

    def get_data_step(self, fov, rgb, depth, camera_pos, camera_rot):
        if len(self.camera_model) == 0:
            self.height, self.width, _ = rgb.shape
            self.fx = (self.width / 2) / (np.tan(np.deg2rad(fov / 2)))
            self.fy = (self.height / 2) / (np.tan(np.deg2rad(fov / 2)))
            self.cx = (self.width - 1) / 2
            self.cy = (self.height - 1) / 2
            print("-----------CAMERA INTRINSICS-----------")
            print("fx, fy, cx, cy : ", self.fx, self.fy, self.cx, self.cy)
            print("---------------------------------------")

            self.camera_model[0] = Camera(
                id=1,
                model="PINHOLE",
                width=self.width,
                height=self.height,
                params=np.array([self.fx, self.fy, self.cx, self.cy]),
            )
            # save camera data
            write_cameras_text(self.camera_model, os.path.join(self.base_path, "sparse/0/cameras.txt"))

        self.image_id += 1
        image_name = "frame_{}.png".format(self.image_id)

        # image
        rgb_image = rgb
        cv2.imwrite(os.path.join(self.img_path, image_name), cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))

        # position
        position = copy.deepcopy(camera_pos)
        pos_trans = [position['x'], position['y'], position['z']]

        # orientation
        rotation = copy.deepcopy(camera_rot)
        rot_matrix = Rotation.from_euler("zxy", [rotation['z'], rotation['x'], rotation['y']], degrees=True).as_matrix()

        # adjust pose to match coordinate system
        adjusted_pose_ = self.adjust_pose(rot_matrix, pos_trans)

        # taking inverse to save the reverse transformation (COLMAP)
        adjusted_pose = np.linalg.inv(adjusted_pose_)
        rotation_matrix = copy.deepcopy(adjusted_pose[:3, :3])
        new_trans_pose = copy.deepcopy(adjusted_pose[:3, 3])

        # saving for viz
        self.trans = copy.deepcopy(new_trans_pose)
        self.camera_pose.append(new_trans_pose)
        self.rot = copy.deepcopy(rotation_matrix)

        rot_new = Rotation.from_matrix(rotation_matrix)
        quaternion = rot_new.as_quat()
        quaternion_reoriented = [quaternion[3], quaternion[0], quaternion[1], quaternion[2]]  # to match COLMAP dataset

        self.images[self.image_id] = ImageModel(id=self.image_id,
                            qvec=np.array(quaternion_reoriented),
                            tvec=np.array(new_trans_pose),
                            camera_id=1,  # pinhole camera
                            name=image_name,
                            xys=np.array([0, 0]),  # initialized to origin, since it is not used
                            point3D_ids=np.array([0]),)  # initialized to origin, since it is not used

        frame_size = depth.shape[:2]
        x = np.arange(0, frame_size[1])
        y = np.arange(frame_size[0], 0, -1)
        xx, yy = np.meshgrid(x, y)

        xx = xx.flatten()
        yy = yy.flatten()
        zz = depth.flatten()

        depth_mask = zz < 10

        x_camera = (xx - self.cx) * zz / self.fx
        y_camera = -(yy - self.cy) * zz / self.fy
        z_camera = zz
        points_camera = np.stack([x_camera, y_camera, z_camera], axis=1)
        points_camera = points_camera[depth_mask]

        # converting it to world frame
        homogenized_points = np.hstack([points_camera, np.ones((points_camera.shape[0], 1))])
        points_world = adjusted_pose_ @ homogenized_points.T
        points_world = points_world.T[..., :3]
        point_color = rgb_image.reshape(-1, 3)
        point_color = point_color[depth_mask]

        # downsampling the pointcloud (frame downsampling)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_world)
        pcd.colors = o3d.utility.Vector3dVector(point_color)
        down_pcd = pcd.voxel_down_sample(voxel_size=0.02)
        down_points = np.asarray(down_pcd.points)
        down_colors = np.asarray(down_pcd.colors)

        # Global downsampling
        self.points_.extend(down_points.tolist())
        self.point_colors.extend(down_colors.tolist())

        print("points before global downsampling : ", len(self.points_))

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.array(self.points_))
        pcd.colors = o3d.utility.Vector3dVector(np.array(self.point_colors))
        down_pcd = pcd.voxel_down_sample(voxel_size=0.02)
        down_points = np.asarray(down_pcd.points)
        down_colors = np.asarray(down_pcd.colors)

        self.points_ = down_points.tolist()
        self.point_colors = down_colors.tolist()

        print("points after global downsampling : ", len(self.points_))

    def save_data(self):
        print("\n saving data .....")
        # write image data
        write_images_text(self.images, os.path.join(self.sparse_path, "images.txt"))

        print(len(self.points_))
        print(np.array(self.points_))
        point_colors = np.asarray(self.point_colors).astype(np.uint32)
        point_colors = point_colors.tolist()

        print("Running pointcloud visualization...")

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.points_)
        pcd_colors = np.asarray(self.point_colors)/255
        pcd.colors = o3d.utility.Vector3dVector(pcd_colors.tolist())

        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=3,
                                                                 origin=[0, 0, 0])

        # axis_cam = copy.deepcopy(axis).translate((self.trans[0], self.trans[1], self.trans[2]), relative=False)
        # axis_cam.rotate(self.rot)
        # o3d.visualization.draw_geometries([pcd, axis, axis_cam])

        li_ = [[i, i + 1] for i in range(1, len(self.camera_pose))]
        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(self.camera_pose),
            lines=o3d.utility.Vector2iVector(li_),
        )

        viz = o3d.visualization.Visualizer()
        viz.create_window()
        viz.add_geometry(pcd)
        viz.add_geometry(line_set)
        viz.add_geometry(axis)
        opt = viz.get_render_option()
        opt.show_coordinate_frame = True
        viz.run()
        viz.destroy_window()

        # saving the pcd data
        for p in range(len(self.points_)):
            self.point_id += 1
            self.pcds[self.point_id] = Point3D(id=self.point_id,
                                               xyz=np.array(self.points_[p]),
                                               rgb=np.array(point_colors[p]),
                                               error=np.array(0),
                                               image_ids=np.array([0]),
                                               point2D_idxs=np.array([0]))

        write_points3D_text(self.pcds, os.path.join(self.sparse_path, "points3D.txt"))
        print("data saved")


class ObjectScene():
    def __init__(self):
        self.object_name = []
        self.object_pos_map = []
        self.object_pos_world = []
        self.task = []
        self.conf = []
        self.clip_ft = []
        self.img_crops = []
        self.n_false_det = 0
        self.n_diff_det = 0
        self.points = None
        self.bbox = None

    def compute_bbox3d(self):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.points)

        if len(self.points) > 4:
            try:
                return np.asarray(pcd.get_oriented_bounding_box(robust=True).get_box_points())
            except RuntimeError as e:
                print(f"Met {e}, use axis aligned bounding box instead")
                return np.asarray(pcd.get_axis_aligned_bounding_box().get_box_points())
        else:
            return np.asarray(pcd.get_axis_aligned_bounding_box().get_box_points())


"""dense feature matching demo -> https://github.com/antmedellin/dinov2/tree/main"""
class Dinov2Matcher:
    def __init__(self, repo_name="facebookresearch/dinov2", model_name="dinov2_vitb14", smaller_edge_size=448, half_precision=False, device="cuda"):
        self.repo_name = repo_name
        self.model_name = model_name
        self.smaller_edge_size = smaller_edge_size
        self.half_precision = half_precision
        self.device = device

        if self.half_precision:
          self.model = torch.hub.load(repo_or_dir=repo_name, model=model_name).half().to(self.device)
        else:
          self.model = torch.hub.load(repo_or_dir=repo_name, model=model_name).to(self.device)

        self.model.eval()

        self.transform = transforms.Compose([
          transforms.Resize(size=smaller_edge_size, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
          transforms.ToTensor(),
          transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),  # imagenet defaults
        ])

    def prepare_image(self, rgb_image_numpy):
        image = Image.fromarray(rgb_image_numpy)
        image_tensor = self.transform(image)
        resize_scale = image.width / image_tensor.shape[2]

        # Crop image to dimensions that are a multiple of the patch size
        height, width = image_tensor.shape[1:]  # C x H x W
        cropped_width, cropped_height = width - width % self.model.patch_size, height - height % self.model.patch_size  # crop a bit from right and bottom parts
        image_tensor = image_tensor[:, :cropped_height, :cropped_width]

        grid_size = (cropped_height // self.model.patch_size, cropped_width // self.model.patch_size)
        return image_tensor, grid_size, resize_scale


    def prepare_mask(self, mask_image_numpy, grid_size, resize_scale):
        cropped_mask_image_numpy = mask_image_numpy[:int(grid_size[0] * self.model.patch_size * resize_scale),
                                   :int(grid_size[1] * self.model.patch_size * resize_scale)]
        image = Image.fromarray(cropped_mask_image_numpy)
        resized_mask = image.resize((grid_size[1], grid_size[0]), resample=Image.Resampling.NEAREST)
        resized_mask = np.asarray(resized_mask).flatten()
        return resized_mask

    def extract_features(self, image_tensor):
        with torch.inference_mode():
            if self.half_precision:
                image_batch = image_tensor.unsqueeze(0).half().to(self.device)
            else:
                image_batch = image_tensor.unsqueeze(0).to(self.device)

            tokens = self.model.get_intermediate_layers(image_batch)[0].squeeze()
        return tokens.cpu().numpy()

    def idx_to_source_position(self, idx, grid_size, resize_scale):
        row = (idx // grid_size[1]) * self.model.patch_size * resize_scale + self.model.patch_size / 2
        col = (idx % grid_size[1]) * self.model.patch_size * resize_scale + self.model.patch_size / 2
        return row, col

    def get_embedding_visualization(self, tokens, grid_size, resized_mask=None):
        pca = PCA(n_components=3)
        if resized_mask is not None:
            tokens = tokens[resized_mask]
        reduced_tokens = pca.fit_transform(tokens.astype(np.float32))
        if resized_mask is not None:
            tmp_tokens = np.zeros((*resized_mask.shape, 3), dtype=reduced_tokens.dtype)
            tmp_tokens[resized_mask] = reduced_tokens
            reduced_tokens = tmp_tokens
        reduced_tokens = reduced_tokens.reshape((*grid_size, -1))
        normalized_tokens = (reduced_tokens - np.min(reduced_tokens)) / (
                    np.max(reduced_tokens) - np.min(reduced_tokens))
        return normalized_tokens

    def get_combined_embedding_visualization(self, tokens1, token2, grid_size1, grid_size2, mask1=None, mask2=None,
                                             random_state=20):
        pca = PCA(n_components=3, random_state=random_state)

        token1_shape = tokens1.shape[0]
        if mask1 is not None:
            tokens1 = tokens1[mask1]
        if mask2 is not None:
            token2 = token2[mask2]
        combinedtokens = np.concatenate((tokens1, token2), axis=0)
        reduced_tokens = pca.fit_transform(combinedtokens.astype(np.float32))

        if mask1 is not None and mask2 is not None:
            resized_mask = np.concatenate((mask1, mask2), axis=0)
            tmp_tokens = np.zeros((*resized_mask.shape, 3), dtype=reduced_tokens.dtype)
            tmp_tokens[resized_mask] = reduced_tokens
            reduced_tokens = tmp_tokens
        elif mask1 is not None and mask2 is None:
            return sys.exit("Either use both masks or none")
        elif mask1 is None and mask2 is not None:
            return sys.exit("Either use both masks or none")

        normalized_tokens = (reduced_tokens - np.min(reduced_tokens)) / (
                    np.max(reduced_tokens) - np.min(reduced_tokens))

        rgbimg1 = normalized_tokens[0:token1_shape, :]
        rgbimg2 = normalized_tokens[token1_shape:, :]

        rgbimg1 = rgbimg1.reshape((*grid_size1, -1))
        rgbimg2 = rgbimg2.reshape((*grid_size2, -1))
        return rgbimg1, rgbimg2, tokens1, token2


class CLIPFeatureExtractor():
    def __init__(self, device):
        self.device = device
        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
            "ViT-B-16", "laion2b_s34b_b88k"
        )
        self.clip_model = self.clip_model.to(device)
        self.clip_tokenizer = open_clip.get_tokenizer("ViT-B-16")

    def tokenize_text(self, text_prompt):
        """
        text_prompt -> list
        """
        tokenized_text = self.clip_tokenizer(text_prompt).to(self.device)
        text_feat = self.clip_model.encode_text(tokenized_text)
        text_feat /= text_feat.norm(dim=-1, keepdim=True)
        return text_feat

    def tokenize_image(self, image):
        image = Image.fromarray(image)
        preprocessed_image = self.clip_preprocess(image).unsqueeze(0).to(self.device)
        img_feat = self.clip_model.encode_image(preprocessed_image).detach()
        img_feat /= img_feat.norm(dim=-1, keepdim=True)
        return img_feat

class GaussianWorldModel():
    def __init__(self, config):
        self.config = config
        self.timestep = None
        # initializing the nerfmodel
        self.nerfmodel = GaussianSplattingWrapper(source_path=config.scene_path,
                                                  output_path=config.gs_checkpoint_path,
                                                  iteration_to_load=config.iteration_to_load,
                                                  load_gt_images=True,
                                                  eval_split=config.use_eval_split,
                                                  eval_split_interval=config.n_skip_images_for_eval_split,
                                                  white_background=config.use_white_background,)
        CONSOLE.print(f'{len(self.nerfmodel.training_cameras)} training images detected.')


        # initializing the sugar model
        coarse_model_path = config.coarse_model_path
        coarse_ckpt = torch.load(coarse_model_path, map_location="cpu")
        self.sugar = SuGaR( nerfmodel=self.nerfmodel,
                            points=coarse_ckpt['state_dict']['_points'].detach().float().cuda(),
                            colors=SH2RGB(coarse_ckpt['state_dict']['_sh_coordinates_dc'][:, 0, :]).detach().float().cuda(),
                            initialize=False,
                            sh_levels=4,
                            triangle_scale=1,
                            learnable_positions=True,
                            keep_track_of_knn=False,
                            knn_to_track=0,
                            freeze_gaussians=False,
                            beta_mode=None,
                            surface_mesh_to_bind=None,
                            surface_mesh_thickness=None,
                            learn_surface_mesh_positions=False,
                            learn_surface_mesh_opacity=False,
                            learn_surface_mesh_scales=False,
                            n_gaussians_per_surface_triangle=1,
                            include_feature=config.include_feature
                        )

        if self.config.debug:
            CONSOLE.print("---------------------------------------------------------------------")
            CONSOLE.print(f'Number of parameters: {sum(p.numel() for p in self.sugar.parameters())}')
            CONSOLE.print("\nModel parameters:")
            for name, param in self.sugar.named_parameters():
                CONSOLE.print(name, param.shape, param.requires_grad)
            CONSOLE.print("---------------------------------------------------------------------")

        if config.include_feature:
            self.sugar.init_lang_ft(config.lang_ft_dim)
        self.sugar.load_state_dict(coarse_ckpt['state_dict'])

        for name, param in self.sugar.named_parameters():
            CONSOLE.print(name, param.shape, param.requires_grad)

        # setting the grads to true (to enable online update of parameters)
        self.sugar.set_grads()

        # decoder for language features
        if config.include_feature:
            autoenc_ckpt_ = config.autoenc_ckpt
            autoenc_ckpt = torch.load(autoenc_ckpt_)
            autoenc_model = Autoencoder(config.encoder_dims, config.decoder_dims).to("cuda:0")
            autoenc_model.load_state_dict(autoenc_ckpt)
            autoenc_model.eval()
            self.autoenc_model = autoenc_model.to(config.decoder_device)
            self.decoded_lang_fts = self.autoenc_model.decode(
                self.sugar.language_feature.clone().detach().to(self.config.decoder_device))
            self.decoded_lang_fts = self.decoded_lang_fts.detach()

        self.radii = None
        self.spatial_mask_gaussians = None
        self.semantic_mask_gaussians = None
        self.gaussian_mask = None
        self.object_bbox_hist = []
        self.scene_info = {}

        # saving info for COLMAP loader
        self.images= []
        self.cam_extrinsics = {}
        self.xyzs = []
        self.rgbs = []
        self.scene_data = {}
        self.camera_model = {}
        self.cam_t = []
        self.images_train = []
        self.image_id = 0
        self.pcds = {}
        self.point_id = 0

    def viz_world_model_images(self):
        for i in trange(73):
            print(self.nerfmodel.get_image_name(camera_indices=i))
            outputs = self.sugar.render_image_gaussian_rasterizer(
                camera_indices=i,
                verbose=False,
                bg_color=None,
                sh_deg=3,
                sh_rotations=None,
                compute_color_in_rasterizer=False,
                compute_covariance_in_rasterizer=True,
                return_2d_radii=True,
                quaternions=None,
                use_same_scale_in_all_directions=False,
                return_opacities=False,
            )
            pred_rgb = outputs['image']
            radii = outputs['radii']
            language_feature = outputs["language_feature_image"]
            torchvision.utils.save_image(pred_rgb.permute(2, 0, 1), 'output/test/{}.png'.format(i))

    def load_virtual_camera(self, cam_pose, time_step):
        cam_center = copy.deepcopy(cam_pose[:3, 3])
        outputs = self.sugar.render_image_for_AI2THOR(
            nerf_cameras=None,
            c2w=cam_pose,
            camera_center=cam_center,
            verbose=False,
            bg_color=None,
            sh_deg=3,
            sh_rotations=None,
            compute_color_in_rasterizer=False,
            compute_covariance_in_rasterizer=True,
            return_2d_radii=True,
            quaternions=None,
            use_same_scale_in_all_directions=False,
            return_opacities=False,
        )
        if self.config.include_feature:
            language_feature = outputs["language_feature_image"]
            torchvision.utils.save_image(language_feature, 'output/WorldModel/lang_field/{:03d}.png'.format(time_step))
        pred_rgb = outputs['image']
        torchvision.utils.save_image(pred_rgb.permute(2, 0, 1), 'output/WorldModel/rendered/{:03d}.png'.format(time_step))
        self.radii = outputs['radii']
        return outputs

    def semantic_selection_gaussians(self, object_name, lang_ft_enc):
        lang_ft = self.autoenc_model.decode(lang_ft_enc.to(self.config.decoder_device))
        gaussian_visibility_mask = self.radii > 0
        masked_gaussians = self.decoded_lang_fts[gaussian_visibility_mask]
        cosine_sim = torch.nn.functional.cosine_similarity(masked_gaussians.to(lang_ft_enc.device), lang_ft)
        max_value = cosine_sim.max()
        min_value = cosine_sim.min()
        normalized_similarities = (cosine_sim - min_value) / (max_value - min_value)
        cosine_sim_mask = normalized_similarities > 0.75

        print(torch.sum(cosine_sim_mask))

        semantic_mask_gaussians = torch.zeros(self.decoded_lang_fts.shape[0], dtype=torch.bool)
        semantic_mask_gaussians[gaussian_visibility_mask.cpu()] = cosine_sim_mask.cpu()
        return semantic_mask_gaussians

    def semantic_selection_gaussians_in_spatial_mask(self, mask, lang_ft):
        #lang_ft = self.autoenc_model.decode(lang_ft_enc.to(self.config.decoder_device))
        gaussians_spatial = self.decoded_lang_fts[mask.to(self.config.decoder_device)]
        cosine_sim = torch.nn.functional.cosine_similarity(gaussians_spatial.to(lang_ft.device), lang_ft)
        # max_value = cosine_sim.max()
        # min_value = cosine_sim.min()
        # normalized_similarities = (cosine_sim - min_value) / (max_value - min_value)
        cosine_sim_mask = cosine_sim > 0.8
        semantic_mask_gaussians = torch.zeros(self.decoded_lang_fts.shape[0], dtype=torch.bool)
        semantic_mask_gaussians[mask.cpu()] = cosine_sim_mask.cpu()
        return semantic_mask_gaussians

    def spatial_selection_gaussians(self, mask, cam2World, depth):

        height, width = depth.shape
        x = np.arange(0, width)
        y = np.arange(height, 0, -1)
        xx, yy = np.meshgrid(x, y)

        xx = xx.flatten()
        yy = yy.flatten()
        zz = depth.flatten()
        x_camera = (xx - width / 2) * zz / self.sugar.fx
        y_camera = -(yy - height / 2) * zz / self.sugar.fy
        z_camera = zz
        points_camera = np.stack([x_camera, y_camera, z_camera], axis=1)
        points_camera = points_camera.reshape((height, width, 3))
        masked_points = points_camera[mask]
        masked_points = masked_points.reshape(-1, 3)

        homogenized_points = np.hstack([masked_points, np.ones((masked_points.shape[0], 1))])
        points_world = cam2World @ torch.tensor(homogenized_points, device=cam2World.device, dtype=torch.float).T
        points_world = points_world.T[..., :3]

        delta = 0.07 # maintain margins
        bbox_3d = {'min': torch.min(points_world, dim=0)[0] - torch.tensor([delta, delta, delta], device=points_world.device),
                   'max': torch.max(points_world, dim=0)[0] + torch.tensor([delta, delta, delta], device=points_world.device)}
        self.object_bbox_hist.append(bbox_3d)

        # for viz TODO: ADD viz
        corners = [
            [bbox_3d['min'][0], bbox_3d['min'][1], bbox_3d['min'][2]],
            [bbox_3d['max'][0], bbox_3d['min'][1], bbox_3d['min'][2]],
            [bbox_3d['max'][0], bbox_3d['max'][1], bbox_3d['min'][2]],
            [bbox_3d['min'][0], bbox_3d['max'][1], bbox_3d['min'][2]],
            [bbox_3d['min'][0], bbox_3d['min'][1], bbox_3d['max'][2]],
            [bbox_3d['max'][0], bbox_3d['min'][1], bbox_3d['max'][2]],
            [bbox_3d['max'][0], bbox_3d['max'][1], bbox_3d['max'][2]],
            [bbox_3d['min'][0], bbox_3d['max'][1], bbox_3d['max'][2]],
        ]
        return bbox_3d

    def update_all_param(self, gauss_dict):
        print("mem2 : ", torch.cuda.memory_allocated(0) / 1024 / 1024 / 1024)
        points = self.sugar._points.clone().detach()
        sh_coords_dc = self.sugar._sh_coordinates_dc.clone().detach()
        sh_coords_rest = self.sugar._sh_coordinates_rest.clone().detach()
        scales = self.sugar._scales.clone().detach()
        quaternions = self.sugar._quaternions.clone().detach()
        all_densities = self.sugar.all_densities.clone().detach()
        lang_fts = self.sugar._language_feature.clone().detach()

        points_len = points.shape[0]

        n_points = points.shape[0] + gauss_dict["points"].shape[0]
        new_points = torch.zeros((n_points, points.shape[1]), device=points.device, dtype=torch.float)
        new_quaternions = torch.zeros((n_points, quaternions.shape[1]), device=quaternions.device, dtype=torch.float)
        new_sh_coords_dc = torch.zeros((n_points, sh_coords_dc.shape[1], sh_coords_dc.shape[2]),
                                       device=sh_coords_dc.device, dtype=torch.float)
        new_sh_coords_rest = torch.zeros((n_points, sh_coords_rest.shape[1], sh_coords_rest.shape[2]),
                                       device=sh_coords_rest.device, dtype=torch.float)
        new_scales = torch.zeros((n_points, scales.shape[1]), device=scales.device, dtype=torch.float)
        new_all_densities = torch.zeros((n_points, all_densities.shape[1]), device=all_densities.device, dtype=torch.float)
        new_lang_fts = torch.zeros((n_points, lang_fts.shape[1]), device=points.device, dtype=torch.float)

        new_points[:points.shape[0], :] = points
        new_quaternions[:quaternions.shape[0], :] = quaternions
        new_sh_coords_dc[:sh_coords_dc.shape[0], :, :] = sh_coords_dc
        new_sh_coords_rest[:sh_coords_rest.shape[0], :, :] = sh_coords_rest
        new_scales[:scales.shape[0], :] = scales
        new_all_densities[:all_densities.shape[0], :] = all_densities
        new_lang_fts[:lang_fts.shape[0], :] = lang_fts

        new_points[points.shape[0]:, :] = gauss_dict["points"]
        new_quaternions[quaternions.shape[0]:, :] = gauss_dict["quaternions"]
        new_sh_coords_dc[sh_coords_dc.shape[0]:, :, :] = gauss_dict["sh_coordinates_dc"]
        new_sh_coords_rest[sh_coords_rest.shape[0]:, :, :] = gauss_dict["sh_coordinates_rest"]
        new_scales[scales.shape[0]:, :] = gauss_dict["scales"]
        new_all_densities[all_densities.shape[0]:, :] = gauss_dict["densities"]
        new_lang_fts[lang_fts.shape[0]:, :] = torch.zeros_like(gauss_dict["points"], device=new_points.device, dtype=torch.float)

        # viz
        colors = torch.zeros_like(new_points)
        colors[points.shape[0]:] = torch.tensor([1, 0, 0])
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(new_points.cpu().numpy().tolist())
        pcd_colors = np.asarray(colors.cpu().numpy())
        pcd.colors = o3d.utility.Vector3dVector(pcd_colors.tolist())
        viz = o3d.visualization.Visualizer()
        viz.create_window()
        viz.add_geometry(pcd)
        opt = viz.get_render_option()
        opt.show_coordinate_frame = True
        viz.run()
        viz.destroy_window()

        new_lang_fts = torch.zeros_like(new_points, device=points.device, dtype=torch.float)

        gaussian_dict = {"points": new_points,
                         "sh_coordinates_dc": new_sh_coords_dc,
                         "sh_coordinates_rest": new_sh_coords_rest,
                         "scales": new_scales,
                         "quaternions": new_quaternions,
                         "densities": new_all_densities,
                         "language_features": new_lang_fts}

        self.sugar.update_all_gaussians(gaussian_dict)

       # return gaussian_dict, points_len

    def update_gaussians_continuous(self, object_name,
                               trans_pose,
                               mask_2d,
                               lang_img,
                               Cam2World,
                               depth,
                               new_pick=False,
                               body_center = None):

        lang_img = lang_img.permute(1, 2, 0)
        lang_fts = lang_img[mask_2d]
        lang_fts = lang_fts.reshape(-1, 3)
        decoded_lang_fts = self.autoenc_model.decode(lang_fts.to(self.config.decoder_device))
        decoded_lang_fts = decoded_lang_fts.mean(dim = 0)

        centers = self.sugar.points.clone().detach()
        quaternions_all = self.sugar.quaternions.clone().detach().cpu().numpy()

        # spatial
        if new_pick:
            bbox_3d = self.spatial_selection_gaussians(mask_2d, Cam2World, depth)
            within_min = centers >= bbox_3d['min']
            within_max = centers <= bbox_3d['max']
            spatial_mask_gaussians = within_max & within_min
            spatial_mask_gaussians = torch.all(spatial_mask_gaussians, dim=-1)
            spatial_mask_gaussians = spatial_mask_gaussians.to(centers.device)
            self.spatial_mask_gaussians = spatial_mask_gaussians

            self.semantic_mask_gaussians = self.semantic_selection_gaussians_in_spatial_mask(spatial_mask_gaussians, decoded_lang_fts)

            self.gaussian_mask = self.semantic_mask_gaussians.to(self.spatial_mask_gaussians.device) & self.spatial_mask_gaussians

        quaternions = copy.deepcopy(quaternions_all[self.gaussian_mask.cpu().numpy()])
        if quaternions.shape[0] == 0:
            raise ValueError("No gaussians in the given region")
        new_center_ = copy.deepcopy(centers[self.gaussian_mask])
        print("selected gauss : ", quaternions.shape)

        rots = Rotation.from_quat(quaternions).as_matrix()
        tfm_ellipsoids = np.zeros((rots.shape[0], 4, 4))
        tfm_ellipsoids[:, :3, :3] = rots
        tfm_ellipsoids[:, :3, 3] = new_center_.cpu().numpy()
        tfm_ellipsoids[:, 3, 3] = 1

        tfm_ellipsoid_new = trans_pose @ tfm_ellipsoids
        new_rot_matrix = tfm_ellipsoid_new[:, :3, :3]
        masked_new_quats = Rotation.from_matrix(new_rot_matrix).as_quat()
        masked_new_center = tfm_ellipsoid_new[:, :3, 3]

        centers = centers.cpu().numpy()
        centers[self.gaussian_mask.cpu().numpy()] = masked_new_center
        quaternions_all[self.gaussian_mask.cpu().numpy()] = masked_new_quats

        self.sugar.move_gaussians(torch.tensor(centers).to(self.nerfmodel.device), torch.tensor(quaternions_all).to(self.nerfmodel.device))

    def get_gaussian_info(self):
        points = self.sugar.points.clone().detach()
        sh_coords_dc = self.sugar.sh_coordinates_dc.clone().detach()
        sh_coords_rest = self.sugar.sh_coordinates_rest.clone().detach()
        scales = self.sugar.scaling.clone().detach()
        quaternions = self.sugar.quaternions.clone().detach()
        all_densities = self.sugar.densities_all.clone().detach()
        lang_fts = self.sugar.language_feature.clone().detach()

        gaussian_info = {
            "points": points.to(self.nerfmodel.device),
            "densities": all_densities.to(self.nerfmodel.device),
            "sh_coordinates_dc": sh_coords_dc.to(self.nerfmodel.device),
            "sh_coordinates_rest": sh_coords_rest.to(self.nerfmodel.device),
            "scales": scales.to(self.nerfmodel.device),
            "quaternions": quaternions.to(self.nerfmodel.device),
            "language_features": lang_fts.to(self.nerfmodel.device),
        }

        return gaussian_info

    def remove_stray_gaussians(self, bbox_3d):
        centers = self.sugar.points.clone().detach()
        within_min = centers >= bbox_3d['min']
        within_max = centers <= bbox_3d['max']
        gaussian_mask = within_max & within_min
        gaussian_mask = torch.all(gaussian_mask, dim=-1)

        gaussian_info = self.get_gaussian_info()

        new_points = gaussian_info["points"][~gaussian_mask]
        new_densities = gaussian_info["densities"][~gaussian_mask]
        new_sh_coords_dc = gaussian_info["sh_coordinates_dc"][~gaussian_mask]
        new_sh_coords_rest = gaussian_info["sh_coordinates_rest"][~gaussian_mask]
        new_scales = gaussian_info["scales"][~gaussian_mask]
        new_quaternions = gaussian_info["quaternions"][~gaussian_mask]
        new_language_features = gaussian_info["language_features"][~gaussian_mask]

        new_gaussians = {
            "points": new_points.type(torch.float).to(self.nerfmodel.device),
            "densities": new_densities.type(torch.float).to(self.nerfmodel.device),
            "sh_coordinates_dc": new_sh_coords_dc.type(torch.float).to(self.nerfmodel.device),
            "sh_coordinates_rest": new_sh_coords_rest.type(torch.float).to(self.nerfmodel.device),
            "scales": new_scales.type(torch.float).to(self.nerfmodel.device),
            "quaternions": new_quaternions.type(torch.float).to(self.nerfmodel.device),
            "language_features": new_language_features.type(torch.float).to(self.nerfmodel.device),
        }
        self.sugar.update_gaussians(new_gaussians)

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

    def save_scene_info_to_update_splat(self, evt, c2w, mask, new_gaussians, bbox_3d):
        self.image_id += 1
        masked = copy.deepcopy(evt.frame)/255
        masked[~mask] = [0, 0, 0]
        self.scene_info[0] = {"c2w": c2w,
                             "rgb_image": masked,
                             "depth": evt.depth_frame,
                             "bbox_3d": bbox_3d,
                             "mask": mask,
                             "gaussian": new_gaussians}

        self.sparse_path = "output/WorldModel/data/sparse/0/"

        self.cam_t.append(c2w)
        self.images_train.append(masked)
        masked = (masked*255).astype(np.uint8)
        cv2.imwrite("output/WorldModel/mask.png", cv2.cvtColor(masked, cv2.COLOR_RGB2BGR))

        cv2.imwrite("output/WorldModel/data/images/{}.png".format(self.image_id), cv2.cvtColor(masked, cv2.COLOR_RGB2BGR))
        masked_read = Image.open("output/WorldModel/data/images/{}.png".format(self.image_id))
        self.images.append(masked_read)

        position = copy.deepcopy(evt.metadata['cameraPosition'])
        pos_trans = [position['x'], position['y'], position['z']]

        # orientation
        rotation = copy.deepcopy(evt.metadata['agent']['rotation'])
        rot_matrix = Rotation.from_euler("zxy", [rotation['z'], rotation['x'], rotation['y']], degrees=True).as_matrix()

        # adjust pose to match coordinate system
        adjusted_pose_ = self.adjust_pose(rot_matrix, pos_trans)

        # taking inverse to save the reverse transformation (COLMAP)
        adjusted_pose = np.linalg.inv(adjusted_pose_)
        rotation_matrix = copy.deepcopy(adjusted_pose[:3, :3])
        new_trans_pose = copy.deepcopy(adjusted_pose[:3, 3])

        rot_new = Rotation.from_matrix(rotation_matrix)
        quaternion = rot_new.as_quat()
        quaternion_reoriented = [quaternion[3], quaternion[0], quaternion[1], quaternion[2]]  # to match COLMAP dataset

        self.cam_extrinsics[self.image_id] = ImageModel(id=self.image_id,
                                                       qvec=np.array(quaternion_reoriented),
                                                       tvec=np.array(new_trans_pose),
                                                       camera_id=1,
                                                       name="{}.png".format(self.image_id),
                                                       xys=np.array([0, 0]),
                                                       point3D_ids=np.array([0]),)


        # TODO: all_points key in dictionary to update multiple objects parallely

        self.xyzs.extend(new_gaussians["points"].cpu().numpy().tolist())
        self.rgbs.extend(new_gaussians["colors"].cpu().numpy().astype(np.uint8).tolist())

        if len(self.camera_model) == 0:
            fov = evt.metadata['fov']
            height, width, _ = evt.frame.shape
            fx = (width / 2) / (np.tan(np.deg2rad(fov / 2)))
            fy = (height / 2) / (np.tan(np.deg2rad(fov / 2)))
            cx = (width - 1) / 2
            cy = (height - 1) / 2

            self.camera_model[1] = Camera(
                id=1,
                model="PINHOLE",
                width=width,
                height=height,
                params=np.array([fx, fy, cx, cy]),
            )

            write_cameras_text(self.camera_model, os.path.join(self.sparse_path, "cameras.txt"))

    def save_data(self):
        print("saving data ...")
        write_images_text(self.cam_extrinsics, os.path.join(self.sparse_path, "images.txt"))

        print("number of points : ", len(self.xyzs))

        for p in range(len(self.xyzs)):
            self.point_id += 1
            self.pcds[self.point_id] = Point3D(id=self.point_id,
                                               xyz=np.array(self.xyzs[p]),
                                               rgb=np.array(self.rgbs[p]),
                                               error=np.array(0),
                                               image_ids=np.array([0]),
                                               point2D_idxs=np.array([0]))

        write_points3D_text(self.pcds, os.path.join(self.sparse_path, "points3D.txt"))
        print("data saved")

    def gaussian_splat_train(self):
        parser = ArgumentParser(description="Training script parameters")
        lp = ModelParams(parser)
        op = optiparams(parser)
        pp = PipelineParams(parser)
        parser.add_argument('--ip', type=str, default="127.0.0.1")
        parser.add_argument('--port', type=int, default=6009)
        parser.add_argument('--debug_from', type=int, default=-1)
        parser.add_argument('--detect_anomaly', action='store_true', default=False)
        parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
        parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
        parser.add_argument("--quiet", action="store_true")
        parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
        parser.add_argument("--start_checkpoint", type=str, default=None)
        args = parser.parse_args(sys.argv[1:])
        args.save_iterations.append(args.iterations)

        string_args = f"""
                --iterations {3000} -m {"output/WorldModel/splat_update"} -s {"/home/nune/gaussian_splatting/gaussian-splatting/dataset/Ai2Thor/FloorPlan20_physics"}
                """
        string_args = string_args.split()
        args = parser.parse_args(string_args)

        xyzs = np.empty((len(self.xyzs), 3))
        rgbs = np.empty((len(self.xyzs), 3))
        for i in range(len(self.xyzs)):
            xyz = np.array(tuple(map(float, self.xyzs[i])))
            rgb = np.array(tuple(map(int, self.rgbs[i])))
            xyzs[i] = xyz
            xyzs[i] = xyz
            rgbs[i] = rgb

        new_gaussians = gaussiantrainer(lp.extract(args),
                                        op.extract(args),
                                        pp.extract(args),
                                        args.test_iterations,
                                        args.save_iterations,
                                        args.checkpoint_iterations,
                                        args.start_checkpoint,
                                        args.debug_from,
                                        from_sim=True,
                                        cam_intrinsics=self.camera_model,
                                        cam_extrinsics=self.cam_extrinsics,
                                        images=self.images,
                                        xyz=xyzs,
                                        rgb=rgbs)

        lang_fts = torch.zeros_like(new_gaussians["points"])
        new_gaussians["language_features"] = lang_fts.to(self.nerfmodel.device).to(torch.float)
        gauss_trained = self.train_surface_aligned_gaussians(new_gaussians)
        self.update_all_param(gauss_trained)
        # self.update_splat(gauss_data, num_pts)

    def train_surface_aligned_gaussians(self, gauss_dict = None):
        if gauss_dict is None:
            gauss_dict = self.scene_info[0]["gaussian"]

        CONSOLE.print(f'{len(self.nerfmodel.training_cameras)} training images detected.')
        with torch.no_grad():
            print("Initializing model from trained 3DGS...")
            with torch.no_grad():
                from sugar_utils.spherical_harmonics import SH2RGB
                points = gauss_dict["points"]

        # initializing the sugar model
        self.sugar_new = SuGaR(nerfmodel=self.nerfmodel,
                               points=points,
                               colors=SH2RGB(gauss_dict['colors'][:, 0].detach().float().cuda()),
                               initialize=True,
                               sh_levels=4,
                               triangle_scale=1,
                               learnable_positions=True,
                               keep_track_of_knn=True,
                               knn_to_track=16,
                               freeze_gaussians=False,
                               beta_mode=None,
                               surface_mesh_to_bind=None,
                               surface_mesh_thickness=None,
                               learn_surface_mesh_positions=False,
                               learn_surface_mesh_opacity=False,
                               learn_surface_mesh_scales=False,
                               n_gaussians_per_surface_triangle=1,
                               include_feature=False
                               )
        if "scales" in gauss_dict:
            print(gauss_dict["sh_coordinates_dc"].shape)
            print(self.sugar_new._sh_coordinates_dc[...].shape)
            with torch.no_grad():
                self.sugar_new._scales[...] = gauss_dict["scales"]
                self.sugar_new._quaternions[...] = gauss_dict["quaternions"]
                self.sugar_new.all_densities[...] = gauss_dict["densities"]
                self.sugar_new._sh_coordinates_dc[...] = gauss_dict["sh_coordinates_dc"]
                self.sugar_new._sh_coordinates_rest[...] = gauss_dict["sh_coordinates_rest"]

        def loss_fn(pred_rgb, gt_rgb):
            return (1.0 - self.config.dssim_factor) * l1_loss(pred_rgb, gt_rgb) + self.config.dssim_factor * (1.0 - ssim(pred_rgb, gt_rgb))

        # TODO: Include densification and pruning
        opt_params = OptimizationParams(
            iterations=self.config.num_iterations,
            position_lr_init=self.config.position_lr_init,
            position_lr_final=self.config.position_lr_final,
            position_lr_delay_mult=self.config.position_lr_delay_mult,
            position_lr_max_steps=self.config.position_lr_max_steps,
            feature_lr=self.config.feature_lr,
            opacity_lr=self.config.opacity_lr,
            scaling_lr=self.config.scaling_lr,
            rotation_lr=self.config.rotation_lr,
            language_feature_lr=self.config.language_feature_lr
        )
        spatial_lr_scale = self.sugar_new.get_cameras_spatial_extent()
        optimizer = SuGaROptimizer(self.sugar_new, opt_params, spatial_lr_scale=spatial_lr_scale)
        self.sugar_new.train()

        # TODO: train semantics
        # self.sugar_new.reset_grads_lang()
        if self.config.debug:
            CONSOLE.print("Optimizer initialized.")
            CONSOLE.print("Optimization parameters:")
            CONSOLE.print(opt_params)
            CONSOLE.print("---------------------------------------------------------------------")
            CONSOLE.print(f'Number of parameters: {sum(p.numel() for p in self.sugar_new.parameters() if p.requires_grad)}')
            CONSOLE.print("\nModel parameters:")
            for name, param in self.sugar_new.named_parameters():
                CONSOLE.print(name, param.shape, param.requires_grad)
            CONSOLE.print("---------------------------------------------------------------------")

        iteration = 0

        start_entropy_regularization_from = 300
        end_entropy_regularization_at = 900
        entropy_regularization_factor = 0.1
        regularize = True
        regularize_from = 300
        start_reset_neighbors_from = 301
        regularity_samples = -1
        regularize_knn = 16
        regularize_sdf = True
        reset_neighbors_every = 150
        start_sdf_regularization_from = 900
        use_sdf_estimation_loss = True
        enforce_samples_to_be_on_surface = True
        start_sdf_estimation_from = 900
        backpropagate_gradients_through_depth = True
        sample_only_in_gaussians_close_to_surface = True
        close_gaussian_threshold = 2
        n_samples_for_sdf_regularization = 1_000_000
        sdf_sampling_scale_factor = 1.5
        sdf_sampling_proportional_to_volume = False
        use_sdf_better_normal_loss = True
        density_threshold = 1
        density_factor = 1/16
        sdf_estimation_mode = "sdf"
        start_sdf_estimation_from = 900
        start_sdf_better_normal_from = 900

        while iteration < self.config.num_iterations:
            for i in range(len(self.images_train)):
                iteration += 1
                gt_image = torch.tensor(self.images_train[i].copy(), dtype=torch.float).to(self.nerfmodel.device)
                gt_rgb = gt_image.view(-1, self.sugar_new.image_height, self.sugar_new.image_width, 3)
                cam_pose = self.cam_t[i]
                cam_center = copy.deepcopy(cam_pose[:3, 3])

                outputs = self.sugar_new.render_image_for_AI2THOR(
                    nerf_cameras=None,
                    c2w=cam_pose,
                    camera_center=cam_center,
                    verbose=False,
                    bg_color=None,
                    sh_deg=3,
                    sh_rotations=None,
                    compute_color_in_rasterizer=False,
                    compute_covariance_in_rasterizer=True,
                    return_2d_radii=True,
                    quaternions=None,
                    use_same_scale_in_all_directions=False,
                    return_opacities=False,
                )

                img_save = outputs["image"].detach().clone()
                torchvision.utils.save_image(img_save.permute(2, 0, 1),
                                             'output/WorldModel/update_new/{:04d}.png'.format(iteration))
                pred_rgb = outputs['image'].view(-1, self.sugar_new.image_height, self.sugar_new.image_width, 3)
                radii = outputs['radii']

                pred_rgb = pred_rgb.transpose(-1, -2).transpose(-2, -3)
                gt_rgb = gt_rgb.transpose(-1, -2).transpose(-2, -3)

                # TODO: Add a semantic loss to optimize the language features in parallel
                loss = loss_fn(pred_rgb, gt_rgb)

                # if iteration > start_entropy_regularization_from and iteration < end_entropy_regularization_at:
                #     if iteration == start_entropy_regularization_from + 1:
                #         CONSOLE.print("\n---INFO---\nStarting entropy regularization.")
                #     if iteration == end_entropy_regularization_at - 1:
                #         CONSOLE.print("\n---INFO---\nStopping entropy regularization.")
                #     visibility_filter = radii > 0
                #     if visibility_filter is not None:
                #         vis_opacities = opacities[visibility_filter]
                #     else:
                #         vis_opacities = opacities
                #     loss = loss + entropy_regularization_factor * (
                #             - vis_opacities * torch.log(vis_opacities + 1e-10)
                #             - (1 - vis_opacities) * torch.log(1 - vis_opacities + 1e-10)
                #     ).mean()
                #
                # if regularize:
                #     if iteration == regularize_from:
                #         CONSOLE.print("Starting regularization...")
                #         # sugar.reset_neighbors()
                #     if iteration > regularize_from:
                #         visibility_filter = radii > 0
                #         if (iteration >= start_reset_neighbors_from) and (
                #                 (iteration == regularize_from + 1) or (iteration % reset_neighbors_every == 0)):
                #             CONSOLE.print("\n---INFO---\nResetting neighbors...")
                #             self.sugar_new.reset_neighbors()
                #         neighbor_idx = self.sugar_new.get_neighbors_of_random_points(num_samples=regularity_samples, )
                #         if visibility_filter is not None:
                #             neighbor_idx = neighbor_idx[visibility_filter]  # TODO: Error here
                #
                #         if regularize_sdf and iteration > start_sdf_regularization_from:
                #             if iteration == start_sdf_regularization_from + 1:
                #                 CONSOLE.print("\n---INFO---\nStarting SDF regularization.")
                #
                #             sampling_mask = visibility_filter
                #
                #             if (use_sdf_estimation_loss or enforce_samples_to_be_on_surface) and iteration > start_sdf_estimation_from:
                #                 if iteration == start_sdf_estimation_from + 1:
                #                     CONSOLE.print("\n---INFO---\nStarting SDF estimation loss.")
                #                 fov_camera = self.nerfmodel.training_cameras.p3d_cameras[0]
                #
                #                 # Render a depth map using gaussian splatting
                #                 if backpropagate_gradients_through_depth:
                #                     point_depth = fov_camera.get_world_to_view_transform().transform_points(
                #                         self.sugar_new.points)[..., 2:].expand(-1, 3)
                #                     max_depth = point_depth.max()
                #                     depth = self.sugar_new.render_image_for_AI2THOR(
                #                         bg_color=max_depth + torch.zeros(3, dtype=torch.float, device=self.sugar_new.device),
                #                         sh_deg=0,
                #                         compute_color_in_rasterizer=False,  # compute_color_in_rasterizer,
                #                         compute_covariance_in_rasterizer=True,
                #                         return_2d_radii=False,
                #                         use_same_scale_in_all_directions=False,
                #                         point_colors=point_depth,
                #                         nerf_cameras=None,
                #                         c2w=cam_pose,
                #                         camera_center=cam_center,
                #                         verbose=False,
                #                         sh_rotations=None,
                #                         quaternions=None,
                #                         return_opacities=False,
                #                     )[..., 0]
                #                 else:
                #                     with torch.no_grad():
                #                         point_depth = fov_camera.get_world_to_view_transform().transform_points(
                #                             self.sugar_new.points)[..., 2:].expand(-1, 3)
                #                         max_depth = point_depth.max()
                #                         depth = self.sugar_new.render_image_for_AI2THOR(
                #                             bg_color=max_depth + torch.zeros(3, dtype=torch.float, device=self.sugar_new.device),
                #                             sh_deg=0,
                #                             compute_color_in_rasterizer=False,  # compute_color_in_rasterizer,
                #                             compute_covariance_in_rasterizer=True,
                #                             return_2d_radii=False,
                #                             use_same_scale_in_all_directions=False,
                #                             point_colors=point_depth,
                #                             nerf_cameras=None,
                #                             c2w=cam_pose,
                #                             camera_center=cam_center,
                #                             verbose=False,
                #                             sh_rotations=None,
                #                             quaternions=None,
                #                             return_opacities=False,
                #                         )[..., 0]
                #
                #                 if sample_only_in_gaussians_close_to_surface:
                #                     with torch.no_grad():
                #                         gaussian_to_camera = torch.nn.functional.normalize(
                #                             fov_camera.get_camera_center() - self.sugar_new.points, dim=-1)
                #                         gaussian_centers_in_camera_space = fov_camera.get_world_to_view_transform().transform_points(
                #                             self.sugar_new.points)
                #
                #                         gaussian_centers_z = gaussian_centers_in_camera_space[..., 2] + 0.
                #                         gaussian_centers_map_z = self.sugar_new.get_points_depth_in_depth_map(fov_camera, depth,
                #                                                                                      gaussian_centers_in_camera_space)
                #
                #                         gaussian_standard_deviations = (self.sugar_new.scaling * quaternion_apply(quaternion_invert(self.sugar_new.quaternions), gaussian_to_camera)).norm(dim=-1)
                #
                #                         gaussians_close_to_surface = (gaussian_centers_map_z - gaussian_centers_z).abs() < close_gaussian_threshold * gaussian_standard_deviations
                #                         sampling_mask = sampling_mask * gaussians_close_to_surface
                #
                #             n_gaussians_in_sampling = sampling_mask.sum()
                #             if n_gaussians_in_sampling > 0:
                #                 sdf_samples, sdf_gaussian_idx = self.sugar_new.sample_points_in_gaussians(
                #                     num_samples=n_samples_for_sdf_regularization,
                #                     sampling_scale_factor=sdf_sampling_scale_factor,
                #                     mask=sampling_mask,
                #                     probabilities_proportional_to_volume=sdf_sampling_proportional_to_volume,
                #                 )
                #
                #                 if use_sdf_estimation_loss or use_sdf_better_normal_loss:
                #                     fields = self.sugar_new.get_field_values(
                #                         sdf_samples, sdf_gaussian_idx,
                #                         return_sdf=(use_sdf_estimation_loss or enforce_samples_to_be_on_surface) and (
                #                                     sdf_estimation_mode == 'sdf') and iteration > start_sdf_estimation_from,
                #                         density_threshold=density_threshold, density_factor=density_factor,
                #                         return_sdf_grad=False, sdf_grad_max_value=10.,
                #                         return_closest_gaussian_opacities=use_sdf_better_normal_loss and iteration > start_sdf_better_normal_from,
                #                         return_beta=(use_sdf_estimation_loss or enforce_samples_to_be_on_surface) and (
                #                                     sdf_estimation_mode == 'density') and iteration > start_sdf_estimation_from,
                #                     )
                #
                #                 if (use_sdf_estimation_loss or enforce_samples_to_be_on_surface) and iteration > start_sdf_estimation_from:
                #                     # Compute the depth of the points in the gaussians
                #                     sdf_samples_in_camera_space = fov_camera.get_world_to_view_transform().transform_points(
                #                         sdf_samples)
                #                     sdf_samples_z = sdf_samples_in_camera_space[..., 2] + 0.
                #                     proj_mask = sdf_samples_z > fov_camera.znear
                #                     sdf_samples_map_z = sugar.get_points_depth_in_depth_map(fov_camera, depth,
                #                                                                             sdf_samples_in_camera_space[
                #                                                                                 proj_mask])
                #                     sdf_estimation = sdf_samples_map_z - sdf_samples_z[proj_mask]
                #
                #                     if not sample_only_in_gaussians_close_to_surface:
                #                         raise NotImplementedError("Not implemented yet.")
                #
                #                     with torch.no_grad():
                #                         if normalize_by_sdf_std:
                #                             sdf_sample_std = gaussian_standard_deviations[sdf_gaussian_idx][proj_mask]
                #                         else:
                #                             sdf_sample_std = sugar.get_cameras_spatial_extent() / 10.
                #
                #                     if use_sdf_estimation_loss:
                #                         if sdf_estimation_mode == 'sdf':
                #                             sdf_values = fields['sdf'][proj_mask]
                #                             if squared_sdf_estimation_loss:
                #                                 sdf_estimation_loss = (
                #                                             (sdf_values - sdf_estimation.abs()) / sdf_sample_std).pow(2)
                #                             else:
                #                                 sdf_estimation_loss = (
                #                                                                   sdf_values - sdf_estimation.abs()).abs() / sdf_sample_std
                #                             loss = loss + sdf_estimation_factor * sdf_estimation_loss.clamp(
                #                                 max=10. * sugar.get_cameras_spatial_extent()).mean()
                #                         elif sdf_estimation_mode == 'density':
                #                             beta = fields['beta'][proj_mask]
                #                             densities = fields['density'][proj_mask]
                #                             target_densities = torch.exp(-0.5 * sdf_estimation.pow(2) / beta.pow(2))
                #                             if squared_sdf_estimation_loss:
                #                                 sdf_estimation_loss = ((densities - target_densities)).pow(2)
                #                             else:
                #                                 sdf_estimation_loss = (densities - target_densities).abs()
                #                             loss = loss + sdf_estimation_factor * sdf_estimation_loss.mean()
                #                         else:
                #                             raise ValueError(f"Unknown sdf_estimation_mode: {sdf_estimation_mode}")
                #
                #                     if enforce_samples_to_be_on_surface:
                #                         if squared_samples_on_surface_loss:
                #                             samples_on_surface_loss = (sdf_estimation / sdf_sample_std).pow(2)
                #                         else:
                #                             samples_on_surface_loss = sdf_estimation.abs() / sdf_sample_std
                #                         loss = loss + samples_on_surface_factor * samples_on_surface_loss.clamp(
                #                             max=10. * sugar.get_cameras_spatial_extent()).mean()
                #
                #                 if use_sdf_better_normal_loss and (iteration > start_sdf_better_normal_from):
                #                     if iteration == start_sdf_better_normal_from + 1:
                #                         CONSOLE.print("\n---INFO---\nStarting SDF better normal loss.")
                #                     closest_gaussians_idx = sugar.knn_idx[sdf_gaussian_idx]
                #                     # Compute minimum scaling
                #                     closest_min_scaling = sugar.scaling.min(dim=-1)[0][
                #                         closest_gaussians_idx].detach().view(len(sdf_samples), -1)
                #
                #                     # Compute normals and flip their sign if needed
                #                     closest_gaussian_normals = sugar.get_normals(estimate_from_points=False)[
                #                         closest_gaussians_idx]
                #                     samples_gaussian_normals = sugar.get_normals(estimate_from_points=False)[
                #                         sdf_gaussian_idx]
                #                     closest_gaussian_normals = closest_gaussian_normals * torch.sign(
                #                         (closest_gaussian_normals * samples_gaussian_normals[:, None]).sum(dim=-1,
                #                                                                                            keepdim=True)
                #                     ).detach()
                #
                #                     # Compute weights for normal regularization, based on the gradient of the sdf
                #                     closest_gaussian_opacities = fields[
                #                         'closest_gaussian_opacities'].detach()  # Shape is (n_samples, n_neighbors)
                #                     normal_weights = ((sdf_samples[:, None] - sugar.points[
                #                         closest_gaussians_idx]) * closest_gaussian_normals).sum(
                #                         dim=-1).abs()  # Shape is (n_samples, n_neighbors)
                #                     if sdf_better_normal_gradient_through_normal_only:
                #                         normal_weights = normal_weights.detach()
                #                     normal_weights = closest_gaussian_opacities * normal_weights / closest_min_scaling.clamp(
                #                         min=1e-6) ** 2  # Shape is (n_samples, n_neighbors)
                #
                #                     # The weights should have a sum of 1 because of the eikonal constraint
                #                     normal_weights_sum = normal_weights.sum(dim=-1).detach()  # Shape is (n_samples,)
                #                     normal_weights = normal_weights / normal_weights_sum.unsqueeze(-1).clamp(
                #                         min=1e-6)  # Shape is (n_samples, n_neighbors)
                #
                #                     # Compute regularization loss
                #                     sdf_better_normal_loss = (samples_gaussian_normals - (
                #                                 normal_weights[..., None] * closest_gaussian_normals).sum(dim=-2)
                #                                               ).pow(2).sum(dim=-1)  # Shape is (n_samples,)
                #                     loss = loss + sdf_better_normal_factor * sdf_better_normal_loss.mean()
                #             else:
                #                 CONSOLE.log("WARNING: No gaussians available for sampling.")

                if iteration % 10 == 0:
                    print("loss : ", loss.item())
                loss.backward()

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        gaussians_trained = self.sugar_new.get_gaussians_scene()
        del self.sugar_new
        if "scales" in gauss_dict:
            return gaussians_trained
        else:
            self.update_all_param(gaussians_trained)

    def update_splat(self):
        gauss_dict = self.scene_info[0]["gaussian"]

        points = self.sugar._points.clone().detach()
        sh_coords_dc = self.sugar._sh_coordinates_dc.clone().detach()
        sh_coords_rest = self.sugar._sh_coordinates_rest.clone().detach()
        scales = self.sugar._scales.clone().detach()
        quaternions = self.sugar._quaternions.clone().detach()
        all_densities = self.sugar.all_densities.clone().detach()
        lang_fts = self.sugar._language_feature.clone().detach()

        n_gaussians_world = points.shape[0]

        n_points = points.shape[0] + gauss_dict["points"].shape[0]
        new_points = torch.zeros((n_points, points.shape[1]), device=points.device, dtype=torch.float)
        new_quaternions = torch.zeros((n_points, quaternions.shape[1]), device=quaternions.device, dtype=torch.float)
        new_sh_coords_dc = torch.zeros((n_points, sh_coords_dc.shape[1], sh_coords_dc.shape[2]),
                                       device=sh_coords_dc.device, dtype=torch.float)
        new_sh_coords_rest = torch.zeros((n_points, sh_coords_rest.shape[1], sh_coords_rest.shape[2]),
                                         device=sh_coords_rest.device, dtype=torch.float)
        new_scales = torch.zeros((n_points, scales.shape[1]), device=scales.device, dtype=torch.float)
        new_all_densities = torch.zeros((n_points, all_densities.shape[1]), device=all_densities.device,
                                        dtype=torch.float)
        new_lang_fts = torch.zeros((n_points, lang_fts.shape[1]), device=points.device, dtype=torch.float)

        new_points[:points.shape[0], :] = points
        new_quaternions[:quaternions.shape[0], :] = quaternions
        new_sh_coords_dc[:sh_coords_dc.shape[0], :, :] = sh_coords_dc
        new_sh_coords_rest[:sh_coords_rest.shape[0], :, :] = sh_coords_rest
        new_scales[:scales.shape[0], :] = scales
        new_all_densities[:all_densities.shape[0], :] = all_densities
        new_lang_fts[:lang_fts.shape[0], :] = lang_fts

        new_points[points.shape[0]:, :] = gauss_dict["points"]

        colors = torch.zeros_like(new_points)
        colors[points.shape[0]:] = torch.tensor([1, 0, 0])
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(new_points.cpu().numpy().tolist())
        pcd_colors = np.asarray(colors.cpu().numpy())
        pcd.colors = o3d.utility.Vector3dVector(pcd_colors.tolist())
        viz = o3d.visualization.Visualizer()
        viz.create_window()
        viz.add_geometry(pcd)
        opt = viz.get_render_option()
        opt.show_coordinate_frame = True
        viz.run()
        viz.destroy_window()


        gaussian_dict = {"points": new_points,
                         "sh_coordinates_dc": new_sh_coords_dc,
                         "sh_coordinates_rest": new_sh_coords_rest,
                         "scales": new_scales,
                         "quaternions": new_quaternions,
                         "densities": new_all_densities,
                         "language_features": new_lang_fts}

        colors = torch.zeros((n_points, points.shape[1]), device=points.device, dtype=torch.float)
        colors[:points.shape[0], :] = SH2RGB(sh_coords_dc[:, 0, :]).detach().float().cuda()
        colors[points.shape[0]:, :] = gauss_dict["colors"]

        del self.sugar
        torch.cuda.empty_cache()
        self.sugar_new = SuGaR(nerfmodel=self.nerfmodel,
                               points=gaussian_dict['points'].detach().float().cuda(),
                               colors=colors,
                               initialize=True,
                               sh_levels=4,
                               triangle_scale=1,
                               learnable_positions=True,
                               keep_track_of_knn=True,
                               knn_to_track=16,
                               freeze_gaussians=False,
                               beta_mode='average',
                               surface_mesh_to_bind=None,
                               surface_mesh_thickness=None,
                               learn_surface_mesh_positions=False,
                               learn_surface_mesh_opacity=False,
                               learn_surface_mesh_scales=False,
                               n_gaussians_per_surface_triangle=1,
                               include_feature=self.config.include_feature
                               )
        self.sugar_new.init_lang_ft(self.config.lang_ft_dim)

        with torch.no_grad():
            self.sugar_new._scales[:n_gaussians_world, :] = gaussian_dict["scales"][:n_gaussians_world, :]
            self.sugar_new._quaternions[:n_gaussians_world, :] = gaussian_dict["quaternions"][:n_gaussians_world, :]
            self.sugar_new.all_densities[:n_gaussians_world, :] = gaussian_dict["densities"][:n_gaussians_world, :]
            self.sugar_new._sh_coordinates_dc[:n_gaussians_world, :] = gaussian_dict["sh_coordinates_dc"][:n_gaussians_world, :]
            self.sugar_new._sh_coordinates_rest[:n_gaussians_world, :, :] = gaussian_dict["sh_coordinates_rest"][:n_gaussians_world, :, :]
            self.sugar_new._language_feature[:n_gaussians_world, :] = gaussian_dict["language_features"][:n_gaussians_world, :]

        for name, param in self.sugar_new.named_parameters():
            CONSOLE.print(name, param.shape, param.requires_grad)

        # setting the grads to true (to enable online update of parameters)
        self.sugar_new.set_grads()

        def loss_fn(pred_rgb, gt_rgb):
            return (1.0 - self.config.dssim_factor) * l1_loss(pred_rgb, gt_rgb) + self.config.dssim_factor * (1.0 - ssim(pred_rgb, gt_rgb))

        # TODO: Include densification and pruning
        opt_params = OptimizationParams(
            iterations=self.config.num_iterations,
            position_lr_init=self.config.position_lr_init,
            position_lr_final=self.config.position_lr_final,
            position_lr_delay_mult=self.config.position_lr_delay_mult,
            position_lr_max_steps=self.config.position_lr_max_steps,
            feature_lr=self.config.feature_lr,
            opacity_lr=self.config.opacity_lr,
            scaling_lr=self.config.scaling_lr,
            rotation_lr=self.config.rotation_lr,
            language_feature_lr=self.config.language_feature_lr
        )
        spatial_lr_scale = self.sugar_new.get_cameras_spatial_extent()
        optimizer = SuGaROptimizer(self.sugar_new, opt_params, spatial_lr_scale=spatial_lr_scale)
        self.sugar_new.train()

        # TODO: train semantics
        self.sugar_new.reset_grads_lang()
        if self.config.debug:
            CONSOLE.print("Optimizer initialized.")
            CONSOLE.print("Optimization parameters:")
            CONSOLE.print(opt_params)
            CONSOLE.print("---------------------------------------------------------------------")
            CONSOLE.print(f'Number of parameters: {sum(p.numel() for p in self.sugar_new.parameters() if p.requires_grad)}')
            CONSOLE.print("\nModel parameters:")
            for name, param in self.sugar_new.named_parameters():
                CONSOLE.print(name, param.shape, param.requires_grad)
            CONSOLE.print("---------------------------------------------------------------------")

        epoch = 0
        iteration = 0
        train_losses = []
        t0 = time.time()
        iteration = 0
        while iteration < self.config.num_iterations:
            for i in range(len(self.images_train)):
                iteration += 1
                gt_image = torch.tensor(self.images_train[i].copy(), dtype=torch.float).to(self.nerfmodel.device)
                gt_rgb = gt_image.view(-1, self.sugar_new.image_height, self.sugar_new.image_width, 3)
                cam_pose = self.cam_t[i]
                cam_center = copy.deepcopy(cam_pose[:3, 3])

                outputs = self.sugar_new.render_image_for_AI2THOR(
                    nerf_cameras=None,
                    c2w=cam_pose,
                    camera_center=cam_center,
                    verbose=False,
                    bg_color=None,
                    sh_deg=3,
                    sh_rotations=None,
                    compute_color_in_rasterizer=False,
                    compute_covariance_in_rasterizer=True,
                    return_2d_radii=True,
                    quaternions=None,
                    use_same_scale_in_all_directions=False,
                    return_opacities=False,
                )

                img_save = outputs["image"].detach().clone()

                torchvision.utils.save_image(img_save.permute(2, 0, 1),
                                             'output/WorldModel/update_new/{:04d}.png'.format(iteration))
                pred_rgb = outputs['image'].view(-1, self.sugar_new.image_height, self.sugar_new.image_width, 3)
                radii = outputs['radii']
                language_feature = outputs["language_feature_image"]

                pred_rgb = pred_rgb.transpose(-1, -2).transpose(-2, -3)
                gt_rgb = gt_rgb.transpose(-1, -2).transpose(-2, -3)

                # TODO: Add a semantic loss to optimize the language features in parallel
                loss = loss_fn(pred_rgb, gt_rgb)

                if iteration % 50 == 0:
                    print("loss : ", loss.item())

                loss.backward()

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            self.sugar_new._scales[:n_gaussians_world, :] = gaussian_dict["scales"][:n_gaussians_world, :]
            self.sugar_new._quaternions[:n_gaussians_world, :] = gaussian_dict["quaternions"][:n_gaussians_world, :]
            self.sugar_new.all_densities[:n_gaussians_world, :] = gaussian_dict["densities"][:n_gaussians_world, :]
            self.sugar_new._sh_coordinates_dc[:n_gaussians_world, :] = gaussian_dict["sh_coordinates_dc"][:n_gaussians_world, :]
            self.sugar_new._sh_coordinates_rest[:n_gaussians_world, :, :] = gaussian_dict["sh_coordinates_rest"][:n_gaussians_world, :, :]
            self.sugar_new._language_feature[:n_gaussians_world, :] = gaussian_dict["language_features"][:n_gaussians_world, :]

        self.sugar = self.sugar_new
        del self.sugar_new
    def create_new_splat(self):

        # TODO: currently testing out the capability to update the scene with a single image
        #  If we need multiple images, combine images from multiple scene captures
        new_gaussians = self.scene_info[0]["gaussian"]
        sugar =  SuGaR(nerfmodel=self.nerfmodel,
                       points=new_gaussians["points"].cuda(),
                       colors=SH2RGB(new_gaussians['sh_coordinates_dc'][:, 0, :]).cuda(),
                       initialize=False,
                       sh_levels=4,
                       triangle_scale=1,
                       learnable_positions=True,
                       keep_track_of_knn=False,
                       knn_to_track=0,
                       freeze_gaussians=False,
                       beta_mode=None,
                       surface_mesh_to_bind=None,
                       surface_mesh_thickness=None,
                       learn_surface_mesh_positions=False,
                       learn_surface_mesh_opacity=False,
                       learn_surface_mesh_scales=False,
                       n_gaussians_per_surface_triangle=1,
                       include_feature=self.config.include_feature
                       )
        sugar.init_lang_ft(self.config.lang_ft_dim)
        sugar.update_gaussians(new_gaussians)

        def loss_fn(pred_rgb, gt_rgb):
            return (1.0 - self.config.dssim_factor) * l1_loss(pred_rgb, gt_rgb) + self.config.dssim_factor * (1.0 - ssim(pred_rgb, gt_rgb))

        # TODO: Include densification and pruning
        opt_params = OptimizationParams(
            iterations=self.config.num_iterations,
            position_lr_init=self.config.position_lr_init,
            position_lr_final=self.config.position_lr_final,
            position_lr_delay_mult=self.config.position_lr_delay_mult,
            position_lr_max_steps=self.config.position_lr_max_steps,
            feature_lr=self.config.feature_lr,
            opacity_lr=self.config.opacity_lr,
            scaling_lr=self.config.scaling_lr,
            rotation_lr=self.config.rotation_lr,
            language_feature_lr=self.config.language_feature_lr
        )
        spatial_lr_scale = sugar.get_cameras_spatial_extent()
        optimizer = SuGaROptimizer(sugar, opt_params, spatial_lr_scale=spatial_lr_scale)
        sugar.train()

        # TODO: train semantics
        sugar.reset_grads_lang()
        if self.config.debug:
            CONSOLE.print("Optimizer initialized.")
            CONSOLE.print("Optimization parameters:")
            CONSOLE.print(opt_params)
            CONSOLE.print("---------------------------------------------------------------------")
            CONSOLE.print(f'Number of parameters: {sum(p.numel() for p in sugar.parameters() if p.requires_grad)}')
            CONSOLE.print("\nModel parameters:")
            for name, param in sugar.named_parameters():
                CONSOLE.print(name, param.shape, param.requires_grad)
            CONSOLE.print("---------------------------------------------------------------------")

        epoch = 0
        iteration = 0
        train_losses = []
        t0 = time.time()
        iteration = 0
        while iteration < self.config.num_iterations:
            for i in range(len(self.scene_info)):
                iteration += 1
                gt_image = torch.tensor(self.scene_info[i]["rgb_image"].copy(), dtype=torch.float).to(self.nerfmodel.device)
                gt_rgb = gt_image.view(-1, sugar.image_height, sugar.image_width, 3)
                cam_pose = self.scene_info[i]["c2w"]
                cam_center = copy.deepcopy(cam_pose[:3, 3])

                # print("mem1 : ", torch.cuda.memory_allocated(0) / 1024 / 1024 / 1024)

                outputs = sugar.render_image_for_AI2THOR(
                    nerf_cameras=None,
                    c2w=cam_pose,
                    camera_center=cam_center,
                    verbose=False,
                    bg_color=None,
                    sh_deg=3,
                    sh_rotations=None,
                    compute_color_in_rasterizer=False,
                    compute_covariance_in_rasterizer=True,
                    return_2d_radii=True,
                    quaternions=None,
                    use_same_scale_in_all_directions=False,
                    return_opacities=False,
                )
                # print("mem2 : ", torch.cuda.memory_allocated(0) / 1024 / 1024 / 1024)

                img_save = outputs["image"].detach().clone()

                # print("mem3 : ", torch.cuda.memory_allocated(0) / 1024 / 1024 / 1024)

                # print("saving image .....")
                torchvision.utils.save_image(img_save.permute(2, 0, 1),
                                             'output/WorldModel/update/{:04d}.png'.format(iteration))
                pred_rgb = outputs['image'].view(-1, sugar.image_height, sugar.image_width, 3)
                radii = outputs['radii']
                language_feature = outputs["language_feature_image"]

                pred_rgb = pred_rgb.transpose(-1, -2).transpose(-2, -3)
                gt_rgb = gt_rgb.transpose(-1, -2).transpose(-2, -3)

                # TODO: Add a semantic loss to optimize the language features in parallel
                loss = loss_fn(pred_rgb, gt_rgb)

                CONSOLE.print("------Stats-----")
                CONSOLE.print(f'\n-------------------\nIteration: {iteration}')
                CONSOLE.print(f"loss: {loss:>7f}  [{iteration:>5d}/{self.config.num_iterations:>5d}]",
                              "computed in", (time.time() - t0) / 60., "minutes.")
                CONSOLE.print("---Min, Max, Mean, Std")
                CONSOLE.print("Points:", sugar.points.min().item(), sugar.points.max().item(),
                              sugar.points.mean().item(), sugar.points.std().item(), sep='   ')
                CONSOLE.print("Scaling factors:", sugar.scaling.min().item(), sugar.scaling.max().item(),
                              sugar.scaling.mean().item(), sugar.scaling.std().item(), sep='   ')
                CONSOLE.print("Quaternions:", sugar.quaternions.min().item(), sugar.quaternions.max().item(),
                              sugar.quaternions.mean().item(), sugar.quaternions.std().item(), sep='   ')
                CONSOLE.print("Sh coordinates dc:", sugar._sh_coordinates_dc.min().item(),
                              sugar._sh_coordinates_dc.max().item(), sugar._sh_coordinates_dc.mean().item(),
                              sugar._sh_coordinates_dc.std().item(), sep='   ')
                CONSOLE.print("Sh coordinates rest:", sugar._sh_coordinates_rest.min().item(),
                              sugar._sh_coordinates_rest.max().item(), sugar._sh_coordinates_rest.mean().item(),
                              sugar._sh_coordinates_rest.std().item(), sep='   ')
                CONSOLE.print("Opacities:", sugar.strengths.min().item(), sugar.strengths.max().item(),
                              sugar.strengths.mean().item(), sugar.strengths.std().item(), sep='   ')

                loss.backward()

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        self.scene_info = {}

class World():
    def __init__(self, device, config):
        self.device = device
        self.save_img_base = "output/WorldModel/"
        self.config = config

        self.pos_before_manipulation = None
        self.rot_before_manipulation = None
        self.pose_init = None
        self.pose_final = None
        self.center = None

        self.o3d_viz = False

    def init_first_step(self, evt):
        self.last_evt = copy.deepcopy(evt)

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

    def get_c2w_transformation(self, evt):
        # position
        position = copy.deepcopy(evt.metadata['cameraPosition'])
        pos_trans = [position['x'], position['y'], position['z']]

        # orientation
        rotation = copy.deepcopy(evt.metadata['agent']['rotation'])
        rot_matrix = Rotation.from_euler("zxy", [rotation['z'], rotation['x'], rotation['y']], degrees=True).as_matrix()

        # adjust pose to match coordinate system between unity and COLMAP
        adjusted_pose_ = self.adjust_pose(rot_matrix, pos_trans)
        adjusted_pose = torch.tensor(adjusted_pose_, device=self.device, dtype=torch.float)
        adjusted_pose[:3, 1:3] *= -1

        return adjusted_pose

    def save_image(self, evt, time_step):
        rgb_image = evt.frame
        rgb_image = rgb_image/255
        rgb_image = torch.tensor(rgb_image).permute(2, 0, 1)
        torchvision.utils.save_image(rgb_image, 'output/WorldModel/sim/{:03d}.png'.format(time_step))

    def save_image_path(self, evt, name, path):
        rgb_image = evt.frame
        rgb_image = rgb_image/255
        rgb_image = torch.tensor(rgb_image).permute(2, 0, 1)
        torchvision.utils.save_image(rgb_image, os.path.join(path, '{}.png'.format(name)))

    def create_traj_video(self):
        image_folder = os.path.join(self.save_img_base, 'sim')
        image_paths = [img for img in os.listdir(image_folder) if img.endswith(".jpg") or img.endswith(".png")]
        image_paths.sort()

        frames = []
        for image_path in image_paths:
            sim_img_path = os.path.join(image_folder, image_path)
            sim_image = torchvision.io.read_image(sim_img_path).permute(1, 2, 0)

            rendered_image_path = os.path.join(self.save_img_base, "rendered", image_path)
            rendered_image = torchvision.io.read_image(rendered_image_path).permute(1, 2, 0)

            lang_field_path = os.path.join(self.save_img_base, "lang_field", image_path)
            lang_field = torchvision.io.read_image(lang_field_path).permute(1, 2, 0)

            frame = torch.zeros((sim_image.shape[0], sim_image.shape[1]*3 + 200, 3))
            frame[:, :sim_image.shape[1], :] = sim_image
            frame[:, sim_image.shape[1] + 100: sim_image.shape[1]*2 + 100, :] = rendered_image
            frame[:, -sim_image.shape[1]:, :] = lang_field

            frames.append(frame)
        frames = torch.stack(frames)
        filename = 'output/WorldModel/output.mp4'
        fps = 2  # frames per second

        # Write the video
        torchvision.io.write_video(filename, frames, fps)

    def transformation_between_pose(self, init_dict, final_dict, new_pick, rot_change=None):
        init_pose = self.adjust_pose(Rotation.from_euler("xyz", init_dict['rotation'], degrees=True).as_matrix(),
                                     init_dict['position'])
        final_pose = self.adjust_pose(Rotation.from_euler("xyz", final_dict['rotation'], degrees=True).as_matrix(),
                                      final_dict['position'])
        init_pose[:3, 1:3] *= -1
        final_pose[:3, 1:3] *= -1

        change_pose = final_pose @ np.linalg.inv(init_pose)
        self.center = init_pose[:3, 3]

        return change_pose

    def get_new_gaussians(self, evt, bbox_3d):
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

        adjusted_pose_ = self.adjust_pose(rot_matrix, pos_trans)
        depth = evt.depth_frame
        frame_size = depth.shape[:2]
        x = np.arange(0, frame_size[1])
        y = np.arange(frame_size[0], 0, -1)
        xx, yy = np.meshgrid(x, y)

        xx = xx.flatten()
        yy = yy.flatten()
        zz = depth.flatten()
        x_camera = (xx - cx) * zz / fx
        y_camera = -(yy - cy) * zz / fy
        z_camera = zz
        points_camera = np.stack([x_camera, y_camera, z_camera], axis=1)
        homogenized_points = np.hstack([points_camera, np.ones((points_camera.shape[0], 1))])
        points_world = adjusted_pose_ @ homogenized_points.T
        points_world = points_world.T[..., :3]
        points_color = rgb_image.reshape(-1, 3)

        # downsampling # TODO: check the speed without downsampling
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(points_world)
        # pcd.colors = o3d.utility.Vector3dVector(point_color)
        # down_pcd = pcd.voxel_down_sample(voxel_size=0.05)
        # down_points = np.asarray(down_pcd.points)
        # down_colors = np.asarray(down_pcd.colors)
        # down_points = torch.tensor(down_points, device=self.device)
        # down_colors = torch.tensor(down_colors, device=self.device)

        down_points = torch.tensor(points_world, device=self.device)
        down_colors = torch.tensor(points_color, device=self.device)

        # create a mask for updating gaussians in that area
        within_min = down_points >= bbox_3d['min'].to(down_points.device)
        within_max = down_points <= bbox_3d['max'].to(down_points.device)
        gaussian_mask = within_min & within_max
        gaussian_mask = torch.all(gaussian_mask, dim=-1)

        new_points = down_points[gaussian_mask]
        new_colors = down_colors[gaussian_mask]

        # new_points = down_points
        # new_colors = down_colors

        if self.o3d_viz:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(new_points.cpu().numpy().tolist())
            pcd_colors = np.asarray(new_colors.cpu().numpy()) / 255
            pcd.colors = o3d.utility.Vector3dVector(pcd_colors.tolist())
            viz = o3d.visualization.Visualizer()
            viz.create_window()
            viz.add_geometry(pcd)
            opt = viz.get_render_option()
            opt.show_coordinate_frame = True
            viz.run()
            viz.destroy_window()

        update_dict = {
            "points": new_points.to(self.device).to(torch.float),
            "colors": new_colors.to(self.device).to(torch.float),
        }
        return update_dict

# Navigation and 2d mapping was inspired from : https://github.com/Gabesarch/TIDEE/tree/main
# Go take a look at their work!
class Agent():
    def __init__(self, config):
        self.config = config

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

        self.navigation = Navigation()

        self.clip_ft_extractor = CLIPFeatureExtractor(self.config.clip_device)

        self.all_objects = OPENABLE_OBJECTS + PICKUPABLE_OBJECTS
        # self.all_objects = ["An object that I can pick up", "An object that I can open"]
        self.all_object_fts = self.clip_ft_extractor.tokenize_text(self.all_objects).detach()

        self.objects_sim = {}
        self.objects_rend = {}
        self.sim_obj_id = 0
        self.rend_obj_id = 0
        self.sim_centers = []
        self.rend_centers = []


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

        return points_world.reshape(rgb_image.shape[0], rgb_image.shape[1], 3)

    def init_success_checker(self, rgb, controller):
        self.navigation.init_success_checker(rgb, controller)

    def update_navigation_obs(self, rgb, depth, action_successful, update_success_checker=True):
        # action_successful = controller.last_event.metadata['lastActionSuccess']
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

    def crop_image_mask(self, image, mask):
        xpadding = 10
        ypadding = 10
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

        cropped_image = image[y_min:y_max + 1, x_min:x_max + 1]
        return cropped_image

    def get_center_from_mask(self, mask):
        y, x = np.where(mask * 1 != 0)
        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()
        xc = int((x_min + x_max) / 2)
        yc = int((y_min + y_max) / 2)
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
        bbox_new = torch.tensor(object_new.bbox)

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

    def check_object_similarty(self, center, centers):
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

    def merge_detections(self, scene_type, new_object, obj_id):
        if scene_type == "sim":
            if new_object.object_name[0] not in self.objects_sim[obj_id].object_name:
                self.objects_sim[obj_id].n_diff_det += 1
            self.objects_sim[obj_id].object_name.append(new_object.object_name[0])
            self.objects_sim[obj_id].object_pos_world.append(new_object.object_pos_world[0])
            self.objects_sim[obj_id].object_pos_map.append(new_object.object_pos_map[0])
            self.objects_sim[obj_id].task.append(new_object.task[0])
            self.objects_sim[obj_id].conf.append(new_object.conf[0])
            self.objects_sim[obj_id].clip_ft.append(new_object.clip_ft[0])
        elif scene_type == "rend":
            if new_object.object_name[0] not in self.objects_rend[obj_id].object_name:
                self.objects_rend[obj_id].n_diff_det += 1
            self.objects_rend[obj_id].object_name.append(new_object.object_name[0])
            self.objects_rend[obj_id].object_pos_world.append(new_object.object_pos_world[0])
            self.objects_rend[obj_id].object_pos_map.append(new_object.object_pos_map[0])
            self.objects_rend[obj_id].task.append(new_object.task[0])
            self.objects_rend[obj_id].conf.append(new_object.conf[0])
            self.objects_rend[obj_id].clip_ft.append(new_object.clip_ft[0])
        else:
            raise Exception("Invalid scene type")

    def reason_about_change(self, evt, sim_image, rendered_image, rendered_lang, masks, step):

        # rendered_lang = rendered_lang.permute(1, 2, 0)
        rendered_image = rendered_image.detach().clone().cpu().numpy()
        rendered_image = (rendered_image*255).astype(np.uint8)
        save_image_sim = copy.deepcopy(sim_image)
        save_image_rend = copy.deepcopy(rendered_image)

        pcd_frame = self.get_pcd(evt)

        for mask in masks:
            sim_object = ObjectScene()
            rend_object = ObjectScene()

            # Extracting clip features for image crops
            cropped_image = self.crop_image_mask(sim_image, mask)
            crop_fts = self.clip_ft_extractor.tokenize_image(cropped_image)
            # masked_lang = rendered_lang[mask]
            # masked_lang = masked_lang.mean()

            cropped_image_rend = self.crop_image_mask(rendered_image, mask)
            crop_fts_rend = self.clip_ft_extractor.tokenize_image(cropped_image_rend)

            # Finding the cosine similarity with the objects
            cs_sim = torch.nn.functional.cosine_similarity(self.all_object_fts, crop_fts)
            cs_rend = torch.nn.functional.cosine_similarity(self.all_object_fts, crop_fts_rend)

            # # Normalizing the cosine similarity
            # cs_sim = (cs_sim - cs_sim.min())/(cs_sim.max() - cs_sim.min())
            # cs_rend = (cs_rend - cs_rend.min())/(cs_rend.max() - cs_rend.min())

            save_image_sim[mask] = (sim_image[mask]*0.65 + np.array([0, 255, 0])*0.35).astype(np.uint8)
            save_image_rend[mask] = (save_image_rend[mask]*0.65 + np.array([0, 255, 255])*0.35).astype(np.uint8)

            center = self.get_center_from_mask(mask)
            pos_map = self.navigation.explorer.mapper.get_goal_position_on_map(
                np.array(pcd_frame[center[1], center[0], 0], pcd_frame[center[1], center[0], 0]))
            if cs_sim.max() > 0.229:
                sim_object.object_name.append(self.all_objects[cs_sim.argmax()])
                sim_object.object_pos_world.append(pcd_frame[center[1], center[0], :])
                sim_object.object_pos_map.append(pos_map)
                sim_object.conf.append(cs_sim.max())
                sim_object.clip_ft.append(crop_fts)
                sim_object.img_crops.append(cropped_image)
                sim_object.points = pcd_frame[mask].reshape(-1, 3)
                sim_object.bbox = sim_object.compute_bbox3d()
                if self.all_objects[cs_sim.argmax()] in OPENABLE_OBJECTS:
                    sim_object.task.append("open")
                else:
                    sim_object.task.append("pick")
            else:
                sim_object.object_name.append("oh")
                sim_object.object_pos_world.append(pcd_frame[center[1], center[0], :])
                sim_object.object_pos_map.append(pos_map)
                sim_object.task.append("nah")
                sim_object.conf.append(0)
                sim_object.clip_ft.append(None)

            if cs_rend.max() > 0.229:
                rend_object.object_name.append(self.all_objects[cs_rend.argmax()])
                rend_object.object_pos_world.append(pcd_frame[center[1], center[0], :])
                rend_object.object_pos_map.append(pos_map)
                rend_object.conf.append(cs_rend.max())
                rend_object.clip_ft.append(crop_fts_rend)
                rend_object.img_crops.append(cropped_image_rend)
                rend_object.points = pcd_frame[mask].reshape(-1, 3)
                rend_object.bbox = rend_object.compute_bbox3d()
                if self.all_objects[cs_rend.argmax()] in OPENABLE_OBJECTS:
                    rend_object.task.append("open")
                else:
                    rend_object.task.append("pick")
            else:
                rend_object.object_name.append("oh")
                rend_object.object_pos_world.append(pcd_frame[center[1], center[0], :])
                rend_object.object_pos_map.append(pos_map)
                rend_object.task.append("nah")
                rend_object.conf.append(0)
                rend_object.clip_ft.append(None)

            # TODO: check for similar objects in memory
            if self.sim_obj_id == 0:
                if cs_sim.max() > 0.229:
                    self.objects_sim[self.sim_obj_id] = sim_object
                    self.sim_centers.append(sim_object.object_pos_world[0])
                    self.sim_obj_id += 1
                    obj_id_save = 0
            else:
                if cs_sim.max() > 0.229:
                    obj_id = self.check_object_similarty(sim_object.object_pos_world[0], self.sim_centers)
                    if obj_id == -1:
                        obj_id_save = copy.deepcopy(self.sim_obj_id)
                        self.objects_sim[self.sim_obj_id] = sim_object
                        self.sim_centers.append(sim_object.object_pos_world[0])
                        self.sim_obj_id += 1
                    else:
                        self.merge_detections("sim", sim_object, obj_id)
                        obj_id_save = copy.deepcopy(obj_id)
                else:
                    obj_id = self.check_object_similarty(sim_object.object_pos_world[0], self.sim_centers)
                    # if it is a new detection, and it is a false detection, then discard it
                    if obj_id != -1:
                        self.objects_sim[obj_id].n_false_det += 1
                        obj_id_save = copy.deepcopy(obj_id)
                    else:
                        obj_id_save = "#"
            if cs_sim.max() > 0.229:
                save_image_sim = cv2.putText(save_image_sim,
                                         self.all_objects[cs_sim.argmax()] + " " + str(obj_id_save), center,
                                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            if self.rend_obj_id == 0:
                if cs_rend.max() > 0.229:
                    self.objects_rend[self.rend_obj_id] = rend_object
                    self.rend_centers.append(rend_object.object_pos_world[0])
                    self.rend_obj_id += 1
                    obj_id_save = 0
            else:
                if cs_rend.max() > 0.229:
                    obj_id = self.check_object_similarty(rend_object.object_pos_world[0], self.rend_centers)
                    if obj_id == -1:
                        obj_id_save = copy.deepcopy(self.rend_obj_id)
                        self.objects_rend[self.rend_obj_id] = rend_object
                        self.rend_centers.append(rend_object.object_pos_world[0])
                        self.rend_obj_id += 1
                    else:
                        self.merge_detections("rend", rend_object, obj_id)
                        obj_id_save = copy.deepcopy(obj_id)
                else:
                    obj_id = self.check_object_similarty(rend_object.object_pos_world[0], self.rend_centers)
                    # if it is a new detection, and it is a false detection, then discard it
                    if obj_id != -1:
                        self.objects_rend[obj_id].n_false_det += 1
                        obj_id_save = copy.deepcopy(obj_id)
                    else:
                        obj_id_save = "#"
            x, y = center
            if cs_rend.max() > 0.229:
                save_image_sim = cv2.putText(save_image_sim,
                                             self.all_objects[cs_rend.argmax()] + " " + str(obj_id_save),
                                             (x, y + 15),
                                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            del sim_object
            del rend_object
        save_image_sim = torch.tensor(save_image_sim / 255)
        save_image_rend = torch.tensor(save_image_rend / 255)
        full_image = torch.zeros(save_image_sim.shape[0], save_image_sim.shape[1]*2, 3)
        full_image[:, :save_image_sim.shape[1], :] = save_image_sim
        full_image[:, save_image_sim.shape[1]:, :] = save_image_rend

        torchvision.utils.save_image(full_image.permute(2, 0, 1),
                                     "output/WorldModel/reason/{:04d}_save.png".format(step))

        print("sim objects at step {}".format(step), len(self.objects_sim))
        print("rend objects at step {}".format(step), len(self.objects_rend))

    def postprocess_detections(self):
        rearrange_dict = {}
        sim_checker = {}
        other_data = {}
        print("==============================================================================")
        print("before formatting : ", len(self.objects_sim), len(self.objects_rend))

        # removing similar detections from sim objects and rend objects
        print("before post process step 1 : ", len(self.objects_sim), len(self.objects_rend))
        # new_sim_objects = {}
        # for sim_index in self.objects_sim:
        #     if len(new_sim_objects) == 0:
        #         new_sim_objects[sim_index] = self.objects_sim[sim_index]
        #     else:
        #         # computing similarity
        #         max_cs_new = 0
        #         argmax_idx_new = 0
        #         for new_sim_index in new_sim_objects:
        #             cs = torch.nn.functional.cosine_similarity(self.objects_sim[sim_index].clip_ft[0], new_sim_objects[new_sim_index].clip_ft[0])
        #             if cs > max_cs_new:
        #                 max_cs_new = cs
        #                 argmax_idx_new = new_sim_index
        #         if max_cs_new < 0.8:
        #             new_sim_objects[sim_index] = self.objects_sim[sim_index]
        # self.objects_sim = new_sim_objects
        #
        # new_rend_objects = {}
        # for rend_index in self.objects_rend:
        #     if len(new_rend_objects) == 0:
        #         new_rend_objects[rend_index] = self.objects_rend[rend_index]
        #     else:
        #         # computing similarity
        #         max_cs_new = 0
        #         argmax_idx_new = 0
        #         for new_rend_index in new_rend_objects:
        #             cs = torch.nn.functional.cosine_similarity(self.objects_rend[rend_index].clip_ft[0],
        #                                                        new_rend_objects[new_rend_index].clip_ft[0])
        #             if cs > max_cs_new:
        #                 max_cs_new = cs
        #                 argmax_idx_new = new_rend_index
        #         if max_cs_new < 0.8:
        #             new_rend_objects[rend_index] = self.objects_rend[rend_index]
        # self.objects_rend = new_rend_objects

        print("after post process step 1 : ", len(self.objects_sim), len(self.objects_rend))

        for i in self.objects_sim:
            if self.objects_sim[i].n_false_det < 4: #and self.objects_sim[i].n_diff_det < 3:
                #print(self.objects_sim[i].object_name, self.objects_sim[i].conf, self.objects_sim[i].n_false_det)
                cosine_sim_max = 0
                idx_max = 0
                img_sim = self.objects_sim[i].img_crops[0]
                for j in self.objects_rend:
                    if self.objects_rend[j].n_false_det < 4: #and self.objects_rend[j].n_diff_det < 3:
                        cosine_sim = torch.nn.functional.cosine_similarity(self.objects_sim[i].clip_ft[0], self.objects_rend[j].clip_ft[0])
                        if cosine_sim > cosine_sim_max:
                            cosine_sim_max = cosine_sim
                            idx_max = j
                if idx_max in rearrange_dict.keys():
                    if sim_checker[idx_max] < cosine_sim_max < 0.9:
                        sim_checker[idx_max] = cosine_sim_max
                        rearrange_dict[idx_max] = i
                        other_data[idx_max] = {"sim": {"name": self.objects_sim[i].object_name,
                                                       "conf": self.objects_sim[i].conf,
                                                       "false_det": self.objects_sim[i].n_false_det},
                                               "rend": {"name": self.objects_rend[idx_max].object_name,
                                                        "conf": self.objects_rend[idx_max].conf,
                                                        "false_det": self.objects_rend[idx_max].n_false_det}
                                               }
                else:
                    rearrange_dict[idx_max] = i
                    sim_checker[idx_max] = cosine_sim_max
                    other_data[idx_max] = {"sim": {"name": self.objects_sim[i].object_name,
                                                   "conf": self.objects_sim[i].conf,
                                                   "false_det": self.objects_sim[i].n_false_det},
                                           "rend": {"name": self.objects_rend[idx_max].object_name,
                                                    "conf": self.objects_rend[idx_max].conf,
                                                    "false_det": self.objects_rend[idx_max].n_false_det}
                                           }

        # for rend_idx in rearrange_dict:
        #     if sim_checker[rend_idx] < 0.9:
        #         sim_idx = rearrange_dict[rend_idx]
        #         img_sim = self.objects_sim[sim_idx].img_crops[0]
        #         img_rend = self.objects_rend[rend_idx].img_crops[0]
        #         image = torch.zeros(img_sim.shape[0] + img_rend.shape[0],
        #                             max(img_sim.shape[1], img_rend.shape[1]), 3)
        #         image[:img_sim.shape[0], :img_sim.shape[1], :] = torch.tensor(img_sim/255)
        #         image[img_sim.shape[0]:, :img_rend.shape[1], :] = torch.tensor(img_rend/255)
        #         torchvision.utils.save_image(image.permute(2, 0, 1),
        #                                      "output/WorldModel/rearr/{:03d}_{:03d}_save.png".format(sim_idx, rend_idx))

        all_sim_idx = np.array(list(rearrange_dict.values()))
        sim_sim_dict = {}
        for rend_idx in rearrange_dict:
            if sim_checker[rend_idx] < 0.9:
                sim_idx = rearrange_dict[rend_idx]
                max_cs = 0
                argmax_id = 0
                for id in all_sim_idx:
                    if id != sim_idx:
                        cs = torch.nn.functional.cosine_similarity(self.objects_sim[id].clip_ft[0], self.objects_sim[sim_idx].clip_ft[0])
                        if cs > max_cs:
                            max_cs = cs
                            argmax_id = id
                sim_sim_dict[sim_idx] = {"sim": max_cs,
                                         "idx": argmax_id}
                img_sim1 = self.objects_sim[sim_idx].img_crops[0]
                img_sim2 = self.objects_sim[argmax_id].img_crops[0]
                image = torch.zeros(img_sim1.shape[0] + img_sim2.shape[0],
                                    max(img_sim1.shape[1], img_sim2.shape[1]), 3)
                image[:img_sim1.shape[0], :img_sim1.shape[1], :] = torch.tensor(img_sim1 / 255)
                image[img_sim1.shape[0]:, :img_sim2.shape[1], :] = torch.tensor(img_sim2 / 255)
                torchvision.utils.save_image(image.permute(2, 0, 1),
                                             "output/WorldModel/sim_sim/{:03d}_{:03d}_save.png".format(sim_idx, argmax_id))

        print("after formatting : ", len(rearrange_dict))

        print("before pruning sim : ", len(rearrange_dict))
        pruned_rearrange_dict = {}
        for rend_idx in rearrange_dict:
            sim_idx = rearrange_dict[rend_idx]
            if len(pruned_rearrange_dict) == 0:
                pruned_rearrange_dict[rend_idx] = rearrange_dict[rend_idx]
            else:
                max_cs = 0
                max_id = 0
                for id_ in pruned_rearrange_dict:
                    cs = torch.nn.functional.cosine_similarity(self.objects_sim[sim_idx].clip_ft[0],
                                                               self.objects_sim[pruned_rearrange_dict[id_]].clip_ft[0])
                    if cs > max_cs:
                        max_cs = cs
                        max_id = id_
                if max_cs > 0.8 :
                    # same object
                    # TODO: check the threshold
                    if sim_checker[max_id] > sim_checker[rend_idx]:
                        # replace detection
                        del pruned_rearrange_dict[max_id]
                        pruned_rearrange_dict[rend_idx] = rearrange_dict[rend_idx]
                else:
                    pruned_rearrange_dict[rend_idx] = rearrange_dict[rend_idx]
        rearrange_dict = pruned_rearrange_dict
        print("after pruning sim : ", len(rearrange_dict))
        # pruned_rearrange_dict = {}
        # print("******************************************************************************")
        # for rend_idx in rearrange_dict:
        #     sim_idx = rearrange_dict[rend_idx]
        #     if len(pruned_rearrange_dict) == 0:
        #         pruned_rearrange_dict[rend_idx] = rearrange_dict[rend_idx]
        #     else:
        #         max_cs = 0
        #         max_id = 0
        #         for id_ in pruned_rearrange_dict:
        #             cs = torch.nn.functional.cosine_similarity(self.objects_rend[rend_idx].clip_ft[0],
        #                                                        self.objects_rend[id_].clip_ft[0])
        #             if cs > max_cs:
        #                 max_cs = cs
        #                 max_id = id_
        #         if max_cs > 0.8:
        #             # same object
        #             # TODO: check the threshold
        #             if sim_checker[max_id] > sim_checker[rend_idx]:
        #                 # replace detection
        #                 del pruned_rearrange_dict[max_id]
        #                 pruned_rearrange_dict[rend_idx] = rearrange_dict[rend_idx]
        #         else:
        #             pruned_rearrange_dict[rend_idx] = rearrange_dict[rend_idx]
        #         print("curr obj, similar obj, score : ", rend_idx, max_id, max_cs)
        # rearrange_dict = pruned_rearrange_dict
        # print("******************************************************************************")
        # print("after pruning rend : ", len(rearrange_dict))

        for rend_idx in rearrange_dict:
            if 0.75 < sim_checker[rend_idx] < 0.9:
                sim_idx = rearrange_dict[rend_idx]
                img_sim = self.objects_sim[sim_idx].img_crops[0]
                img_rend = self.objects_rend[rend_idx].img_crops[0]
                image = torch.zeros(img_sim.shape[0] + img_rend.shape[0],
                                    max(img_sim.shape[1], img_rend.shape[1]), 3)
                image[:img_sim.shape[0], :img_sim.shape[1], :] = torch.tensor(img_sim/255)
                image[img_sim.shape[0]:, :img_rend.shape[1], :] = torch.tensor(img_rend/255)
                torchvision.utils.save_image(image.permute(2, 0, 1),
                                             "output/WorldModel/rearr/{:03d}_{:03d}_save.png".format(sim_idx, rend_idx))

        # for p in sim_checker:
        #     if sim_checker[p] < 0.9:
        #         print("\n------------------------------------------------------------------------------")
        #         print("idx : ", p, rearrange_dict[p])
        #         print("cosine sim : ", sim_checker[p])
        #         print("sim : ", other_data[p]["sim"])
        #         print("rend : ", other_data[p]["rend"])
        #         print("center : ", self.objects_sim[rearrange_dict[p]].object_pos_world[0])
        #         print("------------------------------------------------------------------------------\n")
        #
        # for p in sim_sim_dict:
        #     print("#####################################################################")
        #     print("id : ", p, sim_sim_dict[p]["idx"])
        #     print("sim : ", sim_sim_dict[p]["sim"])
        #     print("#####################################################################")
        # print("\n==============================================================================\n")
        # for i in range(len(self.objects_rend)):
        #     if self.objects_rend[i].n_false_det < 2 and self.objects_rend[i].n_diff_det < 3:
        #         print(self.objects_rend[i].object_name, self.objects_rend[i].conf, self.objects_rend[i].n_false_det)
        return rearrange_dict, sim_checker


class Runner():
    def __init__(self, train=False):
        self.gaussian_config = GaussianConfig(width=500,
                                              height=500)
        if not train:
            self.gaussian_world_model = GaussianWorldModel(self.gaussian_config)
            self.env = World(self.gaussian_world_model.nerfmodel.device, self.gaussian_config)

        self.matcher = Dinov2Matcher()

        self.agent = Agent(self.gaussian_config)

        self.data_logger = Data("/home/nune/gaussian_splatting/lgsplat-mesh/dataset/FloorPlan303_physics")

        self.key_actions = {
            'w': 'MoveAhead',
            's': 'MoveBack',
            'a': 'MoveLeft',
            'd': 'MoveRight',
            'e': {'action': 'RotateRight', 'degrees': 10},
            'q': {'action': 'RotateLeft', 'degrees': 10},
            'p': {'action': 'PickupObject'},
            'l': {'action': 'PutObject'},
            'i': {'action': "MoveHeldObjectAhead"},
            'o': {'action': "RotateHeldObject"},
            'c': 'capture_image',
            'u': 'update_gaussians',
            'b': 'save_data'
        }
        self.timestep = 0
        self.end_run = False
        self.event = None
        self.mask_2d = None
        self.cam_2_world = None
        self.depth_ = None

        self.controller = Controller(scene="FloorPlan303_physics",
                                     renderDepthImage=True,
                                     width=self.gaussian_config.width,
                                     height=self.gaussian_config.height,
                                     renderInstanceSegmentation=True,
                                     gridSize=self.gaussian_config.map_args.STEP_SIZE,
                                     snapToGrid=False)

        # random seed to move objects around
        self.controller.step(dict(action = 'InitialRandomSpawn', randomSeed = 1))
        self.init_event = self.controller.step('Pass')

        self.picked_obj_id = None # none corresponds to no object in hand
        self.track_obj = {}
        self.track_gaussians = {}
        self.new_pick = True

        self.prev_event = None
        if not train:
            self.env.init_first_step(self.init_event)

        self.event = copy.deepcopy(self.init_event)

        self.pickup_stack = {}
        self.place_stack = {}
        self.open_stack = {}

    def match_frames(self, sim_image, rendered_image):
        torchvision.utils.save_image(rendered_image.permute(2, 0, 1),
                                     'output/WorldModel/rendered/rnd_{:03d}.png'.format(self.timestep))
        rendered_image = cv2.imread('output/WorldModel/rendered/rnd_{:03d}.png'.format(self.timestep))
        rendered_image = cv2.cvtColor(rendered_image, cv2.COLOR_BGR2RGB)
        sim_mask = sim_image[..., 0] > -1
        rendered_mask = rendered_image[..., 0] > -1

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
        mask_cs = cosine_similarity < 0.35
        mask_cs = mask_cs.reshape(*grid_size1)
        mask_img = cv2.resize(np.array(mask_cs * 1), (sim_image.shape[1], sim_image.shape[0]),
                              interpolation=cv2.INTER_NEAREST)

        # This mask corresponds to the region where there is a change in the objects
        mask_img = mask_img.astype(np.bool_)

        if self.gaussian_world_model.config.viz_dense_features:
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

            cv2.imwrite(os.path.join(self.gaussian_world_model.config.save_dino_frames, "{}.png".format(self.timestep)),
                        save_image)

        if self.gaussian_config.dilate_object_mask:
            mask_img = mask_img.astype(np.uint8)
            kernel = np.ones((5, 5), np.uint8)
            dilated_mask = cv2.dilate(mask_img, kernel, iterations=self.gaussian_config.dilate_iterations)
            cv2.imwrite('output/WorldModel/dilated_mask.png', dilated_mask)
            dilated_mask = dilated_mask.astype(np.bool_)
            mask_img = dilated_mask
        return mask_img

    def picked_obj(self, obj_id, timestep, evt, cam_2_world, mask_2d, depth_frame, pos_init=None, rot_init=None):
        self.picked_obj_id = obj_id
        if self.new_pick:
            self.track_obj[timestep-1] = {"position": [pos_init['x'], pos_init['y'], pos_init['z']],
                                          "rotation": [-rot_init['x'], -rot_init['y'], rot_init['z']]}

        # TODO: pose from pose tracker or from RGBD image and camera intrinsic, extrinsic
        for j in range(len(evt.metadata["objects"])):
            if evt.metadata["objects"][j]["objectId"] == self.picked_obj_id:
                pos = evt.metadata["objects"][j]["position"]
                rot = evt.metadata["objects"][j]["rotation"]
        self.track_obj[timestep] = {"position": [pos['x'], pos['y'], pos['z']],
                                    "rotation": [-rot['x'], -rot['y'], rot['z']]}
        if self.new_pick:
            trans_pose = self.env.transformation_between_pose(self.track_obj[timestep-1],
                                                                         self.track_obj[timestep], self.new_pick)
        else:
            trans_pose = self.env.transformation_between_pose(self.track_obj[timestep - 1],
                                                                         self.track_obj[timestep], self.new_pick,
                                                                         rot_change=[45, 25, 90])
        out = self.gaussian_world_model.load_virtual_camera(cam_2_world, self.timestep)
        cam_2_world[:3, 1:3] *= -1
        self.gaussian_world_model.update_gaussians_continuous("cardboard",
                                                   trans_pose,
                                                   mask_2d,
                                                   out["language_feature_image"],
                                                   cam_2_world,
                                                   depth_frame,
                                                   self.new_pick,
                                                   self.env.center)
        self.new_pick = False

    def place_obj(self):
        self.new_pick = True

    def check_task_done(self):
        # TODO: based on rearrange task
        return False

    def delineate_masks(self, mask_all):
        mask_all = (mask_all * 255).astype(np.uint8)
        num_labels, labeled_mask = cv2.connectedComponents(mask_all, connectivity=4)
        individual_masks = []

        # Start from 1 to ignore the background
        for label in range(1, num_labels):
            individual_mask = np.uint8(labeled_mask == label) * 255
            individual_masks.append(individual_mask.astype(np.bool_))

        return individual_masks


    def runner(self, train=False, inds_explore=[]):
        self.init_pos = self.event.metadata['agent']['position']
        done_task = False
        step = 0
        exploring = True
        num_sampled = 0
        while not done_task:
            if step == 0:
                self.agent.navigation.init_navigation(None)
                camX0_T_camX = self.agent.navigation.explorer.get_camX0_T_camX()

                # TODO: task
                rgb = self.event.frame
                depth = self.event.depth_frame

                self.agent.init_success_checker(rgb, self.controller)

                # For the first step
                action_successful = True
                self.agent.update_navigation_obs(rgb, depth, action_successful)

                camX0_T_camX = self.agent.navigation.explorer.get_camX0_T_camX()
                camX0_T_camX0 = geom.safe_inverse_single(camX0_T_camX)
                origin_T_camX0 = aithor.get_origin_T_camX(self.controller.last_event, True)
                camX0_T_origin = geom.safe_inverse_single(origin_T_camX0)

            reach_point = False
            if reach_point:
                action, param = self.agent.navigation.act(point_goal=True, goal_loc=np.array([1.3809, -0.62926]))
            else:
                action, param = self.agent.navigation.act()
            action_rearrange = self.agent.nav_action_to_rearrange_action[action]
            action_ind = self.agent.action_to_ind[action_rearrange]

            camX0_T_camX = self.agent.navigation.explorer.get_camX0_T_camX()

            if self.gaussian_config.nav_verbose:
                print("action , index : ", action_rearrange, action_ind)

            if action == "Pass":
                exploring = False
                num_sampled += 1
                try:
                    if not inds_explore:
                        ind_i, ind_j = self.agent.navigation.get_reachable_map_locations(sample=True)
                        if not ind_i:
                            break
                        inds_explore.append([ind_i, ind_j])
                    else:
                        ind_i, ind_j = inds_explore.pop(0)
                except:
                    break

                self.agent.navigation.set_point_goal(ind_i, ind_j)

            else:
                # TODO: implement the step fn with task for the rearrangement task
                event = self.agent.step(action_rearrange, self.controller)

                rgb = event.frame
                depth = event.depth_frame
                action_successful = self.agent.navigation.success_checker.check_successful_action(rgb)
                self.agent.update_navigation_obs(rgb, depth, action_successful)
                if action_successful:
                    if not train:
                        self.env.save_image(event, self.timestep)
                        c2w = self.env.get_c2w_transformation(event)
                        out = self.gaussian_world_model.load_virtual_camera(c2w, self.timestep)
                        mask_all = self.match_frames(event.frame, out['image'])
                        masks = self.delineate_masks(mask_all)
                        self.agent.navigation.explorer.add_change_to_map(depth, masks)
                        # lang_enc = out['language_feature_image'].permute(1, 2, 0)
                        # lang_ft = self.gaussian_world_model.autoenc_model.decode(lang_enc.detach().reshape(-1, 3).to(self.gaussian_config.decoder_device))
                        # lang_ft = lang_ft.reshape(rgb.shape[0], rgb.shape[1], 512)
                        # lang_ft = lang_ft.permute(2, 0, 1)
                        self.agent.reason_about_change(event,
                                                           rgb,
                                                           out['image'].detach().clone(),
                                                           None,
                                                           masks,
                                                           step)
                    else:
                        self.data_logger.get_data_step(fov=event.metadata['fov'],
                                                       rgb=event.frame,
                                                       depth=event.depth_frame,
                                                       camera_pos=event.metadata['cameraPosition'],
                                                       camera_rot=event.metadata['agent']['rotation'])
            if self.agent.navigation.explorer.goal.category != "cover":
                done_task = True
                if train:
                    self.data_logger.save_data()
                else:
                    rearrange_dict, sim_checker = self.agent.postprocess_detections()

            step += 1
            self.timestep += 1
            # done_task = self.check_task_done()

            # agent stuck
            if num_sampled > 25:
                break

        # move to locations in rearrange dict
        if not train:
            for rend_idx in rearrange_dict:
                if sim_checker[rend_idx] < 0.9:
                    self.agent.navigation.explorer.reinit_act_queue()
                    reached = False
                    once = True
                    print("\n\n======================================================================================")
                    while not reached:
                        for i in range(len(event.metadata["objects"])):
                            if event.metadata["objects"][i]["objectType"] == "Book":
                                pos = event.metadata["objects"][i]["position"]
                        pos['x'] -= self.init_pos['x']
                        pos['z'] -= self.init_pos['z']

                        cent_ = self.agent.navigation.explorer.mapper.get_position_on_map_from_aithor_position(pos)
                        print("POS : ", self.agent.navigation.explorer.mapper.get_position_on_map_from_aithor_position(pos))
                        print("======================================================================================\n\n")

                        self.agent.navigation.explorer.place_loc_rend = False
                        sim_idx = rearrange_dict[rend_idx]
                        center = self.agent.objects_sim[sim_idx].object_pos_world[0]
                        self.agent.navigation.explorer.add_indices(str(rend_idx) + " " + str(sim_idx))
                        # for i in range(len(event.metadata["objects"])):
                        #     if event.metadata["objects"][i]["objectType"] == "BaseballBat":
                        #         obj_id = event.metadata["objects"][i]["objectId"]
                        #         pos = event.metadata["objects"][i]["position"]
                        #         rot = event.metadata["objects"][i]["rotation"]
                        if once:
                            print("rend, sim index : ", rend_idx, sim_idx)
                            print("center : ", center)
                            once = False
                        print("curr loc : ", self.agent.navigation.explorer.mapper.get_position_on_map())
                        # action, param = self.agent.navigation.act(point_goal=True, goal_loc=np.array([-center[0], -center[2]]))
                        action, param = self.agent.navigation.act(point_goal=True,
                                                                  goal_loc_map=cent_)
                        action_rearrange = self.agent.nav_action_to_rearrange_action[action]
                        if action_rearrange == "done":
                            reached = True
                            break
                        action_ind = self.agent.action_to_ind[action_rearrange]
                        print("action rearrange : ", action_rearrange)
                        event = self.agent.step(action_rearrange, self.controller)
                        if action_rearrange == "look_down":
                            self.env.save_image_path(event, str(rend_idx) + "_" + str(sim_idx),
                                                     'output/WorldModel/reached/')
                        self.env.save_image(event, self.timestep)
                        rgb = event.frame
                        depth = event.depth_frame
                        action_successful = self.agent.navigation.success_checker.check_successful_action(rgb)
                        self.agent.update_navigation_obs(rgb, depth, action_successful)
                        self.timestep += 1

                    self.agent.navigation.explorer.reinit_act_queue()
                    reached = False
                    once = True
                    print("\n--------------------------------------------------------------------------------------\n")
                    while not reached:
                        self.agent.navigation.explorer.place_loc_rend = True
                        sim_idx = rearrange_dict[rend_idx]
                        center = self.agent.objects_rend[rend_idx].object_pos_world[0]
                        self.agent.navigation.explorer.add_indices(str(rend_idx) + " " + str(sim_idx))
                        # for i in range(len(event.metadata["objects"])):
                        #     if event.metadata["objects"][i]["objectType"] == "BaseballBat":
                        #         obj_id = event.metadata["objects"][i]["objectId"]
                        #         pos = event.metadata["objects"][i]["position"]
                        #         rot = event.metadata["objects"][i]["rotation"]
                        if once:
                            print("rend, sim index : ", rend_idx, sim_idx)
                            print("center : ", center)
                            once = False
                        print("curr loc : ", self.agent.navigation.explorer.mapper.get_position_on_map())
                        action, param = self.agent.navigation.act(point_goal=True,
                                                                      goal_loc=np.array([-center[0], -center[2]]))
                        action_rearrange = self.agent.nav_action_to_rearrange_action[action]
                        if action_rearrange == "done":
                            reached = True
                            break
                        action_ind = self.agent.action_to_ind[action_rearrange]
                        print("action rearrange : ", action_rearrange)
                        event = self.agent.step(action_rearrange, self.controller)
                        if action_rearrange == "look_down":
                            self.env.save_image_path(event, str(rend_idx) + "_" + str(sim_idx),
                                                     'output/WorldModel/reached/')
                        self.env.save_image(event, self.timestep)
                        rgb = event.frame
                        depth = event.depth_frame
                        action_successful = self.agent.navigation.success_checker.check_successful_action(rgb)
                        self.agent.update_navigation_obs(rgb, depth, action_successful)
                        self.timestep += 1

                    print("======================================================================================\n\n")


    def runner_teleop(self):
        while not self.end_run:
            next_step = input("\n move: ")
            next_step = next_step.lower()

            if next_step in self.key_actions:
                action = self.key_actions[next_step]
                if next_step == 'p':
                    for i in range(len(event.metadata["objects"])):
                        if event.metadata["objects"][i]["objectType"] == "Statue":
                            obj_id = event.metadata["objects"][i]["objectId"]
                            pos = event.metadata["objects"][i]["position"]
                            rot = event.metadata["objects"][i]["rotation"]
                            break
                    cam_2_world = self.env.get_c2w_transformation(event)
                    mask_2d = event.instance_masks[obj_id]
                    depth = event.depth_frame
                    event = self.controller.step(action="PickupObject",
                                                 objectId=obj_id,
                                                 manualInteract=False)
                    if event.metadata['lastActionSuccess']:
                        self.picked_obj(obj_id=obj_id,
                                        timestep=self.timestep,
                                        evt=event,
                                        cam_2_world=cam_2_world,
                                        mask_2d=mask_2d,
                                        depth_frame=depth,
                                        pos_init=pos,
                                        rot_init=rot)
                        self.controller.step('Pass')

                elif next_step == 'i':
                    for i in range(len(event.metadata["objects"])):
                        if event.metadata["objects"][i]["objectType"] == "Statue":
                            obj_id = event.metadata["objects"][i]["objectId"]
                            pos = event.metadata["objects"][i]["position"]
                            rot = event.metadata["objects"][i]["rotation"]
                            break
                    cam_2_world = self.env.get_c2w_transformation(event)
                    mask_2d = event.instance_masks[obj_id]
                    depth = event.depth_frame
                    event = self.controller.step(action="MoveHeldObjectAhead",
                                            moveMagnitude=0.1,
                                            forceVisible=False)
                    self.picked_obj(obj_id=obj_id,
                                    timestep=self.timestep,
                                    evt=event,
                                    cam_2_world=cam_2_world,
                                    mask_2d=mask_2d,
                                    depth_frame=depth,
                                    pos_init=pos,
                                    rot_init=rot)
                    self.controller.step('Pass')

                elif next_step == 'o':
                    for i in range(len(event.metadata["objects"])):
                        if event.metadata["objects"][i]["objectType"] == "Statue":
                            obj_id = event.metadata["objects"][i]["objectId"]
                            pos = event.metadata["objects"][i]["position"]
                            rot = event.metadata["objects"][i]["rotation"]
                            break
                    cam_2_world = self.env.get_c2w_transformation(event)
                    mask_2d = event.instance_masks[obj_id]
                    depth = event.depth_frame
                    event = self.controller.step(action="RotateHeldObject",
                                                 pitch=90,
                                                 yaw=25,
                                                 roll=45)
                    self.picked_obj(obj_id=obj_id,
                                    timestep=self.timestep,
                                    evt=event,
                                    cam_2_world=cam_2_world,
                                    mask_2d=mask_2d,
                                    depth_frame=depth,
                                    pos_init=pos,
                                    rot_init=rot)
                    self.controller.step('Pass')

                elif next_step == 'l':
                    for i in range(len(event.metadata["objects"])):
                        if event.metadata["objects"][i]["objectType"] == "Statue":
                            obj_id = event.metadata["objects"][i]["objectId"]
                            pos = event.metadata["objects"][i]["position"]
                            rot = event.metadata["objects"][i]["rotation"]
                        if event.metadata["objects"][i]["objectType"] == "Sofa":
                            rec_id = event.metadata["objects"][i]["objectId"]
                    cam_2_world = self.env.get_c2w_transformation(event)
                    mask_2d = event.instance_masks[obj_id]
                    depth = event.depth_frame
                    event = self.controller.step(action="PutObject",
                                                 objectId=rec_id,
                                                 forceAction=True,
                                                 placeStationary=True)
                    if event.metadata['lastActionSuccess']:
                        self.picked_obj(obj_id=obj_id,
                                        timestep=self.timestep,
                                        evt=event,
                                        cam_2_world=cam_2_world,
                                        mask_2d=mask_2d,
                                        depth_frame=depth,
                                        pos_init=pos,
                                        rot_init=rot)
                        self.controller.step('Pass')
                        self.place_obj()

                elif next_step == 'c':
                    # TODO: automate the image collection
                    c2w = self.env.get_c2w_transformation(event)
                    out = self.gaussian_world_model.load_virtual_camera(c2w, self.timestep)
                    for i in range(len(event.metadata["objects"])):
                        if event.metadata["objects"][i]["objectType"] == "Statue":
                            obj_id = event.metadata["objects"][i]["objectId"]

                    # TODO: make separate masks from DINOV2 for separate objects
                    #  (maybe can utilize the DINOV2 feature representation)
                    mask = self.match_frames(event.frame, out['image'])
                    # mask = event.instance_masks[obj_id]
                    cam2World = copy.deepcopy(c2w)
                    cam2World[:3, 1:3] *= -1
                    bbox_3d = self.gaussian_world_model.spatial_selection_gaussians(mask, cam2World, event.depth_frame)
                    new_gaussians = self.env.get_new_gaussians(event, bbox_3d)
                    self.gaussian_world_model.save_scene_info_to_update_splat(event, c2w, mask, new_gaussians, bbox_3d)

                elif next_step == 'u':
                    # update the splat based on changes in the real world
                    # call this after some movement in the environment
                    # TODO: queued up multiple changes, done parallely
                    # bbox_3d = copy.deepcopy(self.gaussian_world_model.object_bbox_hist[0])
                    # new_gaussian_dict = self.env.get_new_gaussians(event, bbox_3d)
                    # self.gaussian_world_model.remove_stray_gaussians(bbox_3d)
                    # self.gaussian_world_model.init_new_gaussians(new_gaussian_dict)
                    # self.gaussian_world_model.update_splat()


                    #self.gaussian_world_model.create_new_splat()

                    self.gaussian_world_model.gaussian_splat_train()
                    # self.gaussian_world_model.train_surface_aligned_gaussians()
                    # self.gaussian_world_model.update_splat()

                elif next_step == "b":
                    print("hi")
                    self.gaussian_world_model.save_data()

                elif isinstance(action, dict):
                    if not self.new_pick:
                        for i in range(len(event.metadata["objects"])):
                            if event.metadata["objects"][i]["objectType"] == "Statue":
                                obj_id = event.metadata["objects"][i]["objectId"]
                                pos = event.metadata["objects"][i]["position"]
                                rot = event.metadata["objects"][i]["rotation"]
                                break
                        cam_2_world = self.env.get_c2w_transformation(event)
                        mask_2d = event.instance_masks[obj_id]
                        depth = event.depth_frame
                    event = self.controller.step(action=action['action'], degrees=action['degrees'])
                    if not self.new_pick:
                        self.picked_obj(obj_id=obj_id,
                                        timestep=self.timestep,
                                        evt=event,
                                        cam_2_world=cam_2_world,
                                        mask_2d=mask_2d,
                                        depth_frame=depth,
                                        pos_init=pos,
                                        rot_init=rot)
                    self.controller.step('Pass')
                else:
                    if not self.new_pick:
                        for i in range(len(event.metadata["objects"])):
                            if event.metadata["objects"][i]["objectType"] == "Statue":
                                obj_id = event.metadata["objects"][i]["objectId"]
                                pos = event.metadata["objects"][i]["position"]
                                rot = event.metadata["objects"][i]["rotation"]
                                break
                        cam_2_world = self.env.get_c2w_transformation(event)
                        mask_2d = event.instance_masks[obj_id]
                        depth = event.depth_frame
                    event = self.controller.step(action=action)
                    if not self.new_pick:
                        self.picked_obj(obj_id=obj_id,
                                        timestep=self.timestep,
                                        evt=event,
                                        cam_2_world=cam_2_world,
                                        mask_2d=mask_2d,
                                        depth_frame=depth,
                                        pos_init=pos,
                                        rot_init=rot)
                    self.controller.step('Pass')

                if event.metadata['lastActionSuccess']:
                    self.timestep += 1
                    self.gaussian_world_model.timestep = self.timestep
                    c2w = self.env.get_c2w_transformation(event)
                    out = self.gaussian_world_model.load_virtual_camera(c2w, self.timestep)
                    self.env.save_image(event, self.timestep)
                    self.env.update_map(event, out)
                    self.prev_event = copy.deepcopy(event)
                else:
                    print(f"Action {action} failed")
                time.sleep(0.1)

            elif next_step == 'x':
                self.env.create_traj_video()
                self.end_run = True

if __name__ == '__main__':
    runner_ai2thor = Runner(train=False)
    runner_ai2thor.runner(train=False)