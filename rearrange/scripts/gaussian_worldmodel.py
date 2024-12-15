import torch
import torchvision
from rich.console import Console
from tqdm import trange
import numpy as np
import copy
import os
import cv2
from PIL import Image
from scipy.spatial.transform import Rotation

from rearrange.utils.read_write_model import write_images_text, write_cameras_text, write_points3D_text
from rearrange.utils.read_write_model import CameraModel, Camera, Point3D
from rearrange.utils.read_write_model import Image as ImageModel
from sugar_scene.gs_model import GaussianSplattingWrapper
from sugar_scene.sugar_model import SuGaR
from sugar_utils.spherical_harmonics import SH2RGB, RGB2SH
from autoencoder.model import Autoencoder


CONSOLE = Console(width=120)


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
            if not self.config.second_splat:
                torchvision.utils.save_image(pred_rgb.permute(2, 0, 1), 'output/test/{}.png'.format(i))
            else:
                torchvision.utils.save_image(pred_rgb.permute(2, 0, 1), 'output_diff/test/{}.png'.format(i))

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
            torchvision.utils.save_image(language_feature, os.path.join(self.config.run_path, 'lang_field/{:03d}.png'.format(time_step)))
        pred_rgb = outputs['image']
        torchvision.utils.save_image(pred_rgb.permute(2, 0, 1), os.path.join(self.config.run_path, 'rendered/{:03d}.png'.format(time_step)))
        return outputs

    def load_virtual_camera_diff(self, cam_pose):
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
        pred_rgb = outputs['image']
        return pred_rgb

    def render_depth_vis(self, cam_pose, time_step):
        cam_center = copy.deepcopy(cam_pose[:3, 3])
        depth = self.sugar.render_image_for_AI2THOR(
            nerf_cameras=None,
            c2w=cam_pose,
            camera_center=cam_center,
            bg_color=None,
            sh_deg=0,
            compute_color_in_rasterizer=False,  # compute_color_in_rasterizer,
            compute_covariance_in_rasterizer=True,
            return_2d_radii=False,
            use_same_scale_in_all_directions=False,
            point_colors=self.sugar.points,
        )[..., 0]
        depth = torch.abs(depth)
        depth = (depth - depth.min()) / (depth.max() - depth.min())
        depth = depth.detach().cpu().numpy()
        depth = (depth*255).astype(np.uint8)
        return depth

    def render_depth(self, cam_pose, time_step):
        cam_center = copy.deepcopy(cam_pose[:3, 3])
        depth = self.sugar.render_image_for_AI2THOR(
            nerf_cameras=None,
            c2w=cam_pose,
            camera_center=cam_center,
            bg_color=None,
            sh_deg=0,
            compute_color_in_rasterizer=False,  # compute_color_in_rasterizer,
            compute_covariance_in_rasterizer=True,
            return_2d_radii=False,
            use_same_scale_in_all_directions=False,
            point_colors=self.sugar.points,
        )[..., 0]
        return depth

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
