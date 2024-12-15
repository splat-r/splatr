import os
import copy
import torch
import torchvision
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation

class World():
    def __init__(self, device, config):
        self.device = device
        self.save_img_base = config.run_path
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

    def get_c2w_transformation_from_pose(self, pos, rot):
        # position
        position = copy.deepcopy(pos)
        pos_trans = [position['x'], position['y'], position['z']]

        # orientation
        rotation = copy.deepcopy(rot)
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
        torchvision.utils.save_image(rgb_image, os.path.join(self.config.run_path, 'sim/{:03d}.png'.format(time_step)))

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
        filename = os.path.join(self.config.run_path, 'output.mp4')
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