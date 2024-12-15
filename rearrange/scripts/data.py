import numpy as np
import copy
import cv2
import os
import pickle
import open3d as o3d
from scipy.spatial.transform import Rotation
from rearrange.utils.read_write_model import CameraModel, Camera, Point3D
from rearrange.utils.read_write_model import write_images_text, write_cameras_text, write_points3D_text
from rearrange.utils.read_write_model import Image as ImageModel
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

class Data():
    def __init__(self, path, config, diff_splat=False):
        self.config = config
        self.points_ = []
        self.point_colors = []

        self.images = {}
        self.image_id = 0

        self.pcds = {}
        self.point_id = 0

        self.viz_pcd = True
        self.base_path = path

        self.sparse_path = os.path.join(self.base_path, "sparse/0/")
        if not os.path.exists(self.sparse_path):
            os.makedirs(self.sparse_path)

        self.img_path = os.path.join(self.base_path, "images/")
        if not os.path.exists(self.img_path):
            os.makedirs(self.img_path)

        self.diff_splat = diff_splat
        if diff_splat:
            self.diff_save_path = os.path.join(self.base_path, "diff_images/")
            if not os.path.exists(self.diff_save_path):
                os.makedirs(self.diff_save_path)

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

    def get_data_step(self, fov, rgb, depth, camera_pos, camera_rot, diff_image=None, diff_mask=None):
        if len(self.camera_model) == 0:
            self.height, self.width, _ = rgb.shape
            self.fx = (self.width / 2) / (np.tan(np.deg2rad(fov / 2)))
            self.fy = (self.height / 2) / (np.tan(np.deg2rad(fov / 2)))
            self.cx = (self.width - 1) / 2
            self.cy = (self.height - 1) / 2
            # print("-----------CAMERA INTRINSICS-----------")
            # print("fx, fy, cx, cy : ", self.fx, self.fy, self.cx, self.cy)
            # print("---------------------------------------")

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

        # diff feat
        if self.diff_splat:
            diff_image_save_path_ = os.path.join(self.diff_save_path, "frame_{}.pickle".format(self.image_id))
            with open(diff_image_save_path_, "wb") as f:
                pickle.dump(diff_image, f)
            diff_mask_save_path_ = os.path.join(self.diff_save_path, "frame_{}_mask.pickle".format(self.image_id))
            with open(diff_mask_save_path_, "wb") as f:
                pickle.dump(diff_mask, f)

            values = diff_image.reshape(-1)
            cmap = plt.get_cmap('viridis')
            norm = mcolors.Normalize(vmin=0, vmax=1)
            colors = cmap(norm(values))
            colors = colors[:, :3]
            colors = colors.reshape(diff_image.shape[0], diff_image.shape[1], 3)
            # mask_1 = diff_image == 1
            # mask_2 = diff_image == 0.5
            # mask_3 = diff_image == 0
            # colors = np.zeros((diff_image.shape[0], diff_image.shape[1], 3))
            # colors[mask_1] = np.array([1, 0, 0])
            # colors[mask_2] = np.array([0, 1, 0])
            # colors[mask_3] = np.array([0, 0, 1])
            colors = (colors*255).astype(np.uint8)
            cv2.imwrite(os.path.join(self.diff_save_path, "frame_{}.png".format(self.image_id)), colors)


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
        down_pcd = pcd.voxel_down_sample(voxel_size=0.015)
        down_points = np.asarray(down_pcd.points)
        down_colors = np.asarray(down_pcd.colors)

        # Global downsampling
        self.points_.extend(down_points.tolist())
        self.point_colors.extend(down_colors.tolist())

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.array(self.points_))
        pcd.colors = o3d.utility.Vector3dVector(np.array(self.point_colors))
        down_pcd = pcd.voxel_down_sample(voxel_size=0.018)
        down_points = np.asarray(down_pcd.points)
        down_colors = np.asarray(down_pcd.colors)

        self.points_ = down_points.tolist()
        self.point_colors = down_colors.tolist()

    def save_data(self):
        # print("\n saving data .....")
        # write image data
        write_images_text(self.images, os.path.join(self.sparse_path, "images.txt"))
        point_colors = np.asarray(self.point_colors).astype(np.uint32)
        point_colors = point_colors.tolist()

        if self.config.pcd_vis:
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
            # viz.add_geometry(line_set)
            # viz.add_geometry(axis)
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
        print("data saved...")