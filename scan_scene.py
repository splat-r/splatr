import os
import numpy as np
from pynput import keyboard
import time
from ai2thor.controller import Controller
from read_write_model import CameraModel, Camera, Image, Point3D
from read_write_model import write_images_text, write_cameras_text, write_points3D_text
from scipy.spatial.transform import Rotation
import pickle
import cv2
import open3d as o3d
import copy


controller = Controller(scene="FloorPlan20_physics",
                        renderDepthImage=True,
                        width=500,
                        height=500)

# controller = tt.launch_controller({"scene": "FloorPlan202_physics"})

# data base_path
basepath = "/home/nune/gaussian_splatting/gaussian-splatting/dataset/Ai2Thor/FloorPlan20_physics"

key_actions = {
    'w': 'MoveAhead',
    's': 'MoveBack',
    'a': 'MoveLeft',
    'd': 'MoveRight',
    'e': {'action': 'RotateRight', 'degrees': 10},
    'q': {'action': 'RotateLeft', 'degrees': 10},
    'p': 'SaveData',
}

class data():
    def __init__(self, path):
        self.points_ = []
        self.point_colors = []
        print("initializing...")

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

    def get_data_step(self, evt):
        if len(self.camera_model) == 0:
            fov = evt.metadata['fov']
            self.height, self.width, _ = evt.frame.shape
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
            write_cameras_text(self.camera_model, os.path.join(basepath, "sparse/0/cameras.txt"))

        self.image_id += 1
        image_name = "frame_{}.png".format(self.image_id)

        # image
        rgb_image = evt.frame
        cv2.imwrite(os.path.join(self.img_path, image_name), cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))

        # position
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

        # tr_matrix = np.array([[1, 0, 0],
        #                       [0, 1, 0],
        #                       [0, 0, 1]])
        #
        # new_trans_pose = tr_matrix @ new_trans_pose

        # saving for viz
        self.trans = copy.deepcopy(new_trans_pose)
        self.camera_pose.append(new_trans_pose)
        self.rot = copy.deepcopy(rotation_matrix)

        rot_new = Rotation.from_matrix(rotation_matrix)
        quaternion = rot_new.as_quat()
        quaternion_reoriented = [quaternion[3], quaternion[0], quaternion[1], quaternion[2]]  # to match COLMAP dataset

        self.images[self.image_id] = Image(id=self.image_id,
                            qvec=np.array(quaternion_reoriented),
                            tvec=np.array(new_trans_pose),
                            camera_id=1,  # pinhole camera
                            name=image_name,
                            xys=np.array([0, 0]),  # initialized to origin, since it is not used
                            point3D_ids=np.array([0]),)  # initialized to origin, since it is not used

        # pointcloud

        depth = evt.depth_frame

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


        """
        depth = evt.depth_frame
        grid_x, grid_y = np.meshgrid(np.arange(self.width), np.arange(self.height))
        x = (grid_x - self.cx) * depth / self.fx
        y = -(grid_y - self.cy) * depth / self.fy
        z = -depth
        points = np.stack((x, y, z), axis=-1)
        points = points.reshape(-1, 3)

        yaw = np.deg2rad(rotation['y'])
        rotation_matrix = np.array([[np.cos(yaw), 0, np.sin(yaw)],
                                    [0, 1, 0],
                                    [-np.sin(yaw), 0, np.cos(yaw)]])
        c2w = np.eye(4)
        c2w[0:3, 0:3] = rotation_matrix
        c2w[0:3, 3] = [position['x'], position['y'], position['z']]

        transforming the points from camera frame to world frame
        homogenized_points = np.hstack([points, np.ones((points.shape[0], 1))])
        transformed_points = ex_inv @ homogenized_points.T
        transformed_points = transformed_points.T[..., :3]

        transformed_points[:, 1] = -1*transformed_points[:, 1]
        x = copy.deepcopy(transformed_points[:, 0])
        transformed_points[:, 0] = copy.deepcopy(transformed_points[:, 2])
        transformed_points[:, 2] = x
        """

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

        inp = input("continue?")
        if inp == "y":
            return True


        # viewer = o3d.visualization.Visualizer()
        # viewer.add_geometry(pcd)
        # viewer.create_window()
        # opt = viewer.get_render_option()
        # opt.show_coordinate_frame = True
        # viewer.run()

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
        return False

d = data(path=basepath)


def on_press(key):
    try:
        if key.char in key_actions:
            action = key_actions[key.char]
            # save data if triggered
            if action == 'SaveData':
                ret = d.save_data()
                if ret:
                    return False
            # Check if action requires additional parameters like degrees
            elif isinstance(action, dict):
                event = controller.step(action=action['action'], degrees=action['degrees'])
            else:
                event = controller.step(action=action)

            if action != 'SaveData':
                if event.metadata['lastActionSuccess']:
                    d.get_data_step(event)
                    #d.get_pcd(event)
                else:
                    print(f"Action {action} failed")
            time.sleep(0.1)
        elif key.char == 'x':
            return False  # Stop the listener to exit the program

    except AttributeError:
        pass


# Start the keyboard listener in non-blocking mode
listener = keyboard.Listener(on_press=on_press)
listener.start()

# Keep the script running
try:
    while listener.is_alive():
        time.sleep(0.1)
except KeyboardInterrupt:
    pass
