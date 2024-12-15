import open3d as o3d
import numpy as np
import copy
import torch
from scipy.spatial.transform import Rotation as R

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
        self.max_mask_size = 0
        self.nav_pos_map = None
        self.image = None
        self.mask = None
        self.depth_image = None
        self.task_final = None
        self.pcd_frame = None
        self.center_accurate = None
        self.center_current_frame = None
        self.bbox_area = 0
        self.pick_percent = None
        self.accurate_mask = None
        self.avg_dist = None
        self.accurate_image_crop = None
        self.final_clip_ft = []

    def compute_bbox3d(self):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.points)

        if len(self.points) > 4:
            try:
                return torch.tensor(np.asarray(pcd.get_oriented_bounding_box(robust=True).get_box_points()))
            except RuntimeError as e:
                print(f"Met {e}, use axis aligned bounding box instead")
                return torch.tensor(np.asarray(pcd.get_axis_aligned_bounding_box().get_box_points()))
        else:
            return torch.tensor(np.asarray(pcd.get_axis_aligned_bounding_box().get_box_points()))



class Event():
    def __init__(self):
        self.frame = None
        self.depth_frame = None
        self.metadata = {
            "fov": 90,
            "cameraPosition": {},
            "agent": {
                "position": {},
                "rotation": {},
                "cameraHorizon": 0
            }
        }


event = Event()


def convert_rearrange_obs_to_event(obs_agent, rgb, depth):
    event.frame = rgb
    event.depth_frame = depth
    position = {"x": obs_agent["x"],
                "y": obs_agent["y"],
                "z": obs_agent["z"]}
    rotation = {"x": obs_agent["horizon"],
                "y": obs_agent['rotation'],
                "z": 0}
    cam_pos = copy.deepcopy(position)
    cam_pos['y'] += 0.675
    event.metadata["cameraPosition"] = cam_pos
    event.metadata["agent"]["position"] = position
    event.metadata["agent"]["rotation"] = rotation
    event.metadata["agent"]["cameraHorizon"] = obs_agent["horizon"]
    return event


def convert_path_to_event(pos, quat, rgb, depth):
    event.frame = rgb
    event.depth_frame = depth
    position = {"x": pos[0],
                "y": pos[1],
                "z": pos[2]}
    ang = R.from_quat(quat).as_euler('xyz')
    rotation = {"x": ang[0],
                "y": ang[1],
                "z": ang[2]}
    event.metadata["cameraPosition"] = position
    event.metadata["agent"]["position"] = position
    event.metadata["agent"]["rotation"] = rotation
    event.metadata["agent"]["cameraHorizon"] = ang[2]
    return event

class CameraState:
    def __init__(self, x, y, z, quaternion):
        """
        State in GRID coordinate system
        """
        self.position = np.array([x, y, z])
        self.quaternion = quaternion

    def __eq__(self, other):
        return np.array_equal(self.position, other.position)

    def __hash__(self):
        return hash(tuple(self.position)) + hash(tuple(self.quaternion))

    def __lt__(self, other):
        return False

    def distance_to(self, other):
        return np.linalg.norm(self.position - other.position)

    def transform_point(self, point):
        if np.linalg.norm(self.quaternion) != 0:
            rot = R.from_quat(self.quaternion)
            self.rot_matrix = rot.as_matrix()
            h_trans = np.zeros((4, 4))
            h_trans[0:3, 3] = self.position
            h_trans[0:3, 0:3] = self.rot_matrix
            h_trans[3, 3] = 1
            h_trans_inv = np.linalg.inv(h_trans)
            point_cam = h_trans_inv @ np.array([point[0], point[1], point[2], 1])
            return point_cam[:3]
        else:
            return point

