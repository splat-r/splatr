import copy
import logging
import numpy as np
from ast import literal_eval
import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)
from navigation.mapper import Mapper
from navigation.utils.depth_utils import get_camera_matrix
from navigation.fmm_planner import FMMPlanner
import skimage
from skimage.measure import label  
from textwrap import wrap
from queue import LifoQueue 
from matplotlib import pyplot as plt 
import math
import cv2
import torch
import os
import ipdb
st = ipdb.set_trace

from collections import deque

# seed = 42
# np.random.seed(seed)

######## Detectron2 imports start ########
# from .config import retrieval_detector_path
# import detectron2
# from detectron2 import model_zoo
# from detectron2.engine import DefaultPredictor
# from detectron2.config import get_cfg
# from detectron2.utils.visualizer import Visualizer
# from detectron2.data import MetadataCatalog, DatasetCatalog
import matplotlib.pyplot as plt
import tkinter
import matplotlib
import matplotlib.gridspec as gridspec
from scipy.spatial import distance
# matplotlib.use('TkAgg')
do_video = False
video_name = 'images/output.avi'
use_cuda = True
use_rgb = True
######## Detectron2 imports end ########

STOP = 0
FORWARD = 3
BACKWARD = 15
LEFTWARD = 16
RIGHTWARD = 17
LEFT = 2
RIGHT = 1
UNREACHABLE = 5
EXPLORED = 6
DONE = 7
DOWN = 8
# PICKUP = 9
# OPEN = 10
# PUT = 11
# DROP = 12
UP = 13
# CLOSE = 14

# DT = 90
# STEP_SIZE = 0.25

POINT_COUNT = 1
# HORIZON_DT = 30
# STEEP_HORIZON_DT = 30
actions = { 
            # STOP: 'RotateLook', 
            STOP: 'MoveAhead', 
            LEFT: 'RotateLeft', 
            RIGHT: 'RotateRight', 
            FORWARD: 'MoveAhead',
            BACKWARD: 'MoveBack',
            LEFTWARD: 'MoveLeft',
            RIGHTWARD: 'MoveRight',
            DONE: 'Pass',
            DOWN: 'LookDown',
            UP: 'LookUp',
            # PICKUP: 'PickupObject',
            # OPEN: 'OpenObject',
            # CLOSE: 'CloseObject',
            # PUT: 'PutObject',
            # DROP: 'DropObject',
          }
actions_inv = { 
          'RotateLeft': LEFT, 
          'RotateRight': RIGHT, 
          'MoveAhead': FORWARD,
          'MoveBack': BACKWARD,
          'LookDown': DOWN,
          'LookUp': UP,
          'MoveLeft':LEFTWARD,
          'MoveRight':RIGHTWARD,
          'MoveBack':BACKWARD,
        }
# params = { 
#            # STOP: {'rotation': 0}, 
#            STOP: {'degrees': 0.1}, 
#            LEFT: {'degrees': DT}, 
#            RIGHT: {'degrees': DT}, 
#            FORWARD: {'moveMagnitude': STEP_SIZE},
#            BACKWARD: {'moveMagnitude': STEP_SIZE},
#            LEFTWARD: {'moveMagnitude': STEP_SIZE},
#            RIGHTWARD: {'moveMagnitude': STEP_SIZE},
#            DONE: {},
#            DOWN: {'degrees': HORIZON_DT},
#            UP: {'degrees': -HORIZON_DT},
#            PICKUP: {'objectId': None},
#            OPEN: {'objectId': None, 'amount': 0.99},
#            CLOSE: {'objectId': None, 'amount': 0.99},
#            PUT: {'objectId': None, 'receptacleObjectId': None},
#            DROP: {'objectId': None},
#          }

class Explore():
    def __init__(self, obs, goal, bounds, z=[0.15, 2.0], init_pos = None, keep_head_down=False, keep_head_straight=False, dist_thresh=0.25, search_pitch_explore=False, run_path=None, init_rot=None):
        # obs.STEP_SIZE - 0.25
        # obs.DT = 90
        # obs.HORIZON_DT = 30

        self.run_path = run_path

        self.open_mode = False
        self.last_place = True
        self.movement_check = [0, 0]
        self.delta_move_check = 0
        self.omega_move_check = 0
        self.prev_move_act = [0, 0]

        self.params = { 
           # STOP: {'rotation': 0}, 
           STOP: {'degrees': 0}, 
           LEFT: {'degrees': obs.DT}, 
           RIGHT: {'degrees': obs.DT}, 
           FORWARD: {'moveMagnitude': obs.STEP_SIZE},
           BACKWARD: {'moveMagnitude': obs.STEP_SIZE},
           LEFTWARD: {'moveMagnitude': obs.STEP_SIZE},
           RIGHTWARD: {'moveMagnitude': obs.STEP_SIZE},
           DONE: {},
           DOWN: {'degrees': obs.HORIZON_DT},
           UP: {'degrees': -obs.HORIZON_DT},
           # PICKUP: {'objectId': None},
           # OPEN: {'objectId': None, 'amount': 0.99},
           # CLOSE: {'objectId': None, 'amount': 0.99},
           # PUT: {'objectId': None, 'receptacleObjectId': None},
           # DROP: {'objectId': None},
        }

        self.actions_inv = { 
          'RotateLeft': LEFT, 
          'RotateRight': RIGHT, 
          'MoveAhead': FORWARD,
          'MoveBack': BACKWARD,
          'LookDown': DOWN,
          'LookUp': UP,
          'pass': DONE,
        }

        self.DT = obs.DT
        self.STEP_SIZE = obs.STEP_SIZE
        self.HORIZON_DT = obs.HORIZON_DT
        self.head_tilt = obs.head_tilt_init

        self.keep_head_down = keep_head_down
        self.keep_head_straight = keep_head_straight
        self.num_down_explore = 1
        self.num_down_nav = 1
        self.init_down = 1
        self.do_init_down = False
        self.init_on = True
        self.search_pitch_explore = False #search_pitch_explore
        self.goal_loc_viz = None

        self.action_fail_rearr = True

        self.do_visualize = False
        self.step_count = 0
        self.goal = goal
        self.cover_iters = 30
        # logging.error(self.goal.description)
        self.failed_action_count = 0

        self.past_actions = []
        self.loop_stuck = False

        self.reorient_head_flag = False
        self.pick_flag = True
        
        self.rng = np.random.RandomState(0)
        self.fmm_dist = np.zeros((1,1))
        self.acts = iter(())
        self.acts_og = iter(())
        self.explored = False
        # self.selem = skimage.morphology.disk(4) #self.mapper.resolution / self.mapper.resolution)
        # map_size = 20 #12 # 25
        # resolution = 0.05
        self.map_size = 13 # max scene bounds for Ai2thor is ~11.5 #12 # 25
        self.resolution = 0.02
        self.max_depth = 200. # 4. * 255/25.
        self.dist_thresh = dist_thresh # initial dist_thresh
        self.add_obstacle_if_action_fail = True

        self.text_to_add = "None"

        self.updated_goal_pos_map = None

        self.switch_pose = 0
        self.move_sign_x = 0
        self.move_sign_z = 0
        self.set_delta_x = False
        self.set_delta_z = False
        self.pos_world_actual = [0, 0]
        self.head_down_curr = False

        self.not_reachable = False

        # max_depth = 5. * 255/25.
        if True:
            # self.selem = skimage.morphology.disk(8) #self.mapper.resolution / self.mapper.resolution)
            # self.mapper_dilation = 1
            # # self.loc_on_map_selem = skimage.morphology.disk(13)
            # self.loc_on_map_selem = skimage.morphology.disk(2)
            self.selem = skimage.morphology.disk(int(8*(0.02/self.resolution))) #self.mapper.resolution / self.mapper.resolution)
            self.mapper_dilation = 1
            # self.loc_on_map_selem = skimage.morphology.disk(13)
            loc_on_map_size = int(np.floor(self.STEP_SIZE/self.resolution/2))#+5
            self.loc_on_map_selem = np.ones((loc_on_map_size*2+1, loc_on_map_size*2+1)).astype(bool)
            # self.loc_on_map_selem = skimage.morphology.disk(int(6*(0.02/self.resolution)))
        else:
            self.selem = skimage.morphology.disk(5) #self.mapper.resolution / self.mapper.resolution)
            self.mapper_dilation = 20
            self.loc_on_map_selem = skimage.morphology.disk(2)
        # skimage.morphology.square(2) #self.mapper.resolution / self.mapper.resolution)
        self.unexplored_area = np.inf
        self.next_target = 0
        self.opened = []
        self._setup_execution(goal)

        # Default initial position (have camera height)
        # if init_pos is not None:
        #     print("setting position : ", init_pos)
        #     self.position = {'x': init_pos['x'],
        #                      'y': obs.camera_height,
        #                      'z': init_pos['z']}
        # else:
        self.position = {'x': 0,
                         'y': obs.camera_height,
                         'z': 0}
        self.rotation = init_rot
        self.init_rotation = init_rot
        self.prev_act_id = None
        self.obstructed_actions = []
        self.success = False
        self.point_goal = None
        self.action_failed = False

        self.z_bins = z #[0.15, 2.3] # 1.57 is roughly camera height

        self.agent_stuck = False
        self.alternate_stuck_action = False

        ar = obs.camera_aspect_ratio
        vfov = obs.camera_field_of_view*np.pi/180
        focal = ar[1]/(2*math.tan(vfov/2))
        fov = abs(2*math.atan(ar[0]/(2*focal))*180/np.pi)
        self.sc = 1. #255./25. #1 #57.13
        fov, h, w = fov, ar[1], ar[0]
        
        C = get_camera_matrix(w, h, fov=fov)
        self.bounds = bounds # aithor bounds
        self.mapper = Mapper(C, self.sc, self.position, self.map_size, self.resolution,
                                max_depth=self.max_depth, z_bins=self.z_bins,
                                loc_on_map_selem = self.loc_on_map_selem,
                                bounds=self.bounds)
        self.state_xy_prev = self.mapper.origin_map
        self.dist_delta = {'x': 0, 'z': 0}
        self.rotate_delta = 0
        self.delta_steps = 0
        self.y_loc = None
        self.look_down_twice = False
        self.n_steps_to_rotate = None
        self.orient_DT = 10
        self.rotate_left = None
        self.dist_agent = 0

        self.video_ind = 1

        self.invert_pitch = True # invert pitch when fetching roation matrix? 
        self.camX0_T_origin = self.get_camX0_T_camX(get_camX0_T_origin=True)
        self.camX0_T_origin = self.safe_inverse_single(self.camX0_T_origin)

        self.rearr_mode = False
        self.place_loc_rend = False
        self.failed_action_place_count = 0
        self.move_head_back_after_place = False

        '''
        # setup trophy detector
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        if not use_rgb:
            cfg.MODEL.PIXEL_MEAN = [100.0, 100.0, 100.0, 100.0]
            cfg.MODEL.PIXEL_STD = [1.0, 1.0, 1.0, 1.0]
        cfg.MODEL.WEIGHTS = retrieval_detector_path
        if use_cuda:
            cfg.MODEL.DEVICE='cuda'
        else:
            cfg.MODEL.DEVICE='cpu'
        cfg.DATASETS.TEST = ("val",) 
        thing_classes = ['trophy', 'box_closed', 'box_opened']
        d = "train"
        DatasetCatalog.register("val", lambda d=d: val_dataset_function())
        MetadataCatalog.get("val").thing_classes = thing_classes
        self.trophy_cfg = cfg
        self.detector = DefaultPredictor(cfg)
        '''

    def get_traversible_map(self):
        return self.mapper.get_traversible_map(
                          self.selem, 1,loc_on_map_traversible=True)

    def get_explored_map(self):
        return self.mapper.get_explored_map(self.selem, 1)

    def get_map(self):
        return self.mapper.map

    def _setup_execution_new(self, goal):
        fn = lambda: self._init_fn()
        self.execution.put(fn)

    def _setup_execution(self, goal):
        self.execution = LifoQueue(maxsize=200)
        if self.goal.category == 'point_nav':
            fn = lambda: self._point_goal_fn_assigned(np.array([self.goal.targets[0], self.goal.targets[1]]), explore_mode=False)
            self.execution.put(fn)
        elif self.goal.category == 'cover':
            self.exploring = True
            fn = lambda: self._cover_fn(self.goal.targets[0], 20, 1, 1)
            self.execution.put(fn)
        elif self.goal.category == 'retrieval':
            fn = lambda: self._retrieve_fn(self.goal.targets[0]) 
            self.execution.put(fn)
        elif self.goal.category == 'traversal':
            fn = lambda: self._traverse_fn(self.goal.targets[0], 1, 1) 
            self.execution.put(fn)
        elif self.goal.category == 'transferral-top':
            fn = lambda: self._put_fn(self.goal.targets[0], self.goal.targets[1]) 
            self.execution.put(fn)
            fn = lambda: self._traverse_fn(self.goal.targets[1], 1, 1) 
            self.execution.put(fn)
            fn = lambda: self._retrieve_fn(self.goal.targets[0]) 
            self.execution.put(fn)
        elif self.goal.category == 'transferral-next':
            fn = lambda: self._drop_fn()
            self.execution.put(fn)
            fn = lambda: self._traverse_fn(self.goal.targets[1], 1, 1) 
            self.execution.put(fn)
            fn = lambda: self._retrieve_fn(self.goal.targets[0]) 
            self.execution.put(fn)
        else:
            assert(False), f'Incorrext goal category: {self.goal.category}'
        
        
        fn = lambda: self._init_fn()
        self.execution.put(fn)
    
    def _drop_fn(self):
        yield actions[DROP]

    '''
    # Andy: not needed for this task
    def _put_fn(self, uuid, to_uuid):
        yield actions[PUT], {'objectId': uuid, 'receptacleObjectId': to_uuid}
        if self.obs.return_status == 'SUCCESSFUL':
            return

        yield actions[DOWN], {'horizon': STEEP_HORIZON_DT}
        yield actions[PUT], {'objectId': uuid, 'receptacleObjectId': to_uuid}
        if self.obs.return_status == 'SUCCESSFUL':
            yield actions[DOWN], {'horizon': -STEEP_HORIZON_DT}
            return
        yield actions[DOWN], {'horizon': -STEEP_HORIZON_DT}

        yield actions[DOWN], {'horizon': HORIZON_DT}
        yield actions[PUT], {'objectId': uuid, 'receptacleObjectId': to_uuid}
        if self.obs.return_status == 'SUCCESSFUL':
            yield actions[DOWN], {'horizon': -HORIZON_DT}
            return
        yield actions[DOWN], {'horizon': -HORIZON_DT}
        
        yield FORWARD
        yield actions[PUT], {'objectId': uuid, 'receptacleObjectId': to_uuid}
        if self.obs.return_status == 'SUCCESSFUL':
            return
        yield actions[DOWN], {'horizon': HORIZON_DT}
        yield actions[PUT], {'objectId': uuid, 'receptacleObjectId': to_uuid}
        if self.obs.return_status == 'SUCCESSFUL':
            yield actions[DOWN], {'horizon': -HORIZON_DT}
            return
        yield actions[DOWN], {'horizon': -HORIZON_DT}
    '''

    def _traverse_fn(self, uuid, semantic_size_threshold, morph_disk_size):
        fn = lambda: self._object_goal_fn(uuid, 10, semantic_size_threshold, morph_disk_size)
        self.execution.put(fn)
        fn = lambda: self._explore_fn(uuid, 20, semantic_size_threshold, morph_disk_size)
        self.execution.put(fn)
        fn = lambda: self._object_goal_fn(uuid, 10, semantic_size_threshold, morph_disk_size)
        self.execution.put(fn)
        

    def _retrieve_fn(self, uuid):
        fn = lambda: self._pickup_fn(uuid)
        self.execution.put(fn)
        
        fn = lambda: self._traverse_fn(uuid, 0, 0)
        self.execution.put(fn)

    def _init_fn(self):

        self.init_on = True
        self.exploring = False


        # # yield DOWN
        # # yield DOWN




        # for i in range(int(60/self.HORIZON_DT)):
        #     yield DOWN




        # for i in range(int(360/self.DT)):
        #     yield LEFT
        # for i in range(int(60/self.HORIZON_DT)):
        #     yield UP
        # # yield UP
        # # yield UP

        # for i in range(int(45/self.HORIZON_DT)):
        #     yield DOWN
        # for i in range(int(360/self.DT)):
        #     yield LEFT
        
        # if not self.keep_head_down:
        #     for i in range(int(45/self.HORIZON_DT)):
        #         yield UP
        # num_head_down = 1
        # print("HERE2")
        

        # for i in range(self.num_down_explore):
        #     yield DOWN
        for i in range(int(360/self.DT)):
            yield LEFT
        
        # if not self.keep_head_down:
        #     for i in range(self.num_down_explore):
        #         yield UP




        # for i in range(int(60/self.HORIZON_DT)):
        #     yield UP




        self.init_on = False
        self.exploring = True
    
    def _get_target_object(self, obs, target_object, match_metric='id'):
        # Returns the distance to the desired object.
        r = None
        for o in obs.object_list:
            if match_metric == 'id':
                if o.uuid == target_object['id']:
                    r = o
        return r
    
    def _get_action(self, ID):
        if type(ID) == int:
            return actions[ID], self.params[ID]
        else:
            return ID[0], ID[1]

    def _update_coords_lvl1(self, masks, pred_classes, pred_scores):
        self.box_opened_mask_s = []
        self.box_closed_mask_s = []
        self.trophy_mask = None

        if masks.shape[0] == 0:
            # no detections, return
            return

        for idx, mask in enumerate(masks):
            # keep only high confidence detections
            if pred_scores[idx].item() < 0.9:
                continue

            # to numpy
            mask = mask.cpu().numpy()

            cur_class = pred_classes[idx].item()
            if cur_class == 0:
                self.trophy_mask = mask
            elif cur_class == 2:
                self.box_opened_mask_s.append(mask)
            else:
                self.box_closed_mask_s.append(mask)

    def _update_coords_lvl2(self, masks, pred_classes, pred_scores, semantic_map):
        self.box_opened_mask_s = []
        self.box_closed_mask_s = []
        self.trophy_mask = None

        if masks.shape[0] == 0:
            # no detections, return
            return

        for idx, mask in enumerate(masks):
            # keep only high confidence detections:
            if pred_scores[idx].item() < 0.9:
                continue

            # to numpy
            mask = mask.cpu().numpy()
            mask = mask.reshape(*semantic_map.shape[:2],1)
            masked_sem = mask * semantic_map

            unique_colors, counts = np.unique(masked_sem.reshape(-1, 3), return_counts = True, axis=0)

            max_counts = 0
            max_color = None
            for cid in range(len(counts)):
                color = unique_colors[cid]
                if color.sum() == 0:
                    continue
                if counts[cid] > max_counts:
                    max_counts = counts[cid]
                    max_color = color

            # use the mask with max_counts
            mask = (semantic_map == max_color.reshape(1,1,3)).sum(2) == 3

            cur_class = pred_classes[idx].item()
            if cur_class == 0:
                self.trophy_mask = mask
            elif cur_class == 2:
                self.box_opened_mask_s.append(mask)
            else:
                self.box_closed_mask_s.append(mask)
                
    def _update_coords_gt(self, obs):
        self.box_opened_mask_s = []
        self.box_closed_mask_s = []
        self.box_opened_uuid_s = []
        self.box_closed_uuid_s = []
        self.trophy_mask = None
    
        semantic_map = np.array(obs.object_mask_list[0])
        objects = obs.object_list
        
        for obj in objects:
            color = np.array([obj.color['r'], obj.color['g'], obj.color['b']]).reshape(1,1,3)
            mask = (semantic_map == color).sum(2) == 3
            if mask.sum() == 0:
                continue
            
            if 'trophy' in obj.uuid:
                self.trophy_mask = mask
            elif 'box' in obj.uuid:
                if obj.uuid in self.opened:
                    self.box_opened_mask_s.append(mask)
                    self.box_opened_uuid_s.append(obj.uuid)
                else:
                    self.box_closed_mask_s.append(mask)
                    self.box_closed_uuid_s.append(obj.uuid)

    def set_point_goal(self, ind_i, ind_j, dist_thresh=0.25, explore_mode=False, search_mode=False, search_mode_reduced=False):
        self.exploring = False
        self.acts = iter(())
        self.acts_og = iter(())
        self.dists = []
        self.execution = LifoQueue(maxsize=200)
        self.point_goal = [ind_i, ind_j]
        self.dist_thresh = dist_thresh
        self.obstructed_actions = []
        # fn = lambda: self._point_goal_fn(np.array([ind_j, ind_i]), dist_thresh=dist_thresh, explore_mode=explore_mode)
        fn = lambda: self._point_goal_fn_assigned(np.array([ind_j, ind_i]), dist_thresh=dist_thresh, explore_mode=False, search_mode=search_mode, search_mode_reduced=search_mode_reduced)
        self.execution.put(fn)

    def add_observation(self, obs, action, add_obs=True):
        self.step_count += 1
        self.obs = obs

        act_id = actions_inv[action]

    
        if obs.return_status == 'SUCCESSFUL': # and self.prev_act_id is not None:
            # self.obstructed_actions = []
            if 'Rotate' in actions[act_id]:
                if 'Left' in actions[act_id]:
                    self.rotation -= self.params[act_id]['degrees']
                else:
                    self.rotation += self.params[act_id]['degrees']
                self.rotation %= 360
            elif 'Move' in actions[act_id]:
                if act_id == FORWARD:
                    self.position['x'] += np.sin(self.rotation/180*np.pi)*self.params[act_id]['moveMagnitude']
                    self.position['z'] += np.cos(self.rotation/180*np.pi)*self.params[act_id]['moveMagnitude']
                elif act_id == BACKWARD:
                    self.position['x'] -= np.sin(self.rotation/180*np.pi)*self.params[act_id]['moveMagnitude']
                    self.position['z'] -= np.cos(self.rotation/180*np.pi)*self.params[act_id]['moveMagnitude']
                elif act_id == LEFTWARD:
                    self.position['x'] -= np.cos(self.rotation/180*np.pi)*self.params[act_id]['moveMagnitude']
                    self.position['z'] += np.sin(self.rotation/180*np.pi)*self.params[act_id]['moveMagnitude']
                elif act_id == RIGHTWARD:
                    self.position['x'] += np.cos(self.rotation/180*np.pi)*self.params[act_id]['moveMagnitude']
                    self.position['z'] -= np.sin(self.rotation/180*np.pi)*self.params[act_id]['moveMagnitude']
            elif 'Look' in actions[act_id]:
                self.head_tilt += self.params[act_id]['degrees']
        elif obs.return_status == 'OBSTRUCTED' and act_id is not None:
            print("ACTION FAILED.")
            # if self.add_obstacle_if_action_fail:
            #     self.mapper.add_obstacle_in_front_of_agent()
            #     if self.point_goal is not None:
            #         self.execution = LifoQueue(maxsize=200)
            #         ind_i, ind_j = self.point_goal
            #         # fn = lambda: self._point_goal_fn(np.array([ind_j, ind_i]), dist_thresh=dist_thresh, explore_mode=explore_mode)
            #         fn = lambda: self._point_goal_fn_assigned(np.array([ind_j, ind_i]), dist_thresh=self.dist_thresh)
            #         self.execution.put(fn)
            prev_len = len(self.obstructed_actions)
            if prev_len>4000:
                pass
            else:
                # print(prev_len)
                for idx in range(prev_len):
                    obstructed_acts = self.obstructed_actions[idx]
                    self.obstructed_actions.append(obstructed_acts+[act_id])
                self.obstructed_actions.append([act_id])
        # head_tilt = obs.head_tilt
        return_status = obs.return_status
        # print("Step {0}, position {1} / {2}, rotation {3}".format(self.step_count, self.position, obs.position, self.rotation))
        rgb = np.array(obs.image_list[-1])
        depth = np.array(obs.depth_map_list[-1])

        self.mapper.add_observation(self.position, 
                                    self.rotation, 
                                    -self.head_tilt, 
                                    depth, add_obs=add_obs)
        if obs.return_status == 'OBSTRUCTED' and 'Move' in actions[act_id]: # and act_id is not None:
            # print("Ohhhhoooooonnooooooooooooooooonnononononooo")
            # print("ACTION FAILED.")
            if self.add_obstacle_if_action_fail:
                self.mapper.add_obstacle_in_front_of_agent(self.selem)
                if self.point_goal is not None:

                    # self.mapper.dilate_obstacles_around_agent(self.selem)
                    self.execution = LifoQueue(maxsize=200)
                    self.point_goal = self.get_clostest_reachable_map_pos(self.point_goal)
                    ind_i, ind_j = self.point_goal
                    # fn = lambda: self._point_goal_fn(np.array([ind_j, ind_i]), dist_thresh=dist_thresh, explore_mode=explore_mode)
                    fn = lambda: self._point_goal_fn_assigned(np.array([ind_j, ind_i]), dist_thresh=self.dist_thresh)
                    self.execution.put(fn)
        # act_id = self.actions_inv[action]
        # self.prev_act_id = act_id

    def add_change_to_map(self, depth, masks):
        self.mapper.add_change_to_map(position=self.position,
                                      rotation=self.rotation,
                                      elevation=-self.head_tilt,
                                      depth=depth,
                                      masks=masks)

    def get_agent_position_camX0(self):
        '''
        Get agent position in camX0 (first position) reference frame
        '''
        pos = torch.from_numpy(np.array(list(self.position.values())))
        pos = self.apply_4x4(self.camX0_T_origin.unsqueeze(0), pos.unsqueeze(0).unsqueeze(0)).squeeze().numpy()
        return pos

    def get_agent_rotations_camX0(self):
        '''
        Get agent rotation in camX0 (first position) reference frame
        '''
        yaw = self.rotation
        pitch = self.head_tilt
        if self.invert_pitch:
            pitch = -pitch
        roll = 0
        return yaw, pitch, roll

    def get_camX0_T_camX(self, get_camX0_T_origin=False):
        '''
        Get transformation matrix between first position (camX0) and current position (camX)
        '''
        position = np.array(list(self.position.values()))
        # position = position[[2,1,0]]
        # position[0] = position[0] # invert x
        # position[2] = position[2] # invert z
        # in aithor negative pitch is up - turn this on if need the reverse
        head_tilt = self.head_tilt
        if self.invert_pitch:
            head_tilt = -head_tilt
        # print("Head tilt", head_tilt)
        rx = np.radians(head_tilt) #np.radians(event.metadata["agent"]["cameraHorizon"]) # pitch
        rotation = self.rotation
        if rotation >= 180:
            rotation = rotation - 360
        if rotation < -180:
            rotation = 360 + rotation
        # print("ROT PRED", rotation)
        ry = np.radians(rotation) #np.radians(rotation[1]) # yaw
        rz = 0. # roll is always 0
        rotm = self.eul2rotm_py(np.array([rx]), np.array([ry]), np.array([rz]))
        origin_T_camX = np.eye(4)
        origin_T_camX[0:3,0:3] = rotm
        origin_T_camX[0:3,3] = position
        # if position[2]>0:
        #     st()
        # if rotation==180:
        #     position[0] = 
        origin_T_camX = torch.from_numpy(origin_T_camX)
        if get_camX0_T_origin:
            camX0_T_camX = origin_T_camX
        else:
            camX0_T_camX = torch.matmul(self.camX0_T_origin, origin_T_camX)
        
            # if camX_T_origin[1,3]>0.1:
            # print(camX0_T_camX[1,3])
        # print(position)
        # print(np.degrees(np.array([rx])), np.degrees(np.array([ry])), np.degrees(np.array([rz])))
        return camX0_T_camX

    def safe_inverse_single(self, a):
        r, t = self.split_rt_single(a)
        t = t.view(3,1)
        r_transpose = r.t()
        inv = torch.cat([r_transpose, -torch.matmul(r_transpose, t)], 1)
        bottom_row = a[3:4, :] # this is [0, 0, 0, 1]
        # bottom_row = torch.tensor([0.,0.,0.,1.]).view(1,4) 
        inv = torch.cat([inv, bottom_row], 0)
        return inv

    def split_rt_single(self, rt):
        r = rt[:3, :3]
        t = rt[:3, 3].view(3)
        return r, t

    def apply_4x4(self, RT, xyz):
        B, N, _ = list(xyz.shape)
        ones = torch.ones_like(xyz[:,:,0:1])
        xyz1 = torch.cat([xyz, ones], 2)
        xyz1_t = torch.transpose(xyz1, 1, 2)
        # this is B x 4 x N
        xyz2_t = torch.matmul(RT, xyz1_t)
        xyz2 = torch.transpose(xyz2_t, 1, 2)
        xyz2 = xyz2[:,:,:3]
        return xyz2

    def eul2rotm_py(self, rx, ry, rz):
        # inputs are shaped B
        # this func is copied from matlab
        # R = [  cy*cz   sy*sx*cz-sz*cx    sy*cx*cz+sz*sx
        #        cy*sz   sy*sx*sz+cz*cx    sy*cx*sz-cz*sx
        #        -sy            cy*sx             cy*cx]
        rx = rx[:,np.newaxis]
        ry = ry[:,np.newaxis]
        rz = rz[:,np.newaxis]
        # these are B x 1
        sinz = np.sin(rz)
        siny = np.sin(ry)
        sinx = np.sin(rx)
        cosz = np.cos(rz)
        cosy = np.cos(ry)
        cosx = np.cos(rx)
        r11 = cosy*cosz
        r12 = sinx*siny*cosz - cosx*sinz
        r13 = cosx*siny*cosz + sinx*sinz
        r21 = cosy*sinz
        r22 = sinx*siny*sinz + cosx*cosz
        r23 = cosx*siny*sinz - sinx*cosz
        r31 = -siny
        r32 = sinx*cosy
        r33 = cosx*cosy
        r1 = np.stack([r11,r12,r13],axis=2)
        r2 = np.stack([r21,r22,r23],axis=2)
        r3 = np.stack([r31,r32,r33],axis=2)
        r = np.concatenate([r1,r2,r3],axis=1)
        return r

    def set_pos_world(self, pos):
        self.pos_world_actual = pos

    def reinit_act_queue(self):
        print("reinitializing the execution queue...")
        self.execution = LifoQueue(maxsize=200)

    def reorient_head_flag_fn(self, pick_flag):
        self.reorient_head_flag = True
        self.pick_flag = pick_flag

    def act(self, obs, fig=None, point_goal=False, goal_loc=None, goal_loc_map=None, add_obs=True, object_masks=[], held_obj_depth=100., y_loc=None):
        # print("Exploring?", self.exploring)
        self.step_count += 1
        self.obs = obs

        if do_video:
            if self.step_count == 1:
                # self.video_writer = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'FMP4'), 4, (480,480))
                rand_int = np.random.randint(100)
                video_name = f'images/output{rand_int}.avi'
                self.video_writer = cv2.VideoWriter(video_name, 0, 4, (480,480))
            rgb = np.array(obs.image_list[-1]).astype(np.uint8)
            bgr = rgb[:,:,[2,1,0]]
            self.video_writer.write(bgr)
        if self.step_count == 1:
            ar = obs.camera_aspect_ratio
            vfov = obs.camera_field_of_view*np.pi/180
            focal = ar[1]/(2*math.tan(vfov/2))
            fov = abs(2*math.atan(ar[0]/(2*focal))*180/np.pi)
            fov, h, w = fov, ar[1], ar[0]
            C = get_camera_matrix(w, h, fov=fov)
            self.mapper = Mapper(C, self.sc, self.position, self.map_size, self.resolution,
                                 max_depth=self.max_depth, z_bins=self.z_bins,
                                 loc_on_map_selem = self.loc_on_map_selem,
                                 bounds=self.bounds)
        else:
            if obs.return_status == 'SUCCESSFUL' and self.prev_act_id is not None:
                # self.obstructed_actions = []
                if 'Rotate' in actions[self.prev_act_id]:
                    if 'Left' in actions[self.prev_act_id]:
                        self.rotation -= self.params[self.prev_act_id]['degrees']
                    else:
                        self.rotation += self.params[self.prev_act_id]['degrees']
                    self.rotation %= 360
                elif 'Move' in actions[self.prev_act_id]:
                    if self.prev_act_id == FORWARD:
                        move = [np.sin(self.rotation/180*np.pi)*self.params[self.prev_act_id]['moveMagnitude'],
                                np.cos(self.rotation/180*np.pi)*self.params[self.prev_act_id]['moveMagnitude']]
                        self.position['x'] += np.sin(self.rotation/180*np.pi)*self.params[self.prev_act_id]['moveMagnitude']
                        self.position['z'] += np.cos(self.rotation/180*np.pi)*self.params[self.prev_act_id]['moveMagnitude']
                    elif self.prev_act_id == BACKWARD:
                        move = [-np.sin(self.rotation/180*np.pi)*self.params[self.prev_act_id]['moveMagnitude'],
                                -np.cos(self.rotation/180*np.pi)*self.params[self.prev_act_id]['moveMagnitude']]
                        self.position['x'] -= np.sin(self.rotation/180*np.pi)*self.params[self.prev_act_id]['moveMagnitude']
                        self.position['z'] -= np.cos(self.rotation/180*np.pi)*self.params[self.prev_act_id]['moveMagnitude']
                    elif self.prev_act_id == LEFTWARD:
                        move = [-np.cos(self.rotation/180*np.pi)*self.params[self.prev_act_id]['moveMagnitude'],
                                np.sin(self.rotation/180*np.pi)*self.params[self.prev_act_id]['moveMagnitude']]
                        self.position['x'] -= np.cos(self.rotation/180*np.pi)*self.params[self.prev_act_id]['moveMagnitude']
                        self.position['z'] += np.sin(self.rotation/180*np.pi)*self.params[self.prev_act_id]['moveMagnitude']
                    elif self.prev_act_id == RIGHTWARD:
                        move = [np.cos(self.rotation/180*np.pi)*self.params[self.prev_act_id]['moveMagnitude'],
                                -np.sin(self.rotation/180*np.pi)*self.params[self.prev_act_id]['moveMagnitude']]
                        self.position['x'] += np.cos(self.rotation/180*np.pi)*self.params[self.prev_act_id]['moveMagnitude']
                        self.position['z'] -= np.sin(self.rotation/180*np.pi)*self.params[self.prev_act_id]['moveMagnitude']

                    if self.switch_pose == 0:
                        if ((self.position['x'] != 0 and self.pos_world_actual[1] != 0) or
                                (self.position['z'] != 0 and self.pos_world_actual[0] != 0)):
                            self.switch_pose = 1
                        else:
                            self.switch_pose = -1

                    if self.position['x'] != 0 and self.move_sign_x == 0:
                        if self.position['x'] * self.pos_world_actual[0] < 0:
                            self.move_sign_x = -1
                        else:
                            self.move_sign_x = 1

                    if self.position['z'] != 0 and self.move_sign_z == 0:
                        if self.position['z'] * self.pos_world_actual[1] < 0:
                            self.move_sign_z = -1
                        else:
                            self.move_sign_z = 1

                elif 'Look' in actions[self.prev_act_id]:
                    self.head_tilt += self.params[self.prev_act_id]['degrees']

                if 'Move' in actions[self.prev_act_id]:
                    self.check_agent_stuck(prev_act_id=self.prev_act_id, move=move)
                else:
                    self.check_agent_stuck()

            elif obs.return_status == 'OBSTRUCTED' and self.prev_act_id is not None:
                print("ACTION FAILED.")
                self.action_fail_rearr = True
                # check if the agent is stuck, if so dilate the obstacle map
                self.check_agent_stuck()

                if self.place_loc_rend:
                    self.failed_action_place_count += 1
                    if self.failed_action_place_count == 5:
                        # return object to picked position
                        action, param = self._get_action(DONE)
                        return action, param

                # if not self.rearr_mode:
                prev_len = len(self.obstructed_actions)
                if prev_len>4000:
                    pass
                else:
                    # print(prev_len)
                    for idx in range(prev_len):
                        obstructed_acts = self.obstructed_actions[idx]
                        self.obstructed_actions.append(obstructed_acts+[self.prev_act_id])
                    self.obstructed_actions.append([self.prev_act_id])
                self.check_agent_stuck(obstructed=True)

        return_status = obs.return_status

        rgb = np.array(obs.image_list[-1])
        depth = np.array(obs.depth_map_list[-1])

        if self.step_count != 1:
            state_xy = self.mapper.get_position_on_map()
            state_xy = state_xy.astype(np.int32)
            self.state_xy_prev = state_xy

        self.mapper.add_observation(self.position,
                                    self.rotation,
                                    -self.head_tilt,
                                    depth,
                                    add_obs=add_obs)
        if self.reorient_head_flag:
            self.reorient_head_flag = False
            fn = lambda: self.reorient_head(pick=self.pick_flag)
            self.execution.put(fn)

        elif point_goal:
            self.rearr_mode = True

            if self.add_obstacle_if_action_fail and self.action_fail_rearr:
                self.action_fail_rearr = False
                self.mapper.add_obstacle_in_front_of_agent(self.selem)

            if goal_loc_map is None and goal_loc is None:
                raise ValueError("Goal location is not set ...")
            else:
                self.y_loc = y_loc
                if goal_loc is not None:
                    self.goal_to_reach = goal_loc
                    goal_pos_map_ = self.mapper.get_goal_position_on_map(goal_loc)
                    goal_pos_map = np.array([goal_pos_map_[0], goal_pos_map_[1]])
                elif goal_loc_map is not None:
                    self.goal_to_reach = goal_loc_map
                    goal_pos_map = goal_loc_map
                #goal_pos_map = goal_loc_map
                self.goal_pos_map = goal_pos_map
                goal_loc_cell = np.array([goal_pos_map[0], goal_pos_map[1]])  # TODO: changed
                traversible = self.mapper.get_traversible_map(self.selem, POINT_COUNT, loc_on_map_traversible=True)
                planner = FMMPlanner(traversible, 360 // self.DT, int(self.STEP_SIZE / self.mapper.resolution),
                                     self.obstructed_actions)

                reachable = planner.set_goal(goal_loc_cell)
                state_xy = self.mapper.get_position_on_map()
                state_xy = state_xy.astype(np.int32)
                state_theta = self.mapper.get_rotation_on_map() + np.pi / 2

                if not reachable[state_xy[1], state_xy[0]]:
                    indi, indj = self.get_clostest_reachable_within_delta_map_pos([goal_loc_cell[1], goal_loc_cell[0]]) # TODO: changed
                    goal_loc_cell = np.array([indj, indi], dtype=np.int32)
                    self.updated_goal_pos_map = [indj, indi]
                else:
                    self.updated_goal_pos_map = None

                # if obs.return_status == 'OBSTRUCTED' and 'Move' in actions[self.prev_act_id]:
                #     self.action_failed = True
                #     if self.add_obstacle_if_action_fail:
                #         self.mapper.add_obstacle_in_front_of_agent(self.selem)

                fn = lambda: self._point_goal_fn_assigned(goal_loc_cell, dist_thresh=self.dist_thresh)
                self.execution.put(fn)

        elif obs.return_status == 'OBSTRUCTED' and 'Move' in actions[self.prev_act_id]: # and self.prev_act_id is not None:
            # print("ACTION FAILED.")
            self.action_failed = True
            if self.add_obstacle_if_action_fail:
                self.mapper.add_obstacle_in_front_of_agent(self.selem)
                if self.point_goal is not None:

                    # self.mapper.dilate_obstacles_around_agent(self.selem)
                    self.execution = LifoQueue(maxsize=200)
                    # recompute closest navigable point
                    self.point_goal = self.get_clostest_reachable_map_pos(self.point_goal)
                    ind_i, ind_j = self.point_goal
                    # fn = lambda: self._point_goal_fn(np.array([ind_j, ind_i]), dist_thresh=dist_thresh, explore_mode=explore_mode)
                    fn = lambda: self._point_goal_fn_assigned(np.array([ind_j, ind_i]), dist_thresh=self.dist_thresh, explore_mode=True, search_mode=True)
                    self.execution.put(fn)
        else:
            self.action_failed = False

        if self.acts is None:
            act_id = None
        else:
            act_id = next(self.acts, None)
        if act_id is None:
            act_id = DONE
            # num_times = 0
            while self.execution.qsize() > 0:
                op = self.execution.get()
                self.acts = op()
                if self.acts is not None:
                    act_id = next(self.acts, None)
                    if act_id is not None:
                        break
        if act_id is None:
            act_id = DONE

        if type(act_id) != int:
            if act_id[0] in actions_inv:
                act_id = actions_inv[act_id[0]]
                self.prev_act_id = act_id
            else:
                self.prev_act_id = None
                return act_id

        vis = True
        if vis:
            rgb = np.array(obs.image_list[-1])
            self.vis_map(rgb, act_id)

        self.prev_act_id = act_id

        if len(self.past_actions) < 12:
            self.past_actions.append(act_id)
        else:
            del self.past_actions[0]
            self.past_actions.append(act_id)

        action, param = self._get_action(act_id)

        if self.rearr_mode:
            state_xy = self.mapper.get_position_on_map()
            state_xy = state_xy.astype(np.int32)
            state_theta = self.mapper.get_rotation_on_map()
            dx = np.cos(state_theta + np.pi / 2)
            dy = np.sin(state_theta + np.pi / 2)
            v1 = np.array([dx, dy])

            gx = self.goal_pos_map[0] - state_xy[0]
            gy = self.goal_pos_map[1] - state_xy[1]
            v2 = np.array([gx, gy])

            dot = dx * gx + dy * gy
            cross = dx * gy - dy * gx

            phi = np.arccos(dot / (np.linalg.norm(v1) + np.linalg.norm(v2)))
            phi = phi * (180 / np.pi)
        return action, param

    def check_agent_stuck(self, obstructed=False, prev_act_id=None, move=None):
        # if prev_act_id is not None and move is not None:
        #     print("\n==========================================")
        #     print("dmc : ", self.delta_move_check)
        #     print("omc : ", self.omega_move_check)
        #     print("move : ", move)
        #     print("prev_move : ", self.prev_move_act)
        #     print("sum : ", move + self.prev_move_act)
        #     print("==========================================\n")
        if prev_act_id is not None and move is not None:
            if 'Move' in actions[self.prev_act_id]:
                self.movement_check += move
                if np.abs(np.sum(np.array(move) + np.array(self.prev_move_act))) < 0.1:
                    self.delta_move_check += 1
                else:
                    self.omega_move_check += 1
                self.prev_move_act = move
                if self.delta_move_check + self.omega_move_check > 10:
                    if self.delta_move_check > 7:
                        # Agent stuck
                        print("\n\nAgent Stuck in a loop...")
                        print("Dilating the obstacle map...")
                        print("Reinitializing action queue\n\n")
                        self.agent_stuck = True
                        self.mapper.dilate_obstacles_around_agent()
                        self.reinit_act_queue()
                        self.delta_move_check = 0
                        self.omega_move_check = 0
                    else:
                        self.delta_move_check = 0
                        self.omega_move_check = 0

        state_xy = self.mapper.get_position_on_map()
        state_xy = state_xy.astype(np.int32)

        dist = np.linalg.norm(state_xy - self.state_xy_prev)

        self.dist_agent += dist
        self.delta_steps += 1
        if self.delta_steps == 20:
            if self.dist_agent == 0:
                self.agent_stuck = True
                for i in range(4):
                    self.mapper.dilate_obstacles_around_agent()
                self.reinit_act_queue()
            self.dist_agent = 0
            self.delta_steps = 0

        if obstructed:
            if self.place_loc_rend:
                self.failed_action_count += 1

        # check if agent stuck in a loop
        if len(self.past_actions) == 12:
            if self.past_actions[:3] == self.past_actions[3:6] == self.past_actions[6:9] == self.past_actions[9:12]:
                if not self.past_actions[0] == self.past_actions[1] == self.past_actions[2]:
                    print("\n\n AGENT STUCK IN A LOOP \n\n")
                    for i in range(4):
                        self.mapper.dilate_obstacles_around_agent()
                    self.reinit_act_queue()
                    self._setup_execution(self.goal)



    def get_clostest_reachable_map_pos(self, map_pos):
        reachable = self._get_reachable_area()
        inds_i, inds_j = np.where(reachable)
        reachable_where = np.stack([inds_i, inds_j], axis=0)
        dist = abs(distance.cdist(np.expand_dims(map_pos, axis=0), reachable_where.T) - 0.2)
        argmin = np.argmin(dist)
        ind_i, ind_j = inds_i[argmin], inds_j[argmin]
        return ind_i, ind_j

    def get_clostest_reachable_within_delta_map_pos(self, map_pos, delta=0.08):
        reachable = self._get_reachable_area()
        inds_i, inds_j = np.where(reachable)
        reachable_where = np.stack([inds_i, inds_j], axis=0)
        dist = abs(distance.cdist(np.expand_dims(map_pos, axis=0), reachable_where.T)-delta)
        argmin = np.argmin(dist)
        ind_i, ind_j = inds_i[argmin], inds_j[argmin]
        return ind_i, ind_j
    def get_mapper_occ(self, obs, global_downscaling):
        # head_tilt = obs.head_tilt
        depth = np.array(obs.depth_map_list[-1])
        counts2, is_valids2, inds2 = self.mapper.get_occupancy_vars(self.position, 
                                    self.rotation, 
                                    -self.head_tilt, 
                                    depth, global_downscaling)
        return counts2, is_valids2, inds2

    def add_indices(self, text):
        self.text_to_add = text

    def vis_map(self, rgb, act_id):
        fig = plt.figure()
        ax = []
        spec = gridspec.GridSpec(ncols=2, nrows=2,
                                 figure=fig, left=0., right=1., wspace=0.05, hspace=0.5)
        ax.append(fig.add_subplot(spec[0, 0]))
        # ax.append(fig.add_subplot(spec[:2, 3:]))
        ax.append(fig.add_subplot(spec[0, 1]))
        ax.append(fig.add_subplot(spec[1, 1]))
        ax.append(fig.add_subplot(spec[1, 0]))
        # ax.append(fig.add_subplot(spec[2:, 2:4]))
        # ax.append(fig.add_subplot(spec[2:, 4:6]))
        dd = '\n'.join(wrap(self.goal.description, 50))
        fig.suptitle(f"{self.step_count - 1}. {dd} act: {act_id}", fontsize=14)

        for a in ax:
            a.axis('off')

        m_vis = np.invert(self.mapper.get_traversible_map(
            self.selem, 1, loc_on_map_traversible=True))

        # m_vis = self.mapper.get_obs_map_()

        explored_vis = self.mapper.get_explored_map(self.selem, 1)

        track_map = self.mapper.rearrage_map
        track_img = np.ones((track_map.shape[0], track_map.shape[1], 3), dtype=np.uint8)*255

        # track_img[m_vis] = [0, 0, 0]

        # TODO: for loop for multiple images
        track_img[track_map == 1] = [255, 0, 0]
        ax[3].imshow(track_img)

        ax[0].imshow(rgb)
        # ax[1].imshow(depth)
        ax[1].imshow(m_vis, origin='lower', vmin=0, vmax=1,
                     cmap='Reds')
        state_xy = self.mapper.get_position_on_map()

        state_theta = self.mapper.get_rotation_on_map()
        arrow_len = 2.0 / self.mapper.resolution
        ax[1].arrow(state_xy[0], state_xy[1],
                    arrow_len * np.cos(state_theta + np.pi / 2),
                    arrow_len * np.sin(state_theta + np.pi / 2),
                    color='b', head_width=20)

        if self.point_goal is not None:
            ax[1].text(self.point_goal[1], self.point_goal[0], "n", fontsize="medium", color='m')

        if self.rearr_mode:
            dx = np.cos(state_theta + np.pi / 2)
            dy = np.sin(state_theta + np.pi / 2)
            v1 = np.array([dx, dy])

            gx = self.goal_pos_map[0] - state_xy[0]
            gy = self.goal_pos_map[1] - state_xy[1]
            v2 = np.array([gx, gy])

            dot = dx * gx + dy * gy
            cross = dx * gy - dy * gx

            phi = np.arccos(dot / (np.linalg.norm(v1) + np.linalg.norm(v2)))
            phi = phi * (180 / np.pi)
            if not self.place_loc_rend:
                ax[1].text(self.goal_pos_map[0], self.goal_pos_map[1],"P", fontsize="medium", color='b')
                # ax[1].text(self.goal_pos_map[1], self.goal_pos_map[0], "A", fontsize="medium", color='b')
                ax[1].plot([state_xy[0], self.goal_pos_map[0]], [state_xy[1], self.goal_pos_map[1]], color='cyan',
                           linewidth=2)
            else:
                ax[1].text(self.goal_pos_map[0], self.goal_pos_map[1], "G", fontsize="medium", color='b')
                # ax[1].text(self.goal_pos_map[1], self.goal_pos_map[0], "A", fontsize="medium", color='b')
                ax[1].plot([state_xy[0], self.goal_pos_map[0]], [state_xy[1], self.goal_pos_map[1]], color='cyan',
                           linewidth=2)
            if self.updated_goal_pos_map is not None:
                dy = self.updated_goal_pos_map[1] - state_xy[1]
                dx = self.updated_goal_pos_map[0] - state_xy[0]
                theta_ch = np.arctan2(dy, dx) * (180/np.pi)
                theta_ch = np.round(theta_ch, decimals=2)
                theta_ag = (state_theta + np.pi/2) * (180/np.pi)
                if not self.place_loc_rend:
                    ax[1].text(self.updated_goal_pos_map[0], self.updated_goal_pos_map[1], "P", fontsize="medium", color='g')
                else:
                    ax[1].text(self.updated_goal_pos_map[0], self.updated_goal_pos_map[1], "G", fontsize="medium",
                           color='y')
                # ax[1].set_title(str([round(self.position['x'], 2), round(self.position['z'], 2)]) + self.text_to_add)
                ax[1].set_title(self.text_to_add)
            else:
                ax[1].set_title(self.text_to_add)

        # if self.point_goal is not None:
        #     ax[1].plot(self.point_goal[0], self.point_goal[1], color='blue', marker='o', linewidth=10, markersize=12)
        # ax[2].set_title(f"Traversable {self.unexplored_area}")
        if not self.rearr_mode:
            ax[1].set_title(str([round(self.position['x'], 2), round(self.position['z'], 2)]) + self.text_to_add)
        # else:
        #     if self.updated_goal_pos_map is not None:
        #         ax[1].set_title(self.text_to_add)
        #     else:
        #         ax[1].set_title(self.text_to_add + " : Reachable")

        # container_map = np.concatenate([np.zeros_like(self.mapper.get_object_on_map('trophy'))[:,:,np.newaxis].astype('float')]*3, axis=2)
        # for uuid in self.mapper.objects:
        #     if 'box' not in uuid:
        #         continue
        #     box_map = self.mapper.get_object_on_map(uuid)
        #     if uuid in self.opened:
        #     	container_map[:,:,0] += box_map.astype('float') * 255
        #     else:
        #         container_map[:,:,1] += box_map.astype('float') * 255
        # container_map[container_map > 255] = 255
        # ax[3].imshow(container_map[::-1,:,:])
        # ax[3].set_title("Container Map")
        # #ax[3].imshow(self.fmm_dist, origin='lower')
        # #ax[3].set_title('Current FMM')
        # # ax[4].imshow(np.sum(self.mapper.map, 2) > 0, origin='lower')
        # # ax[4].set_title('Sum Points')
        ax[2].imshow(explored_vis > 0, origin='lower')
        ax[2].set_title('Explored Area')
        if self.run_path is not None:
            fig.savefig(os.path.join(self.run_path, 'map2d/{:04d}'.format(self.step_count)))
        else:
            print("SAVE DIR DOES NOT EXISTS")
            fig.savefig(os.path.join("output/WorldModel", 'map2d/{:04d}'.format(self.step_count)))
        # vmax = len(self.mapper.objects) + 1
        # #ax[4].imshow(np.argmax(self.mapper.semantic_map, 2), vmin=0, vmax=vmax, origin='lower')
        # ax[4].imshow(self.mapper.get_object_on_map('trophy')[::-1,:])
        # #ax[4].set_title('Semantic Map')
        # ax[4].set_title('Trophy Map')

    def _vis(self, fig, rgb, depth, act_id, point_goal):
        ax = []
        spec = gridspec.GridSpec(ncols=2, nrows=2, 
            figure=fig, left=0., right=1., wspace=0.05, hspace=0.5)
        ax.append(fig.add_subplot(spec[0, 0]))
        # ax.append(fig.add_subplot(spec[:2, 3:]))
        ax.append(fig.add_subplot(spec[0, 1]))
        ax.append(fig.add_subplot(spec[1, 1]))
        # ax.append(fig.add_subplot(spec[2:, 2:4]))
        # ax.append(fig.add_subplot(spec[2:, 4:6]))
        dd = '\n'.join(wrap(self.goal.description, 50))
        fig.suptitle(f"{self.step_count-1}. {dd} act: {act_id}", fontsize=14)
        
        for a in ax:
            a.axis('off')
        
        m_vis = np.invert(self.mapper.get_traversible_map(
                          self.selem, 1,loc_on_map_traversible=True))
        explored_vis = self.mapper.get_explored_map(self.selem, 1)
        ax[0].imshow(rgb)
        # ax[1].imshow(depth)
        ax[1].imshow(m_vis, origin='lower', vmin=0, vmax=1,
                     cmap='Reds')
        state_xy = self.mapper.get_position_on_map()
        state_theta = self.mapper.get_rotation_on_map()
        arrow_len = 2.0/self.mapper.resolution
        ax[1].arrow(state_xy[0], state_xy[1], 
                    arrow_len*np.cos(state_theta+np.pi/2),
                    arrow_len*np.sin(state_theta+np.pi/2), 
                    color='b', head_width=20)
        if self.point_goal is not None:
            ax[1].plot(self.point_goal[0], self.point_goal[1], color='blue', marker='o',linewidth=10, markersize=12)
        #ax[2].set_title(f"Traversable {self.unexplored_area}")
        ax[1].set_title("Obstacle Map")

        
        # container_map = np.concatenate([np.zeros_like(self.mapper.get_object_on_map('trophy'))[:,:,np.newaxis].astype('float')]*3, axis=2)
        # for uuid in self.mapper.objects:
        #     if 'box' not in uuid:
        #         continue
        #     box_map = self.mapper.get_object_on_map(uuid)
        #     if uuid in self.opened:
        #     	container_map[:,:,0] += box_map.astype('float') * 255
        #     else:
        #         container_map[:,:,1] += box_map.astype('float') * 255
        # container_map[container_map > 255] = 255
        # ax[3].imshow(container_map[::-1,:,:])
        # ax[3].set_title("Container Map")
        # #ax[3].imshow(self.fmm_dist, origin='lower')
        # #ax[3].set_title('Current FMM')
        # # ax[4].imshow(np.sum(self.mapper.map, 2) > 0, origin='lower')
        # # ax[4].set_title('Sum Points')
        ax[2].imshow(explored_vis > 0, origin='lower')
        ax[2].set_title('Explored Area')
        fig.savefig(f'images/{self.step_count}')
        # vmax = len(self.mapper.objects) + 1
        # #ax[4].imshow(np.argmax(self.mapper.semantic_map, 2), vmin=0, vmax=vmax, origin='lower')
        # ax[4].imshow(self.mapper.get_object_on_map('trophy')[::-1,:])
        # #ax[4].set_title('Semantic Map')
        # ax[4].set_title('Trophy Map')

    def _cover_fn(self, uuid, iters, semantic_size_threshold, morph_disk_size):
        self.exploring = True
        unexplored = self._get_unexplored()
        if iters == 0:
            logging.error(f'Coverage iteration limit reached.')
            self.exploring = False
            return
        else:
            explored = np.sum(unexplored) < 20
            # print('sum of unexplored:', np.sum(unexplored))
            if explored:
                self.exploring = False
                logging.error(f'Unexplored area < 20. Exploration finished')
                if do_video:
                    cv2.destroyAllWindows()
                    self.video_writer.release()
                    self.video_ind += 1
            else:
                ind_i, ind_j = self._sample_point_in_unexplored_reachable(unexplored)
                self.point_goal = [ind_i, ind_j]
                
                # logging.error(f'Exploration setting pointgoal: {ind_i}, {ind_j}')

                fn = lambda: self._cover_fn(uuid, iters - 1, semantic_size_threshold, morph_disk_size)
                self.execution.put(fn)
                fn = lambda: self._point_goal_fn_assigned(np.array([ind_j, ind_i]), explore_mode=True, iters=30)
                self.execution.put(fn)


    def _explore_fn(self, uuid, iters, semantic_size_threshold, morph_disk_size):
        if self.success:
            return
        unexplored = self._get_unexplored()
        logging.error(f'Explore fn ({iters}), unexplored area: {np.sum(unexplored)}')
        if iters == 0:
            logging.error(f'Exploration iteration limit reached.')
            return
        else: 
            spotted = self._check_object_goal_spotted(uuid, semantic_size_threshold, morph_disk_size)
            if spotted:
                # Object has been found, do nothing
                logging.error(f'Object {uuid} spotted.')
                return
            else:
                explored = np.sum(unexplored) < 20
                if explored:
                    if do_video:
                        cv2.destroyAllWindows()
                        self.video_writer.release()
                        self.video_ind += 1
                        st()
                    logging.error(f'Unexplored area < 20. Exploration finished')
                    return
                else:
                    ind_i, ind_j = self._sample_point_in_unexplored_reachable(unexplored)
                    logging.error(f'Exploration setting pointgoal: {ind_i}, {ind_j}')
                    fn = lambda: self._explore_fn(uuid, iters-1, semantic_size_threshold, morph_disk_size)
                    self.execution.put(fn)
                    fn = lambda: self._point_goal_fn_assigned(np.array([ind_j, ind_i]), explore_mode=True)
                    self.execution.put(fn)

    def _check_object_goal_spotted(self, uuid, semantic_size_threshold, morph_disk_size):
        disk = skimage.morphology.disk(morph_disk_size)
        object_on_map = self.mapper.get_object_on_map(uuid)
        object_on_map = skimage.morphology.binary_opening(object_on_map, disk)
        return np.sum(object_on_map) > semantic_size_threshold

    def _check_box_large_enough(self, uuid, morph_disk_size):
        disk = skimage.morphology.disk(morph_disk_size)
        object_on_map = self.mapper.get_object_on_map(uuid)
        object_on_map = skimage.morphology.binary_opening(object_on_map, disk)
        if object_on_map.sum() == 0:
            return False
        object_on_map = self._get_largest_cc(object_on_map)

        y, x = np.where(object_on_map)
        y_min, y_max = np.amin(y), np.amax(y)
        x_min, x_max = np.amin(x), np.amax(x)

        y_len = y_max - y_min
        x_len = x_max - x_min

        outer_rect_area = y_len * x_len

        return outer_rect_area >= 25 and y_len >= 5 or x_len >= 5

    # def _object_goal_fn(self, uuid, iters=10, semantic_size_threshold=1, morph_disk_size=1):
    #     if self.success:
    #         return
    #     # Find the location of the object
    #     if iters == 0:
    #         return
    #     else:
    #         spotted = self._check_object_goal_spotted(uuid, semantic_size_threshold, morph_disk_size)
    #         if not spotted:
    #             logging.error(f'Object {uuid} not spotted, going to open')

    #             # Presummably we need to open something before we can see it.
    #             for oid in self.mapper.objects.keys():
    #                 if self._check_box_large_enough(oid,1) and oid not in self.opened:
    #                     # print("spotted", oid)
    #                     fn = lambda: self._object_goal_fn(uuid, iters, semantic_size_threshold, morph_disk_size)
    #                     self.execution.put(fn)
    #                     fn = lambda: self._open_fn(oid)
    #                     self.execution.put(fn)
    #                     fn = lambda: self._object_goal_fn(oid, 10, semantic_size_threshold, morph_disk_size)
    #                     self.execution.put(fn)
    #                     break
    #         else:
    #             disk = skimage.morphology.disk(morph_disk_size)
    #             object_on_map = self.mapper.get_object_on_map(uuid)
    #             object_on_map = skimage.morphology.binary_opening(object_on_map, disk)
    #             if object_on_map.sum() == 0:
    #                 return
    #             object_on_map = self._get_largest_cc(object_on_map)
    #             y, x = np.where(object_on_map)
    #             obj_x = np.mean(x)
    #             obj_y = np.mean(y)
    #             self_x, self_y = self.mapper.get_position_on_map()
    #             goal_x = (self_x + obj_x) / 2
    #             goal_y = (self_y + obj_y) / 2
    #             ind = np.argmin((goal_x-x)**2 + (goal_y-y)**2)
    #             obj_y = y[ind]
    #             obj_x = x[ind]

    #             disk2 = skimage.morphology.disk(1)
    #             # use instead reachable
    #             # traversible = self.mapper.get_traversible_map(self.selem, POINT_COUNT, loc_on_map_traversible=True)
    #             reachable = self._get_reachable_area()

    #             to_gate = np.invert(skimage.morphology.binary_erosion(reachable, disk2))
    #             to_gate_ma = ma.masked_values(to_gate*1, 0)
    #             to_gate_ma[int(obj_y), int(obj_x)] = 0
    #             dd = skfmm.distance(to_gate_ma, dx=1)
    #             # plt.imsave('vis/test.png',reachable)
    #             dd_mask = np.invert(np.isnan(ma.filled(dd, np.nan)))
    #             dd = ma.filled(dd, np.inf)
    #             dd[np.invert(reachable)] = np.NaN

    #             dd_min = np.nanmin(dd)
    #             y, x = np.where(dd == dd_min)
    #             point_goal_target = np.array([x[0],y[0]])
    #             reached = self._check_point_goal_reached(point_goal_target, dist_thresh=0.1)

    #             # Termination condition
    #             if reached:
    #                 # Rotate towards the object and move towards it, so that we get it
    #                 # exactly right. Compute actual location
    #                 disk = skimage.morphology.disk(morph_disk_size)
    #                 object_on_map = self.mapper.get_object_on_map(uuid)
    #                 object_on_map = skimage.morphology.binary_opening(object_on_map, disk)
    #                 if object_on_map.sum() == 0:
    #                     return
    #                 y, x = np.where(object_on_map)
    #                 obj_x = np.mean(x)
    #                 obj_y = np.mean(y)
    #                 agent_x, agent_y = self.mapper.get_position_on_map()
    #                 agent_theta = np.rad2deg(self.mapper.get_rotation_on_map())
    #                 angle = np.rad2deg(np.arctan2(obj_y-agent_y, obj_x-agent_x))
    #                 delta_angle = (angle-90 - agent_theta) % 360
    #                 if delta_angle <= 180:
    #                     for _ in range(int(delta_angle)//self.DT):
    #                         yield 'RotateLeft', {'rotation': self.DT}
    #                 else:
    #                     for _ in range(int((360 - delta_angle)//self.DT)):
    #                         yield 'RotateRight', {'rotation': self.DT}
                
    #             else:
    #                 fn = lambda: self._object_goal_fn(uuid, iters-1, semantic_size_threshold, morph_disk_size)
    #                 self.execution.put(fn)

    #                 # Call point_goal.
    #                 logging.error(f'{self.step_count}: Object goal object at: {obj_x}, {obj_y}')
    #                 logging.error(f'{self.step_count}: Object goal: point goal: {x[0]}, {y[0]}')
    #                 fn = lambda: self._point_goal_fn(point_goal_target, dist_thresh=0.8)
    #                 self.execution.put(fn)

    def _reach_fn(self, uuid):
        yield FORWARD

    def _get_pickup_pixel(self, uuid):
        view_mask = self.mapper.objects[uuid]['view_mask']
        disk = skimage.morphology.disk(5)
        view_mask = skimage.morphology.binary_erosion(view_mask, disk)
        inds = np.where(view_mask == 1)
        indices = np.stack(inds, axis=1).astype(int)
        pixelX = None
        pixelY = None
        if np.prod(indices.shape) == 0:
            return pixelX, pixelY
        mean_loc = indices.mean(0).reshape(1,2)
        ind = np.argmin(np.sum((indices-mean_loc)**2))
        pixelY, pixelX = indices[ind].tolist()
        return pixelY, pixelX

    def _pickup_fn(self, uuid):
        logging.error("Start PICKUP")
        if uuid not in self.mapper.objects:
            return
        yield actions[DOWN], {'horizon': self.HORIZON_DT}
        yield actions[DOWN], {'horizon': self.HORIZON_DT}
        n_lookdown = 2
        for _ in range(10):
            yield actions[LEFT], {'rotation': self.DT}
            yield actions[LEFT], {'rotation': self.DT}
            rot_count = 0
            in_view = self.mapper.objects[uuid]['in_view'] == True and self.mapper.objects[uuid]['view_mask'] is not None
            while not in_view:
                yield actions[LEFT], {'rotation': self.DT}
                in_view = self.mapper.objects[uuid]['in_view'] == True and self.mapper.objects[uuid]['view_mask'] is not None
                rot_count += 1
                if rot_count == 35:
                    break
            if not in_view:
                yield actions[DOWN], {'horizon': self.HORIZON_DT}
                yield actions[DOWN], {'horizon': self.HORIZON_DT}
                n_lookdown += 2
                continue
            pickedup = False
            for i in range(3):
                pixelY, pixelX = self._get_pickup_pixel(uuid)
                if pixelX is None:
                    yield self._get_action(FORWARD)
                    yield self._get_action(FORWARD)
                    break
                yield actions[PICKUP], {'objectImageCoordsX': pixelX, 'objectImageCoordsY': pixelY}
                status = self.obs.return_status
                # print(status)
                if status == 'SUCCESSFUL':
                    print("Picked up {}!!!".format(uuid))
                    pickedup = True
                    self.success = True
                    break
                elif status == 'OUT_OF_REACH':
                    object_on_map = self.mapper.get_object_on_map(uuid)
                    disk = skimage.morphology.disk(1)
                    object_on_map = skimage.morphology.binary_opening(object_on_map, disk)
                    if object_on_map.sum() == 0:
                        continue
                    object_on_map = self._get_largest_cc(object_on_map)
                    yield self._get_action(FORWARD)
                    y, x = np.where(object_on_map)
                    obj_x = np.mean(x)
                    obj_y = np.mean(y)
                    agent_x, agent_y = self.mapper.get_position_on_map()
                    agent_theta = np.rad2deg(self.mapper.get_rotation_on_map())
                    angle = np.rad2deg(np.arctan2(obj_y-agent_y, obj_x-agent_x))
                    delta_angle = (angle-90-agent_theta) % 360
                    if delta_angle <= 180:
                        for _ in range(int(delta_angle)//self.DT):
                            yield 'RotateLeft', {'rotation': self.DT}
                    else:
                        for _ in range(int((360 - delta_angle)//self.DT)):
                            yield 'RotateRight', {'rotation': self.DT}
                    for _ in range(3):
                        yield self._get_action(FORWARD)
            if pickedup:
                break
        for i in range(n_lookdown):
            yield actions[UP], {'horizon': self.HORIZON_DT}
        return

    def _get_largest_cc(self, mask):
        labels = label(mask)
        largestCC = labels == np.argmax(np.array(np.bincount(labels.flat)[1:]))+1
        return largestCC

    def _rotate_look_down_up(self, uuid):
        '''
        Function to rotate toward a box center and look down and then up
        '''
        object_on_map = self.mapper.get_object_on_map(uuid)
        disk = skimage.morphology.disk(1)
        object_on_map = skimage.morphology.binary_opening(object_on_map, disk)
        if object_on_map.sum() == 0:
            return
        if object_on_map.sum() == 400*600:
            object_on_map = object_on_map
        else:
            object_on_map = self._get_largest_cc(object_on_map)
        y, x = np.where(object_on_map)
        obj_x = np.mean(x)
        obj_y = np.mean(y)
        agent_x, agent_y = self.mapper.get_position_on_map()
        agent_theta = np.rad2deg(self.mapper.get_rotation_on_map())
        angle = np.rad2deg(np.arctan2(obj_y-agent_y, obj_x-agent_x))
        delta_angle = (angle-90-agent_theta) % 360
        if delta_angle <= 180:
            for _ in range(int(delta_angle)//self.DT):
                yield 'RotateLeft', {'rotation': self.DT}
        else:
            for _ in range(int((360 - delta_angle)//self.DT)):
                yield 'RotateRight', {'rotation': self.DT}
        max_forward = 5
        forwards = 0
        while self.obs.return_status == "SUCCESSFUL" and forwards < max_forward:
            forwards += 1
            yield self._get_action(FORWARD)
        yield actions[DOWN], {'horizon': self.HORIZON_DT}
        yield actions[DOWN], {'horizon': self.HORIZON_DT}
        yield actions[DOWN], {'horizon': self.HORIZON_DT}
        yield actions[DOWN], {'horizon': self.HORIZON_DT}
        yield actions[DOWN], {'horizon': self.HORIZON_DT}
        yield actions[DOWN], {'horizon': self.HORIZON_DT}
        yield actions[UP], {'horizon': self.HORIZON_DT}
        yield actions[UP], {'horizon': self.HORIZON_DT}
        yield actions[UP], {'horizon': self.HORIZON_DT}
        yield actions[UP], {'horizon': self.HORIZON_DT}
        yield actions[UP], {'horizon': self.HORIZON_DT}
        yield actions[UP], {'horizon': self.HORIZON_DT}
        for _ in range(3):
            yield self._get_action(BACKWARD)

    def _check_corner(self, uuid, corner_loc_cell):
        if self._check_object_goal_spotted('trophy', 0, 0):
            return
        fn = lambda: self._rotate_look_down_up(uuid)
        self.execution.put(fn)
        fn = lambda: self._point_goal_fn(corner_loc_cell)
        self.execution.put(fn)

    def _check_inside(self, uuid):
        logging.error('Start Check Inside')
        max_forward = 5
        forwards = 0
        while self.obs.return_status == 'SUCCESSFUL' and forwards < max_forward:
            forwards += 1
            yield self._get_action(FORWARD)
        # Look down and up
        yield actions[DOWN], {'horizon': self.HORIZON_DT}
        yield actions[DOWN], {'horizon': self.HORIZON_DT}
        yield actions[DOWN], {'horizon': self.HORIZON_DT}
        yield actions[DOWN], {'horizon': self.HORIZON_DT}
        yield actions[DOWN], {'horizon': self.HORIZON_DT}
        yield actions[DOWN], {'horizon': self.HORIZON_DT}
        yield actions[UP], {'horizon': self.HORIZON_DT}
        yield actions[UP], {'horizon': self.HORIZON_DT}
        yield actions[UP], {'horizon': self.HORIZON_DT}
        yield actions[UP], {'horizon': self.HORIZON_DT}
        yield actions[UP], {'horizon': self.HORIZON_DT}
        yield actions[UP], {'horizon': self.HORIZON_DT}
        if self._check_object_goal_spotted('trophy', 0, 0):
            return
        
        # Get four corners
        object_on_map = self.mapper.get_object_on_map(uuid)
        disk = skimage.morphology.disk(1)
        object_on_map = skimage.morphology.binary_opening(object_on_map, disk)
        if object_on_map.sum() == 0:
            object_on_map = self.mapper.get_object_on_map(uuid)
        object_on_map = self._get_largest_cc(object_on_map)
        y, x = np.where(object_on_map)
        y_min, y_max = np.amin(y), np.amax(y)
        x_min, x_max = np.amin(x), np.amax(x)

        ymin_y, ymin_x = y[y == y_min].mean(), x[y == y_min].mean()
        ymax_y, ymax_x = y[y == y_max].mean(), x[y == y_max].mean()
        xmin_y, xmin_x = y[x == x_min].mean(), x[x == x_min].mean()
        xmax_y, xmax_x = y[x == x_max].mean(), x[x == x_max].mean()

        # Go to four corners and look at center, up and down
        agent_x, agent_y = self.mapper.get_position_on_map()
        for _ in range(3):
            yield self._get_action(BACKWARD)
        if self.mapper.resolution*np.sqrt((agent_x-ymin_x)**2+(agent_y-ymin_y)**2) > 0.4:
            fn = lambda: self._check_corner(uuid, np.array([ymin_x, ymin_y-20]))
            self.execution.put(fn)
        if self.mapper.resolution*np.sqrt((agent_x-xmin_x)**2+(agent_y-xmin_y)**2) > 0.4:
            fn = lambda: self._check_corner(uuid, np.array([xmin_x-20, xmin_y]))
            self.execution.put(fn)
        if self.mapper.resolution*np.sqrt((agent_x-ymax_x)**2+(agent_y-ymax_y)**2) > 0.4:
            fn = lambda: self._check_corner(uuid, np.array([ymax_x, ymax_y+20]))
            self.execution.put(fn)
        if self.mapper.resolution*np.sqrt((agent_x-xmax_x)**2+(agent_y-xmax_y)**2) > 0.4:
            fn = lambda: self._check_corner(uuid, np.array([xmax_x+20, xmax_y]))
            self.execution.put(fn)

        return

    def _open_fn(self, uuid):
        # New version of open function with pixel location
        logging.error('Start Opening')
        n_lookdown = 0
        for _ in range(10):
            rot_count = 0
            in_view = self.mapper.objects[uuid]['in_view'] == True and self.mapper.objects[uuid]['view_mask'] is not None
            while not in_view:
                yield actions[LEFT], {'rotation': self.DT}
                in_view = self.mapper.objects[uuid]['in_view'] == True and self.mapper.objects[uuid]['view_mask'] is not None
                rot_count += 1
                if rot_count == 35:
                    break
            if not in_view:
                yield actions[DOWN], {'horizon': self.HORIZON_DT}
                n_lookdown += 1
                continue
            pixelY, pixelX = self._get_pickup_pixel(uuid)
            for i in range(3):
                if pixelX is None:
                    break
                yield actions[OPEN], {'objectImageCoordsX': pixelX, 'objectImageCoordsY': pixelY}
                status = self.obs.return_status
                if status == 'SUCCESSFUL':
                    self.opened.append(uuid)
                    break
                pixelY += 5
                if pixelY > 599:
                    break
            print("open status:", status)
            if status == 'NOT_OPENABLE':
                for i in range(n_lookdown):
                    yield actions[UP], {'horizon': self.HORIZON_DT}
                self.opened.append(uuid)
                return
            if status == 'SUCCESSFUL':
                self.opened.append(uuid)
                for i in range(n_lookdown):
                    yield actions[UP], {'horizon': self.HORIZON_DT}
                return self.execution.put(lambda: self._check_inside(uuid))
            if status == 'OBSTRUCTED':
                yield self._get_action(BACKWARD)
                continue
            if status == 'OUT_OF_REACH':
                for _ in range(5):
                    yield self._get_action(FORWARD)
            yield actions[LEFT], {'rotation': self.DT}
        for i in range(n_lookdown):
            yield actions[UP], {'horizon': self.HORIZON_DT}
        self.opened.append(uuid)
        return

    '''
    def _open_fn(self, uuid):
        # Andy: Need to change the rotation functions if we use this open function
        logging.error('Start Opening')
        self.opened.append(uuid)
        for _ in range(3):
            yield actions[OPEN], {'objectId': uuid, 'amount': 1.}
            if self.obs.return_status == 'NOT_OPENABLE': return
            if self.obs.return_status == 'SUCCESSFUL':
                return self.execution.put(lambda: self._check_inside())
            if self.obs.return_status == 'OBSTRUCTED':
                yield actions[LEFT], {'rotation': 180}
                yield self._get_action(FORWARD)
                yield actions[LEFT], {'rotation': 180}
                continue

            yield actions[DOWN], {'horizon': self.HORIZON_DT}
            yield actions[OPEN], {'objectId': uuid, 'amount': 1.}
            status = self.obs.return_status
            #yield actions[DOWN], {'horizon': -self.HORIZON_DT}
            yield actions[UP], {'horizon': self.HORIZON_DT}
            if status == 'NOT_OPENABLE': return
            if status == 'SUCCESSFUL':
                return self.execution.put(lambda: self._check_inside())
            if status == 'OBSTRUCTED':
                yield actions[LEFT], {'rotation': 180}
                yield self._get_action(FORWARD)
                yield actions[LEFT], {'rotation': 180}
                continue
            yield self._get_action(FORWARD)
    '''

    def _open_fn2(self, uuid):
        # Andy: Need to change the rotation functions if we use this open function
        logging.error('Start Opening')
        self.opened.append(uuid)
        yield actions[OPEN], {'objectId': uuid, 'amount': 1.}
        if self.obs.return_status == 'SUCCESSFUL':
            return

        if self.obs.return_status == 'OBSTRUCTED':
            yield actions[LEFT], {'rotation': 180}
            yield self._get_action(FORWARD)
            yield actions[LEFT], {'rotation': 180}
            yield actions[OPEN], {'objectId': uuid, 'amount': 1.}
            if self.obs.return_status == 'SUCCESSFUL':
                return

        yield actions[DOWN], {'horizon': self.HORIZON_DT}
        yield actions[OPEN], {'objectId': uuid, 'amount': 1.}
        if self.obs.return_status == 'SUCCESSFUL':
            yield actions[DOWN], {'horizon': -self.HORIZON_DT}
            return
        yield actions[DOWN], {'horizon': -self.HORIZON_DT}

        yield FORWARD
        yield actions[OPEN], {'objectId': uuid, 'amount': 1.}
        if self.obs.return_status == 'SUCCESSFUL':
            return
        yield actions[DOWN], {'horizon': self.HORIZON_DT}
        yield actions[OPEN], {'objectId': uuid, 'amount': 1.}
        if self.obs.return_status == 'SUCCESSFUL':
            yield actions[DOWN], {'horizon': -self.HORIZON_DT}
            return
        yield actions[DOWN], {'horizon': -self.HORIZON_DT}

    def _check_point_goal_reached(self, goal_loc_cell, dist_thresh=0.25):
        if self.open_mode:
            dist_thresh = 0.5
        if self.rearr_mode:
            dist_thresh = 0.5
        state_xy = self.mapper.get_position_on_map()
        state_xy = state_xy.astype(np.int32)
        state_theta = self.mapper.get_rotation_on_map() + np.pi/2

        dist = np.sqrt(np.sum(np.square(state_xy - goal_loc_cell)))
        reached = dist*self.mapper.resolution < dist_thresh
        return reached

    # def _point_goal_fn(self, goal_loc_cell, explore_mode=False, search_mode=False, dist_thresh=0.5, iters=10):
    #     state_xy = self.mapper.get_position_on_map()
    #     state_xy = state_xy.astype(np.int32)
    #     state_theta = self.mapper.get_rotation_on_map() + np.pi/2

    #     dist = np.sqrt(np.sum(np.square(state_xy - goal_loc_cell)))
    #     # logging.error(f'{self.step_count}: target: {goal_loc_cell}, current_loc: {state_xy} {np.round(np.rad2deg(state_theta),2)}, dist: {np.round(dist, 2)}')
    #     reached = self._check_point_goal_reached(goal_loc_cell, dist_thresh)
    #     # print(iters)
    #     if reached:
    #         if explore_mode:
    #             yield actions[DOWN], {'horizon': self.HORIZON_DT}
    #             yield actions[DOWN], {'horizon': self.HORIZON_DT}
    #             for _ in range(360//self.DT):
    #                 yield actions[LEFT], {'rotation': self.DT}
    #             yield actions[UP], {'horizon': self.HORIZON_DT}
    #             yield actions[UP], {'horizon': self.HORIZON_DT}
    #         return
    #     else:
    #         if iters==0:
    #             return
    #         traversible = self.mapper.get_traversible_map(self.selem, POINT_COUNT, loc_on_map_traversible=True)
    #         planner = FMMPlanner(traversible, 360//self.DT, int(self.STEP_SIZE/self.mapper.resolution), self.obstructed_actions)

    #         goal_loc_cell = goal_loc_cell.astype(np.int32)
    #         reachable = planner.set_goal(goal_loc_cell)
    #         self.fmm_dist = planner.fmm_dist*1.
    #         # print(reachable[state_xy[1], state_xy[0]])
    #         if reachable[state_xy[1], state_xy[0]]:
    #             a, state, act_seq = planner.get_action(np.array([state_xy[0], state_xy[1], state_theta]))
    #             if act_seq[0] == 0:
    #                 logging.error('FMM failed')
    #                 return
    #             else:
    #                 # print("ACT SEQ", act_seq)
    #                 pass

    #             # Fast Rotation (can't do fast rotation with fixed step size)
    #             if False:
    #                 rotations=act_seq[:-1]
    #                 if len(rotations)>0:
    #                     ty=rotations[0]
    #                     assert all(map(lambda x: x==ty,rotations)), 'bad acts'
    #                     ang = self.params[ty]['rotation'] * len(rotations)
    #                     yield actions[ty], {'rotation': ang}
    #                     # if self.obs.reward >= 1:
    #                     #     logging.error('Reward >= 1: Episode success, returning')
    #                     #     yield DONE
    #                 yield FORWARD
    #                 # if self.obs.reward >= 1:
    #                 #     logging.error('Reward >= 1: Episode success, returning')
    #                 #     yield DONE
    #             else:
    #                 for a in act_seq:
    #                     yield a
    #                     if search_mode:
    #                         yield actions[DOWN], {'horizon': self.HORIZON_DT}
    #                         yield actions[UP], {'horizon': self.HORIZON_DT}
    #             fn = lambda: self._point_goal_fn(goal_loc_cell, explore_mode=explore_mode, dist_thresh=dist_thresh, iters=iters-1)
    #             self.execution.put(fn)

    def reorient_head(self, pick=True):
        if self.rearr_mode and self.head_down_curr:
            # if not pick:
            #     print("move head back : ", self.move_head_back_after_place)
            #     if self.move_head_back_after_place:
            #         self.move_head_back_after_place = False
            #         if self.look_down_twice:
            #             yield actions[UP], {'horizon': self.HORIZON_DT}
            #             yield actions[UP], {'horizon': self.HORIZON_DT}
            #         else:
            #             yield actions[UP], {'horizon': self.HORIZON_DT}
            # else:
            if self.rotate_left is not None:
                if self.rotate_left == True:
                    for _ in range(self.n_steps_to_rotate):
                        yield actions[RIGHT], {'rotation': self.DT}
                else:
                    for _ in range(self.n_steps_to_rotate):
                        yield actions[LEFT], {'rotation': self.DT}
            if self.look_down_twice:
                if not self.open_mode or self.last_place:
                    yield actions[UP], {'horizon': self.HORIZON_DT}
                    yield actions[UP], {'horizon': self.HORIZON_DT}
                    if self.open_mode:
                        self.last_place = False
            else:
                if not self.open_mode or self.last_place:
                    yield actions[UP], {'horizon': self.HORIZON_DT}
                    if self.open_mode:
                        self.last_place = False
            self.head_down_curr = False
            self.rotate_left = None
        yield actions[DONE], None

    def _point_goal_fn_assigned(self, goal_loc_cell, explore_mode=False, dist_thresh=0.25, iters=20, search_mode=False, search_mode_reduced=False):
        state_xy = self.mapper.get_position_on_map()
        state_xy = state_xy.astype(np.int32)
        state_theta = self.mapper.get_rotation_on_map() + np.pi/2

        dist = np.sqrt(np.sum(np.square(np.squeeze(state_xy) - np.squeeze(goal_loc_cell))))
        # self.dists.append(dist)
        # logging.error(f'{self.step_count}: target: {goal_loc_cell}, current_loc: {state_xy} {np.round(np.rad2deg(state_theta),2)}, dist: {np.round(self.mapper.resolution*np.round(dist, 2),2)}')
        self.goal_loc_viz = goal_loc_cell
        reached = self._check_point_goal_reached(goal_loc_cell, dist_thresh)

        if self.place_loc_rend:
            if self.failed_action_count == 10:
                yield actions[DONE], None
                return

        # random resampling
        if self.step_count % 100 == 0:
            print("resampling")
            del self.execution
            self._setup_execution(None)

        if self.agent_stuck:
            self.agent_stuck = False
            self.reinit_act_queue()
            self._setup_execution(None)
            # if self.alternate_stuck_action:
            #     yield actions[RIGHT], {'rotation': self.DT}
            #     yield self._get_action(FORWARD)
            #     self.alternate_stuck_action = not self.alternate_stuck_action
            # else:
            #     yield actions[LEFT], {'rotation': self.DT}
            #     yield self._get_action(FORWARD)
            #     self.alternate_stuck_action = not self.alternate_stuck_action

        if reached:
            print("Reached...")
            if not self.reorient_head_flag:
                self.look_down_twice = False
                if self.rearr_mode:
                    self.head_down_curr = True
                    if self.y_loc is not None:
                        if self.y_loc < 0.3:
                            self.look_down_twice = True

                        # if self.look_down_twice:
                        #     yield actions[DOWN], {'horizon': self.HORIZON_DT}
                        #     yield actions[DOWN], {'horizon': self.HORIZON_DT}
                        # else:
                        #     yield actions[DOWN], {'horizon': self.HORIZON_DT}
                        # yield actions[DONE], None

                    state_theta = self.mapper.get_rotation_on_map()
                    state_xy = self.mapper.get_position_on_map()
                    state_xy = state_xy.astype(np.int32)

                    dx = np.cos(state_theta + np.pi/2)
                    dy = np.sin(state_theta + np.pi/2)
                    v1 =  np.array([dx, dy])

                    gx = self.goal_pos_map[0] - state_xy[0]
                    gy = self.goal_pos_map[1] - state_xy[1]
                    v2 = np.array([gx, gy])

                    dot = dx*gx + dy*gy
                    cross = dx*gy - dy*gx

                    phi = np.arccos(dot/(np.linalg.norm(v1) + np.linalg.norm(v2)))
                    phi = phi * (180 / np.pi)

                    if abs(phi) < 30:
                        self.n_steps_to_rotate = 0
                    elif 30 <= abs(phi) < 135:
                        self.n_steps_to_rotate = 1
                    else:
                        self.n_steps_to_rotate = 2

                    if cross == 0:
                        if dot > 0:
                            # facing goal
                            if self.look_down_twice:
                                if not self.open_mode:
                                    yield actions[DOWN], {'horizon': self.HORIZON_DT}
                                    yield actions[DOWN], {'horizon': self.HORIZON_DT}
                            else:
                                if not self.open_mode:
                                    yield actions[DOWN], {'horizon': self.HORIZON_DT}
                            yield actions[DONE], None
                        else:
                            # rotate 180
                            yield actions[RIGHT], {'rotation': self.DT}
                            yield actions[RIGHT], {'rotation': self.DT}
                            if self.look_down_twice:
                                if not self.open_mode:
                                    yield actions[DOWN], {'horizon': self.HORIZON_DT}
                                    yield actions[DOWN], {'horizon': self.HORIZON_DT}
                            else:
                                if not self.open_mode:
                                    yield actions[DOWN], {'horizon': self.HORIZON_DT}
                            yield actions[DONE], None
                    else:
                        for _ in range(self.n_steps_to_rotate):
                            if cross > 0:
                                self.rotate_left = True
                                yield actions[LEFT], {'rotation': self.DT}

                            else:
                                self.rotate_left = False
                                yield actions[RIGHT], {'rotation': self.orient_DT}
                        if self.look_down_twice:
                            if not self.open_mode:
                                yield actions[DOWN], {'horizon': self.HORIZON_DT}
                                yield actions[DOWN], {'horizon': self.HORIZON_DT}
                        else:
                            if not self.open_mode:
                                yield actions[DOWN], {'horizon': self.HORIZON_DT}

                        yield actions[DONE], None


                # yield actions[UP], {'horizon': self.HORIZON_DT}

            else:
                if self.not_reachable:
                    return
                else:
                    self._setup_execution(self.goal)

            if explore_mode:
                yield actions[DOWN], {'horizon': self.HORIZON_DT}
                yield actions[DOWN], {'horizon': self.HORIZON_DT}
                for _ in range(360//self.DT):
                    yield actions[RIGHT], {'rotation': self.DT}
                yield actions[UP], {'horizon': self.HORIZON_DT}
                yield actions[UP], {'horizon': self.HORIZON_DT}
            return #actions[DONE]
        else:
            # if iters==0:
            #     if self.cover_iters < 0 and self.goal.category == 'cover':
            #         self._setup_execution(self.goal)
            #         self.cover_iters -= 1
            #     return
            #     # return
            traversible = self.mapper.get_traversible_map(self.selem, POINT_COUNT, loc_on_map_traversible=True)
            planner = FMMPlanner(traversible, 360//self.DT, int(self.STEP_SIZE/self.mapper.resolution), self.obstructed_actions)

            goal_loc_cell = goal_loc_cell.astype(np.int32)
            reachable = planner.set_goal(goal_loc_cell)
            self.fmm_dist = planner.fmm_dist*1.
            # print(reachable)
            # print(reachable[state_xy[1], state_xy[0]])
            # exit()
            if reachable[state_xy[1], state_xy[0]]:
                a, state, act_seq = planner.get_action(np.array([state_xy[0], state_xy[1], state_theta]))
                self.act_seq = act_seq
                if act_seq[0] == 0:
                    logging.error('FMM failed')
                    return
                else:
                    pass

                for a in act_seq:
                    yield a
                    if not self.rearr_mode:
                        if self.action_failed:
                            yield actions[DOWN], {'horizon': self.HORIZON_DT}
                            yield actions[UP], {'horizon': self.HORIZON_DT}

                fn = lambda: self._point_goal_fn_assigned(goal_loc_cell, search_mode=search_mode, explore_mode=False, dist_thresh=dist_thresh, iters=iters-1)
                self.execution.put(fn)
            else:
                self.not_reachable = True
                print("!!!!!!!!!! NOT REACHABLE !!!!!!!!!!!")


    def _get_reachable_area(self):
        traversible = self.mapper.get_traversible_map(self.selem, POINT_COUNT, loc_on_map_traversible=True)
        planner = FMMPlanner(traversible, 360//self.DT, int(self.STEP_SIZE/self.mapper.resolution),self.obstructed_actions)
        state_xy = self.mapper.get_position_on_map()
        state_xy = state_xy.astype(np.int32)
        state_theta = self.mapper.get_rotation_on_map() + np.pi/2
        reachable = planner.set_goal(state_xy)
        # if np.sum(reachable)==0:
        #     st()
        # st()
        # plt.figure()
        # plt.imshow(traversible)
        # plt.savefig('images/test.png')
        # plt.figure()
        # plt.imshow(reachable)
        # plt.savefig('images/test2.png')
        return reachable

    def _get_unexplored(self):
        reachable = self._get_reachable_area()
        explored_point_count = 1
        explored = self.mapper.get_explored_map(self.selem, explored_point_count)
        unexplored = np.invert(explored)
        unexplored = np.logical_and(unexplored, reachable)
        # added to remove noise effects
        disk = skimage.morphology.disk(2)
        unexplored = skimage.morphology.binary_opening(unexplored, disk)
        self.unexplored_area = np.sum(unexplored)
        return unexplored

    def _sample_point_in_unexplored_reachable(self, unexplored):
        # Given the map, sample a point randomly in the open space.
        # ind_i, ind_j = np.where(unexplored)
        # map_locs = np.stack((ind_i,ind_j)).T
        # state_xy = self.mapper.get_position_on_map()
        # dists = np.linalg.norm(map_locs - state_xy,axis=1)
        # order = np.argsort(dists)
        # ind_i[order[0]], ind_j[order[0]]
        # # sample nearby location with higher weight
        # ind  = order[int(np.random.random()**5 * len(order))]
        # return ind_i[ind], ind_j[ind]

        # uniform sampling
        # st()
        # ind_i_e, ind_j_e = np.where(unexplored==False)
        # min_i_e = min(ind_i_e)
        # max_i_e = max(ind_i_e)
        # min_j_e = min(ind_j_e)
        # max_j_e = max(ind_j_e)
        # unexplored[unexplored<min_i_e, :] = False
        # unexplored[unexplored>max_i_e, :] = False
        # unexplored[:,unexplored<min_j_e] = False
        # unexplored[:,unexplored>max_j_e] = False
        ind_i, ind_j = np.where(unexplored)
        ind = self.rng.randint(ind_i.shape[0])
        return ind_i[ind], ind_j[ind]
