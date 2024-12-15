# rearrangement task libraries
import os
import sys
import shutil
import glob
import numpy as np
import torch.nn.functional
import argparse
from scipy.spatial import distance
import cv2
import copy
import random

import json
import gzip

from allenact.utils.misc_utils import NumpyJSONEncoder

from baseline_configs.one_phase.one_phase_rgb_base import (
    OnePhaseRGBBaseExperimentConfig,
)
from baseline_configs.two_phase.two_phase_rgb_base import (
    TwoPhaseRGBBaseExperimentConfig,
)
from rearrange.rearrange.tasks import RearrangeTaskSampler, WalkthroughTask, UnshuffleTask

from rearrange.scripts.config import actions, GaussianConfig, _random_scene_, _scene_id_
from rearrange.scripts.clip_feature_extractor import CLIPFeatureExtractor
from rearrange.scripts.data import Data
from rearrange.scripts.dense_feature_matching import Dinov2Matcher
from rearrange.scripts.gaussian_worldmodel import GaussianWorldModel
from rearrange.scripts.interfaces import ObjectScene, convert_rearrange_obs_to_event
from rearrange.scripts.agent import Agent
from rearrange.scripts.runner import Runner
from rearrange.scripts.scene import World
from rearrange.scripts.train_vanilla_gs import train_vanilla_gs
from rearrange.scripts.train_sugar_gs import train_sugar_gs
from rearrange.scripts.sam_mask import SAMPredictor
# from rearrange.scripts.gsam import GSAM

if not os.path.exists("rearrange/test/dataset/"):
    os.makedirs("rearrange/test/dataset/")

idx_to_action = {i: actions[i] for i in range(len(actions))}
action_to_idx = {actions[i]: i for i in range(len(actions))}

# task sampler
# combined dataset -> train + test + validation
task_sampler_params = TwoPhaseRGBBaseExperimentConfig.stagewise_task_sampler_args(
    stage="combined", process_ind=0, total_processes=1,
)
# combined, train, test
two_phase_rgb_task_sampler: RearrangeTaskSampler = TwoPhaseRGBBaseExperimentConfig.make_sampler_fn(
    **task_sampler_params,
    force_cache_reset=True,  # cache used for efficiency during training, should be True during inference
    only_one_unshuffle_per_walkthrough=True,  # used for efficiency during training, should be False during inference
    epochs=1,
)
how_many_unique_datapoints = two_phase_rgb_task_sampler.total_unique
num_tasks_to_do = 1

my_leaderboard_submission = {}
for i_task in range(num_tasks_to_do):
    print(f"\nStarting task {i_task}")

    if _random_scene_:
        scene_id_ = random.randint(1, how_many_unique_datapoints - 1)
        for _ in range(scene_id_):
            walkthrough_task = two_phase_rgb_task_sampler.next_task()
            walkthrough_task.step(action=0)
            unshuffle_task = two_phase_rgb_task_sampler.next_task()
            unshuffle_task.step(action=0)
        walkthrough_task = two_phase_rgb_task_sampler.next_task()

    else:
        if _scene_id_ is not None:
            for ere in range(how_many_unique_datapoints - 1):
                walkthrough_task = two_phase_rgb_task_sampler.next_task()
                if two_phase_rgb_task_sampler.current_task_spec.unique_id == f"{_scene_id_}":
                    break
                else:
                    walkthrough_task.step(action=0)
                    unshuffle_task = two_phase_rgb_task_sampler.next_task()
                    unshuffle_task.step(action=0)
        else:
            raise ValueError("scene id not set in the config file")


    print("--------------------------------------------------------------------------------")
    print("                              WALKTHROUGH TASK                                  ")
    print("                     Episode: ", two_phase_rgb_task_sampler.current_task_spec.unique_id)
    print("--------------------------------------------------------------------------------")

    if not os.path.exists(f"rearrange/test/dataset/{two_phase_rgb_task_sampler.current_task_spec.unique_id}/walkthrough"):
        os.makedirs(f"rearrange/test/dataset/{two_phase_rgb_task_sampler.current_task_spec.unique_id}/walkthrough")
    base_path = f"rearrange/test/dataset/{two_phase_rgb_task_sampler.current_task_spec.unique_id}/walkthrough"

    run_path = f"rearrange/test/runs/{two_phase_rgb_task_sampler.current_task_spec.unique_id}/walkthrough"
    if not os.path.exists(run_path):
        os.makedirs(run_path)
    else:
        shutil.rmtree(run_path)
        os.makedirs(run_path)
    if not os.path.exists(os.path.join(run_path, "reason")):
        os.mkdir(os.path.join(run_path, "reason"))
    if not os.path.exists(os.path.join(run_path, "map2d")):
        os.mkdir(os.path.join(run_path, "map2d"))
    if not os.path.exists(os.path.join(run_path, "reached")):
        os.mkdir(os.path.join(run_path, "reached"))
    if not os.path.exists(os.path.join(run_path, "rearr")):
        os.mkdir(os.path.join(run_path, "rearr"))
    if not os.path.exists(os.path.join(run_path, "rendered")):
        os.mkdir(os.path.join(run_path, "rendered"))
    if not os.path.exists(os.path.join(run_path, "sim")):
        os.mkdir(os.path.join(run_path, "sim"))
    if not os.path.exists(os.path.join(run_path, "dino_frames")):
        os.mkdir(os.path.join(run_path, "dino_frames"))
    if not os.path.exists(os.path.join(run_path, "depth")):
        os.mkdir(os.path.join(run_path, "depth"))
    if not os.path.exists(os.path.join(run_path, "depth_orig")):
        os.mkdir(os.path.join(run_path, "depth_orig"))


    config = GaussianConfig(width=224,
                            height=224,
                            run_path=run_path,
                            dataset_path=base_path,
                            dataset_name=two_phase_rgb_task_sampler.current_task_spec.unique_id)
    if config.include_feature:
        if not os.path.exists(os.path.join(run_path, "lang_field")):
            os.mkdir(os.path.join(run_path, "lang_field"))

    img_files = glob.glob(os.path.join(base_path, "images"))

    if len(img_files) == 0 :
        data_logger = Data(path=base_path, config=config)

        # TODO: load metrics, if already in it then dont compute again

        agent = Agent(config=config,
                      run_path=run_path)

        timestep = 0
        num_sampled = 0
        inds_explore = []
        time_walk_over = False
        while not (walkthrough_task.is_done() or time_walk_over):
            if timestep > 990:
                time_walk_over = True
                walkthrough_task.step(action=0)
                break
            if timestep == 0:
                obs_agent = walkthrough_task.walkthrough_env.get_agent_location()
                obs_cam = walkthrough_task.walkthrough_env.observation
                rgb = obs_cam[0]
                depth = obs_cam[1]
                event = convert_rearrange_obs_to_event(obs_agent, rgb, depth)
                init_pos = event.metadata['agent']['position']
                init_rot = event.metadata['agent']['rotation']['y']
                agent.navigation.init_navigation(bounds=None, init_pos=init_pos, init_rot=init_rot)
                agent.init_success_checker(rgb, walkthrough_task.walkthrough_env.controller)
                # For the first step
                action_successful = True
                agent.update_navigation_obs(rgb, depth, action_successful)
                timestep += 1
            else:
                if agent.navigation.explorer.goal.category != "cover":
                    # The agent has covered the scene and is now ready to train the splat
                    action_ind = 0
                    walkthrough_task.step(action=action_ind)
                    timestep += 1
                else:
                    action, param = agent.navigation.act()
                    if action == "Pass":
                        exploring = False
                        num_sampled += 1
                        try:
                            if not inds_explore:
                                ind_i, ind_j = agent.navigation.get_reachable_map_locations(sample=True)
                                if not ind_i:
                                    break
                                inds_explore.append([ind_i, ind_j])
                            else:
                                ind_i, ind_j = inds_explore.pop(0)
                        except:
                            break

                        agent.navigation.set_point_goal(ind_i, ind_j)
                    else:
                        action_rearrange = agent.nav_action_to_rearrange_action[action]
                        action_ind = agent.action_to_ind[action_rearrange]

                        walkthrough_task.step(action=action_ind)
                        timestep += 1

                        obs_agent = walkthrough_task.walkthrough_env.get_agent_location()
                        obs_cam = walkthrough_task.walkthrough_env.observation
                        rgb = obs_cam[0]
                        depth = obs_cam[1]
                        event = convert_rearrange_obs_to_event(obs_agent, rgb, depth)
                        action_successful = agent.navigation.success_checker.check_successful_action(rgb)
                        agent.update_navigation_obs(rgb, depth, action_successful)
                        if action_successful:
                            data_logger.get_data_step(fov=event.metadata['fov'],
                                                      rgb=event.frame,
                                                      depth=event.depth_frame,
                                                      camera_pos=event.metadata['cameraPosition'],
                                                      camera_rot=event.metadata['agent']['rotation'])
                            depth = np.clip(depth, 0, 10)
                            depth = (depth - depth.min()) / (depth.max() - depth.min())
                            depth = (depth * 255).astype(np.uint8)
                            cv2.imwrite(os.path.join(run_path, "depth_orig", "d_{}.png".format(timestep)), depth)
        if time_walk_over:
            walkthrough_task.step(action=0)


    # training the gaussian splat model from the data collected in the walkthrough phase

        data_logger.save_data()
        del data_logger
        torch.cuda.empty_cache()
    else:
        print("data for this episode was already saved")



    # training vanilla gaussian splatting model
    save_dir = "output/"
    if os.path.exists(save_dir):
        try:
            shutil.rmtree(save_dir)
        except OSError as e:
            print(f"Error: {save_dir} : {e.strerror}")
    os.makedirs(save_dir)

    train_vanilla_gs(base_path, iter=7000)
    torch.cuda.empty_cache()

    # Running SUGAR model on top of this to retrieve surface aligned Gaussians
    train_sugar_gs(base_path, config=config)
    torch.cuda.empty_cache()

    # Unshuffle task
    unshuffle_task = two_phase_rgb_task_sampler.next_task()
    print("--------------------------------------------------------------------------------")
    print("                              UNSHUFFLE TASK                                  ")
    print("                     Episode: ", two_phase_rgb_task_sampler.current_task_spec.unique_id)
    print("--------------------------------------------------------------------------------")

    run_path = f"rearrange/test/runs/{two_phase_rgb_task_sampler.current_task_spec.unique_id}/unshuffle"
    if not os.path.exists(run_path):
        os.makedirs(run_path)
    else:
        shutil.rmtree(run_path)
        os.makedirs(run_path)
    if not os.path.exists(os.path.join(run_path, "reason")):
        os.mkdir(os.path.join(run_path, "reason"))
    if not os.path.exists(os.path.join(run_path, "map2d")):
        os.mkdir(os.path.join(run_path, "map2d"))
    if not os.path.exists(os.path.join(run_path, "reached")):
        os.mkdir(os.path.join(run_path, "reached"))
    if not os.path.exists(os.path.join(run_path, "rearr")):
        os.mkdir(os.path.join(run_path, "rearr"))
    if not os.path.exists(os.path.join(run_path, "rendered")):
        os.mkdir(os.path.join(run_path, "rendered"))
    if not os.path.exists(os.path.join(run_path, "sim")):
        os.mkdir(os.path.join(run_path, "sim"))
    if not os.path.exists(os.path.join(run_path, "dino_frames")):
        os.mkdir(os.path.join(run_path, "dino_frames"))
    if not os.path.exists(os.path.join(run_path, "pick")):
        os.mkdir(os.path.join(run_path, "pick"))
    if not os.path.exists(os.path.join(run_path, "open")):
        os.mkdir(os.path.join(run_path, "open"))
    if not os.path.exists(os.path.join(run_path, "depth")):
        os.mkdir(os.path.join(run_path, "depth"))
    if not os.path.exists(os.path.join(run_path, "depth_orig")):
        os.mkdir(os.path.join(run_path, "depth_orig"))

    dense_feature_matcher = Dinov2Matcher()

    config.run_path = run_path
    config.patch_size = dense_feature_matcher.model.patch_size

    scene = World(device="cuda:0",
                  config=config)

    gaussian_model = GaussianWorldModel(config=config)
    gaussian_model.sugar.reset_grads()

    matcher = Runner(matcher=dense_feature_matcher,
                     config=config)

    agent = Agent(config=config,
                  run_path=run_path)

    # gsam = GSAM(device=config.gdino_device, config=config)

    sam_predictor = SAMPredictor(config=config)

    timestep = 0
    num_sampled = 0
    inds_explore = []
    rearrange_scene = False
    init_pos = None
    time_over = False
    mse_list = []
    while not (unshuffle_task.is_done() or time_over):
        if timestep > 1450:
            time_over = True
            unshuffle_task.step(action=0)
            break
        if timestep == 0:
            obs_agent = unshuffle_task.unshuffle_env.get_agent_location()
            obs_cam = unshuffle_task.unshuffle_env.observation
            rgb = obs_cam[0]
            depth = obs_cam[1]
            event = convert_rearrange_obs_to_event(obs_agent, rgb, depth)
            init_pos = event.metadata['agent']['position']
            init_rot = event.metadata['agent']['rotation']['y']
            agent.navigation.init_navigation(bounds=None, init_pos=init_pos, init_rot=init_rot)
            agent.init_success_checker(rgb, unshuffle_task.unshuffle_env.controller)
            # For the first step
            action_successful = True
            agent.update_navigation_obs(rgb, depth, action_successful)
            timestep += 1

        if (agent.navigation.explorer.goal.category != "cover" and not rearrange_scene):
            # The agent has covered the scene and collected
            # observations, now process them and perform rearrangement
            rearrange_scene = True
            agent.navigation.explorer.goal.category = 'point_nav'
        if not rearrange_scene:
            # navigate through the scene and collect plausible regions of change
            action, param = agent.navigation.act()
            if action == "Pass":
                exploring = False
                num_sampled += 1
                try:
                    if not inds_explore:
                        ind_i, ind_j = agent.navigation.get_reachable_map_locations(sample=True)
                        if not ind_i:
                            break
                        inds_explore.append([ind_i, ind_j])
                    else:
                        ind_i, ind_j = inds_explore.pop(0)
                except:
                    break

                agent.navigation.set_point_goal(ind_i, ind_j)
            else:
                action_rearrange = agent.nav_action_to_rearrange_action[action]
                action_ind = agent.action_to_ind[action_rearrange]

                if action_ind == 0:
                    rearrange_scene = True
                    continue

                unshuffle_task.step(action=action_ind)
                timestep += 1

                obs_agent = unshuffle_task.unshuffle_env.get_agent_location()
                obs_cam = unshuffle_task.unshuffle_env.observation
                rgb = obs_cam[0]
                depth = obs_cam[1]
                event = convert_rearrange_obs_to_event(obs_agent, rgb, depth)

                agent.navigation.explorer.add_indices(str([round(event.metadata['agent']['position']['x'], 2),
                                                           round(event.metadata['agent']['position']['z'], 2)]))

                if agent.navigation.explorer.move_sign_x == 0 or agent.navigation.explorer.move_sign_z == 0:
                    agent.navigation.explorer.set_pos_world([round(event.metadata['agent']['position']['x'], 2) - round(init_pos['x'], 2),
                                                           round(event.metadata['agent']['position']['z'], 2) - round(init_pos['z'], 2)])

                action_successful = agent.navigation.success_checker.check_successful_action(rgb)
                agent.update_navigation_obs(rgb, depth, action_successful)
                if action_successful:
                    scene.save_image(event, timestep)
                    c2w = scene.get_c2w_transformation(event)
                    out = gaussian_model.load_virtual_camera(c2w, timestep)
                    depth_rend = gaussian_model.render_depth_vis(c2w, timestep)
                    cv2.imwrite(os.path.join(run_path, "depth", "d_{}.png".format(timestep)), depth_rend)
                    depth = (depth - depth.min())/(depth.max() - depth.min())
                    depth = (depth*255).astype(np.uint8)
                    cv2.imwrite(os.path.join(run_path, "depth_orig", "d_{}.png".format(timestep)), depth)
                    _, mask_all = matcher.match_frames(event.frame, out['image'].detach().cpu().numpy(), timestep)
                    masks = matcher.delineate_masks(mask_all)
                    agent.reason_about_change(evt=event,
                                              sim_image=rgb,
                                              rendered_image=out['image'].detach().clone(),
                                              rendered_lang=None,
                                              masks=masks,
                                              step=timestep)
                    # agent.reason_about_change_gsam(gsam=gsam,
                    #                                evt=event,
                    #                                sim_image=rgb,
                    #                                rendered_image=out['image'].detach().clone(),
                    #                                rendered_lang=None,
                    #                                masks=masks,
                    #                                step=timestep)
        else:

            # dilate the obstacle map
            agent.navigation.explorer.mapper.dilate_obstacles(iter = 6)

            # navigate to the locations where change was detected
            # and perform rearrangement
            del matcher
            del dense_feature_matcher
            #agent.reinit_clip_cpu()
            rearrange_dict, open_list, open_list_rend = agent.postprocess_detections(sam_predictor)
            torch.cuda.empty_cache()

            placed_loc = []

            for rend_idx in rearrange_dict:
                if timestep > 1450:
                    time_over = True
                    unshuffle_task.step(action=0)
                    break
                if True:
                    sim_idx = rearrange_dict[rend_idx]
                    replace_object = False
                    false_pick = False
                    # Iterate through a pair of pick and place indices
                    for i_ in range(2):
                        if timestep > 1450:
                            time_over = True
                            unshuffle_task.step(action=0)
                            break
                        agent.navigation.explorer.reinit_act_queue()
                        reached = False
                        dense_feature_matcher = Dinov2Matcher()
                        matcher = Runner(matcher=dense_feature_matcher,
                                         config=config)
                        once = True

                        skip_place = False
                        while not reached:
                            if timestep > 1450:
                                time_over = True
                                reached = True
                                unshuffle_task.step(action=0)
                                break
                            if i_ == 0 and once:
                                # Retrieving the center (world frame) of the misplaced object or
                                center = agent.objects_sim[sim_idx].center_accurate
                                agent.navigation.explorer.place_loc_rend = False
                                once = False
                            elif i_ == 1 and once:
                                # Retrieving the region (world frame), to place the object
                                center = agent.objects_rend[rend_idx].object_pos_world[0]
                                agent.navigation.explorer.place_loc_rend = True
                                agent.navigation.explorer.failed_action_place_count = 0
                                once = False

                            if skip_place:
                                reached = True
                                break

                            # pos_world = {'x': agent.navigation.explorer.move_sign_x * (pos['x'] - init_pos['x']),
                            #              'z': agent.navigation.explorer.move_sign_z * (pos['z'] - init_pos['z'])}\

                            pos_world = {'x': agent.navigation.explorer.move_sign_x * (center[0] - init_pos['x']),
                                         'z': agent.navigation.explorer.move_sign_z * (center[2] - init_pos['z'])}

                            tasks = agent.objects_sim[sim_idx].task
                            task_pick = 0
                            for task in tasks:
                                if task == 'pick':
                                    task_pick += 1
                            task_pick_percent = task_pick / len(tasks)
                            if task_pick_percent >= 0.5:
                                task_obj = 'pick'
                            else:
                                task_obj = 'open'

                            # updating viz
                            agent.navigation.explorer.add_indices(str(rend_idx) + " " + str(sim_idx) + " " + task_obj)

                            map_pos = agent.navigation.explorer.mapper.get_position_on_map_from_aithor_position(pos_world)

                            # if pick location is near to previously placed object, there is a good chance that its a wrong pick
                            if i_ == 0:
                                if len(placed_loc) != 0:
                                    min_dist = 1000000
                                    for map_loc_ in placed_loc:
                                        d_ = np.linalg.norm(map_pos - map_loc_)
                                        if d_ < min_dist:
                                            min_dist = d_

                                    if min_dist < 4:
                                        reached = True
                                        false_pick = True
                                        skip_place = True
                                        break

                            action, param = agent.navigation.act(point_goal=True,
                                                                 goal_loc_map=map_pos,
                                                                 y_loc=center[1])

                            action_rearrange = agent.nav_action_to_rearrange_action[action]

                            # if we have reached the location then navigate to the receptacle
                            if action_rearrange == "done":
                                if i_ == 0:
                                    reached = True
                                    break
                                else:
                                    if agent.navigation.explorer.failed_action_place_count == 20:
                                        # place it back to its original location
                                        print("replace object")
                                        agent.navigation.explorer.failed_action_place_count = 0
                                        replace_object = True
                                        agent.navigation.explorer.reinit_act_queue()
                                        reached = True
                                        break
                                    else:
                                        reached = True
                                        break


                            action_ind = agent.action_to_ind[action_rearrange]
                            unshuffle_task.step(action=action_ind)
                            timestep += 1

                            obs_agent = unshuffle_task.unshuffle_env.get_agent_location()
                            obs_cam = unshuffle_task.unshuffle_env.observation
                            rgb = obs_cam[0]
                            depth = obs_cam[1]
                            event = convert_rearrange_obs_to_event(obs_agent, rgb, depth)

                            agent.navigation.explorer.add_indices(str([round(event.metadata['agent']['position']['x'], 2),
                                                                       round(event.metadata['agent']['position']['z'], 2)]))

                            pcd_frame, _ = agent.get_pcd(event)
                            # pcd_frame = pcd_frame.reshape(-1, 3)
                            # pcdx = -pcd_frame[:, 1]
                            # pcdz = -pcd_frame[:, 0]
                            # pcd_new = np.stack((pcdx, pcdz), axis=1)
                            #
                            # pcd_map = agent.navigation.explorer.mapper.get_goal_position_on_map(pcd_new)
                            # agent.navigation.explorer.mapper.add_obstacles_map_from_pcd(pcd_map)

                            # save image if the agent has reached the location
                            if action_rearrange == "look_down":
                                scene.save_image_path(event,
                                                      str(rend_idx) + "_" + str(sim_idx),
                                                      os.path.join(run_path, 'reached/'))

                            action_successful = agent.navigation.success_checker.check_successful_action(rgb)
                            agent.update_navigation_obs(rgb, depth, action_successful)

                        if i_ == 0:
                            if not false_pick:
                                print("picking")

                                rows, cols = np.where(agent.objects_sim[sim_idx].mask)
                                y_min, y_max = rows.min(), rows.max()
                                x_min, x_max = cols.min(), cols.max()

                                current_frame = rgb
                                pcd_current_frame = pcd_frame
                                pcd_current_frame = pcd_current_frame.reshape(-1, 3)

                                dist = np.linalg.norm(pcd_current_frame - agent.objects_sim[sim_idx].center_accurate, axis=-1)
                                dist = dist.reshape(-1).reshape(rgb.shape[0], rgb.shape[1])
                                min_index = np.argmin(dist)
                                row_index, col_index = np.unravel_index(min_index, dist.shape)

                                y_padding = int((y_max - y_min)/2)
                                x_padding = int((x_max - x_min)/2)
                                y_min = max(row_index - y_padding, 0)
                                y_max = min(row_index + y_padding, rgb.shape[0])
                                x_min = max(col_index - x_padding, 0)
                                x_max = min(col_index + x_padding, rgb.shape[1])
                                bbox = [x_min, y_min, x_max, y_max]

                                # Getting a better mask
                                rgb_ = copy.deepcopy(rgb)
                                masks_sam = sam_predictor.get_mask_all(rgb, bbox)
                                torch.cuda.empty_cache()

                                best_mask_idx = 0
                                best_cs = 0
                                for j_ in range(masks_sam.shape[0]):
                                    mask_this = masks_sam[j_]
                                    cropped_image_this = agent.crop_image_along_mask(rgb, mask_this)
                                    crop_ft_this = agent.clip_ft_extractor.tokenize_image(cropped_image_this)
                                    cs_this = torch.nn.functional.cosine_similarity(crop_ft_this, agent.objects_sim[sim_idx].clip_ft[0])
                                    if cs_this > best_cs:
                                        best_cs = cs_this
                                        best_mask_idx = j_

                                mask_sam = masks_sam[best_mask_idx]
                                # picking up this object
                                mask_sam_copy = copy.deepcopy(mask_sam)
                                mask_sam = (mask_sam * 255).astype(np.uint8)
                                kernel = np.ones((3, 3), np.uint8)
                                eroded_mask = cv2.erode(mask_sam, kernel, iterations=2)

                                rows, cols = np.where(eroded_mask != 0)
                                if len(rows) != 0 and len(cols) != 0:
                                    y_sel = rows[int(len(rows) / 2)]
                                    x_sel = cols[int(len(cols) / 2)]
                                    h, w, _ = event.frame.shape
                                    percent_x = x_sel/w
                                    percent_y = y_sel/h

                                    # performing pickup action
                                    unshuffle_task.unshuffle_env.pickup_object(percent_x, percent_y)
                                    timestep += 1

                                    obs_agent = unshuffle_task.unshuffle_env.get_agent_location()
                                    obs_cam = unshuffle_task.unshuffle_env.observation
                                    rgb = obs_cam[0]
                                    depth = obs_cam[1]
                                    event = convert_rearrange_obs_to_event(obs_agent, rgb, depth)
                                    action_successful = agent.navigation.success_checker.check_successful_action(rgb)
                                    agent.update_navigation_obs(rgb, depth, action_successful)
                                    if not action_successful:
                                        skip_place = True
                                        print("skip place")
                                else:
                                    print("Object already moved or No object to move at this location")

                                rgb_copy = copy.deepcopy(rgb_)
                                mask_copy = copy.deepcopy(mask_sam_copy)
                                rows, cols = np.where(mask_copy)
                                y_min, y_max = rows.min(), rows.max()
                                x_min, x_max = cols.min(), cols.max()
                                ind_x, ind_y = (x_min + x_max) / 2, (y_min + y_max) / 2
                                rgb_copy[mask_copy] = (rgb_copy[mask_copy] * 0.65 + np.array([0, 255, 0]) * 0.35).astype(np.uint8)
                                bgr_img = cv2.cvtColor(rgb_copy, cv2.COLOR_RGB2BGR)
                                bgr_img = cv2.putText(bgr_img, "*", org=(int(ind_x), int(ind_y)),
                                                      fontFace=cv2.FONT_HERSHEY_PLAIN, color=(255, 0, 0), fontScale=1)
                                bgr_img = cv2.rectangle(bgr_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=(0, 0, 255), thickness=1)
                                cv2.imwrite(os.path.join(run_path, 'pick/' + str(sim_idx) + "_" + str(x_padding) + "_" + str(y_padding) + '.jpg'), bgr_img)

                                action_successful = agent.navigation.success_checker.check_successful_action(rgb)
                                agent.update_navigation_obs(rgb, depth, action_successful)

                                # TODO: if pickup fails, dont undergo place action
                            else:
                                false_pick = False

                        if not replace_object:
                            if i_ == 1:
                                # PLACE the object
                                # TODO: implement place
                                print("placing")
                                if not skip_place:
                                    unshuffle_task.unshuffle_env.drop_held_object_with_snap()
                                    timestep += 1
                                    placed_loc.append(map_pos)
                                del matcher
                                del dense_feature_matcher
                            torch.cuda.empty_cache()
                        else:
                            break

                        # reorient head
                        reoriented = False
                        agent.navigation.explorer.reinit_act_queue()
                        while not reoriented:
                            if timestep > 1400:
                                time_over = True
                                reoriented = True
                                unshuffle_task.step(action=0)
                                break
                            print("reorienting head")
                            if i_ == 0:
                                agent.navigation.explorer.reorient_head_flag_fn(pick_flag=True)
                            else:
                                if not replace_object:
                                    if i_ == 1:
                                        agent.navigation.explorer.reorient_head_flag_fn(pick_flag=False)

                            action, param = agent.navigation.act(point_goal=True,
                                                                 goal_loc_map=None,
                                                                 y_loc=None)

                            action_rearrange = agent.nav_action_to_rearrange_action[action]

                            # if we have reached the location then navigate to the receptacle
                            if action_rearrange == "done":
                                reoriented = True
                                break

                            action_ind = agent.action_to_ind[action_rearrange]
                            unshuffle_task.step(action=action_ind)
                            timestep += 1

                            obs_agent = unshuffle_task.unshuffle_env.get_agent_location()
                            obs_cam = unshuffle_task.unshuffle_env.observation
                            rgb = obs_cam[0]
                            depth = obs_cam[1]
                            event = convert_rearrange_obs_to_event(obs_agent, rgb, depth)

                            action_successful = agent.navigation.success_checker.check_successful_action(rgb)
                            agent.update_navigation_obs(rgb, depth, action_successful)

                        if skip_place:
                            break

                    if replace_object:
                        print("replacing object")
                        agent.navigation.explorer.place_loc_rend = True
                        agent.navigation.explorer.failed_action_place_count = 0
                        agent.navigation.explorer.reinit_act_queue()
                        reached = False
                        while not reached:
                            if timestep > 1400:
                                time_over = True
                                reached = True
                                unshuffle_task.step(action=0)
                                break
                            center = agent.objects_sim[sim_idx].center_accurate
                            agent.navigation.explorer.place_loc_rend = True
                            pos_world = {'x': agent.navigation.explorer.move_sign_x * (center[0] - init_pos['x']),
                                         'z': agent.navigation.explorer.move_sign_z * (center[2] - init_pos['z'])}

                            tasks = agent.objects_sim[sim_idx].task
                            task_pick = 0
                            for task in tasks:
                                if task == 'pick':
                                    task_pick += 1
                            task_pick_percent = task_pick / len(tasks)
                            if task_pick_percent >= 0.5:
                                task_obj = 'pick'
                            else:
                                task_obj = 'open'

                            # updating viz
                            agent.navigation.explorer.add_indices(str(rend_idx) + " " + str(sim_idx) + " " + task_obj)

                            map_pos = agent.navigation.explorer.mapper.get_position_on_map_from_aithor_position(
                                pos_world)

                            action, param = agent.navigation.act(point_goal=True,
                                                                 goal_loc_map=map_pos,
                                                                 y_loc=center[1])

                            action_rearrange = agent.nav_action_to_rearrange_action[action]


                            # if we have reached the location then navigate to the receptacle
                            if action_rearrange == "done":
                                reached = True
                                break

                            action_ind = agent.action_to_ind[action_rearrange]
                            unshuffle_task.step(action=action_ind)
                            timestep += 1

                            obs_agent = unshuffle_task.unshuffle_env.get_agent_location()
                            obs_cam = unshuffle_task.unshuffle_env.observation
                            rgb = obs_cam[0]
                            depth = obs_cam[1]
                            event = convert_rearrange_obs_to_event(obs_agent, rgb, depth)

                            agent.navigation.explorer.add_indices(
                                str([round(event.metadata['agent']['position']['x'], 2),
                                     round(event.metadata['agent']['position']['z'], 2)]))

                            pcd_frame, _ = agent.get_pcd(event)

                            # save image if the agent has reached the location
                            if action_rearrange == "look_down":
                                scene.save_image_path(event,
                                                      str(rend_idx) + "_" + str(sim_idx),
                                                      os.path.join(run_path, 'reached/'))

                            action_successful = agent.navigation.success_checker.check_successful_action(rgb)
                            agent.update_navigation_obs(rgb, depth, action_successful)
                        unshuffle_task.unshuffle_env.drop_held_object()

                        # reorient head after replace
                        reoriented = False
                        while not reoriented:
                            if timestep > 1400:
                                time_over = True
                                reoriented = True
                                unshuffle_task.step(action=0)
                                break
                            print("reorienting head after replace")
                            agent.navigation.explorer.reorient_head_flag_fn(pick_flag=False)

                            action, param = agent.navigation.act(point_goal=True,
                                                                 goal_loc_map=None,
                                                                 y_loc=None)

                            action_rearrange = agent.nav_action_to_rearrange_action[action]

                            # if we have reached the location then navigate to the receptacle
                            if action_rearrange == "done":
                                reoriented = True

                            action_ind = agent.action_to_ind[action_rearrange]
                            unshuffle_task.step(action=action_ind)
                            timestep += 1

            agent.navigation.explorer.place_loc_rend = False
            agent.navigation.explorer.open_mode = True

            print("pick and place complete")

            for i_open in range(2):
                if timestep > 1400:
                    time_over = True
                    unshuffle_task.step(action=0)
                    break
                if i_open == 0:
                    open_list_ = open_list
                else:
                    open_list_ = open_list_rend
                for open_idx in open_list_:
                    if timestep > 1400:
                        time_over = True
                        unshuffle_task.step(action=0)
                        break

                    agent.navigation.explorer.reinit_act_queue()
                    if i_open == 0:
                        object_open = agent.objects_sim[open_idx]
                    else:
                        object_open = agent.objects_rend[open_idx]
                    center = object_open.object_pos_world[0]
                    pos_world = {'x': agent.navigation.explorer.move_sign_x * (center[0] - init_pos['x']),
                                 'z': agent.navigation.explorer.move_sign_z * (center[2] - init_pos['z'])}
                    map_pos = agent.navigation.explorer.mapper.get_position_on_map_from_aithor_position(pos_world)

                    reached = False
                    while not reached:
                        if timestep > 1400:
                            time_over = True
                            reached = True
                            unshuffle_task.step(action=0)
                            break

                        action, param = agent.navigation.act(point_goal=True,
                                                             goal_loc_map=map_pos)
                        action_rearrange = agent.nav_action_to_rearrange_action[action]

                        if action_rearrange == "done":
                            reached = True
                            break

                        action_ind = agent.action_to_ind[action_rearrange]
                        unshuffle_task.step(action=action_ind)
                        timestep += 1

                        obs_agent = unshuffle_task.unshuffle_env.get_agent_location()
                        obs_cam = unshuffle_task.unshuffle_env.observation
                        rgb = obs_cam[0]
                        depth = obs_cam[1]
                        event = convert_rearrange_obs_to_event(obs_agent, rgb, depth)

                        agent.navigation.explorer.add_indices(str([round(event.metadata['agent']['position']['x'], 2),
                                                                   round(event.metadata['agent']['position']['z'], 2)]))
                        pcd_frame, _ = agent.get_pcd(event)
                        action_successful = agent.navigation.success_checker.check_successful_action(rgb)
                        agent.update_navigation_obs(rgb, depth, action_successful)

                    current_frame = rgb
                    pcd_current_frame = pcd_frame
                    pcd_current_frame = pcd_current_frame.reshape(-1, 3)

                    dist = np.linalg.norm(pcd_current_frame - object_open.center_accurate, axis=-1)
                    dist = dist.reshape(-1).reshape(rgb.shape[0], rgb.shape[1])
                    min_index = np.argmin(dist)
                    row_index, col_index = np.unravel_index(min_index, dist.shape)

                    padding = 10
                    y_min = max(row_index - padding, 0)
                    y_max = min(row_index + padding, rgb.shape[0])
                    x_min = max(col_index - padding, 0)
                    x_max = min(col_index + padding, rgb.shape[1])
                    bbox = [x_min, y_min, x_max, y_max]

                    # performing open action in steps (0.1) until the sim image matches the rendered image
                    completed = False
                    step = 1
                    sim_score = {}
                    best_step = 0
                    best_score = 0

                    mask_sam = sam_predictor.get_mask(rgb, bbox)
                    rgb_copy = copy.deepcopy(rgb)
                    rgb_copy[mask_sam] = (rgb_copy[mask_sam] * 0.65 + np.array([0, 255, 0]) * 0.35).astype(np.uint8)
                    bgr_img = cv2.cvtColor(rgb_copy, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(os.path.join(run_path, "open/{}_init.png".format(open_idx)), bgr_img)

                    torch.cuda.empty_cache()
                    mask_sam_copy = copy.deepcopy(mask_sam)
                    mask_sam = (mask_sam * 255).astype(np.uint8)
                    kernel = np.ones((3, 3), np.uint8)
                    eroded_mask = cv2.erode(mask_sam, kernel, iterations=1)

                    cropped_image = agent.crop_image_along_mask(rgb, mask_sam_copy)
                    crop_ft = agent.clip_ft_extractor.tokenize_image(cropped_image)

                    rows, cols = np.where(eroded_mask != 0)
                    if len(rows) != 0 and len(cols) != 0:
                        y_sel = rows[int(len(rows) / 2)]
                        x_sel = cols[int(len(cols) / 2)]
                        h, w, _ = event.frame.shape
                        percent_x = x_sel / w
                        percent_y = y_sel / h
                    else:
                        # if you cant get the better mask, then use the dino patch center
                        y_sel = (bbox[1] + bbox[3]) / 2
                        x_sel = (bbox[0] + bbox[2]) / 2
                        h, w, _ = event.frame.shape
                        percent_x = x_sel / w
                        percent_y = y_sel / h
                    agent.navigation.explorer.reinit_act_queue()
                    while not completed:
                        if timestep > 1400:
                            time_over = True
                            completed = True
                            unshuffle_task.step(action=0)
                            break
                        # Getting a better mask

                        if step != 0:
                            padding = 70
                            bbox[0] = bbox[0] - padding
                            bbox[2] = bbox[2] + padding
                            bbox[1] = bbox[1] - padding
                            bbox[3] = bbox[3] + padding
                            masks_sam = sam_predictor.get_mask_all(rgb, bbox)
                            #masks_sam = sam_predictor.get_mask_all(rgb)
                            best_mask_idx = 0
                            best_cs = 0
                            for j in range(masks_sam.shape[0]):
                                mask_this = masks_sam[j]
                                cropped_image_this = agent.crop_image_along_mask(rgb, mask_this)
                                crop_ft_this = agent.clip_ft_extractor.tokenize_image(cropped_image_this)
                                cs_this = torch.nn.functional.cosine_similarity(crop_ft_this, crop_ft)
                                if cs_this > best_cs:
                                    best_cs = cs_this
                                    best_mask_idx = j

                            rgb_copy = copy.deepcopy(rgb)
                            rgb_copy[masks_sam[best_mask_idx]] = (rgb_copy[masks_sam[best_mask_idx]] * 0.65 + np.array([0, 255, 0]) * 0.35).astype(np.uint8)
                            bgr_img = cv2.cvtColor(rgb_copy, cv2.COLOR_RGB2BGR)
                            bgr_img = cv2.rectangle(bgr_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=(0, 0, 255),
                                                    thickness=1)

                            mask_sam_ = (masks_sam[best_mask_idx] * 255).astype(np.uint8)
                            kernel = np.ones((3, 3), np.uint8)
                            eroded_mask_ = cv2.erode(mask_sam_, kernel, iterations=1)

                            rows, cols = np.where(eroded_mask_ != 0)
                            if len(rows) != 0 and len(cols) != 0:
                                y_sel = rows[int(len(rows) / 2)]
                                x_sel = cols[int(len(cols) / 2)]
                                h, w, _ = event.frame.shape
                                percent_x = x_sel / w
                                percent_y = y_sel / h
                            else:
                                # if you cant get the better mask, then use the dino patch center
                                y_sel = (bbox[1] + bbox[3]) / 2
                                x_sel = (bbox[0] + bbox[2]) / 2
                                h, w, _ = event.frame.shape
                                percent_x = x_sel / w
                                percent_y = y_sel / h

                        bgr_img = cv2.putText(bgr_img, "*", org=(int(x_sel), int(y_sel)),
                                              fontFace=cv2.FONT_HERSHEY_PLAIN, color=(255, 0, 0), fontScale=1)
                        cv2.imwrite(os.path.join(run_path, "open/{}_{}.png".format(open_idx, step)), bgr_img)

                        unshuffle_task.unshuffle_env.open_object(percent_x, percent_y, openness=step)
                        timestep += 1

                        # Get observation from simulation
                        obs_agent = unshuffle_task.unshuffle_env.get_agent_location()
                        obs_cam = unshuffle_task.unshuffle_env.observation
                        rgb = obs_cam[0]
                        depth = obs_cam[1]
                        event = convert_rearrange_obs_to_event(obs_agent, rgb, depth)
                        action_successful = agent.navigation.success_checker.check_successful_action(rgb)

                        agent.update_navigation_obs(rgb, depth, action_successful)

                        # Get observation from rendered model of the scene
                        c2w = scene.get_c2w_transformation(event)
                        out = gaussian_model.load_virtual_camera(c2w, timestep)
                        rendered_image = out['image'].detach().clone().cpu().numpy()
                        rendered_image = (rendered_image * 255).astype(np.uint8)

                        # Crop out an image with a large padding
                        padding = 20
                        bbox[0] = bbox[0] - padding
                        bbox[2] = bbox[2] + padding
                        bbox[1] = bbox[1] - padding
                        bbox[3] = bbox[3] + padding

                        x_min = max(x_min, 0)
                        x_max = min(x_max, w)
                        y_min = max(y_min, 0)
                        y_max = min(y_max, h)
                        bbox = [x_min, y_min, x_max, y_max]

                        cropped_image_sim = rgb[y_min:y_max, x_min:x_max]
                        cropped_image_rend = rendered_image[y_min:y_max, x_min:x_max]

                        crop_ft_sim = agent.clip_ft_extractor.tokenize_image(cropped_image_sim)
                        crop_ft_rend = agent.clip_ft_extractor.tokenize_image(cropped_image_rend)

                        cs = torch.nn.functional.cosine_similarity(crop_ft_sim, crop_ft_rend)

                        sim_score[step] = cs
                        step -= 0.1
                        if step < 0:
                            # select the best step
                            for step_ in sim_score:
                                if sim_score[step_] > best_score:
                                    best_score = sim_score[step_]
                                    best_step = step_
                            completed = True

                    # Use the best step computed above to open the object
                    unshuffle_task.unshuffle_env.open_object(percent_x, percent_y, openness=best_step)
            print("open complete")
            unshuffle_task.step(action=0)

    if time_over:
        print("time over time over")

    del sam_predictor
    del agent
    del gaussian_model
    del scene
    torch.cuda.empty_cache()

    metrics = unshuffle_task.metrics()
    print(f"Both phases complete, metrics: '{metrics}'")
    task_info = metrics["task_info"]
    del metrics["task_info"]
    my_leaderboard_submission[task_info["unique_id"]] = {**task_info, **metrics}

    save_path = "rearrange/metrics/submission_{}.json.gz".format(two_phase_rgb_task_sampler.current_task_spec.unique_id)
    if os.path.exists(os.path.dirname(save_path)):
        print(f"Saving example submission file to {save_path}")
        submission_json_str = json.dumps(my_leaderboard_submission, cls=NumpyJSONEncoder)
        with gzip.open(save_path, "w") as f:
            f.write(submission_json_str.encode("utf-8"))
    else:
        print("Metrics not saved, as the location does not exist")

    exit()

# saving the metrics
save_path = "rearrange/metrics/submission.json.gz"
if os.path.exists(os.path.dirname(save_path)):
    print(f"Saving example submission file to {save_path}")
    submission_json_str = json.dumps(my_leaderboard_submission, cls=NumpyJSONEncoder)
    with gzip.open(save_path, "w") as f:
        f.write(submission_json_str.encode("utf-8"))
else:
    print("Metrics not saved, as the location does not exist")

two_phase_rgb_task_sampler.close()