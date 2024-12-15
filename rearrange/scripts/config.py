from navigation.utils.arguments import args
import os

class GaussianConfig():
    def __init__(self, width, height, run_path, dataset_path, dataset_name, diff_splat_train=False):
        self.run_path = run_path
        self.scene_path = dataset_path
        self.gs_checkpoint_path = "output/"
        self.iteration_to_load = 7000
        self.use_eval_split = False
        self.n_skip_images_for_eval_split = 8
        self.use_white_background = False
        self.coarse_model_path = "output/coarse/walkthrough/sugarcoarse_3Dgs7000_sdfestim02_sdfnorm02/15000.pt"
        # self.coarse_model_path = "/home/nune/gaussian_splatting/lgsplat-mesh/output/WorldModel/splat_update/coarse/3000.pt"
        self.width = width # set in accordance with AI2THOR
        self.height = height  # set in accordance with AI2THOR

        # Match strategy
        # True -> Greedy Matching
        # False -> Hungarian Matching
        self.greedy_strat = False

        # decoder
        self.decoder_device = "cpu"
        self.autoenc_ckpt = f"ckpt/{dataset_name}/best_ckpt.pth"
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

        # surface align
        self.surface_align = False

        # DINOV2 and CLIP
        self.viz_dense_features = True
        self.save_dino_frames = os.path.join(run_path, "dino_frames")
        self.clip_device = "cuda:0"
        self.dilate_object_mask = False
        self.dilate_iterations = 2
        self.patch_size = 14

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

        # SAM configs
        self.sam_checkpoint = "ckpt/sam_vit_h_4b8939.pth"
        self.sam_model_type = "vit_h"
        self.sam_device = "cpu"

        # gdino
        self.gdino_device = "cuda:0"

        # vis pcd after data collection
        self.pcd_vis = False

        # get mesh of the env
        self.get_mesh = False

        self.walkthrough = True

        # diff splat
        self.second_splat = False
        self.diff_splat_train = diff_splat_train
        if self.diff_splat_train:
            self.coarse_model_path = "output/coarse_lang/sugarcoarse_3Dgs7000_sdfestim02_sdfnorm02/40000.pt"
            self.include_feature = True
            self.lang_ft_dim = 3
        else:
            self.include_feature = False
            self.lang_ft_dim = 0

        # voxalized map
        self.map_config = MapConfig()

        # Autoencoder
        self.encoder_dims = [256, 128, 64, 32, 3]
        self.decoder_dims = [16, 32, 64, 128, 256, 256, 512]
        self.decoder_device = "cuda:0"

class MapConfig:
    def __init__(self):
        self.voxel_size = 0.02
        self.grid_size_m = (15, 15, 15)
        self.fly_y = int((1.57/self.voxel_size) + (self.grid_size_m[1]/(self.voxel_size*2)))


actions = [
           'done', 'move_ahead', 'move_left', 'move_right', 'move_back', 'rotate_right', 'rotate_left',
           'stand', 'crouch', 'look_up', 'look_down', 'drop_held_object_with_snap', 'open_by_type_blinds',
           'open_by_type_cabinet', 'open_by_type_drawer', 'open_by_type_fridge', 'open_by_type_laundry_hamper',
           'open_by_type_microwave', 'open_by_type_safe', 'open_by_type_shower_curtain', 'open_by_type_shower_door',
           'open_by_type_toilet', 'pickup_alarm_clock', 'pickup_aluminum_foil', 'pickup_apple', 'pickup_baseball_bat',
           'pickup_basket_ball', 'pickup_book', 'pickup_boots', 'pickup_bottle', 'pickup_bowl', 'pickup_box',
           'pickup_bread', 'pickup_butter_knife', 'pickup_c_d', 'pickup_candle', 'pickup_cell_phone', 'pickup_cloth',
           'pickup_credit_card', 'pickup_cup', 'pickup_dish_sponge', 'pickup_dumbbell', 'pickup_egg', 'pickup_footstool',
           'pickup_fork', 'pickup_hand_towel', 'pickup_kettle', 'pickup_key_chain', 'pickup_knife', 'pickup_ladle',
           'pickup_laptop', 'pickup_lettuce', 'pickup_mug', 'pickup_newspaper', 'pickup_pan', 'pickup_paper_towel_roll',
           'pickup_pen', 'pickup_pencil', 'pickup_pepper_shaker', 'pickup_pillow', 'pickup_plate', 'pickup_plunger',
           'pickup_pot', 'pickup_potato', 'pickup_remote_control', 'pickup_salt_shaker', 'pickup_scrub_brush',
           'pickup_soap_bar', 'pickup_soap_bottle', 'pickup_spatula', 'pickup_spoon', 'pickup_spray_bottle',
           'pickup_statue', 'pickup_table_top_decor', 'pickup_teddy_bear', 'pickup_tennis_racket', 'pickup_tissue_box',
           'pickup_toilet_paper', 'pickup_tomato', 'pickup_towel', 'pickup_vase', 'pickup_watch', 'pickup_watering_can',
           'pickup_wine_bottle'
        ]