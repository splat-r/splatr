import os
import numpy as np
import torch
import open3d as o3d
from pytorch3d.loss import mesh_laplacian_smoothing, mesh_normal_consistency
from pytorch3d.transforms import quaternion_apply, quaternion_invert
from sugar_scene.gs_model import GaussianSplattingWrapper, fetchPly
from sugar_scene.sugar_model import SuGaR
from sugar_scene.sugar_optimizer import OptimizationParams, SuGaROptimizer
from sugar_scene.sugar_densifier import SuGaRDensifier
from sugar_utils.loss_utils import ssim, l1_loss, l2_loss
import torchvision
from rich.console import Console
import time
import pickle
from torch.utils.tensorboard import SummaryWriter
import cv2
import random
from copy import deepcopy
from tqdm import trange

def coarse_extraction_with_lang(args):
    CONSOLE = Console(width=120)

    # ====================Parameters====================

    num_device = args.gpu
    detect_anomaly = False

    # -----Data parameters-----
    downscale_resolution_factor = 1  # 2, 4

    # -----Model parameters-----
    use_eval_split = True
    n_skip_images_for_eval_split = 8

    freeze_gaussians = False
    initialize_from_trained_3dgs = True  # True or False
    if initialize_from_trained_3dgs:
        prune_at_start = False
        start_pruning_threshold = 0.5
    no_rendering = freeze_gaussians

    n_points_at_start = None  # If None, takes all points in the SfM point cloud

    learnable_positions = False  # True in 3DGS
    use_same_scale_in_all_directions = False  # Should be False
    sh_levels = 4

    # -----Radiance Mesh-----
    triangle_scale = 1.

    # -----Rendering parameters-----
    compute_color_in_rasterizer = False  # TODO: Try True

    # -----Optimization parameters-----

    # Learning rates and scheduling
    num_iterations = 50_000  # Changed

    spatial_lr_scale = None
    position_lr_init = 0.00016
    position_lr_final = 0.0000016
    position_lr_delay_mult = 0.01
    position_lr_max_steps = 30_000
    feature_lr = 0.0025
    opacity_lr = 0.05
    scaling_lr = 0.005
    rotation_lr = 0.001
    language_feature_lr = 0.01

    # Densifier and pruning
    heavy_densification = False
    if initialize_from_trained_3dgs:
        densify_from_iter = 500 + 99999  # 500  # Maybe reduce this, since we have a better initialization?
        densify_until_iter = 7000 - 7000  # 7000
    else:
        densify_from_iter = 500  # 500  # Maybe reduce this, since we have a better initialization?
        densify_until_iter = 7000  # 7000

    if heavy_densification:
        densification_interval = 50  # 100
        opacity_reset_interval = 3000  # 3000

        densify_grad_threshold = 0.0001  # 0.0002
        densify_screen_size_threshold = 20
        prune_opacity_threshold = 0.005
        densification_percent_distinction = 0.01
    else:
        densification_interval = 100  # 100
        opacity_reset_interval = 3000  # 3000

        densify_grad_threshold = 0.0002  # 0.0002
        densify_screen_size_threshold = 20
        prune_opacity_threshold = 0.005
        densification_percent_distinction = 0.01

    # Data processing and batching
    n_images_to_use_for_training = -1  # If -1, uses all images

    train_num_images_per_batch = 1  # 1 for full images

    # Loss functions
    loss_function = 'l1+dssim'  # 'l1' or 'l2' or 'l1+dssim'
    if loss_function == 'l1+dssim':
        dssim_factor = 0.2

    # Regularization
    enforce_entropy_regularization = False
    if enforce_entropy_regularization:
        start_entropy_regularization_from = 7000
        end_entropy_regularization_at = 9000  # TODO: Change
        entropy_regularization_factor = 0.1

    regularize_sdf = False
    if regularize_sdf:
        beta_mode = 'average'  # 'learnable', 'average' or 'weighted_average'

        start_sdf_regularization_from = 9000
        regularize_sdf_only_for_gaussians_with_high_opacity = False
        if regularize_sdf_only_for_gaussians_with_high_opacity:
            sdf_regularization_opacity_threshold = 0.5

        use_sdf_estimation_loss = True
        enforce_samples_to_be_on_surface = False
        if use_sdf_estimation_loss or enforce_samples_to_be_on_surface:
            sdf_estimation_mode = 'sdf'  # 'sdf' or 'density'
            # sdf_estimation_factor = 0.2  # 0.1 or 0.2?
            samples_on_surface_factor = 0.2  # 0.05

            squared_sdf_estimation_loss = False
            squared_samples_on_surface_loss = False

            normalize_by_sdf_std = False  # False

            start_sdf_estimation_from = 9000  # 7000

            sample_only_in_gaussians_close_to_surface = True
            close_gaussian_threshold = 2.  # 2.

            backpropagate_gradients_through_depth = True  # True

        use_sdf_better_normal_loss = True
        if use_sdf_better_normal_loss:
            start_sdf_better_normal_from = 9000
            # sdf_better_normal_factor = 0.2  # 0.1 or 0.2?
            sdf_better_normal_gradient_through_normal_only = True

        density_factor = 1. / 16.  # 1. / 16.
        if (use_sdf_estimation_loss or enforce_samples_to_be_on_surface) and sdf_estimation_mode == 'density':
            density_factor = 1.
        density_threshold = 1.  # 0.3 * density_factor
        n_samples_for_sdf_regularization = 1_000_000  # 300_000
        sdf_sampling_scale_factor = 1.5
        sdf_sampling_proportional_to_volume = False

    bind_to_surface_mesh = False
    if bind_to_surface_mesh:
        learn_surface_mesh_positions = True
        learn_surface_mesh_opacity = True
        learn_surface_mesh_scales = True
        n_gaussians_per_surface_triangle = 6  # 1, 3, 4 or 6

        use_surface_mesh_laplacian_smoothing_loss = True
        if use_surface_mesh_laplacian_smoothing_loss:
            surface_mesh_laplacian_smoothing_method = "uniform"  # "cotcurv", "cot", "uniform"
            surface_mesh_laplacian_smoothing_factor = 5.  # 0.1

        use_surface_mesh_normal_consistency_loss = True
        if use_surface_mesh_normal_consistency_loss:
            surface_mesh_normal_consistency_factor = 0.1  # 0.1

        densify_from_iter = 999_999
        densify_until_iter = 0
        position_lr_init = 0.00016 * 0.01
        position_lr_final = 0.0000016 * 0.01
        scaling_lr = 0.005
    else:
        surface_mesh_to_bind_path = None

    if regularize_sdf:
        regularize = True
        regularity_knn = 16  # 8 until now
        # regularity_knn = 8
        regularity_samples = -1  # Retry with 1000, 10000
        reset_neighbors_every = 500  # 500 until now
        regularize_from = 7000  # 0 until now
        start_reset_neighbors_from = 7000 + 1  # 0 until now (should be equal to regularize_from + 1?)
        prune_when_starting_regularization = False
    else:
        regularize = False
        regularity_knn = 0
    if bind_to_surface_mesh:
        regularize = False
        regularity_knn = 0

    # Opacity management
    prune_low_opacity_gaussians_at = [9000]
    if bind_to_surface_mesh:
        prune_low_opacity_gaussians_at = [999_999]
    prune_hard_opacity_threshold = 0.5

    # Warmup
    do_resolution_warmup = False
    if do_resolution_warmup:
        resolution_warmup_every = 500
        current_resolution_factor = downscale_resolution_factor * 4.
    else:
        current_resolution_factor = downscale_resolution_factor

    do_sh_warmup = False  # Should be True
    if initialize_from_trained_3dgs:
        do_sh_warmup = False
        sh_levels = 4  # nerfmodel.gaussians.active_sh_degree + 1
        CONSOLE.print("Changing sh_levels to match the loaded model:", sh_levels)
    if do_sh_warmup:
        sh_warmup_every = 1000
        current_sh_levels = 1
    else:
        current_sh_levels = sh_levels

    # -----Log and save-----
    print_loss_every_n_iterations = 50
    save_model_every_n_iterations = 1_000_000
    # save_milestones = [9000, 12_000, 15_000]
    save_milestones = [20000, 25000, 30_000, 35_000, 40_000, 45_000, 50_000, 60_000, 70_000]
    save_image_milestones = [2000, 5_000, 7_000, 10_000, 15_000, 20_000, 25_000, 30_000, 35_000, 40_000, 45_000, 50_000,
                             55_000, 60_000, 65_000, 70_000]

    # ====================End of parameters====================

    if args.output_dir is None:
        if len(args.scene_path.split("/")[-1]) > 0:
            args.output_dir = os.path.join("./output/coarse", args.scene_path.split("/")[-1])
        else:
            args.output_dir = os.path.join("./output/coarse", args.scene_path.split("/")[-2])

    source_path = args.scene_path
    gs_checkpoint_path = args.checkpoint_path
    iteration_to_load = args.iteration_to_load

    sdf_estimation_factor = args.estimation_factor
    sdf_better_normal_factor = args.normal_factor

    sugar_checkpoint_path = f'sugarcoarse_3Dgs{iteration_to_load}_sdfestimXX_sdfnormYY/'
    sugar_checkpoint_path = os.path.join(args.lang_output_dir, sugar_checkpoint_path)
    sugar_checkpoint_path = sugar_checkpoint_path.replace(
        'XX', str(sdf_estimation_factor).replace('.', '')
    ).replace(
        'YY', str(sdf_better_normal_factor).replace('.', '')
    )

    use_eval_split = args.eval
    use_white_background = args.white_background

    ply_path = os.path.join(source_path, "sparse/0/points3D.ply")

    CONSOLE.print("-----Parsed parameters-----")
    CONSOLE.print("Source path:", source_path)
    CONSOLE.print("   > Content:", len(os.listdir(source_path)))
    CONSOLE.print("Gaussian Splatting checkpoint path:", gs_checkpoint_path)
    CONSOLE.print("   > Content:", len(os.listdir(gs_checkpoint_path)))
    CONSOLE.print("SUGAR checkpoint path:", sugar_checkpoint_path)
    CONSOLE.print("Iteration to load:", iteration_to_load)
    CONSOLE.print("Output directory:", args.output_dir)
    CONSOLE.print("SDF estimation factor:", sdf_estimation_factor)
    CONSOLE.print("SDF better normal factor:", sdf_better_normal_factor)
    CONSOLE.print("Eval split:", use_eval_split)
    CONSOLE.print("White background:", use_white_background)
    CONSOLE.print("---------------------------")

    # Setup device
    torch.cuda.set_device(num_device)
    CONSOLE.print("Using device:", num_device)
    device = torch.device(f'cuda:{num_device}')
    CONSOLE.print(torch.cuda.memory_summary())

    torch.autograd.set_detect_anomaly(detect_anomaly)

    # Creates save directory if it does not exist
    os.makedirs(sugar_checkpoint_path, exist_ok=True)

    # ====================Load NeRF model and training data====================

    # Load Gaussian Splatting checkpoint
    CONSOLE.print(f"\nLoading config {gs_checkpoint_path}...")
    if use_eval_split:
        CONSOLE.print("Performing train/eval split...")
    nerfmodel = GaussianSplattingWrapper(
        source_path=source_path,
        output_path=gs_checkpoint_path,
        iteration_to_load=iteration_to_load,
        load_gt_images=True,
        eval_split=use_eval_split,
        eval_split_interval=n_skip_images_for_eval_split,
        white_background=use_white_background,
    )

    CONSOLE.print(f'{len(nerfmodel.training_cameras)} training images detected.')
    CONSOLE.print(f'The model has been trained for {iteration_to_load} steps.')

    if downscale_resolution_factor != 1:
        nerfmodel.downscale_output_resolution(downscale_resolution_factor)
    CONSOLE.print(f'\nCamera resolution scaled to '
                  f'{nerfmodel.training_cameras.gs_cameras[0].image_height} x '
                  f'{nerfmodel.training_cameras.gs_cameras[0].image_width}'
                  )

    # Point cloud
    if initialize_from_trained_3dgs:
        with torch.no_grad():
            print("Initializing model from trained 3DGS...")
            with torch.no_grad():
                sh_levels = int(np.sqrt(nerfmodel.gaussians.get_features.shape[1]))

            from sugar_utils.spherical_harmonics import SH2RGB
            points = nerfmodel.gaussians.get_xyz.detach().float().cuda()
            colors = SH2RGB(nerfmodel.gaussians.get_features[:, 0].detach().float().cuda())
            if prune_at_start:
                with torch.no_grad():
                    start_prune_mask = nerfmodel.gaussians.get_opacity.view(-1) > start_pruning_threshold
                    points = points[start_prune_mask]
                    colors = colors[start_prune_mask]
            n_points = len(points)
    else:
        CONSOLE.print("\nLoading SfM point cloud...")
        # pcd = fetchPly(ply_path)
        # points = torch.tensor(pcd.points, device=nerfmodel.device).float().cuda()
        # colors = torch.tensor(pcd.colors, device=nerfmodel.device).float().cuda()
        points = torch.randn(1000, 3, device=nerfmodel.device)
        colors = torch.rand(1000, 3, device=nerfmodel.device)

        if n_points_at_start is not None:
            n_points = n_points_at_start
            pts_idx = torch.randperm(len(points))[:n_points]
            points, colors = points.to(device)[pts_idx], colors.to(device)[pts_idx]
        else:
            n_points = len(points)

    CONSOLE.print(f"Point cloud generated. Number of points: {len(points)}")

    # Mesh to bind to if needed  TODO
    if bind_to_surface_mesh:
        surface_mesh_to_bind_full_path = os.path.join('./results/meshes/', surface_mesh_to_bind_path)
        CONSOLE.print(f'\nLoading mesh to bind to: {surface_mesh_to_bind_full_path}...')
        o3d_mesh = o3d.io.read_triangle_mesh(surface_mesh_to_bind_full_path)
        CONSOLE.print("Mesh to bind to loaded.")
    else:
        o3d_mesh = None
        learn_surface_mesh_positions = False
        learn_surface_mesh_opacity = False
        learn_surface_mesh_scales = False
        n_gaussians_per_surface_triangle = 1

    if not regularize_sdf:
        beta_mode = None

    # Background tensor if needed
    if use_white_background:
        bg_tensor = torch.ones(3, dtype=torch.float, device=nerfmodel.device)
    else:
        bg_tensor = None

    # ====================Initialize SuGaR model====================
    # Construct SuGaR model
    coarse_model_path = args.coarse_model_path
    coarse_ckpt = torch.load(coarse_model_path, map_location="cpu")
    # print(coarse_ckpt['state_dict']['_points'])
    # print(points)
    # print(coarse_ckpt['state_dict']['_points'].shape, points.shape)
    # print(SH2RGB(coarse_ckpt['state_dict']['_sh_coordinates_dc'][:, 0, :]))
    # print(colors)
    # print(SH2RGB(coarse_ckpt['state_dict']['_sh_coordinates_dc'][:, 0, :]).shape, colors.shape)
    sugar = SuGaR(
        nerfmodel=nerfmodel,
        points=coarse_ckpt['state_dict']['_points'].detach().float().cuda(),
        colors=SH2RGB(coarse_ckpt['state_dict']['_sh_coordinates_dc'][:, 0, :]).detach().float().cuda(),
        initialize=False,
        sh_levels=4,
        triangle_scale=triangle_scale,
        learnable_positions=learnable_positions,
        keep_track_of_knn=regularize,
        knn_to_track=regularity_knn,
        freeze_gaussians=freeze_gaussians,
        beta_mode=None,
        surface_mesh_to_bind=o3d_mesh,
        surface_mesh_thickness=None,
        learn_surface_mesh_positions=learn_surface_mesh_positions,
        learn_surface_mesh_opacity=learn_surface_mesh_opacity,
        learn_surface_mesh_scales=learn_surface_mesh_scales,
        n_gaussians_per_surface_triangle=n_gaussians_per_surface_triangle,
        include_feature=args.include_feature
    )
    sugar.init_lang_ft(args.lang_ft_dim)
    sugar.load_state_dict(coarse_ckpt['state_dict'])

    CONSOLE.print(f'\nSuGaR model has been initialized.')
    CONSOLE.print(sugar)
    CONSOLE.print(f'Number of parameters: {sum(p.numel() for p in sugar.parameters() if p.requires_grad)}')
    CONSOLE.print(f'Checkpoints will be saved in {sugar_checkpoint_path}')

    CONSOLE.print("\nModel parameters:")
    for name, param in sugar.named_parameters():
        CONSOLE.print(name, param.shape, param.requires_grad)

    # Compute scene extent
    cameras_spatial_extent = sugar.get_cameras_spatial_extent()

    map_path = os.path.join(args.processed_data_path, "mapping_dict.pickle")
    with open(map_path, 'rb') as map_file:
        mapping_dict = pickle.load(map_file)
    classifier = torch.nn.Conv2d(args.lang_ft_dim, len(mapping_dict) + 1, kernel_size=1)
    classifier.to(args.mapper_device)
    classifier.load_state_dict(torch.load(args.lang_conv_cp, map_location="cuda:0"))
    classifier.eval()

    lang_ft = deepcopy(sugar.language_feature)
    lang_ft = lang_ft.permute(1, 0)
    lang_ft = lang_ft[..., None]
    sem_lang_classes = classifier((lang_ft.to(args.mapper_device)))
    sem_lang_classes = sem_lang_classes.squeeze(2).permute(1, 0)
    sem_lang_classes = torch.softmax(sem_lang_classes, dim=1)
    classes_gaussian = torch.argmax(sem_lang_classes, dim=1)

    # {1: 'beanbag', 2: 'table', 3: 'toy', 4: 'monitor', 5: 'stroller', 6: 'chair'}
    gaussian_mask = classes_gaussian == 2

    sugar.update_coarse_lang_gaussians(gaussian_mask)
    for name, param in sugar.named_parameters():
        CONSOLE.print(name, param.shape, param.requires_grad)

    indices_cam = torch.arange(len(nerfmodel.training_cameras))
    # shuffled_idx = shuffled_idx[190:]
    train_num_images = len(indices_cam)
    render_path = "output/render_images"
    if not os.path.exists(render_path):
        os.makedirs(render_path)
    for d in trange(train_num_images):
        outputs = sugar.render_image_gaussian_rasterizer(
            camera_indices=indices_cam[d].item(),
            verbose=False,
            bg_color=bg_tensor,
            sh_deg=current_sh_levels - 1,
            sh_rotations=None,
            compute_color_in_rasterizer=compute_color_in_rasterizer,
            compute_covariance_in_rasterizer=True,
            return_2d_radii=True,
            quaternions=None,
            use_same_scale_in_all_directions=use_same_scale_in_all_directions,
            return_opacities=enforce_entropy_regularization,
        )
        language_feature_ = outputs["language_feature_image"]
        torchvision.utils.save_image(language_feature_[:3, :, :],
                                     os.path.join(render_path, 'lang_{}'.format(indices_cam[d].item()) + ".png"))
        pred_rgb = outputs['image'].detach().cpu().numpy()
        pred_rgb = (pred_rgb * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(render_path, 'img_{}.png'.format(indices_cam[d].item())), cv2.cvtColor(pred_rgb, cv2.COLOR_RGB2BGR))

    exit()

    # ====================Initialize optimizer====================
    if spatial_lr_scale is None:
        spatial_lr_scale = cameras_spatial_extent
        print("Using camera spatial extent as spatial_lr_scale:", spatial_lr_scale)

    opt_params = OptimizationParams(
        iterations=num_iterations,
        position_lr_init=position_lr_init,
        position_lr_final=position_lr_final,
        position_lr_delay_mult=position_lr_delay_mult,
        position_lr_max_steps=position_lr_max_steps,
        feature_lr=feature_lr,
        opacity_lr=opacity_lr,
        scaling_lr=scaling_lr,
        rotation_lr=rotation_lr,
        language_feature_lr=language_feature_lr
    )

    optimizer = SuGaROptimizer(sugar, opt_params, spatial_lr_scale=spatial_lr_scale, include_feature=args.include_feature)
    CONSOLE.print("Optimizer initialized.")
    CONSOLE.print("Optimization parameters:")
    CONSOLE.print(opt_params)

    CONSOLE.print("-----------!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!-------------")

    CONSOLE.print("Optimizable parameters:")
    for param_group in optimizer.optimizer.param_groups:
        CONSOLE.print(param_group['name'], param_group['lr'])

    CONSOLE.print("-----------!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!-------------")


    # ====================Loss function====================
    if loss_function == 'l1':
        loss_fn = l1_loss
    elif loss_function == 'l2':
        loss_fn = l2_loss
    elif loss_function == 'l1+dssim':
        def loss_fn(pred_rgb, gt_rgb):
            return (1.0 - dssim_factor) * l1_loss(pred_rgb, gt_rgb) + dssim_factor * (1.0 - ssim(pred_rgb, gt_rgb))
    CONSOLE.print(f'Using loss function: {loss_function}')

    # ====================Start training====================
    writer = SummaryWriter()
    color_dict = {}
    save_mask_flag = True
    sugar.train()
    epoch = 0
    iteration = 0
    train_losses = []
    t0 = time.time()


    for batch in range(9_999_999):
        if iteration >= num_iterations:
            break

        # Shuffle images
        shuffled_idx = torch.randperm(len(nerfmodel.training_cameras))
        train_num_images = len(shuffled_idx)

        # We iterate on images
        for i in range(0, train_num_images, train_num_images_per_batch):
            iteration += 1

            # Update learning rates
            optimizer.update_learning_rate(iteration)

            start_idx = i
            end_idx = min(i + train_num_images_per_batch, train_num_images)

            camera_indices = shuffled_idx[start_idx:end_idx]

            # Computing rgb predictions
            if not no_rendering:
                outputs = sugar.render_image_gaussian_rasterizer(
                    camera_indices=camera_indices.item(),
                    verbose=False,
                    bg_color=bg_tensor,
                    sh_deg=current_sh_levels - 1,
                    sh_rotations=None,
                    compute_color_in_rasterizer=compute_color_in_rasterizer,
                    compute_covariance_in_rasterizer=True,
                    return_2d_radii=True,
                    quaternions=None,
                    use_same_scale_in_all_directions=use_same_scale_in_all_directions,
                    return_opacities=enforce_entropy_regularization,
                )
                pred_rgb = outputs['image'].view(-1,
                                                 sugar.image_height,
                                                 sugar.image_width,
                                                 3)
                radii = outputs['radii']
                language_feature = outputs["language_feature_image"]
                viewspace_points = outputs['viewspace_points']
                if enforce_entropy_regularization:
                    opacities = outputs['opacities']

                pred_rgb = pred_rgb.transpose(-1, -2).transpose(-2, -3)  # TODO: Change for torch.permute

                # Gather rgb ground truth
                gt_image = nerfmodel.get_gt_image(camera_indices=camera_indices)
                gt_rgb = gt_image.view(-1, sugar.image_height, sugar.image_width, 3)
                gt_rgb = gt_rgb.transpose(-1, -2).transpose(-2, -3)


                # Compute loss
                image_name = nerfmodel.get_image_name(camera_indices=camera_indices.item())
                processed_data_path = args.processed_data_path
                processed_data_path = os.path.join(processed_data_path, "{}.pickle".format(image_name))
                with open(processed_data_path, "rb") as f:
                    gt_sem_map = pickle.load(f)
                gt_sem_map = gt_sem_map.cpu()
                masked_image = torch.zeros(gt_sem_map.shape[0], gt_sem_map.shape[1], 3)
                gt_seg = torch.zeros(len(mapping_dict) + 1, gt_sem_map.shape[0], gt_sem_map.shape[1])
                masked_image = masked_image.cpu().numpy()
                for c in range(len(mapping_dict) + 1):
                    gt_seg[c, :, :] = (gt_sem_map == c).float()
                    if save_mask_flag:
                        if c not in color_dict:
                            color_dict[c] = random.sample(range(0, 255), 3)
                        color = color_dict[c]
                        masked_image[gt_sem_map == c] = color
                if save_mask_flag:
                    cv2.imwrite("output/dataset/masks/{}.png".format(image_name), masked_image)
                if start_idx == train_num_images - 1:
                    with open("output/dataset/masks/color_dict.pickle", "wb") as cd:
                        pickle.dump(color_dict, cd)
                    save_mask_flag = False

                # compute class
                classes_frame = classifier((language_feature.to(args.mapper_device)))
                gt_seg = gt_seg.to(classes_frame.device)
                bce_loss = cls_criterion(classes_frame.unsqueeze(0), gt_seg.unsqueeze(0)).squeeze(0).mean()
                bce_loss = bce_loss / torch.log(torch.tensor(len(mapping_dict) + 1))
                writer.add_scalar('bce_lang', bce_loss, iteration)
                loss = bce_loss
            else:
                loss = 0.

            # Update parameters
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            cls_optimizer.step()
            cls_optimizer.zero_grad()


            # Print loss
            if iteration == 1 or iteration % print_loss_every_n_iterations == 0:
                CONSOLE.print(f'\n-------------------\nIteration: {iteration}')
                train_losses.append(loss.detach().item())
                CONSOLE.print(f"loss: {loss:>7f}  [{iteration:>5d}/{num_iterations:>5d}]",
                              "computed in", (time.time() - t0) / 60., "minutes.")
                CONSOLE.print("bce loss: ", bce_loss)
                with torch.no_grad():
                    scales = sugar.scaling.detach()
                    if not args.include_feature:
                        CONSOLE.print("------Stats-----")
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
                        if sugar.language_feature is not None:
                            CONSOLE.print("Language feats (mean, std): ", sugar.language_feature.mean().item(),
                                          sugar.language_feature.std().item(), sep=' ')
                    else:
                        CONSOLE.print("------Stats-----")
                        CONSOLE.print("---Mean, Std")
                        CONSOLE.print("Language feats : ", sugar.language_feature.mean().item(),
                                      sugar.language_feature.std().item(), sep=' ')
                        CONSOLE.print("---Min, Max, Mean, Std")
                        CONSOLE.print("Opacities:", sugar.strengths.min().item(), sugar.strengths.max().item(),
                                      sugar.strengths.mean().item(), sugar.strengths.std().item(), sep='   ')
                        CONSOLE.print("Points:", sugar.points.min().item(), sugar.points.max().item(),
                                      sugar.points.mean().item(), sugar.points.std().item(), sep='   ')
                t0 = time.time()

            if iteration in save_image_milestones:
                indices_cam = torch.arange(len(nerfmodel.training_cameras))
                # shuffled_idx = shuffled_idx[190:]
                train_num_images = len(indices_cam)
                render_path = "output/render_lang/test_{}".format(iteration)
                if not os.path.exists(render_path):
                    os.makedirs(render_path)
                for d in range(train_num_images):
                    outputs = sugar.render_image_gaussian_rasterizer(
                        camera_indices=indices_cam[d].item(),
                        verbose=False,
                        bg_color=bg_tensor,
                        sh_deg=current_sh_levels - 1,
                        sh_rotations=None,
                        compute_color_in_rasterizer=compute_color_in_rasterizer,
                        compute_covariance_in_rasterizer=True,
                        return_2d_radii=True,
                        quaternions=None,
                        use_same_scale_in_all_directions=use_same_scale_in_all_directions,
                        return_opacities=enforce_entropy_regularization,
                    )
                    language_feature_ = outputs["language_feature_image"]
                    torchvision.utils.save_image(language_feature_[:3, :, :],
                                                 os.path.join(render_path, '{}'.format(indices_cam[d].item()) + ".png"))

            # Save model
            if (iteration % save_model_every_n_iterations == 0) or (iteration in save_milestones):
                CONSOLE.print("Saving model...")
                model_path = os.path.join(sugar_checkpoint_path, f'{iteration}.pt')
                sugar.save_model(path=model_path,
                                 train_losses=train_losses,
                                 epoch=epoch,
                                 iteration=iteration,
                                 optimizer_state_dict=optimizer.state_dict(),
                                 )
                # if optimize_triangles and iteration >= optimize_triangles_from:
                #     rm.save_model(os.path.join(rc_checkpoint_path, f'rm_{iteration}.pt'))
                CONSOLE.print("Model saved.")

            if iteration >= num_iterations:
                break

        epoch += 1

    CONSOLE.print(f"Training finished after {num_iterations} iterations with loss={loss.detach().item()}.")
    CONSOLE.print("Saving final model...")
    model_path = os.path.join(sugar_checkpoint_path, f'{iteration}.pt')
    sugar.save_model(path=model_path,
                     train_losses=train_losses,
                     epoch=epoch,
                     iteration=iteration,
                     optimizer_state_dict=optimizer.state_dict(),
                     )

    CONSOLE.print("Final model saved.")
    return model_path