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
from sugar_utils.loss_utils import ssim, l1_loss, l2_loss, consistency_loss_3d, weighted_l1_loss
import torchvision
from rich.console import Console
import time
import pickle
from torch.utils.tensorboard import SummaryWriter
import cv2
import random
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import copy


def coarse_training_with_lang(args, gs_model):
    CONSOLE = Console(width=120)

    # ====================Parameters====================

    num_device = args.gpu
    detect_anomaly = False

    # -----Data parameters-----
    downscale_resolution_factor = 1  # 2, 4

    # -----Model parameters-----
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
    num_iterations = 40_000  # Changed

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

    train_num_images_per_batch = 1

    # Loss functions
    loss_function = 'l1+dssim'  # 'l1' or 'l2' or 'l1+dssim'
    if loss_function == 'l1+dssim':
        dssim_factor = 0.2

    # Regularization
    enforce_entropy_regularization = False
    regularize_sdf = False
    bind_to_surface_mesh = False
    regularize = False

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
    save_image_milestones = [7_000, 20_000, 30_000, 40_000]

    # ====================End of parameters====================

    if args.output_dir is None:
        if len(args.scene_path.split("/")[-1]) > 0:
            args.output_dir = os.path.join("./output/coarse_lang", args.scene_path.split("/")[-1])
        else:
            args.output_dir = os.path.join("./output/coarse_lang", args.scene_path.split("/")[-2])

    source_path = args.scene_path
    gs_checkpoint_path = args.checkpoint_path
    iteration_to_load = args.iteration_to_load

    sdf_estimation_factor = args.estimation_factor
    sdf_better_normal_factor = args.normal_factor

    sugar_checkpoint_path = f'sugarcoarse_3Dgs{iteration_to_load}_sdfestimXX_sdfnormYY/'
    print(args.lang_output_dir)
    print(sugar_checkpoint_path)
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
    sugar = SuGaR(
        nerfmodel=nerfmodel,
        points=coarse_ckpt['state_dict']['_points'].detach().float().cuda(),
        colors=SH2RGB(coarse_ckpt['state_dict']['_sh_coordinates_dc'][:, 0, :]).detach().float().cuda(),
        initialize=False,
        sh_levels=4,
        triangle_scale=triangle_scale,
        learnable_positions=learnable_positions,
        keep_track_of_knn=regularize,
        knn_to_track=0,
        freeze_gaussians=freeze_gaussians,
        beta_mode=None,
        surface_mesh_to_bind=o3d_mesh,
        surface_mesh_thickness=None,
        learn_surface_mesh_positions=False,
        learn_surface_mesh_opacity=False,
        learn_surface_mesh_scales=False,
        n_gaussians_per_surface_triangle=1,
        include_feature=args.include_feature
    )

    sugar.load_state_dict(coarse_ckpt['state_dict'])

    CONSOLE.print("\nModel parameters:")
    for name, param in sugar.named_parameters():
        CONSOLE.print(name, param.shape, param.requires_grad)

    sugar.init_lang_ft(args.lang_ft_dim)

    CONSOLE.print(f'\nSuGaR model has been initialized.')
    CONSOLE.print(sugar)
    CONSOLE.print(f'Number of parameters: {sum(p.numel() for p in sugar.parameters() if p.requires_grad)}')
    CONSOLE.print(f'Checkpoints will be saved in {sugar_checkpoint_path}')

    CONSOLE.print("\nModel parameters:")
    for name, param in sugar.named_parameters():
        CONSOLE.print(name, param.shape, param.requires_grad)

    # Compute scene extent
    cameras_spatial_extent = sugar.get_cameras_spatial_extent()

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

    color_dict_ = {}

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
                # depth
                with torch.no_grad():
                    fov_camera = nerfmodel.training_cameras.p3d_cameras[camera_indices.item()]
                    point_depth = fov_camera.get_world_to_view_transform().transform_points(sugar.points)[...,2:].expand(-1, 3)
                    max_depth = point_depth.max()
                    depth = sugar.render_image_gaussian_rasterizer(
                        camera_indices=camera_indices.item(),
                        bg_color=max_depth + torch.zeros(3, dtype=torch.float, device=sugar.device),
                        sh_deg=0,
                        compute_color_in_rasterizer=False,  # compute_color_in_rasterizer,
                        compute_covariance_in_rasterizer=True,
                        return_2d_radii=False,
                        use_same_scale_in_all_directions=False,
                        point_colors=point_depth)
                    depth = depth[0][..., 0]

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

                # Compute loss
                lang_ft_path = os.path.join(args.scene_path, "language_features_dim3")
                gt_language_feature, language_feature_mask, mask_all, save_mask_name, mask_all_ret_0, mask_all_ret_1, mask_all_ret_2, mask_all_ret_3 = nerfmodel.get_gt_lang_ft(camera_indices=camera_indices,
                                                                                                language_feature_dir=lang_ft_path,
                                                                                                to_cuda=True,
                                                                                                langsplat=True,
                                                                                                feature_level=1)

                # for ixc in range(4):
                #     if ixc == 0:
                #         mask_all_curr = mask_all_ret_0
                #     elif ixc == 1:
                #         mask_all_curr = mask_all_ret_1
                #     elif ixc == 2:
                #         mask_all_curr = mask_all_ret_2
                #     else:
                #         mask_all_curr = mask_all_ret_3
                #     save_img_masks = torch.zeros(language_feature.shape[1], language_feature.shape[2], 3)
                #     mask_max_idx = torch.max(mask_all_curr)
                #     if mask_max_idx > -1:
                #         for mask_index in range(mask_max_idx + 1):
                #             if mask_index not in color_dict_:
                #                 color_dict_[mask_index] = torch.tensor([random.random(), random.random(), random.random()])
                #             mask_curr_index = mask_all_curr == mask_index
                #             save_img_masks[mask_curr_index] = color_dict_[mask_index]
                #
                #     save_img_masks = save_img_masks.permute(2, 0, 1)
                #     if args.walkthrough:
                #         render_path_mask = "output/mask/{}".format(ixc)
                #     else:
                #         render_path_mask = "output_diff/mask/{}".format(ixc)
                #     if not os.path.exists(render_path_mask):
                #         os.makedirs(render_path_mask)
                #     torchvision.utils.save_image(save_img_masks,
                #                                  os.path.join(render_path_mask, '{}'.format(save_mask_name) + ".png"))

                if args.walkthrough:
                    if not args.consistancy_loss:
                    # loss_lang = weighted_l1_loss(language_feature.to(gt_language_feature.device) * language_feature_mask,
                    #                              gt_language_feature * language_feature_mask,
                    #                              10/depth)
                        loss_lang = l1_loss(
                            language_feature.to(gt_language_feature.device) * language_feature_mask,
                            gt_language_feature * language_feature_mask)

                        loss = loss_lang
                        writer.add_scalar('lang_loss', loss_lang, iteration)
                    else:
                        if iteration <= args.start_3d_reg_from:
                            loss_lang = l1_loss(
                                language_feature.to(gt_language_feature.device) * language_feature_mask,
                                gt_language_feature * language_feature_mask)

                            loss = loss_lang
                            writer.add_scalar('lang_loss', loss_lang, iteration)

                else: # soft loss to reduce the variance between splats
                    if not args.var_loss:
                        # loss_lang = weighted_l1_loss(
                        #     language_feature.to(gt_language_feature.device) * language_feature_mask,
                        #     gt_language_feature * language_feature_mask,
                        #     10 / depth)
                        loss_lang = l1_loss(
                            language_feature.to(gt_language_feature.device) * language_feature_mask,
                            gt_language_feature * language_feature_mask)

                        loss = loss_lang
                        writer.add_scalar('lang_loss', loss_lang, iteration)
                    else:
                        if iteration < args.start_var_reg_from:
                            # loss_lang = weighted_l1_loss(
                            #     language_feature.to(gt_language_feature.device) * language_feature_mask,
                            #     gt_language_feature * language_feature_mask,
                            #     10 / depth)
                            loss_lang = l1_loss(
                                language_feature.to(gt_language_feature.device) * language_feature_mask,
                                gt_language_feature * language_feature_mask)

                            loss = loss_lang
                            writer.add_scalar('lang_loss', loss_lang, iteration)
                        else:
                            if gs_model is not None:
                                # loss_lang = weighted_l1_loss(
                                #     language_feature.to(gt_language_feature.device) * language_feature_mask,
                                #     gt_language_feature * language_feature_mask,
                                #     10 / depth)
                                loss_lang = l1_loss(
                                    language_feature.to(gt_language_feature.device) * language_feature_mask,
                                    gt_language_feature * language_feature_mask)

                                loss = loss_lang * 0.001
                                writer.add_scalar('lang_loss', loss_lang, iteration)
                                rendered_image_walk = gs_model.sugar.render_image_gaussian_rasterizer(
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
                                language_feature_walk = rendered_image_walk["language_feature_image"].detach()

                                loss_var = l1_loss(language_feature.to(gt_language_feature.device) * language_feature_mask,
                                                   language_feature_walk.to(gt_language_feature.device) * language_feature_mask)
                                #
                                # loss = loss + loss_var
                                # loss_var = torch.sum(1 - torch.nn.functional.cosine_similarity(language_feature.to(gt_language_feature.device).reshape(-1, args.lang_ft_dim),
                                #                                                            language_feature_walk.to(gt_language_feature.device).reshape(-1, args.lang_ft_dim)))
                                loss = loss + loss_var
                                writer.add_scalar('var_loss', loss_var, iteration)
                            else:
                                # loss_lang = weighted_l1_loss(
                                #     language_feature.to(gt_language_feature.device) * language_feature_mask,
                                #     gt_language_feature * language_feature_mask,
                                #     10 / depth)
                                loss_lang = l1_loss(
                                    language_feature.to(gt_language_feature.device) * language_feature_mask,
                                    gt_language_feature * language_feature_mask)

                                loss = loss_lang
                                writer.add_scalar('lang_loss', loss_lang, iteration)

                if args.consistancy_loss and (iteration >= args.start_3d_reg_from):
                    if args.neighbour_consistancy_loss:
                        sampling_mask = radii > 0
                        loss_3d_reg = consistency_loss_3d(sugar.points[sampling_mask],
                                                          sugar.language_feature[sampling_mask],
                                                          k=6,
                                                          max_points=500_000,
                                                          sample_size=1000)

                        loss = loss + loss_3d_reg * 0.005
                        writer.add_scalar('lang_3d_loss', loss_3d_reg, iteration)
                    if args.mask_consistancy_loss:
                        lang_ft_mask_loss = language_feature
                        lang_ft_mask_loss = lang_ft_mask_loss.permute(1, 2, 0)
                        cs_loss_all = torch.tensor(0, device=nerfmodel.device, dtype=torch.float)
                        for mask_ind in mask_all:
                            lang_fts_mask_reg = lang_ft_mask_loss[mask_ind]
                            lang_fts_mask_reg = lang_fts_mask_reg.to(torch.float)
                            avg_lang_ft_mask = torch.mean(lang_fts_mask_reg, dim=0)
                            cs_reg = 1 - torch.nn.functional.cosine_similarity(avg_lang_ft_mask, lang_fts_mask_reg)
                            cs_loss = torch.sum(cs_reg)
                            cs_loss_all += cs_loss

                        loss = cs_loss_all * 0.001
                        writer.add_scalar('mask_loss', cs_loss_all, iteration)

            else:
                loss = 0.

            if loss != 0:
                writer.add_scalar('total_loss', loss, iteration)
            # Update parameters
                loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            # Print loss
            if iteration == 1 or iteration % print_loss_every_n_iterations == 0:
                CONSOLE.print(f'\n-------------------\nIteration: {iteration}')
                train_losses.append(loss.detach().item())
                CONSOLE.print(f"loss: {loss:>7f}  [{iteration:>5d}/{num_iterations:>5d}]",
                              "computed in", (time.time() - t0) / 60., "minutes.")
                with torch.no_grad():
                    scales = sugar.scaling.detach()
                    if not args.include_feature:
                        CONSOLE.print("------Stats-----")
                        # if args.include_3d_reg and (iteration > args.start_3d_reg_from):
                        #     CONSOLE.print("3D loss : ", loss_3d_reg.detach().item())
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
                        # if args.include_3d_reg and (iteration > args.start_3d_reg_from):
                        #     CONSOLE.print("3D loss : ", loss_3d_reg.detach().item())
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
                #indices_cam = torch.arange(len(nerfmodel.training_cameras))
                # shuffled_idx = shuffled_idx[190:]
                #train_num_images = len(indices_cam)
                # if args.walkthrough:
                #     render_path = "output/render_lang_thresh/test_{}".format(iteration)
                # else:
                #     render_path = "output_diff/render_lang_thresh/test_{}".format(iteration)
                # if not os.path.exists(render_path):
                #     os.makedirs(render_path)
                # for d in range(train_num_images):
                #     outputs = sugar.render_image_gaussian_rasterizer(
                #         camera_indices=indices_cam[d].item(),
                #         verbose=False,
                #         bg_color=bg_tensor,
                #         sh_deg=current_sh_levels - 1,
                #         sh_rotations=None,
                #         compute_color_in_rasterizer=compute_color_in_rasterizer,
                #         compute_covariance_in_rasterizer=True,
                #         return_2d_radii=True,
                #         quaternions=None,
                #         use_same_scale_in_all_directions=use_same_scale_in_all_directions,
                #         return_opacities=enforce_entropy_regularization,
                #     )
                #     language_feature_ = outputs["language_feature_image"]
                #
                #     values = language_feature_.permute(1, 2, 0)
                #     values = values.detach().cpu().numpy()
                #     values = np.clip(values, 0, 1)
                #     values = values.reshape(-1)
                #     cmap = plt.get_cmap('viridis')
                #     norm = mcolors.Normalize(vmin=0, vmax=1)
                #     colors = cmap(norm(values))
                #     colors = colors[:, :3]
                #     colors = colors.reshape(language_feature_.shape[1], language_feature_.shape[2], 3)
                #     lang_ft_save = torch.tensor(colors)
                #     lang_ft_save = lang_ft_save.permute(2, 0, 1)
                #     torchvision.utils.save_image(lang_ft_save[:3, :, :],
                #                                  os.path.join(render_path, '{}'.format(indices_cam[d].item()) + ".png"))

                indices_cam = torch.arange(len(nerfmodel.training_cameras))
                # shuffled_idx = shuffled_idx[190:]
                train_num_images = len(indices_cam)
                if args.walkthrough:
                    render_path = "output/render_lang/test_{}".format(iteration)
                else:
                    render_path = "output_diff/render_lang/test_{}".format(iteration)
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
    print("model path : ", model_path)
    sugar.save_model(path=model_path,
                     train_losses=train_losses,
                     epoch=epoch,
                     iteration=iteration,
                     optimizer_state_dict=optimizer.state_dict(),
                     )
    writer.flush()
    CONSOLE.print("Final model saved.")
    return model_path