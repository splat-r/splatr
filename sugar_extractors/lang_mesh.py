import os
import open3d as o3d
import torch
from pytorch3d.renderer import TexturesUV
from pytorch3d.structures import Meshes
from pytorch3d.ops import knn_points
from pytorch3d.io import save_obj
from sugar_scene.gs_model import GaussianSplattingWrapper
from sugar_scene.sugar_model import SuGaR, extract_texture_image_and_uv_from_gaussians
from sugar_utils.spherical_harmonics import SH2RGB
import open_clip
from autoencoder.model import Autoencoder
import cv2
from rich.console import Console
import numpy 
from tqdm import trange
import pickle
import copy
import torch.nn as nn
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_distances as cosine_similarity_matrix
from sklearn.metrics.pairwise import euclidean_distances

from pytorch3d.renderer import (
    AmbientLights,
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    )
from pytorch3d.renderer.blending import BlendParams
from pytorch3d.io import load_objs_as_meshes

def extract_lang_mesh(args):
    CONSOLE = Console(width=120)

    print("processing clip ...")
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
            "ViT-B-16", "laion2b_s34b_b88k"
        )
    clip_model = clip_model.to(args.clip_device)
    clip_tokenizer = open_clip.get_tokenizer("ViT-B-16")

    tokenized_text = clip_tokenizer([args.text_prompt]).to("cpu")
    text_feat = clip_model.encode_text(tokenized_text)
    text_feat /= text_feat.norm(dim=-1, keepdim=True)

    n_skip_images_for_eval_split = 8
            
    # --- Scene data parameters ---
    source_path = args.scene_path
    use_train_test_split = args.eval
    
    # --- Vanilla 3DGS parameters ---
    iteration_to_load = args.iteration_to_load
    gs_checkpoint_path = args.checkpoint_path
    
    # --- Fine model parameters ---
    refined_model_path = args.refined_model_path
    if args.n_gaussians_per_surface_triangle is None:
        n_gaussians_per_surface_triangle = int(refined_model_path.split('/')[-2].split('_gaussperface')[-1])
    else:
        n_gaussians_per_surface_triangle = args.n_gaussians_per_surface_triangle
    
    # --- Output parameters ---
    if args.mesh_output_dir is None:
        if len(args.scene_path.split("/")[-1]) > 0:
            args.mesh_output_dir = os.path.join("./output/refined_mesh_lang", args.scene_path.split("/")[-1])
        else:
            args.mesh_output_dir = os.path.join("./output/refined_mesh_lang", args.scene_path.split("/")[-2])
    mesh_output_dir = args.mesh_output_dir
    os.makedirs(mesh_output_dir, exist_ok=True)
    
    mesh_save_path = refined_model_path.split('/')[-2]
    if args.postprocess_mesh:
        mesh_save_path = mesh_save_path + '_postprocessed'
    mesh_save_path = mesh_save_path + '.obj'
    mesh_save_path = os.path.join(mesh_output_dir, mesh_save_path)
    
    scene_name = source_path.split('/')[-2] if len(source_path.split('/')[-1]) == 0 else source_path.split('/')[-1]
    sugar_mesh_path = os.path.join('./output/coarse_mesh/', scene_name, 
                                refined_model_path.split('/')[-2].split('_normalconsistency')[0].replace('sugarfine', 'sugarmesh') + '.ply')
    
    if args.square_size is None:
        if n_gaussians_per_surface_triangle == 1:
            # square_size = 5  # Maybe 4 already works
            square_size = 10  # Maybe 4 already works
        if n_gaussians_per_surface_triangle == 6:
            square_size = 10
    else:
        square_size = args.square_size
        
    # Postprocessing
    postprocess_mesh = args.postprocess_mesh
    if postprocess_mesh:
        postprocess_density_threshold = args.postprocess_density_threshold
        postprocess_iterations = args.postprocess_iterations
            
    CONSOLE.print('==================================================')
    CONSOLE.print("Starting extracting texture from refined SuGaR model:")
    CONSOLE.print('Scene path:', source_path)
    CONSOLE.print('Iteration to load:', iteration_to_load)
    CONSOLE.print('Vanilla 3DGS checkpoint path:', gs_checkpoint_path)
    CONSOLE.print('Refined model path:', refined_model_path)
    CONSOLE.print('Coarse mesh path:', sugar_mesh_path)
    CONSOLE.print('Mesh output directory:', mesh_output_dir)
    CONSOLE.print('Mesh save path:', mesh_save_path)
    CONSOLE.print('Number of gaussians per surface triangle:', n_gaussians_per_surface_triangle)
    CONSOLE.print('Square size:', square_size)
    CONSOLE.print('Postprocess mesh:', postprocess_mesh)
    CONSOLE.print('==================================================')
    
    # Set the GPU
    # torch.cuda.set_device(args.gpu)
    
    # ==========================    
    
    # --- Loading Vanilla 3DGS model ---
    CONSOLE.print("Source path:", source_path)
    CONSOLE.print("Gaussian splatting checkpoint path:", gs_checkpoint_path)    
    CONSOLE.print(f"\nLoading Vanilla 3DGS model config {gs_checkpoint_path}...")
    
    nerfmodel = GaussianSplattingWrapper(
        source_path=source_path,
        output_path=gs_checkpoint_path,
        iteration_to_load=iteration_to_load,
        load_gt_images=False,  # TODO: Check
        eval_split=use_train_test_split,
        eval_split_interval=n_skip_images_for_eval_split,
        )
    CONSOLE.print("Vanilla 3DGS Loaded.")
    CONSOLE.print(f'{len(nerfmodel.training_cameras)} training images detected.')
    CONSOLE.print(f'The model has been trained for {iteration_to_load} steps.')
    CONSOLE.print(len(nerfmodel.gaussians._xyz) / 1e6, "M gaussians detected.")
    
    # --- Loading coarse mesh ---
    o3d_mesh = o3d.io.read_triangle_mesh(sugar_mesh_path)
    
    print(nerfmodel.device)

    # --- Loading refined SuGaR model ---
    # set this to cpu, to run it on cpu
    checkpoint = torch.load(refined_model_path, map_location="cpu")
    
    for key in checkpoint['state_dict']:
        try:
            print("{}".format(key), checkpoint['state_dict'][key].shape)
        except:
            print("cant : {}".format(key))
    print("mem1 : ", torch.cuda.memory_allocated(0)/1024/1024/1024)
    refined_sugar = SuGaR(
        nerfmodel=nerfmodel,
        points=checkpoint['state_dict']['_points'],
        colors=SH2RGB(checkpoint['state_dict']['_sh_coordinates_dc'][:, 0, :]),
        initialize=False,
        sh_levels=nerfmodel.gaussians.active_sh_degree+1,
        keep_track_of_knn=False,
        knn_to_track=0,
        beta_mode='average',
        surface_mesh_to_bind=o3d_mesh,
        n_gaussians_per_surface_triangle=n_gaussians_per_surface_triangle,
        )
    
    refined_sugar.init_lang_ft(args.lang_ft_dim) 

    for name, param in refined_sugar.named_parameters():
        CONSOLE.print(name, param.shape, param.requires_grad)

    print("mem2 : ", torch.cuda.memory_allocated(0)/1024/1024/1024)

    refined_sugar.load_state_dict(checkpoint['state_dict'])
    refined_sugar.eval()
    refined_sugar.reset_grads()

    print("mem4 : ", torch.cuda.memory_allocated(0)/1024/1024/1024)

    for name, param in refined_sugar.named_parameters():
        CONSOLE.print(name, param.shape, param.requires_grad)

    # Load the autoencoder model
    autoenc_ckpt_ = args.autoenc_ckpt
    autoenc_ckpt = torch.load(autoenc_ckpt_)
    autoenc_model = Autoencoder(args.encoder_dims, args.decoder_dims).to("cuda:0")
    autoenc_model.load_state_dict(autoenc_ckpt)
    autoenc_model.eval()
    autoenc_model = autoenc_model.to(args.decoder_device)
    out_decoder = autoenc_model.decode(refined_sugar.language_feature.to(args.decoder_device))

    # clustering algorthm
    # def clustering(lang_fts):
    #     cluster = {}
    #     i = 0
    #     while len(lang_fts) > 0:
    #         print(i, len(lang_fts))
    #         p = 150000
    #         query = lang_fts[p]
    #         cosine_similarity = torch.nn.functional.cosine_similarity(lang_fts, query, dim=-1)
    #         mask = cosine_similarity > 0.88
    #         cluster[i] = lang_fts[mask]
    #         i = i+1
    #         lang_fts = lang_fts[~mask]
    #         lang_mesh, new_o3d_mesh, param_dict = refined_sugar.language_mesh(text_feat.to(args.decoder_device),
    #                                                                           out_decoder, args.threshold_lang,
    #                                                                           args.n_gaussians_lang_per_triangle,
    #                                                                           mask)
    #
    #         refined_sugar_ = SuGaR(
    #             nerfmodel=nerfmodel,
    #             points=None,
    #             colors=None,
    #             initialize=False,
    #             sh_levels=nerfmodel.gaussians.active_sh_degree + 1,
    #             keep_track_of_knn=False,
    #             knn_to_track=0,
    #             beta_mode='average',
    #             surface_mesh_to_bind=new_o3d_mesh,
    #             n_gaussians_per_surface_triangle=refined_sugar.n_gaussians_per_surface_triangle,
    #         )
    #         refined_sugar_._scales = param_dict['scales']
    #         refined_sugar_._quaternions = param_dict['quaternions']
    #         refined_sugar_.all_densities = param_dict['all_densities']
    #         refined_sugar_._sh_coordinates_dc = param_dict['sh_dc']
    #         refined_sugar_._sh_coordinates_rest = param_dict['sh_rest']
    #
    #         bg_tensor = torch.ones(3, dtype=torch.float, device=nerfmodel.device)  # using white background
    #         shuffled_idx = torch.arange(len(nerfmodel.training_cameras))
    #         camera_indices = shuffled_idx[0:args.num_image_render]
    #         current_sh_levels = 4  # since we initialized from a trained 3DGS
    #         compute_color_in_rasterizer = False
    #         use_same_scale_in_all_directions = False
    #         use_densifier = False
    #         regularize = False
    #         enforce_entropy_regularization = False
    #         shuffled_idx = torch.arange(len(nerfmodel.training_cameras))
    #         n_cams = len(shuffled_idx)
    #
    #         img_save_dir = "output/rendered_images/{}".format(p)
    #         if not os.path.exists(img_save_dir):
    #             os.makedirs(img_save_dir)
    #         for cam_idx in trange(n_cams):
    #             # We iterate on images
    #             output_img = refined_sugar_.render_image_gaussian_rasterizer(
    #                 camera_indices=cam_idx,
    #                 verbose=False,
    #                 bg_color=bg_tensor,
    #                 sh_deg=current_sh_levels - 1,
    #                 sh_rotations=None,
    #                 compute_color_in_rasterizer=compute_color_in_rasterizer,
    #                 compute_covariance_in_rasterizer=True,
    #                 return_2d_radii=use_densifier or regularize,
    #                 quaternions=None,
    #                 use_same_scale_in_all_directions=use_same_scale_in_all_directions,
    #                 return_opacities=enforce_entropy_regularization,
    #             )
    #             output_img = output_img.cpu().detach().numpy()
    #             output_img = (output_img * 255).astype(numpy.uint8)
    #             cv2.imwrite(os.path.join(img_save_dir, "{}.png".format(cam_idx)), output_img)
    #     print(cluster.keys())
    #
    # clustering(out_decoder)

    # # DBSCAN for clustering into objects
    # def clustering_fts(lang_fts_to_cluster, points):
    #     # computing distance based on cosine similarity (semantic similarity)
    #     semantic_sim_matrix = cosine_similarity_matrix(lang_fts_to_cluster)
    #
    #     # computing distance based on spatial similarity
    #     spatial_sim_distance = euclidean_distances(points)
    #
    #     # normalizing the distance matrices
    #     spatial_sim_distance = (spatial_sim_distance - spatial_sim_distance.min())/(spatial_sim_distance.max()-spatial_sim_distance.min())
    #
    #     file = open("output/data.txt", 'w')
    #     file.write(str(list(semantic_sim_matrix[0])))
    #     file.close()
    #
    #     # net distance matrix
    #     phi = args.semantic_sim_distance_coeff
    #     distance_matrix = phi*semantic_sim_matrix + (1-phi)*spatial_sim_distance
    #     print(list(distance_matrix))
    #     # running DBSCAN
    #     eps = 0.05
    #     min_samples = 5  # Minimum number of samples in a neighborhood for a point to be considered as a core point
    #     db = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
    #     db.fit(distance_matrix)
    #     labels = db.labels_
    #
    #     mask = (labels == 1)
    #     print("n_points : ", torch.sum((torch.tensor(mask)*1.0)))
    #
    #     # retrieving the clusters
    #     clusters = {}
    #     cluster_fts = {}
    #     cluster_len = {}
    #     for label in set(labels):
    #         clusters[label] = lang_fts_to_cluster[labels == label]
    #         cluster_len[label] = len(clusters[label])
    #         # cluster mean -> decoder or decode the cluster then mean ?
    #         mean_ft = (lang_fts_to_cluster[labels == label]).mean(dim=0)
    #         cluster_fts[label] = mean_ft
    #     print(cluster_len)
    #
    #     return clusters, cluster_fts, mask
 # {0: 39627, 1: 8, 2: 15, 3: 5, -1: 345}


    # lang_fts_to_cluster_[10_000:20_000] = out_decoder[10_000:20_000, :].clone()
    # lang_fts_to_cluster_[20_000:40_000] = out_decoder[1_110_000:1_130_000, :].clone()
    # #lang_fts_to_cluster_ = refined_sugar.language_feature[1_110_000:1_150_000, :].clone()
    # points_to_cluster = refined_sugar.points[110_000:150_000, :].clone()
    # points_to_cluster[10_000:20_000] = refined_sugar.points[10_000:20_000, :].clone()
    # points_to_cluster[20_000:40_000] = refined_sugar.points[1_110_000:1_130_000, :].clone()
    #
    # lang_fts_to_cluster_ = lang_fts_to_cluster_.detach().cpu()
    # points_to_cluster = points_to_cluster.detach().cpu()
    # clusters_, cluster_fts_, mask = clustering_fts(lang_fts_to_cluster_, points_to_cluster)
    #
    # mask_new = torch.zeros(refined_sugar.points.shape[0])
    # mask_new[110_000:120_000] = torch.tensor(mask[0:10_000])
    # mask_new[10_000:20_000] = torch.tensor(mask[10_000:20_000])
    # mask_new[1_110_000:1_130_000] = torch.tensor(mask[20_000:40_000])

    # map_path = os.path.join(args.processed_data_path, "mapping_dict.pickle")
    # with open(map_path, 'rb') as map_file:
    #     mapping_dict = pickle.load(map_file)
    # classifier = torch.nn.Conv2d(args.lang_ft_dim, len(mapping_dict) + 1, kernel_size=1)
    # classifier.to(args.mapper_device)
    # classifier.load_state_dict(torch.load(args.lang_conv_cp, map_location="cuda:0"))
    # classifier.eval()
    # cls_criterion = torch.nn.CrossEntropyLoss(reduction='none')
    # cls_optimizer = torch.optim.Adam(classifier.parameters(), lr=0.01)




    print("----------------------------------------")
    print(refined_sugar.points.shape)
    print(refined_sugar.sh_coordinates.shape)
    print(refined_sugar.language_feature.shape)
    print(refined_sugar.quaternions.shape)
    print(refined_sugar._points.shape)

    print("computing text sim ...")

    # # computing semantic map
    # lang_ft = copy.deepcopy(refined_sugar.language_feature)
    # lang_ft = lang_ft.permute(1, 0)
    # lang_ft = lang_ft[..., None]
    # sem_lang_classes = classifier((lang_ft.to(args.mapper_device)))
    # sem_lang_classes = sem_lang_classes.squeeze(2).permute(1, 0)
    # sem_lang_classes = torch.softmax(sem_lang_classes, dim=1)
    # classes_gaussian = torch.argmax(sem_lang_classes, dim=1)
    # # loadding id to map dict
    #
    # # {1: 'beanbag', 2: 'table', 3: 'toy', 4: 'monitor', 5: 'stroller', 6: 'chair'}

    # gaussian_mask = classes_gaussian == 5

    lang_mesh, new_o3d_mesh, param_dict = refined_sugar.language_mesh(text_feat.to(args.decoder_device), out_decoder, args.threshold_lang, args.n_gaussians_lang_per_triangle)
    CONSOLE.print("New gaussian parameters")
    for name, param in refined_sugar.named_parameters():
        CONSOLE.print(name, param.shape, param.requires_grad)

    refined_sugar = SuGaR(
                nerfmodel=nerfmodel,
                points=None,
                colors=None,
                initialize=False,
                sh_levels=nerfmodel.gaussians.active_sh_degree+1,
                keep_track_of_knn=False,
                knn_to_track=0,
                beta_mode='average',
                surface_mesh_to_bind=new_o3d_mesh,
                n_gaussians_per_surface_triangle=refined_sugar.n_gaussians_per_surface_triangle,
                )
    refined_sugar._scales = param_dict['scales']
    refined_sugar._quaternions = param_dict['quaternions']
    refined_sugar.all_densities = param_dict['all_densities']
    refined_sugar._sh_coordinates_dc = param_dict['sh_dc']
    refined_sugar._sh_coordinates_rest = param_dict['sh_rest']
    # refined_sugar._points = param_dict['points']
    # refined_sugar._surface_mesh_faces = param_dict['surface_mesh_faces']

    lang_conditioned_model_path = args.refined_model_path
    lang_conditioned_model_ckpt = torch.load(lang_conditioned_model_path, map_location="cpu")

    new_sugar = SuGaR(
        nerfmodel=nerfmodel,
        points=refined_sugar.points,
        colors=SH2RGB(param_dict['sh_dc'][:, 0, :]),
        initialize=True,
        sh_levels=nerfmodel.gaussians.active_sh_degree+1,
        learnable_positions=False,
        triangle_scale=1,
        keep_track_of_knn=False,
        knn_to_track=0,
        beta_mode=False,
        freeze_gaussians=True,
        surface_mesh_to_bind=None,
        surface_mesh_thickness=None,
        learn_surface_mesh_positions=False,
        learn_surface_mesh_opacity=False,
        learn_surface_mesh_scales=False,
        n_gaussians_per_surface_triangle=n_gaussians_per_surface_triangle,
        )

    new_sugar._scales = torch.nn.Parameter(refined_sugar.scaling.detach(), requires_grad=False)
    new_sugar._quaternions = torch.nn.Parameter(refined_sugar.quaternions.detach(), requires_grad=False)
    new_sugar.all_densities = torch.nn.Parameter(refined_sugar.all_densities.detach(), requires_grad=False)
    new_sugar._sh_coordinates_dc = param_dict['sh_dc']
    new_sugar._sh_coordinates_rest = param_dict['sh_rest']

    mesh_model_path = "output/sugar_new.pt"
    new_sugar.save_model(path=mesh_model_path)

    print("model saved ...")
    for key in new_sugar.state_dict():
        print(key, " : ", new_sugar.state_dict()[key].shape)

    del new_sugar

    CONSOLE.print("New mesh initialized")
    for name, param in refined_sugar.named_parameters():
        CONSOLE.print(name, param.shape, param.requires_grad)

    with torch.no_grad():
        verts_uv, faces_uv, texture_img = extract_texture_image_and_uv_from_gaussians(
            refined_sugar, square_size=square_size, n_sh=1, texture_with_gaussian_renders=True)

        textures_uv = TexturesUV(
            maps=texture_img[None], #texture_img[None]),
            verts_uvs=verts_uv[None],
            faces_uvs=faces_uv[None],
            sampling_mode='nearest',
            )
        textured_mesh = Meshes(
            verts=[refined_sugar.surface_mesh.verts_list()[0]],   
            faces=[refined_sugar.surface_mesh.faces_list()[0]],
            textures=textures_uv,
            )

    with torch.no_grad():
        save_obj(  
            mesh_save_path,
            verts=textured_mesh.verts_list()[0],
            faces=textured_mesh.faces_list()[0],
            verts_uvs=textured_mesh.textures.verts_uvs_list()[0],
            faces_uvs=textured_mesh.textures.faces_uvs_list()[0],
            texture_map=textured_mesh.textures.maps_padded()[0].clamp(0., 1.),
            )
    
    # rendering a few images
    bg_tensor = torch.ones(3, dtype=torch.float, device=nerfmodel.device) # using white background
    shuffled_idx = torch.randperm(len(nerfmodel.training_cameras))
    camera_indices = shuffled_idx[0:args.num_image_render]
    current_sh_levels = 4 # since we initialized from a trained 3DGS
    compute_color_in_rasterizer = False
    use_same_scale_in_all_directions = False


    model_path = "output/sugar.pt"
    refined_sugar.save_model(path=model_path)

    print("model saved ...")
    for key in refined_sugar.state_dict():
        print(key, " : ", refined_sugar.state_dict()[key].shape)
    
    use_densifier = False
    regularize = False
    enforce_entropy_regularization = False
    shuffled_idx = torch.arange(len(nerfmodel.training_cameras))
    n_cams = len(shuffled_idx)
    img_save_dir= "output/rendered_images"
    if not os.path.exists(img_save_dir):
        os.makedirs(img_save_dir)
    for cam_idx in trange(n_cams):
        # We iterate on images
        output_img = refined_sugar.render_image_gaussian_rasterizer(
            camera_indices=cam_idx,
            verbose=False,
            bg_color = bg_tensor,
            sh_deg=current_sh_levels-1,
            sh_rotations=None,
            compute_color_in_rasterizer=compute_color_in_rasterizer,
            compute_covariance_in_rasterizer=True,
            return_2d_radii=use_densifier or regularize,
            quaternions=None,
            use_same_scale_in_all_directions=use_same_scale_in_all_directions,
            return_opacities=enforce_entropy_regularization,
            )
        output_img = output_img.cpu().detach().numpy()
        output_img = (output_img*255).astype(numpy.uint8)
        cv2.imwrite(os.path.join(img_save_dir, "{}.png".format(cam_idx)), output_img)

    return model_path, mesh_save_path, new_o3d_mesh