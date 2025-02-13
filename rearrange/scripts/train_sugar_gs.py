import argparse
from sugar_utils.general_utils import str2bool
from sugar_trainers.coarse_density import coarse_training_with_density_regularization
from sugar_trainers.coarse_sdf import coarse_training_with_sdf_regularization
from sugar_trainers.coarse_lang import coarse_training_with_lang
from sugar_extractors.coarse_mesh import extract_mesh_from_coarse_sugar
from rearrange.scripts.config import actions, GaussianConfig
import sys
import os

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self


def train_sugar_gs(data_path, config, gs_model=None):
    # ----- Parser -----
    parser = argparse.ArgumentParser(description='Script to optimize a full SuGaR model.')

    # Data and vanilla 3DGS checkpoint
    parser.add_argument('-s', '--scene_path',
                        type=str,
                        help='(Required) path to the scene data to use.')
    parser.add_argument('-c', '--checkpoint_path',
                        type=str,
                        help='(Required) path to the vanilla 3D Gaussian Splatting Checkpoint to load.')
    parser.add_argument('-i', '--iteration_to_load',
                        type=int, default=7000,
                        help='iteration to load.')

    # Regularization for coarse SuGaR
    parser.add_argument('-r', '--regularization_type', type=str,
                        help='(Required) Type of regularization to use for coarse SuGaR. Can be "sdf" or "density". '
                             'For reconstructing detailed objects centered in the scene with 360° coverage, "density" provides a better foreground mesh. '
                             'For a stronger regularization and a better balance between foreground and background, choose "sdf".')

    # Extract mesh
    parser.add_argument('-l', '--surface_level', type=float, default=0.3,
                        help='Surface level to extract the mesh at. Default is 0.3')
    parser.add_argument('-v', '--n_vertices_in_mesh', type=int, default=1_000_000,
                        help='Number of vertices in the extracted mesh.')
    parser.add_argument('-b', '--bboxmin', type=str, default=None,
                        help='Min coordinates to use for foreground.')
    parser.add_argument('-B', '--bboxmax', type=str, default=None,
                        help='Max coordinates to use for foreground.')
    parser.add_argument('--center_bbox', type=str2bool, default=True,
                        help='If True, center the bbox. Default is False.')

    # Parameters for refined SuGaR
    parser.add_argument('-g', '--gaussians_per_triangle', type=int, default=1,
                        help='Number of gaussians per triangle.')
    parser.add_argument('-f', '--refinement_iterations', type=int, default=15_000,
                        help='Number of refinement iterations.')

    # (Optional) Parameters for textured mesh extraction
    parser.add_argument('-t', '--export_uv_textured_mesh', type=str2bool, default=True,
                        help='If True, will export a textured mesh as an .obj file from the refined SuGaR model. '
                             'Computing a traditional colored UV texture should take less than 10 minutes.')
    parser.add_argument('--square_size',
                        default=10, type=int, help='Size of the square to use for the UV texture.')
    parser.add_argument('--postprocess_mesh', type=str2bool, default=False,
                        help='If True, postprocess the mesh by removing border triangles with low-density. '
                             'This step takes a few minutes and is not needed in general, as it can also be risky. '
                             'However, it increases the quality of the mesh in some cases, especially when an object is visible only from one side.')
    parser.add_argument('--postprocess_density_threshold', type=float, default=0.1,
                        help='Threshold to use for postprocessing the mesh.')
    parser.add_argument('--postprocess_iterations', type=int, default=5,
                        help='Number of iterations to use for postprocessing the mesh.')

    # (Optional) PLY file export
    parser.add_argument('--export_ply', type=str2bool, default=True,
                        help='If True, export a ply file with the refined 3D Gaussians at the end of the training. '
                             'This file can be large (+/- 500MB), but is needed for using the dedicated viewer. Default is True.')

    # (Optional) Default configurations
    parser.add_argument('--low_poly', type=str2bool, default=False,
                        help='Use standard config for a low poly mesh, with 200k vertices and 6 Gaussians per triangle.')
    parser.add_argument('--high_poly', type=str2bool, default=False,
                        help='Use standard config for a high poly mesh, with 1M vertices and 1 Gaussians per triangle.')
    parser.add_argument('--refinement_time', type=str, default=None,
                        help="Default configs for time to spend on refinement. Can be 'short', 'medium' or 'long'.")

    # Evaluation split
    parser.add_argument('--eval', type=str2bool, default=True, help='Use eval split.')

    # GPU
    parser.add_argument('--gpu', type=int, default=0, help='Index of GPU device to use.')
    parser.add_argument('--white_background', type=str2bool, default=False,
                        help='Use a white background instead of black.')
    if config.walkthrough:
        string_args_sugar = f"""
                            -c {"output/"} -r {"sdf"} -s {data_path} --low_poly {True}
                            """
    else:
        string_args_sugar = f"""
                                -c {"output_diff/"} -r {"sdf"} -s {data_path} --low_poly {True}
                                """
    string_args_sugar = string_args_sugar.split()
    # Parse arguments
    args = parser.parse_args(string_args_sugar)

    if args.low_poly:
        args.n_vertices_in_mesh = 200_000
        args.gaussians_per_triangle = 6
        print('Using low poly config.')
    if args.high_poly:
        args.n_vertices_in_mesh = 1_000_000
        args.gaussians_per_triangle = 1
        print('Using high poly config.')
    if args.refinement_time == 'short':
        args.refinement_iterations = 2_000
        print('Using short refinement time.')
    if args.refinement_time == 'medium':
        args.refinement_iterations = 7_000
        print('Using medium refinement time.')
    if args.refinement_time == 'long':
        args.refinement_iterations = 15_000
        print('Using long refinement time.')
    if args.export_uv_textured_mesh:
        print('Will export a UV-textured mesh as an .obj file.')
    if args.export_ply:
        print('Will export a ply file with the refined 3D Gaussians at the end of the training.')

    # ----- Optimize coarse SuGaR -----
    if config.walkthrough:
        if len(args.scene_path.split("/")[-1]) > 0:
            output_dir = os.path.join("./output/coarse", args.scene_path.split("/")[-1])
        else:
            output_dir = os.path.join("./output/coarse", args.scene_path.split("/")[-2])
    else:
        if len(args.scene_path.split("/")[-1]) > 0:
            output_dir = os.path.join("./output_diff/coarse", args.scene_path.split("/")[-1])
        else:
            output_dir = os.path.join("./output_diff/coarse", args.scene_path.split("/")[-2])

    coarse_args = AttrDict({
        'checkpoint_path': args.checkpoint_path,
        'scene_path': args.scene_path,
        'iteration_to_load': args.iteration_to_load,
        'output_dir': output_dir,
        'eval': False,
        'estimation_factor': 0.2,
        'normal_factor': 0.2,
        'gpu': args.gpu,
        'white_background': args.white_background,
        'include_feature': False,
        'feature_level': 0,
        'start_var_reg_from': 20_000,
        'var_loss': False,
        'walkthrough': config.walkthrough,
        'surface_align': config.surface_align
    })
    if args.regularization_type == 'sdf':
        coarse_sugar_path = coarse_training_with_sdf_regularization(coarse_args, gs_model)
    elif args.regularization_type == 'density':
        coarse_sugar_path = coarse_training_with_density_regularization(coarse_args)
    else:
        raise ValueError(f'Unknown regularization type: {args.regularization_type}')

    # Training the diff splat
    if config.diff_splat_train:
        # coarse_sugar_path = "output_diff/coarse/unshuffle/sugarcoarse_3Dgs7000_sdfestim02_sdfnorm02/15000.pt"
        if config.walkthrough:
            lang_dir = "output/coarse_lang"
        else:
            lang_dir = "output_diff/coarse_lang"
        coarse_diff_args = AttrDict({
            'checkpoint_path': args.checkpoint_path,
            'scene_path': args.scene_path,
            'iteration_to_load': args.iteration_to_load,
            'output_dir': None,
            'eval': args.eval,
            'estimation_factor': 0.2,
            'normal_factor': 0.2,
            'gpu': args.gpu,
            'white_background': args.white_background,
            'include_feature': True,
            'coarse_model_path': coarse_sugar_path,
            'lang_ft_dim': 3,
            'lang_output_dir': lang_dir,
            'walkthrough': config.walkthrough,
            "start_3d_reg_from": 20_000,
            "consistancy_loss": True,
            "mask_consistancy_loss": True,
            "neighbour_consistancy_loss": False,
            'start_var_reg_from': 28_000,
            'var_loss': False
        })
        coarse_lang_sugar_path = coarse_training_with_lang(coarse_diff_args, gs_model)


    if config.get_mesh:
        coarse_mesh_args = AttrDict({
            'scene_path': args.scene_path,
            'checkpoint_path': args.checkpoint_path,
            'iteration_to_load': args.iteration_to_load,
            'coarse_model_path': coarse_sugar_path,
            'surface_level': args.surface_level,
            'decimation_target': args.n_vertices_in_mesh,
            'mesh_output_dir': None,
            'bboxmin': args.bboxmin,
            'bboxmax': args.bboxmax,
            'center_bbox': args.center_bbox,
            'gpu': args.gpu,
            'eval': False,
            'use_centers_to_extract_mesh': False,
            'use_marching_cubes': False,
            'use_vanilla_3dgs': False,
            'include_feature': False,
            'lang_output_dir': "output_diff/coarse_lang"
        })
        coarse_mesh_path = extract_mesh_from_coarse_sugar(coarse_mesh_args)[0]

# if "__main__" == __name__:
#     base_path_ = sys.argv[1]
#     config_ = sys.argv[2]
#     train_sugar_gs(base_path_)


