import sys
from argparse import ArgumentParser
from gaussian_splatting.arguments import ModelParams, PipelineParams
from gaussian_splatting.arguments import OptimizationParams as optiparams
from gaussian_splatting.train import training as gaussiantrainer


def train_vanilla_gs(data_path, iter=7000, diff_splat=False):
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = optiparams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 15_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    if not diff_splat:
        string_args_vanilla_gs = f"""
                        --iterations {iter} -m {"output/"} -s {data_path}
                        """
    else:
        string_args_vanilla_gs = f"""
                                --iterations {iter} -m {"output_diff/"} -s {data_path}
                                """
    string_args_vanilla_gs = string_args_vanilla_gs.split()
    args = parser.parse_args(string_args_vanilla_gs)

    gaussiantrainer(lp.extract(args),
                    op.extract(args),
                    pp.extract(args),
                    args.test_iterations,
                    args.save_iterations,
                    args.checkpoint_iterations,
                    args.start_checkpoint,
                    args.debug_from)
