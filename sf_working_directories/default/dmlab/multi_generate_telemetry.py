import sys
import pathlib
import argparse


from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from sample_factory.utils.utils import experiment_dir, log, get_folder_names, get_file_names

from sf_working_directories.default.dmlab.train_hipposlam import parse_dmlab_args, register_dmlab_components, maybe_overwrite_rnn_size
from sf_working_directories.default.dmlab.dmlab_params import add_dmlab_env_args
from sf_working_directories.default.dmlab.custom_params import add_hipposlam_env_args, hipposlam_override_defaults
from sf_working_directories.default.dmlab.enjoy_telemetry import single_run

# ---------------------------------------------------------------------------
# logging helpers (put near the top of the file, after imports)
# ---------------------------------------------------------------------------
import datetime, pathlib, json, pandas as pd, torch, h5py

def _ensure_parent(path: pathlib.Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def _ensure_dir(path: pathlib.Path):
    path.mkdir(parents=True, exist_ok=True)


mapname="openfield_map2_fixed_loc3_noreward"
expname='02_InternalRewardSeparateLoss4_see_99_e.g.coe_1'
train_dir="/work/classic/fr_js1764-sample_factory/workplace_training_directory/train_dir/hipposlam/InternalRewardSeparateLoss4_"

cli = [
    "--algo", "APPO",
    "--env", mapname ,         # pick any DM‑Lab level you have
    "--experiment", expname,
    "--encoder_load_path","/home/fr/fr_js1764/clean_install_mamba/best_000025288_203030528_reward_94.185.pth",
    "--train_dir", train_dir, # anything writable
    "--max_num_frames", "50000",          # short rollout for the test
    "--num_envs", "8",
    "--dmlab_level_cache_path","./.dmlab_cache",
    "--load_checkpoint_kind","latest",
    "--use_jit","False",
    "--with_pos_obs","True",
    "--no_render",        # <-- skip human window; avoid X11 on servers
]

cli_dict={
 'algo': 'APPO',
 'env': mapname,
 'experiment': expname,
 'encoder_load_path': '/home/fr/fr_js1764/clean_install_mamba/best_000025288_203030528_reward_94.185.pth',
 'train_dir': train_dir,
 'max_num_frames': '50000',
 'num_envs': '8',
 'dmlab_level_cache_path': './.dmlab_cache',
 'load_checkpoint_kind': 'latest',
 'no_render': True,
 'use_jit': False,
 'with_pos_obs': True,
}



def experiment_run(cfg, verbose:bool = True):
    """
    For one experiment this allows to generate multiple trajectories as defined by an integer (--number_epochs_analysis). 
    This is distributed over all possible milestones. For specific milestones use a single run in enjoy_telemetry.py
    
    :param cfg: normal Config class with cli_args to overwrite. This is for one experiment with possibly multiple milestones to be evaluated.
    :type cfg: Config
    :param verbose: Whether intermediate messages should be displayed.
    :type verbose: bool
    """
    get_files_old = True
    if get_files_old:
        joint_path = pathlib.Path(cfg.train_dir_path) / pathlib.Path(cfg.expname) / "checkpoint_p0" / "milestones"
        log.debug(f'Joint Path: {joint_path}')
        milestones = get_file_names(joint_path, ending='.pth', sort_files=True)
        log.debug(f'milestones: {milestones}')
        if len(milestones) < cfg.number_epochs_analysis:
            raise ValueError("Too many epochs to analyze specified. Reduce --number_epochs_analysis")
        else:
            step = len(milestones)/cfg.number_epochs_analysis
            log.debug(f'len milestones & step: {len(milestones), step}')
            for i in range(cfg.number_epochs_analysis):
                epoch_i = int(round(i * step))
                if verbose:
                    log.debug(-epoch_i)
                cfg.load_model_path = joint_path / milestones[-epoch_i-1]
                log.debug(f'load model path: {cfg.load_model_path}')
                destination_path = pathlib.Path(cfg.train_dir_path) / pathlib.Path(cfg.expname) / "telemetry"
                _ensure_dir(destination_path)
                if pathlib.Path(milestones[-epoch_i-1]).stem not in get_folder_names(destination_path):
                    single_run(cfg, pathlib.Path(milestones[-epoch_i-1]).stem, verbose)
                else:
                    log.warn(f"There is already telemetry for this run! Please remove if a re-run is wanted. {cfg.expname} and {milestones[-epoch_i-1]}")
    else:
        single_run(cfg, 'telemetry_random', verbose)




def add_gen_args(parser: argparse.ArgumentParser) -> None:
    p = parser
    p.add_argument(
        "--expname",
        default=False,
        type=str,
        help="Name of the folder of the experiment",
    )
    p.add_argument(
        "--train_dir_path",
        default=False,
        type=str,
        help="Path to the folder of the experiment",
    )
    p.add_argument(
        "--number_epochs_analysis",
        default=1,
        type=int,
        help="How many epochs distributed over an experiment should be run",
    )
    p.add_argument(
        "--reset_params",
        default=False,
        type=bool,
        help="Wether to reset all actor-critic parameters. Mostly for evaluating a random agent.",
    )

def parse_gen_args(evaluation=True, argv=None):
    parser, cfg = parse_sf_args(argv, evaluation=evaluation)
    add_hipposlam_env_args(parser)
    add_dmlab_env_args(parser)
    add_gen_args(parser)
    hipposlam_override_defaults(parser)
    cfg = parse_full_cfg(parser, argv)
    # maybe_overwrite_rnn_size(cfg)
    return cfg

def main():
    """Script entry point."""
    register_dmlab_components()
    cfg = parse_gen_args(evaluation=True)#, argv=cli)

    # tweak whatever you like *after* parsing
    # cfg.cli_args=cli_dict
    verbose = False
    experiment_run(cfg, verbose)



if __name__ == "__main__":
    sys.exit(main())