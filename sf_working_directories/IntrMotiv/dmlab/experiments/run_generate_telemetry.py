import os

from sample_factory.launcher.run_description import Experiment, ParamGrid, RunDescription
from sample_factory.utils.utils import get_folder_names

path = "/work/classic/fr_js1764-sample_factory/workplace_training_directory/train_dir/hipposlam/RandomTest_"
folder_names_list = get_folder_names(path)

_params = ParamGrid(
    [
        # ("seed", [8, 99]),
        ("expname", list(folder_names_list)),
    ]
)


vstr = "hipposlam"

cli = (
    "--algo=APPO "
    "--env=openfield_map2_fixed_loc3_noreward "
    "--encoder_load_path=/home/fr/fr_js1764/clean_install_mamba/best_000025288_203030528_reward_94.185.pth "
    "--train_dir_path=" + path + " "
    "--train_dir=" + path + " "
    "--max_num_frames=50000 "
    "--num_envs=8 "
    "--use_jit=False "
    "--with_pos_obs=True "
    "--no_render "
    "--number_epochs_analysis=2 "
    "--reset_params=True "
)


_experiments = [
    Experiment("RandomTest", cli, _params.generate_params(False)),
]

RUN_DESCRIPTION = RunDescription(f"{vstr}", experiments=_experiments)


# Generate Telemtry
# python -m sample_factory.launcher.run --backend=slurm --slurm_workdir=./slurm_grid --slurm_gpus_per_job=0 --slurm_cpus_per_gpu=40 --slurm_sbatch_template=/work/classic/fr_js1764-sample_factory/workplace_training_directory/training_templates/generate_telemetry.sh --pause_between=1 --slurm_print_only=False --run=sf_workingdir.dmlab.experiments.run_generate_telemetry --slurm_partition=cpu --slurm_timeout=3:00:00
