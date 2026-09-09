#!/bin/bash
#SBATCH --job-name=my_job           # Name of your job
#SBATCH --output=./out/output_%j.log      # Output log file (%j is job ID)
#SBATCH --error=./error/error_%j.log        # Error log file
#SBATCH --time=$TIMEOUT             # Time limit (HH:MM:SS)
#SBATCH $PARTITION        # Partition/queue name
#SBATCH --ntasks=1                  # Number of tasks
#SBATCH --cpus-per-task=$CPU           # Number of CPU cores per task
#SBATCH --mem=200G                    # Memory per node

#SBATCH --mail-type=END,FAIL        # When to email you (optional)
#SBATCH --mail-user=xiaoxiong.lin@bcf.uni-freiburg.de  # Your email address

# Load required modules
# module load python/3.10

# Navigate to your working directory
cd /work/classic/fr_xl1014-train/grid/exp_trans_dec
source activate SFgit
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1




# Execute your command
python -m sf_workingdir.dmlab.train_hipposlam $CMD --heartbeat_interval=40 --heartbeat_reporting_interval=600

