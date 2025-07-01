#!/bin/bash
#SBATCH --job-name=birdxbutterfly
#SBATCH --output=job_output_birdxbutterfly.txt
#SBATCH --error=job_error_birdxbutterfly.txt
#SBATCH --ntasks=1
#SBATCH --time=15:59:00
#SBATCH --mem-per-cpu=50Gb
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --partition=long
#SBATCH --array=1-3:1

module load miniconda/3
conda activate new_env
export COMET_API_KEY=$COMET_API_KEY
python train.py args.config="configs/satbirdxsatbutterfly/config_ciso.yaml" args.run_id=$SLURM_ARRAY_TASK_ID
