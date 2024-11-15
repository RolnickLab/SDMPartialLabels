#!/bin/bash
#SBATCH --job-name=splots
#SBATCH --output=job_output_splots.txt
#SBATCH --error=job_error_splots.txt
#SBATCH --ntasks=1
#SBATCH --time=23:59:00
#SBATCH --mem-per-cpu=50Gb
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

module load miniconda/3
conda activate new_env
export COMET_API_KEY=$COMET_API_KEY
python main.py --config="new_src/config_ctran.yaml"
