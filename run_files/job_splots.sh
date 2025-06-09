#!/bin/bash
#SBATCH --job-name=splots_2
#SBATCH --output=job_output_splots_2.txt
#SBATCH --error=job_error_splots_2.txt
#SBATCH --ntasks=1
#SBATCH --time=10:59:00
#SBATCH --mem-per-cpu=50Gb
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --partition=long

module load miniconda/3
conda activate new_env
export COMET_API_KEY=$COMET_API_KEY
python train.py args.config="configs/satbirdxsplot/config_ctran.yaml"
