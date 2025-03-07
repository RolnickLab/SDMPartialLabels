#!/bin/bash
#SBATCH --job-name=satbird_ctran
#SBATCH --output=job_output_test.txt
#SBATCH --error=job_error_test.txt
#SBATCH --ntasks=1
#SBATCH --time=23:59:00
#SBATCH --mem-per-cpu=50Gb
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1

module load miniconda/3
conda activate new_env
export COMET_API_KEY=$COMET_API_KEY
python train.py  args.config=configs/satbird/satbird_ctran_base.yaml
