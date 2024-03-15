#!/bin/bash
#SBATCH --job-name=download_multispectal_sat
#SBATCH --output=job_output_multispectal.txt
#SBATCH --error=job_error_multispectal.txt
#SBATCH --ntasks=1
#SBATCH --time=23:59:00
#SBATCH --mem-per-cpu=20Gb
#SBATCH --cpus-per-task=8
#SBATCH --partition=long

### this specifies the length of array
#SBATCH --array=1-7:1

#SBATCH --mail-user=$USER@mila.quebec
#SBATCH --mail-type=FAIL

# load conda environment
module load miniconda/3
conda activate py38

python3 download_rasters_from_planetary_computer.py --index=$SLURM_ARRAY_TASK_ID
