#!/bin/bash
#SBATCH --job-name=Resnet18_SatBird+SatButterfly
#SBATCH --output=job_output_satbird_satbutterfly.txt
#SBATCH --error=job_error_satbird_satbutterfly.txt
#SBATCH --ntasks=1
#SBATCH --time=23:59:00
#SBATCH --mem-per-cpu=100Gb
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --partition=long

### this specifies the number of runs (we are doing 3 runs for now)
#SBATCH --array=1-3:1

# load conda environment
module load anaconda/3
conda activate eco

# export keys for logging, etc,
export COMET_API_KEY=$COMET_API_KEY
export HYDRA_FULL_ERROR=1

# run training script
python train.py args.config=configs/SatBirdxSatButterfly/resnet18_RGB_ENV.yaml args.run_id=$SLURM_ARRAY_TASK_ID
