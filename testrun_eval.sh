#!/bin/bash

#SBATCH --gres=gpu:1
#SBATCH --mail-user=tankeremail@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --ntasks=12
#SBATCH --mem=32G

#SBATCH --account=notchpeak-shared-short
#SBATCH --partition=notchpeak-shared-short
#SBATCH --job-name=test_run_1
#SBATCH --output=testrun/output_eval.txt
#SBATCH --error=testrun/output_eval.txt
#SBATCH --time=8:00:00


conda init; conda activate datart2
python myevaluate.py