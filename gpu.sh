#!/bin/bash
#SBATCH --job-name=my_gpu_job_concept_3       
#SBATCH --output=gpu_job_output_concept_3.log  
#SBATCH --partition=gpua100            
#SBATCH --gres=gpu:1                   
#SBATCH --cpus-per-task=8              
#SBATCH --mem=32G                       
#SBATCH --time=01:20:00                 


source ~/.bashrc
conda activate ppo


bash ~/conceptshap/conceptSHAP/train_eval_imdb.sh
