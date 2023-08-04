#!/bin/bash
#SBATCH -A uTS22_Pellegri
#SBATCH -p m100_usr_prod
#SBATCH -t 23:59:59
#SBATCH -N 1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=4
#SBATCH --mem=48000
#SBATCH --gres=gpu:1
#SBATCH --job-name=LSA13
#SBATCH --mail-type=ALL
#SBATCH --mail-user=semola96@gmail.com

source ~/.bashrc
module load profile/deeplrn gnu open-ce
conda activate pcb
python main.py --data_dir /m100/home/userexternal/larrighi/Dataset_Low/ --checkpoint_folder /m100/home/userexternal/larrighi/Results/TEST_13_low_long/checkpoints/ --model_save_folder /m100/home/userexternal/larrighi/Results/TEST_13_low_long/models/ --test_folder /m100/home/userexternal/larrighi/Results/TEST_13_low_long/test/
