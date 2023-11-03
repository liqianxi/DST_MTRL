#!/bin/bash
#SBATCH --job-name=fix_must_mt10_sweep
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=96:00:00
#SBATCH --account=rrg-mtaylor3
#SBATCH --output=/home/qianxi/scratch/sparse_training/slurm_log/must/%A.out
#SBATCH --mail-user=qianxi@ualberta.ca
#SBATCH --mail-type=END

module load python/3.8
module load cuda

source /home/qianxi/scratch/sparse_training/sep_t3s/venv/bin/activate
cd /home/qianxi/scratch/sparse_training/sep_t3s/DST_RL
wandb offline
export WANDB_API_KEY=b363daac0bf911130cb2eff814388eaf99942a0b
SECONDS=0

echo "task start"
echo "Use seed $1"

export GROUP=MT10_MUSTSAC
export NAME="10runs_MUST_fixgoal_MT10_exp2_seed${1}"
export TASK_SAMPLE_NUM=10 
ID="1102_itv${2}_pr${3}_sllr${4}_slloss${5}_trajinfo${6}"
GROUP_NAME="1102_sweep"
python starter/mt_must_sac.py --config meta_config/must_configs/mt10/must_mtsac.json --worker_nums 10 --eval_worker_nums 10 --seed $1 --id $ID --mask_update_interval $2 --pruning_ratio $3 --generator_lr $4 --use_sl_loss $5 --use_trajectory_info $6  --wandb_group_name $GROUP_NAME
duration=$SECONDS
echo "$duration seconds elapsed."