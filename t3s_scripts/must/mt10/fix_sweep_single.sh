#!/bin/bash
#SBATCH --job-name=fix_must_mt10_sweep
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=72:00:00
#SBATCH --account=def-mtaylor3
#SBATCH --output=/home/qianxi/scratch/slurm_log/must/sep_sweep/%A.out
#SBATCH --mail-user=qianxi@ualberta.ca
#SBATCH --mail-type=END

module load python/3.8
module load cuda

source /home/qianxi/scratch/t3s/venv/bin/activate
cd /home/qianxi/scratch/t3s/t3s_code
wandb enabled
export WANDB_API_KEY=b363daac0bf911130cb2eff814388eaf99942a0b
SECONDS=0

echo "task start"
echo "Use seed $1"

export GROUP=MT10_MUSTSAC
export NAME="10runs_MUST_fixgoal_MT10_exp2_seed${1}"
export TASK_SAMPLE_NUM=10 
python starter/mt_must_sac.py --config meta_config/must_configs/mt10/must_mtsac.json --worker_nums 10 --eval_worker_nums 10 --seed $1 --id "0909_bool${2}_pr${3}_end${4}" --success_traj_update_only $2 --pruning_ratio $3 --mask_end_update_episode $4
duration=$SECONDS
echo "$duration seconds elapsed."