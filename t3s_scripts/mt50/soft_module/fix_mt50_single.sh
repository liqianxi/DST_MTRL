#!/bin/bash
#SBATCH --job-name=fix_sm_mt50_10runs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=5
#SBATCH --time=3:00:00
#SBATCH --account=rrg-mtaylor3
#SBATCH --output=/home/qianxi/scratch/sparse_training/slurm_log/soft_mod/%A.out
#SBATCH --mail-user=qianxi@ualberta.ca
#SBATCH --mail-type=ALL

module load python/3.8
module load cuda

source /home/qianxi/scratch/sparse_training/t3s/t3s_venv/bin/activate

cd /home/qianxi/scratch/sparse_training/t3s/t3s_root

SECONDS=0

echo "task start"
echo "Use seed $1"

export GROUP="MT50_SOFTMOD_FIX"
export NAME="10runs_SOFTMOD_fixgoal_MT50_exp1_seed${1}" 
export TASK_SAMPLE_NUM=50 

# Be aware that in rl_algo there's a
# task_scheduler = TaskScheduler(num_tasks=50, task_sample_num=TASK_SAMPLE_NUM)
# that controls the num_tasks pasted into the scheduler.
python starter/mt_para_mtsac_modular_gated_cas.py --config meta_config/mt50/modular_2_2_2_256_reweight.json --worker_nums 50 --eval_worker_nums 50 --seed $1

duration=$SECONDS
echo "$duration seconds elapsed."