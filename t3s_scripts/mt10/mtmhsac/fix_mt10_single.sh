#!/bin/bash
#SBATCH --job-name=fix_mt10_10runs
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=16G
#SBATCH --cpus-per-task=1
#SBATCH --time=72:00:00
#SBATCH --account=rrg-mtaylor3
#SBATCH --output=/home/qianxi/scratch/sparse_training/slurm_log/mtmh_sac/%A.out
#SBATCH --mail-user=qianxi@ualberta.ca
#SBATCH --mail-type=END

module load python/3.8
module load cuda

source /home/qianxi/scratch/sparse_training/t3s/t3s_venv/bin/activate

cd /home/qianxi/scratch/sparse_training/t3s/t3s_root
wandb offline
export WANDB_API_KEY=b363daac0bf911130cb2eff814388eaf99942a0b
SECONDS=0

echo "task start"
echo "Use seed $1"

export GROUP=MT10_MHMTSAC
export NAME="10runs_MHMTSAC_fixgoal_MT10_exp1_seed${1}"
export TASK_SAMPLE_NUM=10 
python starter/mt_para_mhmt_sac.py --config meta_config/mt10/mtmhsac.json --worker_nums 10 --eval_worker_nums 10 --seed $1
duration=$SECONDS
echo "$duration seconds elapsed."