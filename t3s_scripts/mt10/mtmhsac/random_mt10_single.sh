#!/bin/bash
#SBATCH --job-name=random_mt10_10runs
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
SECONDS=0

echo "task start"
echo "Use seed $1"

GROUP=MT10_MHMTSAC_RANDOM NAME="10runs_MHMTSAC_randomgoal_MT10_exp1_seed${1}" TASK_SAMPLE_NUM=10 python starter/mt_para_mhmt_sac.py --config meta_config/mt10/mtmhsac_rand.json --worker_nums 10 --eval_worker_nums 10 --seed $1
duration=$SECONDS
echo "$duration seconds elapsed."