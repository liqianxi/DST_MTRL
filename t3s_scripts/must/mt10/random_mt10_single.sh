#!/bin/bash
#SBATCH --job-name=rand_must_mt10_10runs
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=168:00:00
#SBATCH --account=def-mtaylor3
#SBATCH --output=/home/qianxi/scratch/slurm_log/must/%A.out
#SBATCH --mail-user=qianxi@ualberta.ca
#SBATCH --mail-type=END

module load python/3.8
module load cuda

source /home/qianxi/scratch/t3s/venv/bin/activate
cd /home/qianxi/scratch/t3s/t3s_code

export WANDB_API_KEY=b363daac0bf911130cb2eff814388eaf99942a0b
SECONDS=0

echo "task start"
echo "Use seed $1"

export GROUP=MT10_MHMTSAC
export NAME="10runs_MUST_randgoal_MT10_exp1_seed${1}"
export TASK_SAMPLE_NUM=10 
python starter/mt_must_sac.py --config meta_config/must_configs/mt10/random_must_mtsac.json --worker_nums 10 --eval_worker_nums 10 --seed $1
duration=$SECONDS
echo "$duration seconds elapsed."