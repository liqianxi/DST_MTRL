#module load python/3.8
cd /home/qianxi/scratch/t3s/t3s_code
export GROUP=MT10_MHMTSAC
export NAME="10runs_MHMTSAC_fixgoal_MT10_exp1_seed${1}"
export TASK_SAMPLE_NUM=10 
python starter/mt_must_sac.py --config meta_config/must_configs/mt10/must_mtsac.json --worker_nums 10 --eval_worker_nums 10 --seed $1
#python starter/mt_must_sac.py --config meta_config/must_configs/mt10/must_mtsac.json --worker_nums 10 --eval_worker_nums 10 --seed $1

duration=$SECONDS
echo "$duration seconds elapsed."
