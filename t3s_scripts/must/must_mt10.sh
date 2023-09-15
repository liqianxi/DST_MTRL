cd /home/qianxi/scratch/t3s/t3s_code
export GROUP=MT10_MUSTSAC
export NAME="10runs_MUST_fixgoal_MT10_exp1_seed${1}"
export TASK_SAMPLE_NUM=10 
wandb disabled
python starter/mt_must_sac.py --config meta_config/must_configs/mt10/testing_must_mtsac.json --worker_nums 10 --eval_worker_nums 10 --seed $1 --pruning_ratio 0.4 --success_traj_update_only 1 --mask_end_update_episode 10
#python starter/mt_must_sac.py --config meta_config/must_configs/mt10/must_mtsac.json --worker_nums 10 --eval_worker_nums 10 --seed $1

duration=$SECONDS
echo "$duration seconds elapsed."
