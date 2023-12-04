cd /home/qianxi/scratch/DST/DST_RL
export GROUP=MT10_MUSTSAC
export NAME="10runs_MUST_fixgoal_MT10_exp1_seed${1}"
export TASK_SAMPLE_NUM=10 
wandb disabled
python starter/mt_must_sac.py --config meta_config/must_configs/mt10/must_mtsac.json --worker_nums 2 --eval_worker_nums 2 --seed $1 --pruning_ratio 0.4 --success_traj_update_only 1 --use_trajectory_info 0 --use_sl_loss 0 --selected_task_amount 10
#python starter/mt_must_sac.py --config meta_config/must_configs/mt10/must_mtsac.json --worker_nums 10 --eval_worker_nums 10 --seed $1

duration=$SECONDS
echo "$duration seconds elapsed."
