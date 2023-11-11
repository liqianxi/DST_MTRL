<<<<<<< HEAD
cd /home/qianxi/scratch/sparse_training/sep_t3s/DST_RL
=======
cd /home/qianxi/scratch/t3s/t3s_code
>>>>>>> 62bf759bad2fb88b65a7ddf8d02b6641832ddc1e
export GROUP=MT10_MUSTSAC
export NAME="10runs_MUST_fixgoal_MT10_exp1_seed${1}"
export TASK_SAMPLE_NUM=10 
wandb disabled
<<<<<<< HEAD
python starter/mt_must_sac.py --config meta_config/must_configs/mt10/testing_must_mtsac.json --worker_nums 10 --eval_worker_nums 10 --seed $1 --pruning_ratio 0.4 --success_traj_update_only 1 --use_trajectory_info 0 --use_sl_loss 0
=======
python starter/mt_must_sac.py --config meta_config/must_configs/mt10/testing_must_mtsac.json --worker_nums 10 --eval_worker_nums 10 --seed $1 --pruning_ratio 0.9
>>>>>>> 62bf759bad2fb88b65a7ddf8d02b6641832ddc1e
#python starter/mt_must_sac.py --config meta_config/must_configs/mt10/must_mtsac.json --worker_nums 10 --eval_worker_nums 10 --seed $1

duration=$SECONDS
echo "$duration seconds elapsed."
