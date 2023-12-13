#!/bin/bash


# source /home/qianxi/scratch/sparse_training/dec_must/must_env/bin/activate
cd /home/qianxi/scratch/sparse_training/dec_must/DST_RL
wandb disabled
export WANDB_API_KEY=b363daac0bf911130cb2eff814388eaf99942a0b
SECONDS=0

echo "task start"
echo "Use seed $1"

export GROUP=MT10_MUSTSAC
export NAME="10runs_MUST_fixgoal_MT10_exp2_seed${1}"
export TASK_SAMPLE_NUM=$7 
ID="1125_itv${2}_pr${3}_sllr${4}_slloss${5}_trajinfo${6}_selected${7}_select-tsk-id${8}"
GROUP_NAME="1125_sweep_200_max_limit"
python starter/mt_must_sac.py --config meta_config/must_configs/mt10/must_mtsac.json --worker_nums $7 --eval_worker_nums $7 --seed $1 --id $ID --mask_update_interval $2 --pruning_ratio $3 --generator_lr $4 --use_sl_loss $5 --use_trajectory_info $6  --wandb_group_name $GROUP_NAME --selected_task_amount $7 --specify_single_task $8
duration=$SECONDS
echo "$duration seconds elapsed."