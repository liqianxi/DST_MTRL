#!/bin/bash

for seed in {0..2}; 
do
    for mask_update_interval in {25,100};
    do
        for pruning_ratio in {0.5,0.1,0.7};
        do
            for gen_lr in {1e-4,5e-4,2e-5};
            do
  
                sbatch fix_sweep_single.sh $seed $mask_update_interval $pruning_ratio $gen_lr 0 0

            done
        done

    done
done
# for seed in {0..2}; 
# do
#     for mask_update_interval in {25,100};
#     do
#         for pruning_ratio in {0.5,0.1,0.7};
#         do
#             for gen_lr in {1e-4,5e-4,2e-5};
#             do

#                 for use_trajectory_info in {0,1};
#                 do
#                     sbatch fix_sweep_single.sh $seed $mask_update_interval $pruning_ratio $gen_lr 1 $use_trajectory_info
#                 done

#             done
#         done

#     done
# done
