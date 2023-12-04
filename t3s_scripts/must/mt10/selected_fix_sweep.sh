#!/bin/bash

for seed in {0..1}; 
do

    for pruning_ratio in {0.1,0.7};
    do
        for gen_lr in {1e-3,1e-2};
        do
            for selected_task in {2,4};
            do 
                sbatch selected_fix_sweep_single.sh $seed 50 $pruning_ratio $gen_lr 0 0 $selected_task
            done
        done
    done


done
# for seed in {0..1}; 
# do

#     for pruning_ratio in {0.1,0.7};
#     do
#         for gen_lr in {1e-3,1e-2};
#         do

#             for use_trajectory_info in {0,1};
#             do
#                 for selected_task in {2,3,5};
#                 do 
#                     sbatch selected_fix_sweep_single.sh $seed 50 $pruning_ratio $gen_lr 1 $use_trajectory_info $selected_task
#                 done
#             done

#         done
#     done

# done
