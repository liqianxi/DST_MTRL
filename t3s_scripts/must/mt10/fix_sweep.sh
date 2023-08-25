#!/bin/bash

for seed in {0..2}; 
do
    for mask_end_update_episode in {5000,9000};
    do
        for mask_update_interval in {25,50,100};
        do
            for pruning_ratio in {0.5,0.9,0.95};
            do
                sbatch fix_sweep_single.sh $seed $mask_end_update_episode $mask_update_interval $pruning_ratio
            done
        done
    done
done

# for seed in {0..2}; 
# do
#     # for mask_end_update_episode in {4000,5000};
#     # do
#     #     for mask_update_interval in {25,50,100};
#     #     do
#     #         for pruning_ratio in {0.5,0.9,0.95,0.98};
#     #         do
#     sbatch fix_sweep_single.sh $seed 100 100 0.0
#     #         done
#     #     done
#     # done
# done