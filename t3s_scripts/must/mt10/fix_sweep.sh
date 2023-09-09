#!/bin/bash

for seed in {0..2}; 
do
    for success_traj_update_only in {1,0};
    do
        for pruning_ratio in {0.5,0.3};
        do
            for update_end in {5000,7500,10000}
            do
                sbatch fix_sweep_single.sh $seed $success_traj_update_only $pruning_ratio $update_end
            done
        done

    done
done
for seed in {0..2}; 
do

    sbatch fix_sweep_single.sh $seed 0 0 5

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